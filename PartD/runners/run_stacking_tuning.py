#!/usr/bin/env python3
"""
Meta-Learner Optimization for Stacking Ensemble (Fantastic 4)

This script:
1. Generates and caches OOF predictions for all base models
2. Uses Optuna to find the optimal meta-learner type and hyperparameters

Base Models: XGBoost DART, CatBoost Langevin, TrueTabR, TabPFN
Meta-Learner Candidates: Logistic Regression, Ridge, LightGBM, NNLS
"""

import os
import sys
import json
import copy
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import nnls

warnings.filterwarnings('ignore')

# Add parent dir to path
# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.utils import seed_everything
from PartD.sigma_omega.features import apply_feature_view, build_streams
from PartD.sigma_omega.models_trees import get_xgb_dart, get_cat_langevin
from PartD.sigma_omega.models_torch import TrueTabR
from PartD.sigma_omega.losses import prob_meta_features

# Paths
OUTPUT_DIR = Path(__file__).parent / "outputs"
OOF_CACHE_PATH = OUTPUT_DIR / "oof_cache_phe.npz" # New cache for PHE experiments
RESULTS_PATH = OUTPUT_DIR / "meta_learner_results.json"
STORAGE_URL = f"sqlite:///{OUTPUT_DIR / 'nas.db'}"

# Config
SEED = 42
N_FOLDS = 5
TIMEOUT = int(os.getenv('TUNING_TIMEOUT', 1800))  # 30 min default

# Smoke test mode
SMOKE_RUN = os.getenv('SMOKE_RUN', 'False').lower() in ('true', '1', 'yes')
if SMOKE_RUN:
    print(">>> SMOKE RUN: Reduced complexity <<<")
    N_FOLDS = 2
    TIMEOUT = 60


def get_base_models(num_classes: int) -> list:
    """Return list of (name, model) tuples for base models."""
    models = [
        ('XGB_DART', get_xgb_dart(num_classes, iterations=config.GBDT_ITERATIONS)),
        ('Cat_Langevin', get_cat_langevin(num_classes, iterations=config.GBDT_ITERATIONS * 2)),
        ('TrueTabR', TrueTabR(num_classes)),
    ]
    
    # Add TabPFN if available
    if config.USE_TABPFN:
        try:
            from PartD.sigma_omega.models_pfn import TabPFNWrapper
            models.append(('TabPFN', TabPFNWrapper(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                n_estimators=8
            )))
        except ImportError:
            print("Warning: TabPFN not available, skipping.")
    
    return models


def generate_oof_predictions(X: np.ndarray, y: np.ndarray, num_classes: int) -> dict:
    """
    Generate Out-of-Fold predictions for all base models.
    
    Returns:
        dict with keys: oof_preds, test_preds (if needed), y, num_classes
    """
    print("\n=== PHASE 1: Generating OOF Predictions ===")
    
    models = get_base_models(num_classes)
    n_models = len(models)
    
    # Initialize OOF arrays
    oof_preds = {name: np.zeros((len(y), num_classes), dtype=np.float32) for name, _ in models}
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Apply quantile view (best for neural, ok for trees)
    X_view, _ = apply_feature_view(X, X, view='quantile', seed=SEED, allow_transductive=False)
    X_tree, _, X_neural, _, _, _ = build_streams(X_view, X_view)
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        
        X_tree_tr, X_tree_val = X_tree[tr_idx], X_tree[val_idx]
        X_neural_tr, X_neural_val = X_neural[tr_idx], X_neural[val_idx]
        y_tr = y[tr_idx]
        
        for name, model_template in models:
            print(f"  Training {name}...", end=" ", flush=True)
            
            model = copy.deepcopy(model_template)
            
            # Select appropriate feature streams
            if 'TabPFN' in name:
                # TabPFN v2 prefers raw data (no quantile transform)
                # We simply slice the original raw X
                X_f_tr = X[tr_idx]
                X_f_val = X[val_idx]
            else:
                is_tree = 'XGB' in name or 'Cat' in name
                X_f_tr = X_tree_tr if is_tree else X_neural_tr
                X_f_val = X_tree_val if is_tree else X_neural_val
            
            try:
                model.fit(X_f_tr, y_tr)
                preds = model.predict_proba(X_f_val)
                oof_preds[name][val_idx] = preds.astype(np.float32)
                print(f"✓ (val acc: {(preds.argmax(1) == y[val_idx]).mean():.2%})")
            except Exception as e:
                print(f"✗ Error: {e}")
                # Fill with uniform distribution on error
                oof_preds[name][val_idx] = 1.0 / num_classes
    
    return {
        'oof_preds': oof_preds,
        'y': y,
        'num_classes': num_classes,
        'model_names': [name for name, _ in models]
    }


def build_meta_features(oof_dict: dict) -> np.ndarray:
    """
    Build meta-feature matrix from OOF predictions.
    
    Features per model:
    - Raw probabilities (num_classes dims)
    - Top1 confidence (1 dim)
    - Top1-Top2 gap (1 dim)
    - Entropy (1 dim)
    """
    model_names = oof_dict['model_names']
    oof_preds = oof_dict['oof_preds']
    
    meta_parts = []
    for name in model_names:
        p = oof_preds[name]
        meta_parts.append(p)  # Raw probs
        meta_parts.append(prob_meta_features(p))  # top1, gap, entropy
    
    return np.hstack(meta_parts)


def create_objective(meta_X: np.ndarray, y: np.ndarray, num_classes: int):
    """Create Optuna objective function for meta-learner tuning."""
    
    def objective(trial):
        # Choose meta-learner type
        meta_type = trial.suggest_categorical('meta_type', ['lr', 'ridge', 'lgbm', 'nnls'])
        
        if meta_type == 'lr':
            C = trial.suggest_float('lr_C', 0.01, 10.0, log=True)
            model = LogisticRegression(C=C, random_state=SEED, solver='lbfgs', max_iter=500)
            
        elif meta_type == 'ridge':
            alpha = trial.suggest_float('ridge_alpha', 0.01, 100.0, log=True)
            model = RidgeClassifier(alpha=alpha, random_state=SEED)
            
        elif meta_type == 'lgbm':
            try:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int('lgbm_n_estimators', 50, 300),
                    num_leaves=trial.suggest_int('lgbm_num_leaves', 8, 64),
                    max_depth=trial.suggest_int('lgbm_max_depth', 2, 6),
                    learning_rate=trial.suggest_float('lgbm_lr', 0.01, 0.3, log=True),
                    random_state=SEED,
                    verbose=-1,
                    n_jobs=-1
                )
            except ImportError:
                # Fallback if LightGBM not installed
                print("Warning: LightGBM not installed, falling back to LR")
                model = LogisticRegression(C=1.0, random_state=SEED, solver='lbfgs', max_iter=500)
            
        else:  # nnls
            # NNLS: Compute weights via Non-Negative Least Squares
            # We evaluate using leave-one-out approximation
            n_models = len([c for c in meta_X.T[:num_classes*4:num_classes]])  # Count models
            
            # Build target probability matrix (one-hot encoded)
            Z = np.zeros((len(y), 4))  # 4 base models
            
            # Extract just the predicted class probability for each model
            # Assuming meta_X has shape: [probs_m1(C), meta_m1(3), probs_m2(C), ...]
            model_stride = num_classes + 3  # probs + 3 meta features
            for m_i in range(4):
                start_idx = m_i * model_stride
                probs_m = meta_X[:, start_idx:start_idx + num_classes]
                Z[:, m_i] = probs_m[np.arange(len(y)), y]
            
            target = np.ones(len(y))
            weights, _ = nnls(Z, target)
            weights = weights / (weights.sum() + 1e-9)
            
            # Compute blended accuracy
            blended_probs = np.zeros((len(y), num_classes))
            for m_i in range(4):
                start_idx = m_i * model_stride
                probs_m = meta_X[:, start_idx:start_idx + num_classes]
                blended_probs += probs_m * weights[m_i]
            
            acc = (blended_probs.argmax(1) == y).mean()
            return acc
        
        # For sklearn models, use cross-validation
        try:
            scores = cross_val_score(model, meta_X, y, cv=3, scoring='accuracy', n_jobs=-1)
            return scores.mean()
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    return objective


def run_tuning(meta_X: np.ndarray, y: np.ndarray, num_classes: int) -> dict:
    """Run Optuna tuning for meta-learner selection."""
    print("\n=== PHASE 2: Meta-Learner Tuning ===")
    print(f"Meta-feature matrix shape: {meta_X.shape}")
    print(f"Timeout: {TIMEOUT}s")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='meta_learner_tuning',
        storage=STORAGE_URL,
        load_if_exists=True
    )
    
    # Enqueue defaults for faster convergence
    if len(study.trials) == 0:
        study.enqueue_trial({'meta_type': 'lr', 'lr_C': 1.0})
        study.enqueue_trial({'meta_type': 'nnls'})
        study.enqueue_trial({'meta_type': 'ridge', 'ridge_alpha': 1.0})
    
    study.optimize(
        create_objective(meta_X, y, num_classes),
        timeout=TIMEOUT,
        n_jobs=1,  # Sequential for stability
        show_progress_bar=True
    )
    
    print(f"\n✓ Best trial: {study.best_trial.number}")
    print(f"  Accuracy: {study.best_value:.4%}")
    print(f"  Params: {study.best_params}")
    
    return {
        'best_accuracy': float(study.best_value),
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }


def main():
    print("=" * 60)
    print("META-LEARNER OPTIMIZATION FOR STACKING ENSEMBLE")
    print("=" * 60)
    
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    X, y, _ = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"  Samples: {len(X)}, Classes: {num_classes}")
    
    # Phase 1: Generate or load OOF
    if OOF_CACHE_PATH.exists() and not SMOKE_RUN:
        print("\n[2] Loading cached OOF predictions...")
        cache = np.load(OOF_CACHE_PATH, allow_pickle=True)
        oof_dict = {
            'oof_preds': cache['oof_preds'].item(),
            'y': cache['y'],
            'num_classes': int(cache['num_classes']),
            'model_names': list(cache['model_names'])
        }
    else:
        oof_dict = generate_oof_predictions(X, y_enc, num_classes)
        # Cache
        np.savez(
            OOF_CACHE_PATH,
            oof_preds=oof_dict['oof_preds'],
            y=oof_dict['y'],
            num_classes=oof_dict['num_classes'],
            model_names=oof_dict['model_names']
        )
        print(f"\n✓ OOF cached to {OOF_CACHE_PATH}")
    
    # Build meta features
    print("\n[3] Building meta-features...")
    meta_X = build_meta_features(oof_dict)
    print(f"  Shape: {meta_X.shape}")
    
    # Phase 2: Tune meta-learner
    results = run_tuning(meta_X, y_enc, num_classes)
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_PATH}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best Meta-Learner: {results['best_params'].get('meta_type', 'unknown')}")
    print(f"Best CV Accuracy:  {results['best_accuracy']:.4%}")
    print(f"Total Trials:      {results['n_trials']}")
    
    # Recommend config update
    best_type = results['best_params'].get('meta_type', 'lr')
    print(f"\nTo use this config, set: META_LEARNER={best_type}")


if __name__ == "__main__":
    main()
