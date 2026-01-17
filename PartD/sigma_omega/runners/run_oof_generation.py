#!/usr/bin/env python3
"""
OOF Generation Script - For Meta-Learner Tuning

Generates Out-of-Fold predictions with all features enabled:
- TabPFN (64 ensembles, raw data)
- Diffusion Augmentation
- Laplacian Eigenmaps
- Per-model CV Razor

Output: PartD/outputs/oof_cache.npz containing OOF predictions for tuning.

Usage:
    python -m sigma_omega.runners.run_oof_generation

Estimated time: ~30-40 minutes on RTX 3060
"""

import os
import sys
import gc
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Add parent to path for imports
# sys.path hack usually causes issues when running with -m
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .. import config
from ..data import load_data_safe
from ..pipeline import predict_probs_for_view
from ..stacking import fit_predict_stacking
from ..features import apply_feature_view, build_streams
from ..utils import seed_everything
# legacy/imports need fixing too downstream?
# models imports below need fixing too

def main():
    print(">>> OOF GENERATION FOR META-LEARNER TUNING <<<")
    # ... (header prints)
    
    # ... load data ...
    X, y, X_test = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    # 2. PER-MODEL CV RAZOR
    print("[RAZOR] Computing per-model CV-averaged feature importance...")
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb
    
    n_splits = 5
    razor_iterations = 500
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # CatBoost CV Importance - GPU for speed
    cat_importances = []
    print("  [CatBoost] Computing 5-fold CV importance (GPU)...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        scout = CatBoostClassifier(
            iterations=razor_iterations, 
            verbose=0, 
            task_type='GPU',
            random_seed=42 + fold_idx,
        )
        scout.fit(X[train_idx], y_enc[train_idx])
        cat_importances.append(scout.get_feature_importance())
        
        # Cleanup per dfold
        del scout
        gc.collect()
        
    cat_imp_avg = np.mean(cat_importances, axis=0)
    
    # XGBoost CV Importance - Keep on GPU if possible, but handle OOM
    xgb_importances = []
    print("  [XGBoost] Computing 5-fold CV importance...")
    try:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
            xgb_model = xgb.XGBClassifier(
                n_estimators=razor_iterations,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=42 + fold_idx,
                verbosity=0,
            )
            xgb_model.fit(X[train_idx], y_enc[train_idx])
            xgb_importances.append(xgb_model.feature_importances_)
            
            del xgb_model
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [WARN] XGBoost GPU failed ({e}), switching to CPU...")
        xgb_importances = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
            xgb_model = xgb.XGBClassifier(
                n_estimators=razor_iterations,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist',
                device='cpu',
                n_jobs=-1,
                random_state=42 + fold_idx,
                verbosity=0,
            )
            xgb_model.fit(X[train_idx], y_enc[train_idx])
            xgb_importances.append(xgb_model.feature_importances_)
            
    xgb_imp_avg = np.mean(xgb_importances, axis=0)
    
    # cleanup after razor
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create Model-Specific Masks
    razor_threshold = 10
    cat_thresh = np.percentile(cat_imp_avg, razor_threshold)
    xgb_thresh = np.percentile(xgb_imp_avg, razor_threshold)
    
    cat_mask = cat_imp_avg > cat_thresh
    xgb_mask = xgb_imp_avg > xgb_thresh
    
    keep_mask = cat_mask
    X_razor = X[:, keep_mask]
    X_test_razor = X_test[:, keep_mask]
    
    print(f"  > CatBoost mask: {np.sum(cat_mask)}/{X.shape[1]} features kept")
    print(f"  > XGBoost mask: {np.sum(xgb_mask)}/{X.shape[1]} features kept")
    
    razor_masks = {'cat': cat_mask, 'xgb': xgb_mask}

    # 3. COLLECT OOF PREDICTIONS
    all_oof_preds = {}
    all_test_preds = {}
    
    # Import model factories - MOVED HERE TO FIX NAME ERROR
    # Import model factories - MOVED HERE TO FIX NAME ERROR
    from ..models_trees import get_xgb_dart, get_cat_langevin
    from ..models_torch import TrueTabR
    from ..models_pfn import TabPFNWrapper
    
    for seed in config.SEEDS:
        print(f"\n>>> SEED {seed} <<<")
        seed_everything(seed)
        
        for view in config.VIEWS:
            print(f"  [VIEW] {view}")
            key = f"seed{seed}_{view}"
            
            # Define All Potential Models
            # Structure: (Name, Factory/Constructor, UseFullData?)
            model_definitions = [
                ('XGB_DART', lambda: get_xgb_dart(num_classes, iterations=config.GBDT_ITERATIONS), False),
                ('Cat_Langevin', lambda: get_cat_langevin(num_classes, iterations=config.GBDT_ITERATIONS * 2), False),
                ('TrueTabR', lambda: TrueTabR(num_classes), False),
            ]
            
            if config.USE_TABPFN and view == 'raw':
                 model_definitions.append(
                     ('TabPFN', lambda: TabPFNWrapper(device='cuda' if torch.cuda.is_available() else 'cpu', n_estimators=config.TABPFN_N_ENSEMBLES), True)
                 )

            # Store View Results
            view_oof = []
            view_test = []
            view_meta_X = [] 
            view_meta_te = []
            view_names = []

            # --- SEQUENTIAL EXECUTION LOOP ---
            for name, factory, use_full_data in model_definitions:
                print(f"    Running {name} on {view} view...")
                
                # Instantiate
                model_instance = factory()
                
                # [Optimization] Skip Trees on PCA/ICA views
                # Trees work best on axis-aligned data; rotations hurt them.
                is_tree = ('XGB' in name or 'Cat' in name or 'LGBM' in name)
                if is_tree and view not in ['raw', 'quantile']:
                    print(f"    [SKIP] Skipping {name} on {view} (Trees dislike rotations)")
                    continue

                # Select Data
                X_tr_use = X if use_full_data else X_razor
                X_te_use = X_test if use_full_data else X_test_razor
                
                # Run Stacking for Single Model
                # fit_predict_stacking expects a list of (name, model)
                oof, test, _, _, _ = fit_predict_stacking(
                    names_models=[(name, model_instance)],
                    view_name=view,
                    X_train_base=X_tr_use,
                    X_test_base=X_te_use,
                    y=y_enc,
                    num_classes=num_classes,
                    cv_splits=config.N_FOLDS,
                    seed=seed,
                    return_oof=True,
                )
                
                # Collect Results (oof is a list of 1 element)
                view_oof.append(oof[0])
                view_test.append(test[0])
                view_names.append(name)
                
                # Cleanup
                del model_instance, oof, test
                gc.collect()
                torch.cuda.empty_cache()
            
            # Aggregate View Results for Meta-Learner caching
            # We need to reconstruct meta_X (OOFs) and meta_te (Test Preds)
            # fit_predict_stacking usually returns meta_X/Te which includes meta-features
            # But since we ran 1-by-1, we need to handle that if used.
            # Currently fit_predict_stacking returns meta_X from the loop.
            # BUT we should probably just save the raw OOF probs.
            # Rerunning prob_meta_features later is cheap.
            
            # Wait, fit_predict_stacking returns `meta_X` which is [oof_preds] + [meta_feats].
            # Since we want to save RAW OOF predictions for later tuning:
            # We just save the `view_oof` and `view_test` arrays.
            # The cached file expects `seed{seed}_{view}_oof` to be a list of arrays.
            
            # Just mimicking previous structure:
            # meta_X was (N_samples, N_models * (n_classes + n_meta_feats))
            # WE DO NOT need to reconstruct that perfectly right now if we just want OOFs.
            # BUT the cache verification might look for it? 
            # Let's clean up what we write to disk. 
            # We will just write 'oof_preds' and 'test_preds' lists.
            # The 'meta_X' keys in npz might be redundant if we just reload OOFs.
            # Let's keep writing them but as empty/placeholder or just concatenated raw probs.
            # Actually, `run_stacking_tuning.py` (future) will likely load `oof_preds`.
            
            all_oof_preds[key] = {
                'oof_preds': view_oof, 
                'test_preds': view_test,
                'model_names': view_names,
                # Placeholders for compatibility if needed, or remove
                'meta_X': [], 
                'meta_te': [],
            }


    # 4. SAVE OOF CACHE
    output_dir = 'PartD/outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'oof_cache.npz')
    
    np.savez(
        output_path,
        y_true=y_enc,
        seeds=config.SEEDS,
        views=config.VIEWS,
        **{f"{k}_oof": v['oof_preds'] for k, v in all_oof_preds.items()},
        **{f"{k}_test": v['test_preds'] for k, v in all_oof_preds.items()},
        **{f"{k}_meta_X": v['meta_X'] for k, v in all_oof_preds.items()},
        **{f"{k}_meta_te": v['meta_te'] for k, v in all_oof_preds.items()},
    )
    
    print(f"\n[SUCCESS] OOF cache saved to {output_path}")
    print(f"  - Seeds: {config.SEEDS}")
    print(f"  - Views: {config.VIEWS}")
    print(f"  - Models per view: {len(model_definitions)}")
    print("\nNext step: Run run_stacking_tuning.py to tune meta-learner")


if __name__ == "__main__":
    main()
