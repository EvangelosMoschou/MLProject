
import os
import sys
# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import optuna
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.models_torch import ThetaTabM, TrueTabR
from PartD.sigma_omega.utils import seed_everything



# Disable Optuna logging output
optuna.logging.set_verbosity(optuna.logging.INFO)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_thetatabm_objective(X, y, num_classes, epochs=50):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'depth': trial.suggest_int('depth', 2, 5),
            'k': trial.suggest_categorical('k', [8, 16, 32]),
            'num_classes': num_classes
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            # Monkeypatch epochs via config logic overlay or custom loop
            # models_torch.fit() uses hardcoded 20 range(20).
            # To fix this properly, we need to Change models_torch.fit to accept epochs arg.
            # OR we loop fit() multiple times.
            
            model = ThetaTabM(**params)
            
            # HACK: Create a custom fit loop here to override 20 epochs
            # Or simpler: Update models_torch.py to accept epochs in fit (cleaner).
            # Assuming we updated models_torch.py (next step).
            model.fit(X_tr, y_tr, epochs=epochs) 
            
            # Evaluate
            probs = model.predict_proba(X_val)
            preds = np.argmax(probs, axis=1)
            acc = (preds == y_val).mean()
            scores.append(acc)
            
        return np.mean(scores)
    return objective

def get_tabr_objective(X, y, num_classes, epochs=50):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'head_hidden': trial.suggest_categorical('head_hidden', [32, 64, 128]),
            'context_size': trial.suggest_categorical('context_size', [32, 64, 96, 128]),
            'num_classes': num_classes,
            'n_neighbors': 16
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = TrueTabR(**params)
            model.fit(X_tr, y_tr, epochs=epochs)
            
            probs = model.predict_proba(X_val)
            preds = np.argmax(probs, axis=1)
            acc = (preds == y_val).mean()
            scores.append(acc)
            
        return np.mean(scores)
    return objective

def main():
    print(">>> STARTING NEURAL ARCHITECTURE SEARCH (NAS) [PERSISTENT] <<<")
    
    # Check for smoke run
    is_smoke = os.getenv('SMOKE_RUN', 'False').lower() in ('true', '1', 'yes')
    timeout = 60 if is_smoke else 7200  # 1 min vs 2 hours per model
    n_epochs = 1 if is_smoke else 50
    
    # DB Storage for Persistence
    storage_url = "sqlite:///PartD/outputs/nas.db"
    
    # Data Loading
    X, y, _ = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Razor (10% conservative)
    print("[RAZOR] Scanning for noise features...")
    from catboost import CatBoostClassifier
    scout = CatBoostClassifier(iterations=100, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    thresh = np.percentile(imps, 10)
    keep_mask = imps > thresh
    X_razor = X[:, keep_mask]
    print(f"  > Dropped {np.sum(~keep_mask)} features. New Dim: {X_razor.shape[1]}")

    # Quantile Transform
    print("[PREPROC] Applying Gaussian Quantile Transform...")
    qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal', random_state=42)
    X_neural = qt.fit_transform(X_razor)
    
    results = {}
    
    # Tune ThetaTabM
    print(f"\n>>> Tuning ThetaTabM ({timeout}s budget, {n_epochs} epochs) <<<")
    # Load or create study
    theta_study = optuna.create_study(
        direction='maximize', 
        study_name='theta_nas', 
        storage=storage_url, 
        load_if_exists=True
    )
    if len(theta_study.trials) == 0:
        theta_study.enqueue_trial({'hidden_dim': 256, 'depth': 3, 'k': 16}) 
        
    theta_study.optimize(get_thetatabm_objective(X_neural, y_enc, num_classes, epochs=n_epochs), timeout=timeout)
    
    print(f"  > Best Theta Params: {theta_study.best_params}")
    print(f"  > Best CV Score: {theta_study.best_value:.4%}")
    results['thetatabm'] = theta_study.best_params
    results['theta_score'] = theta_study.best_value
    
    # Tune TrueTabR
    print(f"\n>>> Tuning TrueTabR ({timeout}s budget, {n_epochs} epochs) <<<")
    tabr_study = optuna.create_study(
        direction='maximize', 
        study_name='tabr_nas', 
        storage=storage_url, 
        load_if_exists=True
    )
    if len(tabr_study.trials) == 0:
        tabr_study.enqueue_trial({'hidden_dim': 128, 'head_hidden': 64, 'context_size': 96}) 
        
    tabr_study.optimize(get_tabr_objective(X_neural, y_enc, num_classes, epochs=n_epochs), timeout=timeout)
    
    print(f"  > Best TabR Params: {tabr_study.best_params}")
    print(f"  > Best CV Score: {tabr_study.best_value:.4%}")
    results['tabr'] = tabr_study.best_params
    results['tabr_score'] = tabr_study.best_value
    
    # Save Results
    with open('PartD/outputs/nas_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n>>> NAS COMPLETE <<<")
    print(f"Results saved to PartD/outputs/nas_results.json")

if __name__ == "__main__":
    main()
