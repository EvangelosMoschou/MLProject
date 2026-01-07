
import os
import sys
import json
import optuna
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.models_torch import KAN
from PartD.sigma_omega.utils import seed_everything

# Disable Optuna logging output
optuna.logging.set_verbosity(optuna.logging.INFO)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_kan_objective(X, y, num_classes, epochs=50):
    def objective(trial):
        params = {
            'hidden': trial.suggest_categorical('hidden', [64, 128, 256, 512]),
            'depth': trial.suggest_int('depth', 2, 5),
            'num_classes': num_classes
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = KAN(**params)
            model.fit(X_tr, y_tr, epochs=epochs)
            
            probs = model.predict_proba(X_val)
            preds = np.argmax(probs, axis=1)
            acc = (preds == y_val).mean()
            scores.append(acc)
            
        return np.mean(scores)
    return objective

def main():
    print(">>> TUNING KOLMOGOROV-ARNOLD NETWORK (KAN) [PERSISTENT] <<<")
    
    # Check for smoke run
    is_smoke = os.getenv('SMOKE_RUN', 'False').lower() in ('true', '1', 'yes')
    timeout = 60 if is_smoke else 3600  # 1 hour budget for KAN
    n_epochs = 1 if is_smoke else 50
    
    # DB Storage
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
    
    # Quantile Transform
    print("[PREPROC] Applying Gaussian Quantile Transform...")
    qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal', random_state=42)
    X_neural = qt.fit_transform(X_razor)
    
    print(f"\n>>> Optimizing KAN ({timeout}s budget, {n_epochs} epochs) <<<")
    kan_study = optuna.create_study(
        direction='maximize', 
        study_name='kan_nas', 
        storage=storage_url, 
        load_if_exists=True
    )
    if len(kan_study.trials) == 0:
        kan_study.enqueue_trial({'hidden': 64, 'depth': 2}) # Default baseline
        
    kan_study.optimize(get_kan_objective(X_neural, y_enc, num_classes, epochs=n_epochs), timeout=timeout)
    
    print(f"  > Best KAN Params: {kan_study.best_params}")
    print(f"  > Best CV Score: {kan_study.best_value:.4%}")
    
    results = {'kan': kan_study.best_params, 'kan_score': kan_study.best_value}
    
    with open('PartD/outputs/kan_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
