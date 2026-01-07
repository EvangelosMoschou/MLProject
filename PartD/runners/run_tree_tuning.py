
import os
import sys
import json
import optuna
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.utils import seed_everything

# Disable Optuna logging output
optuna.logging.set_verbosity(optuna.logging.INFO)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_xgb_objective(X, y, num_classes):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': 500, # Fixed budget per trial
            'booster': 'dart',
            'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'tree_method': 'hist',
            'device': DEVICE,
            'verbosity': 0
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            acc = (preds == y_val).mean()
            scores.append(acc)
            
        return np.mean(scores)
    return objective

def get_cat_objective(X, y, num_classes):
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'iterations': 500, # Fixed budget
            'langevin': True, # Enable Langevin
            'diffusion_temperature': trial.suggest_float('diffusion_temperature', 100, 5000),
            'loss_function': 'MultiClass',
            'task_type': 'GPU' if DEVICE == 'cuda' else 'CPU',
            'verbose': 0,
            'allow_writing_files': False
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val).flatten()
            acc = (preds == y_val).mean()
            scores.append(acc)
            
        return np.mean(scores)
    return objective

def main():
    print(">>> TUNING TITANS (XGBoost & CatBoost) [PERSISTENT] <<<")
    
    is_smoke = os.getenv('SMOKE_RUN', 'False').lower() in ('true', '1', 'yes')
    timeout = 60 if is_smoke else 1800 # 30 mins each
    
    storage_url = "sqlite:///PartD/outputs/nas.db"
    
    # 1. Load & Razor
    X, y, _ = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print("[RAZOR] Scanning...")
    scout = CatBoostClassifier(iterations=100, verbose=0, task_type='GPU' if DEVICE == 'cuda' else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    X_razor = X[:, imps > np.percentile(imps, 10)]
    
    results = {}
    
    # 2. Tune XGBoost
    print(f"\n>>> Tuning XGBoost (DART) ({timeout}s budget) <<<")
    xgb_study = optuna.create_study(direction='maximize', study_name='xgb_tuning', storage=storage_url, load_if_exists=True)
    if len(xgb_study.trials) == 0:
        xgb_study.enqueue_trial({'max_depth': 8, 'learning_rate': 0.066}) # Current default guess
    xgb_study.optimize(get_xgb_objective(X_razor, y_enc, num_classes), timeout=timeout)
    print(f"  > Best XGB: {xgb_study.best_value:.4%}")
    results['xgb'] = xgb_study.best_params
    
    # 3. Tune CatBoost
    print(f"\n>>> Tuning CatBoost (Langevin) ({timeout}s budget) <<<")
    cat_study = optuna.create_study(direction='maximize', study_name='cat_tuning', storage=storage_url, load_if_exists=True)
    if len(cat_study.trials) == 0:
        cat_study.enqueue_trial({'depth': 9, 'learning_rate': 0.0485}) # Current default guess
    cat_study.optimize(get_cat_objective(X_razor, y_enc, num_classes), timeout=timeout)
    print(f"  > Best Cat: {cat_study.best_value:.4%}")
    results['cat'] = cat_study.best_params
    
    # Save
    with open('PartD/outputs/tree_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
