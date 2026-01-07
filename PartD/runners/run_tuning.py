
import os
import sys
# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import optuna
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.utils import seed_everything

# Disable Optuna logging output
optuna.logging.set_verbosity(optuna.logging.INFO)

def get_xgb_objective(X, y, num_classes):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'verbosity': 0,
            'seed': 42
        }
        
        # 3-Fold for speed during search
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        cv_scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Use accuracy
            preds = model.predict(X_val)
            acc = (preds == y_val).mean()
            cv_scores.append(acc)
            
        return np.mean(cv_scores)
    return objective

def get_cat_objective(X, y, num_classes):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
            'loss_function': 'MultiClass',
            'verbose': 0,
            'random_seed': 42
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
            
            preds = model.predict(X_val).flatten()
            acc = (preds == y_val).mean()
            cv_scores.append(acc)
            
        return np.mean(cv_scores)
    return objective

def main():
    print(">>> STARTING OPTUNA HYPERPARAMETER TUNING <<<")
    
    # Data Loading (Same as main pipeline)
    X, y, _ = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Razor (10% conservative)
    print("[RAZOR] Scanning for noise features...")
    scout = CatBoostClassifier(iterations=100, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    thresh = np.percentile(imps, 10)
    keep_mask = imps > thresh
    X_razor = X[:, keep_mask]
    print(f"  > Dropped {np.sum(~keep_mask)} features. New Dim: {X_razor.shape[1]}")
    
    results = {}
    
    # Check for smoke run
    is_smoke = os.getenv('SMOKE_RUN', 'False').lower() in ('true', '1', 'yes')
    timeout = 60 if is_smoke else 5400  # 1 minute for smoke, 1.5 hours for real
    
    # XGBoost Tuning
    print(f"\n>>> Tuning XGBoost ({timeout}s budget) <<<")
    xgb_study = optuna.create_study(direction='maximize', study_name='xgb_optimization')
    # Add default params as baseline trial
    xgb_study.enqueue_trial({
        'n_estimators': 500,
        'max_depth': 6, 
        'learning_rate': 0.01,
        'subsample': 0.8, 
        'colsample_bytree': 0.8
    })
    xgb_study.optimize(get_xgb_objective(X_razor, y_enc, num_classes), timeout=timeout)
    
    print(f"  > Best XGB Params: {xgb_study.best_params}")
    print(f"  > Best CV Score: {xgb_study.best_value:.4%}")
    results['xgboost'] = xgb_study.best_params
    results['xgboost_score'] = xgb_study.best_value
    
    # CatBoost Tuning
    print(f"\n>>> Tuning CatBoost ({timeout}s budget) <<<")
    cat_study = optuna.create_study(direction='maximize', study_name='cat_optimization')
    cat_study.enqueue_trial({
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.03
    })
    cat_study.optimize(get_cat_objective(X_razor, y_enc, num_classes), timeout=timeout)
    
    print(f"  > Best CatParams: {cat_study.best_params}")
    print(f"  > Best CV Score: {cat_study.best_value:.4%}")
    results['catboost'] = cat_study.best_params
    results['catboost_score'] = cat_study.best_value
    
    # Save Results
    os.makedirs('PartD/outputs', exist_ok=True)
    with open('PartD/outputs/optuna_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n>>> TUNING COMPLETE <<<")
    print(f"Results saved to PartD/outputs/optuna_results.json")

if __name__ == "__main__":
    main()
