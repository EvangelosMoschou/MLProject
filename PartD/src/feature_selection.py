import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from .models import get_xgb_model, get_rf_model
from .utils import log_to_history

def run_feature_selection(X, y):
    """
    Runs Permutation Importance to identify and remove noisy features.
    Uses XGBoost (or RF) as the estimator.
    """
    print("--- Running Feature Selection (Permutation Importance) ---")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use a strong tree-based model
    # We prefer XGBoost if available, else RF
    try:
        clf = get_xgb_model()
        model_name = "XGBoost"
    except:
        clf = get_rf_model()
        model_name = "RandomForest"
        
    print(f"Training {model_name} for feature importance analysis...")
    clf.fit(X_train, y_train)
    
    baseline_acc = clf.score(X_val, y_val)
    print(f"Baseline Accuracy with all features: {baseline_acc:.4f}")
    
    print("Calculating permutation importance...")
    # n_repeats=10 for stability
    result = permutation_importance(clf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Organize results
    perm_sorted_idx = result.importances_mean.argsort()
    
    print("\nFeature Importances (Top=Most Important, Bottom=Least):")
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    important_features = []
    dropped_features = []
    
    # Log detailed importance
    importance_log = {}
    
    for i in perm_sorted_idx:
        name = feature_names[i]
        score = result.importances_mean[i]
        std = result.importances_std[i]
        
        importance_log[name] = float(score)
        
        if score <= 0:
            print(f"❌ {name}: {score:.5f} +/- {std:.5f} (Useless/Harmful)")
            dropped_features.append(i)
        else:
            print(f"✅ {name}: {score:.5f} +/- {std:.5f}")
            important_features.append(i)
            
    print(f"\nIdentified {len(dropped_features)} useless features.")
    
    # Log to history
    log_to_history(
        experiment_name='Feature-Selection-Analysis',
        params={'model': model_name, 'n_repeats': 10},
        metrics={'n_dropped': len(dropped_features), 'baseline_acc': baseline_acc, 'importances': importance_log}
    )
    
    return important_features
