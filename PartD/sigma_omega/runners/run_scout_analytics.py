#!/usr/bin/env python3
"""
Scout Analytics Runner 🕵️
=========================
Fast diagnostic tool to identify Class Confusions and Drift WITHOUT running the full pipeline.
Execution time: < 30 seconds.

Usage:
    python -m sigma_omega.runners.run_scout_analytics
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from ..data import load_data_safe
from ..features import find_confusion_pairs, ConfusionScout, AdversarialDriftRemover
from ..utils import seed_everything

def main():
    start_time = time.time()
    print(">>> SCOUT ANALYTICS PROTOCOL <<<")
    
    # 1. Load Data
    print("1. Loading Data...")
    X, y, X_test = load_data_safe()
    print(f"   Shape: {X.shape}")
    
    # Check for Drift First
    print("\n2. Checking for Covariate Shift (Drift)...")
    drift_detector = AdversarialDriftRemover(threshold=0.70)
    # We fit it to see what it would drop
    # (Re-implementing simplified logic here for reporting)
    n_tr = len(X)
    n_te = len(X_test)
    min_n = min(n_tr, n_te)
    idx_tr = np.random.choice(n_tr, min_n, replace=False)
    idx_te = np.random.choice(n_te, min_n, replace=False)
    X_adv = np.vstack([X[idx_tr], X_test[idx_te]])
    y_adv = np.concatenate([np.zeros(min_n), np.ones(min_n)])
    
    rf = LogisticRegression(max_iter=100) # Fast proxy instead of RF
    rf.fit(X_adv, y_adv)
    auc = roc_auc_score(y_adv, rf.predict_proba(X_adv)[:, 1])
    print(f"   [DRIFT] Dataset Global AUC: {auc:.4f} (0.5 = No Shift, >0.7 = Strong Shift)")
    
    # 3. Find Confusion Pairs
    print("\n3. Identifying Confusion Hotspots...")
    # Use find_confusion_pairs from features.py
    pairs = find_confusion_pairs(X, y, top_k=5)
    
    # 4. Evaluate Scout Performance
    print("\n4. Evaluating Scout Discriminability...")
    print("   (Can we actually separate these pairs linearly detailed?)")
    
    for (c1, c2) in pairs:
        # Subset data
        mask = (y == c1) | (y == c2)
        X_sub = X[mask]
        y_sub = y[mask]
        y_bin = (y_sub == c2).astype(int)
        
        # 5-Fold CV Accuracy
        model = LogisticRegression(C=1.0, max_iter=200)
        preds = cross_val_predict(model, X_sub, y_bin, cv=5)
        acc = accuracy_score(y_bin, preds)
        
        # If Acc is low (~0.5), they are indistinguishable (Hard Confusion)
        # If Acc is high (>0.85), they are separable (Model just needs help)
        status = "SEPARABLE" if acc > 0.8 else "HARD CONFUSION"
        print(f"   Pair {c1} vs {c2}: Scout Accuracy = {acc:.2%} [{status}]")

    elapsed = time.time() - start_time
    print(f"\n>>> ANALYSIS COMPLETE in {elapsed:.2f} seconds <<<")

if __name__ == "__main__":
    main()
