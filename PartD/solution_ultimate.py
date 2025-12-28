
"""
🌌 SOLUTION_ULTIMATE.PY 🌌
--------------------------------------------------------------------------------
"The Magna Carta of Tabular Learning"
Author: Principal Research Engineer (DeepMind)
Date: 2025

OBJECTIVE:
    Achieve theoretical maximum accuracy (>99%) using Multi-Scale Manifold Learning.

STACK:
    1. 🕸️ MULTI-SCALE TABR: Retrieval at Local (k=5), Regional (k=20), Global (k=50) scales.
    2. 🧬 MANIFOLD TTA: PCA-Whitened Noise Injection (Geometry-Aware Inference).
    3. 🛡️ FOUNDATION: TabPFN (Prior-Data Fitted Network).
    4. ⚖️ SIMPLEX STACKING: Gradient-based Weight Optimization (SciPy).

DEPENDENCIES:
    - tabpfn (Optional/Critical)
    - catboost, scikit-learn, scipy
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize
from catboost import CatBoostClassifier

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
SEED = 42

def seed_everything(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)
print("\n[INIT] Initializing ULTIMATE Protocol (Magna Carta)...")

# ------------------------------------------------------------------------------
# 1. MULTI-SCALE TABR (Tabular Retrieval)
# ------------------------------------------------------------------------------
class MultiScaleTabR(BaseEstimator, ClassifierMixin):
    """
    Retrieves neighbors at multiple scales (k) and metrics (euclidean, cosine).
    Injects the Mean and Std of neighbor labels as features.
    """
    def __init__(self, scales=[5, 20, 50], metrics=['euclidean', 'cosine']):
        self.scales = scales
        self.metrics = metrics
        self.indexers = {} # (metric) -> NearestNeighbors
        self.y_train_ohe = None
        self.clf = None
        
    def fit(self, X, y):
        # One-Hot Encode y for aggregation
        n_classes = len(np.unique(y))
        self.y_train_ohe = np.eye(n_classes)[y]
        
        # Build Indices
        for metric in self.metrics:
            print(f"[TABR] Building {metric} Index on {len(X)} samples...")
            nn = NearestNeighbors(n_neighbors=max(self.scales), metric=metric, n_jobs=-1)
            nn.fit(X)
            self.indexers[metric] = nn
            
        # Enrich Features
        print("[TABR] Generating Multi-Scale Retrieval Features...")
        X_enriched = self._enrich(X)
        
        # Train CatBoost Expert
        print(f"[TABR] Training Retrieval Expert (CatBoost) on {X_enriched.shape[1]} features...")
        self.clf = CatBoostClassifier(
            iterations=2000,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3,
            loss_function='MultiClass',
            verbose=False,
            random_seed=SEED,
            allow_writing_files=False
        )
        self.clf.fit(X_enriched, y)
        return self
        
    def _enrich(self, X):
        features = [X]
        
        for metric in self.metrics:
            # Query once for max k
            dists, indices = self.indexers[metric].kneighbors(X)
            
            for k in self.scales:
                # Slicing indices for current k
                # Note: indices shape (N, max_k)
                current_indices = indices[:, :k]
                
                # Gather neighbor labels: (N, k, n_classes)
                neighbor_labels = self.y_train_ohe[current_indices]
                
                # Compute Stats
                # Mean: (N, n_classes) -> "Local Class Probability"
                mean_feats = np.mean(neighbor_labels, axis=1)
                # Std: (N, n_classes) -> "Local Class Uncertainty"
                std_feats = np.std(neighbor_labels, axis=1)
                
                features.append(mean_feats)
                features.append(std_feats)
                
        return np.hstack(features)

    def predict_proba(self, X):
        X_enriched = self._enrich(X)
        return self.clf.predict_proba(X_enriched)

# ------------------------------------------------------------------------------
# 2. MANIFOLD-AWARE TTA (PCA Whitening)
# ------------------------------------------------------------------------------
class ManifoldTTA:
    def __init__(self, n_aug=10, variance_thresh=0.95):
        self.n_aug = n_aug
        self.variance_thresh = variance_thresh
        self.pca = None
        
    def fit(self, X):
        self.pca = PCA(n_components=self.variance_thresh, random_state=SEED)
        self.pca.fit(X)
        print(f"[TTA] PCA fitted. Retained {self.pca.n_components_} components explaining {np.sum(self.pca.explained_variance_ratio_):.2%} variance.")
        
    def predict(self, model, X):
        # 1. Base Prediction
        preds_sum = model.predict_proba(X)
        
        # 2. Generate Manifold Noise
        # Noise in Latent Space: z ~ N(0, I)
        # Scale by Sqrt(Eigenvalues) to match data distribution spread
        eigenvalues = self.pca.explained_variance_
        latent_std = np.sqrt(eigenvalues)
        
        print(f"[TTA] Initializing Double Bagging ({self.n_aug} augs)...")
        for i in range(self.n_aug):
            # Generate latent noise
            N, D_latent = X.shape[0], self.pca.n_components_
            z = np.random.normal(0, 1.0, (N, D_latent))
            
            # Scale noise magnitude (controlled perturbation)
            # We want 'local' manifold noise, not global. Scale down.
            scale_factor = 0.2 
            z_scaled = z * latent_std * scale_factor
            
            # Inverse Transform to Feature Space
            noise_feature_space = self.pca.inverse_transform(z_scaled)
            
            # Augment
            X_aug = X + noise_feature_space
            
            # Predict
            preds_sum += model.predict_proba(X_aug)
            
        return preds_sum / (self.n_aug + 1)

# ------------------------------------------------------------------------------
# 3. MAIN PIPELINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_data
    
    print("---------------------------------------------------------")
    print("      🏛️ THE MAGNA CARTA: ULTIMATE SOTA EXPERIMENT 🏛️      ")
    print("---------------------------------------------------------")
    
    # 1. Load Data
    X, y, X_test = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(np.unique(y_enc))
    
    # 2. Initialize Components
    models = {}
    
    # A. TabPFN (The Foundation)
    try:
        from tabpfn import TabPFNClassifier
        # N=10000 limit for TabPFN context usually, subsampling handled inside or manually
        # Using ensemble_configurations=64 for MAX accuracy
        models['TabPFN'] = TabPFNClassifier(device='cuda', N_ensemble_configurations=64) 
        print("[MODELS] TabPFN Initialized.")
    except Exception as e:
        print(f"⚠️ TabPFN not found ({e}). Skipping.")
        
    # B. Multi-Scale TabR (The Retrieval Expert)
    models['TabR'] = MultiScaleTabR(scales=[5, 20, 50], metrics=['euclidean', 'cosine'])
    
    # C. Manifold TTA Engine
    tta_engine = ManifoldTTA(n_aug=10)
    
    # 3. Cross-Validation (OOF Generation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_preds = {name: np.zeros((len(X), n_classes)) for name in models}
    test_preds_accum = {name: np.zeros((len(X_test), n_classes)) for name in models}
    
    # Prepare TTA on full training data for Test usage later?
    # Actually, TTA PCA should be fitted on Training data of the fold.
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        print(f"\n--- Magna Carta Fold {fold+1}/5 ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]
        
        # Fit TTA PCA on this fold's train data
        tta_engine.fit(X_train)
        
        for name, clf in models.items():
            print(f"Training {name}...")
            
            if name == 'TabPFN':
                # Subsample for TabPFN fit if necessary (>10k)
                if len(X_train) > 10000:
                    sub_idx = np.random.choice(len(X_train), 10000, replace=False)
                    clf.fit(X_train[sub_idx], y_train[sub_idx])
                else:
                    clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train)
            
            # OOF Inference (No TTA usually strictly for OOF to allow fast optim, 
            # but prompt implies TTA is part of the "Inference Strategy". 
            # We will use TTA for OOF to get 'true' performance for stacking.)
            print(f"[{name}] Predicting OOF with Manifold TTA...")
            oof_preds[name][val_idx] = tta_engine.predict(clf, X_val)
            
            # Test Inference
            print(f"[{name}] Predicting Final Test with Manifold TTA...")
            test_preds_accum[name] += tta_engine.predict(clf, X_test) / 5
            
    # 4. Simplex Stacking optimization
    print("\n[OPTIM] Solving for Simplex Weights (Minimize OOF LogLoss)...")
    
    preds_list = [oof_preds[n] for n in models]
    
    def log_loss_objective(w):
        # Weighted sum of probs
        final_probs = np.zeros_like(preds_list[0])
        for i, p in enumerate(preds_list):
            final_probs += w[i] * p
            
        # Clip to avoid log(0)
        final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
        return log_loss(y_enc, final_probs)
    
    init_w = np.ones(len(models)) / len(models)
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)}) # Sum = 1
    bounds = [(0, 1)] * len(models) # Non-negative
    
    res = minimize(log_loss_objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    
    best_weights = res.x
    print(f"[OPTIM] Optimal Weights: {dict(zip(models.keys(), np.round(best_weights, 4)))}")
    print(f"[OPTIM] Final OOF LogLoss: {res.fun:.5f}")
    
    # 5. Final Assembly
    final_test_probs = np.zeros_like(test_preds_accum['TabR']) # Shape init
    test_preds_list = [test_preds_accum[n] for n in models]
    
    for i, w in enumerate(best_weights):
        final_test_probs += w * test_preds_list[i]
        
    final_preds = np.argmax(final_test_probs, axis=1)
    final_labels = le.inverse_transform(final_preds)
    
    # 6. Sanity Check
    assert final_labels.shape[0] == 6955, f"Shape mismatch: {final_labels.shape}"
    
    # 7. Save
    output_path = 'PartD/outputs/labelsX_ultimate.npy'
    if not os.path.exists('PartD/outputs'): os.makedirs('PartD/outputs')
    
    np.save(output_path, final_labels)
    print(f"\n[SUCCESS] The Magna Carta has been sealed. Predictions at {output_path}")
