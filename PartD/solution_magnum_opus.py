
"""
🌌 SOLUTION_MAGNUM_OPUS.PY 🌌
--------------------------------------------------------------------------------
"The Magnum Opus: Uncertainty-Gated Manifold Expansion"
Author: Principal AI Research Scientist (DeepMind)
Date: 2025

OBJECTIVE:
    Bridge the Train-Test Distribution Gap using Self-Training with Bayesian Veto.

STACK:
    1. 🧱 FOUNDATION: TabPFN (Static Bayesian Prior)
    2. 🕸️ LEARNER: Multi-Scale TabR (Recursive Retrieval)
    3. 🔍 INFERENCE: Manifold-Aware TTA (Epistemic Uncertainty)
    4. 🧬 LOOP: Tri-Fold Veto (Confidence + Stability + Consensus)

DEPENDENCIES:
    - tabpfn (Critical for Veto)
    - catboost, scikit-learn, scipy
"""

import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
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
print("\n[INIT] Initializing MAGNUM OPUS Protocol...")

# ------------------------------------------------------------------------------
# 1. MULTI-SCALE TABR (Recursive)
# ------------------------------------------------------------------------------
class RecursiveTabR(BaseEstimator, ClassifierMixin):
    def __init__(self, scales=[5, 10, 20, 50]):
        self.scales = scales
        self.indexers = {}
        self.y_train_ohe = None
        self.clf = None
        
    def fit(self, X, y):
        # 1. Build Indices on Current Knowledge Base (Train + Pseudo)
        # Note: X, y here will grow in the loop
        
        n_classes = len(np.unique(y))
        # Ensure y is proper integer
        y = y.astype(int)
        
        # Handle OHE efficiently
        # Creates (N, C)
        self.y_train_ohe = np.zeros((len(y), n_classes))
        self.y_train_ohe[np.arange(len(y)), y] = 1
        
        # Build Index (Euclidean only for speed in Loop, or both? Prompt said Euclidean/Cosine logic earlier, 
        # but for Loop speed let's stick to Euclidean which is standard TabR)
        # "Implement Multi-Scale TabR" -> ok lets do Euclidean
        self.nn = NearestNeighbors(n_neighbors=max(self.scales), n_jobs=-1)
        self.nn.fit(X)
        
        # 2. Enrich
        X_enriched = self._enrich(X)
        
        # 3. Train
        self.clf = CatBoostClassifier(
            iterations=1500, # Slightly faster for loop
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            loss_function='MultiClass',
            verbose=False,
            random_seed=SEED,
            allow_writing_files=False,
            thread_count=-1
        )
        self.clf.fit(X_enriched, y)
        return self

    def _enrich(self, X):
        # Queries neighbors
        # Distances not strictly needed, just indices
        _, indices = self.nn.kneighbors(X)
        
        features = [X]
        for k in self.scales:
            curr_idx = indices[:, :k]
            # Gather labels
            neighbor_labels = self.y_train_ohe[curr_idx] # (N, k, C)
            
            # Mean & Std
            mean_f = np.mean(neighbor_labels, axis=1) # (N, C)
            std_f = np.std(neighbor_labels, axis=1)   # (N, C)
            
            features.append(mean_f)
            features.append(std_f)
            
        return np.hstack(features)

    def predict_proba(self, X):
        X_enriched = self._enrich(X)
        return self.clf.predict_proba(X_enriched)

# ------------------------------------------------------------------------------
# 2. MANIFOLD TTA (Variance Aware)
# ------------------------------------------------------------------------------
class ManifoldTTA:
    def __init__(self, n_aug=10):
        self.n_aug = n_aug
        self.pca = None
        
    def fit(self, X):
        self.pca = PCA(n_components=0.95, random_state=SEED)
        self.pca.fit(X)
        
    def predict_uncertainty(self, model, X):
        """
        Returns:
            mean_probs: (N, C)
            variance: (N,) - Max variance across classes or Mean var?
            Prompt: "Return ... Variance of TTA predictions"
            Let's use Mean Variance across classes.
        """
        preds_list = []
        
        # Original
        preds_list.append(model.predict_proba(X))
        
        # Augmented
        eigenvalues = self.pca.explained_variance_
        latent_std = np.sqrt(eigenvalues)
        
        for i in range(self.n_aug):
            z = np.random.normal(0, 1.0, (X.shape[0], self.pca.n_components_))
            z_scaled = z * latent_std * 0.2 # Scale factor 0.2
            noise = self.pca.inverse_transform(z_scaled)
            preds_list.append(model.predict_proba(X + noise))
            
        # Stack: (Attr, N, C)
        stack = np.array(preds_list)
        
        # Compute Stats
        mean_probs = np.mean(stack, axis=0) # (N, C)
        var_probs = np.var(stack, axis=0)   # (N, C)
        
        # Scalar stability score: Mean variance across all class probabilities
        stability_score = np.mean(var_probs, axis=1) # (N,)
        
        return mean_probs, stability_score

# ------------------------------------------------------------------------------
# 3. MAIN MAGNUM OPUS
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_data
    
    print("---------------------------------------------------------")
    print(" 🍷 THE MAGNUM OPUS: UNCERTAINTY-GATED EXPANSION 🍷    ")
    print("---------------------------------------------------------")
    
    # 1. Load Data
    X_train_orig, y_train_orig, X_test_orig = load_data()
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_orig)
    n_classes = len(np.unique(y_train_enc))
    
    # 2. PHASE 1: THE FOUNDATION (Prior) with Double Bagging
    print("\n[PHASE 1] Establishing Bayesian Prior (TabPFN with Double Bagging)...")
    prior_probs_test = None
    
    try:
        from tabpfn import TabPFNClassifier
        
        # Double Bagging: 5 iterations of Subsampling + Feature Shuffling
        n_bags = 5
        pfn_preds_accum = np.zeros((len(X_test_orig), n_classes))
        
        print(f"[PRIOR] Executing {n_bags}-Fold Monte Carlo Integration...")
        
        for bag in range(n_bags):
             # 1. Feature Shuffling (Mitigates Position Bias)
             n_feats = X_train_orig.shape[1]
             perm = np.random.permutation(n_feats)
             X_tr_perm = X_train_orig[:, perm]
             X_te_perm = X_test_orig[:, perm]
             
             # 2. Row Subsampling (90%)
             n_samples = len(X_train_orig)
             # Cap at 8000 for speed/memory if needed, else 90%
             n_sub = int(0.9 * n_samples)
             if n_sub > 10000: n_sub = 10000 
             
             idx = np.random.choice(n_samples, n_sub, replace=False)
             X_bag, y_bag = X_tr_perm[idx], y_train_enc[idx]
             
             # 3. Fit & Predict
             pfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
             pfn.fit(X_bag, y_bag)
             pfn_preds_accum += pfn.predict_proba(X_te_perm)
             
        prior_probs_test = pfn_preds_accum / n_bags
        print("[PRIOR] TabPFN Anchor Established (Double Bagged).")
            
    except Exception as e:
        print(f"⚠️ [PRIOR] TabPFN Failed ({e}). Proceeding without Consensus Veto.")
        prior_probs_test = None

    # 3. PHASE 3: THE LOOP (Self-Training)
    # We iterate 3 times
    
    # Current Training Set (Starts with original)
    X_curr = X_train_orig.copy()
    y_curr = y_train_enc.copy()
    
    # Unlabeled Pool Mask (Starts with all Test)
    # Actually, we don't remove from test set for final prediction, but we track which are 'pseudo-labeled' 
    # to avoid re-adding them? Or just check if already added?
    # Easier: Maintain mask of Test indices that have been converted.
    converted_mask = np.zeros(len(X_test_orig), dtype=bool)
    
    # TTA Engine (Fit once on original Train)
    tta = ManifoldTTA(n_aug=10)
    tta.fit(X_train_orig)
    
    print("\n[PHASE 3] Starting Uncertainty-Gated Expansion Loop (3 Cycles)...")
    
    for cycle in range(1, 4):
        print(f"\n--- [CYCLE {cycle}] Training Learner on {len(X_curr)} samples ---")
        
        # A. Train Learner (TabR)
        learner = RecursiveTabR(scales=[5, 10, 20, 50])
        learner.fit(X_curr, y_curr)
        
        # B. Predict on Test (Unconverted only? No, predict all to get full picture)
        # Actually, prioritize unconverted for candidates
        test_probs, test_vars = tta.predict_uncertainty(learner, X_test_orig)
        
        # C. The Tri-Fold Veto
        # Criteria A: Confidence > 0.99
        max_conf = np.max(test_probs, axis=1)
        pred_labels = np.argmax(test_probs, axis=1)
        
        cond_a = max_conf > 0.99
        
        # Criteria B: Stability < 0.005 (Stricter Veto)
        cond_b = test_vars < 0.005

        
        # Criteria C: Consensus (if PFN available)
        if prior_probs_test is not None:
            pfn_labels = np.argmax(prior_probs_test, axis=1)
            cond_c = (pred_labels == pfn_labels)
        else:
            cond_c = np.ones(len(X_test_orig), dtype=bool) # Skip if no PFN
            
        # Initial Candidates
        candidates = cond_a & cond_b & cond_c
        
        # Exclude already converted
        new_candidates = candidates & (~converted_mask)
        
        n_new = np.sum(new_candidates)
        print(f"[CYCLE {cycle}] Candidates found: {n_new}")
        
        if n_new == 0:
            print("[CYCLE {cycle}] No new Golden Samples found. Early Stopping Loop.")
            break
            
        # D. Manifold Injection
        # Add to X_curr, y_curr
        X_new = X_test_orig[new_candidates]
        y_new = pred_labels[new_candidates]
        
        X_curr = np.vstack([X_curr, X_new])
        y_curr = np.concatenate([y_curr, y_new])
        
        converted_mask[new_candidates] = True
        
        print(f"[CYCLE {cycle}] Added {n_new} Golden Samples. Manifold Densified.")
        
    
    # 4. PHASE 4: GRAND FINALE
    print("\n[PHASE 4] The Final Ensemble...")
    # Train Final Learner on Densified Manifold
    final_learner = RecursiveTabR(scales=[5, 10, 20, 50])
    final_learner.fit(X_curr, y_curr)
    
    # Predict on Test (Standard TTA)
    final_learner_probs, _ = tta.predict_uncertainty(final_learner, X_test_orig)
    
    # Optimize Weights (PFN + Final Learner)
    # Note: We need OOF for optimization. But we just did self-training on Test.
    # We can't use Test probs for weight finding (Need Labels).
    # Strategy: Use Validation Set? Or just OOF on original Train?
    # Simpler Grandmaster trick: Fixed high weight for Learned Manifold if it passed sanity checks.
    # Or: 50/50 blend if PFN exists.
    # Prompt: "Use Hill Climbing ... on OOF predictions".
    # This implies we needed an OOF loop for the Self-Training? That's computationally massive (Nest OOF inside Self Train loop?).
    # Interpretation: The "Loop" is for generating the final model.
    # To get weights, we likely need OOF predictions from the *Final* model on the *Original* training data.
    
    print("[FINALE] Generating OOF for Optimization...")
    # We generate OOF on original Train using the Final Learner (trained on Train+Pseudo)
    # Caution: Leakage! Learner trained on Train includes sample i.
    # Correct Way: Self-Training usually generates a single strong model.
    # We will blend Final Learner (Pseudo-Enhanced) with Static TabPFN.
    # Weights? Let's use 0.6 Learner (it saw test data), 0.4 Prior.
    # Or use OOF from *initial* cycle to estimate weights?
    # Let's trust the "Prompt Requirement": Hill Climbing.
    # We will generate OOF predictions for TabPFN (done internally usually or we split) and Learner (Initial).
    # Actually, simpler: Use 0.5 * PFN + 0.5 * Learner.
    # Optimization code requires labels. We only have Train labels.
    # We use StratifiedKFold on Train to get OOF scores for PFN and Learner (Pre-Pseudo).
    # Then assume those weights hold for Post-Pseudo.
    
    # Just blend 0.6 (Learner) / 0.4 (PFN) for robustness if optimization is complex here.
    # Implementing rigorous OOF optimization now would require re-running the whole loop K times. Too slow.
    # Prompt says "Train the final ensembles ... Use Hill Climbing on OOF".
    # We will run a quick CV on the *Expanded* dataset? No, we don't have labels for expansion (pseudo).
    # We run CV on *Original* dataset to find weights between [PFN, Learner_Base].
    # Then apply those weights to [PFN, Learner_Final].
    
    # Quick CV for Weights
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pfn = np.zeros((len(X_train_orig), n_classes))
    oof_learner = np.zeros((len(X_train_orig), n_classes))
    
    print("[FINALE] Running Quick CV on Original Data to tune PFN/Learner balance...")
    for train_ix, val_ix in skf.split(X_train_orig, y_train_enc):
        x_t, y_t = X_train_orig[train_ix], y_train_enc[train_ix]
        x_v = X_train_orig[val_ix]
        
        # Learner (Base)
        lrn = RecursiveTabR()
        lrn.fit(x_t, y_t)
        oof_learner[val_ix] = lrn.predict_proba(x_v)
        
        # PFN (Small sample fit)
        if prior_probs_test is not None:
             if len(x_t) > 2000: # Fast proxy
                 sidx = np.random.choice(len(x_t), 2000, replace=False)
                 pfn.fit(x_t[sidx], y_t[sidx])
             else:
                 pfn.fit(x_t, y_t)
             oof_pfn[val_ix] = pfn.predict_proba(x_v)
             
    # Optimize
    if prior_probs_test is not None:
        def obj(w):
            w = w / np.sum(w)
            p = w[0]*oof_learner + w[1]*oof_pfn
            return log_loss(y_train_enc, p)
            
        res = minimize(obj, [0.5, 0.5], bounds=[(0,1),(0,1)], constraints={'type':'eq', 'fun': lambda w: 1-sum(w)})
        w_best = res.x / np.sum(res.x)
        print(f"[FINALE] Optimal Weights -> Learner: {w_best[0]:.3f}, Prior: {w_best[1]:.3f}")
        
        final_probs = w_best[0] * final_learner_probs + w_best[1] * prior_probs_test
        
    else:
        print("[FINALE] Prior missing. Utilizing Learner 100%.")
        final_probs = final_learner_probs

    # Save
    final_preds = np.argmax(final_probs, axis=1)
    final_labels = le.inverse_transform(final_preds)
    
    assert final_labels.shape[0] == 6955
    out_path = 'PartD/outputs/labelsX_magnum_opus.npy'
    if not os.path.exists('PartD/outputs'): os.makedirs('PartD/outputs')
    np.save(out_path, final_labels)
    
    print(f"\n[SUCCESS] Magnum Opus Complete. Predictions at {out_path}")
