import copy
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import nnls

from . import config
from .losses import prob_meta_features
from .features import apply_feature_view, build_streams, GeometricFeatureGenerator, AnomalyFeatureGenerator, ConfusionScout, find_confusion_pairs, CrossFoldFeatureGenerator
from .generative import synthesize_data
from .postprocessing import neutralize_predictions, align_probabilities

def tabpfn_predict_proba(X_train, y_train, X_eval, n_ensembles=32, seed=42):
    """Prediction helper for TabPFN (Legacy/Fallback use)."""
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        raise RuntimeError("USE_TABPFN=1 but `tabpfn` is not installed/importable.") from e
        
    model = TabPFNClassifier(device=str(config.DEVICE), n_estimators=int(n_ensembles), random_state=int(seed))
    model.fit(X_train, y_train) # Fit just stores data usually
    return model.predict_proba(X_eval).astype(np.float32)


def hill_climbing_optimization(oof_preds, y_true, iterations=100):
    """
    Hill Climbing for Accuracy Optimization.
    Finds optimal weights by iteratively adding the model that maximizes accuracy.
    """
    n_models = len(oof_preds)
    best_weights = np.zeros(n_models)
    current_ensemble = np.zeros_like(oof_preds[0])
    
    # Initialize with best single model
    best_acc = 0
    best_idx = -1
    for i in range(n_models):
        acc = accuracy_score(y_true, np.argmax(oof_preds[i], axis=1))
        if acc > best_acc:
            best_acc = acc
            best_idx = i
            
    current_ensemble += oof_preds[best_idx]
    best_weights[best_idx] += 1
    
    print(f"   [HillClimb] Start: Model {best_idx} (Acc: {best_acc:.4f})")
    
    for it in range(iterations):
        best_gain = -1
        best_candidate = -1
        
        for i in range(n_models):
            # Try adding model i
            temp_ensemble = current_ensemble + oof_preds[i]
            # Normalize implicitly by argmax (scale doesn't matter for hard voting)
            # Actually for probabilities, mean is (A+B)/2. Argmax(A+B) is same as Argmax((A+B)/2).
            # So simple addition is sufficient.
            
            y_pred = np.argmax(temp_ensemble, axis=1)
            acc = accuracy_score(y_true, y_pred)
            
            if acc > best_acc:
                best_gain = acc - best_acc
                best_acc = acc
                best_candidate = i
        
        if best_candidate != -1:
            current_ensemble += oof_preds[best_candidate]
            best_weights[best_candidate] += 1
            # print(f"     Iter {it+1}: Added Model {best_candidate} -> Acc: {best_acc:.4f}")
        else:
            # Convergence
            print(f"   [HillClimb] Converged at iter {it}. Acc: {best_acc:.4f}")
            break
            
    return best_weights / best_weights.sum()


def rank_average(preds_list):
    """
    Rank Averaging: Convert probs to ranks, average, convert back.
    Robust to different model calibrations.
    """
    from scipy.stats import rankdata
    
    n_models = len(preds_list)
    n_samples, n_classes = preds_list[0].shape
    
    # Rank each model's predictions per class
    ranked = np.zeros((n_samples, n_classes))
    for preds in preds_list:
        for c in range(n_classes):
            ranked[:, c] += rankdata(preds[:, c])
    
    # Average ranks
    ranked /= n_models
    
    # Normalize to probabilities (softmax of ranks)
    ranked = (ranked - ranked.mean(axis=1, keepdims=True)) / ranked.std(axis=1, keepdims=True)
    e = np.exp(ranked)
    return e / e.sum(axis=1, keepdims=True)


def geometric_mean(preds_list):
    """
    Power Average: Geometric Mean of probabilities.
    Punishes disagreement harder than arithmetic mean.
    """
    preds_list = [np.clip(p, 1e-8, 1.0) for p in preds_list]
    log_sum = np.sum([np.log(p) for p in preds_list], axis=0)
    geo = np.exp(log_sum / len(preds_list))
    return geo / geo.sum(axis=1, keepdims=True)


def optimize_class_thresholds(oof_probs, y_true):
    """
    Learn a weight vector W (shape n_classes) to multiply probs.
    Maximize ACCURACY (since OOF is calibrated/soft).
    """
    from scipy.optimize import minimize
    n_classes = oof_probs.shape[1]
    
    def loss_func(w):
        # w is [n_classes]
        # Weighted Probs
        p_w = oof_probs * w
        y_pred = np.argmax(p_w, axis=1)
        return -np.mean(y_pred == y_true) # Minimize negative accuracy
        
    res = minimize(loss_func, x0=np.ones(n_classes), method='Nelder-Mead', tol=1e-4) # Nelder-Mead robust for non-diff argmax
    print(f"   [Thresholds] Optimization Result: Acc {-res.fun:.4f} (Base: {np.mean(np.argmax(oof_probs, axis=1) == y_true):.4f})")
    return res.x


def ensemble_failure_analysis(preds, y_true):
    """Print diagnostics for high-confidence errors."""
    y_pred = np.argmax(preds, axis=1)
    probs = np.max(preds, axis=1)
    
    mask_error = (y_pred != y_true)
    mask_conf = (probs > 0.90)
    
    n_errors = mask_error.sum()
    n_conf_errors = (mask_error & mask_conf).sum()
    
    print(f"      [DIAGNOSTICS] Total Errors: {n_errors}/{len(y_true)} ({n_errors/len(y_true):.2%})")
    print(f"      [DIAGNOSTICS] High Confidence (>90%) Errors: {n_conf_errors}")
    
    if n_conf_errors > 0:
        # Show top confusion pairs
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        np.fill_diagonal(cm, 0) # clear correct
        # simple flattening
        pairs = np.argwhere(cm > 0)
        # Sort by count
        counts = cm[pairs[:,0], pairs[:,1]]
        sort_idx = np.argsort(-counts)
        print("      [DIAGNOSTICS] Top Confusions (True -> Pred):")
        for i in range(min(5, len(sort_idx))):
            idx = sort_idx[i]
            t, p = pairs[idx]
            print(f"        {t} -> {p}: {counts[idx]} samples")


def fit_predict_stacking(
    names_models,
    view_name,
    X_train_base,
    X_test_base,
    y,
    num_classes,
    cv_splits=10,
    seed=42,
    sample_weight=None,
    pseudo_idx=None,
    pseudo_y=None,
    pseudo_w=None,
    return_oof=False,
    X_train_raw=None,  # Raw data for TabPFN (bypasses all feature transforms)
    X_test_raw=None,   # Raw data for TabPFN
):
    """
    Cross-Fit Stacking with Meta-Learner Optimization Support.
    Performs Feature Engineering (View + Streams) *inside* the Cross-Validation Loop.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    n_models = len(names_models)
    model_map = {name: i for i, (name, _) in enumerate(names_models)}
    
    oof_preds = [np.zeros((len(y), num_classes), dtype=np.float32) for _ in range(n_models)]
    test_preds_running = [np.zeros((len(X_test_base), num_classes), dtype=np.float32) for _ in range(n_models)]

    print(f"  [STACKING] Cross-Validation ({cv_splits} folds) | View: {view_name} | Models: {len(names_models)}")
    
    if config.TUNE_HYPERPARAMS:
        print("   >>> [OPTUNA] Hyperparameter Tuning Mode Enabled (Stub) <<<")
        # Optimization logic would go here:
        # study = optuna.create_study(direction='maximize')
        # study.optimize(lambda trial: objective(trial, X_train_base, y), n_trials=50)
        # For now, proceeding with fixed params + Geometry.
    
    # Transform View (Split handled inside loop usually for strict correctness, but optimization allows global transform if transductive)
    # [FIX] Moved inside loop to prevent leakage unless transductive logic is explicitly allowed.
    # We still need a reference for SKF split.
    
    for tr_idx, val_idx in skf.split(X_train_base, y):
        # 1. SPLIT DATA (RAW)
        X_tr_raw_fold = X_train_base[tr_idx]
        X_val_raw_fold = X_train_base[val_idx]
        y_tr = y[tr_idx]

        # 2. APPLY VIEW & ENGINEERING (Inside Fold)
        # Check leakage warning if transductive is ON
        if config.ALLOW_TRANSDUCTIVE:
             pass # Warning printed in apply_feature_view usually, or accepted risk.
        
        # Fit on Fold Train, Transform Fold Val & Test
        X_tr_fold, X_val_fold = apply_feature_view(
            X_tr_raw_fold, X_val_raw_fold, view=view_name, seed=seed, allow_transductive=config.ALLOW_TRANSDUCTIVE
        )
        
        # We also need to transform the FULL Test set based on this Fold's pipeline?
        # Standard Stacking: Use the model trained on this fold to predict on the FULL test set.
        # But the Feature Pipeline (PCA/Quantile) needs to be consistent.
        # Correct Approach: Fit pipeline on X_tr_raw_fold, transform X_test_base.
        # apply_feature_view returns (X_train, X_validation/test).
        # We need a 3-way split helper or call it twice?
        # apply_feature_view signature: (X_train, X_test, ...)
        
        # Call 1: Train/Val
        # Done above.
        
        # Call 2: Train/TestFull (To get X_test_view_fold consistent with this fold's scaler/pca)
        # Re-fitting on X_tr_raw_fold is necessary.
        _, X_test_view_fold = apply_feature_view(
             X_tr_raw_fold, X_test_base, view=view_name, seed=seed, allow_transductive=config.ALLOW_TRANSDUCTIVE
        )
        
        sw_tr = None
        if sample_weight is not None:
            sw_tr = sample_weight[tr_idx]

        # [OMEGA] Decoupled Diffusion Augmentation
        # 1. Augment VIEW data (for standard models)
        X_tr_aug, y_tr_aug, sw_tr_aug = X_tr_fold, y_tr, sw_tr
        
        # 2. Augment RAW data (for TabPFN) - Independent Stream
        X_tr_raw_aug, y_tr_raw_aug, sw_tr_raw_aug = X_tr_raw_fold, y_tr, sw_tr

        if config.ENABLE_DIFFUSION and len(X_tr_fold) > 100:
            print(f"   [DIFFUSION] Augmenting View stream...")
            X_tr_aug, y_tr_aug = synthesize_data(X_tr_fold, y_tr, n_new_per_class=config.DIFFUSION_N_SAMPLES // 5)
            # Weights for view stream
            if sw_tr is not None:
                sw_diff = np.ones(len(y_tr_aug) - len(tr_idx), dtype=np.float32) * 0.5
                sw_tr_aug = np.concatenate([sw_tr, sw_diff])
            
            # Independent Augmentation for RAW stream (if TabPFN is present)
            # Check if any TabPFN model exists to avoid wasted compute
            has_tabpfn = any('TabPFN' in m[0] for m in names_models)
            if has_tabpfn:
                 print(f"   [DIFFUSION] Augmenting Raw stream (for TabPFN)...")
                 # We must use fresh synthesis on Raw data
                 # Note: y_tr_raw_aug might differ in content/order from y_tr_aug due to random sampling!
                 # This is fine as long as TabPFN uses (X_raw_aug, y_raw_aug) explicitly.
                 X_tr_raw_aug, y_tr_raw_aug = synthesize_data(X_tr_raw_fold, y_tr, n_new_per_class=config.DIFFUSION_N_SAMPLES // 5)
                 if sw_tr is not None:
                    sw_diff = np.ones(len(y_tr_raw_aug) - len(tr_idx), dtype=np.float32) * 0.5
                    sw_tr_raw_aug = np.concatenate([sw_tr, sw_diff])
            
            import gc; gc.collect()
        
        # NOTE: Updates 'X_tr_fold' to be the augmented version for build_streams?
        # NO. build_streams expects Reference info. 
        # Usually we build streams on the NON-augmented data (to learn manifolds correctly from real dist) 
        # or augmented? 
        # Existing code used 'X_tr_fold' (which was overwritten).
        # Let's keep consistency: Feature/Manifold calculation on Real+Augmented?
        # Computing manifold on synthetic data might be noisy.
        # SAFE BET: Build streams on REAL data (X_tr_fold), then apply to Augmented?
        # Complexity: The synthetic data needs features too.
        # If we synthesized Features directly (which we did, X_tr_aug is features), then we don't need build_streams on them?
        # Wait, 'X_tr_fold' is 'X_tr_view' (PCA/Quantile/Raw).
        # synthesizing it gives more PCA/Quantile samples.
        # BUT `build_streams` adds DAE/Manifold features. 
        # If we synthesize View, we can't easily get DAE features for them unless we run DAE on synthetic views.
        # The standard pipeline: View -> Augment -> DAE/Streams.
        # So we should pass X_tr_aug to build_streams if we want features for them.
        
        # 2. BUILD STREAMS (Local)
        # We use X_tr_aug (View Augmented) for standard models.
        # For TabPFN, we don't use streams (it uses Raw), so we ignore streams for it.
        X_tree_tr, X_tree_val, X_neural_tr, X_neural_val, _, _ = build_streams(X_tr_aug, X_val_fold, y_train=y_tr_aug)
        
        # Test streams: using X_tr_aug as reference? Or X_tr_fold (real)?
        # Using Real data as reference for Test manifold is usually safer/better.
        # But let's stick to X_tr_aug to match training distribution representation.
        _, X_tree_te_fold, _, X_neural_te_fold, _, _ = build_streams(X_tr_aug, X_test_view_fold, y_train=y_tr_aug)
        
        # 3. PREPARE PSEUDO
        pX_tree = None
        pX_neural = None
        py = pseudo_y
        pw = pseudo_w
        
        if pseudo_idx is not None and len(pseudo_idx) > 0:
            pX_tree = X_tree_te_fold[pseudo_idx]
            pX_neural = X_neural_te_fold[pseudo_idx]

        # 4. TRAIN BASE MODELS
        for name, base_template in names_models:
            idx_m = model_map[name]
            
            # Clone
            model = copy.deepcopy(base_template)
            
            # Select Feature Stream
            is_tree = ('XGB' in name or 'Cat' in name or 'LGBM' in name)
            is_tabpfn = 'TabPFN' in name
            is_svm = getattr(base_template, '_is_svm', False) or 'SVM' in name
            
            # TabPFN: Use RAW data only? 
            # [CRITICAL FIX] When Diffusion is ON, X_train_raw is NOT augmented, but y_tr IS augmented.
            # We MUST use the augmented stream (X_f_tr) to match dimensions.
            # The "Raw" bypass is only valid if no augmentation happens.
            # if is_tabpfn and X_train_raw is not None:
            #     X_f_tr = X_train_raw[tr_idx]
            #     X_f_val = X_train_raw[val_idx]
            #     X_f_te = X_test_raw
            #     pX_f = X_test_raw[pseudo_idx] if (pseudo_idx is not None and len(pseudo_idx) > 0) else None
            # else:
            
            # SVM: Use view-only features (no manifold streams - SVM struggles with high-dim)
            if is_svm:
                X_f_tr = X_tr_aug  # SVM uses View Aug
                y_tr_eff_model = y_tr_aug
                sw_tr_eff_model = sw_tr_aug
                
                X_f_val = X_val_fold # Val is always Real
                X_f_te = X_test_view_fold
                pX_f = X_test_view_fold[pseudo_idx] if (pseudo_idx is not None and len(pseudo_idx) > 0) else None
            
            # TabPFN: Use RAW AUG logic
            elif is_tabpfn:
                # TabPFN uses Independent Raw Augmentation stream
                X_f_tr = X_tr_raw_aug
                y_tr_eff_model = y_tr_raw_aug
                sw_tr_eff_model = sw_tr_raw_aug
                
                # Val/Test must be RAW Real
                X_f_val = X_val_raw_fold
                X_f_te = X_test_raw # Full Test Raw
                pX_f = X_test_raw[pseudo_idx] if (pseudo_idx is not None and len(pseudo_idx) > 0) else None
                
                # [OMEGA] TabPFN Geometry + Anomaly Injection
                # Explicitly compute Centroid Distances + Anomalies on RAW Data
                geo = GeometricFeatureGenerator().fit(X_tr_raw_aug, y_tr_raw_aug)
                anom = AnomalyFeatureGenerator().fit(X_tr_raw_aug)
                
                g_tr = np.hstack([geo.transform(X_tr_raw_aug), anom.transform(X_tr_raw_aug)])
                g_val = np.hstack([geo.transform(X_val_raw_fold), anom.transform(X_val_raw_fold)])
                g_te = np.hstack([geo.transform(X_test_raw), anom.transform(X_test_raw)])
                if pX_f is not None: 
                    gp = np.hstack([geo.transform(pX_f), anom.transform(pX_f)])
                
                pass # Just ensuring indentation
                
                X_f_tr = np.hstack([X_f_tr, g_tr])
                X_f_val = np.hstack([X_f_val, g_val])
                X_f_te = np.hstack([X_f_te, g_te])
                if pX_f is not None: pX_f = np.hstack([pX_f, gp])
                
            else:
                # Standard Tree/Neural Streams (Built from X_tr_aug)
                X_f_tr = X_tree_tr if is_tree else X_neural_tr
                y_tr_eff_model = y_tr_aug
                sw_tr_eff_model = sw_tr_aug
                
                X_f_val = X_tree_val if is_tree else X_neural_val
                X_f_te = X_tree_te_fold if is_tree else X_neural_te_fold
                pX_f = pX_tree if is_tree else pX_neural
            
            # Concatenate Pseudo if active
            X_train_final = X_f_tr
            y_train_final = y_tr_eff_model
            w_train_final = sw_tr_eff_model
            
            if pX_f is not None:
                # Handle Concatenation & Label Types
                is_pseudo_soft = (py.ndim > 1) or np.issubdtype(py.dtype, np.floating)
                is_tabpfn = 'TabPFN' in name  # TabPFN requires HARD labels only
                is_torch = hasattr(model, 'finetune_on_pseudo') and not is_tabpfn
                
                y_tr_eff = y_train_final
                py_eff = py
                
                # TabPFN & Trees: Hard Labels (TabPFN doesn't support soft labels)
                if is_pseudo_soft and (not is_torch or is_tabpfn):
                    if py.ndim > 1: py_eff = np.argmax(py, axis=1).astype(np.int64)
                    else: py_eff = py.astype(np.int64)
                elif is_pseudo_soft and is_torch:
                    # Torch models (TrueTabR): Soft Labels
                    if y_tr_eff.ndim == 1:
                        y_tr_eff = np.eye(num_classes, dtype=np.float32)[y_tr_eff]
                
                X_train_final = np.vstack([X_f_tr, pX_f])
                
                # Shape matching for Y
                if y_train_final.ndim == 1 and py_eff.ndim == 1:
                    y_train_final = np.concatenate([y_tr_eff, py_eff])
                else:
                    if y_tr_eff.ndim == 1: y_tr_eff = y_tr_eff[:, None]
                    if py_eff.ndim == 1: py_eff = py_eff[:, None]
                    y_train_final = np.vstack([y_tr_eff, py_eff])
                    if y_train_final.shape[1] == 1: y_train_final = y_train_final.ravel()
                
                # Weights
                w_tr_base = w_train_final if w_train_final is not None else np.ones(len(y_tr), dtype=np.float32)
                
                # [OMEGA] Confusion Weighting (Focus on 2 vs 5)
                # Boost weights for classes that Scout identified as hard
                if config.CONFUSION_WEIGHT_MULTIPLIER > 1.0:
                    # We can use the 'meta_pairs' we detected earlier or re-detect
                    # Re-detecting on current fold is safer
                    fold_pairs = find_confusion_pairs(X_tr_raw, y_tr_raw, top_k=1, seed=seed)
                    for (c_a, c_b) in fold_pairs:
                        mask_conf = (y_tr_eff == c_a) | (y_tr_eff == c_b)
                        if mask_conf.shape[0] == w_tr_base.shape[0]:
                            w_tr_base[mask_conf] *= config.CONFUSION_WEIGHT_MULTIPLIER
                            print(f"    [Weighting] Boosted weights for Class {c_a} & {c_b} (x{config.CONFUSION_WEIGHT_MULTIPLIER})")

                w_p_base = pw if pw is not None else np.ones(len(py), dtype=np.float32)
                w_train_final = np.concatenate([w_tr_base, w_p_base]) 

            # FIT
            try:
                model.fit(X_train_final, y_train_final, sample_weight=w_train_final)
            except TypeError:
                model.fit(X_train_final, y_train_final)
            
            # OOF & TEST PREDICT
            p_oof = model.predict_proba(X_f_val).astype(np.float32)
            oof_preds[idx_m][val_idx] = p_oof
            
            p_test = model.predict_proba(X_f_te).astype(np.float32)
            test_preds_running[idx_m] += p_test
            
            # [MEMORY FIX] Aggressively free GPU memory after each model
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- MEMORY CLEANUP PER FOLD ---
        # Crucial for preventing OOM when using Diffusion + Manifold features
        try:
            del X_train_aug
        except NameError:
            pass
        
        # Explicitly delete fold tensors/arrays
        del X_train_final, y_train_final, w_train_final
        if 'X_tr_fold' in locals(): del X_tr_fold
        if 'X_val_fold' in locals(): del X_val_fold
        
        # Stream deletion
        try:
             del X_tree_tr, X_tree_val, X_neural_tr, X_neural_val
             del X_f_tr, X_f_val, X_f_te
        except:
             pass
             
        # Clear model specific vars
        try:
            del model, p_oof, p_test
        except:
            pass

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average Test Predictions
    for i in range(n_models):
        test_preds_running[i] /= cv_splits

    # >>> OVERFITTING DIAGNOSTIC REPORT <<<
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              OVERFITTING DIAGNOSTIC REPORT                   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    
    model_names = [name for name, _ in names_models]
    oof_accuracies = []
    
    for idx_m, name in enumerate(model_names):
        oof_pred_labels = np.argmax(oof_preds[idx_m], axis=1)
        oof_acc = (oof_pred_labels == y).mean() * 100
        oof_accuracies.append(oof_acc)
        
        # Overfitting indicator: if OOF acc is suspiciously high (>98%) or has high variance
        flag = ""
        if oof_acc > 98:
            flag = " ⚠️ SUSPICIOUS (too high)"
        elif oof_acc < 70:
            flag = " ⚠️ UNDERFITTING"
        
        print(f"  ║  {name:<20} OOF Accuracy: {oof_acc:6.2f}%{flag:<20}║")
    
    # Ensemble OOF (average of all models)
    ensemble_oof = np.mean([oof_preds[i] for i in range(len(oof_preds))], axis=0)
    ensemble_oof_labels = np.argmax(ensemble_oof, axis=1)
    ensemble_oof_acc = (ensemble_oof_labels == y).mean() * 100
    
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'ENSEMBLE OOF':<20} Accuracy: {ensemble_oof_acc:6.2f}%                   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    
    # Compare best single model vs ensemble
    best_single = max(oof_accuracies)
    ensemble_gain = ensemble_oof_acc - best_single
    if ensemble_gain > 0:
        print(f"  ║  Ensemble Gain: +{ensemble_gain:.2f}% over best single model          ║")
    else:
        print(f"  ║  ⚠️ Ensemble WORSE than best single by {-ensemble_gain:.2f}%            ║")
    
    # Overfitting warning
    if ensemble_oof_acc > 95:
        print("  ║  ⚠️ WARNING: OOF acc >95% may indicate data leakage!         ║")
    
    print("  ╚══════════════════════════════════════════════════════════════╝\n")

    # --- META LEARNER ---
    meta_X = np.hstack(oof_preds)
    meta_te = np.hstack(test_preds_running)
    
    # Meta Features (Entropy, Max Prob, etc.)
    meta_feat_oof = [prob_meta_features(oof) for oof in oof_preds]
    meta_feat_te = [prob_meta_features(p) for p in test_preds_running]
    
    # [OMEGA] SCOUT INJECTION (Direct 2-vs-5 Expert Knowledge)
    # The Meta-Learner needs to know: "Is this a 2/5 confusion case?"
    # We generate Scout probabilities on the BASE features.
    print("  [Meta] Injecting Confusion Scout features...")
    
    # 1. Detect Pairs (on raw base data)
    # Use X_train_base (could be raw/pca/etc depending on view, but raw is best for scout)
    # Scout logic handles view data fine usually.
    meta_pairs = find_confusion_pairs(X_train_base, y, top_k=3, seed=seed)
    
    # 2. Generate Scout Features (OOF for Train, Full for Test)
    # We use CrossFoldFeatureGenerator to prevent leakage in Meta-Training
    scout_gen = CrossFoldFeatureGenerator(ConfusionScout, pairs=meta_pairs, n_folds=5).fit(X_train_base, y)
    
    scout_oof = scout_gen.transform(X_train_base, y, mode='train')
    scout_te = scout_gen.transform(X_test_base, mode='test')
    
    # Concatenate: [Ensemble Probs] + [Meta Stats] + [Scout Expert Opinion]
    meta_X = np.hstack([meta_X] + meta_feat_oof + [scout_oof])
    meta_te = np.hstack([meta_te] + meta_feat_te + [scout_te])
    
    # Store OOF Aggregation for Threshold Tuning
    # Simple average for base threshold tuning? Or just use meta output?
    # If meta-learner is used, its output (on train set via cross-val?) is complex.
    # Stacking usually doesn't output 'train' preds.
    # We can use the weighted average of OOF predictions as a proxy for "Ensemble OOF".
    if mode == 'hill_climb':
         # weights already computed.
         oof_meta = np.zeros_like(oof_preds[0])
         for m_i in range(len(oof_preds)):
            oof_meta += oof_preds[m_i] * weights[m_i]
    elif mode in ['rank', 'geo']:
        # recalculate based on mode
        if mode == 'rank': oof_meta = rank_average(oof_preds)
        else: oof_meta = geometric_mean(oof_preds)
    else:
        # For LR/LGBM, we need to cross-val PREDICT on meta_X to get "OOF of Meta".
        # This is nested stacking. Too complex/expensive.
        # Fallback: Just use Hill Climb weights or Simple Average for threshold tuning base.
        # OR: Just use simple average of OOFs.
        print("  [Thresholds] Using Simple Average OOF for threshold learning (Meta is complex).")
        oof_meta = np.mean(oof_preds, axis=0)

    # Return OOF for Tuning
    if return_oof:
        return oof_preds, test_preds_running, meta_X, meta_te, y

    # Meta Optimization
    mode = config.META_LEARNER
    print(f"  > Fitting Meta-Learner ({mode})...")
    
    if mode == 'lr':
        # Optimal C=0.55 from Optuna tuning (87.20% CV accuracy, 3030 trials)
        meta = LogisticRegression(C=0.55, random_state=seed, solver='lbfgs', max_iter=1000)
        meta.fit(meta_X, y)
        final_probs = meta.predict_proba(meta_te)
        
    elif mode == 'ridge':
        meta = RidgeClassifier(alpha=1.0, random_state=seed)
        meta.fit(meta_X, y)
        d = meta.decision_function(meta_te)
        e_d = np.exp(d - np.max(d, axis=1, keepdims=True))
        final_probs = e_d / e_d.sum(axis=1, keepdims=True)

    elif mode == 'lgbm':
        from lightgbm import LGBMClassifier
        meta = LGBMClassifier(n_estimators=100, num_leaves=15, max_depth=3, random_state=seed, verbose=-1)
        meta.fit(meta_X, y)
        final_probs = meta.predict_proba(meta_te)
        
    elif mode == 'hill_climb':
        print("  [Meta] Using Hill Climbing Optimization (Accuracy)...")
        weights = hill_climbing_optimization(oof_preds, y, iterations=100)
        print(f"   [HillClimb Weights] {weights}")
        
        final_probs = np.zeros_like(test_preds_running[0])
        for m_i in range(len(oof_preds)):
            final_probs += test_preds_running[m_i] * weights[m_i]
    
    elif mode == 'rank':
        print("  [Meta] Using Rank Averaging...")
        final_probs = rank_average(test_preds_running)
        
    elif mode == 'geo':
        print("  [Meta] Using Geometric Mean (Power Average)...")
        final_probs = geometric_mean(test_preds_running)
        
    else: # NNLS
        print(f"  [Meta] Using NNLS...")
        n_base_models = len(oof_preds)
        Z = np.zeros((len(y), n_base_models))
        for m_i in range(n_base_models):
            Z[:, m_i] = oof_preds[m_i][np.arange(len(y)), y]
        target = np.ones(len(y))
        
        weights, _ = nnls(Z, target)
        weights /= (weights.sum() + 1e-9)
        print(f"   [NNLS Weights] {weights}")
        
        final_probs = np.zeros_like(test_preds_running[0])
        for m_i in range(n_base_models):
            final_probs += test_preds_running[m_i] * weights[m_i]
    
    # [OMEGA] Threshold Optimization (Pre-Post-Processing)
    # Learn on OOF, apply to Test
    thresh_w = optimize_class_thresholds(oof_meta, y)
    final_probs = final_probs * thresh_w
    final_probs /= final_probs.sum(axis=1, keepdims=True)

    # [OMEGA] Post-Processing (The Silencer & The Equalizer)
    if config.ENABLE_POSTPROCESSING:
        # 1. Label Distribution Alignment (LDA)
        final_probs = align_probabilities(final_probs, y, method=config.LDA_METHOD)
        
        # 2. Feature Neutralization (The Silencer)
        # Neutralize against the Base View features (Test Set)
        final_probs = neutralize_predictions(final_probs, X_test_base, proportion=config.NEUTRALIZE_STRENGTH)

    # [OMEGA] Failure Analysis
    ensemble_failure_analysis(oof_meta, y)

    return final_probs
