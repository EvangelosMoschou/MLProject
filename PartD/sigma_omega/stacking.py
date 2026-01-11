import copy
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import nnls

from . import config
from .losses import prob_meta_features
from .features import apply_feature_view, build_streams
from .generative import synthesize_data

def tabpfn_predict_proba(X_train, y_train, X_eval, n_ensembles=32, seed=42):
    """Prediction helper for TabPFN (Legacy/Fallback use)."""
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        raise RuntimeError("USE_TABPFN=1 but `tabpfn` is not installed/importable.") from e
        
    model = TabPFNClassifier(device=str(config.DEVICE), n_estimators=int(n_ensembles), random_state=int(seed))
    model.fit(X_train, y_train) # Fit just stores data usually
    return model.predict_proba(X_eval).astype(np.float32)


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
    
    # Transform View (Split handled inside loop usually for strict correctness, but optimization allows global transform if transductive)
    # apply_feature_view handles transductive check
    X_tr_view, X_test_view_fold = apply_feature_view(
        X_train_base, X_test_base, view=view_name, seed=seed, allow_transductive=config.ALLOW_TRANSDUCTIVE
    )
    
    for tr_idx, val_idx in skf.split(X_tr_view, y):
        # 1. SPLIT DATA
        X_tr_fold = X_tr_view[tr_idx]
        X_val_fold = X_tr_view[val_idx]
        y_tr = y[tr_idx]
        
        sw_tr = None
        if sample_weight is not None:
            sw_tr = sample_weight[tr_idx]

        # [OMEGA] Diffusion Augmentation
        # Synthesize additional training samples per class to improve generalization
        if config.ENABLE_DIFFUSION and len(X_tr_fold) > 100:
            print(f"   [DIFFUSION] Augmenting fold training data...")
            X_tr_fold, y_tr = synthesize_data(X_tr_fold, y_tr, n_new_per_class=config.DIFFUSION_N_SAMPLES // 5)
            # Update sample weights for augmented data
            if sw_tr is not None:
                sw_aug = np.ones(len(y_tr) - len(tr_idx), dtype=np.float32) * 0.5  # Lower weight for synthetic
                sw_tr = np.concatenate([sw_tr, sw_aug])
        
        # 2. BUILD STREAMS (Local)
        # We need streams for Train and Val
        # X_tr_fold -> build_streams -> X_tree_tr, X_neural_tr
        X_tree_tr, X_tree_val, X_neural_tr, X_neural_val, _, _ = build_streams(X_tr_fold, X_val_fold)
        
        # Test streams: using X_tr_fold as reference
        _, X_tree_te_fold, _, X_neural_te_fold, _, _ = build_streams(X_tr_fold, X_test_view_fold)
        
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
            X_f_tr = X_tree_tr if is_tree else X_neural_tr
            X_f_val = X_tree_val if is_tree else X_neural_val
            X_f_te = X_tree_te_fold if is_tree else X_neural_te_fold
            pX_f = pX_tree if is_tree else pX_neural
            
            # Concatenate Pseudo if active
            X_train_final = X_f_tr
            y_train_final = y_tr
            w_train_final = sw_tr
            
            if pX_f is not None:
                # Handle Concatenation & Label Types
                is_pseudo_soft = (py.ndim > 1) or np.issubdtype(py.dtype, np.floating)
                is_torch = hasattr(model, 'finetune_on_pseudo') or 'TabPFN' in name # TabPFN Wrapper handles hard internally
                
                y_tr_eff = y_train_final
                py_eff = py
                
                # Trees: Hard Labels
                if is_pseudo_soft and not is_torch:
                    if py.ndim > 1: py_eff = np.argmax(py, axis=1).astype(np.int64)
                    else: py_eff = py.astype(np.int64)
                elif is_pseudo_soft and is_torch:
                    # Torch: Soft Labels (if supported)
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

        # --- MEMORY CLEANUP PER FOLD ---
        # Crucial for preventing OOM when using Diffusion + Manifold features
        del X_train_aug, X_val_fold, X_train_final, y_train_final
        try:
             del X_tree_tr, X_tree_val, X_neural_tr, X_neural_val
             del X_f_tr, X_f_val, X_f_te
        except:
             pass
        import gc
        gc.collect()

    # Average Test Predictions
    for i in range(n_models):
        test_preds_running[i] /= cv_splits

    # --- META LEARNER ---
    meta_X = np.hstack(oof_preds)
    meta_te = np.hstack(test_preds_running)
    
    # Meta Features
    meta_feat_oof = [prob_meta_features(oof) for oof in oof_preds]
    meta_feat_te = [prob_meta_features(p) for p in test_preds_running]
    meta_X = np.hstack([meta_X] + meta_feat_oof)
    meta_te = np.hstack([meta_te] + meta_feat_te)

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
    
    return final_probs
