import copy
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import config
from .losses import prob_meta_features
from .features import apply_feature_view, build_streams  # Imported for Cross-Fit

def tabpfn_predict_proba(X_train, y_train, X_eval, n_ensembles=32, seed=42):
    """Single fold prediction for TabPFN."""
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        raise RuntimeError("USE_TABPFN=1 but `tabpfn` is not installed/importable.") from e
        
    model = TabPFNClassifier(device=str(config.DEVICE), N_ensemble_configurations=int(n_ensembles), seed=int(seed))
    # TabPFN can handle small data well.
    # Note: X_train should be standardized or similar? TabPFN handles it internally usually.
    model.fit(X_train, y_train)
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
):
    """
    Cross-Fit Stacking:
    Performs Feature Engineering (View + Streams) *inside* the Cross-Validation Loop
    to prevent data leakage.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    # We will accumulate OOF predictions and Test predictions
    # Dimensions:
    # OOF: (N_train, num_classes) per model
    # Test: (N_test, num_classes) per model (averaged)
    
    n_models = len(names_models) + (1 if config.USE_TABPFN else 0)
    
    oof_preds = [np.zeros((len(X_train_base), num_classes), dtype=np.float32) for _ in range(n_models)]
    test_preds_running = [np.zeros((len(X_test_base), num_classes), dtype=np.float32) for _ in range(n_models)]
    
    # Store meta-features (lid, entropy?) if needed.
    # The original implementation used prob_meta_features on OOF.
    
    # Map model names to index
    model_map = {name: i for i, (name, _) in enumerate(names_models)}
    if config.USE_TABPFN:
        model_map['TabPFN'] = len(names_models)

    print(f"  [STACKING] Starting Cross-Fit Stacking (K={cv_splits}) on View '{view_name}'...")

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train_base, y)):
        # 1. SPLIT RAW DATA
        X_tr_raw, X_val_raw = X_train_base[tr_idx], X_train_base[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
        
        # 2. APPLY VIEW TRANSFORM (Local fit)
        # Transform X_val_raw and X_test_base using X_tr_raw statistics
        X_tr_view, X_val_view = apply_feature_view(
            X_tr_raw, X_val_raw, 
            view=view_name, seed=seed, allow_transductive=config.ALLOW_TRANSDUCTIVE
        )
        # We also need to transform GLOBAL TEST set using CURRENT FOLD statistics for test accumulation
        # Note: redundant to fit transform K times on test, but required for strict correctness if transform is data-dependent.
        # Use a separate call or hacked usage of apply_feature_view.
        # apply_feature_view returns (tr, te). We can pass (X_tr_raw, X_test_base).
        _, X_test_view_fold = apply_feature_view(
             X_tr_raw, X_test_base,
             view=view_name, seed=seed, allow_transductive=config.ALLOW_TRANSDUCTIVE
        )

        # 3. BUILD STREAMS (Local fit)
        # We need X_tree and X_neural for Tr, Val, and Test
        # build_streams returns tr, te.
        # Val:
        X_tree_tr, X_tree_val, X_neural_tr, X_neural_val, _, _ = build_streams(X_tr_view, X_val_view)
        # Test:
        _, X_tree_te_fold, _, X_neural_te_fold, _, _ = build_streams(X_tr_view, X_test_view_fold)
        
        # 4. PREPARE PSEUDO DATA (If active)
        # Pseudo samples are from X_test_base.
        # We assume pseudo_idx refers to indices in X_test_base.
        pX_tree = None
        pX_neural = None
        py = pseudo_y
        pw = pseudo_w
        
        if pseudo_idx is not None and len(pseudo_idx) > 0:
            pX_tree = X_tree_te_fold[pseudo_idx]
            pX_neural = X_neural_te_fold[pseudo_idx]
            
            # Label compatibility check (Soft vs Hard)
            # We do this per model below or setup efficiently here.
            pass

        # 5. TRAIN & PREDICT BASE MODELS
        for name, base_template in names_models:
            idx_m = model_map[name]
            
            # Clone model
            model = copy.deepcopy(base_template)
            
            # Select features
            is_tree = ('XGB' in name or 'Cat' in name)
            X_f_tr = X_tree_tr if is_tree else X_neural_tr
            X_f_val = X_tree_val if is_tree else X_neural_val
            X_f_te = X_tree_te_fold if is_tree else X_neural_te_fold
            pX_f = pX_tree if is_tree else pX_neural
            
            # Prepare Training Data (Fold + Pseudo)
            X_train_final = X_f_tr
            y_train_final = y_tr
            w_train_final = sw_tr
            
            if pX_f is not None:
                # Concatenate Pseudo
                # Check soft/hard label compatibility
                is_pseudo_soft = (py.ndim > 1) or np.issubdtype(py.dtype, np.floating)
                is_torch = hasattr(model, 'finetune_on_pseudo') # Heuristic for Torch
                
                y_tr_eff = y_train_final
                py_eff = py
                
                if is_pseudo_soft and not is_torch:
                    # Tree: Soft -> Hard
                    if py.ndim > 1:
                        py_eff = np.argmax(py, axis=1).astype(np.int64)
                    else:
                        py_eff = py.astype(np.int64)
                elif is_pseudo_soft and is_torch:
                    # Torch: Hard -> Soft
                    if y_tr_eff.ndim == 1:
                         y_tr_eff = np.eye(num_classes, dtype=np.float32)[y_tr_eff]
                
                # Check dims again
                if y_tr_eff.ndim == 1 and py_eff.ndim == 2:
                     y_tr_eff = np.eye(num_classes, dtype=np.float32)[y_tr_eff]
                elif y_tr_eff.ndim == 2 and py_eff.ndim == 1:
                     # e.g. Tree case, both 1D hard
                     pass 
                     
                X_train_final = np.vstack([X_f_tr, pX_f])
                if y_train_final.ndim == 2 or py_eff.ndim == 2:
                    y_train_final = np.vstack([y_tr_eff, py_eff])
                else:
                    y_train_final = np.concatenate([y_tr_eff, py_eff])
                
                # Weights
                w_tr_base = w_train_final if w_train_final is not None else np.ones(len(y_tr), dtype=np.float32)
                w_p_base = pw if pw is not None else np.ones(len(py), dtype=np.float32)
                w_train_final = np.concatenate([w_tr_base, w_p_base])

            # FIT
            try:
                model.fit(X_train_final, y_train_final, sample_weight=w_train_final)
            except TypeError:
                model.fit(X_train_final, y_train_final)
            
            # PREDICT OOF
            p_oof = model.predict_proba(X_f_val).astype(np.float32)
            oof_preds[idx_m][val_idx] = p_oof
            
            # PREDICT TEST
            p_test = model.predict_proba(X_f_te).astype(np.float32)
            test_preds_running[idx_m] += p_test

        # 6. TABPFN (If enabled)
        if config.USE_TABPFN:
            idx_m = model_map['TabPFN']
            # TabPFN uses View features (usually small dim)
            # Use X_tr_view, X_val_view
            # TabPFN usually doesn't support Soft Labels easily (it's a transformer on (x,y) pairs).
            # So if Pseudo is soft, convert to hard.
            
            X_tab_tr = X_tr_view
            y_tab_tr = y_tr
            
            # Mix pseudo? TabPFN context size is limited (e.g. 1024).
            # If we add pseudo, we might exceed context.
            # Strategy: Skip pseudo for TabPFN or sample. 
            # Given TabPFN robustness, maybe train on clean data only is safer/better diversifier.
            # Let's stick to Clean Fold Train for TabPFN for now to avoid context limits.
            
            p_oof_tab = tabpfn_predict_proba(X_tab_tr, y_tab_tr, X_val_view, n_ensembles=config.TABPFN_N_ENSEMBLES)
            oof_preds[idx_m][val_idx] = p_oof_tab
            
            p_te_tab = tabpfn_predict_proba(X_tab_tr, y_tab_tr, X_test_view_fold, n_ensembles=config.TABPFN_N_ENSEMBLES)
            test_preds_running[idx_m] += p_te_tab

    # Average Test Predictions
    for i in range(n_models):
        test_preds_running[i] /= cv_splits

    # META LEARNER
    # We now have OOF matrix [Model 1, Model 2, ...]
    # Stack them
    meta_X = np.hstack(oof_preds)
    meta_te = np.hstack(test_preds_running)
    
    # Meta Features (Entropy etc)
    # Computed on OOF and Test
    meta_feat_oof = [prob_meta_features(oof) for oof in oof_preds]
    meta_feat_te = [prob_meta_features(p) for p in test_preds_running]
    
    meta_X = np.hstack([meta_X] + meta_feat_oof)
    meta_te = np.hstack([meta_te] + meta_feat_te)

    # Note: We lost 'lid_tr' / 'lid_te' global features for Meta Learner 
    # because they are now computed locally.
    # We could compute LID globally just for Meta learner? 
    # Or average local LID?
    # For now, omit LID in Meta (it's minor).
    
    print("  > Fitting Meta-Learner...")
    if config.META_LEARNER == 'lgbm':
        from lightgbm import LGBMClassifier
        meta = LGBMClassifier(
            objective='multiclass',
            num_class=int(num_classes),
            n_estimators=100,
            verbosity=-1
        )
    else:
        meta = LogisticRegression(max_iter=2000, multi_class='multinomial')
        
    meta.fit(meta_X, y, sample_weight=sample_weight) # Meta trains on OOF vs Label
    
    final_probs = meta.predict_proba(meta_te)
    return final_probs
