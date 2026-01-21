import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import nnls

from . import config
from .losses import prob_meta_features
from .features import ConfusionScout, find_confusion_pairs, CrossFoldFeatureGenerator
from .cv_engine import UnifiedCVEngine
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


def optimize_class_thresholds_bounded(oof_probs, y_true, bounds=(0.5, 2.0), init_boost=None):
    """
    Bounded per-class scaling to maximize accuracy.
    Uses honest OOF predictions to avoid leakage.
    """
    from scipy.optimize import minimize

    n_classes = oof_probs.shape[1]
    low, high = float(bounds[0]), float(bounds[1])

    x0 = np.ones(n_classes, dtype=np.float64)
    if init_boost is not None:
        for cls_idx, boost in init_boost.items():
            if 0 <= int(cls_idx) < n_classes:
                x0[int(cls_idx)] = float(boost)

    def loss_func(w):
        w = np.clip(w, low, high)
        p_w = oof_probs * w
        y_pred = np.argmax(p_w, axis=1)
        return -np.mean(y_pred == y_true)

    bounds_vec = [(low, high) for _ in range(n_classes)]
    res = minimize(loss_func, x0=x0, method='L-BFGS-B', bounds=bounds_vec)
    best_w = np.clip(res.x, low, high)
    base_acc = np.mean(np.argmax(oof_probs, axis=1) == y_true)
    best_acc = -res.fun
    print(f"   [Thresholds] Bounded Acc {best_acc:.4f} (Base: {base_acc:.4f})")
    return best_w


def optimize_pairwise_correction(oof_probs, y_true, pair=(2, 5), bounds_diag=(0.5, 2.0), bounds_off=(0.0, 2.0)):
    """Optimize a 2x2 correction matrix for a class pair to maximize accuracy."""
    from scipy.optimize import minimize

    n_classes = oof_probs.shape[1]
    c1, c2 = int(pair[0]), int(pair[1])
    if c1 < 0 or c2 < 0 or c1 >= n_classes or c2 >= n_classes or c1 == c2:
        return np.eye(2, dtype=np.float64)

    def apply_corr(p, m):
        p_adj = p.copy()
        v = p_adj[:, [c1, c2]]
        q = v @ m.T
        p_adj[:, c1] = q[:, 0]
        p_adj[:, c2] = q[:, 1]
        p_adj /= (p_adj.sum(axis=1, keepdims=True) + 1e-12)
        return p_adj

    def loss_func(x):
        m = np.array([[x[0], x[1]], [x[2], x[3]]], dtype=np.float64)
        p_adj = apply_corr(oof_probs, m)
        y_pred = np.argmax(p_adj, axis=1)
        return -np.mean(y_pred == y_true)

    x0 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    bounds = [bounds_diag, bounds_off, bounds_off, bounds_diag]
    res = minimize(loss_func, x0=x0, method='L-BFGS-B', bounds=bounds)
    best = np.array([[res.x[0], res.x[1]], [res.x[2], res.x[3]]], dtype=np.float64)
    return best


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
    
    engine = UnifiedCVEngine(
        names_models=names_models,
        view_name=view_name,
        X_train_base=X_train_base,
        X_test_base=X_test_base,
        y=y,
        num_classes=num_classes,
        cv_splits=cv_splits,
        seed=seed,
        sample_weight=sample_weight,
        pseudo_idx=pseudo_idx,
        pseudo_y=pseudo_y,
        pseudo_w=pseudo_w,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
    )
    oof_preds, test_preds_running = engine.run()

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

    mode = config.META_LEARNER

    def _meta_predict_proba(model, X_eval):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_eval)
        d = model.decision_function(X_eval)
        e_d = np.exp(d - np.max(d, axis=1, keepdims=True))
        return e_d / e_d.sum(axis=1, keepdims=True)

    def _build_meta_model(mode_name, seed_val):
        if mode_name == 'lr':
            return LogisticRegression(C=0.55, random_state=seed_val, solver='lbfgs', max_iter=1000)
        if mode_name == 'ridge':
            return RidgeClassifier(alpha=1.0, random_state=seed_val)
        if mode_name == 'lgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=config.LGBM_N_ESTIMATORS,
                num_leaves=config.LGBM_NUM_LEAVES,
                max_depth=config.LGBM_MAX_DEPTH,
                min_child_samples=20,
                class_weight='balanced',
                random_state=seed_val,
                verbose=-1,
            )
        raise ValueError(f"Unsupported meta mode: {mode_name}")

    nested_meta_oof = None
    if mode in ['lr', 'ridge', 'lgbm']:
        print("  [Meta] Nested CV for threshold tuning...")
        nested_meta_oof = np.zeros((len(y), num_classes), dtype=np.float32)
        meta_skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        for tr_idx, val_idx in meta_skf.split(meta_X, y):
            meta_cv = _build_meta_model(mode, seed)
            meta_cv.fit(meta_X[tr_idx], y[tr_idx])
            nested_meta_oof[val_idx] = _meta_predict_proba(meta_cv, meta_X[val_idx])

    # Store OOF Aggregation for Threshold Tuning
    if nested_meta_oof is not None:
        oof_meta = nested_meta_oof
    elif mode == 'hill_climb':
        oof_meta = np.zeros_like(oof_preds[0])
        for m_i in range(len(oof_preds)):
            oof_meta += oof_preds[m_i] * weights[m_i]
    elif mode in ['rank', 'geo']:
        oof_meta = rank_average(oof_preds) if mode == 'rank' else geometric_mean(oof_preds)
    else:
        print("  [Thresholds] Using Simple Average OOF for threshold learning.")
        oof_meta = np.mean(oof_preds, axis=0)

    # Return OOF for Tuning
    if return_oof:
        return oof_preds, test_preds_running, meta_X, meta_te, y

    # Meta Optimization
    print(f"  > Fitting Meta-Learner ({mode})...")
    
    if mode == 'lr':
        meta = _build_meta_model(mode, seed)
        meta.fit(meta_X, y)
        final_probs = _meta_predict_proba(meta, meta_te)
        
    elif mode == 'ridge':
        meta = _build_meta_model(mode, seed)
        meta.fit(meta_X, y)
        final_probs = _meta_predict_proba(meta, meta_te)

    elif mode == 'lgbm':
        meta = _build_meta_model(mode, seed)
        meta.fit(meta_X, y)
        final_probs = _meta_predict_proba(meta, meta_te)
        
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
    
    # [OMEGA] Honest Threshold Optimization (Pre-Post-Processing)
    # Learn on honest OOF, apply to Test with bounded scaling
    init_boost = {2: 1.1, 5: 1.1}  # Known confusion pair emphasis
    thresh_w = optimize_class_thresholds_bounded(
        oof_meta,
        y,
        bounds=(0.5, 2.0),
        init_boost=init_boost,
    )
    final_probs = final_probs * thresh_w
    final_probs /= final_probs.sum(axis=1, keepdims=True)

    oof_scaled = oof_meta * thresh_w
    oof_scaled /= (oof_scaled.sum(axis=1, keepdims=True) + 1e-12)
    pair_matrix = optimize_pairwise_correction(oof_scaled, y, pair=(2, 5))
    pair_c1, pair_c2 = 2, 5
    pair_block = final_probs[:, [pair_c1, pair_c2]]
    pair_block = pair_block @ pair_matrix.T
    final_probs[:, pair_c1] = pair_block[:, 0]
    final_probs[:, pair_c2] = pair_block[:, 1]
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
