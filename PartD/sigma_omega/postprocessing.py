import numpy as np
from sklearn.preprocessing import QuantileTransformer

def neutralize_predictions(preds, X, proportion=0.5):
    """
    The Silencer: Feature Neutralization.
    Removes the linear component of the predictions that is correlated with the features `X`.
    
    Args:
        preds: (N_samples, N_classes) or (N_samples, )
        X: Features (N_samples, N_features)
        proportion: Strength of neutralization (0.0 to 1.0)
    
    Returns:
        Neutralized predictions.
    """
    if proportion <= 0.0:
        return preds
        
    print(f"   [POST] Neutralizing features (strength={proportion})...")
    
    # Work on ranks to be robust
    preds_rank = preds.copy()
    if preds.ndim == 1:
        preds_rank = _gauss_rank(preds)
    else:
        for c in range(preds.shape[1]):
            preds_rank[:, c] = _gauss_rank(preds[:, c])
    
    # Orthogonalize
    # Pred_neut = Pred - X @ (X.T X)^-1 X.T Pred
    # Using least squares: Pred = X @ w + resid. resid is the neutralized part.
    
    # Center X
    X_cent = X - X.mean(axis=0)
    
    # We do linear regression of Pred_rank on X for each class
    # residuals = y - y_hat
    # Since X is potentially high dim, use lstsq
    
    # Optimization: Only neutralize against top PCA components if X is too large?
    # Or just top correlated features?
    # For full "Silencer", we use all X.
    
    preds_neut = np.zeros_like(preds)
    
    if preds.ndim == 1:
        w, _, _, _ = np.linalg.lstsq(X_cent, preds_rank, rcond=None)
        pred_proj = X_cent @ w
        preds_neut = preds_rank - proportion * pred_proj
        # Rank transform back to uniform/prob space? 
        # Usually just rescale.
        preds_neut = (preds_neut - preds_neut.min()) / (preds_neut.max() - preds_neut.min())
    else:
        # Multiclass: Neutralize each class column independetly
        w, _, _, _ = np.linalg.lstsq(X_cent, preds_rank, rcond=None)
        pred_proj = X_cent @ w
        preds_neut_rank = preds_rank - proportion * pred_proj
        
        # Softmax back?
        # Converting ranks back to probabilities is tricky.
        # Simple approach: Softmax the neut ranks (scaled)
        preds_neut_rank = (preds_neut_rank - preds_neut_rank.mean(axis=0)) / preds_neut_rank.std(axis=0)
        preds_neut = _softmax(preds_neut_rank)

    return preds_neut

def align_probabilities(preds, y_train, method='prior_shift'):
    """
    The Equalizer: Label Distribution Alignment (LDA).
    Corrects predictions based on Train vs Test class priors.
    
    Args:
        preds: (N_test, N_classes)
        y_train: Training labels (to compute train prior)
        method: 'prior_shift' or 'user_sum'
    """
    print(f"   [POST] Running Label Distribution Alignment ({method})...")
    
    # 1. Compute Train Prior
    classes, counts = np.unique(y_train, return_counts=True)
    n_train = len(y_train)
    p_train = np.zeros(preds.shape[1])
    p_train[classes] = counts / n_train
    
    # 2. Estimate Test Prior (using soft preds sum)
    p_test_est = preds.mean(axis=0)
    
    # Avoid division by zero
    p_train = np.clip(p_train, 1e-6, 1.0)
    p_test_est = np.clip(p_test_est, 1e-6, 1.0)
    
    if method == 'prior_shift':
        # Formula: P'(y) = P(y) * (P_test / P_train)
        # Shift Factor
        factor = p_test_est / p_train
        
        # Apply shift
        preds_aligned = preds * factor
        
        # Renormalize
        preds_aligned /= preds_aligned.sum(axis=1, keepdims=True)
        return preds_aligned
        
    return preds

def _gauss_rank(x):
    """Convert to Gaussian Rank."""
    # Use quantile transformer for robust rank gauss
    qt = QuantileTransformer(output_distribution='normal')
    return qt.fit_transform(x.reshape(-1, 1)).flatten()

def _softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
