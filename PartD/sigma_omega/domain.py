import numpy as np

from .config import CORAL_REG
from .utils import get_adversarial_weights


def adversarial_weights(X_train, X_test, seed=42, model='lr', clip=10.0, power=1.0):
    """Estimate importance weights w(x) ~ p_test(x) / p_train(x) via adversarial classifier."""
    return get_adversarial_weights(
        X_train=X_train,
        X_test=X_test,
        seed=seed,
        model=model,
        clip=clip,
        power=power,
    )


def coral_align(X_train, X_test, reg=None):
    """CORAL: align covariance of X_train to X_test. Returns transformed (X_train_a, X_test)."""
    if reg is None:
        reg = CORAL_REG

    X_tr = np.asarray(X_train, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)

    X_trc = X_tr - X_tr.mean(axis=0, keepdims=True)
    X_tec = X_te - X_te.mean(axis=0, keepdims=True)

    cov_tr = (X_trc.T @ X_trc) / max(1, (len(X_trc) - 1))
    cov_te = (X_tec.T @ X_tec) / max(1, (len(X_tec) - 1))

    reg = float(reg)
    cov_tr = cov_tr + reg * np.eye(cov_tr.shape[0])
    cov_te = cov_te + reg * np.eye(cov_te.shape[0])

    # cov^{-1/2}
    evals_tr, evecs_tr = np.linalg.eigh(cov_tr)
    evals_tr = np.clip(evals_tr, 1e-12, None)
    W_tr = evecs_tr @ np.diag(1.0 / np.sqrt(evals_tr)) @ evecs_tr.T

    # cov^{1/2}
    evals_te, evecs_te = np.linalg.eigh(cov_te)
    evals_te = np.clip(evals_te, 1e-12, None)
    C_te = evecs_te @ np.diag(np.sqrt(evals_te)) @ evecs_te.T

    A = W_tr @ C_te
    X_tr_a = X_trc @ A + X_tr.mean(axis=0, keepdims=True)
    return X_tr_a.astype(np.float32), X_te.astype(np.float32)
