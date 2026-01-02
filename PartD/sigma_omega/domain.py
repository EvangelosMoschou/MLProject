import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import CORAL_REG


def adversarial_weights(X_train, X_test, seed=42, model='lr', clip=10.0, power=1.0):
    """Estimate importance weights w(x) ~ p_test(x) / p_train(x) via adversarial classifier."""
    X_all = np.vstack([X_train, X_test])
    y_dom = np.concatenate([
        np.zeros(len(X_train), dtype=np.int64),
        np.ones(len(X_test), dtype=np.int64),
    ])

    if model == 'xgb':
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=int(seed),
            verbosity=0,
        )
    else:
        clf = LogisticRegression(max_iter=2000)

    clf.fit(X_all, y_dom)
    p_test = clf.predict_proba(X_train)[:, 1].astype(np.float64)
    p_test = np.clip(p_test, 1e-6, 1.0 - 1e-6)
    w = p_test / (1.0 - p_test)
    w = np.power(w, float(power))
    w = np.clip(w, 1.0 / float(clip), float(clip))
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float32)


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
