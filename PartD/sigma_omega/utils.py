import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def seed_everything(seed=42):
    import random

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_adversarial_weights(
    X_train,
    X_test,
    seed=42,
    model='lr',
    clip=10.0,
    power=1.0,
    auc_low=0.55,
    auc_high=0.70,
):
    """Estimate importance weights w(x) ~ p_test(x) / p_train(x) with AUC gating."""
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
    p_all = clf.predict_proba(X_all)[:, 1].astype(np.float64)
    auc = roc_auc_score(y_dom, p_all)

    if auc < float(auc_low):
        return np.ones(len(X_train), dtype=np.float32)

    if auc < float(auc_high):
        power = float(power) * 0.5
        clip = min(float(clip), 5.0)

    p_test = clf.predict_proba(X_train)[:, 1].astype(np.float64)
    p_test = np.clip(p_test, 1e-6, 1.0 - 1e-6)
    w = p_test / (1.0 - p_test)
    w = np.power(w, float(power))
    w = np.clip(w, 1.0 / float(clip), float(clip))
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float32)
