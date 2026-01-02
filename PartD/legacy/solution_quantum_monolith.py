"""
SIGMA-OMEGA PROTOCOL: THE GRANDMASTER BUILD (FINAL)
--------------------------------------------------------------------------------
The "Combined Arms" Doctrine + "Data Alchemy" + "Grandmaster Rituals" + "RTX Optimization".
Integrates DART, Langevin, True TabR, ThetaTabM, DAE Refinery, MixUp.
Executes 5-Seed Monte Carlo, 10-Fold CV, Isotonic Calibration, and The Razor.

COMMANDER: Research Commander (Reflexion Core)
EXECUTOR: Antigravity Agent
DATE: 2026

ARSENAL:
1. FUEL (Zeta): Refined Neural / Raw Tree Streams.
2. AIR FORCE: True TabR (Attn), ThetaTabM (SAM, Batch=2048, LR=2e-3).
3. INFANTRY: XGBoost DART (RateDrop 0.1), CatBoost Langevin (Temp 1000).
4. RITUALS: Razor (FeatSelect), 10-Fold CV, Isotonic Calib, Seed Averaging.
"""

import os
import sys
import copy
import time
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import SpectralEmbedding


# CONFIGURATION
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Seeds (defaults preserve prior behavior; can override via SEEDS="1,2,3" or N_SEEDS/SEED_BASE)
_seeds_env = os.getenv('SEEDS')
if _seeds_env:
    SEEDS = [int(s.strip()) for s in _seeds_env.split(',') if s.strip()]
else:
    _n_seeds = os.getenv('N_SEEDS')
    if _n_seeds:
        base = int(os.getenv('SEED_BASE', '42'))
        n = int(_n_seeds)
        SEEDS = [base + i for i in range(n)]
    else:
        SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 2048 # RTX 3060 Optimization
LR_SCALE = 2e-3   # Scaled for Batch 2048
SAM_RHO = 0.08    # Compensated for Batch 2048


def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None and v.strip() != '' else int(default)


def _env_float(name, default):
    v = os.getenv(name)
    return float(v) if v is not None and v.strip() != '' else float(default)


ALLOW_TRANSDUCTIVE = _env_bool('ALLOW_TRANSDUCTIVE', False)
USE_STACKING = _env_bool('USE_STACKING', False)
VIEWS = [v.strip().lower() for v in os.getenv('VIEWS', 'raw,quantile').split(',') if v.strip()]

# Stacking enhancements (opt-in)
META_LEARNER = os.getenv('META_LEARNER', 'lr').strip().lower()  # lr | lgbm
USE_TABPFN = _env_bool('USE_TABPFN', False)
TABPFN_N_ENSEMBLES = _env_int('TABPFN_N_ENSEMBLES', 32)
LGBM_MAX_DEPTH = _env_int('LGBM_MAX_DEPTH', 3)
LGBM_NUM_LEAVES = _env_int('LGBM_NUM_LEAVES', 31)
LGBM_N_ESTIMATORS = _env_int('LGBM_N_ESTIMATORS', 400)

# Adversarial validation reweighting (opt-in)
ENABLE_ADV_REWEIGHT = _env_bool('ENABLE_ADV_REWEIGHT', False)
ADV_MODEL = os.getenv('ADV_MODEL', 'lr').strip().lower()  # lr | xgb
ADV_CLIP = _env_float('ADV_CLIP', 10.0)
ADV_POWER = _env_float('ADV_POWER', 1.0)

# SWA (stochastic weight averaging) for ThetaTabM (opt-in)
ENABLE_SWA = _env_bool('ENABLE_SWA', False)
SWA_START_EPOCH = _env_int('SWA_START_EPOCH', 10)

# CORAL feature alignment (opt-in; usually transductive)
ENABLE_CORAL = _env_bool('ENABLE_CORAL', False)
CORAL_REG = _env_float('CORAL_REG', 1e-3)

# Iterative self-training with stability constraints (opt-in; transductive)
ENABLE_SELF_TRAIN = _env_bool('ENABLE_SELF_TRAIN', False)
SELF_TRAIN_ITERS = _env_int('SELF_TRAIN_ITERS', 0)
SELF_TRAIN_CONF = _env_float('SELF_TRAIN_CONF', 0.92)
SELF_TRAIN_AGREE = _env_float('SELF_TRAIN_AGREE', 1.0)  # fraction of seeds that must agree
SELF_TRAIN_VIEW_AGREE = _env_float('SELF_TRAIN_VIEW_AGREE', 0.66)
SELF_TRAIN_MAX = _env_int('SELF_TRAIN_MAX', 10000)
SELF_TRAIN_WEIGHT_POWER = _env_float('SELF_TRAIN_WEIGHT_POWER', 1.0)


@dataclass(frozen=True)
class PseudoData:
    idx: np.ndarray
    y: np.ndarray
    w: np.ndarray

    @staticmethod
    def empty() -> 'PseudoData':
        return PseudoData(
            idx=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.int64),
            w=np.array([], dtype=np.float32),
        )

    def active(self) -> bool:
        return self.idx is not None and self.y is not None and len(self.idx) > 0


def _normalize_pseudo(pseudo_idx=None, pseudo_y=None, pseudo_w=None) -> PseudoData:
    if pseudo_idx is None or pseudo_y is None:
        return PseudoData.empty()
    idx = np.asarray(pseudo_idx, dtype=np.int64)
    y = np.asarray(pseudo_y, dtype=np.int64)
    if pseudo_w is None:
        w = np.ones((len(idx),), dtype=np.float32)
    else:
        w = np.asarray(pseudo_w, dtype=np.float32)
    if len(idx) == 0:
        return PseudoData.empty()
    return PseudoData(idx=idx, y=y, w=w)


def _vote_mode_and_agreement(votes_2d: np.ndarray):
    """votes_2d: (M, N) int labels. Returns (mode_pred[N], agree_frac[N])."""
    mode_pred = np.zeros((votes_2d.shape[1],), dtype=np.int64)
    agree_frac = np.zeros((votes_2d.shape[1],), dtype=np.float64)
    for j in range(votes_2d.shape[1]):
        vals, counts = np.unique(votes_2d[:, j], return_counts=True)
        k = int(np.argmax(counts))
        mode_pred[j] = int(vals[k])
        agree_frac[j] = float(np.max(counts)) / float(votes_2d.shape[0])
    return mode_pred, agree_frac


def _view_agreement_fraction(preds_tensor_vs_n: np.ndarray, mode_pred: np.ndarray):
    """preds_tensor_vs_n: (V, S, N) int labels. Returns view_agree_frac[N]."""
    view_agree_frac = np.zeros((preds_tensor_vs_n.shape[2],), dtype=np.float64)
    for vi in range(preds_tensor_vs_n.shape[0]):
        view_votes = preds_tensor_vs_n[vi]  # (S, N)
        view_mode, _ = _vote_mode_and_agreement(view_votes)
        view_agree_frac += (view_mode == mode_pred).astype(np.float64)
    view_agree_frac /= float(preds_tensor_vs_n.shape[0])
    return view_agree_frac

# Loss-optimized training knobs (defaults preserve existing behavior)
LOSS_NAME = os.getenv('LOSS', 'ce').strip().lower()  # ce | focal
LABEL_SMOOTHING = _env_float('LABEL_SMOOTHING', 0.0)
FOCAL_GAMMA = _env_float('FOCAL_GAMMA', 2.0)
USE_CLASS_BALANCED = _env_bool('CLASS_BALANCED', False)
CB_BETA = _env_float('CB_BETA', 0.999)
USE_MIXUP = _env_bool('USE_MIXUP', True)

# Efficiency knobs
DAE_EPOCHS = _env_int('DAE_EPOCHS', 30)
DAE_NOISE_STD = _env_float('DAE_NOISE_STD', 0.1)
MANIFOLD_K = _env_int('MANIFOLD_K', 20)
ENABLE_PAGERANK = _env_bool('ENABLE_PAGERANK', True)

# LID temperature scaling (opt-in)
ENABLE_LID_SCALING = _env_bool('ENABLE_LID_SCALING', False)
LID_T_MIN = _env_float('LID_T_MIN', 1.0)
LID_T_MAX = _env_float('LID_T_MAX', 2.5)
LID_T_POWER = _env_float('LID_T_POWER', 1.0)

# Test-time training (TTT) on "silver" samples (opt-in; transductive)
ENABLE_TTT = _env_bool('ENABLE_TTT', False)
TTT_GAP_LOW = _env_float('TTT_GAP_LOW', 0.10)
TTT_GAP_HIGH = _env_float('TTT_GAP_HIGH', 0.35)
TTT_EPOCHS = _env_int('TTT_EPOCHS', 1)
TTT_MAX_SAMPLES = _env_int('TTT_MAX_SAMPLES', 4096)
TTT_LR_MULT = _env_float('TTT_LR_MULT', 0.2)


def compute_class_balanced_weights(y, num_classes, beta=0.999):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective + 1e-12)
    weights = weights / (weights.mean() + 1e-12)
    return weights.astype(np.float32)


def smooth_targets(targets, smoothing):
    # targets: (B, C) probs; returns smoothed probs
    if smoothing <= 0:
        return targets
    n_classes = targets.shape[1]
    return targets * (1.0 - smoothing) + (smoothing / n_classes)


def soft_target_ce(logits, targets, class_weights=None):
    log_probs = F.log_softmax(logits, dim=1)
    if class_weights is not None:
        w = class_weights.view(1, -1)
        return -(targets * w * log_probs).sum(dim=1).mean()
    return -(targets * log_probs).sum(dim=1).mean()


def soft_target_focal(logits, targets, gamma=2.0, class_weights=None):
    probs = torch.softmax(logits, dim=1).clamp(1e-8, 1.0 - 1e-8)
    logp = torch.log(probs)
    mod = torch.pow(1.0 - probs, gamma)
    if class_weights is not None:
        w = class_weights.view(1, -1)
        loss = -(targets * w * mod * logp).sum(dim=1)
    else:
        loss = -(targets * mod * logp).sum(dim=1)
    return loss.mean()


def apply_lid_temperature_scaling(probs, lid_norm, t_min=1.0, t_max=2.5, power=1.0):
    """Scale probabilities via per-sample temperature T(lid); lid_norm expected in [0,1]."""
    p = np.asarray(probs, dtype=np.float64)
    lid = np.asarray(lid_norm, dtype=np.float64).reshape(-1)
    lid = np.clip(lid, 0.0, 1.0)

    T = float(t_min) + (float(t_max) - float(t_min)) * np.power(lid, float(power))
    T = np.clip(T, 1e-3, 1e6).reshape(-1, 1)

    logits = np.log(p + 1e-12)
    logits = logits / T
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / (exps.sum(axis=1, keepdims=True) + 1e-12)


def prob_meta_features(probs, lid=None):
    """Meta-features from probability vectors: max prob, gap, entropy (+ optional lid)."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    part = np.partition(p, kth=(-1, -2), axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    gap = top1 - top2
    entropy = -(p * np.log(p)).sum(axis=1)
    feats = np.column_stack([top1, gap, entropy])
    if lid is not None:
        feats = np.column_stack([feats, np.asarray(lid, dtype=np.float64).reshape(-1)])
    return feats.astype(np.float32)


def tabpfn_oof_and_test_proba(X_train, y, X_test, num_classes, cv_splits=10, seed=42, n_ensembles=32):
    """Compute OOF + test probabilities using TabPFN (fresh model per fold)."""
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        raise RuntimeError("USE_TABPFN=1 but `tabpfn` is not installed/importable.") from e

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(X_train), num_classes), dtype=np.float32)
    te_acc = np.zeros((len(X_test), num_classes), dtype=np.float32)

    for tr_idx, val_idx in skf.split(X_train, y):
        model = TabPFNClassifier(device=str(DEVICE), N_ensemble_configurations=int(n_ensembles), seed=int(seed))
        model.fit(X_train[tr_idx], y[tr_idx])
        oof[val_idx] = model.predict_proba(X_train[val_idx]).astype(np.float32)
        te_acc += model.predict_proba(X_test).astype(np.float32)

    te_acc /= cv_splits
    return oof, te_acc


def adversarial_weights(X_train, X_test, seed=42, model='lr', clip=10.0, power=1.0):
    """Estimate importance weights w(x) ~ p_test(x) / p_train(x) via adversarial classifier."""
    X_all = np.vstack([X_train, X_test])
    y_dom = np.concatenate([np.zeros(len(X_train), dtype=np.int64), np.ones(len(X_test), dtype=np.int64)])

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


def coral_align(X_train, X_test, reg=1e-3):
    """CORAL: align covariance of X_train to X_test. Returns transformed (X_train_a, X_test)."""
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


def select_silver_samples(probs, gap_low=0.10, gap_high=0.35, max_samples=4096, seed=42):
    """Pick indices where top1-top2 confidence gap is in [low, high]."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    part = np.partition(p, kth=(-1, -2), axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    gap = top1 - top2
    mask = (gap >= float(gap_low)) & (gap <= float(gap_high))
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx.astype(np.int64), np.array([], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if idx.size > int(max_samples):
        idx = rng.choice(idx, size=int(max_samples), replace=False)
    y_pseudo = np.argmax(p[idx], axis=1).astype(np.int64)
    return idx.astype(np.int64), y_pseudo


def _is_torch_model(m):
    return hasattr(m, 'finetune_on_pseudo') and callable(getattr(m, 'finetune_on_pseudo'))


def apply_feature_view(X_train, X_test, view, seed, allow_transductive=False):
    view = (view or 'raw').strip().lower()
    if view == 'raw':
        X_tr, X_te = X_train, X_test
        if ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=CORAL_REG)
        return X_tr, X_te

    if view == 'quantile':
        qt = QuantileTransformer(output_distribution='normal', random_state=seed)
        X_tr, X_te = qt.fit_transform(X_train), qt.transform(X_test)
        if ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=CORAL_REG)
        return X_tr, X_te

    if view.startswith('pca'):
        n_components = min(50, X_train.shape[1], max(2, X_train.shape[0] - 1))
        pca = PCA(n_components=n_components, random_state=seed)
        X_tr, X_te = pca.fit_transform(X_train), pca.transform(X_test)
        if ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=CORAL_REG)
        return X_tr, X_te

    if view.startswith('ica'):
        n_components = min(50, X_train.shape[1], max(2, X_train.shape[0] - 1))
        ica = FastICA(n_components=n_components, random_state=seed, max_iter=500)
        X_tr, X_te = ica.fit_transform(X_train), ica.transform(X_test)
        if ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=CORAL_REG)
        return X_tr, X_te

    if view.startswith('rp') or view.startswith('random'):
        n_components = min(50, X_train.shape[1])
        rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
        X_tr, X_te = rp.fit_transform(X_train), rp.transform(X_test)
        if ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=CORAL_REG)
        return X_tr, X_te

    if view.startswith('spectral'):
        if not allow_transductive:
            raise ValueError("Spectral view requires transductive embedding; set ALLOW_TRANSDUCTIVE=1 or remove 'spectral' from VIEWS.")
        X_all = np.vstack([X_train, X_test])
        n_components = min(30, X_all.shape[0] - 1)
        se = SpectralEmbedding(n_components=n_components, random_state=seed)
        Z = se.fit_transform(X_all)
        return Z[: len(X_train)], Z[len(X_train) :]

    raise ValueError(f"Unknown view '{view}'. Supported: raw, quantile, pca, ica, rp/random, spectral(transductive)")

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# ==============================================================================
# SECTION 1: THE FUEL (Zeta Refinery)
# ==============================================================================

class TransductiveDAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.SiLU(), nn.Linear(512, 128))
        self.decoder = nn.Sequential(nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

class DataRefinery:
    def __init__(self, input_dim):
        self.dae = TransductiveDAE(input_dim).to(DEVICE)
    
    def fit(self, X_all, epochs=None, noise_std=None):
        if epochs is None:
            epochs = DAE_EPOCHS
        if noise_std is None:
            noise_std = DAE_NOISE_STD
        X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
        dl = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=True)
        opt = optim.AdamW(self.dae.parameters(), lr=LR_SCALE) # Use fast LR
        crit = nn.MSELoss()
        
        self.dae.train()
        for ep in range(epochs):
            for (xb,) in dl:
                noise = torch.randn_like(xb) * float(noise_std)
                rec = self.dae(xb + noise)
                loss = crit(rec, xb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def get_embedding(self, X):
        self.dae.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        res = []
        for i in range(0, len(X), BATCH_SIZE):
            with torch.no_grad(): res.append(self.dae.encoder(X_t[i:i+BATCH_SIZE]).cpu().numpy())
        return np.vstack(res)

    def get_reconstruction(self, X):
        self.dae.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        res = []
        for i in range(0, len(X), BATCH_SIZE):
            with torch.no_grad(): res.append(self.dae(X_t[i:i+BATCH_SIZE]).cpu().numpy())
        return np.vstack(res)

    def transform(self, X):
        """Return (embedding, reconstruction) in one pass per batch."""
        self.dae.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        emb, rec = [], []
        for i in range(0, len(X), BATCH_SIZE):
            xb = X_t[i:i+BATCH_SIZE]
            with torch.no_grad():
                z = self.dae.encoder(xb)
                r = self.dae.decoder(z)
            emb.append(z.cpu().numpy())
            rec.append(r.cpu().numpy())
        return np.vstack(emb), np.vstack(rec)

def compute_manifold_features(X_train, X_test, allow_transductive=False, k=None, enable_pagerank=None, return_lid=False):
    if k is None:
        k = MANIFOLD_K
    if enable_pagerank is None:
        enable_pagerank = ENABLE_PAGERANK
    if allow_transductive:
        X_all = np.vstack([X_train, X_test])
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_all)
        dists, _ = nbrs.kneighbors(X_all)

        d_k = dists[:, -1]
        d_j = dists[:, 1:]
        lid = k / np.sum(np.log(d_k[:, None] / (d_j + 1e-10) + 1e-10), axis=1)
        lid = (lid - lid.min()) / (lid.max() - lid.min() + 1e-12)

        if enable_pagerank:
            try:
                import networkx as nx

                A = kneighbors_graph(X_all, k, mode='connectivity', include_self=False)
                G = nx.from_scipy_sparse_array(A)
                pr = nx.pagerank(G, alpha=0.85, max_iter=50)
                pagerank = np.array([pr[i] for i in range(len(X_all))], dtype=np.float64)
                pagerank = (pagerank - pagerank.min()) / (pagerank.max() - pagerank.min() + 1e-12)
            except Exception:
                pagerank = np.zeros(len(X_all))
        else:
            pagerank = np.zeros(len(X_all))

        feats = np.column_stack([lid, pagerank])
        feats_tr, feats_te = feats[:len(X_train)], feats[len(X_train):]
        if return_lid:
            return feats_tr, feats_te, lid[:len(X_train)], lid[len(X_train):]
        return feats_tr, feats_te

    # Strict (non-transductive): fit neighborhood + graph on train only.
    # For test, approximate PR by neighbor-average PR and compute LID vs train neighbors.
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X_train)), n_jobs=-1).fit(X_train)
    dists_tr, idx_tr = nbrs.kneighbors(X_train)
    dists_tr = dists_tr[:, 1:]
    idx_tr = idx_tr[:, 1:]
    k_eff = dists_tr.shape[1]

    d_k_tr = dists_tr[:, -1]
    d_j_tr = dists_tr[:, :-1] if k_eff > 1 else dists_tr
    lid_tr = k_eff / np.sum(np.log(d_k_tr[:, None] / (d_j_tr + 1e-10) + 1e-10), axis=1)
    lid_tr_min, lid_tr_max = lid_tr.min(), lid_tr.max()
    lid_tr_n = (lid_tr - lid_tr_min) / (lid_tr_max - lid_tr_min + 1e-12)

    if enable_pagerank:
        try:
            import networkx as nx

            A_tr = kneighbors_graph(X_train, min(k, len(X_train) - 1), mode='connectivity', include_self=False)
            G_tr = nx.from_scipy_sparse_array(A_tr)
            pr_tr_dict = nx.pagerank(G_tr, alpha=0.85, max_iter=50)
            pr_tr = np.array([pr_tr_dict[i] for i in range(len(X_train))], dtype=np.float64)
            pr_tr_min, pr_tr_max = pr_tr.min(), pr_tr.max()
            pr_tr_n = (pr_tr - pr_tr_min) / (pr_tr_max - pr_tr_min + 1e-12)
        except Exception:
            pr_tr_n = np.zeros(len(X_train))
    else:
        pr_tr_n = np.zeros(len(X_train))

    # Test features from train neighbors
    dists_te, idx_te = nbrs.kneighbors(X_test, n_neighbors=min(k, len(X_train)))
    k_te = dists_te.shape[1]
    d_k_te = dists_te[:, -1]
    d_j_te = dists_te[:, :-1] if k_te > 1 else dists_te
    lid_te = k_te / np.sum(np.log(d_k_te[:, None] / (d_j_te + 1e-10) + 1e-10), axis=1)
    lid_te_n = (lid_te - lid_tr_min) / (lid_tr_max - lid_tr_min + 1e-12)
    lid_te_n = np.clip(lid_te_n, 0.0, 1.0)

    pr_te_n = pr_tr_n[idx_te].mean(axis=1) if len(pr_tr_n) else np.zeros(len(X_test))

    feats_tr = np.column_stack([lid_tr_n, pr_tr_n])
    feats_te = np.column_stack([lid_te_n, pr_te_n])
    if return_lid:
        return feats_tr, feats_te, lid_tr_n, lid_te_n
    return feats_tr, feats_te

class TopologyMixUpLoader:
    def __init__(self, X, y, num_classes, batch_size=BATCH_SIZE): # Use global batch size
        self.X, self.y, self.num_classes, self.batch_size = X, y, int(num_classes), batch_size
        self.knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X)
        self.rng = np.random.default_rng()
        
    def __iter__(self):
        idxs = self.rng.permutation(len(self.X))
        for i in range(0, len(self.X), self.batch_size):
            b_idxs = idxs[i:i+self.batch_size]
            X_b, y_b = self.X[b_idxs], self.y[b_idxs]
            
            # MixUp
            mn_idxs = self.knn.kneighbors(X_b, return_distance=False)
            rand_n = self.rng.integers(1, 5, size=len(X_b))
            target_idxs = mn_idxs[np.arange(len(X_b)), rand_n]
            X_target = self.X[target_idxs]
            y_target = self.y[target_idxs]
            
            lam = self.rng.beta(0.4, 0.4, size=(len(X_b), 1))
            lam = np.maximum(lam, 1-lam)
            X_mix = lam * X_b + (1 - lam) * X_target

            y_b_oh = np.eye(self.num_classes, dtype=np.float32)[y_b]
            y_t_oh = np.eye(self.num_classes, dtype=np.float32)[y_target]
            y_mix = lam * y_b_oh + (1 - lam) * y_t_oh

            yield (
                torch.tensor(X_mix, dtype=torch.float32).to(DEVICE),
                torch.tensor(y_mix, dtype=torch.float32).to(DEVICE),
            )

# ==============================================================================
# SECTION 2: THE ARSENAL (Sigma-Omega)
# ==============================================================================

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_((torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p))
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        if zero_grad: self.zero_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
    def step(self): raise NotImplementedError

class TabRModule(nn.Module):
    def __init__(self, input_dim, num_classes, context_size=96):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, context_size))
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(context_size,context_size), nn.Linear(context_size,context_size), nn.Linear(context_size,context_size)
        self.head = nn.Sequential(nn.Linear(context_size, 64), nn.SiLU(), nn.Linear(64, num_classes))
        self.scale = context_size ** -0.5
    def forward(self, x, neighbors):
        q = self.encoder(x).unsqueeze(1)
        B, K, D = neighbors.shape
        kv = self.encoder(neighbors.view(B*K, D)).view(B, K, -1)
        scores = torch.bmm(self.q_proj(q), self.k_proj(kv).transpose(1, 2)) * self.scale
        context = torch.bmm(F.softmax(scores, dim=-1), self.v_proj(kv)).squeeze(1)
        return self.head(context + q.squeeze(1))

class TrueTabR(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, n_neighbors=16):
        self.num_classes, self.n_neighbors = num_classes, n_neighbors
        self.model, self.knn, self.X_train_ = None, None, None
    def fit(self, X, y, sample_weight=None):
        self.X_train_ = np.array(X, dtype=np.float32)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(self.X_train_)
        train_neighbor_idx = self.knn.kneighbors(self.X_train_, return_distance=False)
        self.model = TabRModule(X.shape[1], self.num_classes).to(DEVICE)
        opt = optim.AdamW(self.model.parameters(), lr=LR_SCALE) # Scaled LR
        class_w = None
        if USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y, self.num_classes, beta=CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=DEVICE)
        if LABEL_SMOOTHING > 0:
            crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(LABEL_SMOOTHING))
        else:
            crit = nn.CrossEntropyLoss(weight=class_w)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        idx_t = torch.arange(len(X_t), dtype=torch.long)
        if sample_weight is None:
            dl = DataLoader(TensorDataset(X_t, y_t, idx_t), batch_size=BATCH_SIZE, shuffle=True)
        else:
            w_t = torch.tensor(np.asarray(sample_weight, dtype=np.float32))
            dl = DataLoader(TensorDataset(X_t, y_t, idx_t, w_t), batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        for ep in range(15):
            for batch in dl:
                if sample_weight is None:
                    xb, yb, ib = batch
                    wb = None
                else:
                    xb, yb, ib, wb = batch
                nx = self.X_train_[train_neighbor_idx[ib.numpy()]]
                logits = self.model(xb.to(DEVICE), torch.tensor(nx, dtype=torch.float32).to(DEVICE))
                if LOSS_NAME == 'focal':
                    y_onehot = F.one_hot(yb.to(DEVICE), num_classes=self.num_classes).float()
                    y_onehot = smooth_targets(y_onehot, LABEL_SMOOTHING)
                    probs = torch.softmax(logits, dim=1).clamp(1e-8, 1.0 - 1e-8)
                    pt = (probs * y_onehot).sum(dim=1)
                    loss_vec = -torch.pow(1.0 - pt, float(FOCAL_GAMMA)) * torch.log(pt)
                    if class_w is not None:
                        w_class = (y_onehot * class_w.view(1, -1)).sum(dim=1)
                        loss_vec = loss_vec * w_class
                    if wb is not None:
                        loss_vec = loss_vec * wb.to(DEVICE)
                    loss = loss_vec.mean()
                else:
                    loss_vec = F.cross_entropy(
                        logits,
                        yb.to(DEVICE),
                        reduction='none',
                        weight=class_w,
                        label_smoothing=float(LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else 0.0,
                    )
                    if wb is not None:
                        loss_vec = loss_vec * wb.to(DEVICE)
                    loss = loss_vec.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def finetune_on_pseudo(self, X_pseudo, y_pseudo, epochs=1, lr_mult=0.2):
        if X_pseudo is None or len(X_pseudo) == 0:
            return self
        self.model.train()
        lr = float(LR_SCALE) * float(lr_mult)
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        class_w = None
        if USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y_pseudo, self.num_classes, beta=CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=DEVICE)

        if LABEL_SMOOTHING > 0:
            crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(LABEL_SMOOTHING))
        else:
            crit = nn.CrossEntropyLoss(weight=class_w)

        X_t = torch.tensor(np.asarray(X_pseudo, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.int64))
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(int(epochs)):
            for xb, yb in dl:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                nx = self.X_train_[self.knn.kneighbors(xb.detach().cpu().numpy(), return_distance=False)]
                logits = self.model(xb, torch.tensor(nx, dtype=torch.float32).to(DEVICE))
                if LOSS_NAME == 'focal':
                    y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                    y_onehot = smooth_targets(y_onehot, LABEL_SMOOTHING)
                    loss = soft_target_focal(logits, y_onehot, gamma=FOCAL_GAMMA, class_weights=class_w)
                else:
                    loss = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self
    def predict_proba(self, X):
        self.model.eval()
        p = []
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            nx = self.X_train_[self.knn.kneighbors(xb, return_distance=False)]
            with torch.no_grad(): p.append(torch.softmax(self.model(torch.tensor(xb, dtype=torch.float32).to(DEVICE), torch.tensor(nx).to(DEVICE)), dim=1).cpu().numpy())
        return np.vstack(p)

class ThetaTabM(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes):
        self.num_classes = int(num_classes)
        self.model = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, num_classes)).to(DEVICE)

    def fit(self, X, y, sample_weight=None):
        opt = SAM(self.model.parameters(), optim.AdamW, lr=LR_SCALE, rho=SAM_RHO) # Calibrated SAM
        class_w = None
        if USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y, self.num_classes, beta=CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=DEVICE)

        if LABEL_SMOOTHING > 0:
            hard_crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(LABEL_SMOOTHING))
        else:
            hard_crit = nn.CrossEntropyLoss(weight=class_w)

        self.model.train()
        swa_model = None
        if ENABLE_SWA:
            try:
                from torch.optim.swa_utils import AveragedModel
                swa_model = AveragedModel(self.model)
            except Exception:
                swa_model = None
        for ep in range(20):
            use_mixup_local = USE_MIXUP and (sample_weight is None)
            if use_mixup_local:
                iterator = TopologyMixUpLoader(X, y, num_classes=self.num_classes)
                for xb, yb in iterator:
                    yb = smooth_targets(yb, LABEL_SMOOTHING)
                    opt.zero_grad()
                    logits = self.model(xb)
                    if LOSS_NAME == 'focal':
                        loss = soft_target_focal(logits, yb, gamma=FOCAL_GAMMA, class_weights=class_w)
                    else:
                        loss = soft_target_ce(logits, yb, class_weights=class_w)
                    loss.backward(); opt.first_step(zero_grad=True)

                    logits2 = self.model(xb)
                    if LOSS_NAME == 'focal':
                        loss2 = soft_target_focal(logits2, yb, gamma=FOCAL_GAMMA, class_weights=class_w)
                    else:
                        loss2 = soft_target_ce(logits2, yb, class_weights=class_w)
                    loss2.backward(); opt.second_step(zero_grad=True); opt.base_optimizer.step()
            else:
                if sample_weight is None:
                    w_arr = np.ones(len(X), dtype=np.float32)
                else:
                    w_arr = np.asarray(sample_weight, dtype=np.float32)
                dl = DataLoader(
                    TensorDataset(
                        torch.tensor(X, dtype=torch.float32).to(DEVICE),
                        torch.tensor(y, dtype=torch.long).to(DEVICE),
                        torch.tensor(w_arr, dtype=torch.float32).to(DEVICE),
                    ),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )
                for xb, yb, wb in dl:
                    opt.zero_grad()
                    logits = self.model(xb)
                    if LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, LABEL_SMOOTHING)
                        probs = torch.softmax(logits, dim=1).clamp(1e-8, 1.0 - 1e-8)
                        pt = (probs * y_onehot).sum(dim=1)
                        loss_vec = -torch.pow(1.0 - pt, float(FOCAL_GAMMA)) * torch.log(pt)
                        if class_w is not None:
                            w_class = (y_onehot * class_w.view(1, -1)).sum(dim=1)
                            loss_vec = loss_vec * w_class
                        loss = (loss_vec * wb).mean()
                    else:
                        loss_vec = F.cross_entropy(
                            logits,
                            yb,
                            reduction='none',
                            weight=class_w,
                            label_smoothing=float(LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else 0.0,
                        )
                        loss = (loss_vec * wb).mean()
                    loss.backward(); opt.first_step(zero_grad=True)

                    logits2 = self.model(xb)
                    if LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, LABEL_SMOOTHING)
                        probs2 = torch.softmax(logits2, dim=1).clamp(1e-8, 1.0 - 1e-8)
                        pt2 = (probs2 * y_onehot).sum(dim=1)
                        loss2_vec = -torch.pow(1.0 - pt2, float(FOCAL_GAMMA)) * torch.log(pt2)
                        if class_w is not None:
                            w_class = (y_onehot * class_w.view(1, -1)).sum(dim=1)
                            loss2_vec = loss2_vec * w_class
                        loss2 = (loss2_vec * wb).mean()
                    else:
                        loss2_vec = F.cross_entropy(
                            logits2,
                            yb,
                            reduction='none',
                            weight=class_w,
                            label_smoothing=float(LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else 0.0,
                        )
                        loss2 = (loss2_vec * wb).mean()
                    loss2.backward(); opt.second_step(zero_grad=True); opt.base_optimizer.step()

            if swa_model is not None and ep >= int(SWA_START_EPOCH):
                swa_model.update_parameters(self.model)

        if swa_model is not None and int(SWA_START_EPOCH) < 20:
            self.model.load_state_dict(swa_model.module.state_dict())
        return self

    def finetune_on_pseudo(self, X_pseudo, y_pseudo, epochs=1, lr_mult=0.2):
        if X_pseudo is None or len(X_pseudo) == 0:
            return self
        self.model.train()
        lr = float(LR_SCALE) * float(lr_mult)
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        class_w = None
        if USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y_pseudo, self.num_classes, beta=CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=DEVICE)

        if LABEL_SMOOTHING > 0:
            hard_crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(LABEL_SMOOTHING))
        else:
            hard_crit = nn.CrossEntropyLoss(weight=class_w)

        X_t = torch.tensor(np.asarray(X_pseudo, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.int64))
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(int(epochs)):
            for xb, yb in dl:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = self.model(xb)
                if LOSS_NAME == 'focal':
                    y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                    y_onehot = smooth_targets(y_onehot, LABEL_SMOOTHING)
                    loss = soft_target_focal(logits, y_onehot, gamma=FOCAL_GAMMA, class_weights=class_w)
                else:
                    loss = hard_crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self
    def predict_proba(self, X):
        self.model.eval()
        p = []
        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE): p.append(torch.softmax(self.model(torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy())
        return np.vstack(p)

# ==============================================================================
# SECTION 3: THE RITUAL (Execution)
# ==============================================================================
def load_data_safe():
    try:
        from src.data_loader import load_data
    except Exception as e:
        raise RuntimeError(
            "Failed to import `src.data_loader.load_data`. Ensure `PartD/src` is on PYTHONPATH and dependencies are installed."
        ) from e

    try:
        X, y, X_test = load_data()
    except Exception as e:
        raise RuntimeError("`load_data()` raised an exception.") from e

    if X is None or y is None or X_test is None:
        raise ValueError("`load_data()` returned None(s); expected (X, y, X_test).")
    return X, y, X_test

class CalibratedModel:
    def __init__(self, base_model, name): self.base, self.name, self.ir = base_model, name, None
    def fit(self, X, y, sample_weight=None, pseudo_X=None, pseudo_y=None, pseudo_w=None):
        # 10-Fold CV for Calibration
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_preds, oof_targets = [], []
        
        # Real training happens inside CV? No, Grandmaster Logic: 
        # Train on 9 folds, Predict 1 fold, fit Isotonic. 
        # But we need a Final Model for Test.
        # Strategy: Train on ALL data for Final Prediction. Use CV ONLY to learn Calibration Map?
        # Or standard Stacking approach: OOF predictions -> Calibrator. 
        # For simplicity in this script: Split 90/10 once for calibration fit? No, 10-Fold is requested.
        # We will train 10 models and average their calibrated predictions.
        self.models = []
        self.calibrators = [] # Per class Isotonic
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

            if pseudo_X is not None and len(pseudo_X) > 0:
                X_tr = np.vstack([X_tr, pseudo_X])
                y_tr = np.concatenate([y_tr, pseudo_y])
                if sw_tr is None:
                    sw_tr = np.ones(len(y_tr), dtype=np.float32)
                    sw_tr[: len(tr_idx)] = 1.0
                else:
                    sw_tr = np.concatenate([sw_tr, np.asarray(pseudo_w, dtype=np.float32)])
            
            model = copy.deepcopy(self.base)
            try:
                model.fit(X_tr, y_tr, sample_weight=sw_tr)
            except TypeError:
                model.fit(X_tr, y_tr)
            
            # Calibrate
            val_probs = model.predict_proba(X_val).astype(np.float32)
            c_list = []
            for c in range(val_probs.shape[1]):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(val_probs[:, c], (y_val == c).astype(int))
                c_list.append(iso)
            
            self.models.append(model)
            self.calibrators.append(c_list)
            # print(f"    [CV] Fold {fold+1}/10 done.")
        return self

    def predict_proba(self, X):
        # Average of 10 calibrated models
        total_probs = np.zeros((len(X), len(self.calibrators[0])))
        for model, calib_list in zip(self.models, self.calibrators):
            raw_p = model.predict_proba(X)
            cal_p = np.zeros_like(raw_p)
            for c in range(raw_p.shape[1]):
                cal_p[:, c] = calib_list[c].predict(raw_p[:, c])
            # Re-normalize
            cal_p /= (cal_p.sum(axis=1, keepdims=True) + 1e-10)
            total_probs += cal_p
        return total_probs / len(self.models)


def fit_predict_stacking(
    names_models,
    X_tree_tr,
    X_tree_te,
    X_neural_tr,
    X_neural_te,
    X_view_tr,
    X_view_te,
    y,
    num_classes,
    lid_tr=None,
    lid_te=None,
    cv_splits=10,
    seed=42,
    sample_weight=None,
    pseudo_X_tree=None,
    pseudo_X_neural=None,
    pseudo_y=None,
    pseudo_w=None,
):
    """True stacking: generate OOF probs for base models, fit meta-learner, predict test."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    oof_blocks = []
    test_blocks = []

    meta_feat_oof = []
    meta_feat_te = []

    for name, base in names_models:
        data_tr = X_tree_tr if ('XGB' in name or 'Cat' in name) else X_neural_tr
        data_te = X_tree_te if ('XGB' in name or 'Cat' in name) else X_neural_te

        pX = None
        if pseudo_y is not None:
            pX = pseudo_X_tree if ('XGB' in name or 'Cat' in name) else pseudo_X_neural

        oof = np.zeros((len(data_tr), num_classes), dtype=np.float32)
        te_acc = np.zeros((len(data_te), num_classes), dtype=np.float32)

        for tr_idx, val_idx in skf.split(data_tr, y):
            model = copy.deepcopy(base)
            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
            X_fold = data_tr[tr_idx]
            y_fold = y[tr_idx]
            if pX is not None and len(pX) > 0:
                X_fold = np.vstack([X_fold, pX])
                y_fold = np.concatenate([y_fold, pseudo_y])
                if sw_tr is None and pseudo_w is not None:
                    sw_tr = np.ones(len(y_fold), dtype=np.float32)
                    sw_tr[: len(tr_idx)] = 1.0
                    sw_tr[len(tr_idx) :] = np.asarray(pseudo_w, dtype=np.float32)
                elif sw_tr is not None and pseudo_w is not None:
                    sw_tr = np.concatenate([sw_tr, np.asarray(pseudo_w, dtype=np.float32)])
            try:
                model.fit(X_fold, y_fold, sample_weight=sw_tr)
            except TypeError:
                model.fit(X_fold, y_fold)
            oof[val_idx] = model.predict_proba(data_tr[val_idx]).astype(np.float32)
            te_acc += model.predict_proba(data_te).astype(np.float32)

        te_acc /= cv_splits
        oof_blocks.append(oof)
        test_blocks.append(te_acc)

        meta_feat_oof.append(prob_meta_features(oof))
        meta_feat_te.append(prob_meta_features(te_acc))

    # Optional TabPFN as an extra base model (on the view features, before refinery)
    if USE_TABPFN:
        tab_oof, tab_te = tabpfn_oof_and_test_proba(
            X_view_tr,
            y,
            X_view_te,
            num_classes=num_classes,
            cv_splits=cv_splits,
            seed=seed,
            n_ensembles=TABPFN_N_ENSEMBLES,
        )
        oof_blocks.append(tab_oof)
        test_blocks.append(tab_te)
        meta_feat_oof.append(prob_meta_features(tab_oof))
        meta_feat_te.append(prob_meta_features(tab_te))

    n_experts = len(oof_blocks)

    # Base stacking features
    meta_X = np.hstack(oof_blocks)
    meta_te = np.hstack(test_blocks)

    # Meta-features (gap/entropy/etc.) + optional LID
    meta_X_meta = np.hstack(meta_feat_oof) if len(meta_feat_oof) else None
    meta_te_meta = np.hstack(meta_feat_te) if len(meta_feat_te) else None
    if lid_tr is not None:
        meta_X_meta = np.hstack([meta_X_meta, np.asarray(lid_tr, dtype=np.float32).reshape(-1, 1)]) if meta_X_meta is not None else np.asarray(lid_tr, dtype=np.float32).reshape(-1, 1)
    if lid_te is not None:
        meta_te_meta = np.hstack([meta_te_meta, np.asarray(lid_te, dtype=np.float32).reshape(-1, 1)]) if meta_te_meta is not None else np.asarray(lid_te, dtype=np.float32).reshape(-1, 1)

    # Mixture-of-experts gate: learn which expert to trust per sample, then blend expert probs
    if META_LEARNER == 'moe':
        # Expert label = expert with best per-sample NLL on OOF
        y_int = np.asarray(y, dtype=np.int64)
        nll = np.zeros((len(y_int), n_experts), dtype=np.float64)
        for i, oof in enumerate(oof_blocks):
            p_true = np.clip(oof[np.arange(len(y_int)), y_int], 1e-12, 1.0)
            nll[:, i] = -np.log(p_true)
        expert_label = np.argmin(nll, axis=1).astype(np.int64)

        gate_X = meta_X_meta if meta_X_meta is not None else meta_X
        gate_te = meta_te_meta if meta_te_meta is not None else meta_te

        # Train gating model
        # prefer LightGBM if available
        try:
            import importlib

            LGBMClassifier = importlib.import_module('lightgbm').LGBMClassifier
            gate = LGBMClassifier(
                objective='multiclass',
                num_class=int(n_experts),
                max_depth=2,
                num_leaves=min(31, max(2, 2 ** 2)),
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
        except Exception:
            gate = LogisticRegression(max_iter=2000, multi_class='multinomial')

        if sample_weight is None:
            gate.fit(gate_X, expert_label)
        else:
            gate.fit(gate_X, expert_label, sample_weight=np.asarray(sample_weight, dtype=np.float32))

        gate_w = gate.predict_proba(gate_te).astype(np.float64)  # (N, n_experts)
        gate_w = gate_w / (gate_w.sum(axis=1, keepdims=True) + 1e-12)

        # Blend expert test probs
        p_final = np.zeros_like(test_blocks[0], dtype=np.float64)
        for i, p_i in enumerate(test_blocks):
            p_final += gate_w[:, [i]] * p_i.astype(np.float64)
        p_final = p_final / (p_final.sum(axis=1, keepdims=True) + 1e-12)
        return p_final.astype(np.float32)

    # Otherwise: normal meta-learner on concatenated probs (+ meta-features)
    if meta_X_meta is not None:
        meta_X = np.hstack([meta_X, meta_X_meta])
    if meta_te_meta is not None:
        meta_te = np.hstack([meta_te, meta_te_meta])

    if META_LEARNER == 'lgbm':
        try:
            import importlib

            LGBMClassifier = importlib.import_module('lightgbm').LGBMClassifier
        except Exception as e:
            raise RuntimeError("META_LEARNER=lgbm but `lightgbm` is not installed/importable.") from e
        meta = LGBMClassifier(
            objective='multiclass',
            num_class=int(num_classes),
            max_depth=int(LGBM_MAX_DEPTH),
            num_leaves=int(LGBM_NUM_LEAVES),
            n_estimators=int(LGBM_N_ESTIMATORS),
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        meta = LogisticRegression(max_iter=2000, multi_class='multinomial')

    if sample_weight is None:
        meta.fit(meta_X, y)
    else:
        meta.fit(meta_X, y, sample_weight=np.asarray(sample_weight, dtype=np.float32))
    return meta.predict_proba(meta_te)


def build_streams(X_v, X_test_v):
    ref_fit_X = np.vstack([X_v, X_test_v]) if ALLOW_TRANSDUCTIVE else X_v
    ref = DataRefinery(X_v.shape[1]).fit(ref_fit_X)

    feats_tr, feats_te, lid_tr, lid_te = compute_manifold_features(
        X_v,
        X_test_v,
        allow_transductive=ALLOW_TRANSDUCTIVE,
        k=MANIFOLD_K,
        enable_pagerank=ENABLE_PAGERANK,
        return_lid=True,
    )

    emb_tr, rec_tr = ref.transform(X_v)
    emb_te, rec_te = ref.transform(X_test_v)

    X_neural_tr = np.hstack([X_v, feats_tr, emb_tr])
    X_neural_te = np.hstack([X_test_v, feats_te, emb_te])
    X_tree_tr = np.hstack([X_v, feats_tr, rec_tr])
    X_tree_te = np.hstack([X_test_v, feats_te, rec_te])
    return X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te


def predict_probs_for_view(view, seed, X_train_base, X_test_base, y_enc, num_classes, pseudo_idx=None, pseudo_y=None, pseudo_w=None):
    pseudo = _normalize_pseudo(pseudo_idx=pseudo_idx, pseudo_y=pseudo_y, pseudo_w=pseudo_w)

    X_v, X_test_v = apply_feature_view(
        X_train_base,
        X_test_base,
        view=view,
        seed=seed,
        allow_transductive=ALLOW_TRANSDUCTIVE,
    )

    X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te = build_streams(X_v, X_test_v)

    pseudo_X_tree = None
    pseudo_X_neural = None
    if pseudo.active():
        pseudo_X_tree = X_tree_te[pseudo.idx]
        pseudo_X_neural = X_neural_te[pseudo.idx]

    sample_weight = None
    if ENABLE_ADV_REWEIGHT:
        sample_weight = adversarial_weights(
            X_v,
            X_test_v,
            seed=seed,
            model=ADV_MODEL,
            clip=ADV_CLIP,
            power=ADV_POWER,
        )

    names_models = [
        ('XGB_DART', get_xgb_dart(num_classes)),
        ('Cat_Langevin', get_cat_langevin(num_classes)),
        ('ThetaTabM', ThetaTabM(X_neural_tr.shape[1], num_classes)),
        ('TrueTabR', TrueTabR(num_classes)),
    ]

    if USE_STACKING:
        p = fit_predict_stacking(
            names_models,
            X_tree_tr,
            X_tree_te,
            X_neural_tr,
            X_neural_te,
            X_v,
            X_test_v,
            y_enc,
            num_classes,
            lid_tr=lid_tr,
            lid_te=lid_te,
            cv_splits=10,
            seed=seed,
            sample_weight=sample_weight,
            pseudo_X_tree=pseudo_X_tree,
            pseudo_X_neural=pseudo_X_neural,
            pseudo_y=pseudo.y if pseudo.active() else None,
            pseudo_w=pseudo.w if pseudo.active() else None,
        )
        if ENABLE_LID_SCALING:
            p = apply_lid_temperature_scaling(
                p,
                lid_te,
                t_min=LID_T_MIN,
                t_max=LID_T_MAX,
                power=LID_T_POWER,
            )
        return p

    view_probs = 0
    for name, base in names_models:
        print(f"  > Calibrating {name} (10-Fold)...")
        data_tr = X_tree_tr if ('XGB' in name or 'Cat' in name) else X_neural_tr
        data_te = X_tree_te if ('XGB' in name or 'Cat' in name) else X_neural_te

        calibrated = CalibratedModel(base, name)
        calibrated.fit(
            data_tr,
            y_enc,
            sample_weight=sample_weight,
            pseudo_X=pseudo_X_tree if ('XGB' in name or 'Cat' in name) else pseudo_X_neural,
            pseudo_y=pseudo.y if pseudo.active() else None,
            pseudo_w=pseudo.w if pseudo.active() else None,
        )
        p = calibrated.predict_proba(data_te)

        # Optional TTT for torch models using silver samples from current predictions
        if ENABLE_TTT and _is_torch_model(base):
            if not ALLOW_TRANSDUCTIVE:
                raise RuntimeError("ENABLE_TTT requires ALLOW_TRANSDUCTIVE=1 (it adapts on test features).")
            idx_silver, y_pseudo = select_silver_samples(
                p,
                gap_low=TTT_GAP_LOW,
                gap_high=TTT_GAP_HIGH,
                max_samples=TTT_MAX_SAMPLES,
                seed=seed,
            )
            if idx_silver.size > 0:
                base.finetune_on_pseudo(
                    data_te[idx_silver],
                    y_pseudo,
                    epochs=TTT_EPOCHS,
                    lr_mult=TTT_LR_MULT,
                )
                # refresh probabilities after adaptation
                p = base.predict_proba(data_te)

        # Optional LID temperature scaling on test probs
        if ENABLE_LID_SCALING:
            p = apply_lid_temperature_scaling(
                p,
                lid_te,
                t_min=LID_T_MIN,
                t_max=LID_T_MAX,
                power=LID_T_POWER,
            )

        view_probs += p
    return view_probs / len(names_models)

def main():
    print(">>> INITIATING SIGMA-OMEGA GRANDMASTER PROTOCOL <<<")
    
    # 1. LOAD & RAZOR
    X, y, X_test = load_data_safe()
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Razor (Seed 42 Scout)
    print("[RAZOR] Scanning for noise features...")
    from catboost import CatBoostClassifier
    scout = CatBoostClassifier(iterations=500, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    # Drop bottom 20%
    thresh = np.percentile(imps, 20)
    keep_mask = imps > thresh
    X_razor = X[:, keep_mask]; X_test_razor = X_test[:, keep_mask]
    print(f"  > Dropped {np.sum(~keep_mask)} features. New Dim: {X_razor.shape[1]}")
    
    # 2. MONTE CARLO LOOP
    final_ensemble_probs = 0

    if ENABLE_SELF_TRAIN and SELF_TRAIN_ITERS > 0:
        if not ALLOW_TRANSDUCTIVE:
            raise RuntimeError("ENABLE_SELF_TRAIN requires ALLOW_TRANSDUCTIVE=1 (it uses test features for pseudo-labeling).")

        pseudo = PseudoData.empty()

        last_avg_probs = None
        for it in range(int(SELF_TRAIN_ITERS) + 1):
            if it == 0:
                print("\n>>> SELF-TRAIN ITERATION 0 (no pseudo) <<<")
            else:
                print(f"\n>>> SELF-TRAIN ITERATION {it} (pseudo={len(pseudo.idx)}) <<<")

            probs_per_view = {v: [] for v in VIEWS}
            preds_per_view = {v: [] for v in VIEWS}

            for seed in SEEDS:
                seed_everything(seed)
                for view in VIEWS:
                    p = predict_probs_for_view(
                        view,
                        seed,
                        X_razor,
                        X_test_razor,
                        y_enc,
                        num_classes,
                        pseudo_idx=pseudo.idx,
                        pseudo_y=pseudo.y,
                        pseudo_w=pseudo.w,
                    )
                    probs_per_view[view].append(p)
                    preds_per_view[view].append(np.argmax(p, axis=1))

            # Build tensors
            # probs_tensor: (V, S, N, C); preds_tensor: (V, S, N)
            probs_tensor = []
            preds_tensor = []
            for view in VIEWS:
                probs_tensor.append(np.stack(probs_per_view[view], axis=0))
                preds_tensor.append(np.stack(preds_per_view[view], axis=0))
            probs_tensor = np.stack(probs_tensor, axis=0)
            preds_tensor = np.stack(preds_tensor, axis=0)

            avg_probs = probs_tensor.mean(axis=(0, 1))  # (N, C)
            last_avg_probs = avg_probs

            if it < int(SELF_TRAIN_ITERS):
                # Vote-level stability across seeds×views
                # votes are shape (V*S, N)
                votes = preds_tensor.reshape(preds_tensor.shape[0] * preds_tensor.shape[1], preds_tensor.shape[2])

                mode_pred, agree_frac_votes = _vote_mode_and_agreement(votes)
                view_agree_frac = _view_agreement_fraction(preds_tensor, mode_pred)

                conf = np.max(avg_probs, axis=1)
                mask = (
                    (conf >= float(SELF_TRAIN_CONF))
                    & (agree_frac_votes >= float(SELF_TRAIN_AGREE))
                    & (view_agree_frac >= float(SELF_TRAIN_VIEW_AGREE))
                )
                idx = np.nonzero(mask)[0]

                if idx.size > int(SELF_TRAIN_MAX):
                    top = np.argsort(conf[idx])[::-1][: int(SELF_TRAIN_MAX)]
                    idx = idx[top]

                pseudo_idx = idx.astype(np.int64)
                pseudo_y = mode_pred[pseudo_idx]
                pseudo_w = np.power(conf[pseudo_idx].astype(np.float32), float(SELF_TRAIN_WEIGHT_POWER))
                pseudo = PseudoData(idx=pseudo_idx, y=pseudo_y, w=pseudo_w)

                print(
                    f"  [SELF-TRAIN] mined {len(pseudo_idx)} pseudo (conf>={SELF_TRAIN_CONF}, votes>={SELF_TRAIN_AGREE}, views>={SELF_TRAIN_VIEW_AGREE})"
                )

        final_ensemble_probs = last_avg_probs

    else:
        for seed in SEEDS:
            print(f"\n>>> SEQUENCE START: SEED {seed} <<<")
            seed_everything(seed)

            view_probs_total = 0
            view_count = 0
            for view in VIEWS:
                print(f"  [VIEW] {view}")
                if USE_STACKING:
                    print("  > Stacking meta-learner (OOF -> meta)...")
                view_probs_total += predict_probs_for_view(view, seed, X_razor, X_test_razor, y_enc, num_classes)
                view_count += 1
            final_ensemble_probs += (view_probs_total / max(1, view_count))

        final_ensemble_probs /= len(SEEDS)

    # 3. FINAL OUTPUT
    preds = np.argmax(final_ensemble_probs, axis=1)
    labels = le.inverse_transform(preds)
    
    os.makedirs('PartD/outputs', exist_ok=True)
    np.save('PartD/outputs/labelsX_grandmaster.npy', labels)
    print("\n>>> GRANDMASTER PROTOCOL COMPLETE <<<")

# HELPERS FOR SIGMA MODELS
def get_xgb_dart(n_c):
    from xgboost import XGBClassifier
    # DART: Dropout for Trees
    use_gpu = torch.cuda.is_available()
    params = dict(
        booster='dart',
        rate_drop=0.1,
        skip_drop=0.5,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=int(n_c),
        eval_metric='mlogloss',
        verbosity=0,
    )
    if use_gpu:
        params.update(tree_method='gpu_hist', predictor='gpu_predictor')
    else:
        params.update(tree_method='hist')
    return XGBClassifier(**params)

def get_cat_langevin(n_c):
    from catboost import CatBoostClassifier
    # Langevin: Stochastic Gradient MCMC
    return CatBoostClassifier(
        langevin=True, diffusion_temperature=1000,
        iterations=1000, depth=8, learning_rate=0.03,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        task_type='GPU' if torch.cuda.is_available() else 'CPU',
        verbose=0, allow_writing_files=False
    )

if __name__ == "__main__":
    main()
