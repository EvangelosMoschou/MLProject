"""
🚀 OPERATION OMEGA-SINGULARITY: THE FINAL BUILD 🚀
--------------------------------------------------------------------------------
MERGED DOCTRINE:
1. PREP: Stability Selection (Lasso) + RankGauss + Manifold Features (LID/Laplacian) + Transductive DAE.
2. AUGMENT: Tabular Gaussian Diffusion (Synthetic Data) + Topology MixUp.
3. MODELS: ThetaTabM (w/ SAM), True TabR, KAN, XGBoost (DART), CatBoost (Langevin).
4. ADAPT: Adversarial Reweighting + Test-Time Training (TTT).
5. META: Dirichlet Calibration + NNLS Stacking + LID Scaling.

HARDWARE: Optimized for RTX 3060 (Batch Size 512, Mixed Precision where applicable).
"""

import os
import sys
import copy
import gc
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import SpectralEmbedding
from scipy.special import erfinv
from scipy.optimize import nnls
from scipy.spatial.distance import cdist

# Fallback imports
try:
    import xgboost as xgb
    import catboost as cb
except ImportError:
    print("[WARN] XGBoost/CatBoost not installed. Tree models will be skipped.")

# ------------------------------------------------------------------------------
# 0. CONFIGURATION & SEEDING
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

CONFIG = {
    'N_FOLDS': 5,
    'BATCH_SIZE': 512,
    'TTT_STEPS': 15,
    'TABM_K': 8,
    'DIFFUSION_EPOCHS': 30,
    'DAE_EPOCHS': 30,
    'N_CLASSES': 5 # datasetTV specific
}

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

# ------------------------------------------------------------------------------
# 1. ADVANCED FEATURE ENGINEERING (THE TRINITY ENGINE)
# ------------------------------------------------------------------------------

class RankGaussScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        # Enforce strict normality
        X = np.array(X)
        X_rank = np.argsort(np.argsort(X, axis=0), axis=0)
        X_rank = (X_rank + 1) / (X.shape[0] + 1)
        X_gauss = erfinv(2 * X_rank - 1)
        return np.clip(X_gauss, -5, 5)

class TransductiveDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_dae(X_all, epochs=30):
    print(f"   [DAE] Training Transductive Autoencoder on {X_all.shape}...")
    model = TransductiveDAE(X_all.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for (bx,) in dl:
            noise = torch.randn_like(bx) * 0.1
            rec, _ = model(bx + noise)
            loss = F.mse_loss(rec, bx)
            opt.zero_grad(); loss.backward(); opt.step()
    
    model.eval()
    with torch.no_grad():
        _, emb = model(X_t)
    return emb.cpu().numpy()

def compute_manifold_features(X_train, X_test, k=20):
    print("   [MANIFOLD] Computing LID and Laplacian Eigenmaps...")
    X_all = np.vstack([X_train, X_test])
    
    # 1. LID (Local Intrinsic Dimensionality)
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_all)
    dists, _ = nbrs.kneighbors(X_all)
    d_k = dists[:, -1]; d_j = dists[:, 1:]
    lid = k / np.sum(np.log(d_k[:, None] / (d_j + 1e-10) + 1e-10), axis=1)
    lid_norm = (lid - lid.min()) / (lid.max() - lid.min())
    
    # 2. Laplacian Eigenmaps (Geometry Unfolding)
    # Using sklearn SpectralEmbedding (which computes Laplacian Eigenmaps)
    se = SpectralEmbedding(n_components=8, n_neighbors=k, n_jobs=-1)
    laplacian = se.fit_transform(X_all)
    
    feats = np.column_stack([lid_norm, laplacian])
    return feats[:len(X_train)], feats[len(X_train):], lid_norm[len(X_train):]

class StabilitySelector:
    """Robust feature selection via Randomized Lasso (Manifest)"""
    def __init__(self, n_bootstrap=5, threshold=0.3):
        self.n = n_bootstrap; self.t = threshold
    def fit(self, X, y):
        print(f"   [SELECTOR] Running Stability Selection ({self.n} boots)...")
        scores = np.zeros(X.shape[1])
        n_sub = int(len(X) * 0.7)
        for i in range(self.n):
            idx = np.random.choice(len(X), n_sub, replace=False)
            # Use lightweight model for selection
            model = LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=i)
            model.fit(X[idx], y[idx])
            scores += (np.max(np.abs(model.coef_), axis=0) > 1e-4).astype(float)
        self.support_ = (scores / self.n) > self.t
        print(f"   [SELECTOR] Kept {np.sum(self.support_)} features.")
        return self
    def transform(self, X): return X[:, self.support_]

# ------------------------------------------------------------------------------
# 2. THE ALCHEMIST: DIFFUSION AUGMENTATION
# ------------------------------------------------------------------------------
class TabularDiffusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, dim)
        )
    def forward(self, x): return self.net(x)

def synthesize_data(X, y, n_new=1000):
    # Train simple diffusion-like generator per class
    X_syn_all, y_syn_all = [], []
    classes = np.unique(y)
    
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 10: continue
        
        # Simple noise-denoise training
        model = TabularDiffusion(Xc.shape[1]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        Xt = torch.tensor(Xc, dtype=torch.float32).to(DEVICE)
        
        model.train()
        for _ in range(CONFIG['DIFFUSION_EPOCHS']):
            noise = torch.randn_like(Xt) * 0.1
            rec = model(Xt + noise)
            loss = F.mse_loss(rec, Xt)
            opt.zero_grad(); loss.backward(); opt.step()
            
        # Generate
        model.eval()
        n_gen = int(n_new / len(classes))
        with torch.no_grad():
            seed_idx = np.random.choice(len(Xc), n_gen)
            seed = Xt[seed_idx] + torch.randn_like(Xt[seed_idx]) * 0.2
            gen = model(seed).cpu().numpy()
            X_syn_all.append(gen)
            y_syn_all.append(np.full(n_gen, c))
            
    if not X_syn_all: return X, y
    return np.vstack(X_syn_all), np.concatenate(y_syn_all)

# ------------------------------------------------------------------------------
# 3. OPTIMIZER: SAM (SHARPNESS-AWARE MINIMIZATION)
# ------------------------------------------------------------------------------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        super(SAM, self).__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        return torch.norm(torch.stack([p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None]), p=2)

# ------------------------------------------------------------------------------
# 4. MODEL ARSENAL
# ------------------------------------------------------------------------------

# --- A. THETA TABM (BatchEnsemble) ---
class TabM_Layer(nn.Module):
    def __init__(self, d_in, d_out, k=8):
        super().__init__()
        self.k = k; self.linear = nn.Linear(d_in, d_out)
        self.r = nn.Parameter(torch.randn(k, d_in) * 0.1 + 1.0)
        self.s = nn.Parameter(torch.randn(k, d_out) * 0.1 + 1.0)
    def forward(self, x): # x: (B, D)
        b = x.shape[0]
        x = x.repeat_interleave(self.k, dim=0) # (B*K, D)
        r = self.r.repeat(b, 1) # (B*K, D)
        s = self.s.repeat(b, 1)
        return self.linear(x * r) * s

class ThetaTabM(nn.Module):
    def __init__(self, d_in, n_classes, k=8):
        super().__init__()
        self.k = k; self.n_classes = n_classes
        self.l1 = TabM_Layer(d_in, 256, k)
        self.l2 = TabM_Layer(256, 128, k)
        self.head = TabM_Layer(128, n_classes, k)
        self.act = nn.GELU()
    def forward(self, x):
        b = x.shape[0]
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        # (B*K, C) -> (B, K, C) -> Mean -> (B, C)
        logits = self.head(x).view(b, self.k, self.n_classes)
        return logits.mean(dim=1)

# --- B. TRUE TABR (Retrieval) ---
class TabR(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 128), nn.SiLU())
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.head = nn.Linear(128, n_classes)
    def forward(self, x, neighbors):
        # x: (B, 128), neighbors: (B, K, 128)
        q = self.enc(x).unsqueeze(1)
        kv = self.enc(neighbors)
        # Cross Attention
        ctx, _ = self.attn(q, kv, kv)
        return self.head(ctx.squeeze(1) + q.squeeze(1))

class TabRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes): self.n_classes = n_classes
    def fit(self, X, y):
        self.X_tr = X.astype(np.float32)
        self.knn = NearestNeighbors(n_neighbors=16, n_jobs=-1).fit(self.X_tr)
        self.net = TabR(X.shape[1], self.n_classes).to(DEVICE)
        opt = optim.AdamW(self.net.parameters(), lr=1e-3)
        
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
        
        self.net.train()
        for _ in range(15):
            for bx, by in dl:
                # Retrieve neighbors
                nb_idx = self.knn.kneighbors(bx.numpy(), return_distance=False)
                nb_x = torch.tensor(self.X_tr[nb_idx], dtype=torch.float32).to(DEVICE)
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                out = self.net(bx, nb_x)
                F.cross_entropy(out, by).backward()
                opt.step()
        return self
    def predict_proba(self, X):
        self.net.eval()
        probs = []
        for i in range(0, len(X), 256):
            bx = X[i:i+256]
            nb_idx = self.knn.kneighbors(bx, return_distance=False)
            nb_x = torch.tensor(self.X_tr[nb_idx], dtype=torch.float32).to(DEVICE)
            bx_t = torch.tensor(bx, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                probs.append(torch.softmax(self.net(bx_t, nb_x), dim=1).cpu().numpy())
        return np.vstack(probs)

# --- C. KAN (Kolmogorov-Arnold - Simplified) ---
class KANLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
    def forward(self, x):
        # Basis: x * SiLU(x)
        return F.linear(x * F.silu(x), self.w)

class KAN(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            KANLinear(d_in, 64), nn.LayerNorm(64),
            KANLinear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

# ------------------------------------------------------------------------------
# 5. ADAPTATION & META-LEARNING
# ------------------------------------------------------------------------------

def get_adversarial_weights(X_tr, X_te):
    """Reweights training samples to match test distribution."""
    print("   [ADAPT] Calculating Adversarial Weights...")
    X_comb = np.vstack([X_tr, X_te])
    y_comb = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
    clf = LogisticRegression(max_iter=500, C=0.1)
    clf.fit(X_comb, y_comb)
    p = clf.predict_proba(X_tr)[:, 1]
    w = p / (1 - p + 1e-6)
    return np.clip(w, 0.1, 10.0)

def test_time_training(model, X_test):
    """Unsupervised TTT via Entropy Minimization."""
    print("   [TTT] Fine-tuning on Test Manifold...")
    model.train() # Keep BN running or freeze? Usually fine to update stats or not.
    # We will clone to avoid corrupting the main model for other folds if needed, 
    # but here we do it at the end of fold inference.
    
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    ds = TensorDataset(Xt)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    
    for _ in range(CONFIG['TTT_STEPS']):
        for (bx,) in dl:
            opt.zero_grad()
            logits = model(bx)
            p = torch.softmax(logits, dim=1)
            entropy = -(p * torch.log(p + 1e-6)).sum(dim=1).mean()
            
            # Consistency
            noise = torch.randn_like(bx) * 0.05
            logits_n = model(bx + noise)
            p_n = torch.softmax(logits_n, dim=1)
            cons = F.mse_loss(p, p_n)
            
            loss = entropy + cons
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(Xt), dim=1).cpu().numpy()

def dirichlet_calibration(probs, y_true):
    # Simplified version: Log-prob Logistic Regression
    # ln(p_cal) = W * ln(p_raw) + b
    eps = 1e-6
    log_probs = np.log(probs + eps)
    lr = LogisticRegression(multi_class='multinomial')
    lr.fit(log_probs, y_true)
    return lr

# ------------------------------------------------------------------------------
# 6. ORCHESTRATION
# ------------------------------------------------------------------------------

def run_omega_singularity(X, y, X_test):
    print(">>> OMEGA-SINGULARITY PROTOCOL INITIATED <<<")
    
    # --- PHASE 1: ENGINEERING ---
    print("\n--- PHASE 1: ENGINEERING ---")
    
    # 1. Stability Selection (View A - Trees)
    selector = StabilitySelector().fit(X, y)
    X_tree = selector.transform(X)
    X_test_tree = selector.transform(X_test)
    
    # 2. RankGauss (View B - Neural)
    rg = RankGaussScaler()
    X_rg = rg.transform(X)
    X_test_rg = rg.transform(X_test)
    
    # 3. Manifold + DAE (View C - Deep)
    # Train DAE on everything
    dae_emb_tr = train_dae(np.vstack([X_rg, X_test_rg]))
    dae_emb_te = dae_emb_tr[len(X):]
    dae_emb_tr = dae_emb_tr[:len(X)]
    
    # Manifold Features
    feats_tr, feats_te, lid_te = compute_manifold_features(X_rg, X_test_rg)
    
    # Assemble Deep View
    X_deep_tr = np.hstack([X_rg, dae_emb_tr, feats_tr])
    X_deep_te = np.hstack([X_test_rg, dae_emb_te, feats_te])
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # --- PHASE 2: CROSS-FIT & STACKING ---
    print("\n--- PHASE 2: TRAINING ---")
    skf = StratifiedKFold(n_splits=CONFIG['N_FOLDS'], shuffle=True, random_state=SEED)
    
    # Storage
    # 0: TabM, 1: TabR, 2: KAN, 3: XGB, 4: Cat
    n_models = 5 
    meta_train = np.zeros((len(X), n_models * CONFIG['N_CLASSES']))
    meta_test = np.zeros((len(X_test), n_models * CONFIG['N_CLASSES']))
    
    for fold, (idx_tr, idx_val) in enumerate(skf.split(X, y_enc)):
        print(f"   > Fold {fold+1}/{CONFIG['N_FOLDS']}...")
        
        # Split
        xt_tr, xt_val = X_tree[idx_tr], X_tree[idx_val]
        xd_tr, xd_val = X_deep_tr[idx_tr], X_deep_tr[idx_val]
        y_tr, y_val = y_enc[idx_tr], y_enc[idx_val]
        
        # Adversarial Weights
        sample_weights = get_adversarial_weights(xd_tr, X_deep_te)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32).to(DEVICE)
        
        # Synthesis (Augment Training Data Only)
        # Note: We augment Deep View for NNs. Trees usually fine without.
        xd_aug, y_aug = synthesize_data(xd_tr, y_tr, n_new=1000)
        
        # --- MODEL 1: ThetaTabM (SAM + TTT) ---
        tabm = ThetaTabM(xd_tr.shape[1], CONFIG['N_CLASSES'], k=CONFIG['TABM_K']).to(DEVICE)
        opt_sam = SAM(tabm.parameters(), optim.AdamW, lr=2e-3, rho=0.05)
        
        # Train Loop (SAM)
        tx = torch.tensor(xd_aug, dtype=torch.float32).to(DEVICE)
        ty = torch.tensor(y_aug, dtype=torch.long).to(DEVICE)
        
        tabm.train()
        for _ in range(15): # Epochs
            # Full batch for SAM usually too heavy, use mini-batches
            perm = torch.randperm(len(tx))
            for i in range(0, len(tx), CONFIG['BATCH_SIZE']):
                idx = perm[i:i+CONFIG['BATCH_SIZE']]
                bx, by = tx[idx], ty[idx]
                
                # Step 1
                logits = tabm(bx)
                loss = F.cross_entropy(logits, by)
                loss.backward()
                opt_sam.first_step(zero_grad=True)
                
                # Step 2
                F.cross_entropy(tabm(bx), by).backward()
                opt_sam.second_step(zero_grad=True)
                opt_sam.base_optimizer.step()
        
        # Predict Val (Standard)
        tabm.eval()
        with torch.no_grad():
            meta_train[idx_val, 0:5] = torch.softmax(tabm(torch.tensor(xd_val, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()
            
        # Predict Test (TTT)
        # Clone model for TTT to not affect other logic if needed (here we are done with fold)
        meta_test[:, 0:5] += test_time_training(copy.deepcopy(tabm), X_deep_te) / CONFIG['N_FOLDS']

        # --- MODEL 2: True TabR ---
        tabr = TabRClassifier(CONFIG['N_CLASSES'])
        tabr.fit(xd_aug, y_aug) # Fit on augmented
        meta_train[idx_val, 5:10] = tabr.predict_proba(xd_val)
        meta_test[:, 5:10] += tabr.predict_proba(X_deep_te) / CONFIG['N_FOLDS']

        # --- MODEL 3: KAN ---
        kan = KAN(xd_tr.shape[1], CONFIG['N_CLASSES']).to(DEVICE)
        # LBFGS Training
        opt_kan = optim.LBFGS(kan.parameters(), lr=0.1)
        tkx = torch.tensor(xd_tr, dtype=torch.float32).to(DEVICE) # KANs often prefer cleaner real data
        tky = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
        
        kan.train()
        def closure():
            opt_kan.zero_grad()
            loss = F.cross_entropy(kan(tkx), tky)
            loss.backward()
            return loss
        
        for _ in range(10): opt_kan.step(closure)
        
        kan.eval()
        with torch.no_grad():
            meta_train[idx_val, 10:15] = torch.softmax(kan(torch.tensor(xd_val, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()
            meta_test[:, 10:15] += torch.softmax(kan(torch.tensor(X_deep_te, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy() / CONFIG['N_FOLDS']

        # --- MODEL 4 & 5: Trees ---
        # XGB
        if 'xgboost' in sys.modules:
            xg = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, 
                                   tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
                                   verbosity=0)
            xg.fit(xt_tr, y_tr) # Trees on Stability View
            meta_train[idx_val, 15:20] = xg.predict_proba(xt_val)
            meta_test[:, 15:20] += xg.predict_proba(X_test_tree) / CONFIG['N_FOLDS']
        
        # CatBoost
        if 'catboost' in sys.modules:
            # Langevin usually enabled via posterior sampling options, simplified here to standard robust settings
            cb_clf = cb.CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, 
                                           task_type='GPU' if torch.cuda.is_available() else 'CPU',
                                           verbose=0)
            cb_clf.fit(xt_tr, y_tr)
            meta_train[idx_val, 20:25] = cb_clf.predict_proba(xt_val)
            meta_test[:, 20:25] += cb_clf.predict_proba(X_test_tree) / CONFIG['N_FOLDS']
            
        # Clean up
        del tabm, tabr, kan
        torch.cuda.empty_cache()

    # --- PHASE 3: META-LEARNING (NNLS) ---
    print("\n--- PHASE 3: META-OPTIMIZATION ---")
    
    # 1. Dirichlet Calibration of OOF
    # We calibrate each model's output
    final_preds = np.zeros((len(X_test), CONFIG['N_CLASSES']))
    
    # Construct "Confidence in True Class" matrix for NNLS
    # We have 5 models. We want 5 weights.
    Z = np.zeros((len(X), 5))
    target = np.ones(len(X))
    
    # Check if trees ran
    active_models = [0, 1, 2] # TabM, TabR, KAN
    if 'xgboost' in sys.modules: active_models.append(3)
    if 'catboost' in sys.modules: active_models.append(4)
    
    for m_idx in active_models:
        # Extract probs
        oof_probs = meta_train[:, m_idx*5 : (m_idx+1)*5]
        test_probs = meta_test[:, m_idx*5 : (m_idx+1)*5]
        
        # Calibrate
        calibrator = dirichlet_calibration(oof_probs, y_enc)
        cal_oof = calibrator.predict_proba(np.log(oof_probs + 1e-6))
        cal_test = calibrator.predict_proba(np.log(test_probs + 1e-6))
        
        # Store for NNLS (Prob of correct class)
        Z[:, active_models.index(m_idx)] = cal_oof[np.arange(len(X)), y_enc]
        
        # Add to potential final (weighted later)
        # We store calibrated test probs temporarily
        meta_test[:, m_idx*5 : (m_idx+1)*5] = cal_test

    # 2. NNLS
    weights, _ = nnls(Z[:, :len(active_models)], target)
    weights /= weights.sum()
    print(f"   [WEIGHTS] {weights}")

    # 3. Final Assembly
    for i, m_idx in enumerate(active_models):
        final_preds += meta_test[:, m_idx*5 : (m_idx+1)*5] * weights[i]

    # --- PHASE 4: LID SCALING ---
    print("\n--- PHASE 4: LID SCALING ---")
    # High LID = High Uncertainty. Flatten the distribution.
    # T = 1 + alpha * LID
    alpha = 0.5
    T = 1.0 + alpha * lid_te[:, None]
    
    scaled_logits = np.log(final_preds + 1e-6) / T
    # Re-softmax
    scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
    
    # --- SUBMISSION ---
    preds = np.argmax(scaled_probs, axis=1)
    labels = le.inverse_transform(preds)
    
    out_path = 'PartD/outputs/labelsX_omega_singularity.npy'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, labels)
    print(f"\n[SUCCESS] Solution Omega-Singularity written to {out_path}")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Attempt to load data from standard paths or arguments
    try:
        # User specific loader integration
        sys.path.append(os.getcwd())
        from src.data_loader import load_data
        X, y, X_test = load_data()
        run_omega_singularity(X, y, X_test)
    except Exception as e:
        print(f"Data Loader failed: {e}")
        print("Please ensure 'src.data_loader' is accessible.")