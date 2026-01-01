"""
ZETA-EPSILON PROTOCOL: THE QUANTUM BUILD (FINAL v2)
--------------------------------------------------------------------------------
The "Combined Arms" Doctrine + "Data Alchemy" Refinement + "Infantry Audit".
Integrates True TabR, ThetaTabM (SAM), Transductive DAE (Refinery), and Topology MixUp.
Enforces Zero-Trust constraints on Tree Models (Depth Limits, Early Stopping).

COMMANDER: Research Commander (Reflexion Core)
EXECUTOR: Antigravity Agent
DATE: 2026

ARSENAL:
1. FUEL (Zeta): 
   - Feature Engineering: [LID, PageRank].
   - Streams: Refined (Neural) vs Raw (Trees).
2. AIR FORCE (Neural): True TabR (Attn), ThetaTabM (Mamba-Proxy + SAM).
3. INFANTRY (Trees): 
   - XGBoost (Depth=6, Global View).
   - CatBoost (Depth=8, Early Stopping, Local View).
4. SPECIAL FORCES: Test-Time Training (TTT), LID Scaling.
5. ENGINE: Infinity Loop (Separate Raw/Refined Streams).
"""

import os
import sys
import copy
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import Legacy Anchors
try:
    from src.models import get_catboost_model, get_xgb_model
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from models import get_catboost_model, get_xgb_model

# CONFIGURATION
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

# ==============================================================================
# SECTION 1: THE FUEL (Data Alchemy - "Zeta")
# ==============================================================================

class TransductiveDAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, input_dim))
    
    def forward(self, x): return self.decoder(self.encoder(x))
    def get_embedding(self, x): return self.encoder(x)

class DataRefinery:
    def __init__(self, input_dim):
        self.dae = TransductiveDAE(input_dim).to(DEVICE)
    
    def fit(self, X_all, epochs=50):
        print("[REFINERY] Training Transductive DAE on Manifold...")
        X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        
        opt = optim.AdamW(self.dae.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        
        self.dae.train()
        for ep in range(epochs):
            for (xb,) in dl:
                noise = torch.randn_like(xb) * 0.1
                rec = self.dae(xb + noise)
                loss = crit(rec, xb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def get_embedding(self, X):
        self.dae.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            emb = self.dae.encoder(X_t).cpu().numpy()
        return emb

    def get_reconstruction(self, X):
        self.dae.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            rec = self.dae(X_t).cpu().numpy()
        return rec

def compute_manifold_features(X_train, X_test):
    print("[MANIFOLD] Engineering Topological Features...")
    X_all = np.vstack([X_train, X_test])
    
    # 1. KNN Graph
    k = 20
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_all)
    dists, indices = nbrs.kneighbors(X_all)
    
    # 2. LID (Local Intrinsic Dimensionality)
    d_k = dists[:, -1]
    d_j = dists[:, 1:]
    lid = k / np.sum(np.log(d_k[:, None] / (d_j + 1e-10) + 1e-10), axis=1)
    lid = (lid - lid.min()) / (lid.max() - lid.min()) # Normalize 0-1
    
    # 3. PageRank (Centrality)
    try:
        import networkx as nx
        print("  > Computing PageRank (Graph Centrality)...")
        # Build adjacency matrix (sparse)
        A = kneighbors_graph(X_all, k, mode='connectivity', include_self=False)
        G = nx.from_scipy_sparse_array(A)
        pr = nx.pagerank(G, alpha=0.85)
        pagerank = np.array([pr[i] for i in range(len(X_all))])
        pagerank = (pagerank - pagerank.min()) / (pagerank.max() - pagerank.min())
    except ImportError:
        print("  [WARN] NetworkX not found. Skipping PageRank.")
        pagerank = np.zeros(len(X_all))
        
    feats = np.column_stack([lid, pagerank])
    return feats[:len(X_train)], feats[len(X_train):]

# --- TOPOLOGY MIXUP ---
class TopologyMixUpLoader:
    def __init__(self, X, y, batch_size=256):
        self.X = X # numpy
        self.y = y # numpy
        self.batch_size = batch_size
        self.knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X)
        
    def __iter__(self):
        idxs = np.random.permutation(len(self.X))
        for i in range(0, len(self.X), self.batch_size):
            batch_idxs = idxs[i:i+self.batch_size]
            X_b = self.X[batch_idxs]
            y_b = self.y[batch_idxs]
            
            # Find neighbors for mixup
            batch_mn_idxs = self.knn.kneighbors(X_b, return_distance=False)
            rand_n = np.random.randint(1, 5, size=len(X_b))
            target_idxs = batch_mn_idxs[np.arange(len(X_b)), rand_n]
            
            X_target = self.X[target_idxs]
            # Assumes hard labels, mixing inputs only for regularization
            
            lam = np.random.beta(0.4, 0.4, size=(len(X_b), 1))
            lam = np.maximum(lam, 1-lam)
            
            X_mix = lam * X_b + (1 - lam) * X_target
            yield torch.tensor(X_mix, dtype=torch.float32).to(DEVICE), torch.tensor(y_b, dtype=torch.long).to(DEVICE)

# ==============================================================================
# SECTION 2: THE ENGINE (Epsilon Arsenal)
# ==============================================================================

# --- SAM OPTIMIZER ---
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
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
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
        return torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)

    def step(self, closure=None):
        raise NotImplementedError("SAM requires manual fit loop.")

# --- TRUE TABR ---
class TabRModule(nn.Module):
    def __init__(self, input_dim, num_classes, context_size=96):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, context_size))
        self.q_proj = nn.Linear(context_size, context_size)
        self.k_proj = nn.Linear(context_size, context_size)
        self.v_proj = nn.Linear(context_size, context_size)
        self.head = nn.Sequential(nn.Linear(context_size, 64), nn.SiLU(), nn.Linear(64, num_classes))
        self.scale = context_size ** -0.5

    def forward(self, x, neighbors):
        q = self.encoder(x).unsqueeze(1)    # [B, 1, C]
        B, K, D = neighbors.shape
        kv = self.encoder(neighbors.view(B*K, D)).view(B, K, -1) # [B, K, C]
        
        Q = self.q_proj(q); K_ = self.k_proj(kv); V_ = self.v_proj(kv)
        scores = torch.bmm(Q, K_.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, V_).squeeze(1)
        return self.head(context + q.squeeze(1))

class TrueTabRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, n_neighbors=16):
        self.num_classes, self.n_neighbors = num_classes, n_neighbors
        self.model, self.knn, self.X_train_ = None, None, None

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=np.float32)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(self.X_train_)
        
        self.model = TabRModule(X.shape[1], self.num_classes).to(DEVICE)
        opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        
        ds = TensorDataset(torch.tensor(self.X_train_), torch.tensor(y, dtype=torch.long))
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        
        self.model.train()
        for ep in range(15):
            for xb, yb in dl:
                xb_np = xb.cpu().numpy()
                n_idxs = self.knn.kneighbors(xb_np, return_distance=False)
                n_feats = torch.tensor(self.X_train_[n_idxs]).to(DEVICE)
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                
                opt.zero_grad()
                loss = crit(self.model(xb, n_feats), yb)
                loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        # Process in batches to avoid OOM with neighbors
        probs = []
        for i in range(0, len(X), 256):
            xb_chunk = X[i:i+256]
            n_idxs = self.knn.kneighbors(xb_chunk, return_distance=False)
            n_feats = torch.tensor(self.X_train_[n_idxs]).to(DEVICE)
            
            xb_t = torch.tensor(xb_chunk, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = self.model(xb_t, n_feats)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

# --- THETA TABM (w/ SAM & MIXUP) ---
class ThetaTabMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, num_classes)
        ).to(DEVICE)

    def fit(self, X, y):
        loader = TopologyMixUpLoader(X, y)
        opt = SAM(self.model.parameters(), optim.AdamW, lr=1e-3, rho=0.05)
        crit = nn.CrossEntropyLoss()
        
        self.model.train()
        for ep in range(20):
            for xb, yb in loader:
                opt.zero_grad(); crit(self.model(xb), yb).backward(); opt.first_step(zero_grad=True)
                crit(self.model(xb), yb).backward(); opt.second_step(zero_grad=True); opt.base_optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            p = []
            for i in range(0, len(X), 1024):
                bx = torch.tensor(X[i:i+1024], dtype=torch.float32).to(DEVICE)
                p.append(torch.softmax(self.model(bx), dim=1).cpu().numpy())
        return np.vstack(p)
    
    def test_time_training_single(self, x_sample, y_pseudo, steps=1):
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=0.005)
        xt = torch.tensor(x_sample[np.newaxis, :], dtype=torch.float32).to(DEVICE)
        yt = torch.tensor([y_pseudo], dtype=torch.long).to(DEVICE)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(self.model(xt), yt).backward()
            opt.step()
        self.model.eval()

# ==============================================================================
# SECTION 3: THE EXECUTION LOGIC (Zeta-Epsilon)
# ==============================================================================

def load_data_safe():
    try:
        from src.data_loader import load_data
        return load_data()
    except ImportError: return None, None, None

def main():
    print(">>> INITIATING ZETA-EPSILON PROTOCOL (v2: Infantry Audit) <<<")
    
    # 1. LOAD DATA
    X, y, X_test = load_data_safe()
    if X is None:
        print("[ERR] Data not found. Exiting.")
        return

    le = LabelEncoder(); y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # 2. FEATURE ENGINEERING
    # Base: Quantile
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    X_gauss = qt.fit_transform(X); X_test_gauss = qt.transform(X_test)
    
    # Data Refinery (DAE)
    refinery = DataRefinery(X_gauss.shape[1]).fit(np.vstack([X_gauss, X_test_gauss]), epochs=30)
    
    # Manifold Features (LID + PageRank)
    feats_tr, feats_te = compute_manifold_features(X_gauss, X_test_gauss)
    
    # --- CONSTRUCT STREAMS ---
    # Stream A: Neural (Refined) -> [Gauss, LID, PageRank, DAE_Emb]
    dae_emb_tr = refinery.get_embedding(X_gauss)
    dae_emb_te = refinery.get_embedding(X_test_gauss)
    X_neural_tr = np.hstack([X_gauss, feats_tr, dae_emb_tr])
    X_neural_te = np.hstack([X_test_gauss, feats_te, dae_emb_te])
    
    # Stream B: Trees (Raw fidelity) -> [Raw(here Gauss), LID, PageRank, DAE_Rec]
    # Trees handle raw data well, but DAE reconstruction cleans noise.
    # We use Reconstruction ("Refined feature space") as scalar features.
    dae_rec_tr = refinery.get_reconstruction(X_gauss)
    dae_rec_te = refinery.get_reconstruction(X_test_gauss)
    # Note: Using Gauss as "Raw" base because it's already normalized well.
    X_tree_tr = np.hstack([X_gauss, feats_tr, dae_rec_tr])
    X_tree_te = np.hstack([X_test_gauss, feats_te, dae_rec_te])
    
    # 3. DEPLOY ARSENAL & AUDIT INFANTRY
    models = {}
    print("\n[DEPLOY] Air Force (Neural)...")
    models['ThetaTabM'] = ThetaTabMClassifier(X_neural_tr.shape[1], num_classes)
    models['TrueTabR'] = TrueTabRClassifier(num_classes)
    
    # Infantry Audit
    print("[DEPLOY] Infantry (Trees) with Zero-Trust Constraints...")
    
    # XGBoost: Force Depth 6 (Global)
    xgb_base = get_xgb_model()
    # We can't easily change params of a constructed XGB obj without set_params if it works, 
    # OR we just rely on its kwargs. `get_xgb_model` sets depth=9. 
    # Let's override:
    xgb_base.set_params(max_depth=6) 
    models['XGBoost'] = xgb_base
    
    # CatBoost: Early Stopping (Local)
    cat_base = get_catboost_model()
    # Depth 8 is default in get_catboost_model, verify or keep. keeping 8.
    models['CatBoost'] = cat_base
    
    try:
        from tabpfn import TabPFNClassifier
        models['TabPFN'] = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', N_ensemble_configurations=32)
        print("[DEPLOY] Foundation (TabPFN)...")
    except: pass
    
    # 4. INFINITY LOOP
    probs_map = {}
    
    # Training Loop
    for name, m in models.items():
        print(f"  > Training {name}...")
        
        if name in ['ThetaTabM', 'TrueTabR']:
            m.fit(X_neural_tr, y_enc)
            probs_map[name] = m.predict_proba(X_neural_te)
            
        elif name == 'CatBoost':
            # Create Validation Split for Early Stopping
            # Note: 10% of 8k samples is small (800), safe for validation.
            Xt_tr, Xt_val, yt_tr, yt_val = train_test_split(X_tree_tr, y_enc, test_size=0.1, random_state=SEED, stratify=y_enc)
            m.fit(Xt_tr, yt_tr, eval_set=(Xt_val, yt_val), early_stopping_rounds=50, verbose=False)
            probs_map[name] = m.predict_proba(X_tree_te)
            
        elif name == 'XGBoost':
            # Standard fit (Depth constrained already)
            m.fit(X_tree_tr, y_enc)
            probs_map[name] = m.predict_proba(X_tree_te)
            
        elif name == 'TabPFN':
            # Fallback for TabPFN features
            if X_gauss.shape[1] > 100:
                 from sklearn.decomposition import PCA
                 pca = PCA(n_components=100)
                 p_tr = pca.fit_transform(X_gauss)
                 p_te = pca.transform(X_test_gauss)
                 m.fit(p_tr, y_enc)
                 probs_map[name] = m.predict_proba(p_te)
            else:
                 m.fit(X_gauss, y_enc)
                 probs_map[name] = m.predict_proba(X_test_gauss)
    
    # TTT (Test Time Training) on Silver Samples
    print("\n[TTT] Scanning for Silver Samples...")
    p_tabm = probs_map['ThetaTabM']
    conf = np.max(p_tabm, axis=1)
    silver_idxs = np.where((conf > 0.7) & (conf < 0.95))[0]
    print(f"  > Found {len(silver_idxs)} candidates. Applying TTT (Limit 20)...")
    
    for i in silver_idxs[:20]:
        pseudo_y = np.argmax(p_tabm[i])
        models['ThetaTabM'].test_time_training_single(X_neural_te[i], pseudo_y)
        probs_map['ThetaTabM'][i] = models['ThetaTabM'].predict_proba(X_neural_te[i:i+1])[0]
        
    # Consensus & LID Scaling
    # Use LID as Temperature (High LID = High Noise = High Temp = Soft Prob)
    # feats_te column 0 is normalized LID
    lid_norm = feats_te[:, 0]
    T = 1.0 + 0.2 * lid_norm # Range 1.0 - 1.2
    
    avg_probs = np.zeros_like(p_tabm)
    for name, p in probs_map.items():
        ps = np.power(p, 1.0 / T[:, None])
        ps /= ps.sum(axis=1, keepdims=True)
        avg_probs += ps
    
    final_preds = np.argmax(avg_probs, axis=1)
    labels = le.inverse_transform(final_preds)
    
    os.makedirs('PartD/outputs', exist_ok=True)
    np.save('PartD/outputs/labelsX_zeta_epsilon.npy', labels)
    print("\n>>> ZETA-EPSILON DEPLOYMENT SUCCESSFUL <<<")

if __name__ == "__main__":
    main()
