import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import QuantileTransformer
from sklearn.random_projection import GaussianRandomProjection
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .domain import coral_align
from scipy.special import erfinv
from sklearn.linear_model import LogisticRegression

class RankGaussScaler:
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.array(X)
        # Αριθμητικό rank gauss
        # Για να αποφύγουμε NaN στα όρια, χρησιμοποιούμε (rank+1)/(N+2) ή παρόμοιο
        # Αλλά το erfinv απαιτεί (-1, 1). 
        # Το SKLearn QuantileTransformer το κάνει ήδη robustly.
        # Αλλά για "Purist" RankGauss:
        ns = X.shape[0]
        res = []
        for i in range(X.shape[1]):
            r = np.argsort(np.argsort(X[:, i]))
            r = (r + 1) / (ns + 1)
            # Κόψιμο για να αποφύγουμε infinity
            r = 2 * r - 1
            r = np.clip(r, -0.99999, 0.99999)
            res.append(erfinv(r))
        return np.vstack(res).T

class StabilitySelector:
    """Ανθεκτική επιλογή χαρακτηριστικών μέσω Randomized Lasso."""
    def __init__(self, n_bootstrap=5, threshold=0.3):
        self.n = n_bootstrap; self.t = threshold
        self.support_ = None
    
    def fit(self, X, y):
        print(f"   [SELECTOR] Running Stability Selection ({self.n} boots)...")
        scores = np.zeros(X.shape[1])
        n_sub = int(len(X) * 0.7)
        for i in range(self.n):
            idx = np.random.choice(len(X), n_sub, replace=False)
            model = LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=i)
            model.fit(X[idx], y[idx])
            scores += (np.max(np.abs(model.coef_), axis=0) > 1e-4).astype(float)
        self.support_ = (scores / self.n) > self.t
        print(f"   [SELECTOR] Kept {np.sum(self.support_)} features.")
        return self
    
    def transform(self, X): 
        if self.support_ is None: return X
        return X[:, self.support_]


def apply_feature_view(X_train, X_test, view, seed, allow_transductive=False):
    view = (view or 'raw').strip().lower()
    if view == 'raw':
        X_tr, X_te = X_train, X_test
        if config.ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=config.CORAL_REG)
        return X_tr, X_te

    if view == 'quantile':
        qt = QuantileTransformer(output_distribution='normal', random_state=seed)
        X_tr, X_te = qt.fit_transform(X_train), qt.transform(X_test)
        if config.ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=config.CORAL_REG)
        return X_tr, X_te

    if view.startswith('pca'):
        n_components = min(50, X_train.shape[1], max(2, X_train.shape[0] - 1))
        pca = PCA(n_components=n_components, random_state=seed)
        X_tr, X_te = pca.fit_transform(X_train), pca.transform(X_test)
        if config.ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=config.CORAL_REG)
        return X_tr, X_te

    if view.startswith('ica'):
        n_components = min(50, X_train.shape[1], max(2, X_train.shape[0] - 1))
        ica = FastICA(n_components=n_components, random_state=seed, max_iter=500)
        X_tr, X_te = ica.fit_transform(X_train), ica.transform(X_test)
        if config.ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=config.CORAL_REG)
        return X_tr, X_te

    if view.startswith('rp') or view.startswith('random'):
        n_components = min(50, X_train.shape[1])
        rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
        X_tr, X_te = rp.fit_transform(X_train), rp.transform(X_test)
        if config.ENABLE_CORAL:
            if not allow_transductive:
                raise ValueError("ENABLE_CORAL=1 requires ALLOW_TRANSDUCTIVE=1 (it uses test features for alignment).")
            X_tr, X_te = coral_align(X_tr, X_te, reg=config.CORAL_REG)
        return X_tr, X_te

    if view.startswith('spectral'):
        if not allow_transductive:
            raise ValueError(
                "Spectral view requires transductive embedding; set ALLOW_TRANSDUCTIVE=1 or remove 'spectral' from VIEWS."
            )
        X_all = np.vstack([X_train, X_test])
        n_components = min(30, X_all.shape[0] - 1)
        se = SpectralEmbedding(n_components=n_components, random_state=seed)
        Z = se.fit_transform(X_all)
        return Z[: len(X_train)], Z[len(X_train) :]

    raise ValueError(
        f"Unknown view '{view}'. Supported: raw, quantile, pca, ica, rp/random, spectral(transductive)"
    )


class TransductiveDAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.SiLU(), nn.Linear(512, 128))
        self.decoder = nn.Sequential(nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DataRefinery:
    def __init__(self, input_dim):
        self.dae = TransductiveDAE(input_dim).to(config.DEVICE)

    def fit(self, X_all, epochs=None, noise_std=None):
        if epochs is None:
            epochs = config.DAE_EPOCHS
        if noise_std is None:
            noise_std = config.DAE_NOISE_STD
        X_t = torch.tensor(X_all, dtype=torch.float32).to(config.DEVICE)
        dl = DataLoader(TensorDataset(X_t), batch_size=config.BATCH_SIZE, shuffle=True)
        opt = optim.AdamW(self.dae.parameters(), lr=config.LR_SCALE)
        crit = nn.MSELoss()

        self.dae.train()
        for _ in range(int(epochs)):
            for (xb,) in dl:
                noise = torch.randn_like(xb) * float(noise_std)
                rec = self.dae(xb + noise)
                loss = crit(rec, xb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def transform(self, X):
        """Return (embedding, reconstruction) in one pass per batch."""
        self.dae.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(config.DEVICE)
        emb, rec = [], []
        for i in range(0, len(X), config.BATCH_SIZE):
            xb = X_t[i:i+config.BATCH_SIZE]
            with torch.no_grad():
                z = self.dae.encoder(xb)
                r = self.dae.decoder(z)
            emb.append(z.cpu().numpy())
            rec.append(r.cpu().numpy())
        return np.vstack(emb), np.vstack(rec)


def compute_manifold_features(X_train, X_test, allow_transductive=False, k=None, enable_pagerank=None, return_lid=False):
    if k is None:
        k = config.MANIFOLD_K
    if enable_pagerank is None:
        enable_pagerank = config.ENABLE_PAGERANK

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


def build_streams(X_v, X_test_v):
    ref_fit_X = np.vstack([X_v, X_test_v]) if config.ALLOW_TRANSDUCTIVE else X_v
    ref = DataRefinery(X_v.shape[1]).fit(ref_fit_X)

    feats_tr, feats_te, lid_tr, lid_te = compute_manifold_features(
        X_v,
        X_test_v,
        allow_transductive=config.ALLOW_TRANSDUCTIVE,
        k=config.MANIFOLD_K,
        enable_pagerank=config.ENABLE_PAGERANK,
        return_lid=True,
    )

    emb_tr, rec_tr = ref.transform(X_v)
    emb_te, rec_te = ref.transform(X_test_v)

    X_neural_tr = np.hstack([X_v, feats_tr, emb_tr])
    X_neural_te = np.hstack([X_test_v, feats_te, emb_te])
    X_tree_tr = np.hstack([X_v, feats_tr, rec_tr])
    X_tree_te = np.hstack([X_test_v, feats_te, rec_te])
    return X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te
