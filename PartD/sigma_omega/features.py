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
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.model_selection import StratifiedKFold


def gpu_laplacian_eigenmaps(X, n_components=8, k=20, device=None):
    """
    GPU-accelerated Laplacian Eigenmaps using torch.lobpcg.
    
    Algorithm:
    1. Build k-NN adjacency graph (CPU, sklearn)
    2. Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    3. Use torch.lobpcg(largest=False) for smallest eigenvectors
    
    Args:
        X: Input data (n_samples, n_features)
        n_components: Number of embedding dimensions
        k: Number of neighbors for graph construction
        device: Torch device (defaults to config.DEVICE)
    
    Returns:
        embedding: (n_samples, n_components) numpy array
    """
    if device is None:
        device = config.DEVICE
    
    n = X.shape[0]
    k_eff = min(k, n - 1)
    
    # 1. Build symmetric k-NN adjacency (CPU)
    A_sparse = kneighbors_graph(X, k_eff, mode='connectivity', include_self=False)
    A_sparse = (A_sparse + A_sparse.T) / 2  # Symmetrize
    A_sparse = A_sparse.tocoo()
    
    # 2. Convert to torch sparse tensor
    indices = torch.tensor(np.vstack([A_sparse.row, A_sparse.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_sparse.data, dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    
    # 3. Compute degree and normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    
    # D^{-1/2} A D^{-1/2} as sparse matrix
    row, col = A.indices()
    scaled_vals = A.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    A_norm = torch.sparse_coo_tensor(A.indices(), scaled_vals, (n, n)).coalesce()
    
    # L = I - A_norm (construct as linear operator for lobpcg)
    def laplacian_mv(v):
        # L @ v = v - A_norm @ v
        return v - torch.sparse.mm(A_norm, v)
    
    # 4. Use torch.lobpcg for smallest eigenvectors
    #    Request n_components+1 to skip the trivial all-ones eigenvector
    n_req = min(n_components + 1, n - 1)
    X0 = torch.randn(n, n_req, device=device, dtype=torch.float32)
    
    # Create identity-like sparse for lobpcg (it needs a matrix, not a function)
    # Build L explicitly as sparse: L = I - A_norm
    I_indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    I_values = torch.ones(n, device=device, dtype=torch.float32)
    I_sparse = torch.sparse_coo_tensor(I_indices, I_values, (n, n)).coalesce()
    
    L = I_sparse - A_norm
    L = L.coalesce()
    
    # torch.lobpcg expects dense or sparse, largest=False for smallest eigenvalues
    eigenvalues, eigenvectors = torch.lobpcg(L.to_dense(), k=n_req, largest=False, niter=100)
    
    # Skip the first (near-zero) eigenvalue, take next n_components
    embedding = eigenvectors[:, 1:n_components + 1].cpu().numpy()
    
    # Handle case where we got fewer than requested
    if embedding.shape[1] < n_components:
        padding = np.zeros((n, n_components - embedding.shape[1]))
        embedding = np.hstack([embedding, padding])
    
    return embedding


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

class KMeansFeaturizer:
    """
    Generates cluster-distance features.
    Trees love these as they provide 'global' location context.
    """
    def __init__(self, n_clusters=16, seed=42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans = None
        
    def fit(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init='auto')
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        if self.kmeans is None: return X
        return self.kmeans.transform(X).astype(np.float32)


class PLSFeatureGen:
    """
    Supervised dimensionality reduction. Finds directions of maximum covariance with y.
    Great for guiding Neural Networks towards the target signal.
    """
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = None

    def fit(self, X, y):
        # PLS requires y. If y is regression, ok. If classification, one-hot?
        # Standard PLS works for regression. For classification, one-hot Y is typical.
        # Here y is encoded integer. We can treat as regression (ordinal) or one-hot.
        # Treating as regression is a strong heuristic for ordinal classes or binary.
        # For multi-class, let's use one-hot internal logic if needed, but PLSRegression works with multi-output Y.
        
        # Check if y is multi-class
        n_classes = len(np.unique(y))
        if n_classes > 2:
            # One-hot encode y for PLS2
            y_ohe = np.zeros((len(y), n_classes))
            y_ohe[np.arange(len(y)), y] = 1
            y_target = y_ohe
        else:
            y_target = y
            
        self.pls = PLSRegression(n_components=self.n_components, scale=True)
        try:
            self.pls.fit(X, y_target)
        except Exception as e:
            print(f"   [PLS] Failed to fit: {e}")
            self.pls = None
        return self

    def transform(self, X):
        if self.pls is None: return np.zeros((len(X), 0))
        return self.pls.transform(X)


class GoldenFeatureGenerator:
    """
    Generates arithmetic interactions (sum, diff, mul, div) for feature pairs.
    Keeps only top N features most correlated with target.
    """
    def __init__(self, top_n=20):
        self.top_n = top_n
        self.ops = [] # List of (idx1, idx2, op_name)
    
    def fit(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)
        is_multiclass = len(classes) > 2
        
        # Helper for One-vs-Rest Max Correlation
        def get_score(f_vec, y_vec):
            # Check constant
            if np.std(f_vec) < 1e-9: 
                return 0.0
            
            if not is_multiclass:
                # Binary: Simple correlation
                return abs(np.corrcoef(f_vec, y_vec)[0, 1])
            else:
                # Multiclass: Max correlation with any class binary vector
                # This logic captures "Feature F is strong for Class C"
                scores_c = []
                for c in classes:
                    y_c = (y_vec == c).astype(float)
                    # Check constant target (unlikely unless 1 class)
                    if np.std(y_c) < 1e-9: continue 
                    corr = abs(np.corrcoef(f_vec, y_c)[0, 1])
                    scores_c.append(corr)
                return max(scores_c) if scores_c else 0.0

        print(f"   [GOLDEN] Mining interactions from {n_features} features (OVR-Corr)...")
        
        scores = []
        limit = min(n_features, 50)
        
        # Precompute individual feature OVR to prioritize?
        # Let's iterate pairs.
        
        for i in range(limit):
            for j in range(i+1, limit):
                f1, f2 = X[:, i], X[:, j]
                
                # Sum
                f_sum = f1 + f2
                scores.append((get_score(f_sum, y), i, j, 'sum'))
                
                # Diff
                f_diff = f1 - f2
                scores.append((get_score(f_diff, y), i, j, 'diff'))
                
                # Mul
                f_mul = f1 * f2
                scores.append((get_score(f_mul, y), i, j, 'mul'))
                
                # Div
                # Protect div by zero
                f_div = f1 / (f2 + 1e-6)
                scores.append((get_score(f_div, y), i, j, 'div'))

        scores.sort(key=lambda x: x[0], reverse=True)
        self.ops = scores[:self.top_n]
        if self.ops:
            print(f"   [GOLDEN] Found {len(self.ops)} golden interactions. Top OVR score: {self.ops[0][0]:.4f}")
        return self

    def transform(self, X):
        if not self.ops: return np.zeros((len(X), 0))
        
        out = []
        for _, i, j, op in self.ops:
            f1, f2 = X[:, i], X[:, j]
            if op == 'sum': res = f1 + f2
            elif op == 'diff': res = f1 - f2
            elif op == 'mul': res = f1 * f2
            elif op == 'div': res = f1 / (f2 + 1e-6)
            out.append(res)
        
        return np.column_stack(out).astype(np.float32)


class GeometricFeatureGenerator:
    """
    Calculates distances (Cosine, Euclidean) to Class Centroids.
    Explicitizes geometry for Tree models.
    """
    def __init__(self):
        self.centroids = {} # class_label -> mean_vector
        self.classes = []

    def fit(self, X, y):
        # Compute centroids on Train Only
        self.classes = np.unique(y)
        for c in self.classes:
            self.centroids[c] = np.mean(X[y == c], axis=0)
        return self

    def transform(self, X):
        if not self.centroids: return np.zeros((len(X), 0))
        
        feats = []
        # Centroids matrix
        centers = np.vstack([self.centroids[c] for c in self.classes])
        
        # 1. Cosine Distance
        d_cos = cdist(X, centers, metric='cosine')
        feats.append(d_cos)
        
        # 2. Euclidean Distance
        d_euc = cdist(X, centers, metric='euclidean')
        feats.append(d_euc)
        
        return np.hstack(feats).astype(np.float32)


class AnomalyFeatureGenerator:
    """
    pathology-driven features: Row statistics and Outlier counts.
    Helps models identify 'hard' or 'garbage' samples.
    """
    def __init__(self, sigma=3.0):
        self.sigma = sigma
        self.means = None
        self.stds = None
        
    def fit(self, X, y=None):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0) + 1e-9
        return self
        
    def transform(self, X):
        if self.means is None: return X
        
        # 1. Row Statistics
        row_mean = np.mean(X, axis=1, keepdims=True)
        row_std = np.std(X, axis=1, keepdims=True)
        # kurtosis is expensive/noisy on small N_feat, skip for speed
        
        # 2. Outlier Count (Values > 3 sigma from col mean)
        z_score = np.abs((X - self.means) / self.stds)
        outliers = (z_score > self.sigma).sum(axis=1, keepdims=True)
        
        # 3. High-Magnitude Mean (Mean of top 10% values)
        # Sort each row, take mean of top k
        k = max(1, int(X.shape[1] * 0.1))
        # partitioning is faster than full sort
        # np.partition moves top k to right
        top_k = np.partition(X, -k, axis=1)[:, -k:]
        mag_mean = np.mean(top_k, axis=1, keepdims=True)
        
        return np.hstack([row_mean, row_std, outliers, mag_mean]).astype(np.float32)


class CrossFoldFeatureGenerator:
    """
    Prevents leakage by generating features using K-Fold OOF.
    Wraps another generator (e.g. PLS, Golden).
    """
    def __init__(self, generator_cls, n_folds=5, **kwargs):
        self.generator_cls = generator_cls
        self.kwargs = kwargs
        self.n_folds = n_folds
        self.generators = [] # List of (fold_gen, fold_idx)
        self.full_generator = None

    def fit(self, X, y):
        self.full_generator = self.generator_cls(**self.kwargs).fit(X, y)
        return self

    def transform(self, X, y=None, mode='train'):
        """
        mode='train': Generates OOF features (requires y).
        mode='test': Uses full_generator logic.
        """
        if mode == 'test':
             return self.full_generator.transform(X)
             
        # Train Mode: Cross-Fold Generation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # OOF Matrix
        n_samples = len(X)
        # Determine output dim by running on a small sample
        dummy_gen = self.generator_cls(**self.kwargs).fit(X[:10], y[:10])
        n_feats = dummy_gen.transform(X[:2]).shape[1]
        
        oof_feats = np.zeros((n_samples, n_feats), dtype=np.float32)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            gen = self.generator_cls(**self.kwargs).fit(X_tr, y_tr)
            oof_feats[val_idx] = gen.transform(X_val)
            
        return oof_feats


class AdversarialDriftRemover:
    """
    Prunes features that drift significantly between Train and Test.
    Train RF(X) -> {0:Train, 1:Test}. Drop if AUC > 0.7.
    """
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.keep_mask = None
        
    def fit(self, X_train, X_test):
        # Create adversarial dataset
        n_tr = len(X_train)
        n_te = len(X_test)
        
        # Downsample majority for balanced classification
        min_n = min(n_tr, n_te)
        idx_tr = np.random.choice(n_tr, min_n, replace=False)
        idx_te = np.random.choice(n_te, min_n, replace=False)
        
        X_adv = np.vstack([X_train[idx_tr], X_test[idx_te]])
        y_adv = np.concatenate([np.zeros(min_n), np.ones(min_n)])
        
        # Train simple RF
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_adv, y_adv)
        
        # Get feature importances (proxy for drift contribution)
        # Actually better: Use univariate AUC for each feature.
        # RF importance is multivariate.
        # Let's use ROC AUC for each feature individually.
        
        n_features = X_train.shape[1]
        keep = []
        
        from sklearn.metrics import roc_auc_score
        
        print(f"   [DRIFT] Checking {n_features} features for drift (Threshold AUC > {self.threshold})...")
        for i in range(n_features):
            score = roc_auc_score(y_adv, X_adv[:, i])
            # AUC around 0.5 is good. AUC > 0.7 or < 0.3 is bad drift.
            auc = max(score, 1 - score)
            if auc < self.threshold:
                keep.append(i)
                
        self.keep_mask = np.array(keep)
        print(f"   [DRIFT] Dropping {n_features - len(keep)} drifting features.")
        return self
        
    def transform(self, X):
        if self.keep_mask is None or len(self.keep_mask) == 0:
            return X
        return X[:, self.keep_mask]


class ConfusionScout:
    """
    TRAINS 'SCOUT' MODELS TO SEPARATE CONFUSED CLASS PAIRS.
    For each pair (A, B) in `pairs`, trains a binary classifier.
    Returns probability of Class B for each sample.
    """
    def __init__(self, pairs=None):
        self.pairs = pairs or [] # List of tuples (class_a, class_b)
        self.scouts = [] # List of (pair, model)
        
    def fit(self, X, y):
        if not self.pairs:
            return self
            
        self.scouts = []
        for (a, b) in self.pairs:
            # Filter for A vs B
            mask = (y == a) | (y == b)
            X_sub = X[mask]
            y_sub = y[mask]
            
            # Binary Target: 0=A, 1=B
            y_bin = (y_sub == b).astype(int)
            
            if len(y_bin) < 20 or len(np.unique(y_bin)) < 2:
                continue
                
            model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200, random_state=42)
            model.fit(X_sub, y_bin)
            self.scouts.append(((a,b), model))
            
        return self
        
    def transform(self, X):
        if not self.scouts:
            return np.zeros((len(X), 0))
            
        feats = []
        for (a, b), model in self.scouts:
            # Predict Prob(B)
            p = model.predict_proba(X)[:, 1:]
            feats.append(p)
            
        return np.hstack(feats).astype(np.float32)


def find_confusion_pairs(X, y, top_k=3, seed=42):
    """
    Train a quick global proxy to identify top confusion pairs.
    """
    from sklearn.metrics import confusion_matrix
    
    # Lightweight proxy
    model = LogisticRegression(C=1.0, max_iter=100, random_state=seed)
    model.fit(X, y)
    preds = model.predict(X)
    
    cm = confusion_matrix(y, preds)
    np.fill_diagonal(cm, 0) # Ignore correct predictions
    
    # Find top pairs
    pairs = []
    # Get indices of top_k max values
    # We want distinct pairs (A,B). Note CM is not symmetric (A->B != B->A).
    # We care about misclassification in general.
    # Symmetry: (A,B) and (B,A) are same "boundary".
    # Let's sum symmetric elements? cm + cm.T
    cm_sym = cm + cm.T
    # Taking upper triangle to avoid duplicates
    cm_sym = np.triu(cm_sym)
    
    flat_indices = np.argsort(cm_sym.ravel())[::-1]
    
    for idx in flat_indices:
        if len(pairs) >= top_k: break
        
        row, col = np.unravel_index(idx, cm_sym.shape)
        if cm_sym[row, col] == 0: break # No more errors
        
        # Determine actual class labels
        # Assuming y classes are 0..N-1 matching indices. 
        # Safer: use model.classes_
        c1, c2 = model.classes_[row], model.classes_[col]
        pairs.append((c1, c2))
        
    print(f"   [SCOUT] Identified Top Confusion Pairs: {pairs}")
    return pairs


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


class TabularDiffusion(nn.Module):
    """
    Απλό Diffusion-like μοντέλο για tabular data augmentation.
    Εκπαιδεύεται να αποθορυβοποιεί samples και στη συνέχεια
    χρησιμοποιείται για να δημιουργήσει συνθετικά δεδομένα.
    """
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        return self.net(x)


def synthesize_data(X, y, n_new=1000, epochs=None, device=None):
    """
    Δημιουργεί συνθετικά training samples ανά κλάση μέσω Diffusion.
    
    Args:
        X: Training features (numpy array)
        y: Training labels (numpy array, integers)
        n_new: Συνολικός αριθμός νέων samples
        epochs: Epochs εκπαίδευσης (default: config.DIFFUSION_EPOCHS)
        device: Torch device
    
    Returns:
        X_syn, y_syn: Συνθετικά features και labels
    """
    if epochs is None:
        epochs = config.DIFFUSION_EPOCHS
    if device is None:
        device = config.DEVICE
    
    classes = np.unique(y)
    X_syn_all, y_syn_all = [], []
    
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 10:
            continue  # Skip αν δεν υπάρχουν αρκετά samples
        
        # Train denoising diffusion
        model = TabularDiffusion(Xc.shape[1]).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        Xt = torch.tensor(Xc, dtype=torch.float32).to(device)
        
        model.train()
        for _ in range(int(epochs)):
            noise = torch.randn_like(Xt) * 0.1
            rec = model(Xt + noise)
            loss = nn.functional.mse_loss(rec, Xt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Generate synthetic samples
        model.eval()
        n_gen = max(1, int(n_new / len(classes)))
        with torch.no_grad():
            seed_idx = np.random.choice(len(Xc), n_gen)
            seed = Xt[seed_idx] + torch.randn(n_gen, Xc.shape[1], device=device) * 0.2
            gen = model(seed).cpu().numpy()
            X_syn_all.append(gen)
            y_syn_all.append(np.full(n_gen, c))
    
    if not X_syn_all:
        return np.zeros((0, X.shape[1])), np.zeros(0, dtype=y.dtype)
    
    return np.vstack(X_syn_all), np.concatenate(y_syn_all)


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

        # Laplacian Eigenmaps (geometry unfolding)
        if config.ENABLE_LAPLACIAN:
            n_components = min(8, len(X_all) - 2)
            laplacian = None
            
            # Try GPU path first
            if config.USE_GPU_EIGENMAPS and config.DEVICE.type == 'cuda':
                try:
                    laplacian = gpu_laplacian_eigenmaps(X_all, n_components, k, config.DEVICE)
                    print(f"   [MANIFOLD] GPU Laplacian Eigenmaps: {n_components} components")
                except Exception as e:
                    print(f"   [MANIFOLD] GPU Eigenmaps failed: {e}, falling back to CPU")
            
            # Fallback to sklearn
            if laplacian is None:
                try:
                    se = SpectralEmbedding(n_components=n_components, n_neighbors=k, n_jobs=-1, random_state=42)
                    laplacian = se.fit_transform(X_all)
                    print(f"   [MANIFOLD] CPU Laplacian Eigenmaps: {n_components} components")
                except Exception as e:
                    print(f"   [MANIFOLD] Laplacian failed: {e}")
                    laplacian = np.zeros((len(X_all), 8))
        else:
            laplacian = np.zeros((len(X_all), 0))  # Empty if disabled

        # Combine all manifold features
        if laplacian.shape[1] > 0:
            feats = np.column_stack([lid, pagerank, laplacian])
        else:
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


def build_streams(X_v, X_test_v, y_train=None):
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

    # [OMEGA] K-Means Features
    km = KMeansFeaturizer(n_clusters=32, seed=42).fit(X_v)
    km_tr = km.transform(X_v)
    km_te = km.transform(X_test_v)

    # [OMEGA] Supervised Features (PLS + Golden)
    pls_tr, pls_te = np.zeros((len(X_v), 0)), np.zeros((len(X_test_v), 0))
    gold_tr, gold_te = np.zeros((len(X_v), 0)), np.zeros((len(X_test_v), 0))
    geo_tr, geo_te = np.zeros((len(X_v), 0)), np.zeros((len(X_test_v), 0))
    anom_tr, anom_te = np.zeros((len(X_v), 0)), np.zeros((len(X_test_v), 0))
    conf_tr, conf_te = np.zeros((len(X_v), 0)), np.zeros((len(X_test_v), 0)) # Confusion Features

    if y_train is not None:
        # [OMEGA] Cross-Fold Feature Generation (PLS + Golden) for LEAKAGE PREVENTION
        # Neural Stream PLS
        pls_gen = CrossFoldFeatureGenerator(PLSFeatureGen, n_components=10, n_folds=5).fit(X_v, y_train)
        pls_tr = pls_gen.transform(X_v, y_train, mode='train')
        pls_te = pls_gen.transform(X_test_v, mode='test')

        # Tree Stream Golden
        gold_gen = CrossFoldFeatureGenerator(GoldenFeatureGenerator, top_n=20, n_folds=5).fit(X_v, y_train)
        gold_tr = gold_gen.transform(X_v, y_train, mode='train')
        gold_te = gold_gen.transform(X_test_v, mode='test')
        
        # [OMEGA] Confusion Scout (Auto-Detected Pairs)
        # 1. Detect pairs globally on View Train data
        pairs = find_confusion_pairs(X_v, y_train, top_k=3)
        
        # 2. Train Scouts using Cross-Fold generation (Leakage Proof)
        scout_gen = CrossFoldFeatureGenerator(ConfusionScout, pairs=pairs, n_folds=5).fit(X_v, y_train)
        conf_tr = scout_gen.transform(X_v, y_train, mode='train')
        conf_te = scout_gen.transform(X_test_v, mode='test')

        # [OMEGA] Geometric Features (Centroids)
        geo = GeometricFeatureGenerator().fit(X_v, y_train)
        geo_tr = geo.transform(X_v)
        geo_te = geo.transform(X_test_v)
        
        # [OMEGA] Anomaly Features (Pathology)
        anom = AnomalyFeatureGenerator().fit(X_v)
        anom_tr = anom.transform(X_v)
        anom_te = anom.transform(X_test_v)

    X_neural_tr = np.hstack([X_v, feats_tr, emb_tr, pls_tr, geo_tr, anom_tr, conf_tr])
    X_neural_te = np.hstack([X_test_v, feats_te, emb_te, pls_te, geo_te, anom_te, conf_te])
    
    # Trees get: View + Manifold + DAE_Embedding + KMeans + Golden + Geometric + Anomaly + ConfusionScout
    # Replaced 'rec' with 'emb' (Embeddings are richer)
    X_tree_tr = np.hstack([X_v, feats_tr, emb_tr, km_tr, gold_tr, geo_tr, anom_tr, conf_tr])
    X_tree_te = np.hstack([X_test_v, feats_te, emb_te, km_te, gold_te, geo_te, anom_te, conf_te])
    
    return X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te
