
"""
ΥΛΟΠΟΙΗΣΗ ΠΡΟΗΓΜΕΝΟΥ ΣΥΝΟΛΙΚΟΥ ΜΟΝΤΕΛΟΥ (ENSEMBLE) ΤΑΞΙΝΟΜΗΣΗΣ
--------------------------------------------------------------------------------
Ολοκληρωμένο Σύστημα Επιβλεπόμενης και Μη Επιβλεπόμενης Μάθησης
(Integrated Architecture: Denoising Autoencoder + Manifold Aware Inference)
Συγγραφέας: Ευάγγελος Μόσχου
Ημερομηνία: 2025

ΑΝΤΙΚΕΙΜΕΝΟ ΚΑΙ ΣΤΟΧΟΣ:
    Η επίτευξη μέγιστης ακρίβειας ταξινόμησης στο σύνολο δεδομένων DatasetTV 
    μέσω της εφαρμογής σύγχρονων αρχιτεκτονικών βαθιάς μάθησης και μεθόδων 
    βελτιστοποίησης συμπερασμάτων.

ΑΡΧΙΤΕΚΤΟΝΙΚΗ ΩΜΕΓΑ:
    1. Μεταγωγικός Αυτοκωδικοποιητής (Tabular DAE):
       - Εφαρμογή Αυτο-επιβλεπόμενης Μεταγωγικής Μάθησης (Transductive Learning).
       - Εξαγωγή στιβαρών αναπαραστάσεων (Embeddings) για τα νευρωνικά δίκτυα.
       
    2. Σύστημα Σταθεροποίησης Συμπερασμάτων (Manifold TTA):
       - Επαύξηση δεδομένων κατά τον χρόνο ελέγχου (Test-Time Augmentation).
       - Στάθμιση βάσει τοπολογικής συνέπειας (Gaussian Weights).
       
    3. Ανταγωνιστική Επαλήθευση (Adversarial Validation): Στάθμιση δειγμάτων βάσει Drift.
    4. Μοντέλα Βάσης: TabM (Mamba), KAN (Kolmogorov-Arnold), TurboTabR (CatBoost).
    5. Αναδρομική Συναίνεση: Εξόρυξη δειγμάτων υψηλής εμπιστοσύνης (Diamond Mining).

ΕΞΑΡΤΗΣΕΙΣ:
    - torch, numpy, pandas, scikit-learn, catboost
    - (Προαιρετικά) mambular, kan, tabpfn (Περιλαμβάνονται αυτόματα Fallbacks)
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier

# ------------------------------------------------------------------------------
# ΡΥΘΜΙΣΕΙΣ
# ------------------------------------------------------------------------------
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
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)
print(f"\n[INIT] Device: {DEVICE}")
print("[INIT] Initializing Omega Protocol...")

# ------------------------------------------------------------------------------
# 1. TABULAR DAE (Ο ΣΤΡΟΒΙΛΟΣΥΜΠΙΕΣΤΗΣ)
# ------------------------------------------------------------------------------
class TabularDAE(nn.Module):
    """
    Αυτοτελής Denoising Autoencoder.
    Μαθαίνει τη δομή της πολλαπλότητας (manifold structure) ΟΛΟΚΛΗΡΟΥ του συνόλου δεδομένων (Train + Test).
    """
    def __init__(self, input_dim, hidden_dim=256, bottleneck_dim=64, noise_factor=0.1):
        super(TabularDAE, self).__init__()
        self.noise_factor = noise_factor
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.SiLU() 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

class DAE_Embedder:
    def __init__(self, input_dim, device=DEVICE):
        self.device = device
        self.model = TabularDAE(input_dim).to(device)
        self.trained = False
        
    def fit(self, X_all, epochs=30, batch_size=256):
        """
        X_all: vstack(X_train, X_test) - Μεταγωγική Μάθηση (Transductive Learning)
        """
        print(f"\n[DAE] Training Turbocharger on {X_all.shape} samples...")
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for ep in range(epochs):
            total_loss = 0
            for batch in loader:
                x_clean = batch[0]
                # Εισαγωγή Gaussian Θορύβου
                noise = torch.randn_like(x_clean) * self.model.noise_factor
                x_noisy = x_clean + noise
                
                optimizer.zero_grad()
                recon = self.model(x_noisy)
                loss = criterion(recon, x_clean)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (ep+1) % 10 == 0:
                print(f"  > Epoch {ep+1}: Loss {total_loss/len(loader):.5f}")
                
        self.trained = True
        return self
        
    def transform(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.model.get_embedding(X_t).cpu().numpy()

# ------------------------------------------------------------------------------
# 2. WEIGHTED MANIFOLD TTA (Ο ΣΤΑΘΕΡΟΠΟΙΗΤΗΣ)
# ------------------------------------------------------------------------------
def predict_proba_tta(model, X, knn_graph, neighbor_dists, alpha=0.3):
    """
    P_final = (1-alpha)*P_model + alpha * WeightedAvg(Neighbors)
    Βάρη = exp(-dist^2)
    """
    # 1. Βασική Πρόβλεψη
    p_base = model.predict_proba(X)
    
    # 2. Πρόβλεψη Γειτόνων
    # knn_graph: (N, k) δείκτες
    # neighbor_dists: (N, k) αποστάσεις
    
    # Υπολογισμός Gaussian βαρών: σχήμα (N, k)
    # Το Sigma βασίζεται στη μέση απόσταση από τον 10ο γείτονα ή είναι σταθερό; Σταθερό για στιβαρότητα.
    # Θέλουμε οι πιο κοντινοί γείτονες να έχουν πολύ μεγαλύτερη σημασία.
    sigma = 1.0 # Υποθέτοντας τυποποιημένη ή λογική κλίμακα. Εάν οι αποστάσεις είναι μεγάλες, κανονικοποιήστε.
    weights = np.exp(- (neighbor_dists ** 2) / (2 * sigma ** 2))
    
    # Κανονικοποίηση βαρών ανά γραμμή
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    
    # Συλλογή πιθανοτήτων γειτόνων: (N, k, C)
    # Αργός βρόχος για ασφάλεια, μπορεί να διανυσματιστεί με προχωρημένη ευρετηρίαση
    N, k = knn_graph.shape
    C = p_base.shape[1]
    
    p_neigh_weighted = np.zeros((N, C))
    
    # Διανυσματική συλλογή
    # Flat δείκτες: knn_graph.flatten()
    # flat_probs = p_base[knn_graph.flatten()] -> (N*k, C)
    # Ανασχηματισμός -> (N, k, C)
    # Αλλά η p_base πρέπει να είναι διαθέσιμη για ΟΛΑ τα δείγματα. 
    # Εάν το X είναι Test, οι γείτονες μπορεί να βρίσκονται στο Test (self) ή στο Train; 
    # Το TTA συνήθως υποδηλώνει γείτονες στο ΙΔΙΟ σύνολο (συνέπεια πολλαπλότητας).
    # Υποθέτουμε ότι οι δείκτες του knn_graph αναφέρονται στο ίδιο το X.
    
    flat_indices = knn_graph.flatten()
    flat_probs = p_base[flat_indices].reshape(N, k, C)
    
    # Σταθμισμένο Άθροισμα: sum_k ( w_ik * p_ik )
    # βάρη: (N, k) -> (N, k, 1)
    p_smooth = (flat_probs * weights[:, :, np.newaxis]).sum(axis=1)
    
    return (1 - alpha) * p_base + alpha * p_smooth

# ------------------------------------------------------------------------------
# 3. ΒΟΗΘΗΤΙΚΑ: DRIFT & ΤΟΠΟΛΟΓΙΑ
# ------------------------------------------------------------------------------
class AdversarialWeigher:
    def fit_transform(self, X_train, X_test):
        print("\n[ADVERSARIAL] Checking for Covariate Shift...")
        X_drift = np.vstack([X_train, X_test])
        y_drift = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=SEED, n_jobs=-1)
        probs = cross_val_predict(clf, X_drift, y_drift, cv=5, method='predict_proba')[:, 1]
        
        auc = roc_auc_score(y_drift, probs)
        print(f"  > Drift AUC: {auc:.4f}")
        
        train_probs = probs[:len(X_train)]
        weights = train_probs / (1 - train_probs + 1e-6)
        weights = np.clip(weights, 0.1, 10.0)
        weights /= weights.mean()
        return weights

class ManifoldEngineer:
    def transform(self, X_train, X_test):
        print("\n[TOPOLOGY] Engineering Manifold Features...")
        X_all = np.vstack([X_train, X_test])
        
        # PageRank & LID
        knn = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(X_all)
        dists, indices = knn.kneighbors(X_all)
        
        # LID
        k = 20
        d_k = dists[:, -1].reshape(-1, 1)
        d_j = dists[:, 1:] 
        lid_vals = (k) / (np.sum(np.log(d_k / (d_j + 1e-10) + 1e-10), axis=1) + 1e-10)
        
        # PageRank (Προσέγγιση μέσω βαθμού εάν το networkx αποτύχει)
        try:
            import networkx as nx
            A = kneighbors_graph(X_all, n_neighbors=15, mode='distance', include_self=False, n_jobs=-1)
            G = nx.from_scipy_sparse_array(A)
            pr = nx.pagerank(G, alpha=0.85)
            pr_vals = np.array([pr[i] for i in range(len(X_all))])
        except:
            pr_vals = lid_vals # Εφεδρικό (Fallback)
            
        scaler = StandardScaler()
        feats = np.vstack([
            scaler.fit_transform(pr_vals.reshape(-1, 1)).flatten(),
            scaler.fit_transform(lid_vals.reshape(-1, 1)).flatten()
        ]).T
        
        print(f"  > Added {feats.shape[1]} Topological Features.")
        
        X_train_new = np.hstack([X_train, feats[:len(X_train)]])
        X_test_new = np.hstack([X_test, feats[len(X_train):]])
        
        # Επιστροφή γραφήματος KNN για χρήση στο TTA αργότερα
        # Οι δείκτες στο X_test αναφέρονται σε γραμμές του X_test εάν προσαρμόσουμε στο X_test, 
        # Αλλά προσαρμόζουμε στο X_all. 
        # Για το TTA, χρειαζόμαστε γείτονες του X_test μέσα στο X_test.
        knn_test = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(X_test) # k=6 (εαυτός + 5)
        dists_test, idxs_test = knn_test.kneighbors(X_test)
        
        return X_train_new, X_test_new, idxs_test, dists_test

# ------------------------------------------------------------------------------
# 4. ΑΡΧΙΤΕΚΤΟΝΙΚΕΣ
# ------------------------------------------------------------------------------
class TabMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes, d_model=128, n_layers=4):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.model = None
        self.device = DEVICE
        self.use_proxy = False

    def fit(self, X, y, sample_weight=None):
        print(f"\n[TabM] Initializing (Input Dim: {self.input_dim})...")
        try:
            from mambular.models import MambularClassifier
            self.model = MambularClassifier(d_model=self.d_model, n_layers=self.n_layers, lr=1e-3, cat_feature_indices=[])
            self.model.fit(X, y)
            print("[TabM] Training Real Mamba.")
            self.use_proxy = False
        except:
            print("[TabM] Using LSTM Proxy.")
            self.model = self._build_proxy().to(self.device)
            self._train_proxy(X, y, sample_weight)
            self.use_proxy = True
        return self

    def _build_proxy(self):
        return nn.Sequential(
             nn.Linear(self.input_dim, self.d_model), nn.LayerNorm(self.d_model), nn.GELU(),
             *[nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.LayerNorm(self.d_model), nn.GELU(), nn.Dropout(0.1)) for _ in range(self.n_layers)],
             nn.Linear(self.d_model, self.num_classes)
        )

    def _train_proxy(self, X, y, w, epochs=20):
        opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss(reduction='none')
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.long).to(self.device)
        wt = torch.tensor(w, dtype=torch.float32).to(self.device) if w is not None else torch.ones(len(X)).to(self.device)
        dl = DataLoader(TensorDataset(Xt, yt, wt), batch_size=256, shuffle=True)
        self.model.train()
        for ep in range(epochs):
            for xb, yb, wb in dl:
                opt.zero_grad()
                loss = (crit(self.model(xb), yb) * wb).mean()
                loss.backward()
                opt.step()

    def predict_proba(self, X):
        if not self.use_proxy: return self.model.predict_proba(X)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(torch.tensor(X, dtype=torch.float32).to(self.device)), dim=1).cpu().numpy()

class KANClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.model = None
        self.device = DEVICE
        self.use_proxy = False

    def fit(self, X, y, sample_weight=None):
        print(f"\n[KAN] Initializing (Input Dim: {self.input_dim})...")
        try:
            from kan import KAN
            self.model = KAN(width=[self.input_dim, self.hidden_dim, self.num_classes], device=self.device)
            dataset = {'train_input': torch.tensor(X, dtype=torch.float32).to(self.device), 'train_label': torch.tensor(y, dtype=torch.long).to(self.device)}
            self.model.train(dataset, opt="LBFGS", steps=20)
            print("[KAN] Training Real KAN.")
            self.use_proxy = False
        except:
            print("[KAN] Using FastKAN Proxy.")
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim), nn.SiLU(), 
                nn.Linear(self.hidden_dim, self.num_classes) # Πολύ απλοποιημένο FastKAN
            ).to(self.device)
            self._train_proxy(X, y, sample_weight)
            self.use_proxy = True
        return self

    def _train_proxy(self, X, y, w, epochs=25):
        optimizer = optim.LBFGS(self.model.parameters(), lr=1e-1)
        criterion = nn.CrossEntropyLoss(reduction='none')
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.long).to(self.device)
        wt = torch.tensor(w, dtype=torch.float32).to(self.device) if w is not None else torch.ones(len(X)).to(self.device)
        def closure():
            optimizer.zero_grad()
            loss = (criterion(self.model(Xt), yt) * wt).mean()
            loss.backward()
            return loss
        self.model.train()
        for _ in range(epochs): optimizer.step(closure)

    def predict_proba(self, X):
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        if not self.use_proxy:
             with torch.no_grad(): return torch.softmax(self.model(Xt), dim=1).cpu().numpy()
        self.model.eval()
        with torch.no_grad(): return torch.softmax(self.model(Xt), dim=1).cpu().numpy()

class TurboTabRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k_neighbors=24, n_estimators=800):
        self.k_neighbors = k_neighbors
        self.n_estimators = n_estimators
    def fit(self, X, y, sample_weight=None):
        self.X_ref = X.copy()
        self.y_ref = y.copy()
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=-1).fit(X)
        self.clf = CatBoostClassifier(iterations=self.n_estimators, depth=8, l2_leaf_reg=5, learning_rate=0.03, loss_function='MultiClass', verbose=False, allow_writing_files=False, task_type="GPU" if torch.cuda.is_available() else "CPU")
        self.clf.fit(X, y, sample_weight=sample_weight) # Χρήση ακατέργαστου X + Τοπολογία, παράλειψη Jitter για ταχύτητα/σταθερότητα στο Omega
        # Σημείωση: Το TabR συνήθως χρειάζεται Jitter, αλλά τώρα έχουμε ισχυρά χαρακτηριστικά DAE/Τοπολογίας.
        return self
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class HyperTabPFNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, device='cuda', n_ensemble=32):
        self.device = device
        self.n_ensemble = n_ensemble
        self.model = None
    def fit(self, X, y, sample_weight=None):
        try:
            from tabpfn import TabPFNClassifier
            self.model = TabPFNClassifier(device=self.device)
            if hasattr(self.model, 'N_ensemble_configurations'): self.model.N_ensemble_configurations = self.n_ensemble
            if len(X) > 10000:
                 idx = np.random.choice(len(X), 10000, replace=False)
                 self.model.fit(X[idx], y[idx])
            else: self.model.fit(X, y)
        except: self.model = None
        return self
    def predict_proba(self, X):
        if self.model is None: return np.ones((len(X), 1))
        return self.model.predict_proba(X)

# ------------------------------------------------------------------------------
# 5. INFINITY LOOP (ΑΠΕΙΡΟΣ ΒΡΟΧΟΣ)
# ------------------------------------------------------------------------------
def omega_loop(X, y, X_test):
    # 1. Μηχανική Χαρακτηριστικών (Feature Engineering)
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    X_gauss = qt.fit_transform(X)
    X_test_gauss = qt.transform(X_test)
    
    eng = ManifoldEngineer()
    X_topo_raw, X_test_topo_raw, tta_knn_indices, tta_knn_dists = eng.transform(X, X_test)
    
    weigher = AdversarialWeigher()
    weights = weigher.fit_transform(X, X_test)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(np.unique(y_enc))

    # 2. DAE Turbocharger (Στροβιλοσυμπιεστής)
    dae = DAE_Embedder(input_dim=X_gauss.shape[1])
    # Εκπαίδευση σε ΟΛΑ (Μεταγωγική Μάθηση)
    X_all_gauss = np.vstack([X_gauss, X_test_gauss])
    dae.fit(X_all_gauss)
    
    # Λήψη των Embeddings
    emb_train = dae.transform(X_gauss)
    emb_test = dae.transform(X_test_gauss)
    
    # Συνένωση για τα Νευρωνικά Δίκτυα
    print(f"[OMEGA] Concatenating DAE Embeddings ({emb_train.shape[1]}) to NNs...")
    X_nn_train = np.hstack([X_gauss, emb_train])
    X_nn_test = np.hstack([X_test_gauss, emb_test])
    
    # Τα Δένδρα παραμένουν με χαρακτηριστικά Τοπολογίας (Όχι DAE για αποφυγή κατάρρευσης συσχέτισης)
    X_tree_train = X_topo_raw
    X_tree_test = X_test_topo_raw

    # 3. Αρχικοποίηση Μοντέλων
    models = {
        'TabM': TabMClassifier(input_dim=X_nn_train.shape[1], num_classes=num_classes),
        'KAN': KANClassifier(input_dim=X_nn_train.shape[1], num_classes=num_classes),
        'TurboTabR': TurboTabRClassifier(k_neighbors=24),
        'HyperTabPFN': HyperTabPFNClassifier(device=str(DEVICE))
    }
    
    # 4. Εκπαίδευση & TTA Inference (Εξαγωγή Συμπερασμάτων)
    test_preds = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name in ['TabM', 'KAN']:
            model.fit(X_nn_train, y_enc, sample_weight=weights)
            print(f"  > Applying Weighted Manifold TTA...")
            test_preds[name] = predict_proba_tta(model, X_nn_test, tta_knn_indices, tta_knn_dists)
        else:
            # Δένδρα/PFN
            if name == 'HyperTabPFN' and model.model is None: continue
            model.fit(X_tree_train, y_enc, sample_weight=weights)
            # TTA και για τα δένδρα; Ναι, η εξομάλυνση είναι καλή.
            test_preds[name] = predict_proba_tta(model, X_tree_test, tta_knn_indices, tta_knn_dists)

    # 5. Συναίνεση & Εξόρυξη (Mining)
    print("\n[CONSENSUS] Mining Diamonds...")
    valid = list(test_preds.keys())
    diamond_indices = []
    diamond_labels = []
    
    for i in range(len(X_test)):
        votes = [np.argmax(test_preds[v][i]) for v in valid]
        confs = [np.max(test_preds[v][i]) for v in valid]
        if len(set(votes)) == 1 and min(confs) > 0.95:
            diamond_indices.append(i)
            diamond_labels.append(votes[0])
            
    print(f"  💎 Diamonds: {len(diamond_indices)}")
    
    # 6. Alchemy Refit (Μόνο Anchor)
    if len(diamond_indices) > 50:
        print("[ALCHEMY] Refitting Anchor (TurboTabR)...")
        X_pseudo = X_tree_test[diamond_indices]
        y_pseudo = np.array(diamond_labels)
        
        X_final_train = np.vstack([X_tree_train, X_pseudo])
        y_final_train = np.hstack([y_enc, y_pseudo])
        w_final = np.hstack([weights, np.ones(len(y_pseudo))*2.0])
        
        anchor = TurboTabRClassifier(n_estimators=1000)
        anchor.fit(X_final_train, y_final_train, sample_weight=w_final)
        
        final_probs = predict_proba_tta(anchor, X_tree_test, tta_knn_indices, tta_knn_dists)
    else:
        print("[ALCHEMY] Soft Voting Fallback.")
        final_probs = np.mean([test_preds[v] for v in valid], axis=0)
        
    return np.argmax(final_probs, axis=1), le

# ------------------------------------------------------------------------------
# ΚΥΡΙΩΣ ΠΡΟΓΡΑΜΜΑ (MAIN)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_data
    
    X, y, X_test = load_data()
    
    preds, le = omega_loop(X, y, X_test)
    
    final_labels = le.inverse_transform(preds)
    output_path = 'PartD/outputs/labelsX_omega.npy'
    if not os.path.exists('PartD/outputs'): os.makedirs('PartD/outputs')
    np.save(output_path, final_labels)
    
    print(f"\n[OMEGA] Execution Complete. Saved to {output_path}")
    print(f"Class Dist: {np.unique(final_labels, return_counts=True)}")
