import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .losses import (
    compute_class_balanced_weights,
    smooth_targets,
    soft_target_ce,
    soft_target_focal,
)


def is_torch_model(m):
    return hasattr(m, 'finetune_on_pseudo') and callable(getattr(m, 'finetune_on_pseudo'))


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


class TopologyMixUpLoader:
    def __init__(self, X, y, num_classes, batch_size=None):
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        self.X, self.y, self.num_classes, self.batch_size = X, y, int(num_classes), int(batch_size)
        self.knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X)
        self.rng = np.random.default_rng()

    def __iter__(self):
        idxs = self.rng.permutation(len(self.X))
        for i in range(0, len(self.X), self.batch_size):
            b_idxs = idxs[i:i + self.batch_size]
            X_b, y_b = self.X[b_idxs], self.y[b_idxs]

            mn_idxs = self.knn.kneighbors(X_b, return_distance=False)
            rand_n = self.rng.integers(1, 5, size=len(X_b))
            target_idxs = mn_idxs[np.arange(len(X_b)), rand_n]
            X_target = self.X[target_idxs]
            y_target = self.y[target_idxs]

            lam = self.rng.beta(0.4, 0.4, size=(len(X_b), 1))
            lam = np.maximum(lam, 1 - lam)
            X_mix = lam * X_b + (1 - lam) * X_target

            y_b_oh = np.eye(self.num_classes, dtype=np.float32)[y_b]
            y_t_oh = np.eye(self.num_classes, dtype=np.float32)[y_target]
            y_mix = lam * y_b_oh + (1 - lam) * y_t_oh

            yield (
                torch.tensor(X_mix, dtype=torch.float32).to(config.DEVICE),
                torch.tensor(y_mix, dtype=torch.float32).to(config.DEVICE),
            )


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                p.add_((torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def step(self):
        raise NotImplementedError


class TabRModule(nn.Module):
    def __init__(self, input_dim, num_classes, context_size=96, hidden_dim=128, head_hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, context_size))
        self.q_proj = nn.Linear(context_size, context_size)
        self.k_proj = nn.Linear(context_size, context_size)
        self.v_proj = nn.Linear(context_size, context_size)
        self.head = nn.Sequential(nn.Linear(context_size, head_hidden), nn.SiLU(), nn.Linear(head_hidden, num_classes))
        self.scale = context_size ** -0.5

    def forward(self, x, neighbors):
        q = self.encoder(x).unsqueeze(1)
        B, K, D = neighbors.shape
        kv = self.encoder(neighbors.view(B * K, D)).view(B, K, -1)
        scores = torch.bmm(self.q_proj(q), self.k_proj(kv).transpose(1, 2)) * self.scale
        context = torch.bmm(F.softmax(scores, dim=-1), self.v_proj(kv)).squeeze(1)
        return self.head(context + q.squeeze(1))


class TrueTabR(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, n_neighbors=16, context_size=96, hidden_dim=128, head_hidden=64):
        self.num_classes, self.n_neighbors = num_classes, n_neighbors
        self.context_size, self.hidden_dim, self.head_hidden = context_size, hidden_dim, head_hidden
        self.model, self.knn, self.X_train_ = None, None, None

    def fit(self, X, y, sample_weight=None, epochs=20):
        self.X_train_ = np.array(X, dtype=np.float32)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(self.X_train_)
        train_neighbor_idx = self.knn.kneighbors(self.X_train_, return_distance=False)
        self.model = TabRModule(
            X.shape[1], 
            self.num_classes,
            context_size=self.context_size,
            hidden_dim=self.hidden_dim,
            head_hidden=self.head_hidden
        ).to(config.DEVICE)
        opt = optim.AdamW(self.model.parameters(), lr=config.LR_SCALE)

        class_w = None
        if config.USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y, self.num_classes, beta=config.CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=config.DEVICE)

        if config.LABEL_SMOOTHING > 0:
            crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(config.LABEL_SMOOTHING))
        else:
            crit = nn.CrossEntropyLoss(weight=class_w)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        idx_t = torch.arange(len(X_t), dtype=torch.long)
        if sample_weight is None:
            dl = DataLoader(TensorDataset(X_t, y_t, idx_t), batch_size=config.BATCH_SIZE, shuffle=True)
        else:
            w_t = torch.tensor(np.asarray(sample_weight, dtype=np.float32))
            dl = DataLoader(TensorDataset(X_t, y_t, idx_t, w_t), batch_size=config.BATCH_SIZE, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for batch in dl:
                if sample_weight is None:
                    xb, yb, ib = batch
                    wb = None
                else:
                    xb, yb, ib, wb = batch

                nx = self.X_train_[train_neighbor_idx[ib.numpy()]]
                logits = self.model(xb.to(config.DEVICE), torch.tensor(nx, dtype=torch.float32).to(config.DEVICE))

                if config.LOSS_NAME == 'focal':
                    y_onehot = F.one_hot(yb.to(config.DEVICE), num_classes=self.num_classes).float()
                    y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                    probs = torch.softmax(logits, dim=1).clamp(1e-8, 1.0 - 1e-8)
                    pt = (probs * y_onehot).sum(dim=1)
                    loss_vec = -torch.pow(1.0 - pt, float(config.FOCAL_GAMMA)) * torch.log(pt)
                    if class_w is not None:
                        w_class = (y_onehot * class_w.view(1, -1)).sum(dim=1)
                        loss_vec = loss_vec * w_class
                    if wb is not None:
                        loss_vec = loss_vec * wb.to(config.DEVICE)
                    loss = loss_vec.mean()
                else:
                    loss_vec = F.cross_entropy(
                        logits,
                        yb.to(config.DEVICE),
                        reduction='none',
                        weight=class_w,
                        label_smoothing=float(config.LABEL_SMOOTHING) if config.LABEL_SMOOTHING > 0 else 0.0,
                    )
                    if wb is not None:
                        loss_vec = loss_vec * wb.to(config.DEVICE)
                    loss = loss_vec.mean()

                opt.zero_grad(); loss.backward(); opt.step()

        return self

    def finetune_on_pseudo(self, X_pseudo, y_pseudo, epochs=1, lr_mult=0.2):
        if X_pseudo is None or len(X_pseudo) == 0:
            return self
        self.model.train()
        lr = float(config.LR_SCALE) * float(lr_mult)
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        # Check if soft labels
        y_is_soft = (y_pseudo.ndim > 1) or np.issubdtype(y_pseudo.dtype, np.floating)
        
        class_w = None
        # Class balancing is tricky with soft labels, usually computed on argmax or weighted sum
        if config.USE_CLASS_BALANCED:
            y_for_bal = np.argmax(y_pseudo, axis=1) if y_is_soft else y_pseudo
            class_w_np = compute_class_balanced_weights(y_for_bal, self.num_classes, beta=config.CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=config.DEVICE)

        if not y_is_soft:
            # Hard labels setup
            if config.LABEL_SMOOTHING > 0:
                crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(config.LABEL_SMOOTHING))
            else:
                crit = nn.CrossEntropyLoss(weight=class_w)
            y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.int64))
        else:
            # Soft labels setup
            y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.float32))

        X_t = torch.tensor(np.asarray(X_pseudo, dtype=np.float32))
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=config.BATCH_SIZE, shuffle=True)

        for _ in range(int(epochs)):
            for xb, yb in dl:
                xb = xb.to(config.DEVICE)
                yb = yb.to(config.DEVICE)
                nx = self.X_train_[self.knn.kneighbors(xb.detach().cpu().numpy(), return_distance=False)]
                logits = self.model(xb, torch.tensor(nx, dtype=torch.float32).to(config.DEVICE))
                
                if y_is_soft:
                    # Soft label loss
                    if config.LOSS_NAME == 'focal':
                         loss = soft_target_focal(logits, yb, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                         loss = soft_target_ce(logits, yb, class_weights=class_w)
                else:
                    # Hard label loss
                    if config.LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                        loss = soft_target_focal(logits, y_onehot, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                        loss = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        return self

    def predict_proba(self, X):
        self.model.eval()
        p = []
        for i in range(0, len(X), config.BATCH_SIZE):
            xb = X[i:i + config.BATCH_SIZE]
            nx = self.X_train_[self.knn.kneighbors(xb, return_distance=False)]
            with torch.no_grad():
                p.append(
                    torch.softmax(
                        self.model(
                            torch.tensor(xb, dtype=torch.float32).to(config.DEVICE),
                            torch.tensor(nx).to(config.DEVICE),
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
        return np.vstack(p)

    def get_neighbors(self, X):
        """Retrieve neighbors for TTT."""
        # Check if model handles torch tensors or numpy
        is_torch = torch.is_tensor(X)
        if is_torch:
            x_np = X.detach().cpu().numpy()
        else:
            x_np = X
        
        # Retrieval
        nx = self.X_train_[self.knn.kneighbors(x_np, return_distance=False)]
        
        if is_torch:
            return torch.tensor(nx, dtype=torch.float32).to(X.device)
        return nx


class KANLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
    def forward(self, x):
        return F.linear(x * F.silu(x), self.w)

class KANModule(nn.Module):
    def __init__(self, d_in, n_classes, hidden=64, depth=2):
        super().__init__()
        layers = []
        # Input Layer
        layers.append(KANLinear(d_in, hidden))
        layers.append(nn.LayerNorm(hidden))
        
        # Hidden Layers
        for _ in range(depth - 1): # -1 because input layer is one
             layers.append(KANLinear(hidden, hidden))
             layers.append(nn.LayerNorm(hidden)) # Optional: LayerNorm between KAN layers? Usually yes.
             # Or maybe just one KANLinear is enough? 
             # Standard KAN is 2 layers usually.
             # Let's simple keep it simple: KANLinear -> Norm -> KANLinear ...
        
        # Head (Linear or KANLinear?) Usually final is Linear for logits, but KAN uses Splines everywhere.
        # My implementation `KANLinear` is a linear projection on transformed features.
        # Let's assume the last layer maps to n_classes.
        # But wait, my previous code had `KANLinear(d_in, hidden)` then `KANLinear(hidden, n_classes)`.
        # So "Depth" = number of KAN layers.
        # Reworking loop:
        
        self.layers = nn.ModuleList()
        self.layers.append(KANLinear(d_in, hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden)])
        
        for _ in range(depth - 2):
            self.layers.append(KANLinear(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
            
        # Final Output Layer (KANLinear to classes)
        self.head = KANLinear(hidden, n_classes)
        self.depth = depth

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[i](x)
        return self.head(x)


class KAN(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, num_classes=None, hidden=64, depth=2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden = hidden
        self.depth = depth
        self.model = None

    def fit(self, X, y, sample_weight=None, epochs=20):
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
            
        if self.model is None:
            feat_dim = X.shape[1]
            self.model = KANModule(feat_dim, self.num_classes, hidden=self.hidden, depth=self.depth).to(config.DEVICE)

        opt = SAM(self.model.parameters(), optim.AdamW, lr=config.LR_SCALE, rho=config.SAM_RHO)

        class_w = None
        if config.USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y, self.num_classes, beta=config.CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=config.DEVICE)

        self.model.train()
        
        # Training loop (Simplified without MixUp/SWA for KAN to match simple implementation or minimal template)
        # Using standard training loop from ThetaTabM but simplified if appropriate, or full copy.
        # Given KAN is "experimental" usually, maybe standard training is enough.
        # I will use the robust loop from ThetaTabM for consistency.
        
        dl = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32).to(config.DEVICE),
                torch.tensor(y, dtype=torch.long).to(config.DEVICE),
                torch.tensor(np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else np.ones(len(X), dtype=np.float32)).to(config.DEVICE)
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

        for ep in range(epochs):
            for xb, yb, wb in dl:
                opt.zero_grad()
                logits = self.model(xb)
                
                # Hard coded Loss logic for brevity, or reuse function?
                # Reusing inline logic from ThetaTabM for now to ensure consistency
                if config.LOSS_NAME == 'focal':
                    y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                    y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                    probs = torch.softmax(logits, dim=1).clamp(1e-8, 1.0 - 1e-8)
                    pt = (probs * y_onehot).sum(dim=1)
                    loss_vec = -torch.pow(1.0 - pt, float(config.FOCAL_GAMMA)) * torch.log(pt)
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
                        label_smoothing=float(config.LABEL_SMOOTHING) if config.LABEL_SMOOTHING > 0 else 0.0,
                    )
                    loss = (loss_vec * wb).mean()

                loss.backward()
                opt.first_step(zero_grad=True)
                
                # Second step
                logits2 = self.model(xb)
                if config.LOSS_NAME == 'focal':
                    # ... recompute ...
                    # To save lines I will just use the same loss function call if possible but I'll duplicate for safety
                    y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                    y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                    probs2 = torch.softmax(logits2, dim=1).clamp(1e-8, 1.0 - 1e-8)
                    pt2 = (probs2 * y_onehot).sum(dim=1)
                    loss2_vec = -torch.pow(1.0 - pt2, float(config.FOCAL_GAMMA)) * torch.log(pt2)
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
                        label_smoothing=float(config.LABEL_SMOOTHING) if config.LABEL_SMOOTHING > 0 else 0.0,
                    )
                    loss2 = (loss2_vec * wb).mean()
                    
                loss2.backward()
                opt.second_step(zero_grad=True)
                opt.base_optimizer.step()
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        p = []
        with torch.no_grad():
            for i in range(0, len(X), config.BATCH_SIZE):
                p.append(
                    torch.softmax(
                        self.model(torch.tensor(X[i:i + config.BATCH_SIZE], dtype=torch.float32).to(config.DEVICE)),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
        return np.vstack(p)

    def finetune_on_pseudo(self, X_pseudo, y_pseudo, epochs=1, lr_mult=0.2):
        if X_pseudo is None or len(X_pseudo) == 0:
            return self
        self.model.train()
        lr = float(config.LR_SCALE) * float(lr_mult)
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        y_is_soft = (y_pseudo.ndim > 1) or np.issubdtype(y_pseudo.dtype, np.floating)
        
        class_w = None
        if config.USE_CLASS_BALANCED:
            y_for_bal = np.argmax(y_pseudo, axis=1) if y_is_soft else y_pseudo
            class_w_np = compute_class_balanced_weights(y_for_bal, self.num_classes, beta=config.CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=config.DEVICE)

        if not y_is_soft:
            if config.LABEL_SMOOTHING > 0:
                hard_crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(config.LABEL_SMOOTHING))
            else:
                hard_crit = nn.CrossEntropyLoss(weight=class_w)
            y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.int64))
        else:
            y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.float32))

        X_t = torch.tensor(np.asarray(X_pseudo, dtype=np.float32))
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=config.BATCH_SIZE, shuffle=True)

        for _ in range(int(epochs)):
            for xb, yb in dl:
                xb = xb.to(config.DEVICE)
                yb = yb.to(config.DEVICE)
                logits = self.model(xb)
                
                if y_is_soft:
                    if config.LOSS_NAME == 'focal':
                         loss = soft_target_focal(logits, yb, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                         loss = soft_target_ce(logits, yb, class_weights=class_w)
                else:
                    if config.LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                        loss = soft_target_focal(logits, y_onehot, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                        loss = hard_crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        return self

class TabM_Layer(nn.Module):
    def __init__(self, d_in, d_out, k=None):
        super().__init__()
        self.k = k if k is not None else config.TABM_K
        self.linear = nn.Linear(d_in, d_out)
        self.r = nn.Parameter(torch.randn(self.k, d_in) * 0.1 + 1.0)
        self.s = nn.Parameter(torch.randn(self.k, d_out) * 0.1 + 1.0)
        
    def forward(self, x):
        # x: (B, D) ή (B*K, D)
        # Αν το input είναι (B, D), κάνουμε repeat interleave; Όχι, συνήθως το BatchEnsemble επεκτείνει batch.
        # Αλλά εδώ υποθέτουμε ότι το input `x` αντιστοιχεί σε συγκεκριμένα ensemble members αν έχει επεκταθεί.
        # Απλή περίπτωση: standard batch -> ensemble forward.
        # Αλλά τα PyTorch layers περιμένουν (N, D).
        # Υποθέτουμε ότι το X έχει επαναληφθεί K φορές upstream ή το κάνουμε εδώ.
        # Απλούστερο: Το layer περιμένει (B*K, D).
        return self.linear(x * self.r.repeat(x.shape[0]//self.k, 1)) * self.s.repeat(x.shape[0]//self.k, 1)

class BatchEnsembleTabM(nn.Module):
    def __init__(self, d_in, n_classes, k=None, hidden_dim=256, depth=3):
        super().__init__()
        self.k = k if k is not None else config.TABM_K
        
        layers = []
        # Input Layer
        layers.append(TabM_Layer(d_in, hidden_dim, self.k))
        layers.append(nn.GELU())
        
        # Hidden Layers
        for _ in range(depth - 2):
            layers.append(TabM_Layer(hidden_dim, hidden_dim // 2, self.k))
            layers.append(nn.GELU())
            hidden_dim = hidden_dim // 2
            
        # Head
        self.layers = nn.Sequential(*layers)
        self.head = TabM_Layer(hidden_dim, int(n_classes), self.k)
        
    def forward(self, x):
        b = x.shape[0]
        # Extend for ensemble: (B, D) -> (B*K, D)
        x = x.repeat_interleave(self.k, dim=0) 
        x = self.layers(x)
        # (B*K, C)
        logits = self.head(x)
        # Average over K
        logits = logits.view(b, self.k, -1).mean(dim=1)
        return logits

class ThetaTabM(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, num_classes=None, hidden_dim=256, depth=3, k=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim, self.depth, self.k = hidden_dim, depth, k
        self.model = None

    def _build_model(self, dim):
        # Legacy/Unused
        return nn.Sequential().to(config.DEVICE)

    def fit(self, X, y, sample_weight=None, epochs=20):
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
            
        if self.model is None:
            feat_dim = X.shape[1]
            # Use improved BatchEnsemble architecture
            self.model = BatchEnsembleTabM(
                feat_dim, 
                self.num_classes,
                k=self.k,
                hidden_dim=self.hidden_dim,
                depth=self.depth
            ).to(config.DEVICE)

        opt = SAM(self.model.parameters(), optim.AdamW, lr=config.LR_SCALE, rho=config.SAM_RHO)

        class_w = None
        if config.USE_CLASS_BALANCED:
            class_w_np = compute_class_balanced_weights(y, self.num_classes, beta=config.CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=config.DEVICE)

        if config.LABEL_SMOOTHING > 0:
            hard_crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(config.LABEL_SMOOTHING))
        else:
            hard_crit = nn.CrossEntropyLoss(weight=class_w)

        self.model.train()
        swa_model = None
        if config.ENABLE_SWA:
            try:
                from torch.optim.swa_utils import AveragedModel

                swa_model = AveragedModel(self.model)
            except Exception:
                swa_model = None

        for ep in range(epochs):
            use_mixup_local = config.USE_MIXUP and (sample_weight is None)
            if use_mixup_local:
                iterator = TopologyMixUpLoader(X, y, num_classes=self.num_classes)
                for xb, yb in iterator:
                    yb = smooth_targets(yb, config.LABEL_SMOOTHING)
                    opt.zero_grad()
                    logits = self.model(xb)
                    if config.LOSS_NAME == 'focal':
                        loss = soft_target_focal(logits, yb, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                        loss = soft_target_ce(logits, yb, class_weights=class_w)
                    loss.backward(); opt.first_step(zero_grad=True)

                    logits2 = self.model(xb)
                    if config.LOSS_NAME == 'focal':
                        loss2 = soft_target_focal(logits2, yb, gamma=config.FOCAL_GAMMA, class_weights=class_w)
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
                        torch.tensor(X, dtype=torch.float32).to(config.DEVICE),
                        torch.tensor(y, dtype=torch.long).to(config.DEVICE),
                        torch.tensor(w_arr, dtype=torch.float32).to(config.DEVICE),
                    ),
                    batch_size=config.BATCH_SIZE,
                    shuffle=True,
                )

                for xb, yb, wb in dl:
                    opt.zero_grad()
                    logits = self.model(xb)
                    if config.LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                        probs = torch.softmax(logits, dim=1).clamp(1e-8, 1.0 - 1e-8)
                        pt = (probs * y_onehot).sum(dim=1)
                        loss_vec = -torch.pow(1.0 - pt, float(config.FOCAL_GAMMA)) * torch.log(pt)
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
                            label_smoothing=float(config.LABEL_SMOOTHING) if config.LABEL_SMOOTHING > 0 else 0.0,
                        )
                        loss = (loss_vec * wb).mean()

                    loss.backward(); opt.first_step(zero_grad=True)

                    logits2 = self.model(xb)
                    if config.LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                        probs2 = torch.softmax(logits2, dim=1).clamp(1e-8, 1.0 - 1e-8)
                        pt2 = (probs2 * y_onehot).sum(dim=1)
                        loss2_vec = -torch.pow(1.0 - pt2, float(config.FOCAL_GAMMA)) * torch.log(pt2)
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
                            label_smoothing=float(config.LABEL_SMOOTHING) if config.LABEL_SMOOTHING > 0 else 0.0,
                        )
                        loss2 = (loss2_vec * wb).mean()

                    loss2.backward(); opt.second_step(zero_grad=True); opt.base_optimizer.step()

            if swa_model is not None and ep >= int(config.SWA_START_EPOCH):
                swa_model.update_parameters(self.model)

        if swa_model is not None and int(config.SWA_START_EPOCH) < 20:
            self.model.load_state_dict(swa_model.module.state_dict())

        return self

    def finetune_on_pseudo(self, X_pseudo, y_pseudo, epochs=1, lr_mult=0.2):
        if X_pseudo is None or len(X_pseudo) == 0:
            return self
        self.model.train()
        lr = float(config.LR_SCALE) * float(lr_mult)
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        y_is_soft = (y_pseudo.ndim > 1) or np.issubdtype(y_pseudo.dtype, np.floating)

        class_w = None
        if config.USE_CLASS_BALANCED:
            y_for_bal = np.argmax(y_pseudo, axis=1) if y_is_soft else y_pseudo
            class_w_np = compute_class_balanced_weights(y_for_bal, self.num_classes, beta=config.CB_BETA)
            class_w = torch.tensor(class_w_np, dtype=torch.float32, device=config.DEVICE)

        if not y_is_soft:
            if config.LABEL_SMOOTHING > 0:
                hard_crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(config.LABEL_SMOOTHING))
            else:
                hard_crit = nn.CrossEntropyLoss(weight=class_w)
            y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.int64))
        else:
            y_t = torch.tensor(np.asarray(y_pseudo, dtype=np.float32))

        X_t = torch.tensor(np.asarray(X_pseudo, dtype=np.float32))
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=config.BATCH_SIZE, shuffle=True)

        for _ in range(int(epochs)):
            for xb, yb in dl:
                xb = xb.to(config.DEVICE)
                yb = yb.to(config.DEVICE)
                logits = self.model(xb)
                
                if y_is_soft:
                    if config.LOSS_NAME == 'focal':
                         loss = soft_target_focal(logits, yb, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                         loss = soft_target_ce(logits, yb, class_weights=class_w)
                else:
                    if config.LOSS_NAME == 'focal':
                        y_onehot = F.one_hot(yb, num_classes=self.num_classes).float()
                        y_onehot = smooth_targets(y_onehot, config.LABEL_SMOOTHING)
                        loss = soft_target_focal(logits, y_onehot, gamma=config.FOCAL_GAMMA, class_weights=class_w)
                    else:
                        loss = hard_crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        return self

    def predict_proba(self, X):
        self.model.eval()
        p = []
        with torch.no_grad():
            for i in range(0, len(X), config.BATCH_SIZE):
                p.append(
                    torch.softmax(
                        self.model(torch.tensor(X[i:i + config.BATCH_SIZE], dtype=torch.float32).to(config.DEVICE)),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
        return np.vstack(p)
