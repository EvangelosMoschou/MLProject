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


# TopologyMixUpLoader moved to legacy

# SAM/WSAM moved to legacy


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
        
        # --- Optimizer Selection (AdamW vs Schedule-Free) ---
        try:
            from schedulefree import AdamWScheduleFree
            print("🚀 Using Schedule-Free AdamW for TrueTabR")
            # Schedule-Free usually needs higher LR, e.g., 2x-5x
            opt = AdamWScheduleFree(self.model.parameters(), lr=config.LR_SCALE * 2.0, warmup_steps=100)
            self.optimizer_name = 'schedule_free'
        except ImportError:
            print("⚠️ Schedule-Free not found, falling back to AdamW")
            opt = optim.AdamW(self.model.parameters(), lr=config.LR_SCALE)
            self.optimizer_name = 'adamw'

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
        if hasattr(opt, 'train'):
            opt.train() # Switch optimizer to train mode (for Schedule-Free)
            
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

        # [FIX] Schedule-Free Optimization requires eval() to swap weights with SWA buffer
        if hasattr(self, 'optimizer_name') and self.optimizer_name == 'schedule_free':
             if hasattr(opt, 'eval'):
                 opt.eval()

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


