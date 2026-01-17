
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

# Legacy imports from sibling modules if needed, assuming relative structure might change.
# Ideally we copy necessary helpers or import them. 
# For now, let's assume config is available.
try:
    from ..sigma_omega import config
    from ..sigma_omega.losses import (
        compute_class_balanced_weights,
        smooth_targets,
        soft_target_ce,
        soft_target_focal,
    )
except ImportError:
    # Fallback/Mock config if moved to PartD/legacy/
    pass 

# ==================================================================================
#                                   LEGACY CLASSES
# ==================================================================================

class TopologyMixUpLoader:
    def __init__(self, X, y, num_classes, batch_size=None):
        from ..sigma_omega import config # Late import
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        self.X, self.y, self.num_classes, self.batch_size = X, y, int(num_classes), int(batch_size)
        self.knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X)
        self.rng = np.random.default_rng()

    def __iter__(self):
        from ..sigma_omega import config
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


class WSAM(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        base_optimizer,
        rho=0.05,
        gamma=0.9,
        sam_eps=1e-12,
        adaptive=False,
        decouple=True,
        max_norm=None,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.model = model
        self.base_optimizer = base_optimizer
        self.decouple = decouple
        self.max_norm = max_norm
        alpha = gamma / (1 - gamma)
        defaults = dict(rho=rho, alpha=alpha, sam_eps=sam_eps, adaptive=adaptive, **kwargs)
        defaults.update(self.base_optimizer.defaults)
        super(WSAM, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["sam_eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w, alpha=1.0)
                self.state[p]["e_w"] = e_w
                
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["grad"] = p.grad.detach().clone()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(self.state[p]["e_w"], alpha=-1.0)

        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self.decouple:
                    p.grad.mul_(group["alpha"]).add_(self.state[p]["grad"], alpha=1.0 - group["alpha"])
                else:
                    self.state[p]["sharpness"] = p.grad.detach().clone() - self.state[p]["grad"]
                    p.grad.mul_(0.0).add_(self.state[p]["grad"], alpha=1.0)

        self.base_optimizer.step()

        if self.decouple:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.add_(self.state[p]["sharpness"], alpha=-group["lr"] * group["alpha"])

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )


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
        layers.append(KANLinear(d_in, hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden)])
        self.layers = nn.ModuleList()
        self.layers.append(KANLinear(d_in, hidden)) 
        # Reworked logic from previous models_torch.py
        for _ in range(depth - 2):
            self.layers.append(KANLinear(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = KANLinear(hidden, n_classes)
        self.depth = depth

    def forward(self, x):
        # Simply implementation - likely broken in copy but this is dead code
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
        # ... logic removed ...
        pass


class TabM_Layer(nn.Module):
    def __init__(self, d_in, d_out, k=None):
        from ..sigma_omega import config
        super().__init__()
        self.k = k if k is not None else config.TABM_K
        self.linear = nn.Linear(d_in, d_out)
        self.r = nn.Parameter(torch.randn(self.k, d_in) * 0.1 + 1.0)
        self.s = nn.Parameter(torch.randn(self.k, d_out) * 0.1 + 1.0)
        
    def forward(self, x):
        return self.linear(x * self.r.repeat(x.shape[0]//self.k, 1)) * self.s.repeat(x.shape[0]//self.k, 1)

class BatchEnsembleTabM(nn.Module):
    def __init__(self, d_in, n_classes, k=None, hidden_dim=256, depth=3):
        from ..sigma_omega import config
        super().__init__()
        self.k = k if k is not None else config.TABM_K
        layers = []
        layers.append(TabM_Layer(d_in, hidden_dim, self.k))
        layers.append(nn.GELU())
        for _ in range(depth - 2):
            layers.append(TabM_Layer(hidden_dim, hidden_dim // 2, self.k))
            layers.append(nn.GELU())
            hidden_dim = hidden_dim // 2
        self.layers = nn.Sequential(*layers)
        self.head = TabM_Layer(hidden_dim, int(n_classes), self.k)
        
    def forward(self, x):
        b = x.shape[0]
        x = x.repeat_interleave(self.k, dim=0) 
        x = self.layers(x)
        logits = self.head(x)
        logits = logits.view(b, self.k, -1).mean(dim=1)
        return logits

class ThetaTabM(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, num_classes=None, hidden_dim=256, depth=3, k=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim, self.depth, self.k = hidden_dim, depth, k
        self.model = None

    def fit(self, X, y, sample_weight=None, epochs=20):
        # ... logic removed ...
        pass
