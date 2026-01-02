import numpy as np
import torch
import torch.nn.functional as F


def compute_class_balanced_weights(y, num_classes, beta=0.999):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective + 1e-12)
    weights = weights / (weights.mean() + 1e-12)
    return weights.astype(np.float32)


def smooth_targets(targets, smoothing):
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
