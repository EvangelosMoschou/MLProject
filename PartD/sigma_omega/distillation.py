"""
Self-Distillation Module for Sigma-Omega Protocol.

Implements "Dark Knowledge" transfer where a student model learns
from a teacher's soft predictions (probability distributions).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import config


def distill_model(
    teacher_preds,
    student_model,
    X_unlabeled,
    temperature=3.0,
    epochs=5,
    alpha=0.7,
    y_hard=None,
):
    """
    Train a student model using soft labels from a teacher.
    
    Args:
        teacher_preds: (N, C) Soft probabilities from teacher/ensemble.
        student_model: Sklearn-compatible model with PyTorch backend.
        X_unlabeled: (N, D) Features for distillation (usually test set).
        temperature: Softening temperature (higher = softer distributions).
        epochs: Number of distillation epochs.
        alpha: Weight for soft loss vs hard loss (if y_hard provided).
        y_hard: Optional hard labels for semi-supervised distillation.
    
    Returns:
        Distilled student model.
    """
    print(f"   [DISTILL] Starting distillation (T={temperature}, α={alpha}, epochs={epochs})...")
    
    # Check if student has a PyTorch model
    if not hasattr(student_model, 'model') or student_model.model is None:
        print("   [DISTILL] Student model not initialized. Skipping distillation.")
        return student_model
    
    device = config.DEVICE
    inner_model = student_model.model
    inner_model.train()
    
    # Prepare soft targets (temperature scaling)
    teacher_soft = _soften(teacher_preds, temperature)
    
    X_t = torch.tensor(X_unlabeled, dtype=torch.float32)
    y_soft_t = torch.tensor(teacher_soft, dtype=torch.float32)
    
    if y_hard is not None:
        y_hard_t = torch.tensor(y_hard, dtype=torch.long)
        dl = DataLoader(TensorDataset(X_t, y_soft_t, y_hard_t), batch_size=config.BATCH_SIZE, shuffle=True)
    else:
        dl = DataLoader(TensorDataset(X_t, y_soft_t), batch_size=config.BATCH_SIZE, shuffle=True)
    
    opt = torch.optim.AdamW(inner_model.parameters(), lr=config.LR_SCALE * 0.5)
    
    for ep in range(epochs):
        total_loss = 0
        for batch in dl:
            if y_hard is not None:
                xb, yb_soft, yb_hard = batch
            else:
                xb, yb_soft = batch
                yb_hard = None
            
            xb = xb.to(device)
            yb_soft = yb_soft.to(device)
            
            # Forward pass
            # Handle TrueTabR which needs neighbors
            if hasattr(student_model, 'get_neighbors'):
                neighbors = student_model.get_neighbors(xb)
                logits = inner_model(xb, neighbors)
            else:
                logits = inner_model(xb)
            
            # Soft loss (KL Divergence with temperature)
            student_soft = F.log_softmax(logits / temperature, dim=1)
            soft_loss = F.kl_div(student_soft, yb_soft, reduction='batchmean') * (temperature ** 2)
            
            # Hard loss (if available)
            if yb_hard is not None:
                yb_hard = yb_hard.to(device)
                hard_loss = F.cross_entropy(logits, yb_hard)
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
            else:
                loss = soft_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        print(f"      Epoch {ep+1}/{epochs}: Loss={total_loss/len(dl):.4f}")
    
    inner_model.eval()
    return student_model


def _soften(probs, temperature):
    """Apply temperature scaling to soften probability distribution."""
    # P_soft = exp(log(P)/T) / sum(exp(log(P)/T))
    # Equivalent to: exp(logits/T) / sum(exp(logits/T))
    # Since P = softmax(logits), log(P) ~ logits (approximately)
    
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs)
    logits_scaled = logits / temperature
    logits_scaled = logits_scaled - logits_scaled.max(axis=1, keepdims=True)
    soft = np.exp(logits_scaled)
    return soft / soft.sum(axis=1, keepdims=True)


def ensemble_distill(
    base_models,
    X_train,
    y_train,
    X_test,
    teacher_preds,
    epochs=3,
):
    """
    Distill knowledge from ensemble into each base model.
    
    Args:
        base_models: List of (name, model) tuples.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features (for distillation).
        teacher_preds: Ensemble predictions on X_test.
        epochs: Distillation epochs.
    
    Returns:
        Distilled models.
    """
    print("   [ENSEMBLE DISTILL] Starting cross-distillation...")
    
    for name, model in base_models:
        # Check if model has PyTorch backend
        if hasattr(model, 'model') and model.model is not None:
            print(f"      Distilling {name}...")
            distill_model(
                teacher_preds=teacher_preds,
                student_model=model,
                X_unlabeled=X_test,
                temperature=3.0,
                epochs=epochs,
            )
    
    return base_models
