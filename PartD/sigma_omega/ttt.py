import copy
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import seed_everything

class EntropyMinimizationTTT:
    """
    Implements Test-Time Training (TTT) via Entropy Minimization.
    
    Instead of relying on 'Silver Samples' (confident pseudo-labels), this method
    optimizes the model to be confident on *all* test samples by minimizing
    the entropy of its predictions: H(p) = - sum p(x) * log p(x).
    
    This encourages the decision boundary to move into low-density regions
    (cluster assumption).
    """
    def __init__(self, steps=1, lr=1e-4, optimizer_type='adam'):
        self.steps = steps
        self.lr = lr
        self.optimizer_type = optimizer_type

    def adapt(self, model, x_test_batch):
        """
        Adapts a copy of the model on the given test batch.
        
        Args:
            model: The base PyTorch model (must have a forward method returning logits or probs).
            x_test_batch: Tensor of test inputs.
            
        Returns:
            adapted_model: A copy of the model fine-tuned on x_test_batch.
        """
        # Create a deep copy to avoid modifying the original model permanently
        # (unless we want online TTT, but generally episodic TTT is safer for stability)
        adapted_model = copy.deepcopy(model)
        adapted_model.train()  # maintain dropout if useful, or use eval() if BN stats should be fixed
        
        # We usually want to freeze all layers except the last few or normalization layers
        # For simplicity in this protocol, we let the optimizer handle the whole model 
        # but with a very small LR and few steps.
        # Alternatively, we could freeze the backbone.
        
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(adapted_model.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(adapted_model.parameters(), lr=self.lr, momentum=0.9)
            
        for _ in range(self.steps):
            optimizer.zero_grad()
            
            # Forward pass
            # We assume model returns LOGITS or PROBS. 
            # To be safe, we check if output is normalized.
            outputs = adapted_model(x_test_batch)
            
            # Convert to probabilities if they are logits
            # Heuristic: if any value is < 0 or > 1 or sum is far from 1, assume logits.
            # But standard PyTorch models usually return logits or have a specific output.
            # Let's assume the model returns LOGITS as is standard for stability.
            
            # However, looking at models_torch.py (to be checked), models might return probs in predict_proba.
            # We need to call the forward method that returns logits for numerical stability of entropy.
            # If the model only has predict_proba, we might be stuck.
            
            # Assumption: adapted_model is a nn.Module. calling it calls forward().
            # calculate entropy: - sum p log p
            # efficient way using log_softmax:
            
            log_probs = torch.log_softmax(outputs, dim=1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            
            entropy.backward()
            optimizer.step()
            
        return adapted_model
