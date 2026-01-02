import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import seed_everything

class EntropyMinimizationTTT:
    """
    Υλοποίηση Test-Time Training (TTT) μέσω Entropy Minimization.
    
    Αντί να βασιζόμαστε σε 'Silver Samples' (confident pseudo-labels), αυτή η μέθοδος
    βελτιστοποιεί το model να είναι σίγουρο σε *όλα* τα test samples ελαχιστοποιώντας
    την εντροπία των προβλέψεών του: H(p) = - sum p(x) * log p(x).
    
    Αυτό ενθαρρύνει το decision boundary να μετακινηθεί σε low-density περιοχές
    (cluster assumption).
    """
    def __init__(self, steps=1, lr=1e-4, optimizer_type='adam'):
        self.steps = steps
        self.lr = lr
        self.optimizer_type = optimizer_type

    def adapt(self, model, x_test_batch, **kwargs):
        """
        Προσαρμόζει ένα αντίγραφο του model στο δοθέν test batch.
        
        Args:
            model: Το βασικό PyTorch model.
            x_test_batch: Tensor test inputs.
            **kwargs: Extra inputs for the model (e.g. neighbors).
            
        Returns:
            adapted_model: Αντίγραφο του model fine-tuned.
        """
        # Δημιουργία deep copy για να μην τροποποιήσουμε το αρχικό model μόνιμα
        # (εκτός αν θέλουμε online TTT, αλλά γενικά episodic TTT είναι πιο ασφαλές)
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
            outputs = adapted_model(x_test_batch, **kwargs)
            log_probs = torch.log_softmax(outputs, dim=1)
            probs = torch.exp(log_probs)
            
            # 1. Entropy Loss
            entropy = -(probs * log_probs).sum(dim=1).mean()
            
            # 2. Consistency Loss (Επαύξηση/Θόρυβος)
            noise = torch.randn_like(x_test_batch) * 0.05
            outputs_n = adapted_model(x_test_batch + noise, **kwargs)
            probs_n = torch.softmax(outputs_n, dim=1)
            consistency = F.mse_loss(probs, probs_n)
            
            loss = entropy + consistency
            
            loss.backward()
            optimizer.step()            
        return adapted_model
