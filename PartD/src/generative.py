import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy

class DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, bottleneck_dim=64, noise_factor=0.1):
        super(DAE, self).__init__()
        self.noise_factor = noise_factor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

class GenerativeADClassifier:
    """
    Generative Anomaly/Distribution Classifier.
    Trains one DAE per class.
    Inference: Calculates reconstruction error (Energy) for each class model.
    P(y|x) propto exp(-Energy)
    """
    def __init__(self, input_dim, num_classes, device='cuda', hidden_dim=256, bottleneck_dim=64):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.models = [DAE(input_dim, hidden_dim, bottleneck_dim).to(device) for _ in range(num_classes)]
        self.trained = False
        
    def fit(self, X, y, epochs=50, batch_size=128):
        print(f"[Generative] Training {self.num_classes} DAEs...")
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        
        for c in range(self.num_classes):
            # Select samples for class c
            mask = (y_t == c)
            X_c = X_t[mask]
            
            if len(X_c) < 10:
                print(f"  > Warning: Class {c} has too few samples ({len(X_c)}). Skipping training for this class.")
                continue
                
            model = self.models[c]
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            loader = DataLoader(TensorDataset(X_c), batch_size=batch_size, shuffle=True)
            model.train()
            
            for ep in range(epochs):
                for batch in loader:
                    x_clean = batch[0]
                    noise = torch.randn_like(x_clean) * model.noise_factor
                    x_noisy = x_clean + noise
                    
                    optimizer.zero_grad()
                    recon = model(x_noisy)
                    loss = criterion(recon, x_clean)
                    loss.backward()
                    optimizer.step()
            
            print(f"  > Class {c} DAE Trained.")
            
        self.trained = True
        return self

    def _get_energy(self, X):
        """Returns Energy (Reconstruction Error) for each class: (N, C)"""
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        energies = []
        with torch.no_grad():
            for c in range(self.num_classes):
                recon = self.models[c](X_t)
                # MSE per sample: (x - x_hat)^2
                # shape: (N, D) -> mean(dim=1) -> (N,)
                loss_per_sample = torch.mean((X_t - recon)**2, dim=1)
                energies.append(loss_per_sample.cpu().numpy())
        
        return np.vstack(energies).T # (N, C)

    def predict_proba(self, X):
        energies = self._get_energy(X)
        # Convert Energy to Probability: Softmax(-Energy)
        # Scale energy to avoid numerical issues? 
        # Standard: exp(-E/T)
        logits = -energies
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
        
    def refine_sample(self, x_sample, class_idx, steps=5, lr=0.1):
        """
        Inference Trick: Gradient Descent on Input x to minimize energy for class_idx.
        Returns optimized energy.
        """
        self.eval()
        x_in = torch.tensor(x_sample, dtype=torch.float32).to(self.device).requires_grad_(True)
        model = self.models[class_idx]
        optimizer = optim.SGD([x_in], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            recon = model(x_in)
            energy = torch.mean((x_in - recon)**2)
            energy.backward()
            optimizer.step()
            
        # Final Energy
        with torch.no_grad():
            recon = model(x_in)
            final_energy = torch.mean((x_in - recon)**2).item()
            
        return final_energy

    def predict_with_refinement(self, X, threshold=0.9):
        """
        Apply refinement only to uncertain samples.
        Returns PROBABILITIES (modified).
        """
        probs = self.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        
        uncertain_indices = np.where(max_probs < threshold)[0]
        if len(uncertain_indices) > 0:
            print(f"  > [Inference Trick] Refining {len(uncertain_indices)} silver samples (Energy Descent)...")
            
            for i in uncertain_indices:
                x_i = X[i]
                best_energy = float('inf')
                best_class = np.argmax(probs[i])
                
                # Check initial energy
                # initial_energies = -np.log(probs[i] + 1e-9) # Rough approx
                
                # Try to minimize energy for each class
                results = []
                for c in range(self.num_classes):
                    e_opt = self.refine_sample(x_i, c, steps=10, lr=0.05)
                    results.append(e_opt)
                
                # Softmax the optimized energies to get new probs
                results = np.array(results)
                # logits = -results
                # exp_logits = np.exp(logits - np.max(logits))
                # new_probs = exp_logits / np.sum(exp_logits)
                
                # Or just Hard Vote based on min energy?
                # Energy is reconstruction error. Lower is better.
                best_class_refined = np.argmin(results)
                
                # Boost the probability of the winner
                probs[i] = 0.05 / (self.num_classes - 1)
                probs[i, best_class_refined] = 0.95
                
        return probs

    def eval(self):
        for m in self.models: m.eval()
