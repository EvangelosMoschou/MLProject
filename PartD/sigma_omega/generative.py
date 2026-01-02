import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from . import config

class TabularDiffusion(nn.Module):
    """
    Απλή Gaussian Diffusion για Tabular Data.
    Μαθαίνει να αποθορυβοποιεί x_t -> x_0.
    """
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

def synthesize_data(X, y, n_new_per_class=1000):
    """
    Δημιουργεί συνθετικά δεδομένα χρησιμοποιώντας diffusion models ανά κλάση.
    Εκπαιδεύεται μόνο στο παρεχόμενο X (training split).
    """
    # Συνθέτουμε μόνο αν έχουμε αρκετά δεδομένα
    if len(X) < 50:
        return X, y

    X_syn_all, y_syn_all = [], []
    classes = np.unique(y)
    
    print(f"   [DIFFUSION] Synthesizing {n_new_per_class} samples/class...")

    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 10: 
            continue
        
        # Απλή εκπαίδευση θόρυβος-αποθορυβοποίηση
        model = TabularDiffusion(Xc.shape[1]).to(config.DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        Xt = torch.tensor(Xc, dtype=torch.float32).to(config.DEVICE)
        
        model.train()
        # Εκπαίδευση για σταθερό αριθμό epochs (γρήγορο)
        for _ in range(int(config.DIFFUSION_EPOCHS)):
            noise = torch.randn_like(Xt) * 0.1
            rec = model(Xt + noise)
            loss = F.mse_loss(rec, Xt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # Δημιουργία
        model.eval()
        with torch.no_grad():
            # Αρχικοποίηση με πραγματικά δεδομένα + θόρυβο (Langevin-like)
            idx = np.random.choice(len(Xc), n_new_per_class, replace=True)
            seed = Xt[idx] + torch.randn_like(Xt[idx]) * 0.2
            gen = model(seed).cpu().numpy()
            
            X_syn_all.append(gen)
            y_syn_all.append(np.full(n_new_per_class, c))
            
    if not X_syn_all:
        return X, y
        
    X_syn = np.vstack(X_syn_all)
    y_syn = np.concatenate(y_syn_all)
    
    # Επιστροφή επαυξημένου dataset
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])
