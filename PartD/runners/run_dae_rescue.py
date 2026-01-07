
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.models_torch import ThetaTabM, SAM
from PartD.sigma_omega.utils import seed_everything

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DAE Definition (Simplified) ---
class SwapNoiseDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, swap_noise=0.15):
        super().__init__()
        self.swap_noise = swap_noise
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2), # Expand
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), # Compress
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, input_dim) # Reconstruct

    def forward(self, x):
        # Noise inject done outside or inside? Inside is easier.
        if self.training and self.swap_noise > 0:
            # Swap Noise: Replace token with random token from batch
            mask = torch.rand_like(x) < self.swap_noise
            # Quick swap implementation: shuffle batch
            x_shuffled = x[torch.randperm(x.size(0))]
            x_noised = torch.where(mask, x_shuffled, x)
        else:
            x_noised = x
            
        latent = self.encoder(x_noised)
        recon = self.head(latent)
        return latent, recon

def train_dae(X, epochs=20):
    print("  [DAE] Training Autoencoder...")
    dae = SwapNoiseDAE(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(dae.parameters(), lr=1e-3, weight_decay=1e-5)
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32).to(DEVICE)), batch_size=512, shuffle=True)
    
    loss_fn = nn.MSELoss()
    
    for ep in range(epochs):
        total_loss = 0
        for (xb,) in dl:
            opt.zero_grad()
            _, recon = dae(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
    
    print(f"  [DAE] Final Loss: {total_loss/len(dl):.4f}")
    return dae

def get_latent(dae, X):
    dae.eval()
    latents = []
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32).to(DEVICE)), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for (xb,) in dl:
            l, _ = dae(xb)
            latents.append(l.cpu().numpy())
    return np.vstack(latents)

def main():
    print(">>> RESCUE MISSION: ThetaTabM + DAE <<<")
    seed_everything(42)
    
    # 1. Load & Preproc
    X, y, _ = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Baseline Preproc
    qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal', random_state=42)
    X = qt.fit_transform(X)
    
    # 2. Train DAE
    dae = train_dae(X, epochs=30)
    
    # 3. Transform to Latent
    print("  [Transform] Projecting to Smooth Manifold (1024 dim)...")
    X_latent = get_latent(dae, X)
    
    # 4. Train ThetaTabM on Latent
    print("  [ThetaTabM] Training on Latent Features (100 Epochs)...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, val_idx = next(skf.split(X_latent, y_enc))
    
    X_tr, X_val = X_latent[tr_idx], X_latent[val_idx]
    y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]
    
    model = ThetaTabM(hidden_dim=256, depth=3, k=16, num_classes=len(le.classes_))
    model.fit(X_tr, y_tr, epochs=100) # Give it 100 epochs
    
    probs = model.predict_proba(X_val)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_val).mean()
    
    print("-" * 40)
    print(f"  RESCUE RESULT (DAE + ThetaTabM):")
    print(f"  Accuracy: {acc:.2%}")
    print("-" * 40)
    
    if acc > 0.70:
        print("  SUCCESS! DAE saved the model.")
    else:
        print("  FAILURE. Even DAE couldn't fix it.")

if __name__ == "__main__":
    main()
