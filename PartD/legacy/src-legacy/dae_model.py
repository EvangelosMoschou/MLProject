import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from .config import USE_GPU

class DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, bottleneck_dim=64, noise_factor=0.1):
        super(DAE, self).__init__()
        self.noise_factor = noise_factor
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU() # Bottleneck features are ReLU activated
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_features(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return features.cpu().numpy()

def train_dae(X, epochs=50, batch_size=128):
    """
    Trains a DAE on X and returns the trained model.
    Using Swap Noise (more advanced than Gaussian) is standard for Tabular DAE,
    but we will implement Gaussian for simplicity and speed as a start.
    """
    print("--- Training Denoising Autoencoder (DAE) ---")
    
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"DAE Training on {device}")
    
    # Check for feature selection list to potentially reduce input dims (Optional, skipped for now to keep logic simple)
    input_dim = X.shape[1]
    
    model = DAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x_clean = batch[0]
            
            # Add Gaussian Noise
            noise = torch.randn_like(x_clean) * model.noise_factor
            x_noisy = x_clean + noise
            
            optimizer.zero_grad()
            reconstruction = model(x_noisy)
            loss = criterion(reconstruction, x_clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.5f}")
            
    model.eval()
    return model

def get_dae_features(model, X):
    """Returns the bottleneck features for X."""
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    return model.get_features(X_tensor)
