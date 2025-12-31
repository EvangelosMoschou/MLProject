import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x, context):
        """
        x: (Batch, d_model) -> Query
        context: (Batch, K, d_model) -> Key/Value
        """
        B, d_model = x.shape
        _, K, _ = context.shape
        
        # Reshape for multi-head: (B, H, 1, D_h)
        q = self.q_proj(x).view(B, 1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Reshape context: (B, H, K, D_h)
        k = self.k_proj(context).view(B, K, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(context).view(B, K, self.num_heads, self.d_head).transpose(1, 2)
        
        # Attention scores: (B, H, 1, K)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate: (B, H, 1, D_h)
        out = attn @ v
        
        # Reshape back: (B, d_model)
        out = out.transpose(1, 2).contiguous().view(B, d_model)
        return self.out_proj(out)

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim) # Gating mechanism
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h = self.fc1(x)
        g = F.sigmoid(self.fc2(x))
        x = h * g
        x = self.dropout(x)
        return self.fc3(x)

class TabR(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, num_heads=4, hidden_dim=256, dropout=0.1, context_k=96):
        super().__init__()
        self.encoder = LinearEncoder(input_dim, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Input to MLP is concatenated x (d_model) + context (d_model)
        self.mlp = GatedMLP(d_model * 2, hidden_dim, output_dim, dropout)
        self.context_k = context_k
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, context_emb):
        """
        x: (Batch, InputDim)
        context_emb: (Batch, K, d_model) - Pre-encoded neighbor embeddings
        """
        x_emb = self.encoder(x) # (B, D)
        
        # Cross Attention: Query=x_emb, Key/Val=context_emb
        context_agg = self.attention(x_emb, context_emb) # (B, D)
        context_agg = self.layer_norm(context_agg + x_emb) # Residual + Norm
        
        # Concatenate original X (encoded) with Context
        combined = torch.cat([x_emb, context_agg], dim=1)
        return self.mlp(combined)

class TabRClassifier:
    def __init__(self, input_dim, num_classes, device='cuda', k_neighbors=96):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.k_neighbors = k_neighbors
        self.model = TabR(input_dim, num_classes, context_k=k_neighbors).to(device)
        self.index = None
        self.X_train_raw = None
        self.scaler = None
        
    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=256, lr=1e-3):
        import torch.optim as optim
        
        # 1. Build Index
        self.X_train_raw = X.copy()
        print(f"[TabR] Building Index (K={self.k_neighbors})...")
        self.index = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=-1)
        self.index.fit(X)
        
        # 2. Prepare Tensors
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / batch_size))
        
        print("[TabR] Training...")
        for ep in range(epochs):
            total_loss = 0
            indices = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                idx_batch = indices[i*batch_size : (i+1)*batch_size]
                if len(idx_batch) == 0: continue
                
                # Fetch Batch Data
                X_batch_cpu = X[idx_batch]
                y_batch = y_tensor[idx_batch]
                
                # Retrieve Neighbors (SLOW PART - In real impl, pre-compute or use faiss-gpu)
                # Here we do it on CPU for simplicity as dataset is likely handled in memory
                # Note: TabR usually retrieves neighbors from the TRAINING SET.
                # For a batch item, we should exclude itself if it's in the index? 
                # Scikit-learn finds closest, including self. We usually want K neighbors.
                dists, neighbor_indices = self.index.kneighbors(X_batch_cpu)
                
                # Get neighbor features
                # Neighbors shape: (Batch, K, features)
                X_neighbors = self.X_train_raw[neighbor_indices] 
                
                # To Tensor
                X_batch = torch.tensor(X_batch_cpu, dtype=torch.float32).to(self.device)
                X_neigh = torch.tensor(X_neighbors, dtype=torch.float32).to(self.device)
                
                # Pre-encode neighbors for attention (Usually shared weights with encoder)
                # Optimized: We can encode X_neigh inside the model if we pass raw
                # But our model expects embeddings. 
                # Let's adjust forward: The encoder is part of the model.
                # We need to encode the neighbors using the SAME encoder.
                
                # Forward
                # We need to run encoder on Neighbors: (B*K, D)
                B, K, F_dim = X_neigh.shape
                X_neigh_flat = X_neigh.view(B*K, F_dim)
                emb_neigh = self.model.encoder(X_neigh_flat)
                emb_neigh = emb_neigh.view(B, K, -1)
                
                optimizer.zero_grad()
                logits = self.model(X_batch, emb_neigh)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (ep+1) % 10 == 0:
                print(f"  > Epoch {ep+1}: Loss {total_loss/n_batches:.4f}")
                
    def predict_proba(self, X):
        self.model.eval()
        batch_size = 256
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / batch_size))
        probs_all = []
        
        with torch.no_grad():
            for i in range(n_batches):
                X_batch_cpu = X[i*batch_size : (i+1)*batch_size]
                if len(X_batch_cpu) == 0: break
                
                # Retrieve Neighbors from TRAINING set
                dists, neighbor_indices = self.index.kneighbors(X_batch_cpu)
                X_neighbors = self.X_train_raw[neighbor_indices]
                
                X_batch = torch.tensor(X_batch_cpu, dtype=torch.float32).to(self.device)
                X_neigh = torch.tensor(X_neighbors, dtype=torch.float32).to(self.device)
                
                B, K, F_dim = X_neigh.shape
                emb_neigh = self.model.encoder(X_neigh.view(B*K, F_dim)).view(B, K, -1)
                
                logits = self.model(X_batch, emb_neigh)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_all.append(probs)
                
        return np.vstack(probs_all)
