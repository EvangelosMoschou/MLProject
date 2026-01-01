
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from .config import USE_GPU

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut connection handling
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class TabularResNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_blocks=3, dropout=0.2):
        super(TabularResNet, self).__init__()
        
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.final_layer = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out = self.initial_layer(x)
        for block in self.blocks:
            out = block(out)
        out = self.final_layer(out)
        return out

class ResNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=256, num_blocks=3, dropout=0.2, 
                 lr=0.001, batch_size=256, epochs=50, 
                 validation_fraction=0.1, random_state=42, verbose=False):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        input_dim = X.shape[1]
        
        # Split for internal validation (early stopping could be added here)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, 
            random_state=self.random_state, stratify=y
        )
        
        # Convert to Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize Model
        self.model_ = TabularResNet(
            input_dim=input_dim, 
            num_classes=num_classes, 
            hidden_dim=self.hidden_dim, 
            num_blocks=self.num_blocks, 
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
        # Training Loop
        best_val_loss = float('inf')
        no_improve_count = 0
        patience = 5  # Simple early stopping
        
        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_out = self.model_(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
            
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f}")
                
            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                # Could save best state dict here
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break
                    
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet!")
            
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().numpy()
