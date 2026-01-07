
import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
# We import KANModule directly to manually loop epochs and eval
from PartD.sigma_omega.models_torch import KANModule, SAM, compute_class_balanced_weights
from PartD.sigma_omega.utils import seed_everything

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(">>> KAN EPOCH PROBE (0 to 100) <<<")
    seed_everything(42)
    
    # 1. Load Data
    X, y, X_test = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # 2. Preprocess (Razor + QT)
    print("  [Preproc] Razor + QT...")
    from catboost import CatBoostClassifier
    scout = CatBoostClassifier(iterations=100, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    X = X[:, imps > np.percentile(imps, 10)]
    
    qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal', random_state=42)
    X = qt.fit_transform(X)
    
    # 3. Split
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, val_idx = next(skf.split(X, y_enc))
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]
    
    # 4. Setup Model (Representative Params)
    # Using 'Middle Ground': Hidden=128, Depth=3
    print("  [Model] KAN (Hidden=128, Depth=3)")
    model = KANModule(X_tr.shape[1], num_classes, hidden=128, depth=3).to(DEVICE)
    opt = SAM(model.parameters(), torch.optim.AdamW, lr=2e-3, rho=0.08)
    crit = torch.nn.CrossEntropyLoss()
    
    # 5. Manual Training Loop with Logging
    dl = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32).to(DEVICE), 
            torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
        ), 
        batch_size=512, shuffle=True
    )
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    
    print("\n   Epoch | Val Acc | Delta")
    print("   ------|---------|------")
    
    prev_acc = 0.0
    for ep in range(1, 101): # 100 Epochs
        model.train()
        for xb, yb in dl:
            # SAM Step 1
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.first_step(zero_grad=True)
            
            # SAM Step 2
            crit(model(xb), yb).backward()
            opt.second_step(zero_grad=True)
            
        # Eval every 5 epochs
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_t)
                preds = torch.argmax(logits_val, dim=1)
                acc = (preds == y_val_t).float().mean().item()
            
            delta = acc - prev_acc
            print(f"   {ep:3d}   |  {acc:.2%} | {delta:+.2%}")
            prev_acc = acc

if __name__ == "__main__":
    main()
