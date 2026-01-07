
import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PartD.sigma_omega import config
from PartD.sigma_omega.data import load_data_safe
from PartD.sigma_omega.models_torch import ThetaTabM
from PartD.sigma_omega.utils import seed_everything

# Force 1000 Epochs (This overrides dynamic args if we call fit directly)
EPOCHS = 1000

def main():
    print(f">>> LAUNCHING DEEP PROBE: ThetaTabM x {EPOCHS} Epochs <<<")
    seed_everything(42)
    
    # 1. Load Data
    X, y, X_test = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # 2. Razor (Quick & Dirty)
    print("  [Razor] Scanning...")
    from catboost import CatBoostClassifier
    scout = CatBoostClassifier(iterations=100, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    keep_mask = imps > np.percentile(imps, 10)
    X = X[:, keep_mask]
    
    # 3. QT
    print("  [QT] Transforming...")
    qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal', random_state=42)
    X = qt.fit_transform(X)
    
    # 4. Split (Single Fold for Speed)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, val_idx = next(skf.split(X, y_enc)) # Take first fold only
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]
    
    # 5. Train
    print(f"  [Train] Starting training on {len(X_tr)} samples...")
    model = ThetaTabM(
        hidden_dim=256, 
        depth=3, 
        k=16, 
        num_classes=num_classes
    )
    
    # We rely on our new 'epochs' arg in models_torch.py
    model.fit(X_tr, y_tr, epochs=EPOCHS)
    
    # 6. Evaluate
    probs = model.predict_proba(X_val)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_val).mean()
    
    print("-" * 40)
    print(f"  DEEP PROBE RESULT ({EPOCHS} Epochs):")
    print(f"  Accuracy: {acc:.2%}")
    print("-" * 40)
    
    if acc < 0.30:
        print("  CONCLUSION: DEAD. (Architecture mismatch)")
    elif acc < 0.70:
        print("  CONCLUSION: ALIVE BUT WEAK. (Needs tuning)")
    else:
        print("  CONCLUSION: EXCELLENT. (Epochs were the key!)")

if __name__ == "__main__":
    main()
