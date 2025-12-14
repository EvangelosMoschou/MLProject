import numpy as np
import pandas as pd
import torch
try:
    from .data_loader import load_data
    from .dae_model import train_dae, get_dae_features
except ImportError:
    from src.data_loader import load_data
    from src.dae_model import train_dae, get_dae_features

def generate_super_dataset():
    """
    Generates the final 'Super Dataset':
    1. Loads Original Data
    2. Drops Useless Features (from dropped_features.txt)
    3. Trains DAE on Full Data (Train+Test)
    4. Adds DAE Features
    5. Saves to disk
    """
    print("--- Generating Super Dataset ---")
    
    # 1. Load Data
    X, y, X_test = load_data()
    
    # 2. Drop features
    try:
        with open('PartD/outputs/dropped_features.txt', 'r') as f:
            lines = f.readlines()
            # Extract indices from lines like "✅ Feature_193: ..." 
            # Wait, the file format I wrote was: "Feature indices identified..." then "❌ Feature_193: ..."
            # Need strict parsing.
            
            drop_indices = []
            for line in lines:
                if "❌" in line and "Feature_" in line:
                    parts = line.split("Feature_")[1] # "193: ..."
                    idx = int(parts.split(":")[0])
                    drop_indices.append(idx)
                    
        print(f"Found {len(drop_indices)} features to drop.")
        
        # Drop
        mask = np.ones(X.shape[1], dtype=bool)
        mask[drop_indices] = False
        
        X_pruned = X[:, mask]
        X_test_pruned = X_test[:, mask]
        print(f"Original Shape: {X.shape}, Pruned Shape: {X_pruned.shape}")
        
    except FileNotFoundError:
        print("⚠️ dropped_features.txt not found. Using full feature set.")
        X_pruned = X
        X_test_pruned = X_test
        
    # 3. Train DAE on ALL data (Train + Test) for best unsupervised learning
    X_all = np.vstack((X_pruned, X_test_pruned))
    
    # Train DAE
    dae = train_dae(X_all, epochs=50) # More epochs for final model
    
    # 4. Extract Features
    print("Extracting Deep Features...")
    dae_feat_train = get_dae_features(dae, X_pruned)
    dae_feat_test = get_dae_features(dae, X_test_pruned)
    
    # 5. Concatenate
    X_final = np.hstack((X_pruned, dae_feat_train))
    X_test_final = np.hstack((X_test_pruned, dae_feat_test))
    
    print(f"Final Train Shape: {X_final.shape}")
    print(f"Final Test Shape: {X_test_final.shape}")
    
    # 6. Save
    output_path = 'PartD/outputs/dataset_super.npz'
    np.savez(output_path, X=X_final, y=y, X_test=X_test_final)
    print(f"✅ Saved Super Dataset to {output_path}")

if __name__ == "__main__":
    generate_super_dataset()
