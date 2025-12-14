import os
import numpy as np
import torch
import warnings
from sklearn.metrics import accuracy_score
from .models import get_stacking_ensemble
from .config import USE_GPU

# Suppress warnings
warnings.filterwarnings('ignore')

def run_final_submission():
    print("--- 🚀 STARTING FINAL SUBMISSION RUN ---")
    
    # 1. Load Super Dataset
    print("1. Loading Super Dataset...")
    try:
        data = np.load('PartD/outputs/dataset_super.npz')
        X = data['X']
        y = data['y']
        X_test = data['X_test']
        print(f"   Loaded: Train={X.shape}, Test={X_test.shape}")
    except FileNotFoundError:
        print("❌ Dataset not found! Run --exp gen_data first.")
        return

    # 2. Run TabPFN
    print("\n2. Training TabPFN (n=32)...")
    tabpfn_output_file = 'PartD/outputs/tabpfn_final_probs.npy'
    
    try:
        if os.path.exists(tabpfn_output_file):
             print(f"   Found existing TabPFN predictions at {tabpfn_output_file}. Loading...")
             probs_tabpfn = np.load(tabpfn_output_file)
        else:
            from tabpfn import TabPFNClassifier
            import gc
            
            # TabPFN
            tabpfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', n_estimators=32)
            tabpfn.fit(X, y)
            probs_tabpfn = tabpfn.predict_proba(X_test)
            
            # Save for safety
            np.save(tabpfn_output_file, probs_tabpfn)
            print("   ✅ TabPFN Done and Saved.")
            
            # CLEANUP
            del tabpfn
            gc.collect()
            torch.cuda.empty_cache()
            print("   🧹 GPU Memory Cleared.")
            
    except Exception as e:
        print(f"❌ TabPFN Failed: {e}")
        probs_tabpfn = None

    # 3. Run Stacking Ensemble (with MixUp)
    print("\n3. Training Optimized Stacking Ensemble (w/ MixUp)...")
    
    # MixUp Augmentation
    mixup_alpha = 0.2
    X_mix = []
    y_mix = []
    n_samples = len(X)
    
    # Create 50% more data
    for _ in range(n_samples // 2):
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x_new = lam * X[idx1] + (1 - lam) * X[idx2]
        y_new = y[idx1] if lam > 0.5 else y[idx2]
        X_mix.append(x_new)
        y_mix.append(y_new)
        
    X_aug = np.vstack((X, np.array(X_mix)))
    y_aug = np.hstack((y, np.array(y_mix)))
    print(f"   Augmented Data Shape: {X_aug.shape}")
    
    # Train Stack
    clf = get_stacking_ensemble()
    clf.fit(X_aug, y_aug)
    probs_stack = clf.predict_proba(X_test)
    print("   ✅ Stacking Ensemble Done.")
    
    # 4. Blending
    print("\n4. Blending Predictions...")
    
    if probs_tabpfn is not None:
        # Weighted Blend (Higher weight to TabPFN usually better, but let's do 0.6 Tab / 0.4 Stack)
        # Actually, since our Stack+DAE was 87.0% and TabPFN 87.5%, they are close.
        # Let's trust 0.55 TabPFN + 0.45 Stack
        final_probs = (0.55 * probs_tabpfn) + (0.45 * probs_stack)
    else:
        final_probs = probs_stack
        
    final_preds = np.argmax(final_probs, axis=1)
    
    # 5. Save
    output_path = 'PartD/labels1.npy'
    np.save(output_path, final_preds)
    print(f"\n🎉 SUCCESS! Submission saved to {output_path}")
    print(f"   Shape: {final_preds.shape}")
    print(f"   Classes Pred: {np.unique(final_preds, return_counts=True)}")

if __name__ == "__main__":
    run_final_submission()
