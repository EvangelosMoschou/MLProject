import numpy as np
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .models import get_stacking_ensemble
from .config import USE_GPU
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

def run_final_evaluation():
    print("--- 📊 STARTING FINAL ENSEMBLE EVALUATION (80/20 Split) ---")
    
    # 1. Load Super Dataset (Only Training Part)
    print("1. Loading Super Dataset...")
    try:
        data = np.load('PartD/outputs/dataset_super.npz')
        X_full = data['X']
        y_full = data['y'] - 1  # Fix: Convert 1-5 to 0-4
        print(f"   Total Samples: {X_full.shape}")
        print(f"   Labels: {np.unique(y_full)}")
    except FileNotFoundError:
        print("❌ Dataset not found! Run --exp gen_data first.")
        return

    # 2. Split 80/20
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    print(f"   Split: Train={X_train.shape}, Val={X_val.shape}")

    # 3. Train TabPFN (on 80%)
    print("\n2. Training TabPFN (n=32) on 80% Split...")
    probs_tabpfn = None
    try:
        from tabpfn import TabPFNClassifier
        # TabPFN
        tabpfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', n_estimators=32)
        tabpfn.fit(X_train, y_train)
        probs_tabpfn = tabpfn.predict_proba(X_val)
        acc_tabpfn = accuracy_score(y_val, np.argmax(probs_tabpfn, axis=1))
        print(f"   ✅ TabPFN Validation Accuracy: {acc_tabpfn:.4f} ({acc_tabpfn*100:.2f}%)")
        
        # Cleanup
        del tabpfn
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ TabPFN Failed: {e}")

    # 4. Train Stacking Ensemble (on 80%)
    print("\n3. Training Stacking Ensemble (w/ MixUp) on 80% Split...")
    
    # MixUp Augmentation (Only on Train split)
    mixup_alpha = 0.2
    X_mix = []
    y_mix = []
    n_samples = len(X_train)
    
    for _ in range(n_samples // 2):
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x_new = lam * X_train[idx1] + (1 - lam) * X_train[idx2]
        y_new = y_train[idx1] if lam > 0.5 else y_train[idx2]
        X_mix.append(x_new)
        y_mix.append(y_new)
        
    X_aug = np.vstack((X_train, np.array(X_mix)))
    y_aug = np.hstack((y_train, np.array(y_mix)))
    
    # Train Stack
    clf = get_stacking_ensemble()
    clf.fit(X_aug, y_aug)
    probs_stack = clf.predict_proba(X_val)
    acc_stack = accuracy_score(y_val, np.argmax(probs_stack, axis=1))
    print(f"   ✅ Stacking Validation Accuracy: {acc_stack:.4f} ({acc_stack*100:.2f}%)")
    
    # 5. Blend & Evaluate
    print("\n4. Final Blending & Evaluation...")
    
    if probs_tabpfn is not None:
        # Weighted Blend
        final_probs = (0.55 * probs_tabpfn) + (0.45 * probs_stack)
        final_preds = np.argmax(final_probs, axis=1)
        final_acc = accuracy_score(y_val, final_preds)
        print(f"   🏆 FINAL BLEND ACCURACY: {final_acc:.4f} ({final_acc*100:.2f}%)")
        
        # Comparison
        print(f"   vs TabPFN: {final_acc - acc_tabpfn:+.4f}")
        print(f"   vs Stack:  {final_acc - acc_stack:+.4f}")
    else:
        print("   ❌ Could not blend (TabPFN missing).")

if __name__ == "__main__":
    run_final_evaluation()
