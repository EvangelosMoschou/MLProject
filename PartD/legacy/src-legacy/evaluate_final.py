import numpy as np
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from .models import get_stacking_ensemble
from .config import USE_GPU
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

def run_final_evaluation():
    print("--- 📊 STARTING FINAL ENSEMBLE EVALUATION (80/20 Split) ---")
    
    # 1. Load datasets
    print("1. Loading Datasets...")
    try:
        data_super = np.load('PartD/outputs/dataset_super.npz')
        X_super = data_super['X']
        y_full = data_super['y']
        print(f"   Super: Train={X_super.shape}")
    except FileNotFoundError:
        print("❌ dataset_super.npz not found! Run --exp gen_data first.")
        return

    try:
        data_pruned = np.load('PartD/outputs/dataset_pruned.npz')
        X_pruned = data_pruned['X']
        print(f"   Pruned: Train={X_pruned.shape}")
    except FileNotFoundError:
        print("⚠️ dataset_pruned.npz not found. Falling back to super features for TabPFN.")
        X_pruned = X_super

    # Normalize labels for consistent multiclass probability handling
    le = LabelEncoder()
    y_full_enc = le.fit_transform(y_full)

    # 2. Split 80/20
    X_train_pruned, X_val_pruned, y_train, y_val = train_test_split(
        X_pruned,
        y_full_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_full_enc
    )
    X_train_super, X_val_super, _, _ = train_test_split(
        X_super,
        y_full_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_full_enc
    )
    print(f"   Splits: TabPFN train/val={X_train_pruned.shape}/{X_val_pruned.shape}, Stack train/val={X_train_super.shape}/{X_val_super.shape}")

    # Optional: fit OOF blender on the training split only
    blend_method = 'oof'  # 'oof' recommended; falls back if it fails
    cv_folds_oof = 5
    seed = 42
    tabpfn_n_estimators_oof = 8
    fixed_blend_weight_tabpfn = 0.55

    meta = None
    if blend_method.lower() == 'oof':
        print(f"\n2. Fitting OOF blender on train split (cv={cv_folds_oof})...")
        try:
            from .final_run import _fit_oof_blender
            meta = _fit_oof_blender(
                X_train_pruned,
                X_train_super,
                y_train,
                cv_folds=cv_folds_oof,
                seed=seed,
                tabpfn_n_estimators_oof=tabpfn_n_estimators_oof,
                use_mixup_for_stack=False,
            )
            print("   ✅ OOF blender fitted.")
        except Exception as e:
            print(f"   ⚠️ OOF blender failed, will use fixed blend: {e}")
            meta = None

    # 3. Train TabPFN (on 80%)
    print("\n3. Training TabPFN (n=32) on 80% Split...")
    probs_tabpfn = None
    try:
        from tabpfn import TabPFNClassifier
        # TabPFN
        tabpfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', n_estimators=32)
        tabpfn.fit(X_train_pruned, y_train)
        probs_tabpfn = tabpfn.predict_proba(X_val_pruned)
        acc_tabpfn = accuracy_score(y_val, np.argmax(probs_tabpfn, axis=1))
        print(f"   ✅ TabPFN Validation Accuracy: {acc_tabpfn:.4f} ({acc_tabpfn*100:.2f}%)")
        
        # Cleanup
        del tabpfn
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ TabPFN Failed: {e}")

    # 4. Train Stacking Ensemble (on 80%)
    print("\n4. Training Stacking Ensemble (no MixUp) on 80% Split...")
    clf = get_stacking_ensemble()
    clf.fit(X_train_super, y_train)
    probs_stack = clf.predict_proba(X_val_super)
    acc_stack = accuracy_score(y_val, np.argmax(probs_stack, axis=1))
    print(f"   ✅ Stacking Validation Accuracy: {acc_stack:.4f} ({acc_stack*100:.2f}%)")
    
    # 5. Blend & Evaluate
    print("\n5. Final Blending & Evaluation...")
    
    if probs_tabpfn is not None:
        if meta is not None:
            X_meta_val = np.hstack((probs_tabpfn, probs_stack))
            final_probs = meta.predict_proba(X_meta_val)
            print("   ✅ Used OOF-learned blender.")
        else:
            final_probs = (fixed_blend_weight_tabpfn * probs_tabpfn) + ((1.0 - fixed_blend_weight_tabpfn) * probs_stack)
            print(f"   ⚠️ Used fixed-weight blend (tabpfn={fixed_blend_weight_tabpfn:.2f}).")

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
