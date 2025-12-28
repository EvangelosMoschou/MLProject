import os
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from .models import get_stacking_ensemble
from .config import USE_GPU

# Suppress warnings
warnings.filterwarnings('ignore')


def _mixup_augment(X: np.ndarray, y: np.ndarray, *, alpha: float = 0.2, ratio: float = 0.5, seed: int = 42):
    """Simple MixUp-like augmentation (hard labels). Not true MixUp; kept optional."""
    if ratio <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    n_samples = len(X)
    n_new = int(n_samples * ratio)
    if n_new <= 0:
        return X, y

    X_mix = np.empty((n_new, X.shape[1]), dtype=X.dtype)
    y_mix = np.empty((n_new,), dtype=y.dtype)
    for i in range(n_new):
        idx1, idx2 = rng.choice(n_samples, 2, replace=False)
        lam = rng.beta(alpha, alpha)
        X_mix[i] = lam * X[idx1] + (1 - lam) * X[idx2]
        y_mix[i] = y[idx1] if lam > 0.5 else y[idx2]

    X_aug = np.vstack((X, X_mix))
    y_aug = np.hstack((y, y_mix))
    return X_aug, y_aug


def _fit_oof_blender(
    X_tab: np.ndarray,
    X_stack: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int = 5,
    seed: int = 42,
    tabpfn_n_estimators_oof: int = 8,
    use_mixup_for_stack: bool = False,
):
    """Fits a meta-learner on out-of-fold probabilities from TabPFN + Stacking."""
    from tabpfn import TabPFNClassifier
    import gc

    n_classes = int(np.max(y)) + 1
    oof_tabpfn = np.zeros((len(y), n_classes), dtype=np.float32)
    oof_stack = np.zeros((len(y), n_classes), dtype=np.float32)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y), start=1):
        print(f"   OOF fold {fold}/{cv_folds}...")

        X_tab_train, X_tab_val = X_tab[train_idx], X_tab[val_idx]
        X_stack_train, X_stack_val = X_stack[train_idx], X_stack[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- TabPFN OOF ---
        tabpfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', n_estimators=tabpfn_n_estimators_oof)
        tabpfn.fit(X_tab_train, y_train)
        oof_tabpfn[val_idx] = tabpfn.predict_proba(X_tab_val)

        del tabpfn
        gc.collect()
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

        # --- Stacking OOF ---
        X_stack_train_aug, y_stack_train_aug = (X_stack_train, y_train)
        if use_mixup_for_stack:
            X_stack_train_aug, y_stack_train_aug = _mixup_augment(X_stack_train, y_train, seed=seed + fold)

        stack = get_stacking_ensemble()
        stack.fit(X_stack_train_aug, y_stack_train_aug)
        oof_stack[val_idx] = stack.predict_proba(X_stack_val)

    X_meta = np.hstack((oof_tabpfn, oof_stack))
    meta = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=5000,
        C=1.0,
    )
    meta.fit(X_meta, y)
    return meta

def run_final_submission(
    *,
    blend_method: str = 'oof',
    cv_folds_oof: int = 5,
    seed: int = 42,
    tabpfn_n_estimators_full: int = 32,
    tabpfn_n_estimators_oof: int = 8,
    use_mixup_for_stack: bool = False,
    fixed_blend_weight_tabpfn: float = 0.55,
):
    print("--- 🚀 STARTING FINAL SUBMISSION RUN ---")
    
    # 1. Load datasets (super = DAE-augmented, pruned = raw/pruned for TabPFN)
    print("1. Loading Datasets...")
    try:
        data_super = np.load('PartD/outputs/dataset_super.npz')
        X_super = data_super['X']
        y = data_super['y']
        X_test_super = data_super['X_test']
        print(f"   Super: Train={X_super.shape}, Test={X_test_super.shape}")
    except FileNotFoundError:
        print("❌ Dataset not found! Run --exp gen_data first.")
        return

    try:
        data_pruned = np.load('PartD/outputs/dataset_pruned.npz')
        X_pruned = data_pruned['X']
        X_test_pruned = data_pruned['X_test']
        print(f"   Pruned: Train={X_pruned.shape}, Test={X_test_pruned.shape}")
    except FileNotFoundError:
        print("⚠️ dataset_pruned.npz not found. Falling back to super features for TabPFN.")
        X_pruned = X_super
        X_test_pruned = X_test_super

    # Normalize labels for consistent multiclass proba handling, but save in original label space
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 2. Run TabPFN
    # 2. Fit OOF blender (recommended)
    meta = None
    if blend_method.lower() == 'oof':
        print(f"\n2. Fitting OOF blender (cv={cv_folds_oof}, tabpfn_oof_n={tabpfn_n_estimators_oof})...")
        try:
            meta = _fit_oof_blender(
                X_pruned,
                X_super,
                y_enc,
                cv_folds=cv_folds_oof,
                seed=seed,
                tabpfn_n_estimators_oof=tabpfn_n_estimators_oof,
                use_mixup_for_stack=use_mixup_for_stack,
            )
            print("   ✅ OOF blender fitted.")
        except Exception as e:
            print(f"❌ OOF blender failed: {e}")
            meta = None

    # 3. Run TabPFN (full train)
    print(f"\n3. Training TabPFN (n={tabpfn_n_estimators_full})...")
    tabpfn_output_file = 'PartD/outputs/tabpfn_final_probs_pruned.npy'
    
    try:
        if os.path.exists(tabpfn_output_file):
             print(f"   Found existing TabPFN predictions at {tabpfn_output_file}. Loading...")
             probs_tabpfn = np.load(tabpfn_output_file)
        else:
            from tabpfn import TabPFNClassifier
            import gc
            
            # TabPFN
            tabpfn = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', n_estimators=tabpfn_n_estimators_full)
            tabpfn.fit(X_pruned, y_enc)
            probs_tabpfn = tabpfn.predict_proba(X_test_pruned)
            
            # Save for safety
            np.save(tabpfn_output_file, probs_tabpfn)
            print("   ✅ TabPFN Done and Saved.")
            
            # CLEANUP
            del tabpfn
            gc.collect()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            print("   🧹 GPU Memory Cleared.")
            
    except Exception as e:
        print(f"❌ TabPFN Failed: {e}")
        probs_tabpfn = None

    # 4. Run Stacking Ensemble (optionally w/ MixUp)
    print(f"\n4. Training Stacking Ensemble (mixup={use_mixup_for_stack})...")
    X_stack_train, y_stack_train = (X_super, y_enc)
    if use_mixup_for_stack:
        X_stack_train, y_stack_train = _mixup_augment(X_super, y_enc, seed=seed)
        print(f"   Augmented Data Shape: {X_stack_train.shape}")

    clf = get_stacking_ensemble()
    clf.fit(X_stack_train, y_stack_train)
    probs_stack = clf.predict_proba(X_test)
    print("   ✅ Stacking Ensemble Done.")
    
    # 5. Blending
    print("\n5. Blending Predictions...")

    final_probs = None
    if meta is not None and probs_tabpfn is not None:
        X_meta_test = np.hstack((probs_tabpfn, probs_stack))
        final_probs = meta.predict_proba(X_meta_test)
        print("   ✅ Used OOF-learned blender.")
    elif probs_tabpfn is not None:
        final_probs = (fixed_blend_weight_tabpfn * probs_tabpfn) + ((1.0 - fixed_blend_weight_tabpfn) * probs_stack)
        print(f"   ⚠️ Used fixed-weight blend (tabpfn={fixed_blend_weight_tabpfn:.2f}).")
    else:
        final_probs = probs_stack
        print("   ⚠️ Used stacking only (TabPFN missing).")
        
    final_preds = np.argmax(final_probs, axis=1)
    final_labels = le.inverse_transform(final_preds)
    
    # 5. Save
    output_path = 'PartD/labels1.npy'
    np.save(output_path, final_labels)
    print(f"\n🎉 SUCCESS! Submission saved to {output_path}")
    print(f"   Shape: {final_labels.shape}")
    print(f"   Classes Pred: {np.unique(final_labels, return_counts=True)}")

if __name__ == "__main__":
    run_final_submission()
