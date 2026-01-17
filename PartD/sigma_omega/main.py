import os

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from . import config
from .data import load_data_safe
from .pipeline import predict_probs_for_view
from .pseudo import PseudoData, view_agreement_fraction, vote_mode_and_agreement
from .utils import seed_everything


def main():
    print(">>> INITIATING SIGMA-OMEGA GRANDMASTER PROTOCOL <<<")

    # 1. LOAD & RAZOR
    X, y, X_test = load_data_safe()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    # >>> ADVERSARIAL VALIDATION DIAGNOSTIC <<<
    # Checks for train/test distribution shift before training
    if config.RUN_ADV_DIAGNOSTIC:
        print("\n[ADV DIAGNOSTIC] Checking train/test distribution shift...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        X_all = np.vstack([X, X_test])
        y_dom = np.concatenate([np.zeros(len(X)), np.ones(len(X_test))])
        
        adv_clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        auc_scores = cross_val_score(adv_clf, X_all, y_dom, cv=5, scoring='roc_auc')
        mean_auc = np.mean(auc_scores)
        
        if mean_auc < 0.55:
            print(f"  ✓ AUC = {mean_auc:.3f} (No significant distribution shift detected)")
        elif mean_auc < 0.70:
            print(f"  ⚠ AUC = {mean_auc:.3f} (Mild distribution shift - consider CORAL alignment)")
        else:
            print(f"  ⚠ AUC = {mean_auc:.3f} (SIGNIFICANT shift - enable ENABLE_ADV_REWEIGHT=1)")
        print("")

    # Per-Model CV Razor (5-Fold CV-averaged importance)
    if config.ENABLE_RAZOR:
        print("[RAZOR] Computing per-model CV-averaged feature importance...")
        from catboost import CatBoostClassifier
        from sklearn.model_selection import StratifiedKFold
        import xgboost as xgb
        
        n_splits = 5
        razor_iterations = 500  # More iterations for stable importance
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # --- CatBoost CV Importance ---
        cat_importances = []
        print("  [CatBoost] Computing 5-fold CV importance...")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
            scout = CatBoostClassifier(
                iterations=razor_iterations, 
                verbose=0, 
                task_type='GPU' if torch.cuda.is_available() else 'CPU',
                random_seed=42 + fold_idx
            )
            scout.fit(X[train_idx], y_enc[train_idx])
            cat_importances.append(scout.get_feature_importance())
        cat_imp_avg = np.mean(cat_importances, axis=0)
        
        # --- XGBoost CV Importance ---
        xgb_importances = []
        print("  [XGBoost] Computing 5-fold CV importance...")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
            xgb_model = xgb.XGBClassifier(
                n_estimators=razor_iterations,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=42 + fold_idx,
                verbosity=0,
            )
            xgb_model.fit(X[train_idx], y_enc[train_idx])
            xgb_importances.append(xgb_model.feature_importances_)
        xgb_imp_avg = np.mean(xgb_importances, axis=0)
        
        # --- Create Model-Specific Masks ---
        razor_threshold = 10  # Bottom 10%
        cat_thresh = np.percentile(cat_imp_avg, razor_threshold)
        xgb_thresh = np.percentile(xgb_imp_avg, razor_threshold)
        
        cat_mask = cat_imp_avg > cat_thresh
        xgb_mask = xgb_imp_avg > xgb_thresh
        
        # For backward compatibility, use CatBoost mask as default
        keep_mask = cat_mask
        X_razor = X[:, keep_mask]
        X_test_razor = X_test[:, keep_mask]
        
        print(f"  > CatBoost mask: {np.sum(cat_mask)}/{X.shape[1]} features kept")
        print(f"  > XGBoost mask: {np.sum(xgb_mask)}/{X.shape[1]} features kept")
        
        # Store masks for per-model use in pipeline
        razor_masks = {
            'cat': cat_mask,
            'xgb': xgb_mask,
        }
    else:
        print("[RAZOR] Skipping feature selection (Max Signal Mode)...")
        # Keep all features
        all_mask = np.ones(X.shape[1], dtype=bool)
        X_razor = X
        X_test_razor = X_test
        razor_masks = {
            'cat': all_mask,
            'xgb': all_mask,
        }
        print(f"  > All {X.shape[1]} features kept.")

    # 2. MONTE CARLO LOOP
    final_ensemble_probs = 0

    if config.ENABLE_SELF_TRAIN and config.SELF_TRAIN_ITERS > 0:
        if not config.ALLOW_TRANSDUCTIVE:
            raise RuntimeError(
                "ENABLE_SELF_TRAIN requires ALLOW_TRANSDUCTIVE=1 (it uses test features for pseudo-labeling)."
            )

        pseudo = PseudoData.empty()

        last_avg_probs = None
        for it in range(int(config.SELF_TRAIN_ITERS) + 1):
            if it == 0:
                print("\n>>> SELF-TRAIN ITERATION 0 (no pseudo) <<<")
            else:
                print(f"\n>>> SELF-TRAIN ITERATION {it} (pseudo={len(pseudo.idx)}) <<<")

            probs_per_view = {v: [] for v in config.VIEWS}
            preds_per_view = {v: [] for v in config.VIEWS}

            for seed in config.SEEDS:
                seed_everything(seed)
                for view in config.VIEWS:
                    p = predict_probs_for_view(
                        view,
                        seed,
                        X_razor,
                        X_test_razor,
                        y_enc,
                        num_classes,
                        pseudo_idx=pseudo.idx,
                        pseudo_y=pseudo.y,
                        pseudo_w=pseudo.w,
                        X_train_raw=X,        # Raw data for TabPFN
                        X_test_raw=X_test,    # Raw data for TabPFN
                        razor_masks=razor_masks,  # Per-model masks
                    )
                    probs_per_view[view].append(p)
                    preds_per_view[view].append(np.argmax(p, axis=1))

            probs_tensor = []
            preds_tensor = []
            for view in config.VIEWS:
                probs_tensor.append(np.stack(probs_per_view[view], axis=0))
                preds_tensor.append(np.stack(preds_per_view[view], axis=0))
            probs_tensor = np.stack(probs_tensor, axis=0)  # (V, S, N, C)
            preds_tensor = np.stack(preds_tensor, axis=0)  # (V, S, N)

            avg_probs = probs_tensor.mean(axis=(0, 1))
            last_avg_probs = avg_probs

            if it < int(config.SELF_TRAIN_ITERS):
                votes = preds_tensor.reshape(preds_tensor.shape[0] * preds_tensor.shape[1], preds_tensor.shape[2])

                mode_pred, agree_frac_votes = vote_mode_and_agreement(votes)
                view_agree_frac = view_agreement_fraction(preds_tensor, mode_pred)

                conf = np.max(avg_probs, axis=1)
                mask = (
                    (conf >= float(config.SELF_TRAIN_CONF))
                    & (agree_frac_votes >= float(config.SELF_TRAIN_AGREE))
                    & (view_agree_frac >= float(config.SELF_TRAIN_VIEW_AGREE))
                )
                idx = np.nonzero(mask)[0]

                if idx.size > int(config.SELF_TRAIN_MAX):
                    top = np.argsort(conf[idx])[::-1][: int(config.SELF_TRAIN_MAX)]
                    idx = idx[top]

                pseudo_idx = idx.astype(np.int64)
                # Reflexion Core: Soft Pseudo-Labels
                # We use the full probability vector as the target.
                # Downstream models (Torch) will use Soft Cross Entropy.
                # Tree models will auto-convert to Hard labels in calibration.py/stacking.py.
                pseudo_y = avg_probs[pseudo_idx]
                pseudo_w = np.power(conf[pseudo_idx].astype(np.float32), float(config.SELF_TRAIN_WEIGHT_POWER))
                pseudo = PseudoData(idx=pseudo_idx, y=pseudo_y, w=pseudo_w)

                print(
                    f"  [SELF-TRAIN] mined {len(pseudo_idx)} pseudo (conf>={config.SELF_TRAIN_CONF}, votes>={config.SELF_TRAIN_AGREE}, views>={config.SELF_TRAIN_VIEW_AGREE})"
                )

        final_ensemble_probs = last_avg_probs

    else:
        checkpoint_path = 'PartD/outputs/checkpoint_probs.npy'

        if os.path.exists(checkpoint_path):
            try:
                logger_data = np.load(checkpoint_path, allow_pickle=True).item()
                final_ensemble_probs = logger_data['probs']
                completed_seeds = logger_data['seeds']
                print(f">>> FOUND CHECKPOINT. Resuming... (Completed Seeds: {completed_seeds})")
            except Exception:
                print(">>> CHECKPOINT CORRUPTED. Starting from scratch.")
                final_ensemble_probs = 0
                completed_seeds = []
        else:
            final_ensemble_probs = 0
            completed_seeds = []

        for seed in config.SEEDS:
            if seed in completed_seeds:
                print(f">>> SKIPPING SEED {seed} (Already Completed)")
                continue

            print(f"\n>>> SEQUENCE START: SEED {seed} <<<")
            seed_everything(seed)

            view_probs_total = 0
            view_count = 0
            for view in config.VIEWS:
                print(f"  [VIEW] {view}")
                if config.USE_STACKING:
                    print("  > Stacking meta-learner (OOF -> meta)...")
                view_probs_total += predict_probs_for_view(
                    view, seed, X_razor, X_test_razor, y_enc, num_classes,
                    X_train_raw=X, X_test_raw=X_test,  # Raw data for TabPFN
                    razor_masks=razor_masks,  # Per-model masks
                )
                view_count += 1
            
            if isinstance(final_ensemble_probs, int):
                final_ensemble_probs = (view_probs_total / max(1, view_count))
            else:
                final_ensemble_probs += (view_probs_total / max(1, view_count))
            
            completed_seeds.append(seed)
            np.save(checkpoint_path, {'probs': final_ensemble_probs, 'seeds': completed_seeds})
            print(f"  >>> CHECKPOINT SAVED (Seeds: {completed_seeds})")

        final_ensemble_probs /= len(config.SEEDS)

    # 3. FINAL OUTPUT
    preds = np.argmax(final_ensemble_probs, axis=1)
    labels = le.inverse_transform(preds)

    os.makedirs('PartD/outputs', exist_ok=True)
    np.save('PartD/outputs/labelsX_grandmaster.npy', labels)
    print("\n>>> GRANDMASTER PROTOCOL COMPLETE <<<")


if __name__ == '__main__':
    main()
