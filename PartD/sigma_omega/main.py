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

    # Razor (Seed 42 Scout)
    print("[RAZOR] Scanning for noise features...")
    from catboost import CatBoostClassifier

    scout = CatBoostClassifier(iterations=config.GBDT_ITERATIONS, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    scout.fit(X, y_enc)
    imps = scout.get_feature_importance()
    thresh = np.percentile(imps, 10)  # Conservative: only drop bottom 10%
    keep_mask = imps > thresh
    X_razor = X[:, keep_mask]
    X_test_razor = X_test[:, keep_mask]
    print(f"  > Dropped {np.sum(~keep_mask)} features. New Dim: {X_razor.shape[1]}")

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
                view_probs_total += predict_probs_for_view(view, seed, X_razor, X_test_razor, y_enc, num_classes)
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
