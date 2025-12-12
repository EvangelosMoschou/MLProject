from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from PartD.experiments.datasets import load_dataset_test_features, load_dataset_tv
from PartD.experiments.experiment_logger import ExperimentTimer, append_record, make_record
from PartD.experiments.models import PreprocessConfig, build_mlp, build_rf, build_stacking, build_xgb_multiclass


def _cv_predict_proba(model, X: np.ndarray, y: np.ndarray, *, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    n_classes = int(np.max(y) + 1)
    proba_oof = np.zeros((X.shape[0], n_classes), dtype=float)
    pred_oof = np.zeros((X.shape[0],), dtype=int)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        m = model
        m.fit(X_train, y_train)

        proba = m.predict_proba(X_val)
        proba_oof[val_idx] = proba
        pred_oof[val_idx] = np.argmax(proba, axis=1)

    return pred_oof, proba_oof


def evaluate_multiclass(model, X: np.ndarray, y: np.ndarray, *, folds: int, seed: int) -> dict[str, Any]:
    y_pred, y_proba = _cv_predict_proba(model, X, y, folds=folds, seed=seed)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "logloss": float(log_loss(y, y_proba, labels=list(range(y_proba.shape[1])))),
    }
    return metrics


def adversarial_validation(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    # Binary task: 0=train, 1=test. AUC >> 0.5 indicates shift.
    X = np.vstack([X_train, X_test])
    y = np.hstack([np.zeros(X_train.shape[0], dtype=int), np.ones(X_test.shape[0], dtype=int)])

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba_oof = np.zeros((X.shape[0],), dtype=float)

    # Simple, fast baseline (with scaling).
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=1)),
    ])

    for tr, va in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        proba_oof[va] = clf.predict_proba(X[va])[:, 1]

    auc = float(roc_auc_score(y, proba_oof))

    # Fit on full data to get feature ranking (abs coef).
    clf.fit(X, y)
    lr = clf.named_steps["lr"]
    coefs = np.abs(lr.coef_.ravel())
    top_idx = np.argsort(coefs)[::-1][:25]

    return {
        "adv_auc": auc,
        "adv_top_features": [{"feature": int(i), "abs_coef": float(coefs[i])} for i in top_idx],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Part D experiments runner (NO submission label generation).")
    parser.add_argument("--train", default="Datasets/datasetTV.csv")
    parser.add_argument("--test", default="Datasets/datasetTest.csv")
    parser.add_argument("--history", default="master_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--experiment",
        choices=["xgb", "rf", "mlp", "stacking", "adv"],
        required=True,
        help="Which experiment to run. 'adv' runs adversarial validation only.",
    )
    parser.add_argument("--notes", default="")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    history_csv = (repo_root / args.history).resolve() if not Path(args.history).is_absolute() else Path(args.history)

    timer = ExperimentTimer()

    train_path = (repo_root / args.train).resolve() if not Path(args.train).is_absolute() else Path(args.train)
    test_path = (repo_root / args.test).resolve() if not Path(args.test).is_absolute() else Path(args.test)

    ds = load_dataset_tv(train_path)

    preprocess: dict[str, Any] = {"scale": True, "pca_components": None}
    model_info: dict[str, Any] = {"name": args.experiment}
    metrics: dict[str, Any]

    if args.experiment == "adv":
        X_test = load_dataset_test_features(test_path)
        metrics = adversarial_validation(X_train=ds.X, X_test=X_test, folds=args.folds, seed=args.seed)
        record = make_record(
            repo_root=repo_root,
            script=str(Path(__file__).name),
            experiment_name="adversarial_validation",
            dataset_train_path=str(train_path),
            dataset_test_path=str(test_path),
            cv_folds=args.folds,
            seed=args.seed,
            preprocess=preprocess,
            model=model_info,
            metrics=metrics,
            runtime_seconds=timer.seconds(),
            notes=args.notes,
        )
        append_record(history_csv, record)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.experiment == "xgb":
        pre = PreprocessConfig(scale=True, pca_components=None)
        model = build_xgb_multiclass(pre=pre, seed=args.seed)
        preprocess = {"scale": pre.scale, "pca_components": pre.pca_components}
        model_info = {"name": "xgb", "params": model.named_steps["xgb"].get_params()}
    elif args.experiment == "rf":
        pre = PreprocessConfig(scale=True, pca_components=None)
        model = build_rf(pre=pre, seed=args.seed)
        preprocess = {"scale": pre.scale, "pca_components": pre.pca_components}
        model_info = {"name": "rf", "params": model.named_steps["rf"].get_params()}
    elif args.experiment == "mlp":
        pre = PreprocessConfig(scale=True, pca_components=None)
        model = build_mlp(pre=pre, seed=args.seed)
        preprocess = {"scale": pre.scale, "pca_components": pre.pca_components}
        model_info = {"name": "mlp", "params": model.named_steps["mlp"].get_params()}
    elif args.experiment == "stacking":
        model = build_stacking(seed=args.seed)
        preprocess = {"scale": True, "pca_components": "mixed"}
        model_info = {"name": "stacking", "final_estimator": str(model.final_estimator)}
    else:
        raise ValueError("Unknown experiment")

    metrics = evaluate_multiclass(model, ds.X, ds.y, folds=args.folds, seed=args.seed)
    record = make_record(
        repo_root=repo_root,
        script=str(Path(__file__).name),
        experiment_name=args.experiment,
        dataset_train_path=str(train_path),
        dataset_test_path=str(test_path),
        cv_folds=args.folds,
        seed=args.seed,
        preprocess=preprocess,
        model=model_info,
        metrics=metrics,
        runtime_seconds=timer.seconds(),
        notes=args.notes,
    )

    append_record(history_csv, record)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
