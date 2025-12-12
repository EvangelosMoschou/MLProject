from __future__ import annotations

from dataclasses import dataclass

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class PreprocessConfig:
    scale: bool = True
    pca_components: int | None = None


def _maybe_preprocess(pre: PreprocessConfig) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = []
    if pre.scale:
        steps.append(("scaler", StandardScaler()))
    if pre.pca_components is not None:
        steps.append(("pca", PCA(n_components=pre.pca_components)))
    return steps


def build_xgb_multiclass(*, pre: PreprocessConfig, seed: int) -> Pipeline:
    if xgb is None:
        raise RuntimeError(
            "xgboost is not installed in this environment. "
            "Install it or run a different experiment (rf/mlp/stacking/adv)."
        )
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }
    steps = _maybe_preprocess(pre)
    steps.append(("xgb", xgb.XGBClassifier(**params)))
    return Pipeline(steps)


def build_rf(*, pre: PreprocessConfig, seed: int) -> Pipeline:
    steps = _maybe_preprocess(pre)
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def build_mlp(*, pre: PreprocessConfig, seed: int) -> Pipeline:
    steps = _maybe_preprocess(pre)
    steps.append(
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(512, 256),
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def build_svc_prob(*, pre: PreprocessConfig, seed: int) -> Pipeline:
    steps = _maybe_preprocess(pre)
    steps.append(
        (
            "svc",
            SVC(C=10, gamma="scale", kernel="rbf", probability=True, random_state=seed),
        )
    )
    return Pipeline(steps)


def build_stacking(*, seed: int) -> StackingClassifier:
    # Keep it simple and probability-capable for logloss.
    pre_basic = PreprocessConfig(scale=True, pca_components=100)

    estimators = [
        ("svc", build_svc_prob(pre=pre_basic, seed=seed)),
        ("rf", build_rf(pre=PreprocessConfig(scale=True, pca_components=None), seed=seed)),
        ("mlp", build_mlp(pre=PreprocessConfig(scale=True, pca_components=None), seed=seed)),
    ]

    if xgb is not None:
        estimators.insert(
            2,
            (
                "xgb",
                build_xgb_multiclass(
                    pre=PreprocessConfig(scale=True, pca_components=None),
                    seed=seed,
                ),
            ),
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=3,
        n_jobs=1,
    )
