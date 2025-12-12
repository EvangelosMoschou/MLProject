from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class DatasetTV:
    X: np.ndarray
    y: np.ndarray
    label_encoder: LabelEncoder


def load_dataset_tv(path: str | Path) -> DatasetTV:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training dataset not found: {p}")

    df = pd.read_csv(p, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected training dataset shape: {df.shape}")

    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y_raw = df.iloc[:, -1].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return DatasetTV(X=X, y=y, label_encoder=le)


def load_dataset_test_features(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test dataset not found: {p}")
    df = pd.read_csv(p, header=None)
    return df.to_numpy(dtype=float)
