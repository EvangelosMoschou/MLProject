from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    timestamp_utc: str
    run_id: str
    git_branch: str | None
    git_commit: str | None
    script: str
    experiment_name: str
    dataset_train_path: str
    dataset_test_path: str | None
    cv_folds: int | None
    seed: int | None
    preprocess_json: str
    model_json: str
    metrics_json: str
    runtime_seconds: float
    notes: str


def _safe_git(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def get_git_info(repo_root: Path) -> tuple[str | None, str | None]:
    branch = _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    commit = _safe_git(["git", "rev-parse", "HEAD"], repo_root)
    return branch, commit


def ensure_history_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc",
            "run_id",
            "git_branch",
            "git_commit",
            "script",
            "experiment_name",
            "dataset_train_path",
            "dataset_test_path",
            "cv_folds",
            "seed",
            "preprocess_json",
            "model_json",
            "metrics_json",
            "runtime_seconds",
            "notes",
        ])


def append_record(history_csv: Path, record: ExperimentRecord) -> None:
    ensure_history_file(history_csv)
    with history_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = asdict(record)
        writer.writerow([row[k] for k in ExperimentRecord.__dataclass_fields__.keys()])


class ExperimentTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self._t0


def make_record(
    *,
    repo_root: Path,
    script: str,
    experiment_name: str,
    dataset_train_path: str,
    dataset_test_path: str | None,
    cv_folds: int | None,
    seed: int | None,
    preprocess: dict[str, Any],
    model: dict[str, Any],
    metrics: dict[str, Any],
    runtime_seconds: float,
    notes: str = "",
) -> ExperimentRecord:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = str(uuid.uuid4())
    branch, commit = get_git_info(repo_root)

    return ExperimentRecord(
        timestamp_utc=ts,
        run_id=run_id,
        git_branch=branch,
        git_commit=commit,
        script=script,
        experiment_name=experiment_name,
        dataset_train_path=dataset_train_path,
        dataset_test_path=dataset_test_path,
        cv_folds=cv_folds,
        seed=seed,
        preprocess_json=json.dumps(preprocess, sort_keys=True),
        model_json=json.dumps(model, sort_keys=True),
        metrics_json=json.dumps(metrics, sort_keys=True),
        runtime_seconds=runtime_seconds,
        notes=notes,
    )
