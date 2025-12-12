# Part D Experiments

This folder contains a lightweight experiment runner that logs results into a single project-wide CSV.

## What it does
- Runs cross-validation experiments **only on** `Datasets/datasetTV.csv` (train+labels).
- Optionally runs **adversarial validation** (train vs `Datasets/datasetTest.csv`) to detect distribution shift.
- Appends one row per run to `master_history.csv`.

## What it does NOT do
- It **does not** generate `labels*.npy` submissions.
- It **does not** pseudo-label `datasetTest.csv`.

## Usage
Run from repo root:

```bash
python -m PartD.experiments.runner --experiment adv
python -m PartD.experiments.runner --experiment xgb --folds 3
python -m PartD.experiments.runner --experiment stacking --folds 3
```

Note: `--experiment xgb` requires the `xgboost` package. If `xgboost` is not installed, `stacking` will run without the XGBoost base model.

Results are appended to `master_history.csv`.
