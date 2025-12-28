# Part D Accuracy – Recommended Changes (No Code Applied)

This file documents **specific, high-ROI** changes to improve Part D performance based on a quick review of the current Part D pipeline and the available environment.

## What I checked
- Your environment already has: `catboost`, `optuna`, `tabpfn`, `xgboost`, `scikit-learn`, `torch`.
- Current Part D entry point: `PartD/main.py`.
- Current experiment logging: `PartD/src/utils.py::log_to_history()` writes to `master_history.csv`.

## Highest ROI changes (in order)

### 1) Fix experiment runner issues that affect results stability
**File:** `PartD/main.py`
- Remove the duplicated block that calls `run_optuna_tuning(...)` **twice**.
- Add missing CLI choices for experiments that already exist in the code (e.g. `calib`, `gen_data`, `final`, `eval`).

Why it helps: avoids accidental double-running (wasted time / confusing logs) and makes runs reproducible.

### 2) Upgrade logging so runs are truly comparable
**File:** `PartD/src/utils.py`
- Record **real** `git_branch` and `git_commit` (instead of `'unknown'`).
- Store `preprocess_json`, `model_json`, `metrics_json` as **valid JSON** (use JSON serialization, not `str(...)`).
- Include the actual `cv_folds` and `seed` values used inside each experiment.

Why it helps: accurate tracking is what lets you iterate toward higher accuracy quickly.

### 3) Ensure CV is consistent, leakage-safe, and comparable across experiments
**Files:** `PartD/src/trainer.py`, model pipelines in `PartD/src/models.py`
- Keep **all preprocessing inside the fold** (PCA/scaling/feature selection/DAE feature extraction).
- Use a single shared `seed` parameter threaded through all experiments.
- Report mean±std across folds in history (not only mean).

Why it helps: prevents over-optimistic metrics and makes improvements real.

### 4) Fix MixUp so it matches the method you intended
**File:** `PartD/src/trainer.py` (`run_mixup_experiment`)
- Current implementation mixes features but then assigns a **hard label** based on `lam > 0.5`.
- True MixUp uses **soft labels** (probability targets) and a model that can train on them (usually a neural net with cross-entropy on soft targets).

Why it helps: the current version behaves like a noisy oversampling trick; real MixUp can improve generalization when done properly.

### 5) Use CatBoost explicitly as a strong base learner (often a big gain)
**Files:** `PartD/src/models.py`, `PartD/src/trainer.py`
- Add/confirm a **CatBoostClassifier** baseline and/or include it in the stacking ensemble.
- Start with simple settings (e.g., depth 6–10, learning_rate ~0.03–0.1, 2k–10k iterations with early stopping).

Why it helps: CatBoost is consistently strong on tabular multiclass problems and often boosts ensembles.

### 6) Improve adversarial validation output to guide feature choices
**File:** `PartD/src/trainer.py` (`run_adversarial_validation`)
- In addition to AUC, log **feature importances** (or permutation importance) from the adversarial classifier.
- Consider using a fast gradient-boosted model for adversarial validation to get a sharper signal.

Why it helps: if there is train/test shift, dropping/transforming a few high-shift features can improve test accuracy.

### 7) Optuna: tune the model that actually matters
**Files:** `PartD/src/trainer.py`, `PartD/src/models.py`, `PartD/src/feature_selection.py`
- Use Optuna to tune **one** strong model first (CatBoost or XGBoost), then tune stacking only after you have strong base models.
- Increase trials from `30` to `100–300` if runtime allows.

Why it helps: stacking weak models rarely beats a single well-tuned GBDT.

### 8) TabPFN: treat it as an ensemble component
**File:** `PartD/src/trainer.py` (`run_tabpfn_experiment`)
- TabPFN tends to work best with minimal preprocessing; keep it on raw/standardized numeric features only.
- Log runtime and fold times; TabPFN can be expensive.

Why it helps: TabPFN adds model diversity; even if it’s not the best alone, it often improves ensembles.

## What NOT to do until the final step
- Do not pseudo-label / retrain on `Datasets/datasetTest.csv` labels until the final submission stage.

## Practical next run order (to find gains fast)
1. `adv_val` (check shift)
2. `tabpfn` (strong diverse baseline)
3. `optuna` on CatBoost or XGBoost
4. `stacking` only after base learners are strong

---
If you want, I can also create a **separate checklist** for “final step” actions (pseudo-labeling, calibration, final training on full data) without modifying `PartD/`.
