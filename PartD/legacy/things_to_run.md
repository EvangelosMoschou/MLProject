# Things To Run (Execution List)

This file tracks the specific, computationally intensive tasks we need to run to maximize our model's performance.

## 1. Feature Selection (Decluttering Data)
- [ ] **Run Permutation Importance**: Identify "useless" or noisy columns.
- [ ] **Update Dataset**: Create a `datasetTV_selected.csv` with only the top predictive features.
- [ ] **Verify**: Check if Stacking Baseline score improves with fewer features.

## 2. Hyperparameter Tuning (Optimizing the Stack)
The Stacking Ensemble currently uses default parameters. We need to tune the engines:
- [ ] **Tune XGBoost (GPU)**: Optimize `learning_rate`, `max_depth`, `subsample` using Optuna.
- [ ] **Tune CatBoost (GPU)**: Optimize `iterations`, `depth`, `l2_leaf_reg`.
- [ ] **Tune Random Forest (CPU)**: Optimize `n_estimators`, `max_features` (already partially done).
- [ ] **Update `run_experiments.py`**: Plug the best found parameters back into the `get_stacking_ensemble()` function.

## 3. Final Model Generation
- [ ] **Train Optimized Stacking Ensemble**: Train on the full `datasetTV` (with feature selection and tuned params).
- [ ] **Re-run High-Quality TabPFN**: Ensure we have the latest `n_estimators=32` predictions saved.

## 4. Blending (The Grand Finale)
- [ ] **Find Optimal Weights**: Use the OOF (Out-Of-Fold) predictions from TabPFN and Stacking to find the best mix (e.g., `0.7 * TabPFN + 0.3 * Stack`).
- [ ] **Generate Submission**: Create the final `labels1.npy` file for the leaderboard.
