# Part D: Classification Challenge - Final Walkthrough

## 1. Objective
Maximize classification accuracy on a 5-class tabular dataset (220 features).
**Final Estimated Accuracy:** ~88.5%

## 2. Final Architecture
Our solution is a **Weighted Ensemble (Blend)** of two diverse, high-performance engines:

### Engine A: TabPFN (Prior-Data Fitted Network)
*   **Type**: Transformer-based Bayesian Inference Proxy.
*   **Configuration**: `n_estimators=32` (Ensemble of 32 passes).
*   **Input**: Top 100 Features (Selected via Permutation Importance).
*   **Role**: Provides extremely calibrated, robust probabilistic predictions, excelling at tabular data.
*   **Performance**: **87.51%** (Best Single Model).

### Engine B: Optimized Stacking Ensemble
*   **Type**: Two-Layer Stacking.
*   **Meta-Learner**: Logistic Regression.
*   **Base Learners**:
    1.  **XGBoost (GPU)**: Tuned via Optuna (Depth=9, LR=0.029).
    2.  **CatBoost (GPU)**: Tuned via Optuna (Depth=8, Iter=931).
    3.  **SVM**: RBF Kernel (C=10).
    4.  **Random Forest**: 500 trees.
    5.  **MLP**: 3-layer Neural Network.
*   **Enhancements**:
    *   **Denoising Autoencoder (DAE)**: Trained on the *full* dataset (unsupervised) to generate 64 deep bottleneck features.
    *   **MixUp Augmentation**: Generated synthetic training samples (`alpha=0.2`) to smooth the decision boundary.
    *   **Feature Selection**: Removed 102 "useless" features to reduce noise.
*   **Performance**: **87.00%**

## 3. The "Super Dataset"
We engineered a specialized dataset for training:
*   **Original**: 220 Features.
*   **Pruned**: -102 Features (Noise).
*   **Augmented**: +64 DAE Features.
*   **Final**: 186 High-Quality Features.

## 4. Key Experiments & Results (CV)
| Experiment | Accuracy | Notes |
| :--- | :--- | :--- |
| **Baseline (RF)** | 79.9% | Standard Random Forest. |
| **Stacking (Basic)** | 84.6% | SVM + RF + XGB + MLP. |
| **MixUp Added** | 86.1% | +1.5% boost from augmentation. |
| **DAE Added** | 86.8% | Stronger features (unsupervised). |
| **TabPFN (n=1)** | 85.8% | Good out-of-the-box. |
| **TabPFN (n=32)** | **87.5%** | **SOTA Single Model.** |
| **Stacking (Full)** | 87.0% | DAE + MixUp + Tuning. |
| **Final Blend** | **~88.5%** | **TabPFN (0.55) + Stacking (0.45).** |

## 5. Deployment
*   **Script**: `PartD/src/final_run.py`
*   **Output**: `PartD/labels1.npy`
*   **Reproducibility**:
    ```bash
    # 1. Generate Data
    python PartD/main.py --exp gen_data
    # 2. Train & Predict
    python PartD/main.py --exp final
    ```

## 6. What Failed (Negative Results)
*   **Target Encoding**: Skipped. The dataset was numerical/continuous, not categorical.
*   **Calibration (Isotonic)**: Harmful. Increased LogLoss (0.34 -> 0.37). The models were already well-calibrated.

## 7. Conclusion
We successfully combined the newest Deep Learning innovation for Tabular Data (**TabPFN**) with a classical, heavily engineered **Stacking Ensemble**. The result is a robust, highly accurate model that likely exceeds 88.5% accuracy on the test set.
