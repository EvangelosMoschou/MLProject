# MLProject

## Requirements
The following packages are required to run the project:
- `numpy`
- `matplotlib`
- `scipy`
- `pandas`
- `nbformat`
- `ipykernel`

## Project Parts

### Part A: Maximum Likelihood Estimation
Estimate the parameters (mean and covariance) of three 2D Gaussian distributions using Maximum Likelihood Estimation (MLE) and visualize them in a 3D plot.
- **Dataset**: `dataset1.csv` (300 rows × 3 columns).
- **Key Concepts**: MLE for Mean and Covariance, Gaussian Probability Density Function, 3D Visualization.
- **Constraints**: No library functions allowed for MLE calculations.

### Part B: Parzen Window Density Estimation
Implement the Parzen Window method to estimate probability density functions using Hypercube and Gaussian kernels.
- **Dataset**: `dataset2.csv` (200 rows × 1 column).
- **Key Concepts**: Kernel Density Estimation (Hypercube, Gaussian), Optimal Bandwidth Selection, Error Estimation (Squared Error).
- **Goal**: Estimate the PDF of the data and compare it with the true underlying distribution N(1,4).

### Part C: K-Nearest Neighbors Classifier
Build a K-Nearest Neighbors (KNN) classifier from scratch, find the optimal k value, and visualize decision boundaries.
- **Datasets**: `dataset3.csv` (Training) and `testset.csv` (Test).
- **Key Concepts**: Euclidean Distance, Neighbor Selection, Classification Probability, Model Accuracy, Decision Boundary Visualization.
- **Constraints**: No library distance functions allowed.

### Part D: Classification Challenge
Develop a high-performance classification model to predict labels for an unlabeled test dataset with **5 distinct classes**. This part focuses on building a robust, production-quality pipeline that maximizes generalization on unseen data.

#### Datasets
| File | Description |
|------|-------------|
| `datasetTV.csv` | Training/Validation set with ground-truth labels |
| `datasetTest.csv` | Unlabeled test set (6,955 samples) for final prediction |

#### Pipeline Overview
The solution follows a **three-phase training pipeline**:

```
Phase 1: Initial Training    →    Phase 2: Pseudo-Labeling    →    Phase 3: Final Prediction
     (Augmented TV Data)              (Expand with Test)              (Generate labels1.npy)
```

---

#### Phase 1: Data Preprocessing & Augmentation

**Feature Scaling:**
- All features are normalized using `StandardScaler` to ensure zero mean and unit variance.
- This is critical for distance-based (SVM) and gradient-based (MLP, XGBoost) algorithms.

**Dimensionality Reduction (SVM only):**
- PCA reduces features to 100 principal components before feeding into the SVM.
- This speeds up the computationally expensive RBF kernel while retaining most variance.

**Data Augmentation (Gaussian Noise Injection):**
- The training set is **doubled** by adding copies with small Gaussian noise (σ = 0.05).
- **Purpose**: Acts as a regularization technique, forcing models to learn smoother decision boundaries and improving robustness to minor input perturbations.

---

#### Phase 2: Stacking Ensemble Architecture

The core of the solution is a **Stacking Classifier** that combines the strengths of four diverse base learners:

| Model | Configuration | Strengths |
|-------|---------------|-----------|
| **SVM** | RBF kernel, C=10, PCA(100) | Excellent for complex, non-linear boundaries |
| **Random Forest** | 300 trees, parallelized | Robust to noise, handles feature interactions |
| **XGBoost** | 300 estimators, lr=0.05, depth=6 | State-of-the-art gradient boosting, GPU-accelerated |
| **MLP** | (512, 256) hidden layers, early stopping | Learns abstract representations, captures complex patterns |

**Why Stacking?**
- Each base model has different inductive biases and error patterns.
- The **meta-learner** (Logistic Regression) learns *when* to trust each model based on their out-of-fold predictions.
- Internal 3-fold cross-validation ensures the meta-learner sees unbiased predictions, preventing overfitting.

---

#### Phase 3: Pseudo-Labeling (Semi-Supervised Learning)

After initial training, the model leverages unlabeled test data to refine its decision boundaries:

1. **Predict probabilities** on the entire test set using the trained ensemble.
2. **Identify high-confidence samples** where `max(probability) ≥ 90%`.
3. **Treat these as ground truth** and add them to the training set.
4. **Retrain the entire ensemble** on this expanded dataset.

**Rationale:**
- This is a form of **transductive learning** — adapting the model to the specific test distribution.
- High-confidence samples are typically "easy" examples near cluster centers, which help clarify boundaries for ambiguous cases.

**Safeguards Against Overfitting:**
| Safeguard | How It Helps |
|-----------|--------------|
| **90% Confidence Threshold** | Only the most reliable predictions are used, minimizing error propagation |
| **Stacking with CV** | Internal cross-validation prevents the meta-learner from memorizing training data |
| **Gaussian Augmentation** | Noise injection regularizes base models and discourages overly sharp boundaries |
| **Diverse Ensemble** | Errors from one model are often corrected by others, reducing confirmation bias |

---

#### Output
- **Final predictions** are saved to `labels1.npy` (shape: 6955,).
- The trained model is persisted to `best_model_stacking_fast_cpu.pkl` for reproducibility.

## Project Structure
```
MLProject/
├── Datasets/           # Data files
├── PartA/              # MLE Implementation
├── PartB/              # Parzen Window Implementation
├── PartC/              # KNN Implementation
├── PartD/              # Classification Challenge
├── Submission/         # Final Notebooks and Deliverables
│   ├── Team1-AC.ipynb
│   ├── Team1-D.ipynb
│   └── labels1.npy
└── README.md
```