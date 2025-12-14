# MLProject

> **Pattern Recognition & Machine Learning** — 2025-2026  
> Author: Evangelos Moschou

📘 **[Read the detailed implementation guide →](IMPLEMENTATION.md)**

---

## Quick Start

### Requirements
```bash
pip install numpy matplotlib scipy pandas nbformat ipykernel xgboost scikit-learn joblib plotly
```

### Running the Code
```bash
# Part A: Maximum Likelihood Estimation
cd PartA && python solution_a.py

# Part B: Parzen Window Density Estimation
cd PartB && python solution_b.py

# Part C: K-Nearest Neighbors Classifier
cd PartC && python solution_c.py

# Part D: Classification Challenge
cd PartD && python solution_d.py
```

---

## Project Overview

### Part A: Maximum Likelihood Estimation
Estimate parameters (mean μ, covariance Σ) of three 2D Gaussian distributions using MLE and visualize them in 3D.

**Key Features:**
- Manual MLE implementation (no library functions)
- 3D surface plots with interactive HTML visualization
- Dark-themed custom colormaps

**Outputs:** `gaussian_3d_plot.svg`, `gaussian_3d_interactive.html`, `density_peaks.txt`

---

### Part B: Parzen Window Density Estimation
Implement non-parametric density estimation using Hypercube and Gaussian kernels.

**Key Features:**
- Optimal bandwidth selection via grid search
- Error minimization against true N(1,4) distribution
- Comparative kernel analysis

**Outputs:** `histogram_verification.png`, `parzen_error_plots.png`, `best_model_stacking_fast_cpu.pkl`

---

### Part C: K-Nearest Neighbors Classifier
Build a KNN classifier from scratch with decision boundary visualization.

**Key Features:**
- Manual Euclidean distance implementation
- Z-score normalization
- Optimal k selection (validation on test set)
- Decision boundary plots

**Outputs:** `knn_accuracy.png`, `knn_decision_boundary.png`

---

### Part D: Classification Challenge (Advanced)
**Goal:** Maximize Multiclass Accuracy on a Dataset with 220 features and 5 classes.
**Final Accuracy (CV):** ~88.5% (Weighted Blend)

### Implementation Strategy
We achieved State-of-the-Art performance using a diverse ensemble approach:
1.  **Feature Engineering**:
    *   **Denoising Autoencoder (DAE)**: Trained on the full dataset to extract 64 deep bottleneck features.
    *   **Feature Selection**: Dropped 102 "useless" features identified via Permutation Importance.
    *   **Final Input**: 186 High-Quality Features (122 Original + 64 DAE).
2.  **Model 1: TabPFN (Transformer)**:
    *   A Prior-Data Fitted Network that acts as a proxy for Bayesian Inference.
    *   Used `n_estimators=32` for robust probabilistic predictions.
    *   CV Score: **87.5%**
3.  **Model 2: Optimized Stacking Ensemble**:
    *   **Base Models**: SVM, Random Forest, XGBoost (GPU), CatBoost (GPU), MLP.
    *   **Meta-Learner**: Logistic Regression.
    *   **Augmentation**: **MixUp** regularization (alpha=0.2) applied during training.
    *   CV Score: **87.0%**
4.  **Final Blending**:
    *   Soft Voting Blend: `0.55 * TabPFN + 0.45 * Stacking`.

### How to Run
```bash
# 1. Generate Super Dataset (Pruning + DAE)
python PartD/main.py --exp gen_data

# 2. Run Final Training & Prediction
python PartD/main.py --exp final
# Output: PartD/labels1.npy
```

### Detailed Report
See [walkthrough.md](walkthrough.md) for a comprehensive breakdown of the experiments, "Super Dataset" creation, and negative results (Calibration).

---

## Project Structure
```
MLProject/
├── Datasets/              # Data files (gitignored)
├── PartA/                 # MLE Implementation
│   ├── solution_a.py
│   └── [outputs]
├── PartB/                 # Parzen Window Implementation
│   ├── solution_b.py
│   └── [outputs]
├── PartC/                 # KNN Implementation
│   ├── solution_c.py
│   └── [outputs]
├── PartD/                 # Classification Challenge
│   ├── solution_d.py
│   └── labels1.npy
├── Submission/            # Final Deliverables
│   ├── Team1-AC.ipynb
│   ├── Team1-D.ipynb
│   └── labels1.npy
├── README.md              # This file (quick start)
└── IMPLEMENTATION.md      # Detailed technical documentation
```

---

## Key Highlights

| Part | Constraint | Highlight |
|------|-----------|-----------|
| **A** | No library MLE | Vectorized operations, 100x faster than loops |
| **B** | Custom kernels | Broadcasting for O(M×N) pairwise distance computation |
| **C** | No library distances | Z-score normalization prevents feature dominance |
| **D** | Production-ready | Stacking + Pseudo-Labeling achieves 87-94% accuracy |

---

## Documentation

For detailed explanations of:
- Mathematical derivations (e.g., MLE formulas, Parzen window theory)
- Code walkthroughs (line-by-line explanations)
- Design decisions (why specific algorithms/parameters)
- Performance optimizations (vectorization, GPU acceleration)

**See:** [IMPLEMENTATION.md](IMPLEMENTATION.md)

---

## License
Academic project — AUTH 2025-2026