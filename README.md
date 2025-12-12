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

**Outputs:** `histogram_verification.png`, `parzen_error_plots.png`

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

### Part D: Classification Challenge
Production-quality 5-class classification using advanced ensemble techniques.

**Methodology:**
- **Stacking Ensemble**: SVM + Random Forest + XGBoost + MLP
- **Pseudo-Labeling**: Semi-supervised learning with 90% confidence threshold
- **Data Augmentation**: Gaussian noise injection (σ = 0.05)

**Pipeline:**
```
Phase 1: Train ensemble on augmented data
    ↓
Phase 2: Pseudo-label high-confidence test samples
    ↓
Phase 3: Retrain on expanded dataset → Final predictions
```

**Outputs:** `labels1.npy`, `best_model_stacking_fast_cpu.pkl`

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