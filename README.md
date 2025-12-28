# MLProject

> **Pattern Recognition & Machine Learning** — 2025-2026  
> Author: Evangelos Moschou

---

## Quick Start

### Requirements
```bash
pip install numpy matplotlib scipy pandas xgboost scikit-learn catboost tabpfn torch sentence-transformers
```

### Running the Code
```bash
# Part A: Maximum Likelihood Estimation
cd PartA && python solution_a.py

# Part B: Parzen Window Density Estimation
cd PartB && python solution_b.py

# Part C: K-Nearest Neighbors Classifier
cd PartC && python solution_c.py

# Part D: Classification Challenge (Standard)
python PartD/main.py --exp final

# Part D: Advanced SOTA Solutions
python PartD/solution_god_mode.py      # Mamba + KAN + Diffusion
python PartD/solution_singularity.py   # RF-GNN + LLM Context
python PartD/solution_universal.py     # TabR + TTA + TabPFN
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

### Part D: Classification Challenge (Advanced)
**Goal:** Maximize Multiclass Accuracy on a Dataset with 224 features and 5 classes.

#### Standard Pipeline
- **Feature Engineering**: DAE + Feature Selection
- **Models**: TabPFN + Stacking Ensemble (XGBoost, CatBoost, SVM, RF, MLP)
- **Blending**: Weighted average optimized via Hill Climbing

#### Advanced SOTA Solutions

| Solution | Architecture | Target |
|----------|-------------|--------|
| **God-Mode** | TabM (Mamba) + KAN + Diffusion | >95% |
| **Singularity** | RF-GNN + LLM Context + Nelder-Mead | >93% |
| **Universal** | TabR (Retrieval) + TTA + TabPFN | >92% |

### How to Run
```bash
# Standard Pipeline
python PartD/main.py --exp gen_data  # Generate Super Dataset
python PartD/main.py --exp final     # Train & Predict

# Advanced Solutions (Requires ml_god_mode environment)
conda activate ml_god_mode
python PartD/solution_god_mode.py
python PartD/solution_singularity.py
python PartD/solution_universal.py
```

---

## Project Structure
```
MLProject/
├── Datasets/              # Data files (gitignored)
├── PartA/                 # MLE Implementation
├── PartB/                 # Parzen Window Implementation
├── PartC/                 # KNN Implementation
├── PartD/                 # Classification Challenge
│   ├── main.py            # Standard pipeline entry
│   ├── solution_god_mode.py      # Mamba + KAN
│   ├── solution_singularity.py   # RF-GNN + LLM
│   ├── solution_universal.py     # TabR + TTA
│   └── src/               # Core modules
├── Submission/            # Final Deliverables
│   ├── Team1-AC.ipynb
│   ├── Team1-D.ipynb
│   └── labels1.npy
├── README.md              # This file
└── walkthrough.md         # Detailed experiment log
```

---

## Key Highlights

| Part | Constraint | Highlight |
|------|-----------|-----------|
| **A** | No library MLE | Vectorized operations, 100x faster than loops |
| **B** | Custom kernels | Broadcasting for O(M×N) pairwise distance |
| **C** | No library distances | Z-score normalization prevents feature dominance |
| **D** | SOTA | Multi-architecture ensemble (Mamba, KAN, GNN, TabPFN) |

---

## License
Academic project — AUTH 2025-2026