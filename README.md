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