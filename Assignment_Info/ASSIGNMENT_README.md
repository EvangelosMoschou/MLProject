# Pattern Recognition & Machine Learning Assignment 2025-2026

**Course:** Pattern Recognition & Machine Learning  
**Instructor:** Assoc. Prof. Panagiotis Petrantonakis (ppetrant@ece.auth.gr)  
**Teaching Assistant:** PhD Candidate Stefanos Papadopoulos (stefpapad@iti.gr)  
**Deadline:** Wednesday, January 14, 2026, 23:59

---

## Overview

This assignment consists of **4 parts** (10 points total):
- **Part A** (2 points): Maximum Likelihood Estimation for Gaussian distributions
- **Part B** (2 points): Parzen Window density estimation
- **Part C** (2 points): K-Nearest Neighbors classifier
- **Part D** (4 points): Custom classification challenge (bonus competition)

---

## Part A: Maximum Likelihood Estimation (2 Points)

### Dataset
- **File:** `Datasets/dataset1.csv`
- **Structure:** 300 rows × 3 columns
  - Columns 1-2: 2D sample data (features)
  - Column 3: Class label (0, 1, or 2)
- **Distribution:** 
  - Rows 1-100: Class 0
  - Rows 101-200: Class 1
  - Rows 201-300: Class 2
- **Assumption:** Each class follows a different Normal (Gaussian) distribution

### Task
1. Use **Maximum Likelihood Estimation (MLE)** to find the parameters of three Gaussian distributions:
   - Mean vector (μ) for each class
   - Covariance matrix (Σ) for each class
2. **No library functions allowed** for MLE calculation (basic numpy operations like `np.sum`, `np.mean` are allowed)
3. Create a **single 3D plot** showing all three distributions

### Expected Output
- Mean vectors (μ₀, μ₁, μ₂) for each class
- Covariance matrices (Σ₀, Σ₁, Σ₂) for each class
- 3D visualization of the three Gaussian distributions

---

## Part B: Parzen Window Density Estimation (2 Points)

### Dataset
- **File:** `Datasets/dataset2.csv`
- **Structure:** 200 rows × 1 column (univariate data)
- **Known distribution:** N(1, 4) - Normal with μ=1, σ²=4

### Task
1. Implement **Parzen Window** method for probability density estimation
2. Create a **histogram** of the data to verify it follows N(1, 4)
3. For each h in range [0.1, 10] with step 0.1:
   - Compute predicted likelihood for each data point
   - Compute true likelihood using the Normal distribution
   - Calculate squared error between predicted and true
4. Implement **two kernels**:
   - Hypercube (uniform) kernel
   - Gaussian kernel
5. Find the optimal h for each kernel
6. Create **two plots** (one per kernel): h (x-axis) vs error (y-axis)

### Expected Output
- Histogram of data
- Best h value for hypercube kernel
- Best h value for Gaussian kernel
- Two error plots (h vs squared error)

---

## Part C: K-Nearest Neighbors Classifier (2 Points)

### Datasets
- **Training:** `Datasets/dataset3.csv` (50 rows × 3 columns)
- **Testing:** `Datasets/testset.csv` (50 rows × 3 columns)
- **Structure:** 2D features + class label (0 or 1)

### Task
1. Implement `eucl(x, trainData)`:
   - Returns Euclidean distance from x to all points in trainData
   - **No library functions** except basic numpy operations
   
2. Implement `neighbors(x, trainData, k)`:
   - Computes distances from x to all training points
   - Sorts distances in **descending order**
   - Returns the k nearest points from trainData
   
3. Implement `predict(testData, trainData, k)`:
   - For each test point, call neighbors()
   - Compute probability of belonging to class 0 or class 1
   - Probabilities must sum to 1
   - Return two probabilities per test point

4. Find optimal k:
   - Test k in range [1, 30]
   - Calculate accuracy for each k using testset.csv
   - Print best k and its accuracy
   - Create plot: k (x-axis) vs accuracy (y-axis)

5. Visualize decision boundaries:
   - Use the optimal k value
   - Use `contourf` to plot decision regions

### Expected Output
- eucl(), neighbors(), predict() functions
- Best k value and accuracy
- Accuracy vs k plot
- Decision boundary visualization

---

## Part D: Classification Challenge (4 Points + Bonus)

### Datasets
- **Training:** `Datasets/datasetTV.csv`
  - 8743 samples × 225 columns (224 features + 1 label)
  - Labels: 1, 2, 3, 4, 5 (5 classes)
- **Testing:** `Datasets/datasetTest.csv`
  - 6955 samples × 224 columns (features only, no labels)

### Task
1. Develop a classification algorithm using **any method**
2. Apply any preprocessing/feature engineering as desired
3. Save predictions as `labelsX.npy` (where X is team number)
4. Ensure `numpy.load('labelsX.npy')` works and shape is (6955,)

### Competition
- Best performing teams get bonus points
- Top teams must present their classifier in person

---

## Submission Requirements

### Files to Submit (in TeamX.zip)
1. `TeamX-AC.ipynb` - Code for Parts A, B, C
2. `TeamX-D.ipynb` - Code for Part D
3. `labelsX.npy` - Predictions for Part D test set
4. `TeamX.pdf` - Presentation slides (max 50 slides: 10 per Part A-C, 20 for Part D)

### Important Notes
- Replace X with your team number (1, 2, 3, etc., NOT 01, 02)
- Include names and student IDs in all files
- Each question in Parts A-C must be in a separate cell
- Include brief comments with all code
- Grading based on: code quality, comments, presentation quality, correctness

---

## Project Structure

```
MLProject/
├── Datasets/
│   ├── dataset1.csv    # Part A: 300 × 3 (2D + label)
│   ├── dataset2.csv    # Part B: 200 × 1 (univariate)
│   ├── dataset3.csv    # Part C: 50 × 3 (train)
│   ├── testset.csv     # Part C: 50 × 3 (test)
│   ├── datasetTV.csv   # Part D: 8743 × 225 (train)
│   └── datasetTest.csv # Part D: 6955 × 224 (test)
├── PartA/              # Maximum Likelihood Estimation
├── PartB/              # Parzen Windows
├── PartC/              # K-Nearest Neighbors
├── PartD/              # Classification Challenge
└── ASSIGNMENT_README.md
```
