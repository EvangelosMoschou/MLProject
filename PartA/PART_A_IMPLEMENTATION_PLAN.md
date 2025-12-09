# Part A: Maximum Likelihood Estimation - Step-by-Step Guide

## Objective
Estimate the parameters (mean and covariance) of three 2D Gaussian distributions using Maximum Likelihood Estimation, then visualize them in a 3D plot.

---

## Dataset Information
- **File:** `../Datasets/dataset1.csv`
- **Format:** 300 rows × 3 columns
  - Columns 1-2: Two-dimensional feature values
  - Column 3: Class label (0, 1, or 2)
- **Class distribution:** 
  - Rows 1-100: Class 0
  - Rows 101-200: Class 1
  - Rows 201-300: Class 2

---

## Restrictions
- **NO library functions allowed** for MLE calculations
- Basic numpy operations are permitted (sum, array operations, matrix multiplication)
- You may use numpy's linalg functions for the visualization step only (determinant, inverse)

---

## Step-by-Step Instructions

### STEP 1: Load the Data
1. Read the CSV file `dataset1.csv` using numpy's loadtxt function with comma delimiter
2. Separate the data into features (first two columns) and labels (third column)
3. Verify you have 300 samples total

### STEP 2: Separate Data by Class
1. Create three subsets of the feature data based on the class labels
2. Use boolean indexing to extract samples where label equals 0, 1, or 2
3. Verify each class has exactly 100 samples

### STEP 3: Compute MLE Mean for Each Class
For each of the three classes:
1. Calculate the mean vector using the MLE formula: **μ = (1/N) × Σxᵢ**
2. This is simply the average of all samples in that class
3. Sum all the samples along axis 0 (column-wise), then divide by the number of samples N
4. The result should be a 2-element vector [mean_x1, mean_x2] for each class
5. Store these as μ₀, μ₁, μ₂

### STEP 4: Compute MLE Covariance for Each Class
For each of the three classes:
1. First, center the data by subtracting the mean vector from each sample
2. Calculate the covariance matrix using the MLE formula: **Σ = (1/N) × Σ(xᵢ - μ)(xᵢ - μ)ᵀ**
3. Implementation approach:
   - Take the centered data matrix (N × 2)
   - Compute its transpose (2 × N)
   - Multiply: (2 × N) @ (N × 2) = (2 × 2)
   - Divide the result by N
4. **Important:** Use N in the denominator (not N-1). MLE uses N.
5. The result should be a 2×2 symmetric matrix for each class
6. Store these as Σ₀, Σ₁, Σ₂

### STEP 5: Create Gaussian PDF Function
Create a function that computes the 2D Gaussian probability density:
1. The formula is: **p(x) = (1 / (2π|Σ|^½)) × exp(-½(x-μ)ᵀΣ⁻¹(x-μ))**
2. The function should accept: a point (or array of points), mean vector, covariance matrix
3. Steps inside the function:
   - Compute the determinant of the covariance matrix
   - Compute the inverse of the covariance matrix
   - Calculate the normalization constant: 1 / (2π × sqrt(determinant))
   - For each point, compute the Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
   - Return the product of normalization constant and exp(-0.5 × Mahalanobis)

### STEP 6: Create Meshgrid for Visualization
1. Determine the plotting range by finding min/max of all data in both dimensions
2. Add some margin (e.g., ±5 units) to the range
3. Create two arrays of evenly spaced points covering the x and y ranges (use about 100 points each)
4. Generate a meshgrid from these arrays to get X_grid and Y_grid matrices
5. Combine into a list of all (x,y) coordinate pairs for PDF evaluation

### STEP 7: Compute PDF Values on Grid
1. For each of the three classes:
   - Use the Gaussian PDF function from Step 5
   - Evaluate it at every point on the meshgrid
   - Pass in that class's mean (μ) and covariance (Σ)
   - Reshape the result to match the grid dimensions
2. You should have three Z matrices (Z₀, Z₁, Z₂) containing PDF values

### STEP 8: Create 3D Visualization
1. Create a 3D figure using matplotlib's Axes3D
2. Plot each of the three distributions as a surface:
   - Use plot_surface with X_grid, Y_grid, and Z matrices
   - Use different colormaps for each class (e.g., Reds, Greens, Blues)
   - Set alpha (transparency) to around 0.6 so overlapping surfaces are visible
3. Add axis labels: "Feature 1", "Feature 2", "Probability Density"
4. Add a title: "MLE-Estimated 2D Gaussian Distributions"
5. Add a legend identifying each class
6. Adjust the viewing angle for best visualization (e.g., elevation=30°, azimuth=45°)
7. Save the figure as a PNG file

### STEP 9: Display Results Summary
Print a formatted summary showing for each class:
1. The class number
2. The mean vector μ with values to 4 decimal places
3. The full 2×2 covariance matrix Σ with values to 4 decimal places
4. The individual variances (diagonal elements of Σ)
5. The covariance (off-diagonal element)
6. The correlation coefficient: ρ = σ₁₂ / √(σ₁₁ × σ₂₂)

---

## Expected Outputs
1. **Three mean vectors** (2D each)
2. **Three covariance matrices** (2×2 each, symmetric)
3. **One 3D plot** showing all three Gaussian surfaces
4. **Printed summary** of all parameters

---

## Validation Checks
- Each class should have exactly 100 samples
- Covariance matrices must be symmetric (element [0,1] equals [1,0])
- Covariance matrices should be positive semi-definite (determinant ≥ 0)
- The 3D plot should show three distinct bell-shaped surfaces
- Means should roughly correspond to the visual centers of each class cluster
