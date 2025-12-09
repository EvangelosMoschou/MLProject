# Part B: Parzen Window Density Estimation - Step-by-Step Guide

## Objective
Implement the Parzen Window method to estimate probability density functions using two different kernels (Hypercube and Gaussian), find the optimal bandwidth parameter h, and visualize the results.

---

## Dataset Information
- **File:** `../Datasets/dataset2.csv`
- **Format:** 200 rows × 1 column (univariate data)
- **True underlying distribution:** Normal with μ=1, σ²=4 (meaning standard deviation σ=2)

---

## Mathematical Background

**Parzen Window Formula:**
p̂(x) = (1 / (N × h)) × Σ K((x - xᵢ) / h)

Where:
- N = number of data samples
- h = bandwidth (window width parameter)
- K = kernel function
- xᵢ = individual data points

**Hypercube Kernel:** K(u) = 0.5 if |u| ≤ 1, otherwise 0

**Gaussian Kernel:** K(u) = (1/√(2π)) × exp(-u²/2)

---

## Step-by-Step Instructions

### STEP 1: Load the Data
1. Read the CSV file `dataset2.csv` using numpy
2. Ensure the data is a 1D array (flatten if necessary)
3. Verify you have 200 samples
4. Print basic statistics: mean, variance, min, max
5. Confirm the sample mean is approximately 1 and sample variance is approximately 4

### STEP 2: Create Histogram to Verify Distribution
1. Create a histogram of the data with approximately 30 bins
2. Normalize the histogram (density=True) so it shows probability density
3. Overlay the true N(1,4) probability density function for visual comparison:
   - Create an array of x values spanning the data range
   - For each x, compute: (1/(2×√(2π))) × exp(-(x-1)²/8)
4. Add labels, title, and legend
5. Save as an image file

### STEP 3: Implement the Hypercube Kernel Function
Create a function that:
1. Takes a value (or array of values) u as input
2. Returns 0.5 for each value where |u| ≤ 1
3. Returns 0 for each value where |u| > 1
4. Must handle both single values and arrays

### STEP 4: Implement the Gaussian Kernel Function
Create a function that:
1. Takes a value (or array of values) u as input
2. Computes: (1/√(2π)) × exp(-u²/2)
3. Returns the result for each input value
4. Must handle both single values and arrays

### STEP 5: Implement the Parzen Window Estimator
Create a function that estimates density at point(s) x:
1. Input parameters: evaluation point(s) x, data array, bandwidth h, kernel function
2. For each evaluation point:
   - Calculate scaled distances: u = (x - xᵢ) / h for all data points xᵢ
   - Apply the kernel function to all scaled distances
   - Sum all kernel values
   - Divide by (N × h) to get the density estimate
3. Return the estimated density value(s)

### STEP 6: Implement True Likelihood Function
Create a function that computes the true N(1,4) probability density:
1. Input: point(s) x
2. Use the Gaussian PDF formula with μ=1 and σ²=4:
   - p(x) = (1/(2×√(2π))) × exp(-(x-1)²/8)
3. Return the true density value(s)
4. Note: You MAY use scipy.stats.norm for this step only (computing TRUE likelihood)

### STEP 7: Implement Squared Error Calculation
Create a function that computes total squared error:
1. Input: data, bandwidth h, kernel function
2. For each data point:
   - Compute the Parzen estimate at that point
   - Compute the true likelihood at that point
   - Calculate the squared difference
3. Sum all squared differences
4. Return the total squared error

### STEP 8: Find Optimal h for Each Kernel
1. Create an array of h values from 0.1 to 10.0 with step 0.1 (100 values total)
2. For the Hypercube kernel:
   - Compute squared error for each h value
   - Store all errors in an array
   - Find the h value that gives minimum error
   - Record this as optimal_h_hypercube
3. Repeat the same process for the Gaussian kernel
   - Record this as optimal_h_gaussian
4. Print both optimal h values and their corresponding minimum errors

### STEP 9: Create Error Plots
Create two separate plots (or one figure with two subplots):

**Plot 1 - Hypercube Kernel:**
1. X-axis: h values (0.1 to 10)
2. Y-axis: squared error
3. Plot the error curve
4. Mark the optimal h with a vertical line or point
5. Add title: "Hypercube Kernel: h vs Squared Error"
6. Add axis labels and grid

**Plot 2 - Gaussian Kernel:**
1. Same structure as Plot 1
2. Title: "Gaussian Kernel: h vs Squared Error"

Save both plots as image files.

### STEP 10: (Optional) Compare Estimates Visually
1. Create an array of evaluation points spanning the data range
2. Compute the Parzen estimate using optimal h for each kernel
3. Compute the true N(1,4) density
4. Plot all three curves together with the data histogram
5. Add legend identifying each curve
6. This helps visualize how well the estimates match the true distribution

---

## Expected Outputs
1. **Histogram** confirming data follows N(1,4)
2. **Optimal h value** for Hypercube kernel (printed)
3. **Optimal h value** for Gaussian kernel (printed)
4. **Two error plots** showing h vs squared error for each kernel

---

## Validation Checks
- Sample mean should be close to 1
- Sample variance should be close to 4
- Both kernels should give proper probability densities (non-negative)
- Error curves should typically show a U-shape (high error at very small and very large h)
- Gaussian kernel usually provides smoother estimates
- Optimal h values are typically in the range 0.5 to 3 for this dataset
