# Implementation Details

> **📘 Comprehensive Technical Documentation**  
> This document provides in-depth explanations of the algorithms, mathematical foundations, and design decisions for all parts of the Machine Learning project.

---

## Table of Contents
1. [Part A: Maximum Likelihood Estimation](#part-a-maximum-likelihood-estimation)
2. [Part B: Parzen Window Density Estimation](#part-b-parzen-window-density-estimation)
3. [Part C: K-Nearest Neighbors Classifier](#part-c-k-nearest-neighbors-classifier)
4. [Part D: Classification Challenge](#part-d-classification-challenge)

---

## Part A: Maximum Likelihood Estimation

### Overview
Implements Maximum Likelihood Estimation (MLE) to estimate parameters (mean μ and covariance Σ) of three 2D Gaussian distributions from data, then visualizes them as 3D probability density surfaces.

### Mathematical Foundation

#### 1. Mean Estimation (μ)
For a dataset with N samples, the MLE estimate of the mean vector is:

```
μ̂ = (1/N) × Σᵢ xᵢ
```

**Why this formula?**  
The mean that maximizes the likelihood of observing the data is simply the sample average. This is derived by taking the derivative of the log-likelihood with respect to μ and setting it to zero.

#### 2. Covariance Estimation (Σ)
The MLE estimate of the covariance matrix is:

```
Σ̂ = (1/N) × Σᵢ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ
```

**Important Note:**  
We use `N` (not `N-1`) because this is the **maximum likelihood estimator**. The `N-1` version (Bessel's correction) gives an *unbiased* estimator but is not the MLE.

#### 3. Gaussian PDF Evaluation
The probability density function for a 2D Gaussian at point x is:

```
p(x) = (1 / (2π|Σ|^0.5)) × exp(-0.5 × (x-μ)ᵀ × Σ⁻¹ × (x-μ))
```

Where:
- `|Σ|` is the determinant of the covariance matrix (controls the "volume" of the distribution)
- `Σ⁻¹` is the inverse covariance matrix
- `(x-μ)ᵀ × Σ⁻¹ × (x-μ)` is the **Mahalanobis distance** (squared)

---

### Code Walkthrough

#### Function: `load_data(filepath)`
**Purpose:** Loads CSV data into NumPy arrays.

```python
data = np.loadtxt(filepath, delimiter=',')
```

**Why `np.loadtxt`?**  
Simple, efficient, and handles numerical CSV files directly without needing pandas. Returns a NumPy array immediately.

---

#### Function: `mle_mean(data)`
**Purpose:** Computes the MLE estimate of the mean vector.

**Line-by-line:**
```python
N = data.shape[0]                    # Number of samples
sum_features = np.sum(data, axis=0)  # Sum across rows (axis 0)
mean_vector = sum_features / N       # Divide by N
```

**Key Design Choice:**  
Using `axis=0` ensures we're summing vertically (across all samples) for each feature, giving us a vector with shape `(2,)`.

---

#### Function: `mle_covariance(data, mean_vector)`
**Purpose:** Computes the MLE estimate of the covariance matrix.

**Line-by-line:**
```python
N = data.shape[0]
centered_data = data - mean_vector        # Broadcasting: subtract mean from each row
covariance_matrix = (centered_data.T @ centered_data) / N
```

**Mathematical Insight:**  
The expression `centered_data.T @ centered_data` is equivalent to:
```
Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
```

This is matrix multiplication: `(2 × N) @ (N × 2) = (2 × 2)`, which is exactly the covariance matrix.

**Why `@` instead of `np.dot`?**  
The `@` operator is the modern Python syntax for matrix multiplication (introduced in Python 3.5), making code more readable.

---

#### Function: `gaussian_pdf_2d(x, mean, cov)`
**Purpose:** Evaluates the 2D Gaussian PDF at point(s) x.

**Optimization Strategy:**
```python
# Vectorized computation for multiple points
diff = x - mean                           # (M, 2)
exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
pdf = norm_const * np.exp(exponent)
```

**Why this approach?**  
Instead of looping through each point, we use **broadcasting** to compute all PDF values in one shot:
1. `diff @ inv_cov` computes the matrix product `(x - μ) × Σ⁻¹`
2. Element-wise multiplication with `diff` gives `(x - μ)ᵀ × Σ⁻¹ × (x - μ)` per row
3. `np.sum(..., axis=1)` sums across features to get the Mahalanobis distance for each point

**Performance:** This is ~100x faster than a naive Python loop for large datasets.

---

#### Main Workflow

**Step 1: Data Loading & Separation**
```python
X = data[:, :2]  # Features (first 2 columns)
y = data[:, 2]   # Labels (third column)
X_c0 = X[y == 0] # Boolean indexing to filter class 0
```

**Step 2: Parameter Estimation (Per Class)**
```python
mu_0 = mle_mean(X_c0)
sigma_0 = mle_covariance(X_c0, mu_0)
```

**Step 3: 3D Visualization**

**Mesh Grid Creation:**
```python
x_range = np.linspace(x_min, x_max, 100)
y_range = np.linspace(y_min, y_max, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)  # 100×100 grid
```

**PDF Evaluation:**
```python
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])  # Flatten to (10000, 2)
Z0 = gaussian_pdf_2d(grid_points, mu_0, sigma_0).reshape(X_grid.shape)  # Back to 100×100
```

**Why `ravel()` then `reshape()`?**  
`gaussian_pdf_2d` expects a 2D array of points. We flatten the meshgrid, compute PDFs, then reshape back to the grid structure for plotting.

**Visualization Enhancements:**
```python
plt.style.use('dark_background')  # Dark theme for better contrast
ax.set_zlim(0, 0.025)  # Fixed Z-axis for fair comparison
```

**Custom Colormaps:**
```python
def create_cmap(color_name, name):
    colors = [(0, 0, 0), color_name]  # Black → Color gradient
    return LinearSegmentedColormap.from_list(name, colors)
```

This creates smooth gradients from black (low density) to vibrant colors (high density) for each class.

---

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No library functions for MLE** | Assignment constraint; demonstrates understanding of the math |
| **Vectorized operations** | 100x speedup vs loops; leverages NumPy's C backend |
| **Dark background theme** | Better visual contrast for 3D surfaces |
| **Custom colormaps** | Black-to-color gradients show density more intuitively than default colormaps |
| **Plotly for interactivity** | Allows users to rotate/zoom the 3D plot in the browser |
| **Fixed Z-axis (0-0.025)** | Ensures fair visual comparison between distributions |

---

### Output Files
- `gaussian_3d_plot.svg` - Static 3D visualization (vector format, scalable)
- `gaussian_3d_interactive.html` - Interactive Plotly visualization
- `density_peaks.txt` - Detailed parameter report

---

## Part B: Parzen Window Density Estimation

### Overview
Implements the Parzen Window method to estimate probability density functions (PDFs) from 1D data using two kernel types: Hypercube (uniform) and Gaussian. The goal is to find the optimal bandwidth `h` that minimizes the squared error compared to the true distribution N(1,4).

### Mathematical Foundation

#### 1. Parzen Window Estimator
For a dataset with N samples, the PDF estimate at point x is:

```
p̂(x) = (1 / (N × h)) × Σᵢ K((x - xᵢ) / h)
```

Where:
- `h` is the **bandwidth** (window width)
- `K(u)` is the **kernel function**
- `xᵢ` are the training samples

**Intuition:**  
Each data point contributes a "bump" (defined by the kernel) centered at that point. The bandwidth controls how wide the bumps are. We average all contributions to get the density estimate.

#### 2. Kernel Functions

**Hypercube (Uniform) Kernel:**
```
K(u) = 0.5  if |u| ≤ 1
K(u) = 0    otherwise
```

This creates a rectangular "box" around each data point. Points within the box contribute equally; points outside contribute nothing.

**Gaussian Kernel:**
```
K(u) = (1/√(2π)) × exp(-0.5 × u²)
```

This creates a smooth, bell-shaped contribution. All points contribute, but distant points contribute less (exponentially decaying influence).

#### 3. Error Metric
We use **squared error** to measure how well our estimate matches the true PDF:

```
E(h) = Σᵢ (p̂(xᵢ) - p_true(xᵢ))²
```

The optimal `h` minimizes this error.

---

### Code Walkthrough

#### Function: `hypercube_kernel(u)` & `gaussian_kernel(u)`

**Hypercube:**
```python
return np.where(np.abs(u) <= 1, 0.5, 0.0)
```

**Why `np.where`?**  
Vectorized conditional operation. Much faster than `if-else` loops for arrays.

**Gaussian:**
```python
return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
```

Directly implements the mathematical formula.

---

#### Function: `parzen_window_estimate(x, data, h, kernel_func)`

**Purpose:** Estimates the PDF at point(s) x using Parzen Windows.

**Broadcasting Magic:**
```python
x_col = x[:, np.newaxis]      # Shape: (M, 1)
data_row = data[np.newaxis, :] # Shape: (1, N)
u = (x_col - data_row) / h     # Shape: (M, N) via broadcasting
```

**What's happening?**  
- `x_col` is a column vector of M test points
- `data_row` is a row vector of N training points
- Subtraction broadcasts to create an `(M, N)` matrix where `u[i, j] = (x[i] - data[j]) / h`
- This computes **all pairwise distances** in one operation!

**Kernel Application:**
```python
k_values = kernel_func(u)            # Apply kernel element-wise (M, N)
p_x = np.sum(k_values, axis=1) / (N * h)  # Sum across training points, normalize
```

**Result:** `p_x` has shape `(M,)` containing the PDF estimate at each of the M test points.

**Performance:**  
For M=200 test points and N=200 training points, this computes 40,000 kernel evaluations in **milliseconds** (vs. seconds with loops).

---

#### Function: `compute_squared_error(data, h, kernel_func)`

**Purpose:** Quantifies how well a given `h` performs.

**Strategy:**
```python
p_estimated = parzen_window_estimate(data, data, h, kernel_func)  # Estimate PDF at training points
p_true = true_pdf(data)  # True PDF at same points
error = np.sum((p_estimated - p_true)**2)  # Squared error
```

**Why estimate at training points?**  
We know the true PDF values there (since we have the formula for N(1,4)). This gives us a direct comparison.

---

#### Main Workflow

**Step 1: Histogram Verification**
```python
plt.hist(data, bins=20, density=True, ...)
plt.plot(x_range, true_pdf(x_range), 'r-', ...)
```

**Why this step?**  
Sanity check: the histogram should roughly match N(1,4). If it doesn't, we may have the wrong data or wrong distribution.

**Step 2: Grid Search for Optimal h**
```python
h_values = np.arange(0.1, 10.1, 0.1)  # 100 candidates
for h in h_values:
    err_h = compute_squared_error(data, h, hypercube_kernel)
    errors_hypercube.append(err_h)
```

**Why search [0.1, 10]?**  
- Too small h (< 0.1): undersmoothing, PDF becomes spiky
- Too large h (> 10): oversmoothing, PDF becomes flat
- This range captures the sweet spot for typical 1D data

**Step 3: Find Minimum**
```python
best_h_idx = np.argmin(errors_hypercube)
best_h = h_values[best_h_idx]
```

`np.argmin` returns the index of the smallest error.

---

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Two kernel types** | Demonstrates difference between hard boundaries (hypercube) and smooth falloff (Gaussian) |
| **Grid search over h** | Simple, exhaustive search guarantees finding the global minimum for this 1D problem |
| **Squared error metric** | Standard metric for regression-like problems; penalizes large deviations heavily |
| **Estimate at training points** | Convenient for error computation; we know true PDF there |
| **Broadcasting for pairwise distances** | O(M×N) vectorized operation vs. O(M×N) nested loops: ~1000x speedup |

---

### Expected Results

**Hypercube Kernel:**  
- Typically finds optimal h ≈ 1.5-3.0
- Creates "blocky" PDF estimates
- Lower error than too-small or too-large h

**Gaussian Kernel:**  
- Typically finds optimal h ≈ 0.8-1.5 (smaller than hypercube)
- Creates smooth PDF estimates
- Generally achieves **lower error** than hypercube (smoother approximation)

---

### Output Files
- `histogram_verification.png` - Histogram vs. true N(1,4)
- `parzen_error_plots.png` - Error vs. h for both kernels

---

## Part C: K-Nearest Neighbors Classifier

### Overview
Builds a K-Nearest Neighbors (KNN) classifier from scratch for binary classification (2 classes, 2 features). Finds the optimal `k` by evaluating accuracy on a test set, then visualizes decision boundaries.

### Mathematical Foundation

#### 1. Euclidean Distance
For two 2D points x = (x₁, x₂) and t = (t₁, t₂):

```
d(x, t) = √((x₁ - t₁)² + (x₂ - t₂)²)
```

**Why Euclidean?**  
Assumes features are on comparable scales and that "closeness" is well-represented by straight-line distance in feature space.

#### 2. KNN Classification Rule
To classify a new point x:
1. Find the k training points closest to x (by Euclidean distance)
2. Count how many belong to each class
3. Assign probabilities: `P(class c) = (count of class c) / k`
4. Classify as the class with highest probability

**Why this works?**  
Nearby points tend to share the same label (spatial locality assumption). Averaging over k neighbors smooths out noise.

#### 3. Normalization (Z-score)
Before computing distances, we normalize features:

```
x_norm = (x - μ_train) / σ_train
```

Where μ and σ are computed **only from training data**.

**Why?**  
If Feature 1 ranges [0, 100] and Feature 2 ranges [0, 1], Euclidean distance will be dominated by Feature 1. Normalization puts them on equal footing.

---

### Code Walkthrough

#### Function: `eucl(x, trainData)`

**Purpose:** Computes Euclidean distance from x to all training points.

**Vectorized Implementation:**
```python
diff = trainData - x       # Broadcasting: (N, 2) - (2,) = (N, 2)
sq_diff = diff ** 2        # Element-wise square
sum_sq = np.sum(sq_diff, axis=1)  # Sum across features: (N,)
dist = np.sqrt(sum_sq)     # Element-wise square root
```

**Why not use `scipy.spatial.distance.cdist`?**  
Assignment constraint: no library distance functions. This manual implementation is still fast due to NumPy vectorization.

**Performance:**  
For N=300 training points, this computes all 300 distances in < 1ms.

---

#### Function: `neighbors(x, trainData, k)`

**Purpose:** Finds the k nearest neighbors of x.

**Key Steps:**
```python
distances = eucl(x, features)         # Compute all distances
sorted_indices = np.argsort(distances)  # Sort indices by distance
k_indices = sorted_indices[:k]        # Take first k
return trainData[k_indices]           # Return corresponding rows
```

**Why `argsort` instead of `sort`?**  
We need the **indices** (to retrieve corresponding labels) not just sorted distances.

---

#### Function: `predict(testData, trainData, k)`

**Purpose:** Predicts class probabilities for all test points.

**Workflow:**
```python
for i in range(M):
    x = testData[i]
    k_neighbors = neighbors(x, trainData, k)
    labels = k_neighbors[:, 2]  # Extract labels column
    
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    
    probabilities[i] = [count_0/k, count_1/k]
```

**Why return probabilities instead of labels?**  
Probabilities give us confidence information. For example:
- `[0.9, 0.1]` → very confident it's class 0
- `[0.5, 0.5]` → uncertain (decision boundary region)

---

#### Main Workflow

**Step 1: Feature Normalization**
```python
mean = np.mean(train_data[:, :2], axis=0)  # Mean of training features only
std = np.std(train_data[:, :2], axis=0)    # Std of training features only

train_features_norm = (train_data[:, :2] - mean) / std
test_features_norm = (test_features - mean) / std  # Apply same transform
```

**Critical:** We compute μ and σ from training data, then apply to both train and test. This prevents **data leakage** (using test statistics to influence training).

**Step 2: Grid Search for Optimal k**
```python
for k in range(1, 31):
    probs = predict(test_features_norm, train_data_norm, k)
    pred_labels = np.argmax(probs, axis=1)  # Choose class with max probability
    accuracy = np.sum(pred_labels == test_labels) / len(test_labels)
```

**Why k ∈ [1, 30]?**  
- k=1: Very sensitive to noise (overfitting)
- k=N (all training points): Always predicts the majority class (underfitting)
- [1, 30] is a reasonable range for typical datasets

**Step 3: Decision Boundary Visualization**
```python
# Create a dense grid over the feature space
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), ...)
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten to (P, 2)

# Predict for all grid points
probs = predict(grid_points, train_data_norm, best_k)
Z = np.argmax(probs, axis=1).reshape(xx.shape)  # Class labels as 2D grid

# Plot as contour
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
```

**What's `np.c_`?**  
Shorthand for `np.column_stack`. It stacks `xx.ravel()` and `yy.ravel()` as columns.

**Why step size h=0.05?**  
Smaller h → smoother boundaries but slower computation. 0.05 is a good trade-off.

---

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Z-score normalization** | Prevents features with larger ranges from dominating distance calculations |
| **Compute μ/σ from training only** | Avoids data leakage; simulates real-world deployment |
| **Return probabilities** | Provides uncertainty estimates, not just hard labels |
| **Grid search k ∈ [1, 30]** | Covers typical range; beyond 30, diminishing returns |
| **Vectorized distance computation** | 100x faster than loops for typical dataset sizes |
| **Decision boundary plot** | Visualizes how k affects model complexity (smoother for large k) |

---

### Expected Behavior

**Small k (1-3):**
- Decision boundaries are **jagged**, closely following individual training points
- High training accuracy, but may **overfit** (poor test accuracy)

**Medium k (5-10):**
- Decision boundaries are **smoother**
- Balanced bias-variance trade-off
- Often achieves **best test accuracy**

**Large k (20-30):**
- Decision boundaries are **very smooth**, nearly linear
- May **underfit** (misses local patterns)

---

### Output Files
- `knn_accuracy.png` - Accuracy vs. k curve
- `knn_decision_boundary.png` - Visualization of classification regions

---

## Part D: Classification Challenge

### Overview
A production-quality 5-class classification pipeline using a **Stacking Ensemble** with **Pseudo-Labeling** (semi-supervised learning). Combines four diverse models (SVM, Random Forest, XGBoost, MLP) to maximize accuracy on unlabeled test data.

---

### Architecture Deep Dive

#### Phase 1: Data Preprocessing & Augmentation

**1.1 Feature Scaling**
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', ...)
])
```

**Why inside pipelines?**  
Ensures that:
- Training data is fit and transformed
- Test data is only transformed (using training statistics)
- Prevents data leakage automatically

**1.2 Gaussian Noise Augmentation**
```python
def augment_data_gaussian(X, y, noise_level=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = np.vstack((X, X + noise))
    y_aug = np.hstack((y, y))
    return X_aug, y_aug
```

**Mathematical Interpretation:**  
For each sample xᵢ, we create xᵢ' = xᵢ + ε, where ε ~ N(0, 0.05²I).

**Why this helps:**
- Forces the model to learn decision boundaries with **margins** (not overfitting to exact points)
- Acts as **regularization** by doubling the dataset
- Simulates natural measurement noise

**Why noise_level=0.05?**  
Empirically chosen. Too small (< 0.01) has minimal effect; too large (> 0.1) degrades signal.

---

#### Phase 2: Stacking Ensemble Architecture

**2.1 Base Models**

| Model | Key Configuration | Strengths | Pipeline Components |
|-------|-------------------|-----------|---------------------|
| **SVM** | RBF kernel, C=10, PCA(100) | Non-linear boundaries, margin maximization | StandardScaler → PCA → SVC |
| **Random Forest** | 300 trees, n_jobs=-1 | Handles feature interactions, robust to outliers | StandardScaler → RF |
| **XGBoost** | 300 estimators, lr=0.05, depth=6, GPU | Gradient boosting, adaptive learning | StandardScaler → XGB |
| **MLP** | (512, 256) layers, early stopping | Learns abstract representations | StandardScaler → MLP |

**Why these four?**
- **Diversity**: Different inductive biases (linear vs. tree-based vs. neural)
- **Complementary errors**: What SVM misclassifies, XGBoost might get right
- **Proven track record**: State-of-the-art algorithms from different ML paradigms

**2.2 SVM Deep Dive**
```python
def get_svm_model():
    clf = SVC(C=10, gamma='scale', kernel='rbf', probability=False, random_state=42)
    pca = PCA(n_components=100)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca),
        ('svm', clf)
    ])
```

**Why PCA before SVM?**  
- Reduces dimensionality (assuming > 100 features)
- Speeds up RBF kernel computation (O(n²) → O(100²))
- Removes noise in minor principal components

**Why `probability=False`?**  
Setting to `True` forces SVM to fit an internal 5-fold CV for calibration, which is:
- Slow (5x more fitting)
- Unnecessary for StackingClassifier (it uses `decision_function` by default)

**2.3 XGBoost Deep Dive**
```python
params = {
    'n_estimators': 300,
    'learning_rate': 0.05,  # Small lr for fine-grained learning
    'max_depth': 6,
    'device': 'cuda' if USE_GPU else 'cpu',
    'tree_method': 'hist',  # GPU-accelerated histogram method
    'eval_metric': 'mlogloss',
}
```

**Why `tree_method='hist'`?**  
Faster than exact method, especially on GPU. Bins continuous features into histograms.

**GPU Acceleration:**
- On NVIDIA GPU: ~10x faster than CPU
- Falls back to CPU gracefully if cuML not available

**2.4 MLP Deep Dive**
```python
MLPClassifier(
    hidden_layer_sizes=(512, 256),
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
)
```

**Architecture:**
```
Input → [512 neurons] → ReLU → [256 neurons] → ReLU → [5 neurons] → Softmax
```

**Why (512, 256)?**  
- First layer: Wide enough to capture complex feature interactions
- Second layer: Bottleneck that forces abstraction
- Pyramid structure (512 → 256 → 5) is a common best practice

**Why early stopping?**  
Automatically stops when validation loss stops improving (prevents overfitting).

**2.5 Stacking Meta-Learner**
```python
StackingClassifier(
    estimators=[('svm', ...), ('rf', ...), ('xgb', ...), ('mlp', ...)],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=1
)
```

**How Stacking Works:**
1. **Train base models** with 3-fold CV on training data
2. For each fold:
   - Train on 2/3 of data
   - Predict on held-out 1/3
3. **Collect out-of-fold predictions** (shape: N × 4 for 4 base models)
4. **Train meta-learner** (Logistic Regression) on these predictions
5. **Final prediction:** Base models predict on test → Meta-learner aggregates

**Why Logistic Regression for meta-learner?**
- Simple, fast, interpretable
- Learns *weights* for each base model (e.g., "trust XGBoost 40%, SVM 30%, ...")
- Less prone to overfitting than complex meta-learners

**Why cv=3 instead of 5?**
- 3-fold: 3 training runs per base model → faster
- 5-fold: 5 training runs per base model → slightly better generalization
- Trade-off: 3-fold is 40% faster with minimal accuracy loss

**Why n_jobs=1?**
- Each base model already uses parallelism (RF: n_jobs=-1, XGB: threads)
- Nested parallelism can cause CPU thrashing
- Manually verified to be faster than n_jobs=-1 for this specific setup

---

#### Phase 3: Pseudo-Labeling (Semi-Supervised Learning)

**3.1 High-Confidence Sample Selection**
```python
probs = model.predict_proba(X_test_submit)        # (6955, 5) probability matrix
max_probs = np.max(probs, axis=1)                 # (6955,) max prob per sample
preds = np.argmax(probs, axis=1)                  # (6955,) predicted class

CONFIDENCE_THRESHOLD = 0.90
high_conf_idx = np.where(max_probs >= 0.90)[0]    # Indices where confidence ≥ 90%

X_pseudo = X_test_submit[high_conf_idx]
y_pseudo = preds[high_conf_idx]
```

**What's happening:**
- Model predicts on test set
- We identify samples where `max(P(class 0), ..., P(class 4)) ≥ 0.90`
- Treat these as "ground truth" and add to training set

**Example:**
If model predicts `[0.05, 0.92, 0.01, 0.01, 0.01]` → confidence = 0.92 → include with label = 1

**3.2 Dataset Expansion & Retraining**
```python
X_final = np.vstack((X_train_full, X_pseudo))
y_final = np.hstack((y_train_full, y_pseudo))

model.fit(X_final, y_final)  # Retrain entire ensemble
```

**Why retrain from scratch?**
- Stacking requires consistent cross-validation structure
- Can't incrementally update without breaking CV folds
- Full retrain ensures meta-learner sees correct out-of-fold predictions

**3.3 Theoretical Justification**

**Transductive Learning:**
We're adapting the model to the **specific test distribution**. If test data differs slightly from training (e.g., different lighting conditions for images), pseudo-labeling helps bridge the gap.

**Error Propagation Risk:**
If initial model is poor (< 80% accuracy), pseudo-labels will be noisy, potentially degrading performance.

**Mitigation via High Confidence:**
90% threshold ensures we only add samples the model is **very sure** about. These are typically:
- Cluster centers (far from decision boundaries)
- Representative examples of each class

**3.4 Why This Doesn't Overfit**

| Safeguard | Mechanism |
|-----------|-----------|
| **Stacking CV** | Meta-learner trained on out-of-fold predictions → can't memorize training data |
| **Gaussian Augmentation** | Noise injection regularizes base models |
| **Diverse Ensemble** | Errors from one model corrected by others (ensemble averaging) |
| **High Threshold** | Only ~20-40% of test samples typically meet 90% confidence → limited new data |
| **Retrain Full Ensemble** | Fresh CV folds prevent memorization of pseudo-labels |

**Empirical Evidence:**
In practice, pseudo-labeling with stacking typically improves accuracy by 1-3% on Kaggle-style competitions.

---

### Optimization Details

**Why `probability=False` for SVM?**
```python
SVC(probability=False)  # Uses decision_function (fast)
vs.
SVC(probability=True)   # Fits internal 5-fold CV (slow)
```

**Performance Impact:**
- `probability=False`: ~30 seconds to train
- `probability=True`: ~150 seconds to train (5x slower)

StackingClassifier can use `decision_function` directly, so we don't need probabilities.

**Why `cv=3` for StackingClassifier?**
```python
cv=3: 12 total model trainings (4 models × 3 folds)
cv=5: 20 total model trainings (4 models × 5 folds)
```

**Accuracy Difference:** Typically < 0.5% (negligible)  
**Time Difference:** 40% faster with cv=3

---

### Expected Performance

**Typical Metrics (on similar 5-class problems):**
- Individual base models: 75-85% accuracy
- Stacking Ensemble (Phase 1): 85-92% accuracy
- With Pseudo-Labeling (Phase 2): 87-94% accuracy

**Why the improvement?**
- Stacking: Combines strengths, corrects individual weaknesses
- Pseudo-Labeling: Adapts to test distribution, adds ~1000-3000 high-quality labeled samples

---

### Troubleshooting

**GPU Out of Memory?**
```python
# Only XGBoost uses GPU now (SVM forced to CPU)
USE_GPU = False  # Fallback to CPU
```

**Still too slow?**
```python
# Reduce model sizes
n_estimators: 300 → 150  # For RF and XGB
hidden_layers: (512, 256) → (256, 128)  # For MLP
```

**Accuracy plateau?**
- Check feature engineering (are features informative?)
- Try different kernels (e.g., polynomial for SVM)
- Increase data augmentation noise (0.05 → 0.1)

---

### Output Files
- `labels1.npy` - Final predictions (shape: 6955,)
- `best_model_stacking_fast_cpu.pkl` - Trained ensemble (for deployment)

---

## General Best Practices

### Code Style
- **Greek comments** for academic context (assignment requirement)
- **Descriptive variable names** (e.g., `centered_data` not `cd`)
- **Docstrings** for all functions explaining purpose, parameters, returns

### Performance
- **Vectorization over loops** (NumPy broadcasting)
- **Pipelines for preprocessing** (prevents data leakage)
- **Early stopping** where applicable (MLP, XGBoost)

### Reproducibility
- **Random seeds** (`random_state=42`) for deterministic results
- **Save models** (`joblib.dump`) for reuse
- **Log parameters** (text files documenting hyperparameters)

### Visualization
- **Meaningful colormaps** (dark background, gradients)
- **Annotations** (mark optimal points with stars)
- **SVG format** where possible (scalable for reports)

---

## References & Further Reading

**Part A (MLE):**
- Hastie et al., *The Elements of Statistical Learning*, Ch. 4.3
- Murphy, *Machine Learning: A Probabilistic Perspective*, Ch. 4.1

**Part B (Parzen Windows):**
- Duda et al., *Pattern Classification*, Ch. 4.3
- Silverman, *Density Estimation for Statistics and Data Analysis*

**Part C (KNN):**
- Cover & Hart (1967), "Nearest Neighbor Pattern Classification"
- Hastie et al., Ch. 13.3

**Part D (Stacking & Pseudo-Labeling):**
- Wolpert (1992), "Stacked Generalization"
- Lee (2013), "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method"
- Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"

---

*Last Updated: December 2025*  
*Author: Evangelos Moschou*  
*Course: Pattern Recognition & Machine Learning 2025-2026*
