# Part D: Classification Challenge - Step-by-Step Guide

## Objective
Build a classification model to predict labels for unlabeled test data. This is a competition - best performing teams receive bonus points.

---

## Dataset Information
- **Training file:** `../Datasets/datasetTV.csv`
  - 8743 samples × 225 columns
  - First 224 columns: features
  - Last column (225th): labels (1, 2, 3, 4, or 5)
  - Note: 5-class classification problem
  
- **Test file:** `../Datasets/datasetTest.csv`
  - 6955 samples × 224 columns
  - Features only (no labels provided)
  - You must predict labels for these samples

---

## Output Requirements
- **File name:** `labelsX.npy` (where X is your team number, e.g., labels1.npy)
- **Format:** numpy array saved with np.save()
- **Shape:** (6955,) - one predicted label per test sample
- **Label values:** must be 1, 2, 3, 4, or 5 (same as training labels)
- **Verification:** file must load correctly with np.load()

---

## Step-by-Step Instructions

### STEP 1: Load and Explore Training Data
1. Load `datasetTV.csv` using numpy or pandas
2. Separate features (first 224 columns) from labels (last column)
3. Print basic information:
   - Number of samples and features
   - Label distribution (count per class)
   - Check for class imbalance
4. Check for missing values or anomalies
5. Compute basic statistics for features (mean, std, min, max)

### STEP 2: Load Test Data
1. Load `datasetTest.csv`
2. Verify it has 6955 samples and 224 features
3. Ensure feature columns match the training data structure

### STEP 3: Data Preprocessing (Choose Appropriate Methods)
Consider the following preprocessing steps:

**Handling Missing Values:**
- Check if any features have missing values
- If so, impute with mean, median, or other strategy

**Feature Scaling:**
- Standardization: transform to zero mean and unit variance
- Normalization: scale to [0,1] range
- Important: fit the scaler on training data only, then apply to both train and test

**Dimensionality Reduction (Optional):**
- Apply PCA if you want to reduce the 224 features
- Choose number of components based on explained variance (e.g., 95%)
- Fit on training data, transform both train and test

**Feature Selection (Optional):**
- Remove low-variance features
- Use correlation analysis to remove redundant features
- Apply statistical tests to select most informative features

### STEP 4: Split Training Data for Validation
1. Split training data into training subset and validation subset
2. Recommended split: 80% train, 20% validation
3. Use stratified splitting to maintain class distribution
4. This allows you to estimate model performance before final submission

### STEP 5: Choose and Train Classification Model(s)
You are free to use any method. Suggested approaches:

**Option A - Support Vector Machine (SVM):**
- Works well for high-dimensional data
- Try different kernels: RBF, linear, polynomial
- Tune hyperparameters: C (regularization), gamma (kernel parameter)

**Option B - Random Forest:**
- Ensemble of decision trees
- Robust to overfitting
- Tune: number of trees, max depth, min samples per leaf

**Option C - Neural Network / MLP:**
- Can capture complex patterns
- Define architecture: number of layers, neurons per layer
- Tune: learning rate, regularization, activation functions

**Option D - Gradient Boosting (XGBoost, LightGBM):**
- Often achieves state-of-the-art on tabular data
- Tune: learning rate, max depth, number of estimators

**Option E - Ensemble:**
- Combine multiple different models
- Use voting or averaging for final prediction

### STEP 6: Hyperparameter Tuning
1. Use cross-validation (e.g., 5-fold) on the training subset
2. Try different hyperparameter combinations
3. Use grid search or random search for systematic exploration
4. Select the hyperparameters that give best validation performance
5. Consider using accuracy as the metric (or F1-score for imbalanced classes)

### STEP 7: Evaluate on Validation Set
1. Train your final model with chosen hyperparameters on training subset
2. Predict on validation subset
3. Calculate metrics:
   - Overall accuracy
   - Per-class precision, recall, F1-score
   - Confusion matrix
4. Analyze which classes are hardest to predict
5. If performance is poor, go back to Steps 3-6 and try different approaches

### STEP 8: Train Final Model
1. Once satisfied with validation performance:
2. Retrain the model on the ENTIRE training dataset (all 8743 samples)
3. Use the same hyperparameters selected during tuning
4. This maximizes the amount of data the model learns from

### STEP 9: Generate Predictions for Test Set
1. Apply the trained model to the test data (all 6955 samples)
2. Get predicted labels for each sample
3. Ensure predictions are integers in {1, 2, 3, 4, 5}
4. Verify the shape is (6955,)

### STEP 10: Save Predictions
1. Create a numpy array from your predictions
2. Save using numpy's save function with filename `labelsX.npy`
3. Test that the file loads correctly with np.load()
4. Verify shape is (6955,) and values are in {1, 2, 3, 4, 5}

---

## Suggested Workflow Summary
1. Explore data → understand the problem
2. Preprocess → clean and prepare data
3. Baseline model → get initial results quickly
4. Iterate → try different models, tune hyperparameters
5. Validate → check performance on held-out data
6. Finalize → train on full data, predict test set
7. Submit → save predictions in correct format

---

## Tips for Best Performance
- **Start simple:** Begin with a basic model to establish baseline
- **Feature engineering:** Create new features from existing ones if helpful
- **Cross-validation:** Always use CV to avoid overfitting
- **Error analysis:** Look at misclassified samples to understand weaknesses
- **Ensemble methods:** Combining models often improves performance
- **Don't overfit:** Monitor training vs validation performance gap
- **Time management:** Allow time for experimentation

---

## Expected Outputs
1. **Trained classification model** (your choice of algorithm)
2. **Validation accuracy** (or other metrics)
3. **labelsX.npy** file with 6955 predicted labels
4. **Documentation** of your approach for the presentation

---

## Validation Checks
- predictions array has shape (6955,)
- all values are in {1, 2, 3, 4, 5}
- file loads correctly with np.load()
- no NaN or missing predictions
- validation accuracy is reasonable (significantly above random chance ~20%)
