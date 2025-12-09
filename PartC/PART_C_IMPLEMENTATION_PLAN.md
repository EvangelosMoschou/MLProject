# Part C: K-Nearest Neighbors Classifier - Step-by-Step Guide

## Objective
Build a K-Nearest Neighbors (KNN) classifier from scratch, find the optimal k value, and visualize the decision boundaries.

---

## Dataset Information
- **Training file:** `../Datasets/dataset3.csv`
  - 50 samples × 3 columns (2 features + 1 label)
  - Binary classification: labels are 0 or 1
- **Test file:** `../Datasets/testset.csv`
  - 50 samples × 3 columns (2 features + 1 label)
  - Same format as training data

---

## Restrictions
- **NO library distance functions allowed** (no scipy.distance, no sklearn)
- Basic numpy operations are permitted (sum, sqrt, array operations, sorting)
- matplotlib's contourf is allowed for decision boundary visualization

---

## Step-by-Step Instructions

### STEP 1: Load Both Datasets
1. Load `dataset3.csv` as the training set
2. Load `testset.csv` as the test set
3. For each dataset:
   - Extract features (first two columns)
   - Extract labels (third column)
4. Verify both datasets have 50 samples each
5. Verify labels are binary (0 or 1)

### STEP 2: Implement Euclidean Distance Function `eucl(x, trainData)`
Create a function with the following specifications:
1. **Input:**
   - x: a single 2D point (array of 2 values)
   - trainData: array of N training points (N × 2 matrix)
2. **Process:**
   - For each training point, compute the Euclidean distance to x
   - Distance formula: d = √((x₁ - t₁)² + (x₂ - t₂)²)
   - Do NOT use any library distance functions
   - Manual implementation: square the differences, sum them, take square root
3. **Output:**
   - Array of N distances (one distance per training point)

### STEP 3: Implement Neighbors Function `neighbors(x, trainData, k)`
Create a function with the following specifications:
1. **Input:**
   - x: a single 2D point to classify
   - trainData: the full training dataset including features AND labels
   - k: number of neighbors to return
2. **Process:**
   - Call the eucl function to get distances from x to all training points
   - Get the indices that would sort the distances in ASCENDING order (smallest first)
   - Note: The assignment says "descending" but for finding NEAREST neighbors, you want the smallest distances first
   - Select the first k indices (the k nearest)
   - Extract the corresponding k training points from trainData
3. **Output:**
   - The k nearest training points (including their labels)

### STEP 4: Implement Predict Function `predict(testData, trainData, k)`
Create a function with the following specifications:
1. **Input:**
   - testData: features of test samples (M × 2 matrix)
   - trainData: full training data including labels (N × 3 matrix)
   - k: number of neighbors to use
2. **Process:**
   For each test point:
   - Call the neighbors function to get k nearest neighbors
   - Extract the labels from these k neighbors
   - Count how many neighbors belong to class 0
   - Count how many neighbors belong to class 1
   - Compute probability for class 0: P(class=0) = count_0 / k
   - Compute probability for class 1: P(class=1) = count_1 / k
   - These probabilities should sum to 1
3. **Output:**
   - Array of shape (M, 2) containing [P(class=0), P(class=1)] for each test point

### STEP 5: Find Optimal k Value
1. Create a range of k values from 1 to 30
2. For each k value:
   - Run the predict function on all test points
   - For each test point, the predicted class is the one with higher probability
   - Compare predicted classes with true test labels
   - Calculate accuracy = (number correct) / (total test samples)
   - Store this accuracy
3. Find the k value that gives the highest accuracy
4. Print the optimal k and its accuracy

### STEP 6: Create Accuracy vs k Plot
1. X-axis: k values (1 to 30)
2. Y-axis: accuracy values
3. Plot the accuracy curve
4. Mark the optimal k value with a vertical line or special marker
5. Add title: "KNN Accuracy vs Number of Neighbors (k)"
6. Add axis labels: "k" and "Accuracy"
7. Add grid for readability
8. Save as an image file

### STEP 7: Visualize Decision Boundaries
Using the optimal k value found in Step 5:

1. **Create a meshgrid:**
   - Find min and max values for both features in the training data
   - Add a small margin to the range
   - Create a fine grid of points covering this range (e.g., 100×100 points)

2. **Classify all grid points:**
   - For each point on the grid, use the predict function with optimal k
   - Determine the predicted class (0 or 1) for each grid point
   - Reshape the predictions to match the grid dimensions

3. **Create the visualization:**
   - Use matplotlib's contourf to fill regions based on predicted class
   - Use different colors for class 0 and class 1 regions
   - Overlay the training data points on top
   - Use different markers/colors for class 0 and class 1 training points
   - Add title: "KNN Decision Boundaries (k = optimal_value)"
   - Add legend
   - Save as an image file

---

## Expected Outputs
1. **eucl function** - Returns array of distances
2. **neighbors function** - Returns k nearest training samples
3. **predict function** - Returns probability pairs for each test sample
4. **Best k value** and its accuracy (printed)
5. **Accuracy vs k plot** (saved image)
6. **Decision boundary plot** (saved image)

---

## Validation Checks
- eucl function should return positive distances only
- For k=1, accuracy might fluctuate; for larger k, it typically stabilizes
- Probabilities from predict should always sum to 1.0
- Decision boundaries should show reasonable class separation regions
- Training points should fall within their respective predicted regions (mostly)
- Be careful with the ascending/descending order - you want the NEAREST neighbors

---

## Common Considerations
- If there's a tie in voting (equal probabilities), you can choose either class or use some tiebreaker
- Very small k (like 1) may overfit; very large k (approaching N) may underfit
- The decision boundary becomes smoother as k increases
