#!/usr/bin/env python3
"""
Part C: k-Nearest Neighbors Classifier
Implements k-NN from scratch and finds optimal k
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json

# Load data
train_data = np.loadtxt('../../Datasets/dataset3.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('../../Datasets/testset.csv', delimiter=',', skiprows=1)

X_train = train_data[:, :2]
y_train = train_data[:, 2]
X_test = test_data[:, :2]
y_test = test_data[:, 2]

print("="*60)
print("PART C: k-Nearest Neighbors Classifier")
print("="*60)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Classes: {np.unique(y_train)}")

# Implement k-NN functions
def eucl(x, trainData):
    """Calculate Euclidean distance from x to all points in trainData"""
    distances = np.sqrt(np.sum((trainData - x)**2, axis=1))
    return distances

def neighbors(x, trainData, k):
    """Find k nearest neighbors"""
    distances = eucl(x, trainData)
    indices = np.argsort(distances)[:k]
    return trainData[indices]

def predict(testData, trainData, trainLabels, k):
    """Predict probabilities for test data"""
    n_test = len(testData)
    n_classes = len(np.unique(trainLabels))
    probabilities = np.zeros((n_test, n_classes))
    predictions = np.zeros(n_test)
    
    for i in range(n_test):
        # Find k nearest neighbors
        distances = eucl(testData[i], trainData)
        neighbor_indices = np.argsort(distances)[:k]
        neighbor_labels = trainLabels[neighbor_indices]
        
        # Calculate probabilities
        for c in range(n_classes):
            probabilities[i, c] = np.sum(neighbor_labels == c) / k
        
        # Predict class
        predictions[i] = np.argmax(probabilities[i])
    
    return predictions, probabilities

# Find optimal k
k_values = range(1, 31)
accuracies = []

for k in k_values:
    preds, _ = predict(X_test, X_train, y_train, k)
    accuracy = np.mean(preds == y_test)
    accuracies.append(accuracy)

accuracies = np.array(accuracies)
best_idx = np.argmax(accuracies)
best_k = k_values[best_idx]
best_accuracy = accuracies[best_idx]

print(f"\nOptimal k = {best_k}")
print(f"Best Accuracy = {best_accuracy*100:.2f}%")

results = {
    'best_k': int(best_k),
    'best_accuracy': float(best_accuracy),
    'k_values': list(k_values),
    'accuracies': accuracies.tolist()
}

# Plot Accuracy vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies * 100, 'b-', linewidth=2, marker='o', markersize=4)
plt.axvline(best_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k = {best_k}')
plt.scatter([best_k], [best_accuracy*100], color='r', s=150, zorder=5)
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('k-NN Classifier: Accuracy vs k', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 31, 5))
plt.tight_layout()
plt.savefig('../Presentation/plots/partC_accuracy_vs_k.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: partC_accuracy_vs_k.png")

# Visualize decision boundaries with optimal k
def plot_decision_boundary(X_train, y_train, k, filename):
    h = 0.1  # step size in meshgrid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z, _ = predict(mesh_points, X_train, y_train, k)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['red', 'blue']
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light, levels=1)
    
    # Plot training points
    for c in [0, 1]:
        mask = y_train == c
        plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                   c=cmap_bold[int(c)], s=60, edgecolors='k', alpha=0.7,
                   label=f'Class {int(c)}')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(f'k-NN Decision Boundaries (k={k})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename.split('/')[-1]}")

plot_decision_boundary(X_train, y_train, best_k, '../Presentation/plots/partC_decision_boundary.png')

# Also create visualization of training data
plt.figure(figsize=(10, 8))
cmap_bold = ['red', 'blue']
for c in [0, 1]:
    mask = y_train == c
    plt.scatter(X_train[mask, 0], X_train[mask, 1], 
               c=cmap_bold[int(c)], s=60, edgecolors='k', alpha=0.7,
               label=f'Class {int(c)}')

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Training Data Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../Presentation/plots/partC_training_data.png', dpi=150, bbox_inches='tight')
print("✓ Saved: partC_training_data.png")

# Save results
with open('../Presentation/results_partC.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved: results_partC.json")
print(f"\n{'='*60}")
print("Part C Complete!")
print(f"{'='*60}")
