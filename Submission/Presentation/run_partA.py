#!/usr/bin/env python3
"""
Part A: Maximum Likelihood Estimation for Gaussian Distributions
Generates 3D plots and extracts parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import json

# Load data
data = np.loadtxt('../../Datasets/dataset1.csv', delimiter=',', skiprows=1)
X = data[:, :2]  # Features
y = data[:, 2]   # Labels

# Separate by class
classes = [0, 1, 2]
results = {}

print("="*60)
print("PART A: Maximum Likelihood Estimation")
print("="*60)

for c in classes:
    X_c = X[y == c]
    N_c = len(X_c)
    
    # MLE estimates
    mu = np.mean(X_c, axis=0)
    Sigma = np.cov(X_c.T)
    
    results[f'class_{c}'] = {
        'mu': mu.tolist(),
        'Sigma': Sigma.tolist(),
        'N': int(N_c)
    }
    
    print(f"\nClass {c} (N={N_c}):")
    print(f"  μ = {mu}")
    print(f"  Σ = \n{Sigma}")

# Create 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_grid = np.linspace(x_min, x_max, 100)
y_grid = np.linspace(y_min, y_max, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_grid, Y_grid))

colors = ['red', 'blue', 'green']
labels_text = ['Class 0', 'Class 1', 'Class 2']

for c in classes:
    mu = np.array(results[f'class_{c}']['mu'])
    Sigma = np.array(results[f'class_{c}']['Sigma'])
    
    rv = multivariate_normal(mu, Sigma)
    Z = rv.pdf(pos)
    
    ax.plot_surface(X_grid, Y_grid, Z, alpha=0.5, color=colors[c], label=labels_text[c])

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_zlabel('Probability Density', fontsize=12)
ax.set_title('3D Gaussian Distributions (MLE)', fontsize=14, fontweight='bold')
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('../Presentation/plots/partA_3d_gaussians.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: partA_3d_gaussians.png")

# Create 2D contour plot
fig, ax = plt.subplots(figsize=(10, 8))

for c in classes:
    X_c = X[y == c]
    mu = np.array(results[f'class_{c}']['mu'])
    Sigma = np.array(results[f'class_{c}']['Sigma'])
    
    # Plot data points
    ax.scatter(X_c[:, 0], X_c[:, 1], alpha=0.6, s=30, c=colors[c], label=labels_text[c])
    
    # Plot contours
    rv = multivariate_normal(mu, Sigma)
    Z = rv.pdf(pos)
    ax.contour(X_grid, Y_grid, Z, levels=5, colors=colors[c], alpha=0.6, linewidths=2)
    
    # Plot mean
    ax.plot(mu[0], mu[1], 'x', markersize=15, markeredgewidth=3, color=colors[c])

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('2D Data Points with MLE Gaussian Contours', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../Presentation/plots/partA_2d_contours.png', dpi=150, bbox_inches='tight')
print("✓ Saved: partA_2d_contours.png")

# Save results to JSON
with open('../Presentation/results_partA.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved: results_partA.json")
print(f"\n{'='*60}")
print("Part A Complete!")
print(f"{'='*60}")
