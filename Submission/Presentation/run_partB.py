#!/usr/bin/env python3
"""
Part B: Parzen Windows for Density Estimation  
Tests both Hypercube and Gaussian kernels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json

# Load data
data = np.loadtxt('../../Datasets/dataset2.csv', delimiter=',', skiprows=1)
X = data.flatten()
N = len(X)

# True distribution: N(1, 4) means μ=1, σ²=4, σ=2
true_mu = 1
true_sigma = 2

print("="*60)
print("PART B: Parzen Window Density Estimation")
print("="*60)
print(f"Dataset: {N} samples")
print(f"Assumed distribution: N({true_mu}, {true_sigma**2})")

# Verify with histogram
plt.figure(figsize=(10, 6))
plt.hist(X, bins=30, density=True, alpha=0.6, color='cyan', edgecolor='black', label='Data Histogram')
x_range = np.linspace(X.min()-1, X.max()+1, 1000)
true_pdf = norm.pdf(x_range, true_mu, true_sigma)
plt.plot(x_range, true_pdf, 'r-', linewidth=2, label=f'True N({true_mu}, {true_sigma**2})')
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Data Histogram vs True Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../Presentation/plots/partB_histogram.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: partB_histogram.png")

# Parzen Window Functions
def hypercube_kernel(u):
    return np.where(np.abs(u) <= 0.5, 1.0, 0.0)

def gaussian_kernel(u):
    return (1/np.sqrt(2*np.pi)) * np.exp(-u**2/2)

def parzen_estimate(x_test, X_train, h, kernel_func):
    """Estimate PDF using Parzen windows"""
    n = len(X_train)
    density = np.zeros_like(x_test)
    
    for i, x in enumerate(x_test):
        u = (x - X_train) / h
        density[i] = np.sum(kernel_func(u)) / (n * h)
    
    return density

# Grid search for optimal h
h_values = np.arange(0.1, 10.1, 0.1)
results = {}

for kernel_name, kernel_func in [('hypercube', hypercube_kernel), ('gaussian', gaussian_kernel)]:
    mse_values = []
    
    for h in h_values:
        # Estimate PDF at data points
        estimated_pdf = parzen_estimate(X, X, h, kernel_func)
        
        # True PDF at data points
        true_pdf_at_points = norm.pdf(X, true_mu, true_sigma)
        
        # MSE
        mse = np.mean((true_pdf_at_points - estimated_pdf)**2)
        mse_values.append(mse)
    
    mse_values = np.array(mse_values)
    best_idx = np.argmin(mse_values)
    best_h = h_values[best_idx]
    best_mse = mse_values[best_idx]
    
    results[kernel_name] = {
        'best_h': float(best_h),
        'best_mse': float(best_mse),
        'h_values': h_values.tolist(),
        'mse_values': mse_values.tolist()
    }
    
    print(f"\n{kernel_name.capitalize()} Kernel:")
    print(f"  Optimal h = {best_h:.2f}")
    print(f"  MSE = {best_mse:.6f}")
    
    # Plot MSE vs h
    plt.figure(figsize=(10, 6))
    plt.plot(h_values, mse_values, 'b-', linewidth=2)
    plt.axvline(best_h, color='r', linestyle='--', linewidth=2, label=f'Optimal h = {best_h:.2f}')
    plt.scatter([best_h], [best_mse], color='r', s=100, zorder=5)
    plt.xlabel('Bandwidth h', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title(f'MSE vs Bandwidth ({kernel_name.capitalize()} Kernel)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../Presentation/plots/partB_{kernel_name}_mse.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: partB_{kernel_name}_mse.png")
    
    # Plot estimated PDFs with optimal h
    x_test = np.linspace(X.min()-2, X.max()+2, 1000)
    estimated = parzen_estimate(x_test, X, best_h, kernel_func)
    true_pdf_test = norm.pdf(x_test, true_mu, true_sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, true_pdf_test, 'r-', linewidth=2, label='True N(1,4)')
    plt.plot(x_test, estimated, 'b-', linewidth=2, label=f'Parzen (h={best_h:.2f})')
    plt.hist(X, bins=30, density=True, alpha=0.3, color='cyan', edgecolor='black', label='Data')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Parzen Window Estimation ({kernel_name.capitalize()}, h={best_h:.2f})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../Presentation/plots/partB_{kernel_name}_pdf.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: partB_{kernel_name}_pdf.png")

# Save results
with open('../Presentation/results_partB.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved: results_partB.json")
print(f"\n{'='*60}")
print("Part B Complete!")
print(f"{'='*60}")
