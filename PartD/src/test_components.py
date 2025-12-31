import torch
import numpy as np
import sys
import os

# Add parent dir to path to find src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tabr import TabR, TabRClassifier
from src.generative import GenerativeADClassifier

def test_tabr():
    print("Testing TabR...")
    input_dim = 10
    num_classes = 3
    N = 50
    
    # Mock Data
    X = np.random.randn(N, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, N)
    
    model = TabRClassifier(input_dim, num_classes, device='cpu', k_neighbors=5)
    model.fit(X, y, epochs=2, batch_size=10)
    
    probs = model.predict_proba(X[:5])
    print(f"TabR Probs Shape: {probs.shape}")
    assert probs.shape == (5, num_classes)
    print("TabR Test Passed!")

def test_generative():
    print("\nTesting Generative Classifier...")
    input_dim = 10
    num_classes = 3
    N = 60
    
    X = np.random.randn(N, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, N)
    
    model = GenerativeADClassifier(input_dim, num_classes, device='cpu', hidden_dim=16, bottleneck_dim=4)
    model.fit(X, y, epochs=2, batch_size=10)
    
    probs = model.predict_proba(X[:5])
    print(f"Generative Probs Shape: {probs.shape}")
    assert probs.shape == (5, num_classes)
    
    # Test Refinement
    # print("Testing Refinement...")
    # refined_energy = model.refine_sample(X[0], 0, steps=2)
    # print(f"Refined Energy: {refined_energy}")
    
    print("Generative Test Passed!")

if __name__ == '__main__':
    test_tabr()
    test_generative()
