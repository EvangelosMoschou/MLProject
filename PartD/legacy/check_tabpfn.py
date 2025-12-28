from tabpfn import TabPFNClassifier
import sys

print("Testing default (gated)...")
try:
    clf = TabPFNClassifier(device='cuda', n_estimators=1)
    print("Default init success!")
except Exception as e:
    print(f"Default failed: {e}")

print("\nTesting 'tabpfn-v1'...")
try:
    # Attempting to guess the old model name
    clf = TabPFNClassifier(device='cuda', n_estimators=1, model_path='prior/tabpfn-v1')
    print("v1 init success!")
except Exception as e:
    print(f"v1 failed: {e}")
