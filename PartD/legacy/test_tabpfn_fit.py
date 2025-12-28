from tabpfn import TabPFNClassifier
import numpy as np

X = np.random.randn(10, 10).astype(np.float32)
y = np.random.randint(0, 2, 10)

print("Testing fit with v2 default...")
try:
    # Trying the v2 default specifically
    clf = TabPFNClassifier(device='cuda', n_estimators=1, model_path='tabpfn-v2-classifier.ckpt')
    clf.fit(X, y)
    print("FIT SUCCESS")
except Exception as e:
    print(f"FIT FAILED: {e}")
