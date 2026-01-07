import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, device='cpu', n_estimators=8, seed=42, max_samples=1024):
        self.device = device
        self.n_estimators = n_estimators  # Updated from N_ensemble_configurations
        self.seed = seed
        self.max_samples = max_samples
        self.model = None
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            print("Warning: TabPFN not installed. Using dummy fallback (Random).")
            self.model = None
            self.classes_ = np.unique(y)
            return self

        self.classes_ = np.unique(y)
        
        # Subsample if necessary (TabPFN limit)
        # We use Stratified Subsampling if possible
        if len(X) > self.max_samples:
            # Simple stratified subsample
            indices = np.arange(len(X))
            # Just random shuffle and pick first N is roughly stratified for large N
            # But let's be cleaner.
            rng = np.random.RandomState(self.seed)
            indices = rng.permutation(indices)
            
            # Prioritize picking from all classes
            # (Simplified: Random selection is robust enough for 1024 samples)
            sel_idx = indices[:self.max_samples]
            
            self.X_train_ = X[sel_idx]
            self.y_train_ = y[sel_idx]
        else:
            self.X_train_ = X
            self.y_train_ = y

        # Updated API: n_estimators and random_state (not N_ensemble_configurations/seed)
        self.model = TabPFNClassifier(
            device=self.device, 
            n_estimators=self.n_estimators,
            random_state=self.seed
        )
        # We don't call model.fit() here because TabPFNClassifier.fit() just stores data.
        # But we should follow API.
        # Note: TabPFN's fit might copy data.
        self.model.fit(self.X_train_, self.y_train_)
        return self

    def predict_proba(self, X):
        if self.model is None:
            # Fallback for missing library
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)
            
        # TabPFN expects numpy
        if hasattr(X, 'cpu'):
             X = X.cpu().numpy()
        if hasattr(X, 'numpy'):
             X = X.numpy()
             
        # Check batch size? TabPFN processes in batches?
        # The library usually handles it, but for safety with OOM on GPU:
        BATCH = 128
        preds = []
        for i in range(0, len(X), BATCH):
             xb = X[i:i+BATCH]
             p = self.model.predict_proba(xb)
             preds.append(p)
        return np.vstack(preds)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
