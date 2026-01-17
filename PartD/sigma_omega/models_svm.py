"""
SVM Wrapper for Sigma-Omega Protocol.

Provides an sklearn-compatible SVM classifier with built-in StandardScaler.
Optimized for rotated feature spaces (PCA, ICA).
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


class SVMWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for SVM with StandardScaler.
    
    Marked with _is_svm=True for downstream feature handling
    (excludes manifold features in stacking.py).
    """
    
    _is_svm = True  # Flag for stacking.py to detect SVM
    
    def __init__(self, C=1.0, gamma='scale', kernel='rbf', random_state=42):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        self.pipeline_ = None
        self.classes_ = None
    
    def fit(self, X, y, sample_weight=None):
        """Fit the SVM pipeline."""
        self.classes_ = np.unique(y)
        
        self.pipeline_ = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                C=self.C,
                gamma=self.gamma,
                kernel=self.kernel,
                probability=True,  # Required for predict_proba
                random_state=self.random_state,
                class_weight='balanced',  # Handle class imbalance
            ))
        ])
        
        # SVC doesn't support sample_weight in pipeline, fit directly
        if sample_weight is not None:
            X_scaled = self.pipeline_.named_steps['scaler'].fit_transform(X)
            self.pipeline_.named_steps['svm'].fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.pipeline_.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.pipeline_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.pipeline_.predict_proba(X)


def get_svm(C=1.0, gamma='scale', kernel='rbf', random_state=42):
    """Factory function to create SVM wrapper."""
    return SVMWrapper(C=C, gamma=gamma, kernel=kernel, random_state=random_state)
