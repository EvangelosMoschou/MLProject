"""
Optimized SVM for Sigma-Omega Protocol.

Architecture:
1. Noise Filtration (SelectKBest) -> Remove irrelevant features.
2. Kernel Approximation (Nystroem) -> Project to high-dim space efficiently.
3. Fast Solver (LinearSVC) -> Converge quickly on the projected manifold.
4. Probability Calibration (CalibratedClassifierCV) -> Robust probabilities.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC
import numpy as np

class SVMWrapper(BaseEstimator, ClassifierMixin):
    """
    Advanced SVM Wrapper.
    
    Features:
    - Feature Selection (Filter 50%+ noise)
    - Nystroem Kernel Approximation (RBF mapping)
    - Linear Solver (Speed)
    - Calibrated Probabilities
    """
    
    _is_svm = True  # Flag for stacking.py

    def __init__(self, n_components=400, k_best=100, random_state=42):
        self.n_components = n_components
        self.k_best = k_best
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        
        # Dynamic K selection: at least 50% or capped at 128?
        # User rule: min(n_features, 128)
        k = min(n_features, 128)
        if self.k_best:
             # Allow manual override if specified, else use logic
             k = min(n_features, self.k_best)
        
        # Base Pipeline: Select -> Scale -> Project -> Solve
        # We use SGDClassifier(loss='hinge') which is basically Linear SVM but supports incremental learning if needed
        # Or LinearSVC. LinearSVC is generally more stable for small-medium batches.
        # User requested: "SGDClassifier(loss='hinge') or LinearSVC"
        # LinearSVC(dual=False) is preferred for n_samples > n_features usually.
        
        base_svm = LinearSVC(
            penalty='l2',
            loss='squared_hinge',
            dual=False,
            C=0.5,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=2000
        )
        
        # NOTE: StandardScaler is crucial before Nystroem/SVM
        pipeline = Pipeline([
            ('selector', SelectKBest(score_func=f_classif, k=k)),
            ('scaler', StandardScaler()),
            ('nystroem', Nystroem(
                kernel='rbf',
                gamma=None, # defaults to 1/n_features
                n_components=self.n_components,
                random_state=self.random_state
            )),
            ('svm', base_svm)
        ])
        
        # Wrap in CalibratedClassifierCV for probabilities
        # ensemble=False is slightly faster but True (default) is better calibration
        self.model_ = CalibratedClassifierCV(
            estimator=pipeline,
            method='sigmoid',
            cv=5, 
            n_jobs=-1 
        )
        
        # Fit with sample weights support
        # CalibratedClassifierCV passes sample_weight to the estimator's fit method
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def get_svm(random_state=42):
    """Factory for the optimized SVM."""
    return SVMWrapper(n_components=400, k_best=64, random_state=random_state)
