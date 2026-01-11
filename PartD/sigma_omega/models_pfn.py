"""
TabPFN v2.5 Wrapper για το Sigma-Omega Protocol.

Ο TabPFNClassifier είναι ένα Foundation Model για tabular data που εκτελεί
in-context learning. Η έκδοση 2.5 υποστηρίζει:
- Έως 30,000+ δείγματα εκπαίδευσης (χωρίς το όριο 1024 της v1)
- Έως 500-1000 χαρακτηριστικά (με εξειδικευμένα checkpoints)
- Πολλαπλά checkpoints για διαφορετικές περιπτώσεις χρήσης

Σημείωση: Απαιτεί GPU με >= 8GB VRAM για μεγάλα datasets.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper για τον TabPFNClassifier v2.5.
    
    Παράμετροι:
    -----------
    device : str, default='cuda'
        Συσκευή εκτέλεσης ('cuda' ή 'cpu'). GPU συνιστάται έντονα.
    n_estimators : int, default=32
        Αριθμός ensemble members για post-hoc ensembling.
        Υψηλότερες τιμές (32-64) αυξάνουν την ακρίβεια αλλά και τον χρόνο.
    random_state : int, default=42
        Seed για αναπαραγωγιμότητα.
    inference_precision : str, default='autocast'
        Precision για inference ('autocast', 'float32', 'float16').
        'autocast' είναι η καλύτερη επιλογή για ταχύτητα/ακρίβεια.
    """

    def __init__(
        self,
        device: str = 'cuda',
        n_estimators: int = 32,
        random_state: int = 42,
        inference_precision: str = 'autocast',
    ):
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.inference_precision = inference_precision
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Εκπαίδευση του TabPFN στα δεδομένα.
        
        Σημείωση: Ο TabPFN δεν εκπαιδεύεται πραγματικά - απλώς αποθηκεύει
        τα δεδομένα για in-context learning κατά το inference.
        
        Αν υπάρχει _raw_train attribute, χρησιμοποιεί raw δεδομένα (χωρίς Razor).
        """
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            print("[TabPFN] WARNING: tabpfn not installed. Using random fallback.")
            self.model_ = None
            self.classes_ = np.unique(y)
            return self

        self.classes_ = np.unique(y)

        # Σημείωση: Δεν χρησιμοποιούμε _raw_train για training γιατί το diffusion
        # augmentation αλλάζει το μέγεθος του training set. Χρησιμοποιούμε τα 
        # δεδομένα που μας δόθηκαν (ίσως Razor-filtered + augmented).
        # Το _raw_test χρησιμοποιείται μόνο για test predictions.
        X_use = X
            
        # Δημιουργία TabPFNClassifier με v2.5 defaults
        self.model_ = TabPFNClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            inference_precision=self.inference_precision,
        )

        # Μετατροπή σε numpy αν είναι tensor
        X_np = self._to_numpy(X_use)
        y_np = self._to_numpy(y)

        # Fit (αποθηκεύει δεδομένα για in-context learning)
        self.model_.fit(X_np, y_np)
        
        n_samples, n_features = X_np.shape
        print(f"[TabPFN] Fitted with {n_samples} samples, {n_features} features, "
              f"{self.n_estimators} estimators on {self.device}")

        return self

    def predict_proba(self, X):
        """
        Πρόβλεψη πιθανοτήτων για κάθε κλάση.
        """
        if self.model_ is None:
            # Fallback: τυχαίες πιθανότητες
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)

        # Σημείωση: Πρέπει να χρησιμοποιήσουμε τα ίδια features με το training.
        # Δεν χρησιμοποιούμε _raw_test γιατί η stacking προσθέτει manifold/DAE features.
        X_np = self._to_numpy(X)
        
        # TabPFN v2.5 χειρίζεται batching εσωτερικά
        probs = self.model_.predict_proba(X_np)
        
        return probs

    def predict(self, X):
        """
        Πρόβλεψη ετικετών.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    @staticmethod
    def _to_numpy(arr):
        """Μετατροπή tensor/dataframe σε numpy array."""
        if hasattr(arr, 'cpu'):
            arr = arr.cpu()
        if hasattr(arr, 'numpy'):
            arr = arr.numpy()
        if hasattr(arr, 'values'):
            arr = arr.values
        return np.asarray(arr)
