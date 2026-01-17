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
import torch
import gc
from sklearn.base import BaseEstimator, ClassifierMixin
from . import config


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
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
            from tabpfn import TabPFNClassifier as BaseTabPFN
        except Exception as e:
            print(f"[TabPFN] WARNING: Failed to import AutoTabPFNClassifier. Error: {type(e).__name__}: {e}")
            print("Using random fallback.")
            self.model_ = None
            self.classes_ = np.unique(y)
            return self

        self.classes_ = np.unique(y)

        # Σημείωση: Δεν χρησιμοποιούμε _raw_train για training γιατί το diffusion
        # augmentation αλλάζει το μέγεθος του training set. Χρησιμοποιούμε τα 
        # δεδομένα που μας δόθηκαν (ίσως Razor-filtered + augmented).
        # Το _raw_test χρησιμοποιείται μόνο για test predictions.
        X_use = X
            
        # Δημιουργία custom TabPFN base models με memory_saving_mode
        # This reduces peak VRAM by ~50-70% by internally batching attention
        n_base_models = 8
        custom_models = [
            (f"tabpfn_memsave_{i}", BaseTabPFN(
                device=self.device,
                memory_saving_mode=True,  # <-- Key: internal memory optimization
                random_state=self.random_state + i,
                ignore_pretraining_limits=True,
                n_estimators=4,  # internal ensemble per base model
            ))
            for i in range(n_base_models)
        ]
        
        self.model_ = AutoTabPFNClassifier(
            device=self.device,
            random_state=self.random_state,
            max_time=config.TABPFN_MAX_TIME, 
            ignore_pretraining_limits=True,
            phe_init_args={
                'tabpfn_base_model_source': 'custom',
                'custom_tabpfn_models': custom_models,
            },
        )

        # Μετατροπή σε numpy αν είναι tensor
        X_np = self._to_numpy(X_use)
        y_np = self._to_numpy(y)

        # Fit (αποθηκεύει δεδομένα για in-context learning)
        # Handle OOM by falling back to CPU
        try:
            self.model_.fit(X_np, y_np)
        except torch.cuda.OutOfMemoryError:
            print(f"[TabPFN Auto] CUDA OOM detected on {self.device}. Falling back to CPU (using system RAM).")
            # Cleanup
            self.model_ = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Re-initialize on CPU with memory-optimized models
            from tabpfn import TabPFNClassifier as BaseTabPFN
            cpu_models = [
                (f"tabpfn_cpu_{i}", BaseTabPFN(
                    device='cpu',
                    memory_saving_mode=True,
                    random_state=self.random_state + i,
                    ignore_pretraining_limits=True,
                    n_estimators=4,
                ))
                for i in range(n_base_models)
            ]
            self.model_ = AutoTabPFNClassifier(
                device='cpu',
                random_state=self.random_state,
                max_time=config.TABPFN_MAX_TIME,
                ignore_pretraining_limits=True,
                phe_init_args={
                    'tabpfn_base_model_source': 'custom',
                    'custom_tabpfn_models': cpu_models,
                },
            )
            self.model_.fit(X_np, y_np)
            # Update internal device tracker
            self.device = 'cpu'
        
        n_samples, n_features = X_np.shape
        print(f"[TabPFN Auto] Fitted with {n_samples} samples, {n_features} features, "
              f"on {self.device} (PHE enabled)")

        return self

    def predict_proba(self, X):
        """
        Πρόβλεψη πιθανοτήτων για κάθε κλάση.
        Adaptive batch size: reduces on OOM until prediction succeeds.
        """
        if self.model_ is None:
            # Fallback: τυχαίες πιθανότητες
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)

        X_np = self._to_numpy(X)
        
        # Adaptive batching with OOM fallback
        batch_size = 128
        min_batch_size = 1
        probs_list = []
        i = 0
        
        while i < len(X_np):
            X_batch = X_np[i : i + batch_size]
            
            try:
                p_batch = self.model_.predict_proba(X_batch)
                probs_list.append(p_batch)
                i += batch_size
                
                # Cleanup every few batches
                if len(probs_list) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except torch.cuda.OutOfMemoryError:
                # OOM: reduce batch size
                del X_batch
                gc.collect()
                torch.cuda.empty_cache()
                
                new_batch_size = max(min_batch_size, batch_size // 2)
                if new_batch_size == batch_size:
                    # Already at minimum, cannot reduce further
                    print(f"[TabPFN] FATAL: CUDA OOM at batch_size={batch_size}. Consider rerunning with device='cpu'.")
                    raise
                
                print(f"[TabPFN] CUDA OOM at batch_size={batch_size}. Reducing to {new_batch_size}.")
                batch_size = new_batch_size
                # Don't increment i, retry this batch with smaller size
        
        probs = np.concatenate(probs_list, axis=0)
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
