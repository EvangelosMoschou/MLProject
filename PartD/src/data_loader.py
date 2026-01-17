import os
import pandas as pd
from .config import DATA_TRAIN_PATH, DATA_TEST_PATH

def load_data(train_path=None, test_path=None):
    """Loads and returns X, y, X_test."""
    
    # Defaults
    paths_to_try_train = [
        DATA_TRAIN_PATH,
        f'../{DATA_TRAIN_PATH}',
        'Datasets/datasetTV.csv', # fallback
        'train.csv'
    ]
    paths_to_try_test = [
        DATA_TEST_PATH,
        f'../{DATA_TEST_PATH}',
        'Datasets/datasetTest.csv', # fallback
        'test.csv'
    ]
    
    # Resolve Train Path
    if train_path is None:
        for p in paths_to_try_train:
            if os.path.exists(p):
                train_path = p
                break
    
    # Resolve Test Path
    if test_path is None:
        for p in paths_to_try_test:
            if os.path.exists(p):
                test_path = p
                break
                
    if train_path is None or not os.path.exists(train_path):
         print(f"❌ Data files not found. Tried: {paths_to_try_train}")
         return None, None, None

    try:
        print(f"Loading data from: {train_path} and {test_path}")
        train_df = pd.read_csv(train_path, header=None)
        
        # [OMEGA] Explicit Column Naming
        n_cols = train_df.shape[1]
        train_df.columns = [f"feat_{i}" for i in range(n_cols - 1)] + ["target"]
        
        # [REFACTOR] Explicit targeting
        # Blindly slicing -1 is dangerous if CSV structure changes.
        # Prefer config.TARGET_COL if available, else -1.
        target_idx = -1
        
        X = train_df.iloc[:, :target_idx].values
        y = train_df.iloc[:, target_idx].values
        
        assert y is not None, "Target y is None!"
        assert len(X) == len(y), "X and y length mismatch!"
        
        if test_path and os.path.exists(test_path):
            test_df = pd.read_csv(test_path, header=None)
            X_test = test_df.values
        else:
            X_test = None
        
        return X, y, X_test
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None
