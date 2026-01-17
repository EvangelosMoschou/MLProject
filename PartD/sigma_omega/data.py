def load_data_safe():
    try:
        from src.data_loader import load_data
    except Exception as e:
        raise RuntimeError(
            "Failed to import `src.data_loader.load_data`. Ensure `PartD/src` is on PYTHONPATH and dependencies are installed."
        ) from e

    try:
        X, y, X_test = load_data()
    except Exception as e:
        raise RuntimeError("`load_data()` raised an exception.") from e

    if X is None or y is None or X_test is None:
        raise ValueError("`load_data()` returned None(s); expected (X, y, X_test).")

    # [REFACTOR] Sanitizer Logic
    # 1. Type Enforcement
    import numpy as np
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    if X_test is not None:
        X_test = X_test.astype(np.float32)

    # 2. NaNs Handling (Critical for NN)
    # Check for NaNs
    if np.isnan(X).any() or (X_test is not None and np.isnan(X_test).any()):
        print("   [DATA] NaNs detected! Running SimpleImputer(strategy='mean')...")
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='mean') # Mean is safe for most numerical
        X = imp.fit_transform(X)
        if X_test is not None:
            # Transductive? User said "IF transductive mode is OFF, ensure Imputer fits only on Train".
            # config isn't imported here yet? Wait, config import is needed.
            # Let's inspect imports. None in snippet.
            # Assuming standard strict fit-transform on Train, transform on Test strictly is safer unless explicit.
            # Implementation: fit on X, transform X_test.
            X_test = imp.transform(X_test)
            
    return X, y, X_test
