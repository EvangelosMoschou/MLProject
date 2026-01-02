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
    return X, y, X_test
