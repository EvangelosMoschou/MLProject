import pandas as pd
import numpy as np
import os
import csv
import datetime
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

HISTORY_FILE = 'master_history.csv'

def log_to_history(experiment_name, params, metrics, dataset_train_path='Datasets/datasetTV.csv', dataset_test_path='Datasets/datasetTest.csv'):
    """Logs experiment results to master_history.csv."""
    
    # Ensure file exists and has header
    file_exists = os.path.isfile(HISTORY_FILE)
    
    row = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat(),
        'run_id': os.uname().nodename + '-' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'), # Simple run ID
        'git_branch': 'unknown', # Placeholder
        'git_commit': 'unknown', # Placeholder
        'script': 'run_experiments.py',
        'experiment_name': experiment_name,
        'dataset_train_path': dataset_train_path,
        'dataset_test_path': dataset_test_path,
        'cv_folds': params.get('cv_folds', 'N/A'),
        'seed': params.get('random_state', 'N/A'),
        'preprocess_json': str(params.get('preprocess', {})),
        'model_json': str(params.get('model_params', {})),
        'metrics_json': str(metrics),
        'runtime_seconds': metrics.get('runtime_seconds', 0),
        'notes': params.get('notes', '')
    }
    
    fieldnames = [
        'timestamp_utc', 'run_id', 'git_branch', 'git_commit', 'script', 'experiment_name',
        'dataset_train_path', 'dataset_test_path', 'cv_folds', 'seed', 'preprocess_json',
        'model_json', 'metrics_json', 'runtime_seconds', 'notes'
    ]
    
    try:
        with open(HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"✅ Logged experiment '{experiment_name}' to {HISTORY_FILE}")
    except Exception as e:
        print(f"❌ Failed to log experiment: {e}")

def load_data(train_path=None, test_path=None):
    """Loads and returns X, y, X_test."""
    
    # Defaults
    paths_to_try_train = [
        'Datasets/datasetTV.csv',
        '../Datasets/datasetTV.csv',
        'train.csv'
    ]
    paths_to_try_test = [
        'Datasets/datasetTest.csv',
        '../Datasets/datasetTest.csv',
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
        test_df = pd.read_csv(test_path, header=None)
        
        X = train_df.iloc[:, :-1].values
        y = train_df.iloc[:, -1].values
        X_test = test_df.values
        
        return X, y, X_test
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def evaluate_model(model, X_val, y_val):
    """Calculates metrics for a given model and validation set."""
    preds = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        ll = log_loss(y_val, probs)
    else:
        ll = "N/A"
        
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'log_loss': ll,
        'f1_weighted': f1
    }
