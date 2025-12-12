import os
import csv
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score
from .config import HISTORY_FILE

def log_to_history(experiment_name, params, metrics, dataset_train_path='Datasets/datasetTV.csv', dataset_test_path='Datasets/datasetTest.csv'):
    """Logs experiment results to master_history.csv."""
    
    # Ensure file exists and has header
    file_exists = os.path.isfile(HISTORY_FILE)
    
    row = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat(),
        'run_id': os.uname().nodename + '-' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'), # Simple run ID
        'git_branch': 'unknown', # Placeholder
        'git_commit': 'unknown', # Placeholder
        'script': 'main.py',
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
