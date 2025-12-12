import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .models import get_stacking_ensemble
from .utils import log_to_history, evaluate_model
from .config import USE_GPU

# --- TabPFN / PyTorch Monkeypatch ---
import torch
location = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return location(*args, **kwargs)
torch.load = safe_load

def run_baseline_experiment(X, y, cv_folds=5):
    """Runs a baseline Random Forest experiment."""
    print(f"--- Running Baseline Experiment (Stacking) ---")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []
    log_losses = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Strong Baseline (Stacking Ensemble)
        clf = get_stacking_ensemble()
        
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_val, y_val)
        
        accuracies.append(metrics['accuracy'])
        if metrics['log_loss'] != 'N/A':
            log_losses.append(metrics['log_loss'])
            
        print(f"Fold {fold+1}: Acc={metrics['accuracy']:.4f}")
        
    avg_acc = np.mean(accuracies)
    avg_ll = np.mean(log_losses) if log_losses else 'N/A'
    runtime = time.time() - start_time
    
    print(f"Baseline Results: Avg Acc={avg_acc:.4f}, Runtime={runtime:.2f}s")
    
    log_to_history(
        experiment_name='Baseline-Stacking',
        params={'model_params': 'StackingEnsemble(SVM+RF+XGB+MLP+Cat)', 'cv_folds': cv_folds},
        metrics={'accuracy': avg_acc, 'log_loss': avg_ll, 'runtime_seconds': runtime}
    )

def run_tabpfn_experiment(X, y, cv_folds=5, n_ensemble=8):
    """Runs TabPFN experiment."""
    print(f"--- Running TabPFN Experiment ---")
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        print("❌ TabPFN not installed. Skipping.")
        return

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []
    
    start_time = time.time()
    
    # n_estimators controls compute time/accuracy for v6.0.6 (latest)
    # Using 'cuda' device as requested
    classifier = TabPFNClassifier(device='cuda' if USE_GPU else 'cpu', n_estimators=n_ensemble) 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # TabPFN
        classifier.fit(X_train, y_train)
        y_eval = classifier.predict(X_val)
        acc = accuracy_score(y_val, y_eval)
        accuracies.append(acc)
        print(f"Fold {fold+1}: Acc={acc:.4f}")
        
    avg_acc = np.mean(accuracies)
    runtime = time.time() - start_time
    
    print(f"TabPFN Results: Avg Acc={avg_acc:.4f}, Runtime={runtime:.2f}s")
    
    log_to_history(
        experiment_name='TabPFN',
        params={'model_params': f'TabPFN(n_estimators={n_ensemble})', 'cv_folds': cv_folds},
        metrics={'accuracy': avg_acc, 'runtime_seconds': runtime}
    )

def run_adversarial_validation(X, y, X_test):
    """Checks for train-test distribution shift."""
    print(f"--- Running Adversarial Validation ---")
    
    # Create dataset: Train=0, Test=1
    X_adv = np.vstack((X, X_test))
    y_adv = np.hstack((np.zeros(len(X)), np.ones(len(X_test))))
    
    # Shuffle
    idx = np.arange(len(X_adv))
    np.random.shuffle(idx)
    X_adv = X_adv[idx]
    y_adv = y_adv[idx]
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    scores = cross_val_score(clf, X_adv, y_adv, cv=5, scoring='roc_auc')
    runtime = time.time() - start_time
    
    avg_auc = np.mean(scores)
    print(f"Adversarial Validation AUC: {avg_auc:.4f}")
    if avg_auc > 0.7:
        print("⚠️ Significant distribution shift detected!")
    else:
        print("✅ Distributions look similar.")
        
    log_to_history(
        experiment_name='Adversarial-Validation',
        params={'model_params': 'RF(n=50)', 'task': 'DistShift-Check'},
        metrics={'auc': avg_auc, 'runtime_seconds': runtime}
    )

def run_dae_experiment(X, y, cv_folds=5):
    """Runs experiment using Denoising Autoencoder features (Proxied via Noise Augmentation)."""
    print(f"--- Running DAE Experiment (Augmentation Inside Fold) ---")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply Augmentation ONLY to Training Data
        noise_level = 0.05
        X_train_noisy = X_train + np.random.normal(0, noise_level, X_train.shape)
        X_train_aug = np.vstack((X_train, X_train_noisy))
        y_train_aug = np.hstack((y_train, y_train))
        
        # Train (Using Stacking Ensemble)
        clf = get_stacking_ensemble()
        
        clf.fit(X_train_aug, y_train_aug)
        
        # Evaluate on clean Validation Data
        acc = accuracy_score(y_val, clf.predict(X_val))
        accuracies.append(acc)
        print(f"Fold {fold+1}: Acc={acc:.4f}")
        
    avg_acc = np.mean(accuracies)
    runtime = time.time() - start_time
    
    print(f"DAE (Noise Aug) Results: Avg Acc={avg_acc:.4f}, Runtime={runtime:.2f}s")
    
    log_to_history(
        experiment_name='DAE-Proxy-NoiseAug-Corrected',
        params={'model_params': 'RF-Augmented', 'noise': 0.05},
        metrics={'accuracy': avg_acc, 'runtime_seconds': runtime}
    )

def run_mixup_experiment(X, y, cv_folds=5):
    """Runs MixUp experiment."""
    print(f"--- Running MixUp Experiment (Augmentation Inside Fold) ---")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply MixUp ONLY to Training Data
        mixup_alpha = 0.2
        X_mix = []
        y_mix = []
        
        n_samples = len(X_train)
        # Create 50% more data via mixing
        for _ in range(n_samples // 2):
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            
            x_new = lam * X_train[idx1] + (1 - lam) * X_train[idx2]
            y_new = y_train[idx1] if lam > 0.5 else y_train[idx2]
            
            X_mix.append(x_new)
            y_mix.append(y_new)
            
        X_train_aug = np.vstack((X_train, np.array(X_mix)))
        y_train_aug = np.hstack((y_train, np.array(y_mix)))
        
        # Train (Using Stacking Ensemble)
        clf = get_stacking_ensemble()
        
        clf.fit(X_train_aug, y_train_aug)
        
        # Evaluate
        acc = accuracy_score(y_val, clf.predict(X_val))
        accuracies.append(acc)
        print(f"Fold {fold+1}: Acc={acc:.4f}")
        
    avg_acc = np.mean(accuracies)
    runtime = time.time() - start_time
    
    print(f"MixUp Results: Avg Acc={avg_acc:.4f}, Runtime={runtime:.2f}s")
    
    log_to_history(
        experiment_name='MixUp-Augmentation-Corrected',
        params={'model_params': 'RF-MixUp', 'alpha': 0.2},
        metrics={'accuracy': avg_acc, 'runtime_seconds': runtime}
    )

def run_optuna_tuning(X, y, n_trials=20, cv_folds=3):
    """Runs Optuna hyperparameter tuning for Random Forest."""
    print(f"--- Running Optuna Tuning ---")
    try:
        import optuna
    except ImportError:
        print("❌ Optuna not installed. Skipping.")
        return

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42, 
            n_jobs=-1
        )
        
        # Fast CV for tuning
        score = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.params}")
    
    log_to_history(
        experiment_name='Optuna-Tuning-RF',
        params={'best_params': study.best_trial.params, 'n_trials': n_trials},
        metrics={'best_accuracy': study.best_value}
    )
