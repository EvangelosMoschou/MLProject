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
    """Runs experiment using REAL Denoising Autoencoder features + Original features."""
    print(f"--- Running DAE Experiment (Deep Feature Extraction) ---")
    
    # Import DAE here to avoid circular imports if any
    try:
        from .dae_model import train_dae, get_dae_features
    except ImportError as e:
        print(f"❌ DAE Import Failed: {e}")
        return
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 1. Train DAE on X_train (Unsupervised)
        dae = train_dae(X_train, epochs=30)
        
        # 2. Extract DAE Features
        print(f"Extracting DAE features for Fold {fold+1}...")
        train_features = get_dae_features(dae, X_train)
        val_features = get_dae_features(dae, X_val)
        
        # 3. Concatenate (Original + DAE Features)
        X_train_aug = np.hstack((X_train, train_features))
        X_val_aug = np.hstack((X_val, val_features))
        
        # 4. Train Stacking Ensemble on Augmented Data
        # Ensure we use a robust model that handles many features
        clf = get_stacking_ensemble()
        
        clf.fit(X_train_aug, y_train)
        
        # Evaluate
        acc = accuracy_score(y_val, clf.predict(X_val_aug))
        accuracies.append(acc)
        print(f"Fold {fold+1}: Acc={acc:.4f}")
        
    avg_acc = np.mean(accuracies)
    runtime = time.time() - start_time
    
    print(f"DAE (Real Features) Results: Avg Acc={avg_acc:.4f}, Runtime={runtime:.2f}s")
    
    log_to_history(
        experiment_name='DAE-DeepFeatures',
        params={'model_params': 'Stacking+DAE(64-dim)', 'epochs': 30},
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



def run_calibration_experiment(X, y, cv_folds=5):
    """
    Evaluates if CalibratedClassifierCV improves model probability estimates.
    """
    print(f"--- Running Ensemble Calibration Experiment ---")
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores_raw = []
    scores_cal = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Base Model (Strong Stack)
        clf = get_stacking_ensemble()
        clf.fit(X_train, y_train)
        probs_raw = clf.predict_proba(X_val)
        loss_raw = log_loss(y_val, probs_raw)
        scores_raw.append(loss_raw)
        
        # Calibrated Model (Isotonic = Non-parametric)
        # Note: 'isotonic' requires > 1000 samples usually, which we have.
        cal_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit') 
        # Using 'prefit' is wrong here because we just trained 'clf' on X_train. 
        # Ideally we need a holdout set for calibration.
        # Let's use standard CV calibration (internal splits)
        
        cal_clf_cv = CalibratedClassifierCV(get_stacking_ensemble(), method='isotonic', cv=3)
        cal_clf_cv.fit(X_train, y_train)
        probs_cal = cal_clf_cv.predict_proba(X_val)
        loss_cal = log_loss(y_val, probs_cal)
        scores_cal.append(loss_cal)
        
        print(f"Fold {fold+1}: Raw LL={loss_raw:.4f}, Calibrated LL={loss_cal:.4f}")
        
    avg_raw = np.mean(scores_raw)
    avg_cal = np.mean(scores_cal)
    
    print(f"Calibration Results (Log Loss): Raw={avg_raw:.4f}, Calibrated={avg_cal:.4f}")
    if avg_cal < avg_raw:
        print("✅ Calibration Improved Performance!")
    else:
        print("❌ Calibration Hurnt Performance (or Neutral).")

def run_optuna_tuning(X, y, n_trials=30, cv_folds=5):
    """Runs Optuna hyperparameter tuning for XGBoost and CatBoost on GPU."""
    print(f"--- Running Optuna Tuning ---")
    try:
        import optuna
        import xgboost as xgb
        from catboost import CatBoostClassifier
    except ImportError as e:
        print(f"❌ Dependencies missing for tuning: {e}")
        return

    from .config import USE_GPU
    
    # Check GPU
    import torch
    GPU_AVAILABLE = USE_GPU and torch.cuda.is_available()
    if GPU_AVAILABLE:
        print("✅ GPU Tuning Enabled")
    else:
        print("⚠️ GPU not found. Tuning will be slow on CPU.")

    # --- XGBoost Objective ---
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'verbosity': 0
        }
        
        if GPU_AVAILABLE:
            params.update({'device': 'cuda', 'tree_method': 'hist'})
            
        clf = xgb.XGBClassifier(**params)
        score = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy').mean()
        return score

    # --- CatBoost Objective ---
    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'MultiClass',
            'verbose': 0,
            'random_seed': 42
        }
        
        if GPU_AVAILABLE:
            params.update({'task_type': 'GPU', 'devices': '0'})

        clf = CatBoostClassifier(**params)
        score = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy').mean()
        return score

    # 1. Tune XGBoost
    print("\n⚡ Tuning XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials)
    print(f"🏆 Best XGBoost Params: {study_xgb.best_trial.params}")
    print(f"   Best Acc: {study_xgb.best_value:.4f}")
    
    log_to_history(
        experiment_name='Optuna-XGB-GPU',
        params={'best_params': study_xgb.best_trial.params},
        metrics={'best_accuracy': study_xgb.best_value}
    )

    # 2. Tune CatBoost
    print("\n⚡ Tuning CatBoost...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=n_trials)
    print(f"🏆 Best CatBoost Params: {study_cat.best_trial.params}")
    print(f"   Best Acc: {study_cat.best_value:.4f}")

    log_to_history(
        experiment_name='Optuna-CatBoost-GPU',
        params={'best_params': study_cat.best_trial.params},
        metrics={'best_accuracy': study_cat.best_value}
    )
    
    print("\n✅ Tuning Complete. Please update src/models.py with these values.")
