import argparse
import warnings
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_data
from src.trainer import (
    run_baseline_experiment, 
    run_tabpfn_experiment, 
    run_adversarial_validation, 
    run_dae_experiment, 
    run_mixup_experiment, 
    run_optuna_tuning
)

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Run ML Experiments (Refactored)')
    parser.add_argument('--exp', type=str, default='all', help='Experiment to run: baseline, tabpfn, adv_val, dae, mixup, optuna, all')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()
    
    # Load Data
    X, y, X_test = load_data()
    if X is None:
        return

    # Basic Preprocessing for global usage if needed
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    if args.exp in ['baseline', 'all']:
        run_baseline_experiment(X, y_enc, cv_folds=args.cv)
        
    if args.exp in ['tabpfn', 'all']:
        run_tabpfn_experiment(X, y_enc, cv_folds=args.cv, n_ensemble=32) 
    
    if args.exp in ['adv_val', 'all']:
        if X_test is not None:
             run_adversarial_validation(X, y_enc, X_test)

    if args.exp in ['dae', 'all']:
        run_dae_experiment(X, y_enc, cv_folds=args.cv)

    if args.exp in ['mixup', 'all']:
        run_mixup_experiment(X, y_enc, cv_folds=args.cv)
        
    if args.exp in ['optuna', 'all']:
        run_optuna_tuning(X, y_enc, n_trials=10, cv_folds=3)

if __name__ == "__main__":
    main()
