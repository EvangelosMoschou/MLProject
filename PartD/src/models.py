from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
from .config import USE_GPU

# Make GPU check visible
try:
    import torch
    if USE_GPU and torch.cuda.is_available():
        GPU_AVAILABLE = True
    else:
        GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False

def get_svm_model():
    """Returns SVM Pipeline (Optimized CPU)."""
    clf = SVC(C=10, gamma='scale', kernel='rbf', probability=False, random_state=42)
    pca = PCA(n_components=100)
    return Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', pca), 
        ('svm', clf)
    ])

def get_rf_model():
    """Returns Random Forest Pipeline (CPU)."""
    return Pipeline([
        ('scaler', StandardScaler()), 
        ('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))
    ])

def get_xgb_model():
    """Returns XGBoost Pipeline (GPU if available)."""
    params = {
        'n_estimators': 487, 
        'learning_rate': 0.029, 
        'max_depth': 9, 
        'min_child_weight': 3,
        'subsample': 0.585,
        'colsample_bytree': 0.998,
        'n_jobs': -1, 
        'random_state': 42, 
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    if GPU_AVAILABLE: 
        params.update({'device': 'cuda', 'tree_method': 'hist'})
        print("⚡ XGBoost using GPU")
        
    return Pipeline([
        ('scaler', StandardScaler()), 
        ('xgb', xgb.XGBClassifier(**params))
    ])

def get_catboost_model():
    """Returns CatBoost Pipeline (GPU if available)."""
    params = {
        'iterations': 931,
        'learning_rate': 0.263,
        'depth': 8,
        'l2_leaf_reg': 9.48,
        'border_count': 100,
        'loss_function': 'MultiClass',
        'verbose': 0,
        'random_seed': 42
    }
    if GPU_AVAILABLE:
        params.update({'task_type': 'GPU', 'devices': '0'})
        print("⚡ CatBoost using GPU")
        
    return Pipeline([
        ('scaler', StandardScaler()), # CatBoost doesn't strictly need this but good for consistency
        ('cat', CatBoostClassifier(**params))
    ])

def get_mlp_model():
    """Returns MLP Pipeline (Deep Architecture)."""
    return Pipeline([
        ('scaler', StandardScaler()), 
        ('mlp', MLPClassifier(hidden_layer_sizes=(512, 256), 
                            max_iter=300, 
                            early_stopping=True,
                            validation_fraction=0.1,
                            random_state=42))
    ])

def get_stacking_ensemble():
    """Constructs the Stacking Classifier."""
    estimators = [
        ('svm', get_svm_model()), 
        ('rf', get_rf_model()), 
        ('xgb', get_xgb_model()), 
        ('cat', get_catboost_model()),
        ('mlp', get_mlp_model())
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3, 
        n_jobs=1 
    )
    return stacking_clf
