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
    import subprocess
    subprocess.check_call(['nvidia-smi'])
    GPU_AVAILABLE = True
except (ImportError, FileNotFoundError, subprocess.CalledProcessError):
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
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)

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
        
    return xgb.XGBClassifier(**params)

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
        
    return CatBoostClassifier(**params)

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

def get_resnet_model():
    """Returns ResNet Pipeline (GPU if available)."""
    # Import locally to avoid circular dependencies if any, though likely safe globally
    from .resnet_model import ResNetClassifier
    return Pipeline([
        ('scaler', StandardScaler()),
        ('resnet', ResNetClassifier(
            hidden_dim=256, 
            num_blocks=2, 
            dropout=0.2, 
            epochs=30, 
            batch_size=128
        ))
    ])

def get_stacking_ensemble():
    """Constructs the Stacking Classifier."""
    estimators = [
        ('svm', get_svm_model()), 
        ('rf', get_rf_model()), 
        ('xgb', get_xgb_model()), 
        ('cat', get_catboost_model()),
        ('mlp', get_mlp_model()),
        ('resnet', get_resnet_model())
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            multi_class='multinomial',
            max_iter=5000,
            C=1.0,
            solver='lbfgs'
        ),
        cv=5,
        passthrough=True,
        n_jobs=1
    )
    return stacking_clf
