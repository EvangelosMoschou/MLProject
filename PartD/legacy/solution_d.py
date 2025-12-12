import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# --- GPU Setup (Only for XGBoost now) ---
try:
    import cuml
    print("✅ cuML imported (Only using for XGBoost acceleration).")
    USE_GPU = True
except ImportError:
    print("⚠️ cuML not found. Using CPU completely.")
    USE_GPU = False

DATA_PATH_TRAIN = '../Datasets/datasetTV.csv'
DATA_PATH_TEST = '../Datasets/datasetTest.csv'
MODEL_FILE = 'best_model_stacking_fast_cpu.pkl'
OUTPUT_FILE = 'labels1.npy'

def load_data():
    """Load and return training features/labels and test features."""
    if not os.path.exists(DATA_PATH_TRAIN):
        print("⚠️ Path warning: adjusting paths...")
        train_path = 'Datasets/datasetTV.csv'
        test_path = 'Datasets/datasetTest.csv'
    else:
        train_path = DATA_PATH_TRAIN
        test_path = DATA_PATH_TEST
        
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values
    X_test = test_df.values
    
    print(f"Train Shape: {X.shape}, Test Shape: {X_test.shape}")
    return X, y, X_test

def augment_data_gaussian(X, y, noise_level=0.05):
    """Augment data by adding Gaussian noise."""
    print(f"Augmenting data (noise={noise_level})...")
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = np.vstack((X, X + noise))
    y_aug = np.hstack((y, y))
    print(f"New Training Shape: {X_aug.shape}")
    return X_aug, y_aug

def get_svm_model():
    """Returns SVM Pipeline (Optimized CPU)."""
    # Optimized: probability=False removes internal 5-fold CV
    # StackingSVM works fine with decision_function output
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
        # n_jobs=-1 is usually safe for RF on CPU
        ('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))
    ])

def get_xgb_model():
    """Returns XGBoost Pipeline (GPU if available)."""
    params = {
        'n_estimators': 300, 
        'learning_rate': 0.05, 
        'max_depth': 6, 
        'n_jobs': -1, 
        'random_state': 42, 
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    if USE_GPU: 
        params.update({'device': 'cuda', 'tree_method': 'hist'})
        
    return Pipeline([
        ('scaler', StandardScaler()), 
        ('xgb', xgb.XGBClassifier(**params))
    ])

def get_mlp_model():
    """Returns MLP Pipeline (Deep Architecture)."""
    # Reduced max_iter slightly for speed, early_stopping handles it
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
        ('mlp', get_mlp_model())
    ]
    
    # Optimizations:
    # cv=3 instead of 5 (Reduces fitting runs by 2)
    # n_jobs=1 for safety
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3, 
        n_jobs=1 
    )
    return stacking_clf

def main():
    print("--- Part D: Fast Stacking Ensemble (Optimized) ---")
    
    # 1. Load & Encode
    X, y, X_test_submit = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 2. Initial Augmentation
    X_train_full, y_train_full = augment_data_gaussian(X, y_enc, noise_level=0.05)
    
    # 3. Initialize Model
    model = get_stacking_ensemble()
    
    # 4. First Training Pass
    print("\n[Phase 1] Training Initial Stacking Ensemble...")
    model.fit(X_train_full, y_train_full)
    
    # 5. Pseudo-Labeling Strategy
    print("\n[Phase 2] Pseudo-Labeling High Confidence Predictions...")
    
    # Predict probabilities on Test Set
    probs = model.predict_proba(X_test_submit)
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    CONFIDENCE_THRESHOLD = 0.90
    high_conf_idx = np.where(max_probs >= CONFIDENCE_THRESHOLD)[0]
    
    print(f"Stats: Found {len(high_conf_idx)}/{len(X_test_submit)} samples with confidence >= {CONFIDENCE_THRESHOLD}")
    
    if len(high_conf_idx) > 0:
        # Create Pseudo-Labeled Dataset
        X_pseudo = X_test_submit[high_conf_idx]
        y_pseudo = preds[high_conf_idx]
        
        # Combine
        X_final = np.vstack((X_train_full, X_pseudo))
        y_final = np.hstack((y_train_full, y_pseudo))
        
        print(f"Retraining on Expanded Dataset: {X_final.shape}")
        
        # Retrain
        model.fit(X_final, y_final)
    else:
        print("Not enough high-confidence samples. Skipping retraining.")
        
    # 6. Final Prediction
    print("\n[Phase 3] Generating Final Predictions...")
    final_preds_enc = model.predict(X_test_submit)
    final_preds = le.inverse_transform(final_preds_enc)
    
    # Save
    np.save(OUTPUT_FILE, final_preds.astype(int))
    print(f"Predictions saved to {OUTPUT_FILE}")
    
    # Save Model
    try:
        joblib.dump(model, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save model: {e}")

    # Verify
    print("Verifying file...")
    loaded = np.load(OUTPUT_FILE)
    if loaded.shape[0] == 6955:
        print("✅ Verification SUCCESS")
    else:
        print("❌ Verification FAILED")

if __name__ == "__main__":
    main()

