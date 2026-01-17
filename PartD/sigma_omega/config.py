import os
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')


def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None and v.strip() != '' else int(default)


def _env_float(name, default):
    v = os.getenv(name)
    return float(v) if v is not None and v.strip() != '' else float(default)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seeds (defaults preserve prior behavior; can override via SEEDS="1,2,3" or N_SEEDS/SEED_BASE)
_seeds_env = os.getenv('SEEDS')
if _seeds_env:
    SEEDS = [int(s.strip()) for s in _seeds_env.split(',') if s.strip()]
else:
    _n_seeds = os.getenv('N_SEEDS')
    if _n_seeds:
        base = int(os.getenv('SEED_BASE', '42'))
        n = int(_n_seeds)
        SEEDS = [base + i for i in range(n)]
    else:
        SEEDS = [1337]  # Single seed for fast run

BATCH_SIZE = 512  # Adjusted for RTX 3060 per requirements
LR_SCALE = 2e-3
WSAM_GAMMA = _env_float('WSAM_GAMMA', 0.9)
SAM_RHO = 0.08
TABM_K = _env_int('TABM_K', 4)  # BatchEnsemble members
DIFFUSION_EPOCHS = _env_int('DIFFUSION_EPOCHS', 30)
TTT_STEPS = _env_int('TTT_STEPS', 10)

ALLOW_TRANSDUCTIVE = _env_bool('ALLOW_TRANSDUCTIVE', False)  # DISABLED to prevent overfitting (DAE, Laplacian were using test data)
USE_STACKING = _env_bool('USE_STACKING', True)  # Enable stacking meta-learner
VIEWS = [v.strip().lower() for v in os.getenv('VIEWS', 'raw,quantile,pca,ica').split(',') if v.strip()]
TUNE_HYPERPARAMS = _env_bool('TUNE_HYPERPARAMS', False) # Enable Optuna optimization stub

# Stacking enhancements (opt-in)
# META_LEARNER tuned via Optuna (3030 trials): LR(C=0.55) = 87.20% > Ridge(86.96%) > NNLS(82.39%)
META_LEARNER = os.getenv('META_LEARNER', 'lr').strip().lower()  # lr | lgbm | ridge | nnls
USE_TABPFN = _env_bool('USE_TABPFN', True)  # Enabled: adds diversity to ensemble
TABPFN_N_ENSEMBLES = _env_int('TABPFN_N_ENSEMBLES', 64)  # v2.5 supports higher ensembles
TABPFN_MAX_TIME = _env_int('TABPFN_MAX_TIME', 60) # Fast Production: 60s per fit (OOF used 300s)
LGBM_MAX_DEPTH = _env_int('LGBM_MAX_DEPTH', 3)
LGBM_NUM_LEAVES = _env_int('LGBM_NUM_LEAVES', 31)
LGBM_N_ESTIMATORS = _env_int('LGBM_N_ESTIMATORS', 400)

# Adversarial validation reweighting (opt-in)
ENABLE_ADV_REWEIGHT = _env_bool('ENABLE_ADV_REWEIGHT', False)
ADV_MODEL = os.getenv('ADV_MODEL', 'lr').strip().lower()  # lr | xgb
ADV_CLIP = _env_float('ADV_CLIP', 10.0)
ADV_POWER = _env_float('ADV_POWER', 1.0)
RUN_ADV_DIAGNOSTIC = _env_bool('RUN_ADV_DIAGNOSTIC', True)  # Print adversarial validation AUC at startup

# SWA (stochastic weight averaging) for ThetaTabM (opt-in)
ENABLE_SWA = _env_bool('ENABLE_SWA', False)
SWA_START_EPOCH = _env_int('SWA_START_EPOCH', 10)

# CORAL feature alignment (opt-in; usually transductive)
ENABLE_CORAL = _env_bool('ENABLE_CORAL', False)
CORAL_REG = _env_float('CORAL_REG', 1e-3)

# Iterative self-training with stability constraints (opt-in; transductive)
ENABLE_SELF_TRAIN = _env_bool('ENABLE_SELF_TRAIN', False)  # Disabled for quick run
SELF_TRAIN_ITERS = _env_int('SELF_TRAIN_ITERS', 1)
SELF_TRAIN_CONF = _env_float('SELF_TRAIN_CONF', 0.96)  # 96% confidence for pseudo-labels
SELF_TRAIN_AGREE = _env_float('SELF_TRAIN_AGREE', 1.0)
SELF_TRAIN_VIEW_AGREE = _env_float('SELF_TRAIN_VIEW_AGREE', 0.50)  # Lowered from 0.66
SELF_TRAIN_MAX = _env_int('SELF_TRAIN_MAX', 10000)
SELF_TRAIN_WEIGHT_POWER = _env_float('SELF_TRAIN_WEIGHT_POWER', 1.0)

# Loss-optimized training knobs
LOSS_NAME = os.getenv('LOSS', 'ce').strip().lower()  # ce | focal
LABEL_SMOOTHING = _env_float('LABEL_SMOOTHING', 0.0)
FOCAL_GAMMA = _env_float('FOCAL_GAMMA', 2.0)
USE_CLASS_BALANCED = _env_bool('CLASS_BALANCED', False)
CB_BETA = _env_float('CB_BETA', 0.999)
USE_MIXUP = _env_bool('USE_MIXUP', True)

# Efficiency knobs
DAE_EPOCHS = _env_int('DAE_EPOCHS', 30)
DAE_NOISE_STD = _env_float('DAE_NOISE_STD', 0.1)
MANIFOLD_K = _env_int('MANIFOLD_K', 20)
ENABLE_PAGERANK = _env_bool('ENABLE_PAGERANK', True)
ENABLE_LAPLACIAN = _env_bool('ENABLE_LAPLACIAN', False)  # Disabled for speed & memory
USE_GPU_EIGENMAPS = _env_bool('USE_GPU_EIGENMAPS', True)  # GPU acceleration for Laplacian
ENABLE_DIFFUSION = _env_bool('ENABLE_DIFFUSION', True)  # Diffusion augmentation per-fold
DIFFUSION_N_SAMPLES = _env_int('DIFFUSION_N_SAMPLES', 1000)  # Synthetic samples per fold
ENABLE_RAZOR = _env_bool('ENABLE_RAZOR', True)  # Enable Razor feature selection

# LID temperature scaling (opt-in)
ENABLE_LID_SCALING = _env_bool('ENABLE_LID_SCALING', False)
LID_T_MIN = _env_float('LID_T_MIN', 1.0)
LID_T_MAX = _env_float('LID_T_MAX', 2.5)
LID_T_POWER = _env_float('LID_T_POWER', 1.0)

# Test-time training (TTT) on "silver" samples (opt-in; transductive)
ENABLE_TTT = _env_bool('ENABLE_TTT', False)
TTT_GAP_LOW = _env_float('TTT_GAP_LOW', 0.10)
TTT_GAP_HIGH = _env_float('TTT_GAP_HIGH', 0.35)
TTT_EPOCHS = _env_int('TTT_EPOCHS', 1)
TTT_MAX_SAMPLES = _env_int('TTT_MAX_SAMPLES', 4096)
TTT_LR_MULT = _env_float('TTT_LR_MULT', 0.2)

# Model checkpointing (saves trained models to disk for faster reruns)
SAVE_CHECKPOINTS = _env_bool('SAVE_CHECKPOINTS', True)  # Save models after training
LOAD_CHECKPOINTS = _env_bool('LOAD_CHECKPOINTS', False)  # Load pre-trained models if available

# SMOKE TEST LOGIC (Must be last to override defaults)
SMOKE_RUN = _env_bool('SMOKE_RUN', False)
if SMOKE_RUN:
    print(">>> SMOKE RUN DETECTED: REDUCING COMPLEXITY FOR VERIFICATION <<<")
    SEEDS = [42]
    DIFFUSION_EPOCHS = 5
    DAE_EPOCHS = 5
    GBDT_ITERATIONS = 50
    N_FOLDS = 2
    TABPFN_N_ENSEMBLES = 4  # Reduce from 64 to 4 for speed
    ENABLE_DIFFUSION = False  # Disable diffusion augmentation
    ENABLE_LAPLACIAN = False  # Disable Laplacian Eigenmaps
    USE_TABPFN = False  # Disable TabPFN entirely for smoke test
else:
    # Default iterations for full run
    GBDT_ITERATIONS = 500
    N_FOLDS = 5
