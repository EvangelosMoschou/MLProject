import numpy as np
import torch

from . import config
from .calibration import CalibratedModel
from .domain import adversarial_weights
from .features import apply_feature_view, build_streams
from .losses import apply_lid_temperature_scaling
from .models_torch import TrueTabR, is_torch_model, select_silver_samples
from .models_trees import get_cat_langevin, get_xgb_dart
from .models_svm import get_svm
from .pseudo import normalize_pseudo
from .stacking import fit_predict_stacking


def predict_probs_for_view(view, seed, X_train_base, X_test_base, y_enc, num_classes, 
                           pseudo_idx=None, pseudo_y=None, pseudo_w=None,
                           X_train_raw=None, X_test_raw=None,
                           razor_masks=None):
    """
    Predict probabilities for a single view.
    
    Args:
        X_train_base, X_test_base: Razor-filtered data (default, for backward compat)
        X_train_raw, X_test_raw: Raw unfiltered data (for TabPFN)
        razor_masks: Dict with 'cat' and 'xgb' masks for per-model feature selection
    """
    pseudo = normalize_pseudo(pseudo_idx=pseudo_idx, pseudo_y=pseudo_y, pseudo_w=pseudo_w)

    X_v, X_test_v = apply_feature_view(
        X_train_base,
        X_test_base,
        view=view,
        seed=seed,
        allow_transductive=config.ALLOW_TRANSDUCTIVE,
    )

    X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te = build_streams(X_v, X_test_v, y_train=y_enc)
    
    # [OMEGA] Diffusion Augmentation
    # Μόνο επαύξηση των training data αν δεν κάνουμε validating/stacking (Stacking χειρίζεται strict CV εσωτερικά)
    # Αλλά εδώ, είμαστε έξω από Stacking;
    # Στην πραγματικότητα, το pipeline.py καλεί fit_predict_stacking Ή κάνει ensemble averaging.
    # Αν ensemble averaging, το X_train_base ΕΙΝΑΙ τα training data για ολόκληρο το run (στο seed)
    # Αλλά συνήθως κάνουμε validate χρησιμοποιώντας CV μέσα στο CalibratedModel (που κάνει 10-fold CV).
    # Η επαύξηση πρέπει να γίνει ανά fold στο CV.
    # Το CalibratedModel είναι generic.
    # Το Stacking είναι generic.
    # Το Diffusion πρέπει να γίνει ΜΕΣΑ στο CV split.
    # Το fit_predict_stacking έχει το CV loop. Το Diffusion πρέπει να injectαριστεί ΕΚΕΙ.
    # Για standard ensemble (else block): Το CalibratedModel χρησιμοποιεί CV εσωτερικά.
    # Πρέπει να κάνουμε patch το CalibratedModel ή Diffusion εκεί.
    # Απλούστερο: Μόνο προσθέστε Diffusion στο fit_predict_stacking που είναι το robust path.
    # Αν ο user απενεργοποιήσει stacking, παραλείπουμε diffusion για τώρα ή το προσθέτουμε στο CalibratedModel αργότερα.
    # Ας εστιαστούμε στο Stacking path καθώς είναι το "Grandmaster" default.

    pseudo_X_tree = None
    pseudo_X_neural = None
    if pseudo.active():
        pseudo_X_tree = X_tree_te[pseudo.idx]
        pseudo_X_neural = X_neural_te[pseudo.idx]

    sample_weight = None
    if config.ENABLE_ADV_REWEIGHT:
        sample_weight = adversarial_weights(
            X_v,
            X_test_v,
            seed=seed,
            model=config.ADV_MODEL,
            clip=config.ADV_CLIP,
            power=config.ADV_POWER,
        )

    names_models = [
        ('XGB_DART', get_xgb_dart(num_classes, iterations=config.GBDT_ITERATIONS)),
        ('Cat_Langevin', get_cat_langevin(num_classes, iterations=config.GBDT_ITERATIONS * 2)),
        ('TrueTabR', TrueTabR(num_classes)),
        ('SVM', get_svm()),
    ]
    
    # Store per-model razor masks if provided
    if razor_masks is not None:
        for name, model in names_models:
            if 'XGB' in name:
                model._razor_mask = razor_masks.get('xgb')
                model._X_train_raw = X_train_raw
                model._X_test_raw = X_test_raw
            elif 'Cat' in name:
                model._razor_mask = razor_masks.get('cat')
                model._X_train_raw = X_train_raw
                model._X_test_raw = X_test_raw
            # TrueTabR uses default (cat mask via X_train_base)
    
    if config.USE_TABPFN:
        from .models_pfn import TabPFNWrapper
        # TabPFN uses raw unfiltered data if provided, bypassing Razor
        tabpfn = TabPFNWrapper(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_estimators=config.TABPFN_N_ENSEMBLES,
        )
        # Store raw data references for TabPFN to use instead of Razor-filtered
        if X_train_raw is not None:
            tabpfn._raw_train = X_train_raw
            tabpfn._raw_test = X_test_raw
        names_models.append(('TabPFN', tabpfn))

    # --- SMART VIEW ROUTING ---
    # Filter models based on the view to avoid redundancy and save time.
    # Trees: Need Raw/PCA/ICA (Rotation blind). Invariant to Quantile.
    # TrueTabR: Need Quantile (Scaling). Handle rotations natively.
    # TabPFN: Prefers RAW data (per authors' recommendation - minimal preprocessing).
    
    active_models = []
    view_norm = view.strip().lower()
    
    # 1. Quantile View -> TrueTabR + SVM (scaled data suits both)
    if view_norm == 'quantile':
        for name, model in names_models:
            if name in ['TrueTabR', 'SVM']:
                active_models.append((name, model))
                
    # 2. Raw View -> Trees + TabPFN (TabPFN prefers raw data)
    elif view_norm == 'raw':
        for name, model in names_models:
            if name in ['XGB_DART', 'Cat_Langevin', 'TabPFN']:
                active_models.append((name, model))
    
    # 3. PCA / ICA / RP Views -> SVM Only (Trees can't exploit rotations)
    elif view_norm in ['pca', 'ica', 'rp']:
        for name, model in names_models:
            if name in ['SVM']:
                 active_models.append((name, model))
                 
    # 4. Fallback (Unknown view or simple run) -> All Models
    else:
        active_models = names_models
        
    if not active_models:
        print(f"Warning: Smart Routing left no models for view '{view}'. Using all.")
        active_models = names_models
        
    names_models = active_models
    # --------------------------

    if config.USE_STACKING:
        # Cross-Fit Stacking: Πέρασμα raw data και view name.
        # Οι μετασχηματισμοί γίνονται μέσα στο K-Fold loop.
        p = fit_predict_stacking(
            names_models=names_models,
            view_name=view,
            X_train_base=X_train_base,
            X_test_base=X_test_base,
            y=y_enc,
            num_classes=num_classes,
            cv_splits=config.N_FOLDS,
            seed=seed,
            sample_weight=sample_weight,
            pseudo_idx=pseudo_idx,
            pseudo_y=pseudo_y,
            pseudo_w=pseudo_w,
            X_train_raw=X_train_raw,   # Raw data for TabPFN
            X_test_raw=X_test_raw,     # Raw data for TabPFN
        )
        # Σημείωση: LID scaling για stacking output;
        # Αν δεν υπολογίσαμε LID globally, δεν μπορούμε να scale εύκολα.
        # Αλλά το fit_predict_stacking output είναι ήδη ένα meta-learner output.
        # Συνήθως εμπιστευόμαστε το meta-learner.
        return p

    # Standard Ensemble (Μέσος Όρος) αν το Stacking είναι απενεργοποιημένο
    X_v, X_test_v = apply_feature_view(
        X_train_base,
        X_test_base,
        view=view,
        seed=seed,
        allow_transductive=config.ALLOW_TRANSDUCTIVE,
    )
    X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te = build_streams(X_v, X_test_v, y_train=y_enc)
    
    pseudo_X_tree = None
    pseudo_X_neural = None
    if pseudo.active():
        pseudo_X_tree = X_tree_te[pseudo.idx]
        pseudo_X_neural = X_neural_te[pseudo.idx]

    view_probs = 0
    for name, base in names_models:
        print(f"  > Calibrating {name} ({config.N_FOLDS}-Fold)...")
        data_tr = X_tree_tr if ('XGB' in name or 'Cat' in name) else X_neural_tr
        data_te = X_tree_te if ('XGB' in name or 'Cat' in name) else X_neural_te

        # Model checkpointing
        checkpoint_dir = f"PartD/outputs/models/{view}_{seed}"
        checkpoint_path = f"{checkpoint_dir}/{name}.pkl"
        
        import os
        if config.LOAD_CHECKPOINTS and os.path.exists(checkpoint_path):
            # Load pre-trained model
            calibrated = CalibratedModel.load(checkpoint_path, base)
        else:
            # Train fresh
            calibrated = CalibratedModel(base, name)
            calibrated.fit(
                data_tr,
                y_enc,
                sample_weight=sample_weight,
                pseudo_X=pseudo_X_tree if ('XGB' in name or 'Cat' in name) else pseudo_X_neural,
                pseudo_y=pseudo.y if pseudo.active() else None,
                pseudo_w=pseudo.w if pseudo.active() else None,
            )
            # Save checkpoint
            if config.SAVE_CHECKPOINTS:
                calibrated.save(checkpoint_path)
        
        p = calibrated.predict_proba(data_te)

        if config.ENABLE_TTT and is_torch_model(base):
            if not config.ALLOW_TRANSDUCTIVE:
                raise RuntimeError("ENABLE_TTT requires ALLOW_TRANSDUCTIVE=1 (it adapts on test features).")
        # TTT Logic - Reflexion Core Upgrade
        if config.ENABLE_TTT and is_torch_model(base):
            if not config.ALLOW_TRANSDUCTIVE:
                raise RuntimeError("ENABLE_TTT requires ALLOW_TRANSDUCTIVE=1 (it adapts on test features).")
            
            from .ttt import EntropyMinimizationTTT
            ttt_solver = EntropyMinimizationTTT(
                steps=config.TTT_EPOCHS,
                lr=config.TTT_LR_MULT * 1e-4, 
            )
            
            # Apply TTT to each calibrated fold model
            p_ttt_list = []
            
            # data_te is numpy array.
            # calibrated.models contains the trained models for each fold.
            
            # We need to broadcast data_te to torch for TTT.
            # Note: We are adapting on the FULL test set (data_te) for each fold model.
            # This is standard TTT.
            
            # Assuming models are on device.
            device = config.DEVICE
            X_test_tensor = torch.tensor(data_te, dtype=torch.float32, device=device)
            
            for fold_i, model in enumerate(calibrated.models):
                 # Clone to avoid mutating the original calibrated model if we want to keep it?
                 # Yes, TTT is test-time only.
                 # model is a sklearn wrapper or custom wrapper.
                 # We need the underlying torch module.
                 
                 # Accessing inner module depends on wrapper.
                 # For ThetaTabM/TrueTabR/KAN wrapper, .model is the module.
                 if hasattr(model, 'model'):
                     inner_module = model.model
                 elif hasattr(model, 'module'):
                     inner_module = model.module
                 else:
                     # Fallback
                     inner_module = model
                 
                 # Prepare kwargs for TTT (e.g. neighbors for TabR)
                 ttt_kwargs = {}
                 if hasattr(model, 'get_neighbors'):
                     # Retrieve neighbors for the test batch
                     # Note: X_test_tensor is on device, get_neighbors handles it
                     neighbors = model.get_neighbors(X_test_tensor)
                     ttt_kwargs['neighbors'] = neighbors

                 # Copy for adaptation
                 import copy
                 model_copy = copy.deepcopy(inner_module)
                 model_copy.train() 
                 
                 # Adapt
                 # Pass ttt_kwargs to adapt
                 adapted_model = ttt_solver.adapt(model_copy, X_test_tensor, **ttt_kwargs)
                 
                 adapted_model.eval()
                 with torch.no_grad():
                     # Pass ttt_kwargs to forward
                     logits = adapted_model(X_test_tensor, **ttt_kwargs)
                     p_ttt_list.append(torch.softmax(logits, dim=1).cpu().numpy())

            # Average TTT predictions across folds
            p = np.mean(p_ttt_list, axis=0)

        if config.ENABLE_LID_SCALING:
            p = apply_lid_temperature_scaling(
                p,
                lid_te,
                t_min=config.LID_T_MIN,
                t_max=config.LID_T_MAX,
                power=config.LID_T_POWER,
            )

        # Store predictions for agreement analysis
        if 'model_preds' not in dir():
            model_preds = {}
        model_preds[name] = np.argmax(p, axis=1)
        
        view_probs += p
        
        # Checkpointing per model/view
        # Only saving if not stacking (stacking handles its own states?) 
        # Actually simplest checkpoint is in main loop.
    
    # Model Agreement Analysis
    if len(model_preds) > 1:
        print("      Model Agreement:")
        model_names = list(model_preds.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                agree = (model_preds[m1] == model_preds[m2]).mean()
                # Low agreement = good diversity for ensemble
                diversity = "✓" if agree < 0.90 else "⚠"
                print(f"        {m1} vs {m2}: {agree:.1%} {diversity}")
        
    return view_probs / len(names_models)
