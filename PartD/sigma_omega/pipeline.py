import numpy as np
import torch

from . import config
from .calibration import CalibratedModel
from .domain import adversarial_weights
from .features import apply_feature_view, build_streams
from .losses import apply_lid_temperature_scaling
from .models_torch import KAN, ThetaTabM, TrueTabR, is_torch_model, select_silver_samples
from .models_trees import get_cat_langevin, get_xgb_dart
from .pseudo import normalize_pseudo
from .stacking import fit_predict_stacking


def predict_probs_for_view(view, seed, X_train_base, X_test_base, y_enc, num_classes, pseudo_idx=None, pseudo_y=None, pseudo_w=None):
    pseudo = normalize_pseudo(pseudo_idx=pseudo_idx, pseudo_y=pseudo_y, pseudo_w=pseudo_w)

    X_v, X_test_v = apply_feature_view(
        X_train_base,
        X_test_base,
        view=view,
        seed=seed,
        allow_transductive=config.ALLOW_TRANSDUCTIVE,
    )

    X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te = build_streams(X_v, X_test_v)
    
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
        ('ThetaTabM', ThetaTabM(None, num_classes)), 
        ('TrueTabR', TrueTabR(num_classes)),
        ('KAN', KAN(None, num_classes)),
    ]
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
    X_tree_tr, X_tree_te, X_neural_tr, X_neural_te, lid_tr, lid_te = build_streams(X_v, X_test_v)
    
    pseudo_X_tree = None
    pseudo_X_neural = None
    if pseudo.active():
        pseudo_X_tree = X_tree_te[pseudo.idx]
        pseudo_X_neural = X_neural_te[pseudo.idx]

    view_probs = 0
    for name, base in names_models:
        print(f"  > Calibrating {name} (10-Fold)...")
        data_tr = X_tree_tr if ('XGB' in name or 'Cat' in name) else X_neural_tr
        data_te = X_tree_te if ('XGB' in name or 'Cat' in name) else X_neural_te

        calibrated = CalibratedModel(base, name)
        calibrated.fit(
            data_tr,
            y_enc,
            sample_weight=sample_weight,
            pseudo_X=pseudo_X_tree if ('XGB' in name or 'Cat' in name) else pseudo_X_neural,
            pseudo_y=pseudo.y if pseudo.active() else None,
            pseudo_w=pseudo.w if pseudo.active() else None,
        )
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

        view_probs += p
        
        # Checkpointing per model/view
        # Only saving if not stacking (stacking handles its own states?) 
        # Actually simplest checkpoint is in main loop.
        
    return view_probs / len(names_models)
