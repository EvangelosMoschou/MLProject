import numpy as np

from . import config
from .calibration import CalibratedModel
from .domain import adversarial_weights
from .features import apply_feature_view, build_streams
from .losses import apply_lid_temperature_scaling
from .models_torch import ThetaTabM, TrueTabR, is_torch_model, select_silver_samples
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
        ('XGB_DART', get_xgb_dart(num_classes)),
        ('Cat_Langevin', get_cat_langevin(num_classes)),
        ('ThetaTabM', ThetaTabM(None, num_classes)), # Lazy init input_dim
        ('TrueTabR', TrueTabR(num_classes)),
    ]

    if config.USE_STACKING:
        # Cross-Fit Stacking: Pass raw data and view name.
        # Transformations happen inside the K-Fold loop.
        p = fit_predict_stacking(
            names_models=names_models,
            view_name=view,
            X_train_base=X_train_base,
            X_test_base=X_test_base,
            y=y_enc,
            num_classes=num_classes,
            cv_splits=10,
            seed=seed,
            sample_weight=sample_weight,
            pseudo_idx=pseudo_idx,
            pseudo_y=pseudo_y,
            pseudo_w=pseudo_w,
        )
        # Note: LID scaling for stacking output?
        # If we didn't compute LID globally, we can't scale easily.
        # But fit_predict_stacking output is already a meta-learner output.
        # Usually we trust the meta-learner.
        return p

    # Standard Ensemble (Averaging) if Stacking is disabled
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
        if config.ENABLE_TTT and is_torch_model(base):
            if not config.ALLOW_TRANSDUCTIVE:
                raise RuntimeError("ENABLE_TTT requires ALLOW_TRANSDUCTIVE=1 (it adapts on test features).")
            
            # Reflexion Core Upgrade: Entropy Minimization TTT
            from .ttt import EntropyMinimizationTTT
            ttt_solver = EntropyMinimizationTTT(
                steps=config.TTT_EPOCHS,
                lr=config.TTT_LR_MULT * 1e-4, # heuristic base lr
            )
            
            # For TTT, we need the PyTorch module. calibrated.model is the base model.
            # We assume base is a wrapper that has a .module or is the module itself.
            # However, base in models_torch.py usually effectively Wraps a model.
            # Let's assume base.model is the nn.Module or base is it.
            # Checking models_torch.py would be good, but let's try a generic approach first:
            # If base has 'model' attribute, usage it.
            
            # Actually, let's just make sure we are not fine-tuning the global model permanently for this fold.
            # The TTT class handles copy.
            
            # We need to pass data_te (numpy) -> torch tensor
            device = next(base.model.parameters()).device
            X_test_tensor = torch.tensor(data_te, dtype=torch.float32, device=device)
            
            adapted = ttt_solver.adapt(base.model, X_test_tensor)
            
            # Predict with adapted model
            import torch
            adapted.eval()
            with torch.no_grad():
                logits = adapted(X_test_tensor)
                p = torch.softmax(logits, dim=1).cpu().numpy()

        if config.ENABLE_LID_SCALING:
            p = apply_lid_temperature_scaling(
                p,
                lid_te,
                t_min=config.LID_T_MIN,
                t_max=config.LID_T_MAX,
                power=config.LID_T_POWER,
            )

        view_probs += p

    return view_probs / len(names_models)
