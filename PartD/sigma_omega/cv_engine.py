import copy
import gc

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from . import config
from .features import (
    AnomalyFeatureGenerator,
    GeometricFeatureGenerator,
    build_streams,
    apply_feature_view,
    find_confusion_pairs,
)
from .generative import synthesize_data


class UnifiedCVEngine:
    def __init__(
        self,
        names_models,
        view_name,
        X_train_base,
        X_test_base,
        y,
        num_classes,
        cv_splits=10,
        seed=42,
        sample_weight=None,
        pseudo_idx=None,
        pseudo_y=None,
        pseudo_w=None,
        X_train_raw=None,
        X_test_raw=None,
    ):
        self.names_models = names_models
        self.view_name = view_name
        self.X_train_base = X_train_base
        self.X_test_base = X_test_base
        self.y = y
        self.num_classes = num_classes
        self.cv_splits = cv_splits
        self.seed = seed
        self.sample_weight = sample_weight
        self.pseudo_idx = pseudo_idx
        self.pseudo_y = pseudo_y
        self.pseudo_w = pseudo_w
        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw

    def run(self):
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        n_models = len(self.names_models)
        oof_preds = [np.zeros((len(self.y), self.num_classes), dtype=np.float32) for _ in range(n_models)]
        test_preds_running = [np.zeros((len(self.X_test_base), self.num_classes), dtype=np.float32) for _ in range(n_models)]

        for tr_idx, val_idx in skf.split(self.X_train_base, self.y):
            X_tr_raw_fold = self.X_train_base[tr_idx]
            X_val_raw_fold = self.X_train_base[val_idx]
            y_tr = self.y[tr_idx]

            X_tr_fold, X_val_fold = apply_feature_view(
                X_tr_raw_fold,
                X_val_raw_fold,
                view=self.view_name,
                seed=self.seed,
                allow_transductive=config.ALLOW_TRANSDUCTIVE,
            )

            _, X_test_view_fold = apply_feature_view(
                X_tr_raw_fold,
                self.X_test_base,
                view=self.view_name,
                seed=self.seed,
                allow_transductive=config.ALLOW_TRANSDUCTIVE,
            )

            sw_tr = self.sample_weight[tr_idx] if self.sample_weight is not None else None

            X_tr_aug, y_tr_aug, sw_tr_aug = X_tr_fold, y_tr, sw_tr
            X_tr_raw_aug, y_tr_raw_aug, sw_tr_raw_aug = X_tr_raw_fold, y_tr, sw_tr

            if config.ENABLE_DIFFUSION and len(X_tr_fold) > 100:
                X_tr_aug, y_tr_aug = synthesize_data(
                    X_tr_fold,
                    y_tr,
                    n_new_per_class=config.DIFFUSION_N_SAMPLES // 5,
                )
                if sw_tr is not None:
                    sw_diff = np.ones(len(y_tr_aug) - len(tr_idx), dtype=np.float32) * 0.5
                    sw_tr_aug = np.concatenate([sw_tr, sw_diff])

                has_tabpfn = any('TabPFN' in m[0] for m in self.names_models)
                if has_tabpfn:
                    X_tr_raw_aug, y_tr_raw_aug = synthesize_data(
                        X_tr_raw_fold,
                        y_tr,
                        n_new_per_class=config.DIFFUSION_N_SAMPLES // 5,
                    )
                    if sw_tr is not None:
                        sw_diff = np.ones(len(y_tr_raw_aug) - len(tr_idx), dtype=np.float32) * 0.5
                        sw_tr_raw_aug = np.concatenate([sw_tr, sw_diff])

                gc.collect()

            X_tree_tr, X_tree_val, X_neural_tr, X_neural_val, _, _ = build_streams(
                X_tr_aug,
                X_val_fold,
                y_train=y_tr_aug,
            )
            _, X_tree_te_fold, _, X_neural_te_fold, _, _ = build_streams(
                X_tr_aug,
                X_test_view_fold,
                y_train=y_tr_aug,
            )

            pX_tree = None
            pX_neural = None
            py = self.pseudo_y
            pw = self.pseudo_w

            if self.pseudo_idx is not None and len(self.pseudo_idx) > 0:
                pX_tree = X_tree_te_fold[self.pseudo_idx]
                pX_neural = X_neural_te_fold[self.pseudo_idx]

            for idx_m, (name, base_template) in enumerate(self.names_models):
                model = copy.deepcopy(base_template)

                is_tree = ('XGB' in name or 'Cat' in name or 'LGBM' in name)
                is_tabpfn = 'TabPFN' in name
                is_svm = getattr(base_template, '_is_svm', False) or 'SVM' in name

                if is_svm:
                    X_f_tr = X_tr_aug
                    y_tr_eff_model = y_tr_aug
                    sw_tr_eff_model = sw_tr_aug
                    X_f_val = X_val_fold
                    X_f_te = X_test_view_fold
                    pX_f = X_test_view_fold[self.pseudo_idx] if (self.pseudo_idx is not None and len(self.pseudo_idx) > 0) else None
                elif is_tabpfn:
                    X_f_tr = X_tr_raw_aug
                    y_tr_eff_model = y_tr_raw_aug
                    sw_tr_eff_model = sw_tr_raw_aug
                    X_f_val = X_val_raw_fold
                    X_f_te = self.X_test_raw
                    pX_f = self.X_test_raw[self.pseudo_idx] if (self.pseudo_idx is not None and len(self.pseudo_idx) > 0) else None

                    geo = GeometricFeatureGenerator().fit(X_tr_raw_aug, y_tr_raw_aug)
                    anom = AnomalyFeatureGenerator().fit(X_tr_raw_aug)

                    g_tr = np.hstack([geo.transform(X_tr_raw_aug), anom.transform(X_tr_raw_aug)])
                    g_val = np.hstack([geo.transform(X_val_raw_fold), anom.transform(X_val_raw_fold)])
                    g_te = np.hstack([geo.transform(self.X_test_raw), anom.transform(self.X_test_raw)])
                    if pX_f is not None:
                        gp = np.hstack([geo.transform(pX_f), anom.transform(pX_f)])

                    X_f_tr = np.hstack([X_f_tr, g_tr])
                    X_f_val = np.hstack([X_f_val, g_val])
                    X_f_te = np.hstack([X_f_te, g_te])
                    if pX_f is not None:
                        pX_f = np.hstack([pX_f, gp])
                else:
                    X_f_tr = X_tree_tr if is_tree else X_neural_tr
                    y_tr_eff_model = y_tr_aug
                    sw_tr_eff_model = sw_tr_aug
                    X_f_val = X_tree_val if is_tree else X_neural_val
                    X_f_te = X_tree_te_fold if is_tree else X_neural_te_fold
                    pX_f = pX_tree if is_tree else pX_neural

                X_train_final = X_f_tr
                y_train_final = y_tr_eff_model
                w_train_final = sw_tr_eff_model

                if pX_f is not None and py is not None:
                    is_pseudo_soft = (py.ndim > 1) or np.issubdtype(py.dtype, np.floating)
                    is_torch = hasattr(model, 'finetune_on_pseudo') and not is_tabpfn

                    y_tr_eff = y_train_final
                    py_eff = py

                    if is_pseudo_soft and (not is_torch or is_tabpfn):
                        py_eff = np.argmax(py, axis=1).astype(np.int64) if py.ndim > 1 else py.astype(np.int64)
                    elif is_pseudo_soft and is_torch:
                        if y_tr_eff.ndim == 1:
                            y_tr_eff = np.eye(self.num_classes, dtype=np.float32)[y_tr_eff]

                    X_train_final = np.vstack([X_f_tr, pX_f])

                    if y_train_final.ndim == 1 and py_eff.ndim == 1:
                        y_train_final = np.concatenate([y_tr_eff, py_eff])
                    else:
                        if y_tr_eff.ndim == 1:
                            y_tr_eff = y_tr_eff[:, None]
                        if py_eff.ndim == 1:
                            py_eff = py_eff[:, None]
                        y_train_final = np.vstack([y_tr_eff, py_eff])
                        if y_train_final.shape[1] == 1:
                            y_train_final = y_train_final.ravel()

                    w_tr_base = w_train_final if w_train_final is not None else np.ones(len(y_tr), dtype=np.float32)

                    if config.CONFUSION_WEIGHT_MULTIPLIER > 1.0:
                        fold_pairs = find_confusion_pairs(X_tr_raw_fold, y_tr, top_k=1, seed=self.seed)
                        for (c_a, c_b) in fold_pairs:
                            mask_conf = (y_tr_eff == c_a) | (y_tr_eff == c_b)
                            if mask_conf.shape[0] == w_tr_base.shape[0]:
                                w_tr_base[mask_conf] *= config.CONFUSION_WEIGHT_MULTIPLIER

                    w_p_base = pw if pw is not None else np.ones(len(py), dtype=np.float32)
                    w_train_final = np.concatenate([w_tr_base, w_p_base])

                try:
                    model.fit(X_train_final, y_train_final, sample_weight=w_train_final)
                except TypeError:
                    model.fit(X_train_final, y_train_final)

                p_oof = model.predict_proba(X_f_val).astype(np.float32)
                oof_preds[idx_m][val_idx] = p_oof

                p_test = model.predict_proba(X_f_te).astype(np.float32)
                test_preds_running[idx_m] += p_test

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del X_train_final, y_train_final, w_train_final
            del X_tr_fold, X_val_fold
            del X_tree_tr, X_tree_val, X_neural_tr, X_neural_val
            del X_f_tr, X_f_val, X_f_te
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for i in range(n_models):
            test_preds_running[i] /= self.cv_splits

        self._print_diagnostics(oof_preds)

        return oof_preds, test_preds_running

    def _print_diagnostics(self, oof_preds):
        print("\n  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║              OVERFITTING DIAGNOSTIC REPORT                   ║")
        print("  ╠══════════════════════════════════════════════════════════════╣")

        model_names = [name for name, _ in self.names_models]
        oof_accuracies = []

        for idx_m, name in enumerate(model_names):
            oof_pred_labels = np.argmax(oof_preds[idx_m], axis=1)
            oof_acc = (oof_pred_labels == self.y).mean() * 100
            oof_accuracies.append(oof_acc)
            flag = ""
            if oof_acc > 98:
                flag = " ⚠️ SUSPICIOUS (too high)"
            elif oof_acc < 70:
                flag = " ⚠️ UNDERFITTING"
            print(f"  ║  {name:<20} OOF Accuracy: {oof_acc:6.2f}%{flag:<20}║")

        ensemble_oof = np.mean([oof_preds[i] for i in range(len(oof_preds))], axis=0)
        ensemble_oof_labels = np.argmax(ensemble_oof, axis=1)
        ensemble_oof_acc = (ensemble_oof_labels == self.y).mean() * 100

        print("  ╠══════════════════════════════════════════════════════════════╣")
        print(f"  ║  {'ENSEMBLE OOF':<20} Accuracy: {ensemble_oof_acc:6.2f}%                   ║")
        print("  ╠══════════════════════════════════════════════════════════════╣")

        best_single = max(oof_accuracies)
        ensemble_gain = ensemble_oof_acc - best_single
        if ensemble_gain > 0:
            print(f"  ║  Ensemble Gain: +{ensemble_gain:.2f}% over best single model          ║")
        else:
            print(f"  ║  ⚠️ Ensemble WORSE than best single by {-ensemble_gain:.2f}%            ║")

        if ensemble_oof_acc > 95:
            print("  ║  ⚠️ WARNING: OOF acc >95% may indicate data leakage!         ║")

        print("  ╚══════════════════════════════════════════════════════════════╝\n")
