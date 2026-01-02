import copy

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


class CalibratedModel:
    def __init__(self, base_model, name):
        self.base, self.name, self.ir = base_model, name, None

    def fit(self, X, y, sample_weight=None, pseudo_X=None, pseudo_y=None, pseudo_w=None):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        self.models = []
        self.calibrators = []

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

            if pseudo_X is not None and len(pseudo_X) > 0:
                # Handle label compatibility
                y_tr_eff = y_tr
                pseudo_y_eff = pseudo_y
                
                is_pseudo_soft = pseudo_y.ndim > 1 or np.issubdtype(pseudo_y.dtype, np.floating)
                
                # Check if model is Torch (supports soft labels)
                # We assume if it has 'finetune_on_pseudo', it's our custom Torch wrapper
                is_torch = hasattr(self.base, 'finetune_on_pseudo')
                
                if is_pseudo_soft and not is_torch:
                    # Tree model -> Convert Soft Pseudo to Hard (Argmax)
                    # Use argmax to get int labels
                    if pseudo_y.ndim > 1:
                        pseudo_y_eff = np.argmax(pseudo_y, axis=1).astype(np.int64)
                    else:
                        pseudo_y_eff = pseudo_y.astype(np.int64)
                
                elif is_pseudo_soft and is_torch:
                    # Torch model -> Convert Hard Train to Soft (One-Hot)
                    num_classes = pseudo_y.shape[1]
                    y_tr_eff = np.eye(num_classes, dtype=np.float32)[y_tr]
                    # pseudo_y_eff is already soft
                
                elif not is_pseudo_soft and is_torch:
                     # Both hard, nothing to do, unless we want to force soft? No.
                     pass

                X_tr = np.vstack([X_tr, pseudo_X])
                
                # Concatenate labels
                # Ensure dimensions match
                if y_tr_eff.ndim == 1 and pseudo_y_eff.ndim == 2:
                     # This shouldn't happen with logic above, but safety check:
                     # If y_tr is hard and pseudo is soft -> make y_tr soft
                     num_classes = pseudo_y_eff.shape[1]
                     y_tr_eff = np.eye(num_classes, dtype=np.float32)[y_tr_eff]
                elif y_tr_eff.ndim == 2 and pseudo_y_eff.ndim == 1:
                     # If y_tr is soft and pseudo is hard -> make pseudo soft
                     num_classes = y_tr_eff.shape[1]
                     pseudo_y_eff = np.eye(num_classes, dtype=np.float32)[pseudo_y_eff]

                if y_tr_eff.ndim == 2:
                    y_tr = np.vstack([y_tr_eff, pseudo_y_eff])
                else:
                    y_tr = np.concatenate([y_tr_eff, pseudo_y_eff])

                if sw_tr is None:
                    sw_tr = np.ones(len(y_tr), dtype=np.float32)
                    sw_tr[: len(tr_idx)] = 1.0
                else:
                    sw_tr = np.concatenate([sw_tr, np.asarray(pseudo_w, dtype=np.float32)])

            model = copy.deepcopy(self.base)
            try:
                model.fit(X_tr, y_tr, sample_weight=sw_tr)
            except TypeError:
                model.fit(X_tr, y_tr)

            val_probs = model.predict_proba(X_val).astype(np.float32)
            c_list = []
            for c in range(val_probs.shape[1]):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(val_probs[:, c], (y_val == c).astype(int))
                c_list.append(iso)

            self.models.append(model)
            self.calibrators.append(c_list)

        return self

    def predict_proba(self, X):
        total_probs = np.zeros((len(X), len(self.calibrators[0])))
        for model, calib_list in zip(self.models, self.calibrators):
            raw_p = model.predict_proba(X)
            cal_p = np.zeros_like(raw_p)
            for c in range(raw_p.shape[1]):
                cal_p[:, c] = calib_list[c].predict(raw_p[:, c])
            cal_p /= (cal_p.sum(axis=1, keepdims=True) + 1e-10)
            total_probs += cal_p
        return total_probs / len(self.models)
