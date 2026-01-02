from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PseudoData:
    idx: np.ndarray
    y: np.ndarray
    w: np.ndarray

    @staticmethod
    def empty() -> 'PseudoData':
        return PseudoData(
            idx=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.int64),
            w=np.array([], dtype=np.float32),
        )

    def active(self) -> bool:
        return self.idx is not None and self.y is not None and len(self.idx) > 0

    def is_soft(self) -> bool:
        return self.y.ndim > 1 or np.issubdtype(self.y.dtype, np.floating)


def normalize_pseudo(pseudo_idx=None, pseudo_y=None, pseudo_w=None) -> PseudoData:
    if pseudo_idx is None or pseudo_y is None:
        return PseudoData.empty()
    idx = np.asarray(pseudo_idx, dtype=np.int64)
    
    # Check if soft labels (probs) or hard labels (int)
    y = np.asarray(pseudo_y)
    if y.ndim == 1:
         # Hard labels: ensure int64
         y = y.astype(np.int64)
    else:
         # Soft labels: ensure float32
         y = y.astype(np.float32)

    if pseudo_w is None:
        w = np.ones((len(idx),), dtype=np.float32)
    else:
        w = np.asarray(pseudo_w, dtype=np.float32)
    if len(idx) == 0:
        return PseudoData.empty()
    return PseudoData(idx=idx, y=y, w=w)


def vote_mode_and_agreement(votes_2d: np.ndarray):
    """votes_2d: (M, N) int labels. Returns (mode_pred[N], agree_frac[N])."""
    mode_pred = np.zeros((votes_2d.shape[1],), dtype=np.int64)
    agree_frac = np.zeros((votes_2d.shape[1],), dtype=np.float64)
    for j in range(votes_2d.shape[1]):
        vals, counts = np.unique(votes_2d[:, j], return_counts=True)
        k = int(np.argmax(counts))
        mode_pred[j] = int(vals[k])
        agree_frac[j] = float(np.max(counts)) / float(votes_2d.shape[0])
    return mode_pred, agree_frac


def view_agreement_fraction(preds_tensor_vs_n: np.ndarray, mode_pred: np.ndarray):
    """preds_tensor_vs_n: (V, S, N) int labels. Returns view_agree_frac[N]."""
    view_agree_frac = np.zeros((preds_tensor_vs_n.shape[2],), dtype=np.float64)
    for vi in range(preds_tensor_vs_n.shape[0]):
        view_votes = preds_tensor_vs_n[vi]  # (S, N)
        view_mode, _ = vote_mode_and_agreement(view_votes)
        view_agree_frac += (view_mode == mode_pred).astype(np.float64)
    view_agree_frac /= float(preds_tensor_vs_n.shape[0])
    return view_agree_frac
