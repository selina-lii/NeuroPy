"""Training and leave-one-rat-out cross-validation.

Splitting is by animal, never at random: pairs from one session share neurons, so
a random split leaks the same neuron into train and test and reports a score the
model would not reach on a new animal.
"""
from __future__ import annotations

import numpy as np

from neuropy.classifier.dataset import LabeledSet
from neuropy.classifier.models import MODELS, BaseModel

MODEL_DIR = 'data/classifier'


def per_label_scores(Y: np.ndarray, P: np.ndarray, names: list[str],
                     threshold: np.ndarray | float = 0.5) -> dict[str, dict]:
    """Precision / recall / F1 / AUC per label, plus its positive count."""
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), (len(names),))
    out = {}
    for j, name in enumerate(names):
        y, p = Y[:, j], P[:, j]
        pred = p >= thr[j]
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {'n': int(y.sum()), 'precision': prec, 'recall': rec,
                     'f1': f1, 'auc': _auc(y, p)}
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via rank statistic; 0.5 when only one class is present."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return 0.5
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def leave_one_rat_out(ls: LabeledSet, model_name: str = 'conv',
                      duration: float = 0.02, **kw) -> dict:
    """Hold out each animal in turn; return pooled out-of-sample predictions."""
    ccg, null, Y, rats = ls.X_ccg, ls.X_null, ls.Y, ls.rats
    hi, hi_null = _highres_for(ls, model_name)
    P = np.zeros_like(Y, dtype=float)
    T = np.zeros_like(Y, dtype=float)     # each row cut by its own fold's thresholds
    for rat in sorted(set(rats)):
        test = rats == rat
        model = MODELS[model_name](ls.label_names, duration=duration, **kw)
        model.fit(ccg[~test], null[~test], Y[~test],
                  *_slice_highres(hi, hi_null, ~test))
        P[test] = model.predict_proba(ccg[test], null[test],
                                      *_slice_highres(hi, hi_null, test))
        T[test] = model.thresholds
        print(f"  fold {rat}: trained on {(~test).sum()}, tested on {test.sum()}",
              flush=True)
    hits = (P >= T).astype(int)
    return {'proba': P, 'Y': Y, 'thresholds': T, 'pred': hits,
            'scores': _scores_from_hits(Y, P, hits, ls.label_names),
            'label_names': ls.label_names}


def _highres_for(ls: LabeledSet, model_name: str):
    """Highres arrays when the model wants them and the set has them."""
    if not MODELS[model_name].uses_highres or not ls.has_highres:
        return None, None
    return ls.X_ccg_hi, ls.X_null_hi


def _slice_highres(hi, hi_null, mask):
    return (None, None) if hi is None else (hi[mask], hi_null[mask])


def _scores_from_hits(Y, P, hits, names) -> dict[str, dict]:
    """Score a precomputed decision matrix (thresholds vary by fold)."""
    out = {}
    for j, name in enumerate(names):
        y, pred = Y[:, j], hits[:, j].astype(bool)
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {'n': int(y.sum()), 'precision': prec, 'recall': rec,
                     'f1': f1, 'auc': _auc(y, P[:, j])}
    return out


def fit_final(ls: LabeledSet, model_name: str = 'conv',
              duration: float = 0.02, **kw) -> BaseModel:
    """Train on every labeled pair — the model that ships to the UI."""
    model = MODELS[model_name](ls.label_names, duration=duration, **kw)
    hi, hi_null = _highres_for(ls, model_name)
    return model.fit(ls.X_ccg, ls.X_null, ls.Y, hi, hi_null)


def report(scores: dict) -> str:
    rows = ["  label            n   prec   rec    F1    AUC",
            "  " + "-" * 44]
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]['f1']):
        rows.append(f"  {name:<14} {s['n']:>4}  {s['precision']:.2f}  "
                    f"{s['recall']:.2f}  {s['f1']:.2f}  {s['auc']:.2f}")
    mean_f1 = np.mean([s['f1'] for s in scores.values()])
    mean_auc = np.mean([s['auc'] for s in scores.values()])
    rows.append(f"  mean F1 {mean_f1:.3f}   mean AUC {mean_auc:.3f}")
    return "\n".join(rows)
