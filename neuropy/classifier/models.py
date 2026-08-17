"""Neural-net CCG classifiers — all multi-label, all sharing one interface.

Every label is predicted **in parallel** by its own sigmoid output rather than by
splitting a softmax: half the labeled pairs carry two or more labels ('rhythm'
and 'good' describe different axes, so they co-occur freely), and a softmax would
force them to compete. A pair whose every score is low is reported as ``'?'``.

Interface shared by all models::

    m.fit(X_ccg, X_null, Y)          -> self
    m.predict_proba(X_ccg, X_null)   -> [n, n_label] in [0, 1]
    m.save(path) / Model.load(path)
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from neuropy.classifier.features import shape_features, trace_stack

UNSURE = '?'


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float(((~pred) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


class BaseModel:
    """Shared fit/predict plumbing; subclasses only define ``_encode``."""

    def __init__(self, label_names: list[str], duration: float = 0.02, **kw):
        self.label_names = list(label_names)
        self.duration = duration
        self.params = kw
        self.scaler = StandardScaler()
        self.nets: list[MLPClassifier] = []
        self.constant: list[float | None] = []
        self.thresholds = np.full(len(self.label_names), 0.5)

    # Highres inputs are accepted by every model and used by the ones that ask
    # for them, so callers pass whatever they have without branching per model.
    uses_highres = False

    def _encode(self, ccg: np.ndarray, null: np.ndarray,
                ccg_hi: np.ndarray = None, null_hi: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    def fit(self, ccg: np.ndarray, null: np.ndarray, Y: np.ndarray,
            ccg_hi: np.ndarray = None, null_hi: np.ndarray = None):
        X = self.scaler.fit_transform(self._encode(ccg, null, ccg_hi, null_hi))
        self.nets, self.constant = [], []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            # A label absent (or universal) in this fold has nothing to separate.
            if y.min() == y.max():
                self.nets.append(None)
                self.constant.append(float(y[0]))
                continue
            net = MLPClassifier(**self._net_kw())
            net.fit(X, y)
            self.nets.append(net)
            self.constant.append(None)
        self._calibrate(self._scores(X), Y)
        return self

    def _scores(self, X: np.ndarray) -> np.ndarray:
        """Raw per-label probabilities from scaled features — the one scoring path."""
        out = np.zeros((len(X), len(self.label_names)))
        for j, net in enumerate(self.nets):
            out[:, j] = self.constant[j] if net is None else net.predict_proba(X)[:, 1]
        return out

    def _calibrate(self, P: np.ndarray, Y: np.ndarray):
        """Pick each label's decision threshold by maximizing F1 on training scores.

        Labels here are rare (83–919 of 3115), so the natural 0.5 cut classifies
        almost everything negative. Thresholds are fit on training data only —
        an outer CV fold never contributes to its own threshold.
        """
        grid = np.arange(0.05, 0.95, 0.01)
        for j in range(Y.shape[1]):
            y, p = Y[:, j], P[:, j]
            if y.min() == y.max():
                continue
            f1 = [_f1(y, p >= t) for t in grid]
            self.thresholds[j] = grid[int(np.argmax(f1))]

    def _net_kw(self) -> dict:
        kw = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600,
                  early_stopping=True, n_iter_no_change=25, random_state=0)
        kw.update(self.params)
        return kw

    def predict_proba(self, ccg: np.ndarray, null: np.ndarray,
                      ccg_hi: np.ndarray = None,
                      null_hi: np.ndarray = None) -> np.ndarray:
        X = self._encode(ccg, null, ccg_hi, null_hi)
        return self._scores(self.scaler.transform(X))

    def save(self, path: str):
        import pickle
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)
        meta = {'model': type(self).__name__, 'labels': self.label_names,
                'duration': self.duration}
        with open(os.path.splitext(path)[0] + '.json', 'w') as fh:
            json.dump(meta, fh, indent=1)

    @staticmethod
    def load(path: str) -> 'BaseModel':
        import pickle
        with open(path, 'rb') as fh:
            return pickle.load(fh)


class ShapeFeatureNet(BaseModel):
    """MLP over the 14 interpretable shape descriptors.

    Smallest input, easiest to explain, and the baseline every richer model
    has to beat before its extra capacity is worth anything.
    """

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        return shape_features(ccg, null, self.duration)


class TraceNet(BaseModel):
    """MLP over the full residual trace plus its 1st and 2nd derivatives.

    The derivative channels are what let a fixed-width net distinguish a sharp
    1 ms peak from a broad hump of the same height — the distinction the user's
    own 'msconn' vs '0rhythm' notes turn on.
    """

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        return trace_stack(ccg, null).reshape(len(np.atleast_2d(ccg)), -1)


class HybridNet(BaseModel):
    """Trace + derivatives + the scalar descriptors, concatenated.

    Gives the net both the raw shape and the summary statistics, so it need not
    re-derive peak SNR or lag from the trace with limited data.
    """

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        n = len(np.atleast_2d(ccg))
        return np.hstack([trace_stack(ccg, null).reshape(n, -1),
                          shape_features(ccg, null, self.duration)])


class DualResNet(BaseModel):
    """Both resolutions, each through its own embedding head before fusion.

    The two traces cover the *same* ±duration/2 window — lowres at 1 ms bins,
    highres at 1/30 ms — so they are different views of one interval, not
    different extents. Padding lowres out to the highres length would misalign
    lag 0 and destroy exactly the information the fine bins exist to carry.

    Each resolution instead gets its own scaler and PCA head, because highres
    bins hold ~1/30 the counts of lowres bins: one shared scaler over the
    concatenation would let 601 highres columns swamp 21 lowres ones by sheer
    count rather than by information. The heads compress each view to a
    comparable size, and the fused embedding keeps the scalar descriptors from
    both so peak SNR and lag survive the compression.
    """

    uses_highres = True

    def __init__(self, label_names, duration=0.02, n_components=24, **kw):
        super().__init__(label_names, duration=duration, **kw)
        self.n_components = n_components
        self.head_lo = _EmbedHead(n_components)
        self.head_hi = _EmbedHead(n_components)

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        n = len(np.atleast_2d(ccg))
        lo_trace = trace_stack(ccg, null).reshape(n, -1)
        lo_feat = shape_features(ccg, null, self.duration)
        parts = [self.head_lo.apply(lo_trace), lo_feat]
        if ccg_hi is not None:
            # Smoothed harder: at 1/30 ms bins single spikes dominate raw counts,
            # so the sigma is scaled by the bin-count ratio to match lowres.
            sigma = max(1.0, ccg_hi.shape[-1] / ccg.shape[-1] / 3)
            hi_trace = trace_stack(ccg_hi, null_hi, sigma=sigma).reshape(n, -1)
            hi_feat = shape_features(ccg_hi, null_hi, self.duration)
            parts += [self.head_hi.apply(hi_trace), hi_feat]
        return np.hstack(parts)


class _EmbedHead:
    """Per-resolution scaler + PCA, fitted on first use then reused."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Fit on the first (training) call, transform on every later one."""
        if self.pca is None:
            k = min(self.n_components, X.shape[0], X.shape[1])
            self.pca = PCA(n_components=k, random_state=0)
            return self.pca.fit_transform(self.scaler.fit_transform(X))
        return self.pca.transform(self.scaler.transform(X))


QUALITY_LABELS = ('best', 'good', 'ok')


class TwoHeadNet(HybridNet):
    """Hybrid input, but quality and shape are decided by different rules.

    The two label families behave differently in the ground truth: quality tiers
    are near mutually exclusive (1427 pairs carry exactly one, 14 carry two)
    while shape labels co-occur freely (half of all pairs have 2+). So quality
    gets one softmax head that must choose, and shape keeps its parallel sigmoid
    heads that may all fire or none.
    """

    def __init__(self, label_names, duration=0.02, **kw):
        super().__init__(label_names, duration=duration, **kw)
        self.quality_idx = [i for i, n in enumerate(self.label_names)
                            if n in QUALITY_LABELS]

    def _scores(self, X):
        P = super()._scores(X)
        if self.quality_idx:
            q = P[:, self.quality_idx]
            winner = q.max(axis=1, keepdims=True)
            # Runners-up are zeroed so only one tier can survive; the winner still
            # faces its own calibrated threshold, so a poor pair gets no tier at all.
            P[:, self.quality_idx] = np.where(q >= winner, q, 0.0)
        return P


MODELS = {'dualres': DualResNet, 'hybrid': HybridNet, 'shape': ShapeFeatureNet,
          'trace': TraceNet, 'twohead': TwoHeadNet}


def decide(proba: np.ndarray, label_names: list[str],
           thresholds: np.ndarray | float = 0.5) -> list[list[str]]:
    """Per-pair label lists, best score first; a row clearing nothing becomes ``['?']``.

    ``thresholds`` is normally a model's calibrated per-label vector, so a common
    label and a rare one are each cut where that label separates best.
    """
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), proba.shape[1:])
    out = []
    for row in proba:
        hits = [label_names[j] for j in np.argsort(-row) if row[j] >= thr[j]]
        out.append(hits or [UNSURE])
    return out
