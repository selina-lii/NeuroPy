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

import datetime
import json
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from neuropy.classifier.dataset import loaded_ccg
from neuropy.classifier.features import (bank_response, kernel_bank, kernel_response,
                                         learned_bank, residual, shape_features,
                                         smooth, trace_stack)

UNSURE = '?'


BIAS_BETA = {'accurate': 0.5, 'balanced': 1.0, 'discover': 2.0}


def fbeta_from_pr(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta from precision and recall — the one definition of the run's objective.

    Thresholds are calibrated with it and labels are routed with it, so the two
    decisions cannot drift apart.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def _fbeta(y: np.ndarray, pred: np.ndarray, beta: float = 1.0) -> float:
    """F-beta; beta>1 favours recall (find more), beta<1 favours precision."""
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float(((~pred) & (y == 1)).sum())
    if not tp:
        return 0.0
    return fbeta_from_pr(tp / (tp + fp), tp / (tp + fn), beta)


class BaseModel:
    """Shared fit/predict plumbing; subclasses only define ``_encode``."""

    # 'mlp' or 'gb'; gradient boosting copes better with wide correlated inputs
    # (a kernel bank is highly correlated by construction) at this sample size.
    head = 'mlp'

    def __init__(self, label_names: list[str], duration: float = 0.02,
                 head: str = None, bias: str = 'balanced', **kw):
        self.label_names = list(label_names)
        self.duration = duration
        self.bias = bias   # 'discover' | 'balanced' | 'accurate'; see BIAS_BETA
        self.params = kw
        if head is not None:
            self.head = head
        self.scaler = StandardScaler()
        self.nets: list = []
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
        # Pin what was actually trained on: a highres-capable model fitted
        # without it emits narrow features and must be scored the same way.
        self.uses_highres = self.uses_highres and ccg_hi is not None
        X = self.scaler.fit_transform(self._encode(ccg, null, ccg_hi, null_hi))
        self.nets, self.constant = [], []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            # A label absent (or universal) in this fold has nothing to separate.
            if y.min() == y.max():
                self.nets.append(None)
                self.constant.append(float(y[0]))
                continue
            net = self._new_head()
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
        """Pick each label's threshold by maximizing F-beta on training scores.

        Labels here are rare (83–919 of 3115), so the natural 0.5 cut classifies
        almost everything negative. Thresholds are fit on training data only —
        an outer CV fold never contributes to its own threshold.
        """
        beta = BIAS_BETA[self.bias]
        grid = np.arange(0.05, 0.95, 0.01)
        for j in range(Y.shape[1]):
            y, p = Y[:, j], P[:, j]
            if y.min() == y.max():
                continue
            score = [_fbeta(y, p >= t, beta) for t in grid]
            self.thresholds[j] = grid[int(np.argmax(score))]

    def _new_head(self):
        """One binary classifier for one label, per ``self.head``."""
        if self.head == 'gb':
            kw = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                      l2_regularization=1.0, early_stopping=True,
                      validation_fraction=0.15, random_state=0)
            kw.update(self.params)
            return HistGradientBoostingClassifier(**kw)
        kw = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600,
                  early_stopping=True, n_iter_no_change=25, random_state=0)
        kw.update(self.params)
        return MLPClassifier(**kw)

    def predict_proba(self, ccg: np.ndarray, null: np.ndarray,
                      ccg_hi: np.ndarray = None,
                      null_hi: np.ndarray = None) -> np.ndarray:
        X = self._encode(ccg, null, ccg_hi, null_hi)
        return self._scores(self.scaler.transform(X))

    def save(self, path: str, provenance: dict = None):
        """Write the model plus a sidecar recording what produced it.

        The sidecar is the point: a trained model applied to another project is
        only interpretable if you can see which selections taught it, so the
        training sources travel with the weights.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)
        self.meta = {'model': type(self).__name__,
                     'model_key': self.model_key(),
                     'head': self.head,
                     'bias': self.bias,
                     'labels': self.label_names,
                     'duration': self.duration,
                     'thresholds': {n: float(t) for n, t
                                    in zip(self.label_names, self.thresholds)},
                     'saved_at': datetime.datetime.now().isoformat(),
                     'trained_on': provenance or getattr(self, 'meta', {}).get(
                         'trained_on', {})}
        with open(os.path.splitext(path)[0] + '.json', 'w') as fh:
            json.dump(self.meta, fh, indent=1)
        return path

    def model_key(self) -> str:
        """Registry name for this class (inverse of ``MODELS``)."""
        for name, cls in MODELS.items():
            if cls is type(self):
                return name
        return type(self).__name__

    @staticmethod
    def load(path: str) -> 'BaseModel':
        with open(path, 'rb') as fh:
            model = pickle.load(fh)
        side = os.path.splitext(path)[0] + '.json'
        if os.path.isfile(side):
            with open(side) as fh:
                model.meta = json.load(fh)
        return model

    def compatible_with(self, cd) -> list[str]:
        """Reasons this model cannot score *cd*'s CCGs; empty means it can.

        A model encodes traces of a fixed bin count and window, so applying one
        trained elsewhere is only valid when the target project was computed the
        same way. Checked explicitly rather than failing inside numpy.
        """
        problems = []
        if abs(float(cd.conf.duration) - float(self.duration)) > 1e-9:
            problems.append(f"window {cd.conf.duration * 1e3:.1f} ms "
                            f"!= trained {self.duration * 1e3:.1f} ms")
        trained = getattr(self, 'meta', {}).get('trained_on', {})
        for res, attr in (('n_bins_lowres', 'lowres'), ('n_bins_highres', 'highres')):
            want = trained.get(res)
            if want is None:
                continue
            got = _n_bins(cd, attr)
            if got is None:
                # The encoder emits a fixed width, so a resolution the model was
                # trained with must be present for every session it scores.
                problems.append(f"{attr} CCGs missing (model was trained with them)")
            elif got != want:
                problems.append(f"{attr} bins {got} != trained {want}")
        return problems


def _n_bins(cd, resolution: str) -> int | None:
    """Bin count at *resolution*, or None unless every pointer session has it."""
    widths = set()
    for key in cd.ptr:
        data = loaded_ccg(cd, key, resolution)
        if data is None:
            return None
        widths.add(int(data.ccg.shape[-1]))
    return widths.pop() if len(widths) == 1 else None


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


class KernelNet(BaseModel):
    # Same reasoning as ConvNet: 1012 correlated columns suit trees, not an MLP.
    head = 'gb'

    """Both resolutions read through a fixed bank of localized shape kernels.

    Replaces the PCA heads with matched filters — Gaussian derivatives at a grid
    of lags and widths — because PCA learns whole-window templates that maximize
    dataset variance, while visual inspection judges *local* features: a peak's
    width, a dip's depth, where a rift sits relative to 0 ms.

    Nothing in the bank is fitted, so it cannot overfit ~3000 samples, and every
    coefficient names a lag and a width. This is the older ``PeakRule`` vocabulary
    (lag range, FWHM, polarity, per-resolution) expressed as a continuous basis
    the net can weight instead of hand-set thresholds.
    """

    uses_highres = True

    def __init__(self, label_names, duration=0.02, lag_step_ms=0.5, **kw):
        super().__init__(label_names, duration=duration, **kw)
        self.lag_step_ms = lag_step_ms
        self._bank_lo = self._bank_hi = None

    def _bank(self, which: str, n_bin: int):
        """Cache one bank per resolution — it depends only on the bin count."""
        attr = f'_bank_{which}'
        bank = getattr(self, attr)
        if bank is None or bank.shape[1] != n_bin:
            bank, _meta = kernel_bank(n_bin, self.duration,
                                      lag_step_ms=self.lag_step_ms)
            setattr(self, attr, bank)
        return getattr(self, attr)

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        ccg = np.atleast_2d(ccg)
        parts = [kernel_response(ccg, null, self.duration,
                                 self._bank('lo', ccg.shape[1])),
                 shape_features(ccg, null, self.duration)]
        if ccg_hi is not None:
            ccg_hi = np.atleast_2d(ccg_hi)
            parts += [kernel_response(ccg_hi, null_hi, self.duration,
                                      self._bank('hi', ccg_hi.shape[1])),
                      shape_features(ccg_hi, null_hi, self.duration)]
        return np.hstack(parts)


class ConvNet(BaseModel):
    """Filters discovered from the data, then convolved and pooled per resolution.

    The literature's CNNs (CoNNECT and successors) learn their first-layer filters
    rather than fixing them, but they do it on ~80k *simulated* pairs. With ~3000
    real ones, learning a filter per lag would overfit, so the filters here are
    learned from sliding **patches**: one small filter is reused at every lag, and
    therefore trains on n_pos times more examples than a whole-window basis.

    That keeps the CNN's useful inductive bias — local, translation-covariant
    features — at a parameter count this dataset can actually support. Pooling
    keeps argmax-lag alongside max and mean, so *where* a feature occurred
    survives, which plain max-pooling would discard.
    """

    uses_highres = True
    # Boosted trees, not an MLP: the pooled bank is wide and correlated, which an
    # MLP handles poorly at this sample size (+0.06 F1 on a fixed kernel bank).
    head = 'gb'

    def __init__(self, label_names, duration=0.02, n_filters=16,
                 width_lo=7, width_hi=31, stride=2, **kw):
        super().__init__(label_names, duration=duration, **kw)
        self.n_filters = n_filters
        self.width_lo, self.width_hi, self.stride = width_lo, width_hi, stride
        self.bank_lo = self.bank_hi = None

    def _side(self, res, which: str, width: int):
        """Learn this side's bank on the first (training) call, then reuse it."""
        attr = f'bank_{which}'
        if getattr(self, attr) is None:
            setattr(self, attr, learned_bank(res, self.n_filters,
                                             min(width, res.shape[1]), self.stride))
        return bank_response(res, getattr(self, attr), self.stride)

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        ccg, null = np.atleast_2d(ccg), np.atleast_2d(null)
        res_lo = smooth(residual(ccg, null), 1.0)
        parts = [self._side(res_lo, 'lo', self.width_lo),
                 shape_features(ccg, null, self.duration)]
        if ccg_hi is not None:
            ccg_hi, null_hi = np.atleast_2d(ccg_hi), np.atleast_2d(null_hi)
            sigma = max(1.0, ccg_hi.shape[1] / ccg.shape[1] / 3)
            res_hi = smooth(residual(ccg_hi, null_hi), sigma)
            parts += [self._side(res_hi, 'hi', self.width_hi),
                      shape_features(ccg_hi, null_hi, self.duration)]
        return np.hstack(parts)


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


class RoutedNet(BaseModel):
    """Per-label routing: each label answered by the strategy that scores it best.

    The one-vs-rest heads inside a single model already learn each label
    separately, but they all read the *same* features, because ``_encode``
    belongs to the strategy. A quality grade turns on peak SNR while a rhythm
    label turns on structure at long lags, so one encoding cannot be right for
    both. This holds one fitted sub-model per strategy and takes each label's
    column from its winner, which is chosen by cross-validated F-beta.

    ``routes`` maps label -> strategy key. Labels with no entry fall back to the
    first sub-model, so a route table never has to be exhaustive. Everything a
    sub-model owns is read back from it rather than copied here: a threshold
    recalibrated on a sub-model is the threshold this model uses.
    """

    def __init__(self, label_names, duration=0.02, routes: dict = None, **kw):
        super().__init__(label_names, duration=duration, **kw)
        self.routes = dict(routes or {})
        self.subs: dict[str, BaseModel] = {}

    @property
    def uses_highres(self) -> bool:
        """True when any routed sub-model reads the fine bins."""
        return any(m.uses_highres for m in self.subs.values())

    @property
    def thresholds(self) -> np.ndarray:
        """Each label's cut, taken live from the sub-model that answers it."""
        return np.array([self.subs[k].thresholds[self.subs[k].label_names.index(n)]
                         for n, k in ((n, self._key_for(n))
                                      for n in self.label_names)])

    @thresholds.setter
    def thresholds(self, value):
        pass          # BaseModel.__init__ seeds a default; the sub-models own it

    def _key_for(self, label: str) -> str:
        """The strategy answering *label* — the one route-resolution rule."""
        key = self.routes.get(label)
        return key if key in self.subs else next(iter(self.subs))

    def _encode(self, ccg, null, ccg_hi=None, null_hi=None):
        raise NotImplementedError("RoutedNet delegates encoding to its sub-models")

    def _scores(self, X):
        raise NotImplementedError("RoutedNet scores through its sub-models")

    def fit(self, ccg, null, Y, ccg_hi=None, null_hi=None):
        for sub in self.subs.values():
            sub.fit(ccg, null, Y, ccg_hi, null_hi)
        return self

    def predict_proba(self, ccg, null, ccg_hi=None, null_hi=None):
        # Only strategies that actually answer a label are run: an unrouted
        # sub-model would encode and score every pair for nothing.
        used = {self._key_for(n) for n in self.label_names}
        cache = {k: self.subs[k].predict_proba(ccg, null, ccg_hi, null_hi)
                 for k in used}
        out = np.zeros((len(np.atleast_2d(ccg)), len(self.label_names)))
        for j, name in enumerate(self.label_names):
            key = self._key_for(name)
            out[:, j] = cache[key][:, self.subs[key].label_names.index(name)]
        return out

    def model_key(self) -> str:
        """Not a MODELS key: this model is a composition, not a registry entry."""
        return 'routed(' + '+'.join(self.subs) + ')'


MODELS = {'conv': ConvNet, 'kernel': KernelNet, 'dualres': DualResNet,
          'hybrid': HybridNet, 'shape': ShapeFeatureNet, 'trace': TraceNet,
          'twohead': TwoHeadNet}


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
