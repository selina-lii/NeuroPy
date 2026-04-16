"""
CCG shape-based pair classifiers.

Two parallel classifiers — both return ClassifyResult and NEVER modify the
caller's ground-truth group/deletion sets:

CCGClassifier
    Trains one binary LogisticRegression per group (plus a trash detector)
    from 12 hand-crafted CCG shape features.

CCGClusterClassifier
    Normalises the raw CCG waveform to unit L2 norm (shape-only), reduces
    dimensionality with PCA fitted on ALL local pairs, then assigns by
    nearest per-group centroid with softmax-distance confidence.
    Works well with rough/noisy high-res CCGs via optional Gaussian
    pre-smoothing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# sklearn is a soft dependency
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA
except ImportError:
    LogisticRegression = StandardScaler = make_pipeline = PCA = None  # type: ignore


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClassifyResult:
    """Prediction for a single pair."""
    pair: Tuple[int, int]
    action: str                         # 'assign' | 'delete' | 'review'
    groups: List[str] = field(default_factory=list)    # non-empty when action='assign'
    confidences: Dict[str, float] = field(default_factory=dict)  # group → probability
    trash_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CCGClassifier:
    """
    Lightweight CCG shape classifier.

    Parameters
    ----------
    ccg_data : CCGData
        Loaded CCGData object; provides `.ccg` and `.ccg_null`.
    conf : CCGConfig
        Config providing `bin_size`, `duration`, `center_bin`, `nbins`.
    """

    # Central window for peak/trough detection: ±5 ms expressed in seconds.
    _CENTRAL_HALF_MS = 5e-3   # 5 ms
    # Flank region starts this far from center.
    _FLANK_START_MS  = 25e-3  # 25 ms

    def __init__(self, ccg_data, conf):
        self._cd   = ccg_data
        self._conf = conf
        self._clfs: Dict[str, object]  = {}   # group_name → fitted LogisticRegression
        self._trash_clf                = None
        self._groups_fitted: List[str] = []

        # Infer bin params from actual array shape — conf.bin_size / conf.nbins /
        # conf.center_bin are all derived from bin_size and can be mutated by
        # load_highres().  Derive everything from the array and conf.duration only.
        nb = ccg_data.ccg.shape[-1]           # actual number of bins in this CCGData
        bs = (conf.duration / (nb - 1)        # robust bin_size from actual array
              if nb > 1 else conf.bin_size)
        cb = (nb - 1) // 2                    # center bin (lag=0) — always symmetric
        self._bin_ms = (np.arange(nb) - cb) * bs * 1e3   # lag in ms per bin index
        self._bs_ms  = bs * 1e3                           # bin size in ms

        # Central window (bin indices)
        half_bins = int(self._CENTRAL_HALF_MS / bs)
        self._cwin = slice(max(0, cb - half_bins), min(nb, cb + half_bins + 1))

        # Flank window (bin indices)
        flank_bins = int(self._FLANK_START_MS / bs)
        flank_lo = max(0, cb - flank_bins)
        flank_hi = min(nb, cb + flank_bins)
        self._flank_mask = np.ones(nb, dtype=bool)
        self._flank_mask[flank_lo:flank_hi] = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, ref: int, tgt: int) -> np.ndarray:
        """Return 12-element feature vector for pair (ref, tgt)."""
        cd = self._cd
        # All-segments average
        ccg_avg  = cd.ccg[:, ref, tgt, :].mean(axis=0).astype(float)
        null_avg = (cd.ccg_null[:, ref, tgt, :].mean(axis=0).astype(float)
                    if cd.ccg_null is not None else np.zeros_like(ccg_avg))
        residual = ccg_avg - null_avg
        return self._compute_features(residual)

    def fit(
        self,
        labeled_pairs: Dict[str, Set[Tuple[int, int]]],
        deleted_pairs: Set[Tuple[int, int]],
    ) -> Dict[str, int]:
        """
        Train one binary LogisticRegression per group and one trash detector.

        Parameters
        ----------
        labeled_pairs : dict  group_name → set of (ref, tgt) in current session
        deleted_pairs : set of (ref, tgt) treated as trash class

        Returns
        -------
        dict  group_name → number of positive training examples used
        """
        # Collect all labeled pairs (union of all groups + deleted)
        all_labeled: Set[Tuple[int, int]] = set()
        for pairs in labeled_pairs.values():
            all_labeled.update(pairs)
        all_labeled.update(deleted_pairs)

        if len(all_labeled) < 2:
            return {}

        # Feature matrix for every labeled pair
        pair_list = sorted(all_labeled)
        X = np.array([self.extract_features(r, t) for r, t in pair_list])
        pair_index = {p: i for i, p in enumerate(pair_list)}

        # NaN/inf guard
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        counts: Dict[str, int] = {}
        self._clfs = {}
        self._groups_fitted = []

        for gname, pos_pairs in labeled_pairs.items():
            pos = sorted(pos_pairs & all_labeled)
            neg = sorted(all_labeled - pos_pairs)
            if len(pos) < 1 or len(neg) < 1:
                continue
            y = np.array([1 if p in pos_pairs else 0 for p in pair_list])
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight='balanced',
                    solver='lbfgs',
                ),
            )
            clf.fit(X, y)
            self._clfs[gname] = clf
            self._groups_fitted.append(gname)
            counts[gname] = len(pos)

        # Trash detector
        if len(deleted_pairs) >= 1:
            neg_trash = sorted(all_labeled - deleted_pairs)
            if neg_trash:
                y_trash = np.array(
                    [1 if p in deleted_pairs else 0 for p in pair_list])
                trash_clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1000, C=1.0, class_weight='balanced',
                        solver='lbfgs',
                    ),
                )
                trash_clf.fit(X, y_trash)
                self._trash_clf = trash_clf

        return counts

    def predict(
        self,
        pairs: List[Tuple[int, int]],
        high_thresh: float = 0.75,
        trash_thresh: float = 0.80,
    ) -> List[ClassifyResult]:
        """
        Classify a list of (ref, tgt) pairs.

        Parameters
        ----------
        pairs       : pairs to classify
        high_thresh : minimum probability to confidently assign a group
        trash_thresh: minimum trash probability to mark for deletion

        Returns
        -------
        List of ClassifyResult — one per pair, in input order.
        """
        if not self._clfs and self._trash_clf is None:
            return [ClassifyResult(p, 'review') for p in pairs]

        results = []
        for pair in pairs:
            ref, tgt = pair
            try:
                feats = self.extract_features(ref, tgt)
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                feats2d = feats.reshape(1, -1)
            except Exception:
                results.append(ClassifyResult(pair, 'review'))
                continue

            # Trash score
            trash_score = 0.0
            if self._trash_clf is not None:
                trash_score = float(self._trash_clf.predict_proba(feats2d)[0, 1])

            # Group scores
            group_scores: Dict[str, float] = {}
            for gname, clf in self._clfs.items():
                group_scores[gname] = float(clf.predict_proba(feats2d)[0, 1])

            # Decision
            if trash_score >= trash_thresh:
                action = 'delete'
                assigned = []
            else:
                assigned = [g for g, s in group_scores.items() if s >= high_thresh]
                action = 'assign' if assigned else 'review'

            results.append(ClassifyResult(
                pair=pair,
                action=action,
                groups=assigned,
                confidences=group_scores,
                trash_confidence=trash_score,
            ))

        return results

    # ------------------------------------------------------------------
    # Serialisation helpers (called by UI)
    # ------------------------------------------------------------------

    @staticmethod
    def save_training_snapshot(
        path: str,
        labeled_pairs: Dict[str, Set[Tuple[int, int]]],
        deleted_pairs: Set[Tuple[int, int]],
        session: str,
        conn_type: str,
    ) -> None:
        """Save ground-truth labels to JSON before training."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'session': session,
            'conn_type': conn_type,
            'groups': {g: [[int(r), int(t)] for r, t in sorted(ps)]
                       for g, ps in labeled_pairs.items()},
            'deleted': [[int(r), int(t)] for r, t in sorted(deleted_pairs)],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def save_speculated(
        path: str,
        results: List[ClassifyResult],
        session: str,
        conn_type: str,
    ) -> None:
        """Save speculated labels to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'session': session,
            'conn_type': conn_type,
            'predictions': [
                {
                    'pair': [int(r.pair[0]), int(r.pair[1])],
                    'action': r.action,
                    'groups': r.groups,
                    'confidences': {g: float(v) for g, v in r.confidences.items()},
                    'trash_confidence': float(r.trash_confidence),
                }
                for r in results
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def load_speculated(path: str) -> List[ClassifyResult]:
        """Load a previously saved speculated file."""
        with open(path) as f:
            data = json.load(f)
        return [
            ClassifyResult(
                pair=tuple(p['pair']),
                action=p['action'],
                groups=p.get('groups', []),
                confidences=p.get('confidences', {}),
                trash_confidence=p.get('trash_confidence', 0.0),
            )
            for p in data.get('predictions', [])
        ]

    # ------------------------------------------------------------------
    # Internal feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, residual: np.ndarray) -> np.ndarray:
        nb      = len(residual)
        bs_ms   = self._bs_ms
        bin_ms  = self._bin_ms

        # Flank noise (background std)
        flank   = residual[self._flank_mask]
        flank_noise = float(np.std(flank)) if len(flank) > 1 else 1.0
        noise_floor = max(flank_noise, 1e-9)

        # ── Peak detection ──────────────────────────────────────────────
        cwin_residual = residual[self._cwin]
        cwin_bins     = bin_ms[self._cwin]

        peak_height_snr  = 0.0
        peak_lag         = 0.0
        peak_width_fwhm  = 0.0
        n_peaks          = 0
        peak_spacing_reg = 0.0   # 0 = perfectly rhythmic
        peak_overlap     = 0.0   # bool as float

        peak_idx, peak_props = self._find_peaks(cwin_residual, noise_floor, polarity=1)
        if peak_idx.size > 0:
            best  = int(np.argmax(cwin_residual[peak_idx]))
            h     = float(cwin_residual[peak_idx[best]]) / noise_floor
            lag   = float(cwin_bins[peak_idx[best]])
            w     = self._fwhm(cwin_residual, peak_idx[best], polarity=1) * bs_ms

            peak_height_snr = h
            peak_lag        = lag
            peak_width_fwhm = w
            n_peaks         = len(peak_idx)

            if n_peaks >= 2:
                spacings = np.diff(cwin_bins[peak_idx])
                peak_spacing_reg = (float(np.std(spacings) / (np.mean(spacings) + 1e-9))
                                    if len(spacings) > 0 else 0.0)
                peak_overlap = float(any(
                    peak_props['widths'][i] * bs_ms > spacings[i]
                    for i in range(len(spacings))
                ))

        # ── Trough detection ────────────────────────────────────────────
        trough_depth_snr = 0.0
        trough_lag       = 0.0
        trough_width_fwhm = 0.0

        trough_idx, _ = self._find_peaks(-cwin_residual, noise_floor, polarity=-1)
        if trough_idx.size > 0:
            best  = int(np.argmax(-cwin_residual[trough_idx]))
            d     = float(-cwin_residual[trough_idx[best]]) / noise_floor
            lag   = float(cwin_bins[trough_idx[best]])
            w     = self._fwhm(cwin_residual, trough_idx[best], polarity=-1) * bs_ms

            trough_depth_snr  = d
            trough_lag        = lag
            trough_width_fwhm = w

        # ── Rhythmicity (FFT of full residual) ──────────────────────────
        oscillation_freq  = 0.0
        oscillation_power = 0.0

        if nb >= 8:
            fft_mag   = np.abs(np.fft.rfft(residual))
            freqs     = np.fft.rfftfreq(nb, d=self._bs_ms * 1e-3)  # Hz
            # Ignore DC and very low frequencies (<5 Hz)
            valid     = (freqs >= 5.0) & (freqs <= 300.0)
            if valid.any():
                fft_valid = fft_mag[valid]
                freqs_v   = freqs[valid]
                broadband = float(np.mean(fft_valid)) + 1e-9
                best_f    = int(np.argmax(fft_valid))
                oscillation_freq  = float(freqs_v[best_f])
                oscillation_power = float(fft_valid[best_f]) / broadband

        return np.array([
            peak_lag,           # 0
            peak_height_snr,    # 1
            peak_width_fwhm,    # 2
            float(n_peaks),     # 3
            peak_spacing_reg,   # 4
            peak_overlap,       # 5
            trough_depth_snr,   # 6
            trough_lag,         # 7
            trough_width_fwhm,  # 8
            oscillation_freq,   # 9
            oscillation_power,  # 10
            flank_noise,        # 11
        ], dtype=float)

    @staticmethod
    def _find_peaks(
        x: np.ndarray,
        noise_floor: float,
        polarity: int = 1,
        min_prominence: float = 1.5,   # × noise_floor
    ) -> Tuple[np.ndarray, dict]:
        """
        Find local maxima in x that are at least min_prominence * noise_floor
        above the surrounding baseline.  Returns (indices, {'widths': widths}).
        """
        n = len(x)
        threshold = min_prominence * noise_floor
        peaks = []
        widths = []
        for i in range(1, n - 1):
            if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] >= threshold:
                peaks.append(i)
                # Crude width: bins where x > half-max
                hm = x[i] / 2.0
                lo, hi = i, i
                while lo > 0 and x[lo - 1] >= hm:
                    lo -= 1
                while hi < n - 1 and x[hi + 1] >= hm:
                    hi += 1
                widths.append(hi - lo)
        return np.array(peaks, dtype=int), {'widths': widths}

    @staticmethod
    def _fwhm(x: np.ndarray, idx: int, polarity: int = 1) -> float:
        """Full-width at half-maximum of feature at `idx`, in bins."""
        n  = len(x)
        hm = x[idx] / 2.0 * polarity
        lo, hi = idx, idx
        while lo > 0 and x[lo - 1] * polarity >= hm:
            lo -= 1
        while hi < n - 1 and x[hi + 1] * polarity >= hm:
            hi += 1
        return float(hi - lo) if hi > lo else 1.0


# ---------------------------------------------------------------------------
# Cluster-based classifier — PCA + Nearest Centroid on raw waveforms
# ---------------------------------------------------------------------------

class CCGClusterClassifier:
    """
    Shape-similarity classifier using PCA + Nearest-Centroid.

    Extracts the all-segments-averaged CCG residual (ccg - null), applies
    optional Gaussian smoothing, normalises to unit L2 norm (amplitude
    removed — shape only), then reduces with PCA fitted on ALL local pairs
    for the best possible embedding.  Assignment is by nearest per-group
    centroid in PCA space; confidence is a softmax over negative distances.

    Parameters
    ----------
    ccg_data    : CCGData
    conf        : CCGConfig
    smooth_sigma: float
        Gaussian smoothing sigma in **bins** applied before normalisation.
        Default 1.0 — tames rough high-res peaks without blurring structure.
        Set to 0 to disable.
    """

    def __init__(self, ccg_data, conf, smooth_sigma: float = 1.0):
        self._cd          = ccg_data
        self._conf        = conf
        self._smooth      = float(smooth_sigma)
        self._pca         = None
        self._centroids: Dict[str, np.ndarray]  = {}   # group → PCA embedding
        self._trash_centroid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_waveform(self, ref: int, tgt: int) -> np.ndarray:
        """
        Return normalised CCG residual waveform for pair (ref, tgt).

        Steps: average across segments → subtract null baseline →
        optional Gaussian smooth → unit L2 normalise.
        """
        cd = self._cd
        ccg_avg  = cd.ccg[:, ref, tgt, :].mean(axis=0).astype(float)
        null_avg = (cd.ccg_null[:, ref, tgt, :].mean(axis=0).astype(float)
                    if cd.ccg_null is not None else np.zeros_like(ccg_avg))
        residual = ccg_avg - null_avg

        if self._smooth > 0:
            residual = gaussian_filter1d(residual, sigma=self._smooth)

        norm = float(np.linalg.norm(residual))
        if norm > 1e-12:
            residual = residual / norm
        return residual

    def fit(
        self,
        labeled_pairs: Dict[str, Set[Tuple[int, int]]],
        deleted_pairs: Set[Tuple[int, int]],
        all_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, int]:
        """
        Fit PCA on ``all_pairs`` (or labeled + deleted if None), then compute
        per-group centroids in PCA space.

        Parameters
        ----------
        labeled_pairs : group_name → set of (ref, tgt) in current session
        deleted_pairs : set of (ref, tgt) treated as trash
        all_pairs     : full pair list to fit PCA on; pass list(ptr.inds2) for
                        broadest coverage.  Falls back to labeled + deleted.

        Returns
        -------
        dict  group_name → number of positive training examples
        """
        # ── Collect pairs for PCA fitting ────────────────────────────────
        labeled_set: Set[Tuple[int, int]] = set()
        for ps in labeled_pairs.values():
            labeled_set.update(ps)
        labeled_set.update(deleted_pairs)

        pca_pairs: List[Tuple[int, int]] = (
            list(all_pairs) if all_pairs is not None else sorted(labeled_set))
        if not pca_pairs:
            return {}

        # ── Build waveform matrix ─────────────────────────────────────────
        X = np.array([self.extract_waveform(r, t) for r, t in pca_pairs],
                     dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        pair_to_idx = {p: i for i, p in enumerate(pca_pairs)}

        # ── Fit PCA ───────────────────────────────────────────────────────
        n_labeled = max(len(labeled_set), 1)
        n_comp = max(1, min(20, n_labeled - 1, X.shape[0] - 1, X.shape[1]))
        self._pca = PCA(n_components=n_comp)
        X_pca = self._pca.fit_transform(X)   # (n_all_pairs, n_comp)

        # ── Per-group centroids ───────────────────────────────────────────
        counts: Dict[str, int] = {}
        self._centroids = {}
        for gname, pos_pairs in labeled_pairs.items():
            idx = [pair_to_idx[p] for p in pos_pairs if p in pair_to_idx]
            if not idx:
                continue
            self._centroids[gname] = X_pca[idx].mean(axis=0)
            counts[gname] = len(idx)

        # ── Trash centroid ────────────────────────────────────────────────
        self._trash_centroid = None
        if deleted_pairs:
            idx = [pair_to_idx[p] for p in deleted_pairs if p in pair_to_idx]
            if idx:
                self._trash_centroid = X_pca[idx].mean(axis=0)

        return counts

    def predict(
        self,
        pairs: List[Tuple[int, int]],
        high_thresh: float = 0.75,
        trash_thresh: float = 0.80,
    ) -> List['ClassifyResult']:
        """
        Classify pairs by nearest centroid in PCA space.

        Confidence is softmax of negative Euclidean distances to centroids
        (including trash).  Returns same ``ClassifyResult`` format as
        ``CCGClassifier``.
        """
        if self._pca is None or not self._centroids:
            return [ClassifyResult(p, 'review') for p in pairs]

        group_names = list(self._centroids.keys())
        all_names   = group_names + (['__trash__']
                                     if self._trash_centroid is not None else [])
        all_centroids = np.array(
            [self._centroids[g] for g in group_names]
            + ([self._trash_centroid] if self._trash_centroid is not None else [])
        )  # (n_groups[+1], n_comp)

        results: List['ClassifyResult'] = []
        for pair in pairs:
            ref, tgt = pair
            try:
                wav = self.extract_waveform(ref, tgt)
                wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
                emb = self._pca.transform(wav.reshape(1, -1))[0]   # (n_comp,)
            except Exception:
                results.append(ClassifyResult(pair, 'review'))
                continue

            # Euclidean distance to every centroid
            diffs = all_centroids - emb[np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=1)          # (n_all,)

            # Softmax of negative distances (temperature = 1 in PCA space)
            neg = -dists
            neg -= neg.max()                                # numerical stability
            exp_neg = np.exp(neg)
            probs = exp_neg / (exp_neg.sum() + 1e-300)

            group_probs = {g: float(probs[i]) for i, g in enumerate(group_names)}
            trash_prob  = (float(probs[len(group_names)])
                           if self._trash_centroid is not None else 0.0)

            # Decision
            if trash_prob >= trash_thresh:
                action   = 'delete'
                assigned = []
            else:
                assigned = [g for g, p in group_probs.items() if p >= high_thresh]
                action   = 'assign' if assigned else 'review'

            results.append(ClassifyResult(
                pair=pair,
                action=action,
                groups=assigned,
                confidences=group_probs,
                trash_confidence=trash_prob,
            ))

        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_embedding(
        self,
        labeled_pairs: Dict[str, Set[Tuple[int, int]]],
        deleted_pairs: Set[Tuple[int, int]],
        unlabeled_pairs: Optional[List[Tuple[int, int]]] = None,
        title: str = '',
    ):
        """
        Plot PCA embedding of all pairs coloured by group label.

        Must be called after ``fit()``.  Returns a ``matplotlib.figure.Figure``.

        Parameters
        ----------
        labeled_pairs   : group_name → set of (ref, tgt) — training set
        deleted_pairs   : set of (ref, tgt) — trash class
        unlabeled_pairs : optional list of (ref, tgt) to show in gray
        title           : figure suptitle (e.g. session + conn_type)
        """
        if self._pca is None:
            raise RuntimeError("Call fit() before plot_embedding().")

        # ── Collect all pairs and their labels ───────────────────────────
        all_group_pairs: List[Tuple[int, int]] = []
        pair_labels:     List[str]             = []

        for gname, ps in labeled_pairs.items():
            for p in sorted(ps):
                all_group_pairs.append(p)
                pair_labels.append(gname)
        for p in sorted(deleted_pairs):
            all_group_pairs.append(p)
            pair_labels.append('__trash__')

        unlabeled = list(unlabeled_pairs) if unlabeled_pairs else []

        # ── Embed ────────────────────────────────────────────────────────
        def _embed(pairs):
            X = np.array([self.extract_waveform(r, t) for r, t in pairs],
                         dtype=float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return self._pca.transform(X)

        emb_labeled   = _embed(all_group_pairs) if all_group_pairs else None
        emb_unlabeled = _embed(unlabeled)        if unlabeled      else None

        # ── Colour map ───────────────────────────────────────────────────
        group_names = sorted(labeled_pairs.keys())
        has_trash   = bool(deleted_pairs)
        all_keys    = group_names + (['__trash__'] if has_trash else [])

        cmap   = cm.get_cmap('tab10', max(len(all_keys), 1))
        colors = {k: cmap(i) for i, k in enumerate(all_keys)}

        # ── Figure: PC1 vs PC2 (+ PC3 if available) ─────────────────────
        n_comp   = self._pca.n_components_
        ev_ratio = self._pca.explained_variance_ratio_

        n_plots = 1 + (1 if n_comp >= 3 else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5),
                                 squeeze=False)
        ax0 = axes[0, 0]

        # Unlabeled background
        if emb_unlabeled is not None and len(emb_unlabeled):
            ax0.scatter(emb_unlabeled[:, 0], emb_unlabeled[:, 1],
                        c='#CCCCCC', s=18, linewidths=0, alpha=0.4,
                        zorder=1, label='unlabeled')

        # Labeled points
        if emb_labeled is not None:
            for i, (pair, lbl) in enumerate(zip(all_group_pairs, pair_labels)):
                marker = 'x' if lbl == '__trash__' else 'o'
                ax0.scatter(emb_labeled[i, 0], emb_labeled[i, 1],
                            c=[colors[lbl]], s=55, marker=marker,
                            linewidths=1.2, edgecolors='k' if marker == 'o' else None,
                            alpha=0.85, zorder=3,
                            label=lbl if i == pair_labels.index(lbl) else '_')

        # Centroids
        for gname, centroid in self._centroids.items():
            ax0.scatter(centroid[0], centroid[1],
                        c=[colors[gname]], s=250, marker='*',
                        edgecolors='k', linewidths=0.8, zorder=5)
        if self._trash_centroid is not None:
            ax0.scatter(self._trash_centroid[0], self._trash_centroid[1],
                        c=[colors['__trash__']], s=250, marker='*',
                        edgecolors='k', linewidths=0.8, zorder=5)

        ax0.set_xlabel(f'PC1 ({ev_ratio[0]:.1%})')
        ax0.set_ylabel(f'PC2 ({ev_ratio[1]:.1%})')
        ax0.set_title('PC1 vs PC2')
        # Deduplicated legend
        handles, labels_ = ax0.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels_):
            if l not in seen:
                seen[l] = h
        ax0.legend(seen.values(), seen.keys(), fontsize=8, markerscale=0.9)

        # Optional PC1 vs PC3
        if n_comp >= 3:
            ax1 = axes[0, 1]
            if emb_unlabeled is not None and len(emb_unlabeled):
                ax1.scatter(emb_unlabeled[:, 0], emb_unlabeled[:, 2],
                            c='#CCCCCC', s=18, linewidths=0, alpha=0.4, zorder=1)
            if emb_labeled is not None:
                for i, (_, lbl) in enumerate(zip(all_group_pairs, pair_labels)):
                    marker = 'x' if lbl == '__trash__' else 'o'
                    ax1.scatter(emb_labeled[i, 0], emb_labeled[i, 2],
                                c=[colors[lbl]], s=55, marker=marker,
                                linewidths=1.2,
                                edgecolors='k' if marker == 'o' else None,
                                alpha=0.85, zorder=3)
            for gname, centroid in self._centroids.items():
                ax1.scatter(centroid[0], centroid[2],
                            c=[colors[gname]], s=250, marker='*',
                            edgecolors='k', linewidths=0.8, zorder=5)
            if self._trash_centroid is not None:
                ax1.scatter(self._trash_centroid[0], self._trash_centroid[2],
                            c=[colors['__trash__']], s=250, marker='*',
                            edgecolors='k', linewidths=0.8, zorder=5)
            ax1.set_xlabel(f'PC1 ({ev_ratio[0]:.1%})')
            ax1.set_ylabel(f'PC3 ({ev_ratio[2]:.1%})')
            ax1.set_title('PC1 vs PC3')

        ev_total = float(ev_ratio[:min(n_comp, 5)].sum())
        suptitle = f"{title}\n" if title else ''
        suptitle += (f"PCA — top-{min(n_comp,5)} PCs explain {ev_total:.1%} variance  |  "
                     f"n_labeled={len(all_group_pairs)}  "
                     f"n_unlabeled={len(unlabeled)}")
        fig.suptitle(suptitle, fontsize=9)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Template-based classifier — rule-driven, no training data required
# ---------------------------------------------------------------------------

@dataclass
class PeakRule:
    """One peak criterion within a GroupTemplate."""
    lag_min:   float                    # ms — left edge of acceptable lag range
    lag_max:   float                    # ms — right edge
    width_min: Optional[float] = None   # ms FWHM lower bound (None = unconstrained)
    width_max: Optional[float] = None   # ms FWHM upper bound
    polarity:  int             = 1      # 1=peak, -1=trough, 0=baseline_low, 2=not_in
    required:  bool            = True   # False = optional; adds bonus but not penalised
    resolution: str            = 'both' # 'hi', 'lo', or 'both'
    conn_type:  str            = '(all)'# conn_type string or '(all)'
    ref_group:  str            = ''     # for polarity==2: name of group to exclude
    smooth_ms:  Optional[float]= None   # per-rule Gaussian smooth; None = use global
    min_count:  Optional[float]= None   # for polarity==5 (min_bincount): raw spike count threshold


@dataclass
class GroupTemplate:
    """Template for one CCG group defined by peak rules."""
    name:                     str
    peak_rules:               List[PeakRule]
    n_peaks_min:              Optional[int]  = None   # total peaks detected (None = any)
    n_peaks_max:              Optional[int]  = None
    baseline_low_before_zero: bool           = False  # counts at lag<0 expected low
    tier:                     str            = 'primary'  # 'primary' | 'secondary'


def ccgconfig_to_main_template(conf) -> 'GroupTemplate':
    """
    Auto-generate a read-only 'Main' GroupTemplate from a CCGConfig.

    """
    half_ms = (conf.spkcnt_scope / 2) * 1000
    rules = [
        PeakRule(
            lag_min=conf.min_lag * 1000,
            lag_max=conf.max_lag * 1000,
            polarity=1,
            required=True,
            conn_type='(all)',
        ),
        PeakRule(
            lag_min=-half_ms,
            lag_max=half_ms,
            polarity=5,
            required=True,
            conn_type='(all)',
            min_count=conf.min_spkcount,
        ),
    ]
    return GroupTemplate(
        name='Main',
        peak_rules=rules,
        tier='primary',
    )


class CCGTemplateClassifier:
    """
    Rule-based CCG classifier.

    Each group is described by a ``GroupTemplate``: one or more ``PeakRule``
    entries (lag range, FWHM bounds, polarity) plus optional constraints on the
    total number of peaks and on whether the baseline before t=0 is expected to
    be low.

    Peak detection runs on a Gaussian-smoothed all-segments-averaged CCG
    (``smooth_ms`` parameter, default 2 ms).  Raw CCG is not modified.
    ``scipy.signal.find_peaks`` + ``peak_widths`` are used for robustness.

    Parameters
    ----------
    ccg_data  : CCGData
    conf      : CCGConfig
    smooth_ms : float
        Gaussian smoothing sigma in **milliseconds** before peak detection.
        Default 2.0 — matches ``gaussian_filter1d(x, sigma=2.0)`` at 1 ms bins.
        Converted to bins internally using the actual array resolution.
    """

    # Optional penalty multiplier when baseline_low_before_zero fails:
    # score is multiplied by this factor (< 1 degrades match).
    _BASELINE_PENALTY = 0.5

    # Cosine similarity threshold: a rule "matches" if sim > this value.
    # Typical range: 0.85–0.99 = near-identical shape; 0.3–0.6 = moderate match.
    _CONV_MATCH_THRESH: float = 0.3

    def __init__(self, ccg_data, conf, smooth_ms: float = 2.0,
                 resolution: str = 'both', conn_type_str: str = '(all)'):
        self._cd        = ccg_data
        self._conf      = conf
        self._templates: Dict[str, GroupTemplate] = {}
        self._resolution    = resolution      # 'hi', 'lo', or 'both'
        self._conn_type_str = conn_type_str   # e.g. 'pyr-pyr' or '(all)'

        # Derive bin size from actual array so loading hi-res never breaks this.
        nb = ccg_data.ccg.shape[-1]
        bs = conf.duration / (nb - 1) if nb > 1 else conf.bin_size
        self._bs_ms     = bs * 1e3
        self._cb        = (nb - 1) // 2
        self._nb        = nb
        self._bin_ms    = (np.arange(nb) - self._cb) * self._bs_ms
        self._sigma_bins = float(smooth_ms) / self._bs_ms if self._bs_ms > 0 else 0.0

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def load_templates(self, templates: Dict[str, 'GroupTemplate']) -> None:
        self._templates = dict(templates)

    def save_templates(self, path: str, session: str = '', conn_type: str = '',
                       smooth_ms: float = None, classify_with: list = None) -> None:
        """Serialise all templates to JSON."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = {
            'session':       session,
            'conn_type':     conn_type,
            'smooth_ms':     smooth_ms,
            'classify_with': classify_with,
            'templates': {
                name: {
                    'peak_rules': [
                        {'lag_min':    r.lag_min,
                         'lag_max':    r.lag_max,
                         'width_min':  r.width_min,
                         'width_max':  r.width_max,
                         'polarity':   r.polarity,
                         'required':   r.required,
                         'resolution': r.resolution,
                         'conn_type':  r.conn_type,
                         'ref_group':  r.ref_group,
                         'smooth_ms':  r.smooth_ms,
                         'min_count':  r.min_count}
                        for r in tmpl.peak_rules
                    ],
                    'n_peaks_min':              tmpl.n_peaks_min,
                    'n_peaks_max':              tmpl.n_peaks_max,
                    'baseline_low_before_zero': tmpl.baseline_low_before_zero,
                }
                for name, tmpl in self._templates.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def load_templates_from_file(path: str) -> Dict[str, 'GroupTemplate']:
        """Load templates from a JSON file; returns empty dict on any error."""
        try:
            with open(path) as f:
                data = json.load(f)
            result: Dict[str, GroupTemplate] = {}
            for name, td in data.get('templates', {}).items():
                rules = [
                    PeakRule(
                        lag_min=r['lag_min'],
                        lag_max=r['lag_max'],
                        width_min=r.get('width_min'),
                        width_max=r.get('width_max'),
                        polarity=r.get('polarity', 1),
                        required=r.get('required', True),
                        resolution=r.get('resolution', 'both'),
                        conn_type=r.get('conn_type', '(all)'),
                        ref_group=r.get('ref_group', ''),
                        smooth_ms=r.get('smooth_ms'),
                        min_count=r.get('min_count'),
                    )
                    for r in td.get('peak_rules', [])
                ]
                result[name] = GroupTemplate(
                    name=name,
                    peak_rules=rules,
                    n_peaks_min=td.get('n_peaks_min'),
                    n_peaks_max=td.get('n_peaks_max'),
                    baseline_low_before_zero=td.get('baseline_low_before_zero', False),
                )
            return result
        except Exception:
            return {}

    @staticmethod
    def save_templates_to_file(path: str, templates: Dict[str, 'GroupTemplate'],
                               smooth_ms: float = None,
                               classify_with: list = None) -> None:
        """Serialise a templates dict to JSON without requiring a live instance."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = {
            'smooth_ms':     smooth_ms,
            'classify_with': classify_with,
            'templates': {
                name: {
                    'peak_rules': [
                        {'lag_min':    r.lag_min,
                         'lag_max':    r.lag_max,
                         'width_min':  r.width_min,
                         'width_max':  r.width_max,
                         'polarity':   r.polarity,
                         'required':   r.required,
                         'resolution': r.resolution,
                         'conn_type':  r.conn_type,
                         'ref_group':  r.ref_group,
                         'smooth_ms':  r.smooth_ms,
                         'min_count':  r.min_count}
                        for r in tmpl.peak_rules
                    ],
                    'n_peaks_min':              tmpl.n_peaks_min,
                    'n_peaks_max':              tmpl.n_peaks_max,
                    'baseline_low_before_zero': tmpl.baseline_low_before_zero,
                }
                for name, tmpl in templates.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def load_file_metadata(path: str) -> dict:
        """Return smooth_ms and classify_with stored in a templates file."""
        try:
            with open(path) as f:
                data = json.load(f)
            return {
                'smooth_ms':     data.get('smooth_ms'),
                'classify_with': data.get('classify_with'),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smooth_ccg(self, ref: int, tgt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(ccg_raw, ccg_smooth)`` for pair (ref, tgt), both 1-D arrays
        of length n_bins.  ccg_raw is the all-segments average; ccg_smooth has
        Gaussian smoothing applied (sigma = smooth_ms / bs_ms bins).
        """
        cd = self._cd
        ccg_raw = cd.ccg[:, ref, tgt, :].mean(axis=0).astype(float)
        if self._sigma_bins > 0:
            ccg_smooth = gaussian_filter1d(ccg_raw, sigma=self._sigma_bins)
        else:
            ccg_smooth = ccg_raw.copy()
        return ccg_raw, ccg_smooth

    def score_pair(self, ref: int, tgt: int) -> Dict[str, float]:
        """
        Compute a match score in [0, 1] for every loaded template.

        Peak/trough rules use local find_peaks + peak_widths (detect on smoothed,
        measure FWHM on raw) so the score is not diluted by CCG features at other lags.

        Score = 1.0 if all required rules matched, 0.0 if any required rule fails.
        Optional rules each add 0.1 (capped at 1.0).

        Returns ``{group_name: score}``.
        """
        ccg_raw, ccg_smooth = self.smooth_ccg(ref, tgt)
        bin_ms = self._bin_ms

        scores: Dict[str, float] = {}
        for name, tmpl in self._templates.items():
            if (not tmpl.peak_rules and tmpl.n_peaks_min is None
                    and tmpl.n_peaks_max is None
                    and not tmpl.baseline_low_before_zero):
                continue   # no criteria defined — skip entirely
            score = self._score_template(tmpl, ccg_raw, ccg_smooth, bin_ms)
            scores[name] = score
        return scores

    def score_pair_detail(
        self, ref: int, tgt: int,
    ) -> Dict[str, dict]:
        """
        Like score_pair but returns per-rule breakdown for each template.

        Returns ``{group_name: {'score': float, 'rules': [{'label': str,
        'sim': float, 'matched': bool, 'required': bool}, ...]}}``
        """
        ccg_raw, ccg_smooth = self.smooth_ccg(ref, tgt)
        bin_ms = self._bin_ms

        result: Dict[str, dict] = {}
        for name, tmpl in self._templates.items():
            if (not tmpl.peak_rules and tmpl.n_peaks_min is None
                    and tmpl.n_peaks_max is None
                    and not tmpl.baseline_low_before_zero):
                continue   # no criteria defined — skip entirely
            details: list = []
            score = self._score_template(tmpl, ccg_raw, ccg_smooth, bin_ms, _details=details)
            rule_rows = []
            for d in details:
                rule = d['rule']
                pol_str = {1: 'peak', -1: 'trough', 0: 'baseline_low',
                           2: f'not_in({rule.ref_group})',
                           3: 'trough-peak', 4: 'peak-trough',
                           5: 'min_bincount', 6: 'min_bincount_avg'}.get(rule.polarity, '?')
                label = (f"{pol_str} [{rule.lag_min:.1f},{rule.lag_max:.1f}]ms"
                         f"{'(req)' if rule.required else ''}")
                rule_rows.append({
                    'label':    label,
                    'sim':      d['sim'],
                    'matched':  d['matched'],
                    'required': rule.required,
                })
            result[name] = {'score': score, 'rules': rule_rows}
        return result

    def predict(
        self,
        pairs: List[Tuple[int, int]],
        threshold: float = 0.80,
    ) -> List[ClassifyResult]:
        """
        Classify ``pairs`` against all loaded templates.

        Parameters
        ----------
        pairs     : list of (ref, tgt)
        threshold : minimum score to assign a group (0–1)

        Returns ``List[ClassifyResult]``.  ``confidences`` contains the raw
        scores for every template (interpretable as fraction of rules matched).
        """
        if not self._templates:
            return [ClassifyResult(p, 'review') for p in pairs]

        results: List[ClassifyResult] = []
        for pair in pairs:
            ref, tgt = pair
            try:
                scores = self.score_pair(ref, tgt)
            except Exception:
                results.append(ClassifyResult(pair, 'review'))
                continue

            assigned = [g for g, s in scores.items() if s >= threshold]
            action   = 'assign' if assigned else 'review'
            results.append(ClassifyResult(
                pair=pair,
                action=action,
                groups=assigned,
                confidences=scores,
                trash_confidence=0.0,
            ))
        return results

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_template(
        self,
        tmpl: 'GroupTemplate',
        ccg_raw: np.ndarray,
        ccg_smooth: np.ndarray,
        bin_ms: np.ndarray,
        _details: Optional[list] = None,   # if provided, per-rule dicts appended here
    ) -> float:
        # Global dynamic range — used as a significance floor for window excursions.
        dyn      = float(ccg_smooth.max() - ccg_smooth.min())
        peak_val = float(ccg_smooth.max()) if ccg_smooth.max() > 0 else 1.0
        # Minimum excursion a window feature must have to be considered real:
        # at least 10% of the global CCG swing.
        _SIG = 0.10 * dyn if dyn > 0 else 0.01

        # Global CCG baseline from far flanks (|lag| > 20 ms).  Used inside
        # _window_feature so that peaks/troughs must actually cross the resting
        # level — window-edge baseline is unreliable when a large feature sits
        # adjacent to the rule window (e.g. a peak's shoulder depresses the
        # edge_mean for a trough check, producing a spurious excursion).
        _far = np.abs(bin_ms) > 20.0
        _global_baseline = (float(np.median(ccg_smooth[_far]))
                            if _far.sum() >= 5 else float(np.median(ccg_smooth)))

        def _window_feature(polarity: int, lag_min: float, lag_max: float,
                            rule_smooth_ms: Optional[float] = None):
            """
            Find the strongest peak (polarity=1) or trough (-1) within
            [lag_min, lag_max] independently of features outside the window.
            Returns (lag_ms, fwhm_ms, excursion) or None if nothing significant.
            Both detection and FWHM measurement use the per-rule smoothed signal.
            Excursion is relative to the global far-lag baseline, not window edges.
            """
            # Use rule-specific smooth if given, else fall back to global ccg_smooth
            if rule_smooth_ms is not None:
                _s = rule_smooth_ms / self._bs_ms
                _sig = gaussian_filter1d(ccg_raw.astype(float), _s) if _s > 0 else ccg_raw.astype(float)
            else:
                _sig = ccg_smooth
            mask = (bin_ms >= lag_min) & (bin_ms <= lag_max)
            if not mask.any():
                return None
            sig_win = _sig[mask]
            g_idx   = np.where(mask)[0]              # global indices for this window
            # Excursion measured from the global baseline (not window edges).
            # For peaks: candidate must be above baseline; for troughs: below it.
            if polarity == 1:
                feat_i    = int(sig_win.argmax())
                excursion = float(sig_win[feat_i]) - _global_baseline
            else:
                feat_i    = int(sig_win.argmin())
                excursion = _global_baseline - float(sig_win[feat_i])
            if excursion < _SIG:                     # not a real feature
                return None
            global_i = int(g_idx[feat_i])
            lag       = float(bin_ms[global_i])
            # FWHM on the same smoothed signal used for detection.
            # Measuring on ccg_raw is wrong for noisy hi-res CCGs: noise fluctuations
            # drop below half-height within 1-2 bins, giving artificially narrow FWHM
            # that makes any Wmax check trivially pass even for a 3ms-wide peak.
            sig_clip = np.clip(_sig if polarity == 1 else -_sig, 0.0, None)
            try:
                fwhm_ms = float(peak_widths(sig_clip, [global_i], rel_height=0.5)[0][0]
                                 * self._bs_ms)
            except Exception:
                fwhm_ms = float('inf')   # unmeasurable → don't apply width constraint
            return lag, fwhm_ms, excursion

        n_required    = sum(1 for r in tmpl.peak_rules if r.required)
        n_opt_matched = 0
        rule_details: List[dict] = []
        n_req_matched = 0

        for rule in tmpl.peak_rules:
            # Skip rules that don't apply to this classifier's resolution/conn_type
            if rule.resolution != 'both' and self._resolution != 'both':
                if rule.resolution != self._resolution:
                    continue
            if rule.conn_type not in ('(all)', 'Excitatory', 'Inhibitory'):
                if self._conn_type_str not in ('(all)',) and rule.conn_type != self._conn_type_str:
                    continue
            elif rule.conn_type == 'Excitatory' and self._conn_type_str != '(all)':
                _ct_tuple = tuple(self._conn_type_str.split('-'))
                if _ct_tuple not in [tuple(c) for c in getattr(self._conf, 'conn_types_E', [])]:
                    continue
            elif rule.conn_type == 'Inhibitory' and self._conn_type_str != '(all)':
                _ct_tuple = tuple(self._conn_type_str.split('-'))
                if _ct_tuple not in [tuple(c) for c in getattr(self._conf, 'conn_types_I', [])]:
                    continue

            if rule.polarity == 5:
                # min_bincount: sum of raw CCG in [lag_min, lag_max] >= min_count
                mask  = (bin_ms >= rule.lag_min) & (bin_ms <= rule.lag_max)
                total = float(ccg_raw[mask].sum()) if mask.any() else 0.0
                threshold = rule.min_count if rule.min_count is not None else 0.0
                matched = total >= threshold
                sim     = 1.0 if matched else float(np.clip(total / threshold, 0.0, 0.99)
                                                     if threshold > 0 else 0.0)
            elif rule.polarity == 6:
                # min_bincount_avg: mean of raw CCG in [lag_min, lag_max] >= min_count
                mask  = (bin_ms >= rule.lag_min) & (bin_ms <= rule.lag_max)
                avg   = float(ccg_raw[mask].mean()) if mask.any() else 0.0
                threshold = rule.min_count if rule.min_count is not None else 0.0
                matched = avg >= threshold
                sim     = 1.0 if matched else float(np.clip(avg / threshold, 0.0, 0.99)
                                                     if threshold > 0 else 0.0)
            elif rule.polarity == 0:
                # baseline_low: mean in lag range must be < 50% of peak value
                mask = (bin_ms >= rule.lag_min) & (bin_ms <= rule.lag_max)
                ratio = float(ccg_smooth[mask].mean()) / peak_val if mask.any() else 1.0
                sim   = float(np.clip(1.0 - ratio, 0.0, 1.0))
                matched = ratio < 0.5
            elif rule.polarity == 2:
                # not_in: evaluate only the non-"not in" rules of the referenced
                # template to avoid circular dependency.
                ref_tmpl = self._templates.get(rule.ref_group)
                if ref_tmpl is not None:
                    filtered = GroupTemplate(
                        name=ref_tmpl.name,
                        peak_rules=[r for r in ref_tmpl.peak_rules
                                    if r.polarity != 2],
                        n_peaks_min=ref_tmpl.n_peaks_min,
                        n_peaks_max=ref_tmpl.n_peaks_max,
                        baseline_low_before_zero=ref_tmpl.baseline_low_before_zero,
                    )
                    sim = self._score_template(filtered, ccg_raw, ccg_smooth, bin_ms)
                else:
                    sim = 0.0
                matched = sim < 0.5
            elif rule.polarity in (3, 4):
                # Biphasic: need both a peak and a trough within [lag_min, lag_max],
                # ordered correctly — trough before peak (3) or peak before trough (4).
                p_feat = _window_feature( 1, rule.lag_min, rule.lag_max, rule.smooth_ms)
                t_feat = _window_feature(-1, rule.lag_min, rule.lag_max, rule.smooth_ms)
                if p_feat is None or t_feat is None:
                    sim, matched = 0.0, False
                else:
                    p_lag, t_lag = p_feat[0], t_feat[0]
                    # trough-peak (3): trough must be at earlier lag than peak
                    ordered = (t_lag < p_lag) if rule.polarity == 3 else (p_lag < t_lag)
                    sim     = 1.0 if ordered else 0.3
                    matched = ordered
            else:
                # polarity 1 (peak) or -1 (trough): window-local detection.
                # Each rule evaluates its own lag window independently — no
                # global prominence competition from adjacent larger peaks.
                feat = _window_feature(rule.polarity, rule.lag_min, rule.lag_max,
                                       rule.smooth_ms)
                if feat is None:
                    sim, matched = 0.0, False
                else:
                    _, fwhm_ms, _ = feat
                    w_ok = ((rule.width_min is None or fwhm_ms >= rule.width_min) and
                            (rule.width_max is None or fwhm_ms <= rule.width_max))
                    matched = w_ok
                    sim = 1.0 if matched else 0.5   # 0.5 = found but wrong FWHM

            rule_details.append({'rule': rule, 'sim': sim, 'matched': matched})

            if rule.required:
                n_req_matched += int(matched)
            else:
                n_opt_matched += int(matched)

        # ALL required rules must match — any failure vetoes the template entirely
        if n_required > 0 and n_req_matched < n_required:
            if _details is not None:
                _details.extend(rule_details)
            return 0.0

        # Optional bonus (capped so total ≤ 1.0)
        score = min(1.0, 1.0 + 0.1 * n_opt_matched)

        # n_peaks total constraint — only counting, so find_peaks is fine here
        if tmpl.n_peaks_min is not None or tmpl.n_peaks_max is not None:
            try:
                from scipy.signal import find_peaks
                pf  = 0.1 * dyn if dyn > 0 else 0.1
                idx, _ = find_peaks(ccg_smooth, prominence=pf)
                n_peaks_total = len(idx)
            except ImportError:
                n_peaks_total = 0
            if tmpl.n_peaks_min is not None and n_peaks_total < tmpl.n_peaks_min:
                score *= 0.5
            if tmpl.n_peaks_max is not None and n_peaks_total > tmpl.n_peaks_max:
                score *= 0.5

        # baseline_low_before_zero: full veto if the pre-zero region is not low
        if tmpl.baseline_low_before_zero:
            pre_zero_mask = bin_ms < 0
            if pre_zero_mask.any():
                pre_mean = float(ccg_smooth[pre_zero_mask].mean())
                if (pre_mean / peak_val) >= 0.5:
                    if _details is not None:
                        _details.extend(rule_details)
                    return 0.0   # veto — baseline before t=0 is not low

        if _details is not None:
            _details.extend(rule_details)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _conv_rule_score(
        rule: 'PeakRule',
        ccg_smooth: np.ndarray,
        bin_ms: np.ndarray,
    ) -> float:
        """
        Cosine similarity between the smoothed CCG and a Gaussian template built
        from this rule's lag/width parameters (the same Gaussian shown in the
        template editor preview).  Returns a value in [0, 1].
        """
        center_ms = (rule.lag_min + rule.lag_max) / 2.0
        if rule.width_max is not None:
            sigma_ms = rule.width_max / 2.355
        elif rule.width_min is not None:
            sigma_ms = rule.width_min / 2.355
        else:
            sigma_ms = max((rule.lag_max - rule.lag_min) / 4.0, 0.5)

        kernel = np.exp(-0.5 * ((bin_ms - center_ms) / sigma_ms) ** 2)
        if rule.polarity == -1:
            kernel = -kernel

        ccg_z  = ccg_smooth - ccg_smooth.mean()
        kern_z = kernel - kernel.mean()
        nc, nk = float(np.linalg.norm(ccg_z)), float(np.linalg.norm(kern_z))
        if nc == 0.0 or nk == 0.0:
            return 0.0
        return float(np.clip(np.dot(ccg_z / nc, kern_z / nk), 0.0, 1.0))
