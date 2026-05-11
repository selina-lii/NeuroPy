"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

from enum import Enum as _Enum, auto as _auto
from neuropy.io import NeuroscopeIO
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

_CUPY_AVAILABLE = cp is not None

import neuropy.analyses.correlations as correlations
from neuropy.analyses.utils import _san, _san_np, _hasvalue, Config, AnalysisDataset, Savable, SetOp, ConfigOption
from neuropy.core.neurons import Neurons
from neuropy.analyses.neurons_dataset import Key, EpochSlicingConfig, NeuronsDatasetConfig, NeuronsDataset
from scipy.signal import windows
from scipy.stats import poisson
from scipy import ndimage
from typing import Union, Optional, Dict, Any, Tuple
import h5py
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field, replace
from collections import defaultdict
import imageio
import neuropy.plotting.ccg as plot_ccg
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
SAVE_ROOT = "~/Documents/Neuropy/outputs"
DATA_ROOT = str(_REPO_ROOT / "data")

# CCG resolution presets (bin_size in seconds)
_CCG_RESOLUTION = {
    'lowres':  1e-3,    # 1 ms   — default, fast
    'highres': 1e-4,  # 0.1 ms — finer temporal resolution (must exceed 1/sample_rate)
}


class NormalizeBy(_Enum):
    REF_FRATE = _auto()
    TARGET_FRATE = _auto()
    TIME_SPAN = _auto()    # divide by effective recording time in hours
    TIME_SECOND = _auto()  # divide by effective recording time in seconds
    TOTAL_AREA = _auto()   # divide by sum of all bin counts (area under CCG)
    BASELINE = _auto()     # subtract conv baseline (ccg_null) from ccg


def apply_norms_to_ccg(ccg_raw, ccg_null_raw, ref, tgt, seg,
                        active_norms, neurons=None, nd=None, nd_key=None,
                        n_segments=0, is_custom_seg=False, custom_time_hours=None):
    """Return (ccg, ccg_null) with active normalizations applied (copies).

    For TIME_SPAN on custom segments pass ``custom_time_hours`` (active recording
    duration in hours for that window).  Without it the norm is silently skipped.
    """
    if not active_norms:
        return ccg_raw, ccg_null_raw
    ccg = ccg_raw.copy().astype(float)
    ccg_null = ccg_null_raw.copy().astype(float) if ccg_null_raw is not None else None
    if NormalizeBy.REF_FRATE in active_norms and neurons is not None:
        fr = float(neurons.firing_rate[ref])
        ccg /= max(fr, 1e-12)
        if ccg_null is not None:
            ccg_null /= max(fr, 1e-12)
    if NormalizeBy.TARGET_FRATE in active_norms and neurons is not None:
        fr = float(neurons.firing_rate[tgt])
        ccg /= max(fr, 1e-12)
        if ccg_null is not None:
            ccg_null /= max(fr, 1e-12)
    if NormalizeBy.TIME_SPAN in active_norms:
        et = None
        if is_custom_seg and custom_time_hours is not None:
            et = float(custom_time_hours)
        elif nd is not None and not is_custom_seg:
            if seg == n_segments and n_segments > 0:
                et = sum(float(nd.edge_times[nd_key].iloc[s]['effective_time_hours'])
                         for s in range(n_segments))
            else:
                et = float(nd.edge_times[nd_key].iloc[seg]['effective_time_hours'])
        if et is not None:
            ccg /= max(et, 1e-12)
            if ccg_null is not None:
                ccg_null /= max(et, 1e-12)
    if NormalizeBy.TIME_SECOND in active_norms:
        et_s = None
        if is_custom_seg and custom_time_hours is not None:
            et_s = float(custom_time_hours) * 3600.0
        elif nd is not None and not is_custom_seg:
            if seg == n_segments and n_segments > 0:
                et_s = sum(float(nd.edge_times[nd_key].iloc[s]['effective_time_hours'])
                           for s in range(n_segments)) * 3600.0
            else:
                et_s = float(nd.edge_times[nd_key].iloc[seg]['effective_time_hours']) * 3600.0
        if et_s is not None:
            ccg /= max(et_s, 1e-12)
            if ccg_null is not None:
                ccg_null /= max(et_s, 1e-12)
    if NormalizeBy.TOTAL_AREA in active_norms:
        total = float(np.sum(np.abs(ccg)))
        if total > 1e-12:
            ccg /= total
            if ccg_null is not None:
                ccg_null /= total
    if NormalizeBy.BASELINE in active_norms:
        if ccg_null is not None:
            ccg -= ccg_null  # subtract already-normalized baseline; result can be negative
            ccg_null = None  # baseline is now at 0; suppress the overlay
    return ccg, ccg_null


@dataclass
class CCGPanelData:
    """Fully-prepared data for rendering one CCG panel.

    Produced by :func:`compute_ccg_panel_data`.  All arrays are 1-D and
    already normalized.  Callers pass fields directly into
    ``plot_ccg.plot_ccg_panel`` without further processing.

    Attributes
    ----------
    ccg : ndarray
        Normalized CCG (BASELINE-subtracted if ``NormalizeBy.BASELINE`` was
        active).
    ccg_null : ndarray or None
        Null/expected trace for display.  ``None`` when baseline subtraction
        was applied (baseline now sits at zero).
    baseline_1d : ndarray or None
        The baseline array used by the active CS method:
        - conv   → copy of ccg_null (EranConv expected)
        - tailed → flat array at tail-mean value
        - global → flat array at max-outside-window value
        Passed as ``conn_strength_baseline`` to ``plot_ccg_panel`` for the
        green CS-overlay bars.
    cs_val : float or None
        Connection-strength scalar (AUC above baseline in test window).
    eff_min_lag : float or None
        Effective minimum test-window lag (seconds), after adaptive-TW
        resolution.
    eff_max_lag : float or None
        Effective maximum test-window lag (seconds).
    """
    ccg: np.ndarray
    ccg_null: Optional[np.ndarray]
    baseline_1d: Optional[np.ndarray]
    cs_val: Optional[float]
    eff_min_lag: Optional[float]
    eff_max_lag: Optional[float]


def deconv_autocorr(ccg, acg1, nspks1, acg2, nspks2):
    """
    Deconvolve ACGs from a single CCG trace (1-D, n_bins).
    Translated from Eran Stark's cchdeconv.m.
    """
    m = len(ccg)
    if m % 2 == 0:
        m -= 1
        ccg = ccg[:m]
        acg1 = acg1[:m]
        acg2 = acg2[:m]
    hw = (m - 1) // 2

    a1 = acg1.copy()
    a1 = (a1 - np.mean(a1)) / max(nspks1, 1)
    hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])
    a1[hw] = 1 - np.sum(a1[hidx])
    den = np.fft.fft(a1)

    a2 = acg2.copy()
    a2 = (a2 - np.mean(a2)) / max(nspks2, 1)
    a2[hw] = 1 - np.sum(a2[hidx])
    den = den * np.fft.fft(a2)

    den = np.where(np.abs(den) < 1e-10, 1e-10, den)
    dcccg = np.real(np.fft.ifft(np.fft.fft(ccg) / den))
    dcccg = np.concatenate([dcccg[1:], [dcccg[0]]])
    dcccg[dcccg < 0] = 0
    return dcccg


def compute_pair_conn_strength_1d(ccg, ccg_null, conf, method,
                                   acg_ref=None, acg_tgt=None,
                                   nspks_ref=1.0, nspks_tgt=1.0,
                                   min_lag_override=None, max_lag_override=None):
    """Compute connection strength scalar and baseline for a single 1-D CCG.

    Parameters
    ----------
    ccg : ndarray (n_bins,)  — already normalized
    ccg_null : ndarray or None
    conf : CCGConfig
    method : str — 'conv', 'tailed', or 'global'
    acg_ref, acg_tgt : ndarray or None — required for 'tailed'
    nspks_ref, nspks_tgt : float
    min_lag_override, max_lag_override : float or None
        Override conf.min_lag / conf.max_lag (e.g. for adaptive test window).

    Returns
    -------
    cs : float or None
    baseline_1d : ndarray or None
    """
    n_bins = len(ccg)
    bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
    center = n_bins // 2
    eff_min_lag = min_lag_override if min_lag_override is not None else conf.min_lag
    eff_max_lag = max_lag_override if max_lag_override is not None else conf.max_lag
    lo = max(0, center + int(eff_min_lag / bin_size_eff))
    hi_bin = min(n_bins, center + int(eff_max_lag / bin_size_eff) + 1)

    if method == 'conv':
        if ccg_null is not None:
            baseline = ccg_null.copy()
            cs = float(np.sum((ccg - ccg_null)[lo:hi_bin]))
        else:
            baseline = np.zeros_like(ccg)
            cs = float(np.sum(ccg[lo:hi_bin]))
        return cs, baseline

    if method == 'tailed':
        # Tailed baseline: mean of far-lag "tail" bins of the CURRENT CCG.
        # No ACG deconvolution is performed here; deconvolution (if desired) is a
        # separate upstream display transform.
        try:
            hw = max(1, int(11e-3 / bin_size_eff))
            l_idx = center - hw
            r_idx = center + hw + 1
            if l_idx > 0 and r_idx < n_bins:
                tail = np.concatenate([ccg[:l_idx], ccg[r_idx:]])
            else:
                edge = max(1, n_bins // 10)
                tail = np.concatenate([ccg[:edge], ccg[-edge:]])
            baseline_val = float(np.mean(tail))
            baseline = np.full(n_bins, baseline_val)
            cs = float(np.sum((ccg - baseline_val)[lo:hi_bin]))
            return cs, baseline
        except Exception:
            return None, None

    if method == 'global':
        # Global baseline: flat baseline at the maximum of the ACTIVE baseline.
        # - If convolution null exists, use its maximum.
        # - Otherwise fall back to the maximum outside the test window of the current CCG.
        try:
            if ccg_null is not None:
                baseline_val = float(np.max(ccg_null))
            else:
                outside_mask = np.ones(n_bins, dtype=bool)
                outside_mask[lo:hi_bin] = False
                if not np.any(outside_mask):
                    baseline_val = float(np.max(ccg))
                else:
                    baseline_val = float(np.max(ccg[outside_mask]))
            baseline = np.full(n_bins, baseline_val)
            cs = float(np.sum((ccg - baseline_val)[lo:hi_bin]))
            return cs, baseline
        except Exception:
            return None, None

    return None, None


def compute_ccg_panel_data(
    ccg_raw: np.ndarray,
    ccg_null_raw: Optional[np.ndarray],
    conf: 'CCGConfig',
    method: str,
    active_norms: set,
    ref: int,
    tgt: int,
    segment: int,
    n_segments: int,
    is_custom: bool,
    custom_time_hours: Optional[float],
    eff_min_lag: Optional[float],
    eff_max_lag: Optional[float],
    neurons=None,
    nd=None,
    nd_key=None,
    acg_ref: Optional[np.ndarray] = None,
    acg_tgt: Optional[np.ndarray] = None,
    nspks_ref: float = 1.0,
    nspks_tgt: float = 1.0,
) -> CCGPanelData:
    """Normalize a raw CCG pair and compute connection-strength in one pass.

    This is the single authoritative pipeline for preparing CCG data for
    display and CS computation.  It must be used instead of calling
    :func:`apply_norms_to_ccg` and :func:`compute_pair_conn_strength_1d`
    separately, because the ordering of normalization and baseline
    computation matters:

    1. Apply all normalisations **except** ``NormalizeBy.BASELINE``.
    2. Compute CS + ``baseline_1d`` on the resulting signal (so the global
       or tailed baseline is derived from the pre-subtraction CCG).
    3. Apply ``NormalizeBy.BASELINE`` using the *method's own* baseline
       (not always the EranConv null).

    Parameters
    ----------
    ccg_raw, ccg_null_raw : ndarray
        Raw spike-count CCG and its EranConv null, as returned by
        :meth:`_resolve_segment_data`.
    conf : CCGConfig
    method : str
        ``'conv'``, ``'tailed'``, or ``'global'``.
    active_norms : set of NormalizeBy
        Currently active normalizations (may include NormalizeBy.BASELINE).
    ref, tgt : int
        Neuron indices.
    segment, n_segments : int
        Current segment index and total count.
    is_custom : bool
        Whether *segment* is a custom (user-defined) time window.
    custom_time_hours : float or None
        Recording duration for a custom segment (for TIME_SPAN norm).
    eff_min_lag, eff_max_lag : float or None
        Effective test-window bounds in seconds, already resolved by the
        caller (e.g. via ``_effective_lags``).  Passed directly as
        ``min_lag_override``/``max_lag_override`` to
        :func:`compute_pair_conn_strength_1d`.
    neurons, nd, nd_key : optional
        Passed through to :func:`apply_norms_to_ccg` for firing-rate and
        time-span normalizations.
    acg_ref, acg_tgt : ndarray or None
        Auto-correlograms required by the ``'tailed'`` method.
    nspks_ref, nspks_tgt : float
        Spike counts used by ``'tailed'`` deconvolution.

    Returns
    -------
    CCGPanelData
    """
    # ── Step 1: normalize without BASELINE ───────────────────────────────
    has_baseline_norm = NormalizeBy.BASELINE in active_norms
    norms_no_bl = active_norms - {NormalizeBy.BASELINE}
    ccg, ccg_null = apply_norms_to_ccg(
        ccg_raw, ccg_null_raw, ref, tgt, segment, norms_no_bl,
        neurons, nd, nd_key, n_segments, is_custom, custom_time_hours)

    # ── Step 2: compute CS + baseline_1d on the step-1 signal ────────────
    cs_val, baseline_1d = compute_pair_conn_strength_1d(
        ccg, ccg_null, conf, method,
        acg_ref=acg_ref, acg_tgt=acg_tgt,
        nspks_ref=nspks_ref, nspks_tgt=nspks_tgt,
        min_lag_override=eff_min_lag,
        max_lag_override=eff_max_lag)

    # ── Step 3: apply BASELINE norm using the method's own baseline ───────
    if has_baseline_norm:
        bl = baseline_1d if baseline_1d is not None else ccg_null
        if bl is not None:
            ccg = ccg - bl
            ccg_null = None     # baseline is now at zero; suppress the null overlay

    # ── Step 4: choose which null trace to show (method-dependent) ────────
    # conv  → EranConv predicted null (varies per bin)
    # tailed → flat array at tail-mean value
    # global → flat array at max-outside-window value (same shape, drawn as h-line)
    # All three are suppressed when BASELINE norm is active because the CCG is
    # already baseline-subtracted and the baseline sits at zero.
    if has_baseline_norm:
        display_null = None
    elif method == 'conv':
        display_null = ccg_null
    else:
        # tailed and global both produce a flat baseline_1d; display identically
        display_null = baseline_1d

    return CCGPanelData(
        ccg=ccg,
        ccg_null=display_null,
        baseline_1d=baseline_1d,
        cs_val=cs_val,
        eff_min_lag=eff_min_lag,
        eff_max_lag=eff_max_lag,
    )


class ConnStrengthMethod(ConfigOption):
    PEAKSIZE = 0
    TAILED = 1


def example(var: dict):
    """
    Get an example from a dictionary
    """
    k, v = next(iter(var.items()))
    return v


class CCGConfig(Config):
    """
    Configuration for CCG computation and significance detection.

    Fields are split into two groups:
      COMPUTE_FIELDS  — affect the raw CCG arrays; changing these requires
                        re-running spike_correlations (expensive).
      SIGNIF_FIELDS   — affect significance detection only; changing these
                        only requires re-running EranConv (cheap).
    """

    # Fields that affect the CCG computation (spike_correlations + EranConv conv)
    COMPUTE_FIELDS = [
        'name', 'resolution', 'bin_size', 'duration', 'conv_window',
        'conn_types_E', 'conn_types_I',
        'use_acceleration', 'symmetrize_ccg',
    ]
    # Fields that affect significance detection only (rerunnable without recomputing CCG)
    SIGNIF_FIELDS = [
        'alpha', 'alpha2', 'min_lag', 'max_lag',
        'min_spkcount', 'spkcnt_scope', 'multiple_correction',
    ]

    def __init__(
        self,
        name="default",
        conn_types_E: Union[list[list], list] = [('pyr', 'pyr'),
                                                 ('pyr', 'inter')],
        conn_types_I: Union[list[list], list] = [('inter', 'inter'),
                                                 ('inter', 'pyr')],
        duration: float = 20e-3,
        bin_size: float = None,
        resolution: str = 'lowres',
        conv_window: float = 5e-3,
        alpha: float = 0.05,
        alpha2: float = 0.1,
        min_lag: float = 1e-3,
        max_lag: float = 3e-3,
        min_spkcount=2.5,
        spkcount_scope=12e-3,
        multiple_correction: str = 'bonferroni',  # 'bonferroni' or 'fdr_bh'
        use_acceleration=None,  # None → auto-detect CuPy; True/False to override
        symmetrize_ccg=True,
        conn_strength_method: ConnStrengthMethod = ConnStrengthMethod.PEAKSIZE,
    ):
        super().__init__()
        self.name = name
        self.resolution = resolution

        # bin_size: explicit value takes priority; otherwise use resolution preset
        if bin_size is None:
            bin_size = _CCG_RESOLUTION.get(resolution, 1e-3)
        self.bin_size = bin_size

        self.conn_types_E = conn_types_E
        self.conn_types_I = conn_types_I
        self.duration = duration
        self.conv_window = conv_window
        self.alpha = alpha
        self.alpha2 = alpha2
        self.multiple_correction = multiple_correction
        self.center_bin = int(self.duration / self.bin_size // 2)
        self.nbins = int(self.duration / self.bin_size) + 1  # NOTE

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_spkcount = min_spkcount
        self.spkcnt_scope = spkcount_scope
        self.spkcnt_bins = int(self.spkcnt_scope / self.bin_size)

        self.min_lag_bin = self.center_bin + int(
            self.min_lag / self.bin_size)  # leftmost bin for p value test
        self.max_lag_bin = self.center_bin + int(
            self.max_lag / self.bin_size) + 1  # rightmost bin for p value test
        self.min_spkcnt_bin = self.center_bin - self.spkcnt_bins // 2  # leftmost bin requiring minimum spike count
        self.max_spkcnt_bin = self.center_bin + self.spkcnt_bins // 2 + 1  # rightmost bin requiring minimum spike count

        self.use_acceleration = _CUPY_AVAILABLE if use_acceleration is None else use_acceleration
        self.symmetrize_ccg = symmetrize_ccg

        self.conn_strength_method = conn_strength_method

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.filepath}\n"
        return s

    @property
    def conn_types(self):
        return {'E': self.conn_types_E, 'I': self.conn_types_I}

    @property
    def conn_types_flat(self):
        return self.conn_types_E + self.conn_types_I

    @property
    def conn_types_labeled(self) -> list:
        """Return [(EI, (ref, tgt)), ...] for all connection types, E first then I."""
        return ([('E', ct) for ct in self.conn_types_E] +
                [('I', ct) for ct in self.conn_types_I])

    def excitability_for(self, conn_type) -> str:
        """Return 'E' or 'I' for a given (ref, tgt) connection type tuple."""
        ct = tuple(conn_type)
        if ct in [tuple(c) for c in self.conn_types_E]:
            return 'E'
        if ct in [tuple(c) for c in self.conn_types_I]:
            return 'I'
        raise ValueError(f"Unknown conn_type: {conn_type}")

    @property
    def conv_window_bins(self):
        return self.conv_window / self.bin_size

    def time2bin(self, x):
        """time in SECONDS to bin#"""
        return x / self.bin_size

    def bin2time(self, x):
        """bin# to time in SECONDS"""
        return x * self.bin_size

    @property
    def filepath(self):
        return f"{SAVE_ROOT}/{self.name}.ccg.meta.h5"

    def cache_key(self, session_key_str: str) -> str:
        """
        Deterministic short hash of all computation-affecting config fields + session.
        Used as part of cache filenames.

        Returns e.g. 'sess_RatJ_Day2_a3f91bc7e2d04a1c'
        """
        import hashlib, json

        def _enum_name(v):
            if isinstance(v, list):
                return [_enum_name(x) for x in v]
            return v.name if hasattr(v, 'name') else v

        def _to_list(v):
            if isinstance(v, (list, tuple)):
                return [_to_list(x) for x in v]
            return v

        fields = {
            "duration":             self.duration,
            "bin_size":             self.bin_size,
            "conv_window":          self.conv_window,
            "alpha":                self.alpha,
            "alpha2":               self.alpha2,
            "min_lag":              self.min_lag,
            "max_lag":              self.max_lag,
            "min_spkcount":         self.min_spkcount,
            "spkcount_scope":       self.spkcnt_scope,
            "multiple_correction":            self.multiple_correction,
            "symmetrize_ccg":       self.symmetrize_ccg,
            "conn_types_E":         _to_list(self.conn_types_E),
            "conn_types_I":         _to_list(self.conn_types_I),
            "session":              session_key_str,
        }
        digest = hashlib.sha256(
            json.dumps(fields, sort_keys=True).encode()
        ).hexdigest()[:12]
        return digest

    @property
    def save_path(self) -> str:
        """Base path (no extension) for saving this config's CCGDataset."""
        root = os.path.join(DATA_ROOT, "ccg", self.name)
        return os.path.join(root, f"{self.name}_{self.resolution}")


class CCGPointer(Savable):
    """
    A positional pointer to CCGdata locations
    """
    def __init__(
        self,
        key,
        inds,
        conf: CCGConfig = None,
        significant=None,
        edge_times=None,
    ):
        super().__init__()
        self.key = key
        self._inds = _san_np(inds)
        self.edge_times = edge_times
        self.conf = conf
        self.significant = significant

    # def __repr__(self):
    #     printstr = "CCG name: " + str(self.key) + "\n===\n"
    #     printstr += "Legend\n" + self.segment_labels + "\n===\n"
    #     conn = self.connectivity
    #     if conn is None: return 'No connectivity'
    #     printstr += f"{'Pair indices':<15}\n"
    #     for (x, y) in sorted(conn.keys()):
    #         printstr += f"{f'({x}, {y})':<15}\tIn segments {str(conn[(x,y)]):<20}\t"
    #         printstr += "\n"
    #     return printstr

    @property
    def connectivity_array(self):
        conn = self.connectivity
        inds = self.inds2
        a = pd.DataFrame(0,
                         index=pd.MultiIndex.from_arrays(inds.T),
                         columns=range(self.n_segments))
        for ind in inds:
            a.loc[tuple(ind), conn[tuple(ind)]] = 1
        return a

    @property
    def segment_labels(self):
        printstr = ""
        for i, l in enumerate(self.edge_times.label.values):
            printstr += f"{str(i):2}:{l:10}"
            if (i + 1) % 5 == 0:
                printstr += "\n"
        return printstr

    @property
    def stored_by_segment(self):
        if not _hasvalue(self._inds):
            return False
        return self._inds.shape[-1] == 3  #otherwise 2

    @property
    def connectivity(self):
        if not _hasvalue(self.inds):
            return None
        d = defaultdict(list)
        if self.stored_by_segment:
            d = defaultdict(list)
            for i, x, y in self.inds:
                d[(x, y)].append(i)
        else:
            for x, y in self.inds:
                d[(x, y)].append(list(np.arange(self.n_segments)))
        return d

    def __str__(self):
        s = 'CCG Pointer\n'
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray) or isinstance(val, list):
                s += f"{key}\tshape={np.array(val).shape}"
                sval = "\n".join(str(val[0:2]).splitlines()[:3])
                s += f"\tval={sval}...\n"
            elif isinstance(val, dict):
                k, v = next(iter(val.items()))
                s += f"{key} dict keys={k}...\n"
                item_str = str(v)
                for line in item_str.splitlines()[:3]:
                    s += f"\t\t{line}\n"
            elif key != '_conf':
                sval = "\n".join(str(val).splitlines()[:3])
                s += f"{key}: {sval}\n"
        return s

    @property
    def inds2(self):
        if self.stored_by_segment:
            return SetOp.unique(self._inds[:, -2:])
        else:
            return self.inds

    @property
    def inds3(self):
        if self.stored_by_segment:
            return self._inds
        else:
            x = np.arange(self.n)
            yz = self.inds[:, 1:]
            return np.column_stack(
                [np.repeat(x, yz.shape[0]),
                 np.tile(yz, (x.shape[0], 1))])

    @property
    def indsplit(self) -> list[np.ndarray]:
        """Indices listed by each segment"""
        return [
            self.inds[np.where(self.inds[:, 0] == i)[0]][:, 1:]
            for i in range(self.n_segments)
        ]

    @property
    def n_pairs(self):
        if not _hasvalue(self._inds):
            return 0
        if self.stored_by_segment:
            return self.inds2.shape[0]
        else:
            return self.inds.shape[0]

    def n_pairs_segment(self, i: int):
        if self.stored_by_segment:
            return np.sum(np.where(self.inds[:, 0] == i)[0])
        else:
            assert i < self.n_segments
            return self.n_pairs

    @property
    def inds(self):
        if self._inds is None:
            return np.empty((0, 2), dtype=int)
        return self._inds
    
    @property
    def ref_inds(self):
        return self.inds[:, -2]

    @property
    def target_inds(self):
        return self.inds[:, -1]

    @property
    def ref_ind(self):
        return self.inds[-2]

    @property
    def unique_inds(self):
        """Returns list of unique neuron inds, not pairs!"""
        return np.unique(self.inds)

    @property
    def n_segments(self):
        return self.edge_times.shape[0]

    def get_segment(self, i: int) -> 'CCGPointer':
        if self._inds is None:
            inds = None
        elif self.stored_by_segment is False:
            assert i < self.n_segments
            inds = np.hstack([np.zeros(self.n_pairs), self.inds])
        else:
            inds = self.inds[np.where(self.inds[:, 0] == i)[0]][:, 1:]
        return CCGPointer(
            key=self.key.add(segment=i),
            inds=inds,
            edge_times=self.edge_times.iloc[i],
        )

    def split(self) -> list['CCGPointer']:
        return [self.get_segment(i) for i in range(self.edge_times.shape[0])]

    def print_connectivity(self):
        grouped = defaultdict(list)
        for seg_i, x, y in self.inds:
            grouped[(x, y)].append(seg_i)
        for (x, y), seg_inds in grouped.items():
            print(f"[{x},{y}] appearing in [{', '.join(map(str, seg_inds))}]")

    def plotdir(self, root):
        root = os.path.expanduser(root)
        tag = ''
        for v in [self.key.epoch, self.key.segment]:
            if v is not None:
                tag += v
        if tag != '':
            tag = '-' + tag
        if self.key.conn_type is None:
            return f"{root}/{self.key.session}/{self.key.session}{tag}/{self.key.excitability}_any"
        return f"{root}/{self.key.session}/{self.key.session}{tag}/{self.key.excitability}_{self.key.conn_type[0]}-{self.key.conn_type[1]}"


@dataclass
class CCGData(Savable):
    """
    Stores the whole CCG array and its p values

        ccg         [N, Np, Nbins]
        N = number of data segments
        Np = number of neuron pairs
        Nbins = number of bins per CCG 
        ccg_null    [N, Np, Nbins]
    """
    key: Key
    _conf: CCGConfig

    # [n_seg, n_ref, n_tgt, n_bins]
    ccg: np.ndarray
    ccg_null: np.ndarray
    pval: np.ndarray
    qval: np.ndarray
    pval_corrected: np.ndarray
    qval_corrected: np.ndarray
    significant: np.ndarray
    norm_factors: list[np.ndarray]

    # [n_seg, n_ref, n_tgt]
    conn_strength: np.ndarray

    def __post_init__(self):
        super().__init__()  # initializes parent attributes

    @property
    def conf(self):
        return self._conf

    @property
    def n_segment(self):
        return self.ccg.shape[0]

    @staticmethod
    def get_autocorr_locations(shape):
        """
        Genearte a mask of autocorrelation locations shaped (ngroups, nneurons, nneurons)
        """
        n_auto = min(shape[-3], shape[-2])
        auto_mask = np.eye(n_auto, dtype=bool)
        auto_mask = np.pad(auto_mask,
                           ((0, shape[-3] - n_auto), (0, shape[-2] - n_auto)))
        autocorr_locations = np.broadcast_to(auto_mask, shape[:-1])
        return autocorr_locations

    def get_conn_strength(self, method: ConnStrengthMethod, **kwargs):
        """
        Wrapper
        """
        if self.ccg is None:
            return  # no connection
        if method == ConnStrengthMethod.PEAKSIZE:
            self.__get_conn_strength_peaksize(**kwargs)
        elif method == ConnStrengthMethod.TAILED:
            self.__get_conn_strength_tailed(**kwargs)
        else:
            raise NotImplementedError("Unknown connection strength method")

    @property
    def conn_strength_change(self):
        return self.conn_strength[-1, ...] - self.conn_strength[0, ...]

    def save_plots(self,
                   pt: CCGPointer,
                   neurons: Neurons,
                   neurons_config: NeuronsDatasetConfig,
                   frates_cut: np.ndarray,
                   plotdir: str,
                   split_all_plots=False,
                   overwrite=False):

        if not os.path.exists(plotdir):
            os.makedirs(plotdir, exist_ok=True)

        if self.n_segment > 1 or split_all_plots:
            self.__save_gif(
                plotdir=plotdir,
                pt=pt,
                neurons=neurons,
                neurons_config=neurons_config,
                frates_cut=frates_cut,
                overwrite=overwrite,
            )
        else:
            self.__save_img(
                plotdir=plotdir,
                pt=pt,
                neurons=neurons,
                neurons_config=neurons_config,
                frates_cut=frates_cut,
                overwrite=overwrite,
            )

    def __save_img(
        self,
        plotdir,
        pt: CCGPointer,
        neurons: Neurons,
        neurons_config: NeuronsDatasetConfig,
        frates_cut: np.ndarray,
        overwrite=False,
    ):
        """Save a single PNG for each significant pair (single-segment case)."""
        if pt.inds is None:
            print(f"nothing to plot: {pt}")
            return

        for inds in pt.inds2:
            save_path = f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png"
            if not overwrite and os.path.exists(save_path):
                print(f"{save_path} already exists")
                continue

            loc = (0, *inds)
            fig = plot_ccg.plot_ccg_figure(
                inds=inds,
                ids=neurons.ind2id(inds),
                neuron_types=neurons.neuron_type[inds],
                frates_cut=frates_cut[0][inds],
                frates_all=neurons.firing_rate[inds],
                waveforms=neurons.waveforms[inds],
                shank_ids=neurons.shank_ids[inds],
                discarded_channels=neurons_config.recinfo.skipped_channels,
                ch_per_shank=neurons_config.ch_per_shank,
                peak_channels=(neurons.peak_channels[inds]
                               if getattr(neurons, 'peak_channels', None) is not None
                               else None),
                save_path=save_path,
                window_size=self.conf.duration * 1e3,
                bin_size=self.conf.bin_size * 1e3,
                ccg=self.ccg[loc],
                ccg_null=self.ccg_null[loc]
                if self.ccg_null is not None else None,
                pval=self.pval[loc] if self.pval is not None else None,
                pval_corrected=self.pval_corrected[loc]
                if self.pval_corrected is not None else None,
                alpha=self.conf.alpha if self.conf.alpha is not None else None,
                is_significant_pair=self.significant[loc]
                if self.significant is not None else None,
                show=False,
                save=True,
                segment_id=0,
            )
            plt.close(fig)

        print("done saving plots")

    def __get_conn_strength_peaksize(self, norm_factor: np.ndarray = None):
        """
        Connection strength:
        
            Area under CCG curve minus baseline, within temporal ROI
            The ROI is by default the same as the interval tested for peak/trough signficance
            Can be negative
        """
        auc = self.ccg - self.ccg_null  # area under curve
        cs = np.sum(auc[..., self.conf.min_lag_bin:self.conf.max_lag_bin],
                    axis=-1)  # (inds,)
        if norm_factor is not None:
            cs = cs / norm_factor  # e.g. presynaptic element firing rate
        self.conn_strength = cs

    def __get_conn_strength_tailed(self,
                                   nspks: np.ndarray,
                                   norm_factor: np.ndarray = None):
        """
        Connection strength via ACG deconvolution + tail baseline.

        Deconvolves each neuron's autocorrelogram out of its cross-CCGs (FFT
        method from Eran Stark's cchdeconv.m), then subtracts the mean of the
        "tail" bins (|t| > 11 ms from center) as the baseline.  Area under the
        resulting curve in [min_lag, max_lag] gives the connection strength.

        Parameters
        ----------
        nspks : ndarray, shape (n_seg, n_neurons)
            Spike counts per segment per neuron.
        norm_factor : ndarray or None
            Optional element-wise divisor applied to conn_strength.
        """

        n_seg, n_ref, n_tgt, n_bins = self.ccg.shape
        n_neurons = min(n_ref, n_tgt)
        center = self.conf.center_bin

        # ACGs: shape (n_seg, n_neurons, n_bins) — diagonal of the full matrix
        acgs = np.stack([self.ccg[:, i, i, :] for i in range(n_neurons)], axis=1)

        # Deconvolve ACGs from every cross-pair (ref ≠ tgt)
        dcccg = self.ccg.copy().astype(float)
        for s in range(n_seg):
            for ref in range(n_ref):
                for tgt in range(n_tgt):
                    if ref == tgt:
                        continue
                    dcccg[s, ref, tgt] = deconv_autocorr(
                        self.ccg[s, ref, tgt].copy(),
                        acgs[s, ref], float(nspks[s, ref]),
                        acgs[s, tgt], float(nspks[s, tgt]),
                    )

        # Tail baseline: mean of bins with |t| > 11 ms from center
        hw = int(11e-3 / self.conf.bin_size)
        l = center - hw       # first bin of the central region
        r = center + hw + 1   # one past the last bin of the central region

        if l > 0 and r < n_bins:
            tail = np.concatenate([dcccg[..., :l], dcccg[..., r:]], axis=-1)
        else:
            # Window too narrow for a ±11 ms tail (e.g. only 20 ms window);
            # fall back to the outermost 10 % of bins on each side.
            edge = max(1, n_bins // 10)
            tail = np.concatenate([dcccg[..., :edge], dcccg[..., -edge:]], axis=-1)

        baseline = np.mean(tail, axis=-1, keepdims=True)
        self.ccg_null = np.broadcast_to(baseline, self.ccg.shape).copy()

        # AUC within the significance window
        auc = dcccg - baseline
        cs = np.sum(auc[..., self.conf.min_lag_bin:self.conf.max_lag_bin], axis=-1)
        if norm_factor is not None:
            cs = cs / norm_factor
        self.conn_strength = cs

    def __save_gif(
        self,
        plotdir,
        pt: CCGPointer,
        neurons: Neurons,
        neurons_config: NeuronsDatasetConfig,
        frates_cut: np.ndarray,
        overwrite=False,
    ):
        if pt.inds is None:
            print(f"nothing to plot: {pt}")
            return

        for i, inds in enumerate(pt.inds2):
            sig = self.significant[:,
                                   *inds] if self.significant is not None else None
            where_sig = np.where(sig)[0]
            save_path = f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}-{'-'.join(['sig'+str(s) for s in where_sig])}.gif"
            if not overwrite and os.path.exists(save_path):
                print(f"{save_path} already exists")
                continue

            figs = []
            ymin, ymax, ymin2, ymax2 = [], [], [], []
            print(i, inds)
            for i_seg in range(self.n_segment):
                loc = (i_seg, *inds)
                fig = plot_ccg.plot_ccg_figure(
                    inds=inds,
                    ids=neurons.ind2id(inds),
                    neuron_types=neurons.neuron_type[inds],
                    frates_cut=frates_cut[i_seg][inds],
                    frates_all=neurons.firing_rate[inds],
                    waveforms=neurons.waveforms[inds],
                    shank_ids=neurons.shank_ids[inds],
                    discarded_channels=neurons_config.recinfo.skipped_channels,
                    ch_per_shank=neurons_config.ch_per_shank,
                    peak_channels=(neurons.peak_channels[inds]
                                   if getattr(neurons, 'peak_channels', None) is not None
                                   else None),
                    save_path=None,
                    window_size=self.conf.duration * 1e3,
                    bin_size=self.conf.bin_size * 1e3,
                    ccg=self.ccg[loc],
                    ccg_null=self.ccg_null[loc]
                    if self.ccg_null is not None else None,
                    pval=self.pval[loc] if self.pval is not None else None,
                    pval_corrected=self.pval_corrected[loc]
                    if self.pval_corrected is not None else None,
                    alpha=self.conf.alpha
                    if self.conf.alpha is not None else None,
                    is_significant_pair=self.significant[loc]
                    if self.significant is not None else None,
                    show=False,
                    save=False,
                    segment_id=i_seg,
                )
                figs.append(fig)
                _ymin, _ymax = fig.axes[0].get_ylim()
                ymin.append(_ymin)
                ymax.append(_ymax)
            ymin, ymax = min(ymin), max(ymax)
            frames = []
            for fig in figs:
                fig.axes[0].set_ylim(ymin, ymax)
                fig.canvas.draw_idle()
                frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
                plt.close(fig)

            imageio.mimsave(save_path, frames, duration=0.8)

        print("done saving plots")


class CCGDataset(AnalysisDataset):
    """
    Data and operations on CCGs from an experiment.

    Requires a NeuronsDataset to be processed first, and a configuration object
    (see :class:`CCGConfig`).

    Tests CCGs and stores them separately by significance criteria.

    Attributes
    ----------
    data : dict[CCGPointer, ...]
        Neuron pairs that are significant connections.
    spurious : dict[CCGPointer, ...]
        Neuron pairs that passed rough significance checks but do not belong to a
        certain connection type.
    conf : CCGConfig
        Configuration.
    nd : NeuronsDataset
        Source neurons for the CCGs.
    """

    _ccg: dict[CCGData]
    data: dict[CCGPointer]
    spurious: dict[CCGPointer]
    _conf: CCGConfig
    nd: NeuronsDataset

    def __init__(
        self,
        conf=None,
        nd=None,
    ):
        super().__init__(conf)
        self.nd = nd
        self._ccg = {}
        self.data = {}
        self.spurious = {}
        # Jitter significance testing lives in neuropy.analyses.jitter (Jitter/JitterDataset).
        self._ccg_highres = {}  # nd_key → CCGData (raw only; loaded by load_highres)
        self._jitter_results = {}  # nd_key → {(ref, tgt, res_key): (j_avg, j_pval, j_pval_bins)}
        self.get_ccg()

    @property
    def filepath(self):
        return ''

    def save_path(self) -> str:
        return self.conf.save_path

    def highres_save_path(self) -> str:
        """Base path (no extension) for saving the high-resolution CCGData dict."""
        return self.conf.save_path + '_highres'

    def find(self, query: str):
        """Return the key of the first CCGPointer whose session matches *query*.

        Matching is case-insensitive and ignores underscores/spaces, so e.g.
        ``cd.find("RatUDay2")`` matches ``RatU_Day2NSD``.

        Prefers the excitatory pyr→pyr type when present; otherwise returns
        the first matching key.  Pass the returned key directly to
        ``launch_ccg_review``::

            launch_ccg_review(cd, cd.find("RatUDay2"))

        Raises KeyError if no session matches.
        """
        q = query.replace('_', '').replace(' ', '').lower()
        # Collect all full keys from cd.data whose session matches
        matches = [
            k for k in self.data.keys()
            if k.session and q in k.session.replace('_', '').replace(' ', '').lower()
        ]
        if not matches:
            raise KeyError(f"No CCG session matching {query!r}")
        # Prefer E excitability + pyr-pyr conn_type; fall back to first match
        preferred = [
            k for k in matches
            if getattr(k, 'excitability', None) == 'E'
            and getattr(k, 'conn_type', None) == ('pyr', 'pyr')
        ]
        return (preferred or matches)[0]

    # ------------------------------------------------------------------
    # Metadata helpers (I.3: cache invalidation)
    # ------------------------------------------------------------------

    def _metadata_path(self, suffix='compute') -> str:
        """Return path to a .meta.json file.

        suffix='compute' → tracks CCG computation parameters.
        suffix='signif'  → tracks significance-detection parameters.
        """
        return os.path.expanduser(self.save_path()) + f'.{suffix}.meta.json'

    @staticmethod
    def _serialize_conf_value(v):
        """Recursively serialize a config value to a JSON-safe type."""
        if hasattr(v, 'name'):          # ConfigOption / Enum
            return v.name
        if isinstance(v, (list, tuple)):
            return [CCGDataset._serialize_conf_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): CCGDataset._serialize_conf_value(val)
                    for k, val in v.items()}
        try:
            import json as _json
            _json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)

    def _save_metadata(self):
        """Write two .meta.json files — one for compute params, one for signif params."""
        import json, datetime as _dt

        _s = self._serialize_conf_value

        def _write(suffix, fields):
            conf_dict = {f: _s(getattr(self.conf, f, None)) for f in fields}
            meta = {
                'version': '1.1',
                'saved_at': _dt.datetime.now().isoformat(),
                'conf': conf_dict,
            }
            p = os.path.expanduser(self._metadata_path(suffix))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as fh:
                json.dump(meta, fh, indent=2)
            print(f"[CCGDataset] {suffix} metadata saved → {p}")

        _write('compute', CCGConfig.COMPUTE_FIELDS)
        _write('signif',  CCGConfig.SIGNIF_FIELDS)

    def _check_metadata(self, suffix='compute') -> bool:
        """Return True if the saved metadata for *suffix* matches the current config."""
        import json
        p = os.path.expanduser(self._metadata_path(suffix))
        if not os.path.isfile(p):
            # Fall back: try the old single-file format
            old_p = os.path.expanduser(self.save_path()) + '.meta.json'
            if not os.path.isfile(old_p):
                return False
            p = old_p
        try:
            with open(p) as fh:
                meta = json.load(fh)
        except Exception as exc:
            print(f"[CCGDataset] metadata read error ({suffix}): {exc}")
            return False
        saved_conf = meta.get('conf', {})
        fields = (CCGConfig.COMPUTE_FIELDS if suffix == 'compute'
                  else CCGConfig.SIGNIF_FIELDS)
        _s = self._serialize_conf_value
        for field in fields:
            current_val = _s(getattr(self.conf, field, None))
            saved_val = saved_conf.get(field)
            if saved_val != current_val:
                print(f"[CCGDataset] cache miss ({suffix}): '{field}' "
                      f"saved={saved_val!r} vs current={current_val!r}")
                return False
        return True

    # ------------------------------------------------------------------
    # Separate save / load for CCGData (raw arrays) vs CCGPointers (pairs)
    # ------------------------------------------------------------------

    def _ccgdata_path(self) -> str:
        return os.path.expanduser(self.save_path()) + '_ccgdata'

    def _ccgpointers_path(self) -> str:
        return os.path.expanduser(self.save_path()) + '_ccgpointers'

    def save_ccgdata(self):
        """Save only the raw CCG arrays (``_ccg`` dict) to a separate file.

        This covers the expensive spike_correlations + convolution output.
        It is invalidated only when COMPUTE_FIELDS change.
        """
        import hickle as hkl
        p = self._ccgdata_path() + '.hkl'
        os.makedirs(os.path.dirname(p), exist_ok=True)
        hkl.dump(self._ccg, p)
        # Write compute metadata next to it
        import json, datetime as _dt
        _s = self._serialize_conf_value
        meta = {
            'version': '1.1',
            'saved_at': _dt.datetime.now().isoformat(),
            'conf': {f: _s(getattr(self.conf, f, None))
                     for f in CCGConfig.COMPUTE_FIELDS},
        }
        mp = self._ccgdata_path() + '.meta.json'
        with open(mp, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f"[CCGDataset] ccgdata saved → {p}")

    def load_ccgdata(self) -> str:
        """Load raw CCG arrays from the separate ccgdata file.

        Returns
        -------
        'loaded'  — successfully loaded (config matches).
        'missing' — file does not exist on disk.
        'stale'   — file exists but compute config has changed.
        """
        import hickle as hkl, json
        mp = self._ccgdata_path() + '.meta.json'
        p  = self._ccgdata_path() + '.hkl'
        if not os.path.isfile(p):
            return 'missing'
        # Validate compute config if metadata exists
        if os.path.isfile(mp):
            try:
                with open(mp) as fh:
                    meta = json.load(fh)
                saved = meta.get('conf', {})
                _s = self._serialize_conf_value
                for field in CCGConfig.COMPUTE_FIELDS:
                    current = _s(getattr(self.conf, field, None))
                    if saved.get(field) != current:
                        print(f"[CCGDataset] ccgdata cache miss: "
                              f"'{field}' saved={saved.get(field)!r} "
                              f"current={current!r}")
                        return 'stale'
            except Exception as exc:
                print(f"[CCGDataset] ccgdata metadata error: {exc}")
                return 'stale'
        try:
            self._ccg = hkl.load(p)
            print(f"[CCGDataset] ccgdata loaded ← {p}")
            return 'loaded'
        except Exception as exc:
            print(f"[CCGDataset] ccgdata load failed: {exc}")
            return 'stale'

    def save_ccgpointers(self):
        """Save only the CCGPointer dicts (``data`` + ``spurious``) to a separate file.

        This covers the significance-detection output and can be re-saved
        whenever SIGNIF_FIELDS change without re-running spike_correlations.
        """
        import hickle as hkl, json, datetime as _dt
        p = self._ccgpointers_path() + '.hkl'
        os.makedirs(os.path.dirname(p), exist_ok=True)
        hkl.dump({'data': self.data, 'spurious': self.spurious}, p)
        _s = self._serialize_conf_value
        meta = {
            'version': '1.1',
            'saved_at': _dt.datetime.now().isoformat(),
            'conf': {f: _s(getattr(self.conf, f, None))
                     for f in CCGConfig.SIGNIF_FIELDS},
        }
        mp = self._ccgpointers_path() + '.meta.json'
        with open(mp, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f"[CCGDataset] ccgpointers saved → {p}")

    def load_ccgpointers(self) -> str:
        """Load CCGPointer dicts from the separate ccgpointers file.

        Returns
        -------
        'loaded'  — successfully loaded (config matches).
        'missing' — file does not exist on disk.
        'stale'   — file exists but significance config has changed.
        """
        import hickle as hkl, json
        mp = self._ccgpointers_path() + '.meta.json'
        p  = self._ccgpointers_path() + '.hkl'
        if not os.path.isfile(p):
            return 'missing'
        if os.path.isfile(mp):
            try:
                with open(mp) as fh:
                    meta = json.load(fh)
                saved = meta.get('conf', {})
                _s = self._serialize_conf_value
                for field in CCGConfig.SIGNIF_FIELDS:
                    current = _s(getattr(self.conf, field, None))
                    if saved.get(field) != current:
                        print(f"[CCGDataset] ccgpointers cache miss: "
                              f"'{field}' saved={saved.get(field)!r} "
                              f"current={current!r}")
                        return 'stale'
            except Exception as exc:
                print(f"[CCGDataset] ccgpointers metadata error: {exc}")
                return 'stale'
        try:
            obj = hkl.load(p)
            self.data     = obj.get('data', {})
            self.spurious = obj.get('spurious', {})
            # Detect hickle RecoveredDataset fallbacks (class not found)
            for d in (self.data, self.spurious):
                for v in d.values():
                    if v is not None and not isinstance(v, CCGPointer):
                        print(f"[CCGDataset] ccgpointers stale: "
                              f"found {type(v).__name__} instead of CCGPointer")
                        self.data, self.spurious = {}, {}
                        return 'stale'
            print(f"[CCGDataset] ccgpointers loaded ← {p}")
            return 'loaded'
        except Exception as exc:
            print(f"[CCGDataset] ccgpointers load failed: {exc}")
            return 'stale'

    # ------------------------------------------------------------------
    # High-res save / load (I.4)
    # ------------------------------------------------------------------

    def save_highres(self, path: str = None):
        """Save self._ccg_highres dict to a separate hickle file.

        Parameters
        ----------
        path : str, optional
            Base path (without extension).  Defaults to ``highres_save_path()``.
        """
        if not self._ccg_highres:
            print("[save_highres] Nothing to save (run load_highres() first).")
            return
        import hickle as hkl
        p = os.path.expanduser((path or self.highres_save_path()) + '.hkl')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        hkl.dump(self._ccg_highres, p)
        print(f"[CCGDataset] highres saved → {p}")

    def _load_highres_from_disk(self, path: str = None) -> bool:
        """Try to load high-res CCGData from a previously saved file.

        Returns True on success, False if the file does not exist.
        """
        import hickle as hkl
        p = os.path.expanduser((path or self.highres_save_path()) + '.hkl')
        if not os.path.isfile(p):
            return False
        try:
            data = hkl.load(p)
            self._ccg_highres = data
            print(f"[CCGDataset] highres loaded ← {p}")
            return True
        except Exception as exc:
            print(f"[CCGDataset] highres load failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Jitter save / load
    # ------------------------------------------------------------------

    def jitter_save_path(self) -> str:
        """Base path (no extension) for saving jitter results."""
        return self.conf.save_path + '_jitter'

    def save_jitter(self, path: str = None):
        """Save _jitter_results to a hickle file.

        Structure on disk:
            nd_str → pair_key → {ref, tgt, res_key, j_avg, j_pval, j_pval_bins}
        """
        if not self._jitter_results:
            print("[save_jitter] Nothing to save.")
            return
        import hickle as hkl, numpy as np
        p = os.path.expanduser((path or self.jitter_save_path()) + '.hkl')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        serializable = {}
        for nd_key, pairs in self._jitter_results.items():
            nd_str = str(nd_key)
            serializable[nd_str] = {}
            for cache_key, (j_avg, j_pval, j_pval_bins) in pairs.items():
                ref, tgt, res_key = cache_key[0], cache_key[1], cache_key[2]
                seg = cache_key[3] if len(cache_key) > 3 else None
                k = f"{ref}_{tgt}_{res_key}" + (f"_s{seg}" if seg is not None else "")
                serializable[nd_str][k] = {
                    'ref': np.int64(ref),
                    'tgt': np.int64(tgt),
                    'res_key': res_key,
                    'seg': np.int64(seg) if seg is not None else np.int64(-1),
                    'j_avg': np.asarray(j_avg, dtype=float),
                    'j_pval': np.float64(j_pval),
                    'j_pval_bins': np.asarray(j_pval_bins, dtype=float),
                }
        hkl.dump(serializable, p)
        total = sum(len(v) for v in self._jitter_results.values())
        print(f"[CCGDataset] jitter saved ({total} pairs) → {p}")

    def load_jitter(self, path: str = None) -> bool:
        """Load jitter results from disk into _jitter_results.

        Returns True on success, False if file not found.
        """
        import hickle as hkl
        p = os.path.expanduser((path or self.jitter_save_path()) + '.hkl')
        if not os.path.isfile(p):
            return False
        try:
            data = hkl.load(p)
            nd_key_map = {str(k): k for k in self._ccg}
            for nd_str, pairs in data.items():
                nd_key = nd_key_map.get(nd_str)
                if nd_key is None:
                    continue
                if nd_key not in self._jitter_results:
                    self._jitter_results[nd_key] = {}
                for k, v in pairs.items():
                    ref = int(v['ref'])
                    tgt = int(v['tgt'])
                    res_key = str(v['res_key'])
                    seg_raw = v.get('seg', None)
                    seg = None if (seg_raw is None or int(seg_raw) < 0) else int(seg_raw)
                    self._jitter_results[nd_key][(ref, tgt, res_key, seg)] = (
                        v['j_avg'], float(v['j_pval']), v['j_pval_bins'])
            total = sum(len(v) for v in self._jitter_results.values())
            print(f"[CCGDataset] jitter loaded ({total} pairs) ← {p}")
            return True
        except Exception as exc:
            print(f"[CCGDataset] jitter load failed: {exc}")
            return False

    def get_example_key(self):
        """Get an example key from data for testing"""
        if self.data:
            return next(iter(self.data.keys()))
        return None

    @staticmethod
    def _ask_overwrite(path: str, label: str) -> bool:
        """Prompt user before overwriting an existing cache file.

        Returns True if the user agrees to overwrite, False otherwise.
        """
        print(f"\n[CCGDataset] {label} file already exists at:\n  {path}")
        try:
            answer = input(f"  Overwrite with new computation? [y/N]: ").strip().lower()
        except EOFError:
            answer = ''
        return answer in ('y', 'yes')

    def get_ccg(self, baseline_method="eran_conv", use_segments=True):
        """
        main function of the class

        Cache strategy (split files):
          1. Try loading ccgdata (raw arrays) — only re-computed when COMPUTE_FIELDS change.
          2. If ccgdata loaded, try loading ccgpointers (significant pairs) — only
             re-computed when SIGNIF_FIELDS change.
          3. If ccgpointers stale, re-run EranConv (cheap) on cached ccgdata and save.
          4. If ccgdata missing, run full spike_correlations + EranConv then save both.

        When a named file exists but config has changed, the user is asked
        before overwriting.
        """
        if self.nd is None:
            return

        if baseline_method == "eran_conv":
            conv = EranConv(self.conf)

            # --- Step 1: try loading cached raw CCG arrays ---
            ccgdata_status = self.load_ccgdata()
            if ccgdata_status == 'loaded':
                # --- Step 2: try loading cached CCGPointers ---
                ptr_status = self.load_ccgpointers()
                if ptr_status == 'loaded':
                    print("[CCGDataset] Loaded CCGData + CCGPointers from split cache.")
                    return
                # Pointers stale/missing → re-run significance detection on cached CCGData
                if ptr_status == 'stale':
                    if not self._ask_overwrite(
                            self._ccgpointers_path() + '.hkl', 'CCGPointers'):
                        print("[CCGDataset] Aborted — keeping existing ccgpointers.")
                        return
                print("[CCGDataset] CCGData cached; re-running significance detection.")
                for nd_key, ccg_data in self._ccg.items():
                    self.__run_eranconv_on_ccgdata(nd_key, ccg_data, conv)
                self.save_ccgpointers()
                self._save_metadata()
                return

            if ccgdata_status == 'stale':
                # File exists but config changed — ask before overwriting
                if not self._ask_overwrite(
                        self._ccgdata_path() + '.hkl', 'CCGData'):
                    print("[CCGDataset] Aborted — keeping existing files.")
                    return

            # --- Fallback: try old monolithic cache ---
            if self._check_metadata() and self.load_data():
                print("[CCGDataset] Loaded from legacy monolithic cache.")
                return

            # --- Step 3: full computation ---
            missing_keys = [k for k in self.nd.edge_times.keys()
                            if k not in self._ccg]

            for key in self.nd.edge_times.keys():
                if key not in missing_keys:
                    print(self._session_summary(key))

            if not missing_keys:
                print("[CCGDataset] All sessions in cache, skipping computation.")
                return

            for key in missing_keys:
                self.__ccg_eranconv(key=key,
                                    conv=conv,
                                    edge_times=self.nd.edge_times[key],
                                    use_segments=use_segments)
            # Save both files separately
            self.save_ccgdata()
            self.save_ccgpointers()
            self._save_metadata()
        elif baseline_method == "jitter":
            raise NotImplementedError(
                "CCG jitter is implemented in neuropy.analyses.jitter. "
                "Use Jitter/JitterDataset (or the GUI on-demand jitter)."
            )
        else:
            raise ValueError(f"Unknown baseline_method: {baseline_method!r}")

    def reselect_pairs(self, new_alpha: float, method: str = 'bonferroni'):
        """Re-run pair selection with a new significance threshold.

        Reuses the cached CCG arrays and p-values (skips the expensive
        ``spike_correlations`` computation) and only re-runs convolution
        and significance masking.

        Updates ``self.conf.alpha``, ``self.data``, ``self.spurious``, and the
        ``pval_corrected``, ``qval_corrected``, ``significant`` fields in each
        ``CCGData`` entry.

        Parameters
        ----------
        new_alpha : float
            New significance threshold for EranConv pair selection.
        method : str
            Multiple-comparison method (e.g. ``'bonferroni'``).
        """
        self.conf.alpha = new_alpha
        self.data = {}
        self.spurious = {}

        for nd_key, ccg_data in self._ccg.items():
            neurons = self.nd.data[nd_key]
            edge_times = self.nd.edge_times[nd_key]

            conv = EranConv(self.conf)
            _, _, _, ccg_pointers, spur_pointers, printstr = conv.eranconv(
                neurons_key=nd_key,
                ccg=ccg_data.ccg,
                edge_times=edge_times,
                neuron_type=neurons.neuron_type,
                conf=self.conf,
            )

            ccg_data.pval_corrected = conv._pvals
            ccg_data.qval_corrected = conv._qvals
            ccg_data.significant    = conv._significant

            self._attr_append(nd_key, ccg_pointers, 'data')
            self._attr_append(nd_key, spur_pointers, 'spurious')
            print(printstr)

        n = sum(1 for v in self.data.values() if v is not None and v.n_pairs > 0)
        print(f"[reselect_pairs] alpha={new_alpha} → {len(self.data)} conn-type keys, "
              f"{n} non-empty")
        # Persist updated pointers (cheap) but leave ccgdata untouched
        self.save_ccgpointers()
        self._save_metadata()

    def load_highres(self, conf_highres: 'CCGConfig'=None, force_recompute: bool=False):
        """
        Load (or compute) high-resolution CCG arrays for all sessions.

        Tries to load from a previously saved highres file first; falls back to
        computing from spike trains if no file is found or ``force_recompute``
        is True.

        Only raw CCG spike-count arrays are computed — no significance test is
        run.  Low-res significance data in ``self._ccg`` and ``self.data``
        remain the authoritative source for pair selection.

        Results are stored in ``self._ccg_highres[nd_key]`` as :class:`CCGData`
        objects whose significance fields are all ``None``.

        Parameters
        ----------
        conf_highres : CCGConfig, optional
            Configuration specifying at minimum ``bin_size`` and ``duration``
            for the high-resolution CCG.
        force_recompute : bool, optional
            If True, skip the on-disk cache and always recompute from spikes.
        """
        from neuropy.analyses import correlations as _corr

        # Try loading from disk first (unless forced to recompute)
        if not force_recompute:
            if self._load_highres_from_disk():
                return
        else:
            # Force recompute requested — check if file exists and ask
            p = os.path.expanduser(self.highres_save_path() + '.hkl')
            if os.path.isfile(p):
                if not self._ask_overwrite(p, 'High-res CCG'):
                    print("[load_highres] Aborted — keeping existing file.")
                    return

        if conf_highres is None:
            import copy as _copy
            conf_highres = _copy.copy(self.conf)
            conf_highres.bin_size = _CCG_RESOLUTION['highres']

        if self.nd is None:
            raise RuntimeError(
                "CCGDataset.load_highres: nd is None — no NeuronsDataset attached.")

        for nd_key, neurons in self.nd.data.items():
            edge_times = self.nd.edge_times[nd_key]
            print(f"[load_highres] {nd_key} — bin_size="
                  f"{conf_highres.bin_size * 1e3:.2f} ms …", flush=True)
            ccg = _corr.spike_correlations(
                neurons=neurons,
                neuron_inds=np.arange(neurons.n_neurons),
                bin_size=conf_highres.bin_size,
                window_size=conf_highres.duration,
                use_acceleration=conf_highres.use_acceleration,
                symmetrize=conf_highres.symmetrize_ccg,
                edge_times=edge_times,
            )
            self._ccg_highres[nd_key] = CCGData(
                key=nd_key,
                _conf=conf_highres,
                ccg=ccg,
                ccg_null=None,
                pval=None,
                qval=None,
                pval_corrected=None,
                qval_corrected=None,
                significant=None,
                conn_strength=None,
                norm_factors=None,
            )
            print(f"[load_highres]   → shape {ccg.shape}")

        n = len(self._ccg_highres)
        print(f"[CCGDataset] load_highres complete — {n} session(s), "
              f"{conf_highres.bin_size * 1e3:.2f} ms bins.")
        # Run EranConv significance on the freshly computed high-res CCGs
        self.run_highres_eranconv()
        # Auto-save so future calls can skip re-computation
        self.save_highres()

    def run_highres_eranconv(self):
        """Run EranConv on high-resolution CCG data and store results in _ccg_highres.

        Infers bin_size from the actual CCG array shape (robust to conf.bin_size
        mutation in load_highres).  Results are stored directly on each CCGData
        object inside self._ccg_highres so that the high-res plot shows the
        EranConv null distribution and significance.
        """
        if not self._ccg_highres:
            print("[run_highres_eranconv] No high-res CCG loaded; call load_highres() first.")
            return
        for nd_key, ccg_hi in self._ccg_highres.items():
            conf = ccg_hi._conf
            ccg = ccg_hi.ccg  # [n_seg, n_ref, n_tgt, n_bins]
            n_bins = ccg.shape[-1]
            bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
            W = max(1, int(round(conf.conv_window / bin_size_eff)))
            print(f"[run_highres_eranconv] {nd_key} — "
                  f"shape={ccg.shape}, W={W} bins "
                  f"({W * bin_size_eff * 1e3:.1f} ms conv window)")
            pvals, pred, qvals = EranConv._conv(ccg, W=W)
            sig, pvals_corrected = EranConv.multiple_correction(
                pvals, alpha=conf.alpha, method=conf.multiple_correction)
            ccg_hi.ccg_null = pred
            ccg_hi.pval = pvals
            ccg_hi.qval = qvals
            ccg_hi.pval_corrected = pvals_corrected
            ccg_hi.significant = sig
            n_sig = int(sig.any(axis=-1).sum()) if sig is not None else 0
            print(f"[run_highres_eranconv]   → null shape {pred.shape}, "
                  f"{n_sig} significant pair-segments")
        print(f"[run_highres_eranconv] complete — {len(self._ccg_highres)} session(s).")

    def _session_summary(self, key) -> str:
        """Build the session summary string from stored CCGPointers.

        Produces the same output as the inline block in __ccg_eranconv so the
        identical summary is printed whether the session was freshly computed
        or restored from cache.

        Parameters
        ----------
        key : Key
            The nd-level key (from self.nd.edge_times.keys()) for the session.
        """
        neurons = self.nd.data[key.nd()]
        edge_times = self.nd.edge_times[key]
        et = edge_times.effective_time_hours.values

        s = f"======={key.session}=======\n"
        s += f"Segment(s) are {[f'{_:.2f}' for _ in et]} hours long "
        if self.nd.conf.sleep is not None:
            s += f"\nand contain {[f'{_:.2f}' for _ in et]} hours of actual sleep "
        for _ in self.nd.conf.neuron_types:
            s += f"{_}={neurons.get_neuron_type(_).n_neurons} "
        s += "\n"

        # Non-None fields of key used to filter stored CCGPointers to this session.
        nd_attrs = {
            f: getattr(key, f)
            for f in key.__dataclass_fields__
            if getattr(key, f) is not None
        }

        def belongs(k):
            return all(getattr(k, attr, None) == val for attr, val in nd_attrs.items())

        def count_seg(pointer, seg_i):
            if pointer is None or pointer.inds is None or len(pointer.inds) == 0:
                return 0
            if pointer.stored_by_segment:
                return int((pointer.inds[:, 0] == seg_i).sum())
            return pointer.n_pairs  # not segment-split; applies to all segments

        printstr = ''
        for i, (_, edge_time) in enumerate(edge_times.iterrows()):
            N_totalE, N_totalI = 0, 0
            for k, ptr in {**self.data, **self.spurious}.items():
                if not belongs(k):
                    continue
                n = count_seg(ptr, i)
                if k.excitability == 'E':
                    N_totalE += n
                elif k.excitability == 'I':
                    N_totalI += n

            printstr += f"{edge_time['label']:10}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "

            for EI, (ref, target) in self.conf.conn_types_labeled:
                ct = (ref, target)
                N = 0
                for k, ptr in self.data.items():
                    if not belongs(k):
                        continue
                    if (k.excitability == EI
                            and k.conn_type is not None
                            and tuple(k.conn_type) == ct):
                        N = count_seg(ptr, i)
                        break
                printstr += f"{ref}-{target}/{EI} {f'{N:02d}' if N > 0 else ' -'} | "
            printstr += '\n'

        return s + printstr

    def copy(self) -> "CCGDataset":
        """Copy only conf and nd (nd is a shallow reference)"""
        new = self.__class__(conf=self._conf)
        new.nd = self.nd
        return new

    def change_timescale(self,
                         bin_size,
                         duration=None,
                         jscale=None) -> 'CCGDataset':
        """
        Run CCG and convolution based significance test for all neurons

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        # change timescale in my configurations.
        print(
            f"recalculated CCG from binsize={self._conf.bin_size} to binsize={bin_size}"
        )
        self._conf.bin_size = bin_size
        if duration:
            self._conf.duration = duration
        if jscale:
            self._conf.jscale = jscale
        self.get_ccg()
        print("rescale completed")

    def save_plots(self,
                   root=str(_REPO_ROOT / "images" / "ccg_plots"),
                   source='data',
                   overwrite=False,
                   **filters):
        from datetime import datetime

        root = os.path.expanduser(root)
        os.makedirs(root, exist_ok=True)

        folder_name = input("Enter folder name (press Enter for auto-generated): ").strip()

        if folder_name:
            plot_folder = os.path.join(root, folder_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"tmp_{timestamp}"
            plot_folder = os.path.join(root, folder_name)

            # Clear previous tmp folders
            for f in os.listdir(root):
                if f.startswith("tmp_"):
                    p = os.path.join(root, f)
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                        print(f"Cleared: {f}")

        os.makedirs(plot_folder, exist_ok=True)
        print(f"Saving plots to: {plot_folder}")

        itergroup = self.filter(attrname=source, **filters)
        if itergroup is None:
            return

        for key, ccg_pointer in itergroup.items():
            print(f"ccg {key.session} {key.conn_type}")

            self._ccg[key.nd()].save_plots(
                pt=ccg_pointer,
                neurons=self.nd.data[key.nd()],
                frates_cut=self.nd.segment_firing_rates[key.nd()],
                neurons_config=self.nd.conf,
                plotdir=ccg_pointer.plotdir(plot_folder),
                overwrite=overwrite,
            )
        print(f"Done! Plots saved to: {plot_folder}")

    def get_connection_strengths(self, method=ConnStrengthMethod.PEAKSIZE):
        """
        Set connection_strengths value for ccg data based on given method.
        Values can be found in self._ccg[key].conn_strengths.
        """
        for key, ccg_data in self._ccg.items():
            if method == ConnStrengthMethod.TAILED:
                # frate: (n_seg, n_neurons) in Hz
                # total_time: (n_seg,) in hours  → multiply by 3600 to get seconds
                frate = self.nd.segment_firing_rates[key]
                total_time = self.nd.edge_times[key]['effective_time_hours'].values
                nspks = np.round(frate * total_time[:, None] * 3600)
                ccg_data.get_conn_strength(method=method, nspks=nspks)
            elif method == ConnStrengthMethod.PEAKSIZE:
                ccg_data.get_conn_strength(method=method)
            else:
                raise NotImplementedError()

    def plot_connection_strengths(
            self,
            n_segments_threshold=None,
            norm_by_n_sess=False,
            norm_by_total_strength=False,
            zero_first_timepoint=False,
            show_legend=False,
            save=False,
            root=str(_REPO_ROOT / "images" / "conn_strengths"),
            debug=False):

        for k, cp in self.data.items():
            pairs = cp.inds
            if n_segments_threshold is not None:
                ccg_data = self._ccg[k.nd()]
                mask = np.sum(ccg_data.significant[(
                    slice(None), *pairs.T)], axis=1) >= n_segments_threshold
                pairs = pairs[mask]
            ccg_data = self._ccg[k.nd()]
            plot_ccg.plot_strength(
                key=k,
                n_segments_threshold=n_segments_threshold,
                plot_data=ccg_data.conn_strength[:, pairs[:, 0], pairs[:, 1]],
                pairs=pairs,
                significant=ccg_data.significant[:, pairs[:, 0], pairs[:, 1]],
                n_segments=cp.n_segments,
                save=save,
                root=root,
                norm_by_n_sess=norm_by_n_sess,
                norm_by_total_strength=norm_by_total_strength,
                zero_first_timepoint=zero_first_timepoint,
                show_legend=show_legend,
                debug=debug)

    def __run_eranconv_on_ccgdata(self, nd_key, ccg_data, conv):
        """Run EranConv significance detection on already-loaded CCGData.

        Used when COMPUTE_FIELDS match (ccgdata cached) but SIGNIF_FIELDS have changed.
        Updates ``pval_corrected``, ``qval_corrected``, ``significant`` on the CCGData,
        and rebuilds ``self.data`` / ``self.spurious`` for this nd_key.
        """
        neurons = self.nd.data[nd_key]
        edge_times = self.nd.edge_times[nd_key]

        pvals, pred, qvals, ccg_pointers, spur_pointers, printstr = conv.eranconv(
            neurons_key=nd_key,
            ccg=ccg_data.ccg,
            edge_times=edge_times,
            neuron_type=neurons.neuron_type,
            conf=self.conf)

        ccg_data.ccg_null       = pred
        ccg_data.pval           = pvals
        ccg_data.qval           = qvals
        ccg_data.pval_corrected = conv._pvals
        ccg_data.qval_corrected = conv._qvals
        ccg_data.significant    = conv._significant

        self._attr_append(nd_key, ccg_pointers, 'data')
        self._attr_append(nd_key, spur_pointers, 'spurious')
        print(printstr)

    def __ccg_eranconv(self, key, conv, edge_times, use_segments=True):
        """
        Run CCG and generate a convolution-based baseline for all neurons in my NeuronsDataset.
        Run significance tests.
        Store results in objects:
            self._ccg
            self.data
            self.spurious.

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        print("EranConv significant pairs")

        neurons = self.nd.data[key.nd()]

        ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(neurons.n_neurons),  # all
            bin_size=self.conf.bin_size,
            window_size=self.conf.duration,
            use_acceleration=self.conf.use_acceleration,
            symmetrize=self.conf.symmetrize_ccg,
            edge_times=edge_times if use_segments else None,
        )

        pvals, pred, qvals, ccg_pointers, spur_pointers, printstr = conv.eranconv(
            neurons_key=key,
            ccg=ccg,
            edge_times=edge_times,
            neuron_type=neurons.neuron_type,
            conf=self.conf)

        ccg_data = CCGData(key=key,
                           _conf=self.conf,
                           ccg=ccg,
                           ccg_null=pred,
                           pval=pvals,
                           qval=qvals,
                           pval_corrected=conv._pvals,
                           qval_corrected=conv._qvals,
                           significant=conv._significant,
                           conn_strength=None,
                           norm_factors=None)

        self._ccg[key] = ccg_data
        self._attr_append(key, ccg_pointers, 'data')
        self._attr_append(key, spur_pointers, 'spurious')

        print(self._session_summary(key))


class EranConv:
    """
    A device for running EranConv and other significance tests    
    """

    def __init__(self, conf):
        self._pvals = []
        self._qvals = []
        self._significant = []  # final filtering results
        self.conf = conf

    @staticmethod
    def _conv(ccg, W=5, wintype="gauss", hollow_frac=None):
        """
        Estimate chance-level correlations using convolution method from Stark and Abeles (2009, J. Neuro Methods).
        Referencing MATLAB script EranConv.m written by the authors

        Parameters
        ----------
        ccg: np.array. 
            1D or 2D. (CCGs in columns)
            If 2D, elements in the first dimension are individual ccgs and second dimension are bins.
        W: 
            defines the width (unit: ms) of the convolution window, should be same as size of jitter window if were to use one
            `gauss`: W is standard deviation (sigma). Total window length will be 
            `rect`: Half size of window = W, total length is always odd
            `triang`: Window length is W rounded up to the nearest odd number

        wintype: ["gauss", "rect", "triang"]
            Type of convolution window.
            `gauss`: Gaussian kernel
            `rect`: rectangular kernel
            `triang`: triangular kernel

        hollow_frac: weight of the current bin
        
        Returns
        -------
        pvals: p-values (bin-wise)
        pred: predictor (expected values) 
        qvals: p-values (bin-wise) for inhibition
        """
        if len(ccg.shape) == 1:
            ccg = ccg[np.newaxis, ...]

        assert wintype in ["gauss", "rect", "triang"]
        assert W <= ccg.shape[-1]

        # Auto-assign appropriate hollow fraction if not specified
        # generate window
        # get center indices of window
        if wintype == "gauss":
            hollow_frac = hollow_frac or 0.6
            sigma = W / 2
            W = int(6 * sigma + (2 if W % 2 else 1))
            center = int(3 * sigma + (0.5 if W % 2 else 0))
            window = windows.gaussian(W, std=sigma) / (2 * np.pi * sigma)
        elif wintype == "rect":
            hollow_frac = hollow_frac or 0.42
            if W % 2 == 0:
                W += 1
            center = W // 2
            window = windows.boxcar(W)
        elif wintype == "triang":
            hollow_frac = hollow_frac or 0.63
            W = 2 * W + (-1 if W % 2 else 1)
            center = W // 2
            window = windows.triang(W)

        # hollow and normalize window
        window[center] *= (1 - hollow_frac)
        window /= np.sum(window)
        # padding
        ccg_pad = np.concatenate(
            [ccg[..., :W][..., ::-1], ccg, ccg[..., -W:][..., ::-1]], axis=-1)

        # convolve window with ccg
        pred = ndimage.convolve1d(ccg_pad, window, axis=-1)
        pred = pred[..., W:-W]

        # mid-p Poisson test: P( val<=pred ) + half of P ( val==pred )
        pvals = 1 - poisson.cdf(ccg - 1, pred) - poisson.pmf(ccg, pred) * 0.5
        qvals = 1 - pvals
        return pvals, pred, qvals

    @staticmethod
    def multiple_correction(pvals: np.ndarray,
                            alpha: float,
                            method: str = 'bonferroni'
                            ) -> tuple:
        """Per-pair multiple-comparison correction over bins only.

        For each (seg, ref, tgt) triple independently, correct the ``n_bins``
        p-values for that pair.  Segments and pairs are fully decoupled — they
        never inflate each other's correction penalty.

        Parameters
        ----------
        pvals : ndarray, shape ``[n_seg, n_ref, n_tgt, n_bins]``
            Raw Poisson p-values from :meth:`_conv`.
        alpha : float
            Significance threshold (applied to corrected p-values).
        method : str
            ``'bonferroni'`` (default) — multiply each p-value by ``n_bins``,
            clip at 1.  Fast, conservative, and transparent.
            Any other string accepted by
            ``statsmodels.stats.multitest.multipletests`` also works
            (e.g. ``'fdr_bh'``).

        Returns
        -------
        significance : bool ndarray, same shape as *pvals*
        corrected_pvals : float ndarray, same shape as *pvals*
        """
        if method == 'bonferroni':
            n_bins = pvals.shape[-1]
            corrected = np.minimum(pvals * n_bins, 1.0)
            return corrected <= alpha, corrected

        # Fallback for FDR-BH and other statsmodels methods.
        significance = np.zeros_like(pvals, dtype=bool)
        corrected_pvals = np.ones_like(pvals, dtype=float)
        for idx in np.ndindex(pvals.shape[:-1]):
            row = pvals[idx]
            s, pc, _, _ = multipletests(row, alpha=alpha, method=method)
            significance[idx] = s
            corrected_pvals[idx] = pc
        return significance, corrected_pvals

    def spkcount_mask(self, ccg):
        min_bin = self.conf.min_spkcnt_bin
        max_bin = self.conf.max_spkcnt_bin
        threshold = self.conf.min_spkcount
        # Use mean across the spkcount window so that a hollow center bin
        # (zero spike count at lag=0) doesn't discard the whole pair.
        # Previously used .all(axis=-1) which required EVERY bin >= threshold.
        pair_inds = np.argwhere(
            ccg[..., min_bin:max_bin].mean(axis=-1) >= threshold)
        # NOTE right now it's the same criteria for excitation/inhibition
        return pair_inds

    def significance_mask(self, p, excitability):
        """Return pair indices with significant CCG peaks.

        Excitatory (E): a bin in [min_lag, max_lag) must survive MC correction.
        Inhibitory (I): a surviving bin must have a surviving neighbour at the
        looser ``alpha2`` threshold (ensures trough, not just noise).
        """
        conf = self.conf
        method = conf.multiple_correction if conf.multiple_correction is not None else 'bonferroni'

        if excitability == 'E':
            sig, self._pvals = EranConv.multiple_correction(p, conf.alpha, method=method)
            # At least one corrected-significant bin in the excitatory test window.
            has_valid_peak = sig[..., conf.min_lag_bin:conf.max_lag_bin].any(axis=-1)
            pair_inds = np.argwhere(has_valid_peak)
        elif excitability == 'I':
            sig1, self._qvals = EranConv.multiple_correction(p, conf.alpha, method=method)
            sig2, _ = EranConv.multiple_correction(p, conf.alpha2, method=method)
            # Bin must be significant at alpha AND have a neighbour at alpha2.
            neighbor = sig1 & (np.roll(sig2, 1, -1) | np.roll(sig2, -1, -1))
            pair_inds = np.argwhere(neighbor.any(-1))
        else:
            raise ValueError(f"Unknown excitability: {excitability!r}")
        return pair_inds

    def _cell_type_mask(self, pair_inds, neuron_type, conn_types):
        sig_pairs = {}
        # Conn types with no pairs are marked with None
        if not _hasvalue(pair_inds):
            for ct in conn_types:
                sig_pairs[ct] = None

        # Condition 3: Ref/Target are specific cell types
        for ct in conn_types:
            inds = np.where(
                np.isin(pair_inds[:, -2], np.where(neuron_type == ct[0])) &
                np.isin(pair_inds[:, -1], np.where(neuron_type == ct[1])))[0]
            sig_pairs[ct] = pair_inds[inds] if inds.shape[0] else None
        return sig_pairs

    def eranconv(
        self,
        neurons_key: Key,
        ccg,
        edge_times: pd.DataFrame,
        neuron_type,
        conf: CCGConfig,
    ):
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
        print("running eranconv (1st pass)")
        key = neurons_key
        self.conf = conf
        self.n_segments = edge_times.shape[0]

        pvals, pred, qvals = EranConv._conv(ccg,
                                            W=conf.conv_window_bins,
                                            wintype="gauss",
                                            hollow_frac=None)

        def build_inds(p, EI, conn_types):
            rough_inds = SetOp.intersect(self.significance_mask(p, EI),
                                         self.spkcount_mask(ccg))
            inds = self._cell_type_mask(rough_inds, neuron_type, conn_types)
            return rough_inds, inds

        # [n_seg, n_pair, 2]
        rough_inds_E, inds_E = build_inds(pvals, 'E', conf.conn_types_E)
        rough_inds_I, inds_I = build_inds(qvals, 'I', conf.conn_types_I)

        # Record a global map of significant pairs
        self._significant = np.zeros(ccg.shape[:3], dtype=bool)
        for inds in [inds_E, inds_I]:
            if inds is None:
                continue
            for k, v in inds.items():
                if v is None:
                    continue
                self._significant[tuple(v.T)] = True

        ccg_inds_by_type, spur_inds_by_type = {}, {}

        # Force CCG to be 4D
        if ccg.ndim == 3:
            ccg = ccg[None]
            pred = pred[None]
            for attr in (
                    "_pvals",
                    "_qvals",
            ):
                setattr(self, attr, getattr(self, attr)[None])

        count = np.zeros((edge_times.shape[0], len(self.conf.conn_types_flat)),
                         dtype=int)
        j = 0
        # Update return values
        for EI in ['E', 'I']:
            spurious = rough_inds_E if EI == 'E' else rough_inds_I  # initialize spurious pairs

            for conn_type in self.conf.conn_types[EI]:
                inds = inds_E[conn_type] if EI == 'E' else inds_I[conn_type]
                ccg_key = key.add(conn_type=conn_type, excitability=EI)
                ccg_pointer = CCGPointer(key=ccg_key,
                                         conf=self.conf,
                                         inds=inds if _hasvalue(inds) else None,
                                         edge_times=edge_times)
                for i, ccg in enumerate(ccg_pointer.split()):
                    count[i, j] = ccg.n_pairs if ccg is not None else 0
                ccg_inds_by_type[ccg_key] = ccg_pointer
                spurious = SetOp.setdiff(
                    spurious, inds)  # remove these pairs from spurious
                j += 1

            spur_key = key.add(excitability=EI)
            spur_inds_by_type[spur_key] = CCGPointer(
                key=spur_key,
                conf=self.conf,
                inds=spurious if _hasvalue(spurious) else None,
                edge_times=edge_times)

        printstr = ''
        for i, (segment_i, edge_time) in enumerate(edge_times.iterrows()):
            N_totalE = (rough_inds_E[:, 0] == i).sum()
            N_totalI = (rough_inds_I[:, 0] == i).sum()
            printstr += f"{edge_time['label']:10}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "

            for N, (EI, (ref, target)) in zip(count[i], self.conf.conn_types_labeled):
                printstr += f"{ref}-{target}/{EI} {f'{N:02d}' if N>0 else ' -'} | "
            printstr += '\n'

        print("eranconv done")

        return pvals, pred, qvals, ccg_inds_by_type, spur_inds_by_type, printstr


