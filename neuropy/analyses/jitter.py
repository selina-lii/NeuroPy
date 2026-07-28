from enum import Enum
from pathlib import Path as _Path
from typing import Union

import numpy as np
import pandas as pd

import neuropy.analyses.correlations as correlations
from neuropy.analyses.utils import _san, AnalysisDataset
from neuropy.core.neurons import Neurons

_REPO_ROOT = _Path(__file__).resolve().parents[2]

try:
    import cupy as cp
except ImportError:
    cp = None


import collections as _collections

JitterTask = _collections.namedtuple(
    'JitterTask',
    ['tag', 'ref', 'tgt', 'njitter', 'res_key', 'bin_size_eff', 'seg_arg', 't0', 't1', 'nd_key'],
    defaults=[None, None, None, None],
)


class JitterType(Enum):
    SPIKE_TIMING = 0
    INTERVAL = 1


class JitterConfig:
    def __init__(
        self,
        ccg,                             # CCGConfig
        njitter: int = 100,
        jitter_type: JitterType = JitterType.INTERVAL,
        jscale: float = 5e-3,
        alpha: float = 0.05,
        use_acceleration: bool = True,
        ccg_batch_bytes: int = 256_000_000,
    ):
        """
        Parameters
        ----------
        ccg : CCGConfig
            CCG configuration (bin_size, duration, test window, etc.)
        njitter : int
            Number of jitter repetitions.
        jitter_type : JitterType
            INTERVAL (shift spikes within jscale-wide intervals) or SPIKE_TIMING (uniform spike shift).
        jscale : float
            Jitter interval width in seconds. Spikes placed uniformly within each interval.
        alpha : float
            Significance threshold.
        use_acceleration : bool
            Use CuPy GPU acceleration where available.
        ccg_batch_bytes : int
            Memory budget (bytes) for jitter trials per batch. Larger = more parallelism.
        """
        self.ccg = ccg
        self.njitter = njitter
        self.jitter_type = jitter_type
        self.jscale = jscale
        self.alpha = alpha
        self.use_acceleration = use_acceleration
        self.ccg_batch_bytes = int(ccg_batch_bytes) if ccg_batch_bytes is not None else 256_000_000

    def __str__(self):
        return (
            f"njitter:{self.njitter}, jitter_type:{self.jitter_type}, "
            f"jscale:{self.jscale}, alpha:{self.alpha}, "
            f"use_acceleration:{self.use_acceleration}"
        )

    @property
    def jscale_ms(self):
        return self.jscale * 1e3

    @property
    def jscale_bins(self):
        return self.jscale / self.ccg.bin_size


class Jitter:
    """
    Jitter significance test over CCGPointer pairs (one session).

    Groups pairs by target so each target is jittered once. Per (seg, ref, tgt):
    p = fraction of njitter jitter-CCG window sums ≥ real CCG window sum.
    """

    def __init__(self, key, neurons, conf: JitterConfig,
                 ccg_ptr, ccg_data):
        """
        Parameters
        ----------
        key : Key
            Session / epoch key.
        neurons : Neurons
            Neuron object for this session (spike trains, IDs, etc.)
        conf : JitterConfig
        ccg_ptr : CCGPointer
            Holds the selected pair indices (from EranConv).
        ccg_data : CCGData
            Holds the raw CCG array for this session
            (shape [n_seg, n_ref_all, n_tgt_all, n_bins]).
        """
        self.key = key
        self.conf = conf
        self.neurons = neurons
        self.ccg_ptr = ccg_ptr
        self.ccg_data = ccg_data

        # Filled by run()
        self.j_sig = None    # [n_pairs] bool
        self.pval = None     # [n_pairs] float
        self.pval_bins = None  # [n_pairs, n_bins] per-bin empirical p-values
        self.JBSI = None     # [n_pairs, n_bins] float
        # pair_idx → (j_avg [n_bins], j_lo [n_bins], j_hi [n_bins])
        self._j_ccg_cache: dict = {}

        self._group_by_target()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _group_by_target(self):
        """Group pair indices by target neuron for efficient computation."""
        inds = self.ccg_ptr.inds
        if inds is None or len(inds) == 0:
            self.jtgt_inds = np.array([], dtype=int)
            self.jref_inds = []
            self.pair_pos = {}
            return

        tgt_col = inds[:, -1]
        keys, inv = np.unique(tgt_col, return_inverse=True)
        self.jtgt_inds = keys
        # All unique refs per target (across every segment of that target)
        self.jref_inds = [
            np.unique(inds[inv == i, -2]).tolist() for i in range(len(keys))
        ]
        self.pair_pos = {int(k): np.where(inv == i)[0] for i, k in enumerate(keys)}

    @property
    def n_pairs(self):
        inds = self.ccg_ptr.inds
        return len(inds) if inds is not None else 0

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def run(self):
        """Run jitter and populate self.j_sig, self.pval, self.JBSI."""
        inds = self.ccg_ptr.inds
        if inds is None or len(inds) == 0:
            self.j_sig = np.array([], dtype=bool)
            self.pval = np.array([], dtype=float)
            self.JBSI = np.zeros((0, self.conf.ccg.nbins))
            return

        n_pairs = len(inds)
        self.j_sig = np.zeros(n_pairs, dtype=bool)
        self.pval = np.ones(n_pairs, dtype=float)
        self.pval_bins = np.ones((n_pairs, self.conf.ccg.nbins), dtype=float)
        self.JBSI = np.zeros((n_pairs, self.conf.ccg.nbins))

        stored_by_segment = self.ccg_ptr.stored_by_segment
        ccg_conf = self.conf.ccg
        lb, ub = ccg_conf.min_lag_bin, ccg_conf.max_lag_bin

        for refs, tgt in zip(self.jref_inds, self.jtgt_inds):
            tgt = int(tgt)
            pos = self.pair_pos[tgt]   # indices into inds for this target

            # 1. Generate jittered spike trains for this target
            j_trains = self._jitter_trains(tgt)  # list of njitter arrays

            # 2. Compute CCG: refs × jittered-targets
            #    j_ccg shape: [n_ref, njitter, n_bins]
            j_ccg = self._compute_jitter_ccg(refs, tgt, j_trains)

            ref_list = list(refs)

            for pair_idx in pos:
                ref = int(inds[pair_idx, -2])
                ref_i = ref_list.index(ref)

                # Jitter distribution: summed count in test window for each jitter
                j_vals = j_ccg[ref_i, :, lb:ub].sum(axis=-1)   # [njitter]

                # Real CCG in test window (from ccg_data)
                if stored_by_segment:
                    seg = int(inds[pair_idx, 0])
                    real_val = self.ccg_data.ccg[seg, ref, tgt, lb:ub].sum()
                    j_ccg_avg = j_ccg[ref_i].mean(axis=0)  # [n_bins]
                    real_ccg = self.ccg_data.ccg[seg, ref, tgt]
                else:
                    real_val = self.ccg_data.ccg[:, ref, tgt, lb:ub].sum()
                    j_ccg_avg = j_ccg[ref_i].mean(axis=0)
                    real_ccg = self.ccg_data.ccg[:, ref, tgt].sum(axis=0)

                # Cache null distribution for verification plotting
                j_ccg_lo = np.percentile(j_ccg[ref_i], 5, axis=0)
                j_ccg_hi = np.percentile(j_ccg[ref_i], 95, axis=0)
                self._j_ccg_cache[pair_idx] = (j_ccg_avg, j_ccg_lo, j_ccg_hi)

                # Per-bin empirical p-values: fraction of jitter trials
                # where the jitter count >= real count at each bin
                # j_ccg[ref_i] shape: [njitter, n_bins], real_ccg shape: [n_bins]
                self.pval_bins[pair_idx] = np.mean(
                    j_ccg[ref_i] >= real_ccg[np.newaxis, :], axis=0)

                pval = float(np.mean(j_vals >= real_val))
                self.pval[pair_idx] = pval
                self.j_sig[pair_idx] = pval <= self.conf.alpha
                self.JBSI[pair_idx] = self._jbsi(
                    ref, tgt, real_ccg, j_ccg_avg)

    def _compute_jitter_ccg_one(self, ref_ind, tgt_ind, j_trains):
        """Single-ref jitter CCG via spike_correlations. Returns shape [1, njitter, n_bins].

        Used by jitter_worker (always n_ref=1). Builds one combined Neurons
        object with all njitter jittered trains and calls spike_correlations once.
        """
        ccg_conf = self.conf.ccg
        j_list = j_trains if isinstance(j_trains, list) else list(j_trains)
        njitter = min(int(self.conf.njitter), len(j_list))
        tgt_id_base = int(self.neurons.neuron_ids[tgt_ind])
        tgt_type = (self.neurons.neuron_type[tgt_ind][0]
                    if getattr(self.neurons, 'neuron_type', None) is not None else None)
        ref_neurons = self.neurons.neuron_slice(neuron_inds=np.array([ref_ind]))
        j_neurons = Neurons(
            spiketrains=j_list[:njitter],
            t_start=self.neurons.t_start,
            t_stop=self.neurons.t_stop,
            neuron_ids=[tgt_id_base * 100000 + i for i in range(njitter)],
            neuron_type=([tgt_type] * njitter) if tgt_type is not None else None,
        )
        combined = ref_neurons
        combined.merge(j_neurons)
        return correlations.spike_correlations(
            neurons=combined,
            ref_neuron_inds=np.array([0]),
            neuron_inds=np.arange(1, 1 + njitter),
            bin_size=ccg_conf.bin_size,
            window_size=ccg_conf.duration,
            use_acceleration=ccg_conf.use_acceleration,
            symmetrize=ccg_conf.symmetrize_ccg,
            one_to_many=True,
        )  # shape [1, njitter, n_bins]

    def _compute_jitter_ccg_batch(self, refs, tgt_ind, j_trains):
        """Multi-ref jitter CCG with memory-bounded batching. Returns [n_ref, njitter, n_bins]."""
        ccg_conf = self.conf.ccg
        j_list = j_trains if isinstance(j_trains, list) else list(j_trains)
        njitter = min(int(self.conf.njitter), len(j_list))
        n_ref = len(refs)
        tgt_id_base = int(self.neurons.neuron_ids[tgt_ind])
        tgt_type = (self.neurons.neuron_type[tgt_ind][0]
                    if getattr(self.neurons, 'neuron_type', None) is not None else None)
        try:
            n_bins_est = int(round(ccg_conf.duration / ccg_conf.bin_size)) + 1
        except Exception:
            n_bins_est = 64
        budget = int(getattr(self.conf, 'ccg_batch_bytes', 256_000_000) or 256_000_000)
        batch = max(1, min(njitter, budget // max(1, n_ref * n_bins_est * 8)))
        out = None
        n_bins = None
        done = 0
        while done < njitter:
            b = min(batch, njitter - done)
            ref_neurons = self.neurons.neuron_slice(neuron_inds=np.asarray(refs))
            j_neurons = Neurons(
                spiketrains=j_list[done:done + b],
                t_start=self.neurons.t_start,
                t_stop=self.neurons.t_stop,
                neuron_ids=[tgt_id_base * 100000 + (done + i) for i in range(b)],
                neuron_type=([tgt_type] * b) if tgt_type is not None else None,
            )
            combined = ref_neurons
            combined.merge(j_neurons)
            ccg_j = correlations.spike_correlations(
                neurons=combined,
                ref_neuron_inds=np.arange(n_ref),
                neuron_inds=np.arange(n_ref, n_ref + b),
                bin_size=ccg_conf.bin_size,
                window_size=ccg_conf.duration,
                use_acceleration=ccg_conf.use_acceleration,
                symmetrize=ccg_conf.symmetrize_ccg,
                one_to_many=(n_ref == 1),
            )
            if out is None:
                n_bins = int(ccg_j.shape[-1])
                if not n_bins:
                    raise ValueError("spike_correlations returned empty CCG")
                out = np.empty((n_ref, njitter, n_bins), dtype=float)
            elif int(ccg_j.shape[-1]) != n_bins:
                raise ValueError(f"jitter CCG bin mismatch: expected {n_bins}, got {ccg_j.shape[-1]}")
            out[:, done:done + b, :] = ccg_j
            done += b
        return out

    def _compute_jitter_ccg(self, refs, tgt_ind, j_trains):
        """Dispatch to one- or batch-ref jitter CCG computation."""
        if len(refs) == 1:
            return self._compute_jitter_ccg_one(refs[0], tgt_ind, j_trains)
        return self._compute_jitter_ccg_batch(refs, tgt_ind, j_trains)

    def _jbsi(self, ref, tgt, real_ccg, j_ccg_avg):
        """
        Jitter-Based Synchrony Index (Agmon 2012).

        real_ccg : [n_bins]   — real CCG for this pair/segment
        j_ccg_avg: [n_bins]   — mean jitter CCG across njitter
        """
        fr_ref = self.neurons.firing_rate[ref]
        fr_tgt = self.neurons.firing_rate[tgt]
        return compute_jbsi(
            real_ccg=real_ccg,
            j_ccg_avg=j_ccg_avg,
            fr_ref=fr_ref,
            fr_tgt=fr_tgt,
            bin_size=self.conf.ccg.bin_size,
            jscale=self.conf.jscale,
        )

    # ------------------------------------------------------------------
    # Jitter generation
    # ------------------------------------------------------------------

    def _jitter_trains(self, tgt_ind):
        """
        INTERVAL: each spike placed uniformly within its jscale-wide window.
        SPIKE_TIMING: each spike shifted by uniform draw over [-jscale, +jscale].
        """
        spiketrain = np.asarray(self.neurons.spiketrains[tgt_ind])
        n_spikes = len(spiketrain)
        sr = self.neurons.sampling_rate
        njitter = int(self.conf.njitter)
        jscale_samples = int(self.conf.jscale * sr)

        if self.conf.jitter_type == JitterType.INTERVAL:
            if self.conf.use_acceleration and cp is not None:
                trains = (
                    cp.sort(cp.floor(
                        (cp.floor(
                            cp.round(cp.array(spiketrain) * sr) / jscale_samples
                        ) + cp.random.rand(njitter, n_spikes)) * jscale_samples
                    )) / sr
                ).get()
            else:
                trains = (
                    np.sort(np.floor(
                        (np.floor(
                            np.round(spiketrain * sr) / jscale_samples
                        ) + np.random.rand(njitter, n_spikes)) * jscale_samples
                    )) / sr
                )
        else:  # SPIKE_TIMING
            trains = (
                np.round(
                    (spiketrain
                     + 2 * self.conf.jscale * np.random.rand(njitter, n_spikes)
                     - self.conf.jscale) * sr
                ) / sr
            )

        # Keep as a numpy array (njitter, n_spikes); callers can slice without
        # materializing Python lists.
        return trains


def compute_jbsi(*, real_ccg, j_ccg_avg, fr_ref, fr_tgt, bin_size: float, jscale: float):
    """Compute Jitter-Based Synchrony Index (Agmon 2012) trace.

    Parameters
    ----------
    real_ccg : array-like, shape (n_bins,)
        Real CCG counts for the pair.
    j_ccg_avg : array-like, shape (n_bins,)
        Mean jitter CCG across jitter trials.
    fr_ref, fr_tgt : float
        Reference/target firing rates (Hz) used in the normalization factor.
    bin_size : float
        CCG bin size (seconds).
    jscale : float
        Full jitter interval width (seconds).
    """
    real_ccg = np.asarray(real_ccg, dtype=float)
    j_ccg_avg = np.asarray(j_ccg_avg, dtype=float)
    n1 = np.minimum(np.asarray(fr_ref, dtype=float), np.asarray(fr_tgt, dtype=float))
    ts = float(bin_size)
    tj = float(jscale)
    b = tj / (tj - ts) if tj / ts > 2 else 2.0
    factor = b / (n1 + 1e-12)
    if np.ndim(factor) > 0:
        factor = factor[..., None]  # per-pair factor broadcasts over the bin axis
    return factor * (real_ccg - j_ccg_avg)


def plot_jitter_verification(jitter_obj,
                             max_pairs: int = 16,
                             sort_by_pval: bool = True,
                             figsize_per_panel: tuple = (3.5, 2.8),
                             ncols: int = 4,
                             save_path: str = None):
    """Standalone verification plot for a :class:`Jitter` result.

    Requires that ``jitter_obj._j_ccg_cache`` is populated (set during
    :meth:`Jitter.run`).  If the cache is not available (e.g. a loaded
    result), only the stored JBSI and p-values are shown.

    Parameters
    ----------
    jitter_obj : Jitter
        A fully-run Jitter instance.
    max_pairs : int
        Maximum number of panels to plot.
    sort_by_pval : bool
        Show lowest p-value pairs first.
    figsize_per_panel : tuple
        ``(width, height)`` in inches per panel.
    ncols : int
        Number of grid columns.
    save_path : str, optional
        If provided, save figure here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    j = jitter_obj
    inds = j.ccg_ptr.inds
    if inds is None or len(inds) == 0:
        print("[plot_jitter_verification] No pairs to plot.")
        return None

    n_pairs = len(inds)
    order = (np.argsort(j.pval) if sort_by_pval else np.arange(n_pairs))
    order = order[:max_pairs]

    ncols = min(ncols, len(order))
    nrows = int(np.ceil(len(order) / ncols))
    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)

    ccg_conf = j.conf.ccg
    bins = np.linspace(-ccg_conf.duration / 2,
                        ccg_conf.duration / 2,
                        ccg_conf.nbins)
    bins_ms = bins * 1e3
    lb, ub = ccg_conf.min_lag_bin, ccg_conf.max_lag_bin

    # Retrieve real CCG and jitter cache
    cd = j.ccg_data
    stored_by_seg = j.ccg_ptr.stored_by_segment
    j_ccg_cache = getattr(j, '_j_ccg_cache', None)  # populated by patched run()

    for plot_i, pair_i in enumerate(order):
        row, col = plot_i // ncols, plot_i % ncols
        ax = axes[row][col]

        ref = int(inds[pair_i, -2])
        tgt = int(inds[pair_i, -1])

        # Real CCG: sum across segments if stored_by_segment
        if stored_by_seg:
            seg = int(inds[pair_i, 0])
            real_ccg = cd.ccg[seg, ref, tgt]
        else:
            real_ccg = cd.ccg[:, ref, tgt].sum(axis=0)

        ax.bar(bins_ms, real_ccg, width=(bins_ms[1] - bins_ms[0]) * 0.9,
               color='#1565C0', alpha=0.7, label='real CCG')

        # Mark test window
        ax.axvspan(bins_ms[lb], bins_ms[ub - 1], alpha=0.15, color='green',
                   label='test window')

        # Null distribution from cache: mean ± 5–95 % band
        cache_entry = j_ccg_cache.get(pair_i) if j_ccg_cache else None
        if cache_entry is not None:
            j_avg, j_lo, j_hi = cache_entry
            ax.plot(bins_ms, j_avg, color='orange', lw=1.4, label='jitter mean')
            ax.fill_between(bins_ms, j_lo, j_hi,
                            color='orange', alpha=0.25, label='jitter 5–95%')
        elif j.JBSI is not None and pair_i < j.JBSI.shape[0]:
            # Fallback: show JBSI on a twin axis when no raw cache is available
            ax2 = ax.twinx()
            ax2.plot(bins_ms, j.JBSI[pair_i], color='orange', lw=1.2, label='JBSI')
            ax2.set_ylabel('JBSI', fontsize=6, color='orange')
            ax2.tick_params(axis='y', labelsize=5, colors='orange')

        pval = j.pval[pair_i] if j.pval is not None else float('nan')
        sig = bool(j.j_sig[pair_i]) if j.j_sig is not None else False
        ax.set_title(f"{'*' if sig else ''}  {ref}→{tgt}  p={pval:.3f}",
                     fontsize=8, pad=2)
        ax.set_xlabel('lag (ms)', fontsize=7)
        ax.set_ylabel('spikes', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axvline(0, color='k', lw=0.5, ls='--')

    # Hide unused panels
    for plot_i in range(len(order), nrows * ncols):
        axes[plot_i // ncols][plot_i % ncols].set_visible(False)

    key_str = str(j.key)
    fig.suptitle(
        f"Jitter verification — {key_str}\n"
        f"njitter={j.conf.njitter}, jscale={j.conf.jscale*1e3:.0f} ms, "
        f"α={j.conf.alpha}  |  "
        f"{int(j.j_sig.sum()) if j.j_sig is not None else '?'}/{n_pairs} sig",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[plot_jitter_verification] Saved to {save_path}")

    return fig


class JitterDataset(AnalysisDataset):
    """
    Runs and stores jitter results for all sessions in a CCGDataset.

    Typical usage::

        jconf = JitterConfig(ccg=conf, njitter=100)
        jd = JitterDataset(nd=nd, cd=ccgs, conf=jconf)
        jd.run_jitter()
    """

    def __init__(self, nd, cd, conf: JitterConfig):
        super().__init__(conf)
        self.nd = nd
        self.cd = cd
        self.data = {}   # Key → Jitter

    def run_jitter(self):
        for key, ccg_ptr in self.cd.ptr.items():
            neurons = self.nd.get_neurons(key)
            ccg_data = self.cd.ccg_for(key)

            if ccg_ptr.n_pairs == 0:
                self.data[key] = None
                continue

            j = Jitter(
                key=key,
                neurons=neurons,
                conf=self.conf,
                ccg_ptr=ccg_ptr,
                ccg_data=ccg_data,
            )
            j.run()
            self.data[key] = j
            print(
                f"[Jitter] {key}: "
                f"{j.j_sig.sum()}/{j.n_pairs} pairs passed"
            )


import collections as _collections
import multiprocessing as _mp
import threading as _threading
import time as _time

_MAX_JITTER_QUEUE = 50


class JitterManager:
    """Backend manager for on-demand single-pair jitter computation.

    Handles subprocess/thread task queue and LRU result cache.
    UI callbacks (on_jitter_done, on_custom_done) are called on completion.
    """
    CACHE_MAX = 500
    UNVIEWED_BG = '#FFEE58'
    UNVIEWED_FG = '#333333'
    VIEWED_BG   = '#FFF9C4'
    VIEWED_FG   = '#333333'

    def __init__(self, key, neurons, ccg_data, ccg_ptr,
                 compute_custom_fn, on_jitter_done, on_custom_done):
        self.key = key
        self.neurons = neurons
        self.ccg_data = ccg_data
        self.ccg_ptr = ccg_ptr
        self._compute_custom = compute_custom_fn
        self._on_jitter_done = on_jitter_done
        self._on_custom_done = on_custom_done
        self._cache = _collections.OrderedDict()
        self._unviewed: set = set()
        self._pending: _collections.deque = _collections.deque()
        self._proc: _mp.Process = None
        self._result_queue: _mp.Queue = None
        self._thread: _threading.Thread = None
        self._thread_result: list = []

    # ── Cache ──────────────────────────────────────────────────────────

    def cache_get(self, key):
        return self._cache.get(key)

    def cache_put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self.CACHE_MAX:
            self._cache.popitem(last=False)

    def cache_pop(self, key):
        return self._cache.pop(key, None)

    def has_any_res(self, ref, tgt):
        return any((ref, tgt, r) in self._cache for r in ('hi', 'lo'))

    def mark_viewed(self, ref, tgt):
        """Mark pair as viewed; returns True if it was unviewed."""
        pair = (ref, tgt)
        if pair in self._unviewed:
            self._unviewed.discard(pair)
            return True
        return False

    def discard_unviewed(self, ref, tgt):
        self._unviewed.discard((ref, tgt))

    def is_unviewed(self, ref, tgt):
        return (ref, tgt) in self._unviewed

    # ── Queue management ───────────────────────────────────────────────

    def is_running(self):
        return ((self._proc is not None and self._proc.is_alive()) or
                (self._thread is not None and self._thread.is_alive()))

    def queue_size(self):
        return len(self._pending)

    def current_task(self):
        return self._pending[0] if self._pending else None

    def enqueue_jitter(self, ref, tgt, njitter, res_key, bin_size_eff):
        """Enqueue a jitter task. Returns False if queue full."""
        running = 1 if self.is_running() else 0
        if running + len(self._pending) >= _MAX_JITTER_QUEUE:
            return False
        self._pending.append(('jitter', ref, tgt, njitter, res_key, bin_size_eff))
        self.start_next()
        return True

    def enqueue_custom_ccg(self, t0, t1, name, neurons_override, active_duration, filter_state):
        """Enqueue a custom-CCG task. Returns False if queue full."""
        running = 1 if self.is_running() else 0
        if running + len(self._pending) >= _MAX_JITTER_QUEUE:
            return False
        self._pending.append(('custom_ccg', t0, t1, name, neurons_override, active_duration, filter_state))
        self.start_next()
        return True

    def start_next(self):
        """Start the next queued task if none is running."""
        if self.is_running() or not self._pending:
            return
        task = self._pending[0]
        if task[0] == 'jitter':
            from neuropy.analyses._jitter_worker import jitter_worker
            _, ref, tgt, njitter, res_key, bin_size_eff = task
            self._result_queue = _mp.Queue()
            self._proc = _mp.Process(
                target=jitter_worker,
                args=(self._result_queue, self.key, self.neurons,
                      self.ccg_data, None,
                      ref, tgt, njitter, bin_size_eff),
                daemon=True,
            )
            self._proc.start()
        elif task[0] == 'custom_ccg':
            _, t0, t1, name, neurons_override, active_duration, filter_state = task
            self._thread_result.clear()
            t_start = _time.monotonic()

            def _worker(_t0=t0, _t1=t1, _name=name, _no=neurons_override,
                        _ad=active_duration, _fs=filter_state):
                try:
                    result = self._compute_custom(_t0, _t1, _name,
                                                  neurons_override=_no,
                                                  active_duration=_ad)
                    if result is not None:
                        result['filter_state'] = _fs
                        result['compute_sec'] = _time.monotonic() - t_start
                    self._thread_result.append(
                        result if result is not None else {'error': 'compute returned None'})
                except Exception as ex:
                    self._thread_result.append({'error': str(ex)})

            self._thread = _threading.Thread(target=_worker, daemon=True)
            self._thread.start()

    def poll(self):
        """Poll for completed tasks; call callbacks on completion. Returns True if still running."""
        if self.is_running():
            return True
        completed = self._pending.popleft() if self._pending else None
        if completed is None:
            return False
        task_type = completed[0]

        if task_type == 'jitter':
            result = None
            try:
                if self._result_queue is not None and not self._result_queue.empty():
                    result = self._result_queue.get_nowait()
            except Exception:
                pass
            if self._proc is not None:
                self._proc.join(timeout=1)
                self._proc = None
            self._result_queue = None
            if result is not None and not result.get('error') and result.get('j_avg') is not None:
                res_key = completed[4]
                cache_key = (result['ref'], result['tgt'], res_key)
                self.cache_put(cache_key, (result['j_avg'], result['j_pval'], result['j_pval_bins']))
                self._unviewed.add((result['ref'], result['tgt']))
                self._on_jitter_done(result['ref'], result['tgt'], res_key,
                                     result['j_avg'], result['j_pval'], result['j_pval_bins'])
            elif result is not None and result.get('error'):
                self._on_jitter_done(None, None, None, None, None, None,
                                     error=result['error'])
        elif task_type == 'custom_ccg':
            if self._thread is not None:
                self._thread.join(timeout=1)
                self._thread = None
            result = self._thread_result[0] if self._thread_result else None
            self._thread_result.clear()
            self._on_custom_done(result)

        self.start_next()
        return self.is_running()


class JitterResults:
    """On-disk jitter surrogate results (separate from CCG compute cache)."""

    def __init__(self, conf: 'CCGConfig'):
        self.conf = conf
        self.results: dict = {}

    def save_path(self) -> str:
        return self.conf.save_path + '_jitter'

    def save(self, path: str = None) -> None:
        if not self.results:
            print("[save_jitter] Nothing to save.")
            return
        import hickle as hkl
        p = os.path.expanduser((path or self.save_path()) + '.hkl')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        serializable = {}
        for nd_key, pairs in self.results.items():
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
        total = sum(len(v) for v in self.results.values())
        print(f"[CCGDataset] jitter saved ({total} pairs) → {p}")

    def load(self,
             nd_key=None,
             path: str = None,
             known_keys=None) -> bool:
        import hickle as hkl
        p = os.path.expanduser((path or self.save_path()) + '.hkl')
        if not os.path.isfile(p):
            return False
        known_keys = known_keys or []
        nd_key_map = {str(k): k for k in known_keys}
        try:
            data = hkl.load(p)
            for nd_str, pairs in data.items():
                mapped_key = nd_key_map.get(nd_str)
                if mapped_key is None:
                    continue
                if nd_key is not None and mapped_key != nd_key:
                    continue
                if mapped_key not in self.results:
                    self.results[mapped_key] = {}
                for k, v in pairs.items():
                    ref = int(v['ref'])
                    tgt = int(v['tgt'])
                    res_key = str(v['res_key'])
                    seg_raw = v.get('seg', None)
                    seg = (None if (seg_raw is None or int(seg_raw) < 0)
                           else int(seg_raw))
                    self.results[mapped_key][(ref, tgt, res_key, seg)] = (
                        v['j_avg'], float(v['j_pval']), v['j_pval_bins'])
            loaded = (sum(len(v) for v in self.results.values())
                      if nd_key is None
                      else len(self.results.get(nd_key, {})))
            label = f"[{nd_key}]" if nd_key is not None else "(all sessions)"
            print(f"[CCGDataset] jitter loaded {label} ({loaded} pairs) ← {p}")
            return True
        except Exception as exc:
            print(f"[CCGDataset] jitter load failed: {exc}")
            return False

