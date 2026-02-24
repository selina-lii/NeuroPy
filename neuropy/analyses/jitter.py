from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

import neuropy.analyses.correlations as correlations
from neuropy.analyses.utils import _san, AnalysisDataset
from neuropy.core.neurons import Neurons

try:
    import cupy as cp
except ImportError:
    cp = None


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
    ):
        """
        Parameters
        ----------
        ccg : CCGConfig
            CCG configuration (bin_size, duration, test window, etc.)
        njitter : int
            Number of jitter repetitions.
        jitter_type : JitterType
            INTERVAL (Agmon 2012) or SPIKE_TIMING (uniform spike shift).
        jscale : float
            Full jitter interval width in seconds (default 5 ms intervals).
            Spikes are placed uniformly within the jscale-wide interval they
            belong to.  To match the paper (Agmon 2012, 20 ms intervals) set
            jscale=20e-3.
        alpha : float
            Significance threshold.
        use_acceleration : bool
            Use CuPy GPU acceleration where available.
        """
        self.ccg = ccg
        self.njitter = njitter
        self.jitter_type = jitter_type
        self.jscale = jscale
        self.alpha = alpha
        self.use_acceleration = use_acceleration

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
    Jitter-based significance test for a set of selected neuron pairs (one session).

    Uses only the pairs identified by EranConv (the "selected indices" stored in a
    CCGPointer) rather than all N² pairs, making it much faster.  Pairs are grouped
    by target neuron so that each target's spike train is jittered only once.

    Significance test (per pair, per segment when inds are segment-stored):
        1. Generate `njitter` interval-jittered spike trains for the target neuron.
        2. Compute CCG between each ref neuron and each jittered train
           → j_ccg shape [n_ref, njitter, n_bins].
        3. For each (seg?, ref, tgt) in the pointer's inds:
             real_val  = sum of real CCG in the test window (from ccg_data)
             j_vals    = sum of jitter CCG in the test window across njitter trials
             p-value   = fraction of j_vals ≥ real_val
             significant if p-value ≤ alpha.
    """

    def __init__(self, key, neurons, conf: JitterConfig,
                 ccg_pointer, ccg_data):
        """
        Parameters
        ----------
        key : Key
            Session / epoch key.
        neurons : Neurons
            Neuron object for this session (spike trains, IDs, etc.)
        conf : JitterConfig
        ccg_pointer : CCGPointer
            Holds the selected pair indices (from EranConv).
        ccg_data : CCGData
            Holds the raw CCG array for this session
            (shape [n_seg, n_ref_all, n_tgt_all, n_bins]).
        """
        self.key = key
        self.conf = conf
        self.neurons = neurons
        self.ccg_pointer = ccg_pointer
        self.ccg_data = ccg_data

        # Filled by run()
        self.j_sig = None    # [n_pairs] bool
        self.pval = None     # [n_pairs] float
        self.JBSI = None     # [n_pairs, n_bins] float
        # pair_idx → (j_avg [n_bins], j_lo [n_bins], j_hi [n_bins])
        self._j_ccg_cache: dict = {}

        self._group_by_target()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _group_by_target(self):
        """Group pair indices by target neuron for efficient computation."""
        inds = self.ccg_pointer.inds
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
        inds = self.ccg_pointer.inds
        return len(inds) if inds is not None else 0

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def run(self):
        """Run jitter and populate self.j_sig, self.pval, self.JBSI."""
        inds = self.ccg_pointer.inds
        if inds is None or len(inds) == 0:
            self.j_sig = np.array([], dtype=bool)
            self.pval = np.array([], dtype=float)
            self.JBSI = np.zeros((0, self.conf.ccg.nbins))
            return

        n_pairs = len(inds)
        self.j_sig = np.zeros(n_pairs, dtype=bool)
        self.pval = np.ones(n_pairs, dtype=float)
        self.JBSI = np.zeros((n_pairs, self.conf.ccg.nbins))

        stored_by_segment = self.ccg_pointer.stored_by_segment
        ccg_conf = self.conf.ccg
        lb, ub = ccg_conf.min_lag_bin, ccg_conf.max_lag_bin

        for refs, tgt in zip(self.jref_inds, self.jtgt_inds):
            tgt = int(tgt)
            pos = self.pair_pos[tgt]   # indices into inds for this target

            # 1. Generate jittered spike trains for this target
            j_trains = self._jitter_trains(tgt)  # list of njitter arrays

            # 2. Compute CCG: refs × jittered-targets
            #    j_ccg shape: [n_ref, njitter, n_bins]
            j_ccg = self._compute_jitter_ccg(refs, j_trains)

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

                pval = float(np.mean(j_vals >= real_val))
                self.pval[pair_idx] = pval
                self.j_sig[pair_idx] = pval <= self.conf.alpha
                self.JBSI[pair_idx] = self._jbsi(
                    ref, tgt, real_ccg, j_ccg_avg)

    def _compute_jitter_ccg(self, refs, j_trains):
        """
        Compute CCG between ref neurons and njitter jittered target spike trains.

        Returns j_ccg of shape [n_ref, njitter, n_bins].
        """
        njitter = self.conf.njitter
        ccg_conf = self.conf.ccg

        # Slice ref neurons out of the full neuron set
        ref_neurons = self.neurons.neuron_slice(neuron_inds=refs)

        # Build a Neurons object for the jittered spike trains.
        # Give each jitter a distinct ID so they're treated as separate neurons.
        tgt_id = self.neurons.neuron_ids[refs[0]]  # use ref's ID range as base
        j_ids = [tgt_id * 100000 + j for j in range(njitter)]
        j_type = [self.neurons.neuron_type[refs[0]][0]] * njitter
        j_neurons = Neurons(
            spiketrains=j_trains,
            t_start=self.neurons.t_start,
            t_stop=self.neurons.t_stop,
            neuron_ids=j_ids,
            neuron_type=j_type,
        )
        combined = ref_neurons
        combined.merge(j_neurons)

        # 2groups: refs are group 0 (inds 0..n_ref-1),
        #          jitters are group 1 (inds n_ref..n_ref+njitter-1)
        # Returns shape [n_ref, njitter, n_bins]
        j_ccg = correlations.spike_correlations(
            neurons=combined,
            ref_neuron_inds=np.arange(len(refs)),
            neuron_inds=len(refs) + np.arange(njitter),
            bin_size=ccg_conf.bin_size,
            window_size=ccg_conf.duration,
            use_acceleration=ccg_conf.use_acceleration,
            symmetrize=ccg_conf.symmetrize_ccg,  # match real CCG shape
        )
        return j_ccg

    def _jbsi(self, ref, tgt, real_ccg, j_ccg_avg):
        """
        Jitter-Based Synchrony Index (Agmon 2012).

        real_ccg : [n_bins]   — real CCG for this pair/segment
        j_ccg_avg: [n_bins]   — mean jitter CCG across njitter
        """
        ccg_conf = self.conf.ccg
        fr_ref = self.neurons.firing_rate[ref]
        fr_tgt = self.neurons.firing_rate[tgt]
        n1 = np.minimum(fr_ref, fr_tgt)

        ts = ccg_conf.bin_size
        tj = self.conf.jscale
        b = tj / (tj - ts) if tj / ts > 2 else 2.0
        return b / (n1 + 1e-12) * (real_ccg - j_ccg_avg)

    # ------------------------------------------------------------------
    # Jitter generation
    # ------------------------------------------------------------------

    def _jitter_trains(self, tgt_ind):
        """
        Generate self.conf.njitter jittered spike trains for neuron tgt_ind.

        INTERVAL jitter (Agmon 2012): each spike is placed uniformly at random
        within the jscale-wide interval it originally fell in.  jscale is the
        full interval width (not a half-width).
        SPIKE_TIMING: each spike is shifted by a uniform draw over [-jscale, +jscale].
        """
        spiketrain = np.asarray(self.neurons.spiketrains[tgt_ind])
        n_spikes = len(spiketrain)
        sr = self.neurons.sampling_rate
        njitter = self.conf.njitter
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

        return list(trains)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def significant_inds(self):
        """
        Return the subset of CCGPointer.inds that passed jitter.
        Returns None if nothing survived.
        """
        if self.j_sig is None or not self.j_sig.any():
            return None
        return self.ccg_pointer.inds[self.j_sig]

    # ------------------------------------------------------------------
    # Save / inspect intermediates
    # ------------------------------------------------------------------

    def save(self, save_dir: str = None) -> str:
        """Save all jitter result arrays to *save_dir* as .npy files.

        Files written
        -------------
        ``jitter_inds.npy``   — pair indices, shape ``[n_pairs, 2 or 3]``
        ``jitter_pval.npy``   — empirical p-values, shape ``[n_pairs]``
        ``jitter_jsig.npy``   — significance flags, shape ``[n_pairs]``
        ``jitter_JBSI.npy``   — JBSI traces, shape ``[n_pairs, n_bins]``
        ``jitter_meta.txt``   — text summary (key, conf, n_sig / n_pairs)

        Parameters
        ----------
        save_dir : str, optional
            Destination directory.  Defaults to
            ``~/Documents/ms_synchrony/NeuroPy/data/jitter/<key_str>``.

        Returns
        -------
        str  — the directory where files were written.
        """
        import os

        if save_dir is None:
            key_str = str(self.key).replace(' ', '_').replace('/', '-')
            save_dir = os.path.expanduser(
                f"~/Documents/ms_synchrony/NeuroPy/data/jitter/{key_str}")
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'jitter_inds.npy'), self.ccg_pointer.inds)
        np.save(os.path.join(save_dir, 'jitter_pval.npy'), self.pval)
        np.save(os.path.join(save_dir, 'jitter_jsig.npy'), self.j_sig)
        np.save(os.path.join(save_dir, 'jitter_JBSI.npy'), self.JBSI)

        # Save null distribution cache if available
        if self._j_ccg_cache:
            n_pairs = len(self.ccg_pointer.inds)
            n_bins = self.conf.ccg.nbins
            j_avg_arr = np.full((n_pairs, n_bins), np.nan)
            j_lo_arr  = np.full((n_pairs, n_bins), np.nan)
            j_hi_arr  = np.full((n_pairs, n_bins), np.nan)
            for idx, (avg, lo, hi) in self._j_ccg_cache.items():
                j_avg_arr[idx] = avg
                j_lo_arr[idx]  = lo
                j_hi_arr[idx]  = hi
            np.save(os.path.join(save_dir, 'jitter_null_avg.npy'), j_avg_arr)
            np.save(os.path.join(save_dir, 'jitter_null_lo.npy'),  j_lo_arr)
            np.save(os.path.join(save_dir, 'jitter_null_hi.npy'),  j_hi_arr)

        has_null = bool(self._j_ccg_cache)
        meta = (
            f"key:        {self.key}\n"
            f"njitter:    {self.conf.njitter}\n"
            f"jscale:     {self.conf.jscale * 1e3:.1f} ms\n"
            f"alpha:      {self.conf.alpha}\n"
            f"n_pairs:    {self.n_pairs}\n"
            f"n_sig:      {int(self.j_sig.sum()) if self.j_sig is not None else 'n/a'}\n"
            f"inds_shape: {self.ccg_pointer.inds.shape}\n"
            f"null_cache: {'yes (jitter_null_avg/lo/hi.npy)' if has_null else 'no'}\n"
        )
        with open(os.path.join(save_dir, 'jitter_meta.txt'), 'w') as fh:
            fh.write(meta)

        print(f"[Jitter.save] Saved to {save_dir}")
        print(meta)
        return save_dir

    def plot_verification(self,
                          max_pairs: int = 16,
                          sort_by_pval: bool = True,
                          figsize_per_panel: tuple = (3.5, 2.8),
                          ncols: int = 4,
                          save_path: str = None):
        """Plot real CCG vs mean jitter CCG for every pair — for eyeballing.

        Each panel shows:
        - Blue bars  : real CCG
        - Orange line: mean jitter CCG (null model)
        - Grey region: 5–95 % percentile band of jitter CCGs
        - Title      : ``ref→tgt  p={:.3f}  {'*' if sig}``

        Parameters
        ----------
        max_pairs : int
            Cap the number of panels (sorted by p-value, lowest first).
        sort_by_pval : bool
            Show most-significant pairs first.
        figsize_per_panel : tuple
            (width, height) in inches per panel.
        ncols : int
            Grid columns.
        save_path : str, optional
            If given, save the figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        return plot_jitter_verification(
            self,
            max_pairs=max_pairs,
            sort_by_pval=sort_by_pval,
            figsize_per_panel=figsize_per_panel,
            ncols=ncols,
            save_path=save_path,
        )


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
    inds = j.ccg_pointer.inds
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
    stored_by_seg = j.ccg_pointer.stored_by_segment
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

    def run_jitter(self, save_progress=False):
        for key, ccg_pointer in self.cd.data.items():
            nd_key = key.nd()
            neurons = self.nd.data[nd_key]
            ccg_data = self.cd._ccg[nd_key]

            if ccg_pointer.n_pairs == 0:
                self.data[key] = None
                continue

            j = Jitter(
                key=key,
                neurons=neurons,
                conf=self.conf,
                ccg_pointer=ccg_pointer,
                ccg_data=ccg_data,
            )
            j.run()
            self.data[key] = j
            print(
                f"[Jitter] {key}: "
                f"{j.j_sig.sum()}/{j.n_pairs} pairs passed"
            )
