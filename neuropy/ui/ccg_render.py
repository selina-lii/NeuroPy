"""
Standalone (Tk-free) CCG PNG renderer.

Used by neuropy/ui/pregen.py as a subprocess worker; does NOT import tkinter.
Mirrors the caching logic of CCGReviewUI._png_path and _render_png for
normal (non-custom, non-jitter) segments using a fixed cache_config dict.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — no display required
from matplotlib.figure import Figure
import matplotlib.pyplot as _plt

from neuropy.plotting import ccg as plot_ccg
from neuropy.analyses.ms_connectivity import NormalizeBy, apply_norms_to_ccg

_ALL_SEGS_LABEL = "All_segments"


# ---------------------------------------------------------------------------
# Thin proxy classes so apply_norms_to_ccg works without real neurons/nd
# ---------------------------------------------------------------------------

class _NeuronsProxy:
    def __init__(self, firing_rate, shank_ids):
        self.firing_rate = np.asarray(firing_rate, dtype=float) if firing_rate is not None else None
        self.shank_ids   = np.asarray(shank_ids,   dtype=int)   if shank_ids   is not None else None


class _NDProxy:
    """Minimal nd proxy supporting nd.edge_times[nd_key].iloc[seg][col]."""
    def __init__(self, edge_times_by_key: dict):
        self.edge_times = _EdgeTimesLookup(edge_times_by_key)


class _EdgeTimesLookup:
    def __init__(self, d):
        self._d = d

    def __getitem__(self, nd_key):
        times = self._d.get(str(nd_key), [])
        return _EdgeTimesDF(times)


class _EdgeTimesDF:
    def __init__(self, times):
        self.iloc = _ILocProxy(times)


class _ILocProxy:
    def __init__(self, times):
        self._t = times

    def __getitem__(self, seg):
        val = float(self._t[seg]) if seg < len(self._t) else 1.0
        return {'effective_time_hours': val}


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class CCGRenderer:
    """
    Standalone renderer for CCG pair PNGs (no Tk dependency).

    Produces files with the same naming scheme as CCGReviewUI._png_path so
    they are recognised as cache hits when the UI is running.

    Parameters
    ----------
    ccg_lo, ccg_hi : CCGData or None
        Low- and (optional) high-resolution CCGData objects.
    n_segments : int
        Number of real recording segments.
    segment_names : list[str]
        Label for each segment; length == n_segments.
    tmp_dir : str
        Directory where PNG files are written.
    cache_config : dict
        Display-state snapshot produced by CCGReviewUI._current_display_config().
    neurons : _NeuronsProxy or None
        Neuron attributes used by normalisation (firing_rate, shank_ids).
    edge_times : list[float] or None
        ``effective_time_hours`` per segment for TIME_SPAN normalisation.
    nd_key : str
        Session identifier used as the edge_times lookup key.
    """

    def __init__(
        self,
        ccg_lo,
        ccg_hi,
        n_segments: int,
        segment_names: list[str],
        tmp_dir: str,
        cache_config: dict,
        neurons: _NeuronsProxy | None = None,
        edge_times: list[float] | None = None,
        nd_key: str = '',
    ):
        self.ccg_lo = ccg_lo
        self.ccg_hi = ccg_hi
        self.n_segments = n_segments
        self.segment_names = segment_names
        self.tmp_dir = tmp_dir
        self.cache_config = cache_config
        self.neurons = neurons
        self.edge_times = edge_times
        self.nd_key = nd_key

    # ------------------------------------------------------------------ paths

    def png_path(self, ref: int, tgt: int, segment: int, hires: bool) -> str:
        """Return canonical cache path — must match CCGReviewUI._png_path output."""
        seg_name = (
            _ALL_SEGS_LABEL if segment == self.n_segments
            else self.segment_names[segment]
        )
        cfg = self.cache_config
        norms = cfg.get('active_norms') or []
        norm_key = '_'.join(sorted(norms)) if norms else 'raw'

        cd = self.ccg_hi if (hires and self.ccg_hi is not None) else self.ccg_lo
        alpha = cfg.get('active_alpha', 0.001)
        alpha_key = (f'_a{alpha:.3f}'
                     if (cd is not None and getattr(cd, 'pval_corrected', None) is not None)
                     else '')
        res_key = '_hi' if hires else '_lo'
        # j_key and scale_key are always '' in the subprocess (no jitter, no scale mode)
        return os.path.join(
            self.tmp_dir,
            f"pair_{ref}_{tgt}_{seg_name}_{norm_key}{alpha_key}{res_key}.png"
        )

    # ---------------------------------------------------------------- resolve

    def _resolve(self, ref: int, tgt: int, segment: int, hires: bool):
        """Return (ccg_raw, ccg_null_raw, pval, pval_c, seg_label, conf)."""
        cd = self.ccg_hi if (hires and self.ccg_hi is not None) else self.ccg_lo
        is_all = (segment == self.n_segments)
        if is_all:
            ccg_raw      = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            ccg_null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                            if cd.ccg_null is not None else None)
            pval, pval_c = None, None
            seg_label    = "All segments"
        else:
            ccg_raw      = cd.ccg[segment, ref, tgt, :]
            ccg_null_raw = (cd.ccg_null[segment, ref, tgt, :]
                            if cd.ccg_null is not None else None)
            pval         = (cd.pval[segment, ref, tgt, :]
                            if getattr(cd, 'pval', None) is not None else None)
            pval_c       = (cd.pval_corrected[segment, ref, tgt, :]
                            if getattr(cd, 'pval_corrected', None) is not None else None)
            seg_label    = self.segment_names[segment]
        return ccg_raw, ccg_null_raw, pval, pval_c, seg_label, cd.conf

    # ----------------------------------------------------------------- render

    def render_png(self, ref: int, tgt: int, segment: int, hires: bool) -> str:
        """Render the CCG panel, save to disk, return path."""
        cfg = self.cache_config
        ccg_raw, ccg_null_raw, pval, pval_c, seg_label, conf = self._resolve(
            ref, tgt, segment, hires)

        # Build active norms set
        norms_set: set = set()
        for n in NormalizeBy:
            if n.name in (cfg.get('active_norms') or []):
                norms_set.add(n)

        nd_proxy = (
            _NDProxy({self.nd_key: self.edge_times})
            if self.edge_times is not None else None
        )
        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw, ccg_null_raw, ref, tgt, segment,
            norms_set, self.neurons, nd_proxy, self.nd_key,
            self.n_segments, False,
        )

        n_bins = len(ccg)
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

        _method = cfg.get('_conn_str_method_var', 'conv')
        _null_raw = ccg_null if _method == 'conv' else None
        show_null = (_null_raw if cfg.get('_baseline_show_var', True)
                     else None)
        show_pval   = pval     if cfg.get('_sig_conv_p_var')         else None
        show_pval_c = pval_c   if cfg.get('_sig_conv_pc_var')        else None
        alpha = cfg.get('active_alpha', 0.001)

        ids = (
            str(int(self.neurons.shank_ids[ref]))
            if (self.neurons and self.neurons.shank_ids is not None) else str(ref),
            str(int(self.neurons.shank_ids[tgt]))
            if (self.neurons and self.neurons.shank_ids is not None) else str(tgt),
        )

        fig = Figure(figsize=(7, 5))
        ax  = fig.add_subplot(111)
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg, ids=ids, inds=(ref, tgt),
            window_size=conf.duration, bin_size=bin_size_eff,
            pval=show_pval, pval_corrected=show_pval_c,
            alpha=alpha, ccg_null=show_null,
            j_ccg=None, j_pval=None,
            segment_id=seg_label,
            is_significant_pair=False,   # cosmetic only in subprocess
            min_lag=conf.min_lag if cfg.get('_sig_test_window_var') else None,
            max_lag=conf.max_lag if cfg.get('_sig_test_window_var') else None,
            normalize_info=(', '.join(cfg['active_norms'])
                            if cfg.get('active_norms') else None),
            acg_ref=None, acg_tgt=None,
            acg_yscale_ref=1.0, acg_yscale_tgt=1.0,
            acg_match_ccg=False,
            show_ccg=cfg.get('_ccg_show_var', True),
            line_ccg=cfg.get('_line_ccg_var', False),
            line_baseline=cfg.get('_line_baseline_var', False),
            line_ref=cfg.get('_line_ref_var', False),
            line_tgt=cfg.get('_line_tgt_var', False),
            line_jitter=False,
            conn_strength_baseline=None,
        )

        path = self.png_path(ref, tgt, segment, hires)
        fig.savefig(path, dpi=100, bbox_inches='tight')
        _plt.close(fig)
        return path
