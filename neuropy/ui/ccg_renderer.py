"""
CCGRenderEngine: data preparation and PNG rendering for CCGReviewUI.

CCGRenderEngine.build_context() resolves all display parameters, loads and
transforms data (deconvolution, normalization, CS/baseline, jitter, waveform).
CCGRenderEngine.write_png() handles only figure creation and saving.

This keeps _render_png in ccg_ui.py as a thin ~12-line coordinator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot

from neuropy.analyses.ms_connectivity import (
    NormalizeBy, apply_norms_to_ccg, compute_ccg_panel_data, deconv_autocorr,
)
import neuropy.plotting.ccg as plot_ccg


# ---------------------------------------------------------------------------
# Pure helpers (no UI dependency)
# ---------------------------------------------------------------------------

def deconvolve_ccg(
    ccg_raw, null_raw,
    acg_ref, nspks_ref,
    acg_tgt, nspks_tgt,
    dref: bool, dtgt: bool,
):
    """Apply ACG deconvolution to CCG and null arrays.  Returns (ccg_out, null_out)."""
    def _deconv_1d(x):
        if x is None:
            return None
        x = x.copy().astype(float)
        if dref and dtgt:
            return deconv_autocorr(x, acg_ref, nspks_ref, acg_tgt, nspks_tgt)
        if dref:
            return deconv_autocorr(x, acg_ref, nspks_ref,
                                   np.zeros_like(acg_tgt, dtype=float), 1.0)
        return deconv_autocorr(x, np.zeros_like(acg_ref, dtype=float), 1.0,
                               acg_tgt, nspks_tgt)

    try:
        return _deconv_1d(ccg_raw), _deconv_1d(null_raw)
    except Exception:
        return ccg_raw, null_raw


def _fill_waveform(wf_neuron, shank_id: int, ch_per_shank: int, discarded):
    """Expand a (possibly trimmed) per-neuron waveform to a full (ch_per_shank, T) array.

    Discarded channels are filled with np.nan.
    """
    if wf_neuron.ndim == 1:
        return np.tile(wf_neuron, (ch_per_shank, 1))
    sid = int(shank_id)
    channel_ids = ch_per_shank * sid + np.arange(ch_per_shank)
    if discarded is None:
        start  = int(ch_per_shank * sid)
        length = ch_per_shank
        mask   = np.ones(ch_per_shank, dtype=bool)
    else:
        mask   = ~np.isin(channel_ids, discarded)
        start  = int(ch_per_shank * sid - np.sum(discarded < ch_per_shank * sid))
        length = int(np.sum(mask))
    clean = np.full((ch_per_shank, wf_neuron.shape[-1]), np.nan)
    clean[mask] = wf_neuron[start:start + length]
    return clean


# ---------------------------------------------------------------------------
# RenderContext — everything write_png needs, no UI state
# ---------------------------------------------------------------------------

@dataclass
class RenderContext:
    """All processed data and parameters for rendering one CCG PNG."""
    # Core CCG
    ccg: np.ndarray
    ccg_null_plot: Optional[np.ndarray]       # None when baseline hidden
    pval: Optional[np.ndarray]
    pval_corrected: Optional[np.ndarray]
    # Jitter overlay
    j_ccg: Optional[np.ndarray]
    j_pval: Optional[np.ndarray]
    j_ccg_lo: Optional[np.ndarray]
    j_ccg_hi: Optional[np.ndarray]
    # ACG overlays
    acg_ref: Optional[np.ndarray]
    acg_tgt: Optional[np.ndarray]
    # Peak waveform inset
    wf_peak_ms: Optional[np.ndarray]
    wf_peak_amp: Optional[np.ndarray]
    # CS baseline overlay
    cs_baseline_arg: Optional[np.ndarray]
    # Timing
    window_size_eff: float
    bin_size_eff: float
    # Display metadata
    alpha: float
    norm_info: Optional[str]
    seg_id_display: str
    min_lag_plot: Optional[float]
    max_lag_plot: Optional[float]
    extend_on: bool
    # Pair identity
    inds: tuple
    ids: tuple                    # (shank_label_ref, shank_label_tgt)
    neuron_type: tuple
    is_significant_pair: bool
    # Style
    show_ccg: bool
    line_ccg: bool
    line_baseline: bool
    line_ref: bool
    line_tgt: bool
    line_jitter: bool
    acg_yscale_ref: float
    acg_yscale_tgt: float
    acg_match_ccg: bool
    # Y-axis scale (None = auto)
    ylim: Optional[tuple]
    # Export/preview style overrides (None = use plot_ccg_panel defaults)
    ccg_color:          Optional[str]
    baseline_color:     Optional[str]
    ccg_alpha:          Optional[float]
    baseline_alpha:     Optional[float]
    cs_shade_color:     Optional[str]
    show_legend:        bool
    xticks_ms:          Optional[list]
    mirror_xticks:      bool
    min_text_size:      Optional[float]
    show_test_window:   Optional[bool]   # None = legacy (show when lags provided)
    test_window_color:  Optional[str]
    test_window_alpha:  Optional[float]
    pval_line_color:    Optional[str]
    alpha_line_color:   Optional[str]
    cs_annotation_lines: list[str]
    # Title / ylabel visibility flags
    title_show_shanks:      bool
    title_show_inds:        bool
    title_show_type:        bool
    title_show_seg:         bool
    title_show_norm_details: bool
    title_show_session:     bool
    title_session_label:    str   # precomputed "session=NSD 2" string
    dark_mode:              bool = False


# ---------------------------------------------------------------------------
# CCGRenderEngine
# ---------------------------------------------------------------------------

class CCGRenderEngine:
    """Builds RenderContext objects and writes PNG files for CCGReviewUI.

    Holds a back-reference to the UI so data sources and display state are
    accessible, but write_png() is a pure function of RenderContext — no UI
    state reads happen there.
    """

    def __init__(self, ui):
        self._ui = ui

    # ----------------------------------------------------------------
    # Display-state resolution helpers
    # ----------------------------------------------------------------

    def _rsig(self, name: str, render_cfg) -> bool:
        _map = {
            'conv_p':      '_sig_conv_p_var',
            'conv_pc':     '_sig_conv_pc_var',
            'test_window': '_sig_test_window_var',
            'jitter_pc':   '_sig_jitter_pc_var',
        }
        if render_cfg:
            return bool(render_cfg.get(_map[name], False))
        return self._ui._sig(name)

    def _rline(self, attr: str, render_cfg) -> bool:
        """Read a line-mode bool from render_cfg or correlogram panel var."""
        if render_cfg:
            return bool(render_cfg.get(attr, False))
        v = getattr(self._ui.center_container.correlogram_panel, attr, None)
        return bool(v.get()) if v is not None else False

    def _racg(self, attr: str, render_cfg, default=None):
        """Read an ACG/correlogram display var from render_cfg or live UI."""
        if render_cfg:
            return render_cfg.get(attr, default)
        return self._ui._acg_var_get(attr, default)

    # ----------------------------------------------------------------
    # Waveform loading
    # ----------------------------------------------------------------

    def _load_waveform(self, ref: int, render_cfg, highres: bool):
        """Return (wf_ms, wf_amp) for the peak channel of neuron `ref`, or (None, None)."""
        if not highres or not self._racg('_peak_wf_var', render_cfg, False):
            return None, None
        neurons = self._ui.neurons
        if neurons is None:
            return None, None
        wf_all = getattr(neurons, 'waveforms',     None)
        pc      = getattr(neurons, 'peak_channels', None)
        sids    = getattr(neurons, 'shank_ids',     None)
        if wf_all is None or pc is None or sids is None:
            return None, None

        nd_conf   = getattr(getattr(self._ui.cd, 'nd', None), '_conf', None)
        ch_ps     = int(getattr(nd_conf, 'ch_per_shank', 16) or 16)
        recinfo   = getattr(nd_conf, 'recinfo', None) if nd_conf else None
        skipped   = getattr(recinfo, 'skipped_channels', None) if recinfo else None
        discarded = None if skipped is None else np.asarray(skipped, dtype=int)

        try:
            peak_ch = int(pc[ref])
            rs      = int(sids[ref])
        except (IndexError, TypeError, ValueError):
            return None, None

        disc_set = (set(int(x) for x in discarded.ravel())
                    if discarded is not None and discarded.size else set())
        if peak_ch in disc_set:
            return None, None
        local_idx = peak_ch - ch_ps * rs
        if not (0 <= local_idx < ch_ps):
            return None, None

        ref_full = _fill_waveform(wf_all[ref], rs, ch_ps, discarded)
        tr = ref_full[local_idx]
        if not np.any(np.isfinite(tr)):
            return None, None

        fs = float(getattr(neurons, 'sampling_rate', None) or 30000.0)
        if not np.isfinite(fs) or fs <= 0:
            fs = 30000.0
        n   = int(tr.shape[0])
        ctr = n // 2
        return (np.arange(n, dtype=float) - ctr) / fs * 1000.0, np.asarray(tr, dtype=float)

    # ----------------------------------------------------------------
    # Main build method
    # ----------------------------------------------------------------

    def build_context(
        self,
        inds,
        segment: int,
        highres: bool,
        render_cfg,
        ccg_data_override,
    ) -> RenderContext:
        """Resolve display state, load and transform data, return a RenderContext.

        Side effects on the UI (both match the old _render_png behaviour):
          - writes _conn_strength_cache when method is conv/tailed/global
          - writes _display_pair_temp for downstream label/value reads
        """
        ui  = self._ui
        ref = int(inds[0])
        tgt = int(inds[1])
        cd  = ccg_data_override if ccg_data_override is not None else ui.ccg_data
        conf = cd.conf

        # ── Segment data: CCG + pval ────────────────────────────────
        d = ui._resolve_segment_data(ref, tgt, segment, highres=highres,
                                     include_pval=True, include_acg=False, _cd=cd)
        ccg_raw      = d['ccg_raw']
        ccg_null_raw = d['ccg_null_raw']
        pval_arg     = d['pval']
        pval_c_arg   = d['pval_corrected']
        seg_label    = d['seg_label']
        if segment == ui.n_segments:
            seg_label = ""

        window_size_eff = float(conf.duration)
        try:
            _n0 = int(len(ccg_raw)) if ccg_raw is not None else 0
            bin_size_eff0 = float(conf.duration) / (_n0 - 1) if _n0 > 1 else float(conf.bin_size)
        except Exception:
            bin_size_eff0 = float(conf.bin_size)

        # ── Extend window ───────────────────────────────────────────
        _extend_on     = bool(self._racg('_extend_enable_var', render_cfg, False))
        _extend_ms     = self._racg('_extend_ms_var',     render_cfg, 0)
        _extend_bin_ms = self._racg('_extend_bin_ms_var', render_cfg, None)
        if _extend_on:
            try:
                bin_ext = (float(_extend_bin_ms) / 1000.0
                           if _extend_bin_ms is not None else float(bin_size_eff0))
                ext = ui._resolve_extended_ccg(
                    ref, tgt, segment, bool(highres), int(_extend_ms), float(bin_ext), cd)
                if ext is not None:
                    ccg_raw         = ext['ccg_raw']
                    ccg_null_raw    = ext['ccg_null_raw']
                    pval_arg        = ext.get('pval')
                    pval_c_arg      = None
                    seg_label       = ext.get('seg_label', seg_label)
                    window_size_eff = float(ext['window_size_s'])
                    bin_size_eff0   = float(ext['bin_size_eff'])
            except Exception as _ext_exc:
                import traceback as _tb
                print(f"[CCGRenderer] extend error: {_ext_exc}")
                _tb.print_exc()

        # ── Norms + alpha ───────────────────────────────────────────
        if render_cfg is not None and NormalizeBy is not None:
            _norms = {n for n in NormalizeBy
                      if n.name in (render_cfg.get('active_norms') or [])}
        else:
            _norms = ui.active_norms
        _alpha = (render_cfg.get('active_alpha', ui.active_alpha)
                  if render_cfg else ui.active_alpha)

        _custom_time_h = None
        if ui._is_custom_segment(segment):
            _ci = ui._custom_seg_index(segment)
            _custom_time_h = ui._custom_segments[_ci].get('total_time_hours')

        norm_info = (', '.join(nm.name for nm in _norms)
                     if _norms and NormalizeBy is not None else None)

        # ── Method + CS show ────────────────────────────────────────
        method  = (render_cfg.get('_conn_str_method_var', 'conv')
                   if render_cfg else
                   ui.center_container.baseline_panel._conn_str_method_var.get())
        cs_show = (render_cfg.get('_conn_str_show_var', False)
                   if render_cfg else
                   ui.center_container.cs_panel._conn_str_show_var.get())

        eff_min_lag, eff_max_lag = ui._effective_lags(ref, tgt)
        _tw_active = self._rsig('test_window', render_cfg)

        # ── ACG data ────────────────────────────────────────────────
        show_acg_ref = self._racg('_acg_ref_var',        render_cfg, False)
        show_acg_tgt = self._racg('_acg_tgt_var',        render_cfg, False)
        _dref        = bool(self._racg('_acg_deconv_ref_var', render_cfg, False))
        _dtgt        = bool(self._racg('_acg_deconv_tgt_var', render_cfg, False))
        _need_acg    = (method == 'tailed') or show_acg_ref or show_acg_tgt or _dref or _dtgt
        acg_ref_raw = acg_tgt_raw = None
        nspks_ref = nspks_tgt = 1.0
        if _need_acg:
            d_acg = ui._resolve_segment_data(ref, tgt, segment,
                                             include_pval=False, include_acg=True, _cd=cd)
            acg_ref_raw = d_acg['acg_ref'].copy().astype(float)
            acg_tgt_raw = d_acg['acg_tgt'].copy().astype(float)
            nspks_ref   = max(float(np.sum(acg_ref_raw)), 1.0)
            nspks_tgt   = max(float(np.sum(acg_tgt_raw)), 1.0)

        # ── Deconvolution ────────────────────────────────────────────
        _deconv_active = _dref or _dtgt
        if _deconv_active and acg_ref_raw is not None and acg_tgt_raw is not None:
            ccg_raw, ccg_null_raw = deconvolve_ccg(
                ccg_raw, ccg_null_raw,
                acg_ref_raw, nspks_ref,
                acg_tgt_raw, nspks_tgt,
                _dref, _dtgt,
            )

        # ── Normalization + CS/baseline computation ──────────────────
        if method in ('conv', 'tailed', 'global'):
            panel_data = compute_ccg_panel_data(
                ccg_raw, ccg_null_raw, conf, method,
                _norms, ref, tgt, segment, ui.n_segments,
                ui._is_custom_segment(segment), _custom_time_h,
                eff_min_lag, eff_max_lag,
                neurons=ui.neurons, nd=ui.cd.nd, nd_key=ui.key.nd(),
                acg_ref=acg_ref_raw if method == 'tailed' else None,
                acg_tgt=acg_tgt_raw if method == 'tailed' else None,
                nspks_ref=nspks_ref, nspks_tgt=nspks_tgt,
            )
            ccg_out     = panel_data.ccg
            show_null   = panel_data.ccg_null
            baseline_1d = panel_data.baseline_1d
            cs_val      = panel_data.cs_val
            if not _deconv_active:
                ui._conn_strength_cache[
                    ui._cs_cache_key(ref, tgt, segment, method, highres,
                                     eff_min_lag, eff_max_lag)
                ] = (cs_val, baseline_1d)
        else:
            ccg_out, show_null = apply_norms_to_ccg(
                ccg_raw, ccg_null_raw, ref, tgt, segment,
                _norms - {NormalizeBy.BASELINE},
                ui.neurons, ui.cd.nd, ui.key.nd(), ui.n_segments,
                ui._is_custom_segment(segment),
                custom_time_hours=_custom_time_h,
            )
            baseline_1d = None
            cs_val      = None

        n_bins       = len(ccg_out)
        bin_size_eff = window_size_eff / (n_bins - 1) if n_bins > 1 else conf.bin_size

        # ── _display_pair_temp side effect ──────────────────────────
        try:
            resk = 'hi' if bool(highres) else 'lo'
            ui._display_pair_temp[(ref, tgt, int(segment), resk)] = {
                'deconv_ref':   bool(_dref),
                'deconv_tgt':   bool(_dtgt),
                'method':       str(method),
                'ccg':          ccg_out,
                'ccg_null':     show_null,
                'baseline_1d':  baseline_1d,
                'cs_val':       cs_val,
                'bin_size_eff': float(bin_size_eff),
                'min_lag': float(eff_min_lag) if eff_min_lag is not None else None,
                'max_lag': float(eff_max_lag) if eff_max_lag is not None else None,
            }
        except Exception:
            pass

        # ── Jitter data ─────────────────────────────────────────────
        resk  = 'hi' if bool(highres) else 'lo'
        _jseg = ui._jitter_seg(segment)
        j_data = ui._jitter_cache.get((ref, tgt, resk, _jseg))
        if j_data is None and _jseg is not None:
            j_data = ui._jitter_cache.get((ref, tgt, resk, None))
        j_ccg_arg    = j_data[0] if j_data is not None else None
        j_pval_arg   = j_data[2] if j_data is not None and len(j_data) > 2 else None
        j_ccg_lo_arg = j_data[3] if j_data is not None and len(j_data) > 3 else None
        j_ccg_hi_arg = j_data[4] if j_data is not None and len(j_data) > 4 else None

        ui._dbg_log(
            "H1",
            "ccg_render_engine.py:build_context:jitter_lookup",
            "Jitter lookup before plot",
            {
                "highres":     bool(highres),
                "method":      str(method),
                "segment":     int(segment),
                "j_key":       [int(ref), int(tgt), "lo", ui._jitter_seg(segment)],
                "len_ccg":     int(len(ccg_out)) if ccg_out is not None else None,
                "len_j_ccg":   int(len(j_ccg_arg))   if j_ccg_arg   is not None else None,
                "len_j_pval":  int(len(j_pval_arg))  if j_pval_arg  is not None else None,
            },
        )

        # ── p-value overlay selection ────────────────────────────────
        show_pval = show_pval_c = None
        show_j_ccg = show_j_pval = None
        show_j_ccg_lo = show_j_ccg_hi = None
        if method == 'conv':
            show_pval   = pval_arg   if self._rsig('conv_p',  render_cfg) else None
            show_pval_c = pval_c_arg if self._rsig('conv_pc', render_cfg) else None
            if self._rsig('conv_p', render_cfg) and pval_arg is None and render_cfg is None:
                is_custom = ui._is_custom_segment(segment)
                is_all    = (segment == ui.n_segments)
                reason    = ('all-segments view' if is_all
                             else 'custom segment' if is_custom else 'cd.pval is None')
                print(f"[CCGReviewUI] p-value ON but unavailable "
                      f"({ref},{tgt}) seg={segment}: {reason}")
        elif method == 'jitter':
            if not highres:
                show_j_ccg    = j_ccg_arg
                show_j_pval   = j_pval_arg if self._rsig('jitter_pc', render_cfg) else None
                show_j_ccg_lo = j_ccg_lo_arg
                show_j_ccg_hi = j_ccg_hi_arg
            ui._dbg_log(
                "H2",
                "ccg_render_engine.py:build_context:jitter_show",
                "Jitter overlay selection",
                {
                    "highres":       bool(highres),
                    "show_j_ccg":    bool(show_j_ccg    is not None),
                    "show_j_pval":   bool(show_j_pval   is not None),
                    "len_ccg":       int(len(ccg_out))    if ccg_out    is not None else None,
                    "len_show_j_ccg": int(len(show_j_ccg)) if show_j_ccg is not None else None,
                },
            )

        # ── CS baseline overlay argument ─────────────────────────────
        if cs_show:
            _has_bl_norm = NormalizeBy.BASELINE in _norms
            if _has_bl_norm and baseline_1d is not None:
                cs_baseline_arg = np.zeros(len(panel_data.ccg))
            else:
                cs_baseline_arg = baseline_1d
        else:
            cs_baseline_arg = None

        # ── ACG overlay arrays ───────────────────────────────────────
        # When extend is active, ACG bin count may differ from extended CCG — suppress to avoid mismatch.
        _acg_bins_ok = (acg_ref_raw is None or len(acg_ref_raw) == n_bins)
        acg_ref_out = acg_ref_raw if show_acg_ref and acg_ref_raw is not None and _acg_bins_ok else None
        acg_tgt_out = acg_tgt_raw if show_acg_tgt and acg_tgt_raw is not None and _acg_bins_ok else None

        # ── Waveform ─────────────────────────────────────────────────
        wf_peak_ms, wf_peak_amp = self._load_waveform(ref, render_cfg, bool(highres))

        # ── Baseline visibility ──────────────────────────────────────
        ccg_null_plot = show_null if bool(self._racg('_baseline_show_var', render_cfg, True)) else None

        # ── lag plot range ───────────────────────────────────────────
        _min_lag_plot = eff_min_lag if (_tw_active or cs_show) else None
        _max_lag_plot = eff_max_lag if (_tw_active or cs_show) else None

        _sh = getattr(ui.neurons, 'shank_ids', None) if ui.neurons is not None else None
        def _shank_label(idx):
            if _sh is not None:
                try:
                    return str(int(_sh[idx]))
                except Exception:
                    pass
            return str(idx)
        ids = (_shank_label(ref), _shank_label(tgt))
        try:
            nt = (ui.neurons.neuron_type[ref], ui.neurons.neuron_type[tgt])
        except Exception:
            nt = None

        # Export/preview style overrides — read from ui._export_overrides if set
        _eo = getattr(ui, '_export_overrides', None) or {}
        _exp_ccg_color        = _eo.get('ccg_color')
        _exp_base_color       = _eo.get('baseline_color')
        _exp_ccg_alpha        = _eo.get('ccg_alpha')
        _exp_base_alpha       = _eo.get('baseline_alpha')
        _exp_cs_shade         = _eo.get('cs_shade_color')
        _exp_show_legend      = bool(_eo.get('show_legend', True))
        _exp_xticks_ms        = _eo.get('xticks_ms')
        _exp_mirror_xticks    = bool(_eo.get('mirror_xticks', True))
        _exp_min_text_size    = _eo.get('min_text_size')
        # test_window visibility: always driven by _tw_active (not export overrides),
        # but color/alpha can be customised for export.
        _exp_test_window_color = _eo.get('test_window_color')
        _exp_test_window_alpha = _eo.get('test_window_alpha')
        _exp_pval_line_color   = _eo.get('pval_line_color')
        _exp_alpha_line_color  = _eo.get('alpha_line_color')
        _print_stg  = bool(_eo.get('print_cs_stg',  False))
        _print_jbsi = bool(_eo.get('print_cs_jbsi', False))
        _title_show_shanks       = bool(_eo.get('title_show_shanks',       True))
        _title_show_inds         = bool(_eo.get('title_show_inds',         True))
        _title_show_type         = bool(_eo.get('title_show_type',         True))
        _title_show_seg          = bool(_eo.get('title_show_seg',          True))
        _title_show_norm_details = bool(_eo.get('title_show_norm_details', True))
        _title_show_session      = bool(_eo.get('title_show_session',      False))
        try:
            _sess_str = str(getattr(ui.key, 'session', ''))
            _title_sess_label = ui._sess_title_label(_sess_str) if _title_show_session else ""
        except Exception:
            _title_sess_label = ""
        try:
            _cs_lines = (ui._cs_annotation_lines(ref, tgt, segment, highres,
                                                  _print_stg, _print_jbsi)
                         if (_print_stg or _print_jbsi) else [])
        except Exception:
            _cs_lines = []

        return RenderContext(
            ccg             = ccg_out,
            ccg_null_plot   = ccg_null_plot,
            pval            = show_pval,
            pval_corrected  = show_pval_c,
            j_ccg           = show_j_ccg,
            j_pval          = show_j_pval,
            j_ccg_lo        = show_j_ccg_lo,
            j_ccg_hi        = show_j_ccg_hi,
            acg_ref         = acg_ref_out,
            acg_tgt         = acg_tgt_out,
            wf_peak_ms      = wf_peak_ms,
            wf_peak_amp     = wf_peak_amp,
            cs_baseline_arg = cs_baseline_arg,
            window_size_eff = window_size_eff,
            bin_size_eff    = bin_size_eff,
            alpha           = _alpha,
            norm_info       = norm_info,
            seg_id_display  = seg_label,
            min_lag_plot    = _min_lag_plot,
            max_lag_plot    = _max_lag_plot,
            extend_on       = _extend_on,
            inds            = tuple(inds),
            ids             = ids,
            neuron_type     = nt,
            is_significant_pair = ui._is_significant(ref, tgt, segment),
            show_ccg        = bool(self._racg('_ccg_show_var',    render_cfg, True)),
            line_ccg        = self._rline('_line_ccg_var',        render_cfg),
            line_baseline   = self._rline('_line_baseline_var',   render_cfg),
            line_ref        = self._rline('_line_ref_var',        render_cfg),
            line_tgt        = self._rline('_line_tgt_var',        render_cfg),
            line_jitter     = self._rline('_line_jitter_var',     render_cfg),
            acg_yscale_ref  = float(self._racg('_acg_yscale_ref_var', render_cfg, 1.0)),
            acg_yscale_tgt  = float(self._racg('_acg_yscale_tgt_var', render_cfg, 1.0)),
            acg_match_ccg   = bool(self._racg('_acg_match_ccg_var',   render_cfg, False)),
            ylim            = ui._get_current_scale_ylim(ref, tgt),
            ccg_color          = _exp_ccg_color,
            baseline_color     = _exp_base_color,
            ccg_alpha          = _exp_ccg_alpha,
            baseline_alpha     = _exp_base_alpha,
            cs_shade_color     = _exp_cs_shade,
            show_legend        = _exp_show_legend,
            xticks_ms          = _exp_xticks_ms,
            mirror_xticks      = _exp_mirror_xticks,
            min_text_size      = _exp_min_text_size,
            show_test_window   = bool(_tw_active),
            test_window_color  = _exp_test_window_color,
            test_window_alpha  = _exp_test_window_alpha,
            pval_line_color    = _exp_pval_line_color,
            alpha_line_color   = _exp_alpha_line_color,
            cs_annotation_lines      = _cs_lines,
            title_show_shanks        = _title_show_shanks,
            title_show_inds          = _title_show_inds,
            title_show_type          = _title_show_type,
            title_show_seg           = _title_show_seg,
            title_show_norm_details  = _title_show_norm_details,
            title_show_session       = _title_show_session,
            title_session_label      = _title_sess_label,
            dark_mode                = getattr(ui, '_dark', False),
        )

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------

    def write_png(self, ctx: RenderContext, png_path: str, dpi: int = 100) -> None:
        """Create figure, call plot_ccg_panel, apply post-processing, save PNG."""
        fig = Figure(figsize=(7, 5))
        ax  = fig.add_subplot(111)

        plot_ccg.plot_ccg_panel(
            ax               = ax,
            ccg              = ctx.ccg,
            ids              = ctx.ids,
            inds             = ctx.inds,
            neuron_type      = ctx.neuron_type,
            window_size      = ctx.window_size_eff,
            bin_size         = ctx.bin_size_eff,
            pval             = ctx.pval,
            pval_corrected   = ctx.pval_corrected,
            alpha            = ctx.alpha,
            ccg_null         = ctx.ccg_null_plot,
            j_ccg            = ctx.j_ccg,
            j_pval           = ctx.j_pval,
            segment_id       = ctx.seg_id_display,
            is_significant_pair = ctx.is_significant_pair,
            min_lag          = ctx.min_lag_plot,
            max_lag          = ctx.max_lag_plot,
            normalize_info   = ctx.norm_info,
            acg_ref          = ctx.acg_ref,
            acg_tgt          = ctx.acg_tgt,
            acg_yscale_ref   = ctx.acg_yscale_ref,
            acg_yscale_tgt   = ctx.acg_yscale_tgt,
            acg_match_ccg    = ctx.acg_match_ccg,
            show_ccg         = ctx.show_ccg,
            line_ccg         = ctx.line_ccg,
            line_baseline    = ctx.line_baseline,
            line_ref         = ctx.line_ref,
            line_tgt         = ctx.line_tgt,
            line_jitter      = ctx.line_jitter,
            conn_strength_baseline = ctx.cs_baseline_arg,
            ccg_color          = ctx.ccg_color,
            baseline_color     = ctx.baseline_color,
            ccg_alpha          = ctx.ccg_alpha,
            baseline_alpha     = ctx.baseline_alpha,
            cs_shade_color     = ctx.cs_shade_color,
            show_legend        = ctx.show_legend,
            show_test_window   = ctx.show_test_window,
            test_window_color  = ctx.test_window_color,
            test_window_alpha  = ctx.test_window_alpha,
            pval_line_color    = ctx.pval_line_color,
            alpha_line_color   = ctx.alpha_line_color,
            title_show_shanks       = ctx.title_show_shanks,
            title_show_inds         = ctx.title_show_inds,
            title_show_type         = ctx.title_show_type,
            title_show_seg          = ctx.title_show_seg,
            title_show_norm_details = ctx.title_show_norm_details,
            title_show_session      = ctx.title_show_session,
            title_session_label     = ctx.title_session_label,
        )

        # Extend view: readable x-ticks on long windows
        if ctx.extend_on:
            try:
                half_ms = float(ctx.window_size_eff) * 1000.0 / 2.0
                nice    = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
                step    = next((s for s in nice if (2 * half_ms / s) <= 10.5), nice[-1])
                start   = -np.floor(half_ms / step) * step
                ticks   = np.arange(start, half_ms + 0.5 * step, step)
                if not np.any(np.isclose(ticks, 0.0)):
                    ticks = np.sort(np.append(ticks, 0.0))
                ax.set_xticks(ticks)
            except Exception:
                pass

        # Jitter 95% confidence band
        if ctx.j_ccg_lo is not None and ctx.j_ccg_hi is not None:
            try:
                jlo = np.asarray(ctx.j_ccg_lo, dtype=float)
                jhi = np.asarray(ctx.j_ccg_hi, dtype=float)
                if len(jlo) == len(ctx.ccg) and len(jhi) == len(ctx.ccg):
                    bs     = ctx.bin_size_eff
                    ws     = ctx.window_size_eff
                    bins_s = np.arange(-ws / 2, ws / 2 + bs, bs)
                    bins   = bins_s * 1000.0
                    edges  = np.append(bins - bs * 500.0, bins[-1] + bs * 500.0)
                    x_step = np.repeat(edges, 2)[1:-1]
                    for arr in (jlo, jhi):
                        ax.plot(x_step, np.repeat(arr, 2),
                                color='#C62828', linewidth=1.15,
                                alpha=0.9, linestyle=(0, (4, 3)), zorder=4)
            except Exception:
                pass

        if ctx.ylim is not None:
            ax.set_ylim(ctx.ylim)

        # Export xtick override
        if ctx.xticks_ms:
            try:
                ticks = list(ctx.xticks_ms)
                if ctx.mirror_xticks:
                    ticks = sorted(set(ticks + [-t for t in ticks]))
                ax.set_xticks(ticks)
            except Exception:
                pass

        # CS annotation lines below x-axis label
        if ctx.cs_annotation_lines:
            try:
                cur_xlabel = ax.get_xlabel() or ''
                ax.set_xlabel(cur_xlabel + '\n' + '\n'.join(ctx.cs_annotation_lines))
            except Exception:
                pass

        # Export min text size
        if ctx.min_text_size is not None:
            try:
                ms = float(ctx.min_text_size)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                             + ax.get_xticklabels() + ax.get_yticklabels()):
                    try:
                        if item.get_fontsize() < ms:
                            item.set_fontsize(ms)
                    except Exception:
                        pass
            except Exception:
                pass

        if ctx.dark_mode:
            _bg = '#2b2b2b'
            _fg = 'white'
            _sp = '#666666'
            fig.set_facecolor(_bg)
            ax.set_facecolor(_bg)
            ax.tick_params(colors=_fg)
            ax.xaxis.label.set_color(_fg)
            ax.yaxis.label.set_color(_fg)
            ax.title.set_color(_fg)
            for sp in ax.spines.values():
                sp.set_edgecolor(_sp)

        fig.savefig(png_path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        matplotlib.pyplot.close(fig)
