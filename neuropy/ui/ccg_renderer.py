"""CCGContextBuilder — data prep for CCG PNGs (UI layer).

Pure rendering (RenderContext, render_ccg_png) lives in neuropy.plotting.ccg.
Pure computation (deconvolve_ccg) lives in neuropy.analyses.ms_connectivity.
"""
from __future__ import annotations

import traceback

import numpy as np

from neuropy.analyses.ms_connectivity import (
    NormalizeBy, apply_norms_to_ccg, compute_ccg_panel_data,
    deconvolve_ccg,
)
from neuropy.plotting.ccg import (
    JitterOverlay, PlotStyle, TitleConfig, RenderContext,
    _fill_waveform, render_ccg_png,
)
import neuropy.plotting.ccg as plot_ccg


# ---------------------------------------------------------------------------
# CCGContextBuilder (formerly CCGRenderEngine)
# ---------------------------------------------------------------------------

class CCGContextBuilder:
    """Resolves UI display state, loads and transforms data, returns a RenderContext.

    Holds a back-reference to CCGReviewUI; all methods read from self._ui.
    Pure rendering is delegated to render_ccg_png() in neuropy.plotting.ccg.
    """

    def __init__(self, ui):
        self._ui = ui

    # ----------------------------------------------------------------
    # Display-state resolution helpers
    # ----------------------------------------------------------------

    def _rsig(self, name: str, render_cfg) -> bool:
        _map = {
            'conv_p':      '_sig_conv_p',
            'conv_pc':     '_sig_conv_pc',
            'test_window': '_sig_test_window',
            'jitter_pc':   '_sig_jitter_pc',
        }
        if render_cfg:
            return bool(render_cfg.get(_map[name], False))
        return self._ui._sig(name)

    def _rline(self, attr: str, render_cfg) -> bool:
        if render_cfg:
            return bool(render_cfg.get(attr, False))
        v = getattr(self._ui.center_container.correlogram_panel, attr, None)
        return bool(v.get()) if v is not None else False

    def _racg(self, attr: str, render_cfg, default=None):
        if render_cfg:
            return render_cfg.get(attr, default)
        return self._ui._acg_var_get(attr, default)

    # ----------------------------------------------------------------
    # Waveform loading
    # ----------------------------------------------------------------

    def _load_waveform(self, ref: int, render_cfg, highres: bool):
        if not highres or not self._racg('_peak_wf', render_cfg, False):
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

        if discarded is not None and discarded.size and np.isin(peak_ch, discarded):
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
        """Resolve display state, load and transform data, return a RenderContext."""
        ui  = self._ui
        ref = int(inds[0])
        tgt = int(inds[1])
        cd  = ccg_data_override if ccg_data_override is not None else ui.ccg_data
        conf = cd.conf

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
        _n0 = len(ccg_raw) if ccg_raw is not None else 0
        bin_size_eff0 = conf.duration / (_n0 - 1) if _n0 > 1 else conf.bin_size

        _extend_on     = bool(self._racg('_extend_enable', render_cfg, False))
        _extend_ms     = self._racg('_extend_ms',     render_cfg, 0)
        _extend_bin_ms = self._racg('_extend_bin_ms', render_cfg, None)
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
                print(f"[CCGRenderer] extend error: {_ext_exc}")
                traceback.print_exc()

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

        method  = (render_cfg.get('_conn_str_method', 'conv')
                   if render_cfg else
                   ui.center_container.baseline_panel._conn_str_method.get())
        cs_show = (render_cfg.get('_conn_str_show', False)
                   if render_cfg else
                   ui.center_container.cs_panel._conn_str_show.get())

        eff_min_lag, eff_max_lag = ui._effective_lags(ref, tgt)
        _tw_active = self._rsig('test_window', render_cfg)

        show_acg_ref = self._racg('_acg_ref',        render_cfg, False)
        show_acg_tgt = self._racg('_acg_tgt',        render_cfg, False)
        _dref        = bool(self._racg('_acg_deconv_ref', render_cfg, False))
        _dtgt        = bool(self._racg('_acg_deconv_tgt', render_cfg, False))
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

        _deconv_active = _dref or _dtgt
        if _deconv_active and acg_ref_raw is not None and acg_tgt_raw is not None:
            ccg_raw, ccg_null_raw = deconvolve_ccg(
                ccg_raw, ccg_null_raw,
                acg_ref_raw, nspks_ref,
                acg_tgt_raw, nspks_tgt,
                _dref, _dtgt,
            )

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
            "H1", "ccg_renderer.py:build_context:jitter_lookup", "Jitter lookup before plot",
            {
                "highres": bool(highres), "method": str(method), "segment": int(segment),
                "j_key": [int(ref), int(tgt), "lo", ui._jitter_seg(segment)],
                "len_ccg": int(len(ccg_out)) if ccg_out is not None else None,
                "len_j_ccg": int(len(j_ccg_arg)) if j_ccg_arg is not None else None,
                "len_j_pval": int(len(j_pval_arg)) if j_pval_arg is not None else None,
            },
        )

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
                _j_len = len(j_ccg_arg) if j_ccg_arg is not None else None
                _bins_match = (_j_len == n_bins)
                show_j_ccg    = j_ccg_arg    if _bins_match else None
                show_j_pval   = (j_pval_arg  if self._rsig('jitter_pc', render_cfg) else None) if _bins_match else None
                show_j_ccg_lo = j_ccg_lo_arg if _bins_match else None
                show_j_ccg_hi = j_ccg_hi_arg if _bins_match else None
                if not _bins_match and j_ccg_arg is not None:
                    print(f"[jitter] bin mismatch ({_j_len} vs {n_bins}) — overlay suppressed")
            ui._dbg_log(
                "H2", "ccg_renderer.py:build_context:jitter_show", "Jitter overlay selection",
                {
                    "highres": bool(highres),
                    "show_j_ccg": bool(show_j_ccg is not None),
                    "show_j_pval": bool(show_j_pval is not None),
                    "len_ccg": int(len(ccg_out)) if ccg_out is not None else None,
                    "len_show_j_ccg": int(len(show_j_ccg)) if show_j_ccg is not None else None,
                },
            )

        if cs_show:
            _has_bl_norm = NormalizeBy.BASELINE in _norms
            if _has_bl_norm and baseline_1d is not None:
                cs_baseline_arg = np.zeros(len(panel_data.ccg))
            else:
                cs_baseline_arg = baseline_1d
        else:
            cs_baseline_arg = None

        _acg_bins_ok = (acg_ref_raw is None or len(acg_ref_raw) == n_bins)
        acg_ref_out = acg_ref_raw if show_acg_ref and acg_ref_raw is not None and _acg_bins_ok else None
        acg_tgt_out = acg_tgt_raw if show_acg_tgt and acg_tgt_raw is not None and _acg_bins_ok else None

        wf_peak_ms, wf_peak_amp = self._load_waveform(ref, render_cfg, bool(highres))

        ccg_null_plot = show_null if bool(self._racg('_baseline_show', render_cfg, True)) else None

        _min_lag_plot = eff_min_lag if (_tw_active or cs_show) else None
        _max_lag_plot = eff_max_lag if (_tw_active or cs_show) else None

        _neurons_obj = ui.neurons
        if _neurons_obj is None:
            try:
                _neurons_obj = ui.cd.nd.data[ui.key.nd()]
            except Exception:
                pass
        _sh = getattr(_neurons_obj, 'shank_ids', None) if _neurons_obj is not None else None
        def _sh_label(idx):
            try: return str(int(_sh[idx]))
            except Exception: return str(idx)
        shank_ids = tuple(_sh_label(i) for i in (ref, tgt)) if _sh is not None else (str(ref), str(tgt))
        try:
            nt = (_neurons_obj.neuron_type[ref], _neurons_obj.neuron_type[tgt])
        except Exception:
            nt = None

        _eo = getattr(ui, '_export_overrides', None) or {}
        print_stg  = bool(_eo.get('print_cs_stg',  False))
        print_jbsi = bool(_eo.get('print_cs_jbsi', False))
        title_show_session = bool(_eo.get('title_show_session', False))
        try:
            _title_sess_label = (ui._sess_title_label(str(getattr(ui.key, 'session', '')))
                                 if title_show_session else "")
        except Exception:
            _title_sess_label = ""

        try:
            _cs_lines = (ui._cs_annotation_lines(ref, tgt, segment, highres, print_stg, print_jbsi)
                         if (print_stg or print_jbsi) else [])
        except Exception:
            _cs_lines = []

        return RenderContext(
            ccg             = ccg_out,
            ccg_null_plot   = ccg_null_plot,
            pval            = show_pval,
            pval_corrected  = show_pval_c,
            jitter          = JitterOverlay(
                j_ccg=show_j_ccg, j_pval=show_j_pval,
                j_ccg_lo=show_j_ccg_lo, j_ccg_hi=show_j_ccg_hi),
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
            cs_annotation_lines = _cs_lines,
            inds            = tuple(inds),
            shank_ids       = shank_ids,
            neuron_type     = nt,
            is_significant_pair = ui._is_significant(ref, tgt, segment),
            show_ccg        = bool(self._racg('_ccg_show',    render_cfg, True)),
            line_ccg        = self._rline('_line_ccg',        render_cfg),
            line_baseline   = self._rline('_line_baseline',   render_cfg),
            line_ref        = self._rline('_line_ref',        render_cfg),
            line_tgt        = self._rline('_line_tgt',        render_cfg),
            line_jitter     = self._rline('_line_jitter',     render_cfg),
            acg_yscale_ref  = float(self._racg('_acg_yscale_ref', render_cfg, 1.0)),
            acg_yscale_tgt  = float(self._racg('_acg_yscale_tgt', render_cfg, 1.0)),
            acg_match_ccg   = bool(self._racg('_acg_match_ccg',   render_cfg, False)),
            ylim            = ui._get_current_scale_ylim(ref, tgt),
            style           = PlotStyle(
                ccg_color          = _eo.get('ccg_color'),
                baseline_color     = _eo.get('baseline_color'),
                ccg_alpha          = _eo.get('ccg_alpha'),
                baseline_alpha     = _eo.get('baseline_alpha'),
                cs_shade_color     = _eo.get('cs_shade_color'),
                show_legend        = bool(_eo.get('show_legend', True)),
                xticks_ms          = _eo.get('xticks_ms'),
                mirror_xticks      = bool(_eo.get('mirror_xticks', True)),
                min_text_size      = _eo.get('min_text_size'),
                show_test_window   = bool(_tw_active),
                test_window_color  = _eo.get('test_window_color'),
                test_window_alpha  = _eo.get('test_window_alpha'),
                pval_line_color    = _eo.get('pval_line_color'),
                alpha_line_color   = _eo.get('alpha_line_color')),
            title           = TitleConfig(
                title_show_shanks        = bool(_eo.get('title_show_shanks',       True)),
                title_show_inds          = bool(_eo.get('title_show_inds',         True)),
                title_show_type          = bool(_eo.get('title_show_type',         True)),
                title_show_seg           = bool(_eo.get('title_show_seg',          True)),
                title_show_norm_details  = bool(_eo.get('title_show_norm_details', True)),
                title_show_session       = title_show_session,
                title_session_label      = _title_sess_label),
            dark_mode       = ui.theme.dark,
        )

    def write_png(self, ctx: RenderContext, png_path: str, dpi: int = 100) -> None:
        render_ccg_png(ctx, png_path, dpi)


# Backward-compat alias
CCGRenderEngine = CCGContextBuilder
