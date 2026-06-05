"""CCG main view — UI panel classes for the center panel of CCGReviewUI."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neuropy.analyses.spike_attribution import find_spike_pairs
from neuropy.plotting.ccg import plot_spike_attribution_raster
from neuropy.ui.utils import ArrowScroller, WrapFrame

if TYPE_CHECKING:
    from neuropy.analyses.ms_connectivity import CCGKey
    from neuropy.ui.ccg_ui import CCGReviewUI


def _collapsible_section(parent: tk.Widget, title: str, expanded: bool = True):
    """Separator-line collapsible section.  Returns (inner_frame, fold_var)."""
    outer = ttk.Frame(parent)
    outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(3, 0))

    hdr = ttk.Frame(outer)
    hdr.pack(fill=tk.X, pady=(2, 0))
    hdr.columnconfigure(2, weight=1)

    fold_var = tk.BooleanVar(value=expanded)
    tri = ttk.Label(hdr, text='▾' if expanded else '▸',
                    cursor='hand2', font=('Arial', 10))
    tri.grid(row=0, column=0, padx=(4, 0))
    ttk.Label(hdr, text=title, font=('Arial', 9, 'bold')).grid(
        row=0, column=1, padx=(3, 6))
    ttk.Separator(hdr, orient='horizontal').grid(
        row=0, column=2, sticky='ew', padx=(0, 4), pady=6)

    inner = ttk.Frame(outer, padding=(8, 2, 4, 2))
    if expanded:
        inner.pack(fill=tk.X)

    def _toggle(e=None):
        v = not fold_var.get()
        fold_var.set(v)
        (inner.pack(fill=tk.X) if v else inner.pack_forget())
        tri.config(text='▾' if v else '▸')

    tri.bind('<Button-1>', _toggle)
    hdr.bind('<Button-1>', _toggle)
    return inner, fold_var


class CCGPlotPanel:
    """Main CCG figure + canvas + horizontal split pane for waveforms."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._build(parent)

    def _build(self, parent: tk.Widget):
        self._plot_pw = tk.PanedWindow(parent, orient=tk.HORIZONTAL,
                                       sashrelief=tk.RAISED, sashwidth=4,
                                       bg='#CCCCCC')
        self._plot_pw.pack(fill=tk.BOTH, expand=True)
        ccg_inner = ttk.Frame(self._plot_pw)
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=ccg_inner)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().bind('<Button-2>', self._ui._ccg_context_menu)
        self.canvas.get_tk_widget().bind('<Button-3>', self._ui._ccg_context_menu)
        self._plot_pw.add(ccg_inner, stretch='always')


class CorrelogramPanel:
    """Correlograms sub-section: tri-state style buttons, ACG sliders, extend window.

    Owns all style BooleanVars.  Bridge aliases on CCGReviewUI keep backward-compat
    with _CACHE_CONFIG_ATTRS and rendering code.
    """

    _STYLE_VARS = {
        'ccg':      ('_line_ccg',      '_ccg_show'),
        'baseline': ('_line_baseline', '_baseline_show'),
        'ref':      ('_line_ref',      '_acg_ref'),
        'tgt':      ('_line_tgt',      '_acg_tgt'),
        'peak_wf':  ('_line_peak_wf',  '_peak_wf'),
    }

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        # CCG / Baseline style
        self._line_ccg      = tk.BooleanVar(value=False)
        self._ccg_show      = tk.BooleanVar(value=True)
        self._line_baseline = tk.BooleanVar(value=False)
        self._baseline_show = tk.BooleanVar(value=True)
        # ACG style (default: hidden)
        self._line_ref  = tk.BooleanVar(value=True)
        self._acg_ref   = tk.BooleanVar(value=False)
        self._line_tgt  = tk.BooleanVar(value=True)
        self._acg_tgt   = tk.BooleanVar(value=False)
        # Peak waveform style
        self._line_peak_wf = tk.BooleanVar(value=True)
        self._peak_wf      = tk.BooleanVar(value=False)
        # Jitter overlay style
        self._line_jitter  = tk.BooleanVar(value=False)
        # ACG deconvolution
        self._acg_deconv_ref = tk.BooleanVar(value=False)
        self._acg_deconv_tgt = tk.BooleanVar(value=False)
        # ACG Y-scale
        self._acg_yscale_ref = tk.DoubleVar(value=1.0)
        self._acg_yscale_tgt = tk.DoubleVar(value=1.0)
        self._acg_match_ccg  = tk.BooleanVar(value=False)
        # Extend window
        self._extend_enable  = tk.BooleanVar(value=False)
        self._extend_ms      = tk.IntVar(value=50)
        self._extend_bin_ms  = tk.DoubleVar(value=1.0)
        self._build(parent)

    def _get_style_vars(self, item: str):
        la, sa = self._STYLE_VARS[item]
        return getattr(self, la), getattr(self, sa)

    def _build(self, parent: tk.Widget):
        acg_frame, self._acg_fold = _collapsible_section(parent, "Correlograms")
        self._acg_inner_frame = acg_frame
        self._build_style_row(acg_frame)
        self._build_deconv_row(acg_frame)
        self._build_extend_row(parent)
        self._update_style_btns()
        self._update_acg_deconv_btns()

    def _make_style_btn(self, parent, text: str, cycle_fn) -> tk.Label:
        """Raised Label with click binding for tri-state style cycling."""
        btn = tk.Label(parent, text=text, font=('TkDefaultFont', 9),
                       relief='raised', bd=1, padx=2, cursor='hand2')
        btn.pack(side=tk.LEFT, padx=2)
        btn.bind('<Button-1>', lambda e: cycle_fn())
        return btn

    def _make_acg_yscale(self, parent, label: str, var, side: str) -> ttk.Entry:
        """Label + Scale + Entry for one ACG y-scale control. Returns the Entry."""
        ttk.Label(parent, text=label, font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 1))
        ttk.Scale(parent, from_=0.1, to=1.5, variable=var,
                  orient=tk.HORIZONTAL, length=50,
                  command=lambda v: self._on_acg_scale_change()).pack(side=tk.LEFT, padx=1)
        entry = ttk.Entry(parent, width=5, font=('Courier', 8))
        entry.insert(0, "1.0")
        entry.pack(side=tk.LEFT, padx=(0, 4))
        entry.bind('<Return>',   lambda e: self._on_acg_entry_submit(side))
        entry.bind('<FocusOut>', lambda e: self._on_acg_entry_submit(side))
        return entry

    def _build_style_row(self, parent: tk.Widget) -> None:
        """Row 1: CCG/baseline/ACG style buttons + y-scale controls."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 1))
        self._ccg_style_btn      = self._make_style_btn(row, "■ CCG",              lambda: self._cycle_style('ccg'))
        self._baseline_style_btn = self._make_style_btn(row, "■ baseline",         lambda: self._cycle_style('baseline'))
        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(row, text="ACG display", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 4))
        self._ref_style_btn      = self._make_style_btn(row, "□ ref",              lambda: self._cycle_style_acg('ref'))
        self._tgt_style_btn      = self._make_style_btn(row, "□ tgt",              lambda: self._cycle_style_acg('tgt'))
        self._peak_wf_style_btn  = self._make_style_btn(row, "X waveform overlay", lambda: self._cycle_style_acg('peak_wf'))
        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._acg_scale_ref_entry = self._make_acg_yscale(row, "ref Y:", self._acg_yscale_ref, 'ref')
        self._acg_scale_tgt_entry = self._make_acg_yscale(row, "tgt Y:", self._acg_yscale_tgt, 'tgt')
        ttk.Checkbutton(row, text="Match ACG to CCG scale",
                        variable=self._acg_match_ccg,
                        command=self._ui._on_sig_toggle).pack(side=tk.LEFT, padx=4)

    def _build_deconv_row(self, parent: tk.Widget) -> None:
        """Row 2: ACG deconvolution buttons (ref + tgt)."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(1, 0))
        ttk.Label(row, text="Deconvolve ACG from CCG",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 4))
        self._deconv_ref_btn = self._make_style_btn(row, "□ ref", lambda: self._toggle_acg_deconv('ref'))
        self._deconv_tgt_btn = self._make_style_btn(row, "□ tgt", lambda: self._toggle_acg_deconv('tgt'))

    def _build_extend_row(self, parent: tk.Widget) -> None:
        """Bottom: extend-window enable toggle + ms spinbox + resolution spinbox."""
        row = ttk.Frame(parent)
        row.pack(side=tk.BOTTOM, fill=tk.X, anchor='w', pady=(2, 0))
        ttk.Checkbutton(row, text="Extend", variable=self._extend_enable,
                        command=self._on_extend_toggle).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(row, text="ms:").pack(side=tk.LEFT)
        self._extend_ms_spin = ttk.Spinbox(row, from_=1, to=5000, increment=1, width=6,
                                           textvariable=self._extend_ms,
                                           command=self._on_extend_ms_commit)
        self._extend_ms_spin.pack(side=tk.LEFT, padx=(2, 6))
        self._extend_ms_spin.bind('<Return>',   lambda e: self._on_extend_ms_commit())
        self._extend_ms_spin.bind('<FocusOut>', lambda e: self._on_extend_ms_commit())
        ttk.Label(row, text="resolution (ms):").pack(side=tk.LEFT, padx=(8, 2))
        self._extend_bin_spin = ttk.Spinbox(row, from_=0.0, to=100.0, increment=0.1, width=6,
                                            format='%.3f', textvariable=self._extend_bin_ms,
                                            command=self._on_extend_ms_commit)
        self._extend_bin_spin.pack(side=tk.LEFT, padx=(2, 6))
        self._extend_bin_spin.bind('<Return>',   lambda e: self._on_extend_ms_commit())
        self._extend_bin_spin.bind('<FocusOut>', lambda e: self._on_extend_ms_commit())

    # ── Style cycle methods ────────────────────────────────────────────

    def _cycle_style(self, item: str):
        """Tri-state cycle: ■ solid → □ outline → X hidden → ■ solid."""
        line_var, show_var = self._get_style_vars(item)
        if show_var.get() and not line_var.get():
            line_var.set(True)
        elif show_var.get() and line_var.get():
            show_var.set(False); line_var.set(False)
        else:
            show_var.set(True); line_var.set(False)
        self._update_style_btns()
        self._ui._refresh()

    def _cycle_style_acg(self, item: str):
        """Tri-state cycle: X hidden → □ outline → ■ solid → X hidden."""
        line_var, show_var = self._get_style_vars(item)
        if not show_var.get():
            show_var.set(True); line_var.set(True)
        elif line_var.get():
            line_var.set(False)
        else:
            show_var.set(False); line_var.set(False)
        self._update_style_btns()
        self._ui._refresh()

    def _update_style_btns(self):
        """Refresh tri-state button labels: ■ / □ / X."""
        for item, btn_attr, name in [
            ('ccg',     '_ccg_style_btn',     'CCG'),
            ('baseline','_baseline_style_btn', 'baseline'),
            ('ref',     '_ref_style_btn',      'ref'),
            ('tgt',     '_tgt_style_btn',      'tgt'),
            ('peak_wf', '_peak_wf_style_btn',  'ref peak'),
        ]:
            btn = getattr(self, btn_attr, None)
            if not btn:
                continue
            line_var, show_var = self._get_style_vars(item)
            if not show_var.get():
                btn.config(text=f"X {name}")
            elif line_var.get():
                btn.config(text=f"□ {name}")
            else:
                btn.config(text=f"■ {name}")

    def _toggle_acg_deconv(self, which: str):
        var = self._acg_deconv_ref if which == 'ref' else self._acg_deconv_tgt
        var.set(not var.get())
        self._update_acg_deconv_btns()
        self._ui._refresh()

    def _update_acg_deconv_btns(self):
        cur_ref = bool(self._acg_deconv_ref.get())
        cur_tgt = bool(self._acg_deconv_tgt.get())
        btn = getattr(self, '_deconv_ref_btn', None)
        if btn is not None:
            btn.config(text=('■ ' if cur_ref else '□ ') + 'ref')
        btn = getattr(self, '_deconv_tgt_btn', None)
        if btn is not None:
            btn.config(text=('■ ' if cur_tgt else '□ ') + 'tgt')

    def _on_acg_scale_change(self):
        ref_val = self._acg_yscale_ref.get()
        tgt_val = self._acg_yscale_tgt.get()
        self._acg_scale_ref_entry.delete(0, tk.END)
        self._acg_scale_ref_entry.insert(0, f"{ref_val:.1f}")
        self._acg_scale_tgt_entry.delete(0, tk.END)
        self._acg_scale_tgt_entry.insert(0, f"{tgt_val:.1f}")
        if self._acg_ref.get() or self._acg_tgt.get() or self._peak_wf.get():
            self._ui._refresh()

    def _on_acg_entry_submit(self, which: str):
        if which == 'ref':
            entry, var = self._acg_scale_ref_entry, self._acg_yscale_ref
        else:
            entry, var = self._acg_scale_tgt_entry, self._acg_yscale_tgt
        try:
            val = max(0.01, float(entry.get()))
            var.set(val)
            entry.delete(0, tk.END)
            entry.insert(0, f"{val:.1f}")
            if self._acg_ref.get() or self._acg_tgt.get() or self._peak_wf.get():
                self._ui._refresh()
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, f"{var.get():.1f}")

    def _clear_extend_cache(self):
        try:
            self._ui._extend_cache.clear()
        except Exception:
            pass

    def _on_extend_toggle(self):
        self._clear_extend_cache()
        self._ui._refresh()

    def _on_extend_ms_commit(self):
        ms = max(1, min(5000, self._extend_ms.get()))
        self._extend_ms.set(ms)
        try:
            raw_bms = float(str(self._extend_bin_ms.get()))
            neurons = getattr(self._ui, 'neurons', None)
            fs = float(getattr(neurons, 'sampling_rate', None) or 30000.0)
            min_bms = 1000.0 / fs
            bms = max(min_bms, min(100.0, raw_bms))
            self._extend_bin_ms.set(round(bms, 4))
            if raw_bms < min_bms:
                from tkinter import messagebox
                messagebox.showerror(
                    "Resolution too small",
                    f"Minimum resolution is {min_bms:.4f} ms (1 sample at {fs:.0f} Hz).\n"
                    f"Value set to {round(bms, 4)} ms."
                )
        except Exception:
            pass
        if self._extend_enable.get():
            self._clear_extend_cache()
            self._ui._refresh()


class CSPanel:
    """Connection strength overlay: show toggle, metric selector, CS label, non-negative."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._conn_str_show   = tk.BooleanVar(value=False)
        self._conn_str_nonneg = tk.BooleanVar(value=False)
        self._conn_str_metric = tk.StringVar(
            value=ui._ui_state_cache.get('conn_str_metric', 'STG'))
        self._cs_metric_rbs: dict = {}
        self._build(parent)

    def _build(self, parent: tk.Widget):
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, anchor='w')
        ttk.Checkbutton(row1, text="Show CS overlay",
                        variable=self._conn_str_show,
                        command=lambda: (self._ui._cs_mgr._on_conn_str_toggle(),
                                        self._ui._build_sig_chips())).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Label(row1, text="Measure:").pack(side=tk.LEFT, padx=(8, 2))
        for val in ("STG", "JBSI"):
            rb = ttk.Radiobutton(
                row1, text=val, value=val, variable=self._conn_str_metric,
                command=lambda: (self._ui._refresh(),
                                 self._ui._cs_mgr._update_conn_str_label())
            )
            rb.pack(side=tk.LEFT, padx=(0, 6))
            self._cs_metric_rbs[val] = rb
        self._ui._cs_mgr._update_conn_str_metric_availability()

        row_cs = ttk.Frame(parent)
        row_cs.pack(fill=tk.X, anchor='w', pady=(2, 0))
        self._conn_str_label = ttk.Label(row_cs, text="CS: —")
        self._conn_str_label.pack(side=tk.LEFT)
        ttk.Checkbutton(
            row_cs,
            text="non-negative",
            variable=self._conn_str_nonneg,
            command=lambda: (self._ui._refresh(),
                             self._ui._cs_mgr._update_conn_str_label(),
                             getattr(self._ui, '_stats_panel', None)
                             and self._ui._stats_panel.on_parent_display_option_changed()),
        ).pack(side=tk.LEFT, padx=(10, 0))

class BaselinePanel:
    """Baseline method radio buttons, test window, adaptive TW, p-value row."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._conn_str_method = tk.StringVar(
            value=ui._ui_state_cache.get('conn_str_method', 'conv'))
        self._sig_test_window = tk.BooleanVar(value=True)
        self._adaptive_tw     = tk.BooleanVar(value=False)
        self._sig_conv_p      = tk.BooleanVar(value=False)
        self._sig_conv_pc     = tk.BooleanVar(value=False)
        self._sig_jitter_pc   = tk.BooleanVar(value=False)
        self._global_rb           = None
        self._jitter_rb           = None
        self._adaptive_tw_btn     = None
        self._build(parent)

    def _build(self, parent: tk.Widget):
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, anchor='w', pady=(2, 0))
        ttk.Label(row2, text="Baseline:").pack(side=tk.LEFT)
        for val, lbl in [('conv', 'Conv'), ('tailed', 'Tailed'),
                         ('global', 'Global'), ('jitter', 'Jitter')]:
            rb = ttk.Radiobutton(row2, text=lbl, variable=self._conn_str_method,
                                 value=val, command=self._ui._on_baseline_method_change)
            rb.pack(side=tk.LEFT, padx=3)
            if val == 'global':
                self._global_rb = rb
                rb.state(['disabled'])
            if val == 'jitter':
                self._jitter_rb = rb
                rb.state(['disabled'])
        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Checkbutton(row2, text="test window",
                        variable=self._sig_test_window,
                        command=self._ui._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._adaptive_tw_btn = ttk.Checkbutton(
            row2, text="adaptive test window",
            variable=self._adaptive_tw,
            command=self._ui._on_adaptive_tw_toggle)
        self._adaptive_tw_btn.pack(side=tk.LEFT, padx=2)
        self._adaptive_tw_btn.state(['disabled'])

        self._cs_pval_row = ttk.Frame(parent)
        self._cs_pval_row.pack(fill=tk.X, anchor='w', pady=(2, 0))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(6, 2))
        wf_row = ttk.Frame(parent)
        wf_row.pack(fill=tk.X, anchor='w')
        ttk.Checkbutton(wf_row, text="Waveform",
                        variable=self._ui._panel_vars['Waveforms'],
                        command=self._ui._toggle_waveforms_panel).pack(side=tk.LEFT)


class NormPanel:
    """Normalization checkboxes, same-scale toggles, and Normalize All button."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._pair_scale = tk.BooleanVar(
            value=ui._ui_state_cache.get('pair_scale', False))
        self._sess_scale = tk.BooleanVar(
            value=ui._ui_state_cache.get('sess_scale', False))
        self._norm_checkbuttons: list = []
        self._build(parent)

    def _build(self, parent: tk.Widget):
        from neuropy.ui.ccg_norms import NormalizeBy, NormBackend
        norm_inner, self._norm_fold = _collapsible_section(parent, "Normalization")

        saved_norms = self._ui._ui_state_cache.get('active_norms', [])
        options = [
            (NormalizeBy.REF_FRATE,    "Ref f-rate"),
            (NormalizeBy.TARGET_FRATE, "Tgt f-rate"),
            (NormalizeBy.TIME_SPAN,    "Time (hr)"),
            (NormalizeBy.TIME_SECOND,  "Time (sec)"),
            (NormalizeBy.TOTAL_AREA,   "CCG total area"),
            (NormalizeBy.BASELINE,     "Subtract baseline"),
        ]

        wrap = WrapFrame(norm_inner)
        wrap.pack(fill=tk.X)

        for nm, label in options:
            if self._ui.neurons is None and nm in (
                    NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE):
                continue
            var = tk.BooleanVar(value=(nm.name in saved_norms))
            self._ui.norm_vars[nm] = var
            cb = ttk.Checkbutton(wrap, text=label, variable=var,
                                 command=self._ui._on_norm_toggle)
            wrap.add(cb)
            self._norm_checkbuttons.append((cb, nm))

        bottom_row = ttk.Frame(norm_inner)
        bottom_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(bottom_row, text="Same scale (pair)",
                        variable=self._pair_scale,
                        command=self._ui._on_pair_scale_toggle).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(bottom_row, text="Same scale (session)",
                        variable=self._sess_scale,
                        command=self._ui._on_session_scale_toggle).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom_row, text="Normalize all…",
                   command=self._ui._finalize_normalization).pack(side=tk.RIGHT, padx=6)



class JitterPanel:
    """Jitter controls: n, run, clear, save, resolution selector."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._njitter       = tk.IntVar(value=100)
        self._jitter_btn_text   = tk.StringVar(value="Run Jitter")
        self._jitter_run_lo = tk.BooleanVar(
            value=bool(ui._ui_state_cache.get('jitter_run_lo', True)))
        self._jitter_run_hi = tk.BooleanVar(
            value=bool(ui._ui_state_cache.get('jitter_run_hi', False)))
        self._build(parent)

    def _build(self, parent: tk.Widget):
        jitter_inner, self._jitter_fold = _collapsible_section(parent, "Jitter")
        ttk.Label(jitter_inner, text="n=").pack(side=tk.LEFT)
        ttk.Spinbox(jitter_inner, from_=10, to=5000, increment=50,
                    textvariable=self._njitter, width=6).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(jitter_inner, textvariable=self._jitter_btn_text,
                   command=self._ui._on_run_jitter).pack(side=tk.LEFT, padx=6)
        ttk.Button(jitter_inner, text="Clear",
                   command=self._ui._on_clear_jitter).pack(side=tk.LEFT)
        ttk.Button(jitter_inner, text="Save",
                   command=self._ui._on_save_jitter).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(jitter_inner, text="Resolution:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Checkbutton(jitter_inner, text="lo",
                        variable=self._jitter_run_lo).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(jitter_inner, text="hi",
                        variable=self._jitter_run_hi).pack(side=tk.LEFT)
        ttk.Separator(jitter_inner, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=2)


class SpikeAttributionPanel:
    """Spike attribution: toggle, bin entry, Set button, and all computation."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._enabled  = tk.BooleanVar(value=False)
        self._bin_entry_var      = tk.StringVar(value="0")
        self._bin_ms: float       = 0.0
        self._spike_pairs: list   = []
        self._selected_idx: int   = -1
        self._raster_window: float = 0.050  # ±50 ms in seconds
        self._build(parent)

    def _build(self, parent: tk.Widget):
        sa_inner, self._fold = _collapsible_section(parent, "Spike Attribution")

        row1 = ttk.Frame(sa_inner)
        row1.pack(fill=tk.X)
        ttk.Checkbutton(row1, text="Allow spike attribution",
                        variable=self._enabled,
                        command=self._on_sa_toggle).pack(side=tk.LEFT)

        row2 = ttk.Frame(sa_inner)
        row2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row2, text="Bin (ms):").pack(side=tk.LEFT)
        self._bin_entry = ttk.Entry(row2, textvariable=self._bin_entry_var,
                                       width=6, state='disabled')
        self._bin_entry.pack(side=tk.LEFT, padx=2)
        self._bin_entry.bind('<Return>', lambda _: self._on_sa_set())
        self._set_btn = ttk.Button(row2, text="Set",
                                      command=self._on_sa_set,
                                      state='disabled')
        self._set_btn.pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Toggle / Set
    # ------------------------------------------------------------------

    def _on_sa_toggle(self):
        """Toggle unlock: enable/disable the bin entry and Set button."""
        enabled = self._enabled.get()
        state = 'normal' if enabled else 'disabled'
        self._bin_entry.config(state=state)
        self._set_btn.config(state=state)
        if not enabled:
            self._spike_pairs = []
            self._selected_idx = -1
            if hasattr(self._ui, 'left_container'):
                self._ui.left_container.spike_pairs.clear()
            self._ui._plot_mgr.update_plot()

    def _on_sa_set(self):
        """Query spike pairs for the current CCG pair + bin offset."""
        ui = self._ui
        if not self._enabled.get() or ui.neurons is None:
            return
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        try:
            self._bin_ms = float(self._bin_entry_var.get())
        except ValueError:
            if hasattr(ui, 'left_container'):
                ui.left_container.spike_pairs._spike_pairs_count.set("Invalid bin")
            return
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        self._compute_spike_pairs(ref, tgt, self._bin_ms)
        if hasattr(ui, 'left_container'):
            ui.left_container.spike_pairs.activate()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute_spike_pairs(self, ref: int, tgt: int, bin_ms: float):
        """Find spike pairs contributing to the given CCG bin."""
        ui = self._ui
        conf = ui.ccg_data.conf if ui.ccg_data is not None else None
        if conf is None:
            return
        n_bins = ui.ccg_data.ccg.shape[-1] if ui.ccg_data.ccg is not None else conf.nbins
        bin_size = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
        seg = ui.current_segment
        t0 = t1 = None
        if seg < ui.n_segments:
            et = ui.ccg_ptr.edge_times
            t0 = float(et.iloc[seg]['start']) if 'start' in et.columns else None
            t1 = float(et.iloc[seg]['stop']) if 'stop' in et.columns else None
        pairs = find_spike_pairs(
            ui.neurons.spiketrains[ref],
            ui.neurons.spiketrains[tgt],
            bin_ms, bin_size, t0=t0, t1=t1,
        )
        self._spike_pairs = pairs
        self._selected_idx = -1
        ui.left_container.spike_pairs.populate(pairs)

    # ------------------------------------------------------------------
    # Raster drawing
    # ------------------------------------------------------------------

    def _exit_spike_attribution_view(self):
        """Exit spike attribution raster and restore normal CCG view."""
        if self._selected_idx < 0:
            return
        self._selected_idx = -1
        self._ui._plot_mgr.update_plot()

    def _draw_sa_raster(self, idx: int):
        """Unpack UI state and call the pure plotting function."""
        ui = self._ui
        ref_t, tgt_t = self._spike_pairs[idx]
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        plot_spike_attribution_raster(
            ui.fig,
            ui.neurons.spiketrains[ref], ui.neurons.spiketrains[tgt],
            ref_t, tgt_t, self._raster_window,
            f"Ref {ui.network_panel._shank_label(ref)}",
            f"Tgt {ui.network_panel._shank_label(tgt)}",
            idx,
        )
        ui.canvas.draw()

    def _draw_spike_pairs_raster(self, idx: int, spike_pairs: list):
        """Called from SpikePairsPanel when a spike pair is clicked."""
        self._spike_pairs = spike_pairs
        self._selected_idx = idx
        self._draw_sa_raster(idx)


class SegmentChipsPanel:
    """Segment chip row and sbs / stacked-segment controls.

    Owns ``frame`` (the chip-row tk.Frame).  Mutable state
    (_lo_hi_mode, _stacked_segments, seg_sig_labels) stays on CCGReviewUI so
    the render path can read it without an extra indirection.
    """

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self.frame = ttk.Frame(parent)
        self.frame.pack(side=tk.BOTTOM, pady=2, fill=tk.X)
        self._scroller = ArrowScroller()

    def rebuild(self):
        """Rebuild all chip widgets — call whenever segment list changes."""
        ui = self._ui
        for widget in self.frame.winfo_children():
            widget.destroy()
        ui.seg_sig_labels = {}

        _fs = max(8, ui._settings_mgr.min_font_size())

        # CS chip (right-aligned)
        cs_panel = ui.center_container.cs_panel
        cs_on = cs_panel._conn_str_show.get()
        cs_btn = tk.Label(
            self.frame, text="CS",
            relief=tk.SUNKEN if cs_on else tk.RAISED,
            font=('Arial', _fs, 'bold'),
            bg='#B0C4FF' if cs_on else '#E0E0E0',
            padx=4, pady=2, cursor='hand2')
        cs_btn.pack(side=tk.RIGHT, padx=(2, 2))
        cs_btn.bind('<Button-1>', lambda e: (
            cs_panel._conn_str_show.set(not cs_panel._conn_str_show.get()),
            ui._cs_mgr._on_conn_str_toggle(),
            self.rebuild(),
        ))

        # lo|hi side-by-side toggle (right-aligned)
        lo_hi_bg     = '#B0C4FF' if ui._lo_hi_mode else '#E0E0E0'
        lo_hi_relief = tk.SUNKEN  if ui._lo_hi_mode else tk.RAISED
        lo_hi_btn = tk.Label(
            self.frame, text="lo|hi",
            relief=lo_hi_relief, font=('Arial', _fs, 'bold'),
            bg=lo_hi_bg, padx=4, pady=2, cursor='hand2')
        lo_hi_btn.pack(side=tk.RIGHT, padx=(2, 6))
        lo_hi_btn.bind('<Button-1>', lambda e: self._toggle_lo_hi_mode())

        # Build ordered chip list: All (idx=n_segments), real segs (0..n-1), custom segs
        _custom = getattr(ui, '_custom_segments', [])
        n_total = 1 + ui.n_segments + len(_custom)

        ttk.Label(self.frame, text="Segments:").pack(side=tk.LEFT, padx=(4, 0))
        self._scroller.install(self.frame, n_total, lambda _: self.rebuild(),
                               font_size=_fs, dark=ui.theme.dark)
        offset = self._scroller.offset

        def _bind_chip(lbl, seg_idx):
            lbl.bind('<Button-1>',
                     lambda e, i=seg_idx:
                         self._on_segment_chip_primary_click(i) if not (e.state & 0x4) else None)
            for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
                lbl.bind(_seq,
                         lambda e, i=seg_idx: (self._toggle_segment_chip_multi(i), 'break')[1])
            for _b in ('<Button-2>', '<Button-3>'):
                lbl.bind(_b, lambda e, i=seg_idx: self._segment_chip_ctx_menu(e, i))

        def _make_all_chip():
            lbl = tk.Label(self.frame, text="All", relief=tk.RAISED,
                           font=('Arial', _fs, 'bold'), bg='#E0E0E0',
                           padx=4, pady=2, cursor='hand2')
            _bind_chip(lbl, ui.n_segments)
            return lbl, ui.n_segments

        def _make_real_chip(i, name):
            lbl = tk.Label(self.frame, text=name, relief=tk.RAISED,
                           font=('Arial', _fs), bg='#E0E0E0', padx=4, pady=2)
            _bind_chip(lbl, i)
            return lbl, i

        def _make_custom_chip(ci, cs):
            seg_idx = ui.n_segments + 1 + ci
            lbl = tk.Label(self.frame, text=cs['name'], relief=tk.SUNKEN,
                           font=('Arial', _fs, 'italic'), bg='#FFF9C4',
                           fg='#5D4037', padx=4, pady=2)
            _bind_chip(lbl, seg_idx)
            lbl.bind('<Double-Button-1>',
                     lambda e, idx=ci: ui._custom_mgr._remove_custom_segment(idx))
            return lbl, seg_idx

        # Build scrollable chips: real segs + custom (All is pinned separately)
        scrollable_chips = []  # [(lbl, seg_idx)] in display order
        for i, name in enumerate(ui.ccg_ptr.segment_names):
            scrollable_chips.append(_make_real_chip(i, name))
        for ci, cs in enumerate(_custom):
            scrollable_chips.append(_make_custom_chip(ci, cs))

        # All chip: always visible, packed before sash
        all_lbl, all_seg_idx = _make_all_chip()
        all_lbl.pack(side=tk.LEFT, padx=2)

        # Separator (always visible)
        tk.Frame(self.frame, width=1, bg='#AAAAAA').pack(
            side=tk.LEFT, fill=tk.Y, padx=3, pady=2)

        # Pack scrollable chips from offset onward
        for lbl, _ in scrollable_chips[offset:]:
            lbl.pack(side=tk.LEFT, padx=2)

        # seg_sig_labels: dict seg_idx → label widget
        ui.seg_sig_labels = {all_seg_idx: all_lbl}
        ui.seg_sig_labels.update({seg_idx: lbl for lbl, seg_idx in scrollable_chips})

    def _toggle_lo_hi_mode(self):
        self._ui._lo_hi_mode = not self._ui._lo_hi_mode
        self.rebuild()
        self._ui._plot_mgr.update_plot()

    def _on_segment_chip_primary_click(self, idx: int):
        self._ui._stacked_segments = set()
        self._ui._jump_to_segment(idx)

    def _toggle_segment_chip_multi(self, idx: int):
        ui = self._ui
        sel = ui._stacked_segments
        if idx in sel:
            sel.discard(idx)
        else:
            sel.add(idx)
        ui._stacked_segments = sel
        inds = ui._current_inds()
        if inds is not None:
            ui._pair_mgr._update_sig_indicators(inds)

    def _segment_chip_ctx_menu(self, event, idx: int):
        ui = self._ui
        menu = tk.Menu(ui.root, tearoff=0)
        menu.add_command(
            label="Stack selected segments",
            command=lambda: ui._plot_mgr.update_plot() if ui._stacked_segments else None)
        menu.add_command(label="Select only this segment",
                         command=lambda i=idx: self._select_only_segment(i))
        menu.add_command(label="Clear segment selection",
                         command=self._clear_segment_selection)
        menu.tk_popup(event.x_root, event.y_root)

    def _select_only_segment(self, idx: int):
        ui = self._ui
        ui._stacked_segments = {int(idx)}
        inds = ui._current_inds()
        if inds is not None:
            ui._pair_mgr._update_sig_indicators(inds)

    def _clear_segment_selection(self):
        ui = self._ui
        ui._stacked_segments = set()
        inds = ui._current_inds()
        if inds is not None:
            ui._pair_mgr._update_sig_indicators(inds)


class CenterPanelContainer:
    """Assembles all center-panel sections (CCG plot, controls, chips).

    Receives separate ``plot_frame`` (top PanedWindow pane) and ``ctrl_frame``
    (scrollable bottom area) from setup_center_panel.  Panels pack into these
    with side=BOTTOM so that creation order determines visual order (first
    created = bottommost).
    """

    def __init__(self, plot_frame: tk.Widget, ctrl_frame: tk.Widget,
                 ui: 'CCGReviewUI'):
        self._ui = ui

        # ── Plot panel (top pane) ────────────────────────────────────────
        self.plot_panel = CCGPlotPanel(plot_frame, ui)

        # ── Control panels — BOTTOM stack: create in order bottom → top ─
        # 1. Baseline & CS (bottommost)
        cs_frame, self._cs_fold_var = _collapsible_section(
            ctrl_frame, "Baseline & Connection Strength")
        self.cs_panel       = CSPanel(cs_frame, ui)
        self.baseline_panel = BaselinePanel(cs_frame, ui)

        # 2. Correlograms (above Baseline & CS)
        self.correlogram_panel = CorrelogramPanel(ctrl_frame, ui)

        # 3. Normalization
        self.norm_panel = NormPanel(ctrl_frame, ui)

        # 4. Spike Attribution
        self.spike_attribution_panel = SpikeAttributionPanel(ctrl_frame, ui)

        # 5. Jitter (topmost control panel)
        self.jitter_panel = JitterPanel(ctrl_frame, ui)

        # 6. Segment chips (topmost — above jitter, below plot)
        self.seg_chips_panel = SegmentChipsPanel(ctrl_frame, ui)


