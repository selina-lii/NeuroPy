"""
CCG main view — display-state dataclasses, computation engines, and UI panel classes.

Frozen dataclasses at the top define the interface between Tk panels and the render
engine.  Each panel exposes get_config() returning the appropriate dataclass.
CenterPanelContainer.get_display_state() composes them into a single DisplayState.

Architecture
------------
    CCGReviewUI
      └── CenterPanelContainer.get_display_state() → DisplayState
              ├── CCGPlotPanel         .fig  .canvas
              ├── CSPanel              .get_config() → CSConfig
              ├── BaselinePanel        .get_config() → BaselineConfig
              ├── CorrelogramPanel     .get_config() → CorrelogramStyle
              ├── NormPanel            .get_config() → NormConfig
              ├── SpikeAttributionPanel .get_config() → SpikeAttributionConfig
              └── JitterPanel          .get_config() → JitterOverlayConfig

The render engine (ccg_renderer.py) reads DisplayState — no Tk vars cross that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


if TYPE_CHECKING:
    from neuropy.analyses.ms_connectivity import CCGKey
    from neuropy.ui.ccg_ui import CCGReviewUI


# ---------------------------------------------------------------------------
# Display-state dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtendConfig:
    """Parameters for the extended-window CCG view."""
    duration_ms: int
    bin_ms: float


@dataclass(frozen=True)
class NormConfig:
    """Active normalization methods and y-axis scale mode."""
    active: frozenset
    scale_mode: str          # 'none' | 'pair' | 'session'


@dataclass(frozen=True)
class CorrelogramStyle:
    """Display style for CCG, baseline, ACG overlays and the extend window."""
    ccg: str
    baseline: str
    acg_ref: str
    acg_tgt: str
    peak_wf: str
    acg_deconv_ref: bool
    acg_deconv_tgt: bool
    acg_scale_ref: float
    acg_scale_tgt: float
    acg_match_ccg: bool
    extend: Optional[ExtendConfig]


@dataclass(frozen=True)
class BaselineConfig:
    """Baseline method and significance display settings."""
    method: str              # 'conv' | 'tailed' | 'global' | 'jitter'
    test_window: bool
    adaptive_tw: bool
    show_conv_p: bool
    show_conv_pc: bool
    show_jitter_pc: bool


@dataclass(frozen=True)
class CSConfig:
    """Connection strength overlay settings."""
    show: bool
    metric: str              # 'STG' | 'JBSI'
    nonneg: bool


@dataclass(frozen=True)
class JitterOverlayConfig:
    """Jitter overlay display settings."""
    line_mode: bool


@dataclass(frozen=True)
class SpikeAttributionConfig:
    """Spike attribution mode and which spike pair (if any) is displayed as raster."""
    enabled: bool
    selected_idx: int        # -1 = CCG view, >=0 = raster view


@dataclass(frozen=True)
class DisplayState:
    """Complete display configuration. Frozen and hashable — used as PNG cache key."""
    norm: NormConfig
    style: CorrelogramStyle
    baseline: BaselineConfig
    cs: CSConfig
    jitter: JitterOverlayConfig
    spike_attribution: SpikeAttributionConfig
    highres: bool
    alpha: float


def style_from_bools(show: bool, line: bool) -> str:
    """Convert legacy (show_var, line_var) bool pair to tri-state style string."""
    if not show:
        return 'hidden'
    return 'line' if line else 'solid'


# ---------------------------------------------------------------------------
# SpikeAttributionEngine
# ---------------------------------------------------------------------------

class SpikeAttributionEngine:
    """Stateless engine for spike attribution computation.

    ``compute_pairs`` finds all (ref_spike_t, tgt_spike_t) pairs whose lag
    falls within the half-open interval for a given CCG time bin.  All inputs
    are plain numpy arrays — no CCGReviewUI or Tk dependency.
    """

    @staticmethod
    def compute_pairs(
        ref_spikes: np.ndarray,
        tgt_spikes: np.ndarray,
        bin_ms: float,
        bin_size_sec: float,
        t0: float | None = None,
        t1: float | None = None,
    ) -> list[tuple[float, float]]:
        """Find contributing spike pairs for a CCG bin at ``bin_ms`` lag.

        Parameters
        ----------
        ref_spikes, tgt_spikes : spike time arrays (seconds)
        bin_ms : bin centre lag in milliseconds
        bin_size_sec : bin width in seconds
        t0, t1 : optional segment time bounds (seconds); if given, spikes
                 outside [t0, t1] are excluded before searching
        """
        lag_sec = bin_ms / 1000.0
        bin_lo = lag_sec - bin_size_sec / 2.0
        bin_hi = lag_sec + bin_size_sec / 2.0

        if t0 is not None and t1 is not None:
            ref_spikes = ref_spikes[(ref_spikes >= t0) & (ref_spikes <= t1)]
            tgt_spikes = tgt_spikes[(tgt_spikes >= t0) & (tgt_spikes <= t1)]

        pairs: list[tuple[float, float]] = []
        tgt_sorted = np.sort(tgt_spikes)
        for rt in ref_spikes:
            idx_lo = np.searchsorted(tgt_sorted, rt + bin_lo, side='left')
            idx_hi = np.searchsorted(tgt_sorted, rt + bin_hi, side='right')
            for j in range(idx_lo, idx_hi):
                pairs.append((float(rt), float(tgt_sorted[j])))
        return pairs


# ---------------------------------------------------------------------------
# Shared UI helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CCGPlotPanel
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CorrelogramPanel
# ---------------------------------------------------------------------------

class CorrelogramPanel:
    """Correlograms sub-section: tri-state style buttons, ACG sliders, extend window.

    Owns all style BooleanVars.  Bridge aliases on CCGReviewUI keep backward-compat
    with _CACHE_CONFIG_ATTRS and rendering code.
    """

    _STYLE_VARS = {
        'ccg':      ('_line_ccg_var',      '_ccg_show_var'),
        'baseline': ('_line_baseline_var', '_baseline_show_var'),
        'ref':      ('_line_ref_var',      '_acg_ref_var'),
        'tgt':      ('_line_tgt_var',      '_acg_tgt_var'),
        'peak_wf':  ('_line_peak_wf_var',  '_peak_wf_var'),
    }

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        # CCG / Baseline style
        self._line_ccg_var      = tk.BooleanVar(value=False)
        self._ccg_show_var      = tk.BooleanVar(value=True)
        self._line_baseline_var = tk.BooleanVar(value=False)
        self._baseline_show_var = tk.BooleanVar(value=True)
        # ACG style (default: hidden)
        self._line_ref_var  = tk.BooleanVar(value=True)
        self._acg_ref_var   = tk.BooleanVar(value=False)
        self._line_tgt_var  = tk.BooleanVar(value=True)
        self._acg_tgt_var   = tk.BooleanVar(value=False)
        # Peak waveform style
        self._line_peak_wf_var = tk.BooleanVar(value=True)
        self._peak_wf_var      = tk.BooleanVar(value=False)
        # Jitter overlay style
        self._line_jitter_var  = tk.BooleanVar(value=False)
        # ACG deconvolution
        self._acg_deconv_ref_var = tk.BooleanVar(value=False)
        self._acg_deconv_tgt_var = tk.BooleanVar(value=False)
        # ACG Y-scale
        self._acg_yscale_ref_var = tk.DoubleVar(value=1.0)
        self._acg_yscale_tgt_var = tk.DoubleVar(value=1.0)
        self._acg_match_ccg_var  = tk.BooleanVar(value=False)
        # Extend window
        self._extend_enable_var  = tk.BooleanVar(value=False)
        self._extend_ms_var      = tk.IntVar(value=50)
        self._extend_bin_ms_var  = tk.DoubleVar(value=1.0)
        self._build(parent)

    def _get_style_vars(self, item: str):
        la, sa = self._STYLE_VARS[item]
        return getattr(self, la), getattr(self, sa)

    def _build(self, parent: tk.Widget):
        outer = ttk.Frame(parent)
        outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(3, 0))

        hdr = ttk.Frame(outer)
        hdr.pack(fill=tk.X, pady=(2, 0))
        hdr.columnconfigure(2, weight=1)
        self._acg_fold_var = tk.BooleanVar(value=True)
        tri = ttk.Label(hdr, text='▾', cursor='hand2', font=('Arial', 10))
        tri.grid(row=0, column=0, padx=(4, 0))
        ttk.Label(hdr, text='Correlograms', font=('Arial', 9, 'bold')).grid(
            row=0, column=1, padx=(3, 6))
        ttk.Separator(hdr, orient='horizontal').grid(
            row=0, column=2, sticky='ew', padx=(0, 4), pady=6)
        acg_frame = ttk.Frame(outer, padding=(8, 2, 4, 2))
        acg_frame.pack(fill=tk.X)
        self._acg_inner_frame = acg_frame

        def _toggle_acg(e=None):
            v = not self._acg_fold_var.get()
            self._acg_fold_var.set(v)
            (acg_frame.pack(fill=tk.X) if v else acg_frame.pack_forget())
            tri.config(text='▾' if v else '▸')

        tri.bind('<Button-1>', _toggle_acg)
        hdr.bind('<Button-1>', _toggle_acg)

        # CCG / Baseline tri-state buttons
        self._ccg_style_btn = tk.Label(
            acg_frame, text="■ CCG", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._ccg_style_btn.pack(side=tk.LEFT, padx=2)
        self._ccg_style_btn.bind('<Button-1>', lambda e: self._cycle_style('ccg'))

        self._baseline_style_btn = tk.Label(
            acg_frame, text="■ baseline", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._baseline_style_btn.pack(side=tk.LEFT, padx=2)
        self._baseline_style_btn.bind('<Button-1>', lambda e: self._cycle_style('baseline'))

        ttk.Separator(acg_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        ttk.Label(acg_frame, text="ACG", font=('Arial', 9, 'bold')).pack(
            side=tk.LEFT, padx=(0, 4))

        self._ref_style_btn = tk.Label(
            acg_frame, text="□ ref", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._ref_style_btn.pack(side=tk.LEFT, padx=2)
        self._ref_style_btn.bind('<Button-1>', lambda e: self._cycle_style_acg('ref'))

        self._tgt_style_btn = tk.Label(
            acg_frame, text="□ tgt", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._tgt_style_btn.pack(side=tk.LEFT, padx=2)
        self._tgt_style_btn.bind('<Button-1>', lambda e: self._cycle_style_acg('tgt'))

        self._peak_wf_style_btn = tk.Label(
            acg_frame, text="X ref peak", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._peak_wf_style_btn.pack(side=tk.LEFT, padx=2)
        self._peak_wf_style_btn.bind('<Button-1>', lambda e: self._cycle_style_acg('peak_wf'))

        ttk.Label(acg_frame, text="ACG deconvolution",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(10, 4))

        self._deconv_ref_btn = tk.Label(
            acg_frame, text="□ ref", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._deconv_ref_btn.pack(side=tk.LEFT, padx=2)
        self._deconv_ref_btn.bind('<Button-1>', lambda e: self._toggle_acg_deconv('ref'))

        self._deconv_tgt_btn = tk.Label(
            acg_frame, text="□ tgt", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._deconv_tgt_btn.pack(side=tk.LEFT, padx=2)
        self._deconv_tgt_btn.bind('<Button-1>', lambda e: self._toggle_acg_deconv('tgt'))

        ttk.Separator(acg_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # Ref ACG Y-scale
        ttk.Label(acg_frame, text="ref Y:", font=('Arial', 8)).pack(
            side=tk.LEFT, padx=(0, 1))
        ttk.Scale(acg_frame, from_=0.1, to=1.5,
                  variable=self._acg_yscale_ref_var,
                  orient=tk.HORIZONTAL, length=50,
                  command=lambda v: self._on_acg_scale_change()
                  ).pack(side=tk.LEFT, padx=1)
        self._acg_scale_ref_entry = ttk.Entry(acg_frame, width=5, font=('Courier', 8))
        self._acg_scale_ref_entry.insert(0, "1.0")
        self._acg_scale_ref_entry.pack(side=tk.LEFT, padx=(0, 1))
        self._acg_scale_ref_entry.bind('<Return>',
            lambda e: self._on_acg_entry_submit('ref'))
        self._acg_scale_ref_entry.bind('<FocusOut>',
            lambda e: self._on_acg_entry_submit('ref'))

        # Tgt ACG Y-scale
        ttk.Label(acg_frame, text="tgt Y:", font=('Arial', 8)).pack(
            side=tk.LEFT, padx=(0, 1))
        ttk.Scale(acg_frame, from_=0.1, to=1.5,
                  variable=self._acg_yscale_tgt_var,
                  orient=tk.HORIZONTAL, length=50,
                  command=lambda v: self._on_acg_scale_change()
                  ).pack(side=tk.LEFT, padx=1)
        self._acg_scale_tgt_entry = ttk.Entry(acg_frame, width=5, font=('Courier', 8))
        self._acg_scale_tgt_entry.insert(0, "1.0")
        self._acg_scale_tgt_entry.pack(side=tk.LEFT, padx=(0, 1))
        self._acg_scale_tgt_entry.bind('<Return>',
            lambda e: self._on_acg_entry_submit('tgt'))
        self._acg_scale_tgt_entry.bind('<FocusOut>',
            lambda e: self._on_acg_entry_submit('tgt'))

        ttk.Checkbutton(acg_frame, text="Match ACG to CCG scale",
                        variable=self._acg_match_ccg_var,
                        command=self._ui._on_sig_toggle).pack(side=tk.LEFT, padx=4)

        # Extend-window row
        extend_row = ttk.Frame(outer)
        extend_row.pack(fill=tk.X, anchor='w', pady=(2, 0))
        ttk.Checkbutton(
            extend_row, text="Extend",
            variable=self._extend_enable_var,
            command=self._on_extend_toggle,
        ).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(extend_row, text="ms:").pack(side=tk.LEFT)
        self._extend_ms_spin = ttk.Spinbox(
            extend_row, from_=1, to=5000, increment=1, width=6,
            textvariable=self._extend_ms_var,
            command=self._on_extend_ms_commit,
        )
        self._extend_ms_spin.pack(side=tk.LEFT, padx=(2, 6))
        self._extend_ms_spin.bind('<Return>', lambda e: self._on_extend_ms_commit())
        self._extend_ms_spin.bind('<FocusOut>', lambda e: self._on_extend_ms_commit())
        ttk.Label(extend_row, text="resolution (ms):").pack(side=tk.LEFT, padx=(8, 2))
        self._extend_bin_spin = ttk.Spinbox(
            extend_row, from_=0.0, to=100.0, increment=0.1, width=6,
            format='%.3f',
            textvariable=self._extend_bin_ms_var,
            command=self._on_extend_ms_commit,
        )
        self._extend_bin_spin.pack(side=tk.LEFT, padx=(2, 6))
        self._extend_bin_spin.bind('<Return>', lambda e: self._on_extend_ms_commit())
        self._extend_bin_spin.bind('<FocusOut>', lambda e: self._on_extend_ms_commit())

        self._update_style_btns()
        self._update_acg_deconv_btns()

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
        self._ui._clear_all_png_cache()
        self._ui.update_plot()

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
        self._ui._clear_all_png_cache()
        self._ui.update_plot()

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

    def toggle_plot_style(self):
        """Toggle all visible items between filled and outline (Ctrl+L)."""
        visible_lines = [self._get_style_vars(item)[0]
                         for item in ('ccg', 'baseline', 'ref', 'tgt', 'peak_wf')
                         if self._get_style_vars(item)[1].get()]
        any_line = any(v.get() for v in visible_lines)
        for v in visible_lines:
            v.set(not any_line)
        self._update_style_btns()
        self._ui._clear_all_png_cache()
        self._ui.update_plot()

    def _toggle_acg_deconv(self, which: str):
        if which == 'ref':
            self._acg_deconv_ref_var.set(not bool(self._acg_deconv_ref_var.get()))
        elif which == 'tgt':
            self._acg_deconv_tgt_var.set(not bool(self._acg_deconv_tgt_var.get()))
        self._update_acg_deconv_btns()
        self._ui._clear_all_png_cache()
        self._ui.update_plot()

    def _update_acg_deconv_btns(self):
        cur_ref = bool(self._acg_deconv_ref_var.get())
        cur_tgt = bool(self._acg_deconv_tgt_var.get())
        btn = getattr(self, '_deconv_ref_btn', None)
        if btn is not None:
            btn.config(text=('■ ' if cur_ref else '□ ') + 'ref')
        btn = getattr(self, '_deconv_tgt_btn', None)
        if btn is not None:
            btn.config(text=('■ ' if cur_tgt else '□ ') + 'tgt')

    def _on_acg_scale_change(self):
        ref_val = self._acg_yscale_ref_var.get()
        tgt_val = self._acg_yscale_tgt_var.get()
        if hasattr(self, '_acg_scale_ref_entry'):
            self._acg_scale_ref_entry.delete(0, tk.END)
            self._acg_scale_ref_entry.insert(0, f"{ref_val:.1f}")
        if hasattr(self, '_acg_scale_tgt_entry'):
            self._acg_scale_tgt_entry.delete(0, tk.END)
            self._acg_scale_tgt_entry.insert(0, f"{tgt_val:.1f}")
        if self._acg_ref_var.get() or self._acg_tgt_var.get() or self._peak_wf_var.get():
            self._ui._clear_all_png_cache()
            self._ui.update_plot()

    def _on_acg_entry_submit(self, which: str):
        if which == 'ref':
            entry, var = self._acg_scale_ref_entry, self._acg_yscale_ref_var
        else:
            entry, var = self._acg_scale_tgt_entry, self._acg_yscale_tgt_var
        try:
            val = max(0.01, float(entry.get()))
            var.set(val)
            entry.delete(0, tk.END)
            entry.insert(0, f"{val:.1f}")
            if self._acg_ref_var.get() or self._acg_tgt_var.get() or self._peak_wf_var.get():
                self._ui._clear_all_png_cache()
                self._ui.update_plot()
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, f"{var.get():.1f}")

    def _on_extend_toggle(self):
        try:
            self._ui._extend_cache.clear()
        except Exception:
            pass
        self._ui._clear_all_png_cache()
        self._ui.update_plot()

    def _on_extend_ms_commit(self):
        try:
            ms = max(1, min(5000, int(self._extend_ms_var.get())))
            self._extend_ms_var.set(ms)
        except Exception:
            pass
        try:
            raw_bms = float(str(self._extend_bin_ms_var.get()))
            # Minimum = 1 sample = 1000/fs ms
            try:
                neurons = getattr(self._ui, 'neurons', None)
                fs = float(getattr(neurons, 'sampling_rate', None) or 30000.0)
                if not (fs > 0):
                    fs = 30000.0
            except Exception:
                fs = 30000.0
            min_bms = 1000.0 / fs
            bms = max(min_bms, min(100.0, raw_bms))
            self._extend_bin_ms_var.set(round(bms, 4))
        except Exception:
            pass
        if bool(self._extend_enable_var.get()):
            try:
                self._ui._extend_cache.clear()
            except Exception:
                pass
            self._ui._clear_all_png_cache()
            self._ui.update_plot()

    def get_config(self) -> CorrelogramStyle:
        ext = None
        if self._extend_enable_var.get():
            ext = ExtendConfig(
                duration_ms=int(self._extend_ms_var.get()),
                bin_ms=float(self._extend_bin_ms_var.get()),
            )
        return CorrelogramStyle(
            ccg=style_from_bools(self._ccg_show_var.get(), self._line_ccg_var.get()),
            baseline=style_from_bools(self._baseline_show_var.get(), self._line_baseline_var.get()),
            acg_ref=style_from_bools(self._acg_ref_var.get(), self._line_ref_var.get()),
            acg_tgt=style_from_bools(self._acg_tgt_var.get(), self._line_tgt_var.get()),
            peak_wf=style_from_bools(self._peak_wf_var.get(), self._line_peak_wf_var.get()),
            acg_deconv_ref=bool(self._acg_deconv_ref_var.get()),
            acg_deconv_tgt=bool(self._acg_deconv_tgt_var.get()),
            acg_scale_ref=float(self._acg_yscale_ref_var.get()),
            acg_scale_tgt=float(self._acg_yscale_tgt_var.get()),
            acg_match_ccg=bool(self._acg_match_ccg_var.get()),
            extend=ext,
        )


# ---------------------------------------------------------------------------
# CSPanel
# ---------------------------------------------------------------------------

class CSPanel:
    """Connection strength overlay: show toggle, metric selector, CS label, non-negative."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._conn_str_show_var   = tk.BooleanVar(value=False)
        self._conn_str_nonneg_var = tk.BooleanVar(value=False)
        self._conn_str_metric_var = tk.StringVar(
            value=ui._ui_state_cache.get('conn_str_metric', 'STG'))
        self._cs_metric_rbs: dict = {}
        self._build(parent)

    def _build(self, parent: tk.Widget):
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, anchor='w')
        ttk.Checkbutton(row1, text="Show CS overlay",
                        variable=self._conn_str_show_var,
                        command=lambda: (self._ui._on_conn_str_toggle(),
                                        self._ui._build_sig_ccs())).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Label(row1, text="Measure:").pack(side=tk.LEFT, padx=(8, 2))
        for val in ("STG", "JBSI"):
            rb = ttk.Radiobutton(
                row1, text=val, value=val, variable=self._conn_str_metric_var,
                command=lambda: (self._ui._clear_all_png_cache(),
                                 self._ui.update_plot(),
                                 self._ui._update_conn_str_label())
            )
            rb.pack(side=tk.LEFT, padx=(0, 6))
            self._cs_metric_rbs[val] = rb
        self._ui._update_conn_str_metric_availability()

        row_cs = ttk.Frame(parent)
        row_cs.pack(fill=tk.X, anchor='w', pady=(2, 0))
        self._conn_str_label = ttk.Label(row_cs, text="CS: —")
        self._conn_str_label.pack(side=tk.LEFT)
        ttk.Checkbutton(
            row_cs,
            text="non-negative",
            variable=self._conn_str_nonneg_var,
            command=lambda: (self._ui._clear_all_png_cache(),
                             self._ui.update_plot(),
                             self._ui._update_conn_str_label(),
                             getattr(self._ui, '_stats_panel', None)
                             and self._ui._stats_panel.on_parent_display_option_changed()),
        ).pack(side=tk.LEFT, padx=(10, 0))

    def get_config(self) -> CSConfig:
        return CSConfig(
            show=bool(self._conn_str_show_var.get()),
            metric=self._conn_str_metric_var.get(),
            nonneg=bool(self._conn_str_nonneg_var.get()),
        )


# ---------------------------------------------------------------------------
# BaselinePanel
# ---------------------------------------------------------------------------

class BaselinePanel:
    """Baseline method radio buttons, test window, adaptive TW, p-value row."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._conn_str_method_var = tk.StringVar(
            value=ui._ui_state_cache.get('conn_str_method', 'conv'))
        self._sig_test_window_var = tk.BooleanVar(value=True)
        self._adaptive_tw_var     = tk.BooleanVar(value=False)
        self._sig_conv_p_var      = tk.BooleanVar(value=False)
        self._sig_conv_pc_var     = tk.BooleanVar(value=False)
        self._sig_jitter_pc_var   = tk.BooleanVar(value=False)
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
            rb = ttk.Radiobutton(row2, text=lbl, variable=self._conn_str_method_var,
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
                        variable=self._sig_test_window_var,
                        command=self._ui._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._adaptive_tw_btn = ttk.Checkbutton(
            row2, text="adaptive test window",
            variable=self._adaptive_tw_var,
            command=self._ui._on_adaptive_tw_toggle)
        self._adaptive_tw_btn.pack(side=tk.LEFT, padx=2)
        self._adaptive_tw_btn.state(['disabled'])

        self._cs_pval_row = ttk.Frame(parent)
        self._cs_pval_row.pack(fill=tk.X, anchor='w', pady=(2, 0))

    def get_config(self) -> BaselineConfig:
        return BaselineConfig(
            method=self._conn_str_method_var.get(),
            test_window=bool(self._sig_test_window_var.get()),
            adaptive_tw=bool(self._adaptive_tw_var.get()),
            show_conv_p=bool(self._sig_conv_p_var.get()),
            show_conv_pc=bool(self._sig_conv_pc_var.get()),
            show_jitter_pc=bool(self._sig_jitter_pc_var.get()),
        )


# ---------------------------------------------------------------------------
# NormPanel
# ---------------------------------------------------------------------------

class NormPanel:
    """Normalization checkboxes, same-scale toggles, and Normalize All button."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._pair_scale_var = tk.BooleanVar(
            value=ui._ui_state_cache.get('pair_scale', False))
        self._sess_scale_var = tk.BooleanVar(
            value=ui._ui_state_cache.get('sess_scale', False))
        self._norm_checkbuttons: list = []
        self._build(parent)

    def _build(self, parent: tk.Widget):
        from neuropy.analyses.ms_connectivity import NormalizeBy
        norm_inner, self._norm_fold_var = _collapsible_section(parent, "Normalization")

        saved_norms = self._ui._ui_state_cache.get('active_norms', [])
        options = [
            (NormalizeBy.REF_FRATE,    "Ref f-rate"),
            (NormalizeBy.TARGET_FRATE, "Tgt f-rate"),
            (NormalizeBy.TIME_SPAN,    "Time (hr)"),
            (NormalizeBy.TIME_SECOND,  "Time (sec)"),
            (NormalizeBy.TOTAL_AREA,   "CCG total area"),
            (NormalizeBy.BASELINE,     "Subtract baseline"),
        ]

        top_row = ttk.Frame(norm_inner)
        top_row.pack(fill=tk.X)
        norm_body = tk.Frame(top_row, height=22)
        norm_body.pack(side=tk.LEFT, fill=tk.X, expand=True)
        norm_body.pack_propagate(False)
        self._norm_body = norm_body

        for nm, label in options:
            if self._ui.neurons is None and nm in (
                    NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE):
                continue
            var = tk.BooleanVar(value=(nm.name in saved_norms))
            self._ui.norm_vars[nm] = var
            cb = ttk.Checkbutton(norm_body, text=label, variable=var,
                                 command=self._ui._on_norm_toggle)
            self._norm_checkbuttons.append((cb, nm))

        norm_body.bind('<Configure>',
                       lambda e: self._ui.root.after_idle(self._rewrap))

        bottom_row = ttk.Frame(norm_inner)
        bottom_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(bottom_row, text="Same scale (pair)",
                        variable=self._pair_scale_var,
                        command=self._ui._on_pair_scale_toggle).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(bottom_row, text="Same scale (session)",
                        variable=self._sess_scale_var,
                        command=self._ui._on_session_scale_toggle).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom_row, text="Normalize all…",
                   command=self._ui._finalize_normalization).pack(side=tk.RIGHT, padx=6)

        self._ui.root.after(150, self._rewrap)

    def _rewrap(self):
        """Reposition normalization checkboxes into wrapping rows."""
        if not hasattr(self, '_norm_body') or not self._norm_checkbuttons:
            return
        body = self._norm_body
        body.update_idletasks()
        avail_w = body.winfo_width()
        if avail_w <= 1:
            body.after(100, self._rewrap)
            return
        PAD_X, PAD_Y = 2, 1
        x, y, row_h = PAD_X, PAD_Y, 0
        for cb, _ in self._norm_checkbuttons:
            if not cb.winfo_exists():
                continue
            cb.update_idletasks()
            ww = cb.winfo_reqwidth()
            wh = cb.winfo_reqheight()
            if x + ww > avail_w - PAD_X and x > PAD_X:
                y += row_h + PAD_Y
                x, row_h = PAD_X, 0
            cb.place(x=x, y=y)
            x += ww + PAD_X
            row_h = max(row_h, wh)
        body.configure(height=max(y + row_h + PAD_Y * 2, 22))

    def get_config(self) -> NormConfig:
        active = frozenset(nm for nm, v in self._ui.norm_vars.items() if v.get())
        if self._pair_scale_var.get():
            scale_mode = 'pair'
        elif self._sess_scale_var.get():
            scale_mode = 'session'
        else:
            scale_mode = 'none'
        return NormConfig(active=active, scale_mode=scale_mode)


# ---------------------------------------------------------------------------
# JitterPanel
# ---------------------------------------------------------------------------

class JitterPanel:
    """Jitter controls: n, run, clear, save, resolution selector."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._njitter_var       = tk.IntVar(value=100)
        self._jitter_btn_text   = tk.StringVar(value="Run Jitter")
        self._jitter_run_lo_var = tk.BooleanVar(
            value=bool(ui._ui_state_cache.get('jitter_run_lo', True)))
        self._jitter_run_hi_var = tk.BooleanVar(
            value=bool(ui._ui_state_cache.get('jitter_run_hi', False)))
        self._build(parent)

    def _build(self, parent: tk.Widget):
        jitter_inner, self._jitter_fold_var = _collapsible_section(parent, "Jitter")
        ttk.Label(jitter_inner, text="n=").pack(side=tk.LEFT)
        ttk.Spinbox(jitter_inner, from_=10, to=5000, increment=50,
                    textvariable=self._njitter_var, width=6).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(jitter_inner, textvariable=self._jitter_btn_text,
                   command=self._ui._on_run_jitter).pack(side=tk.LEFT, padx=6)
        ttk.Button(jitter_inner, text="Clear",
                   command=self._ui._on_clear_jitter).pack(side=tk.LEFT)
        ttk.Button(jitter_inner, text="Save",
                   command=self._ui._on_save_jitter).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(jitter_inner, text="Resolution:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Checkbutton(jitter_inner, text="lo",
                        variable=self._jitter_run_lo_var).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(jitter_inner, text="hi",
                        variable=self._jitter_run_hi_var).pack(side=tk.LEFT)
        ttk.Separator(jitter_inner, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=2)
        ttk.Checkbutton(jitter_inner, text="Waveform",
                        variable=self._ui._panel_vars['Waveforms'],
                        command=self._ui._toggle_waveforms_panel).pack(side=tk.LEFT)


# ---------------------------------------------------------------------------
# SpikeAttributionPanel
# ---------------------------------------------------------------------------

class SpikeAttributionPanel:
    """Spike attribution: toggle, bin entry, Set button, and all computation."""

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._sa_enabled_var  = tk.BooleanVar(value=False)
        self._sa_bin_var      = tk.StringVar(value="0")
        self._sa_bin_ms: float       = 0.0
        self._sa_spike_pairs: list   = []
        self._sa_selected_idx: int   = -1
        self._sa_raster_window: float = 0.050  # ±50 ms in seconds
        self._build(parent)

    def _build(self, parent: tk.Widget):
        sa_inner, self._sa_fold_var = _collapsible_section(parent, "Spike Attribution")

        row1 = ttk.Frame(sa_inner)
        row1.pack(fill=tk.X)
        ttk.Checkbutton(row1, text="Allow spike attribution",
                        variable=self._sa_enabled_var,
                        command=self._on_sa_toggle).pack(side=tk.LEFT)

        row2 = ttk.Frame(sa_inner)
        row2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row2, text="Bin (ms):").pack(side=tk.LEFT)
        self._sa_bin_entry = ttk.Entry(row2, textvariable=self._sa_bin_var,
                                       width=6, state='disabled')
        self._sa_bin_entry.pack(side=tk.LEFT, padx=2)
        self._sa_bin_entry.bind('<Return>', lambda _: self._on_sa_set())
        self._sa_set_btn = ttk.Button(row2, text="Set",
                                      command=self._on_sa_set,
                                      state='disabled')
        self._sa_set_btn.pack(side=tk.LEFT, padx=4)

    def get_config(self) -> SpikeAttributionConfig:
        return SpikeAttributionConfig(
            enabled=bool(self._sa_enabled_var.get()),
            selected_idx=self._sa_selected_idx,
        )

    # ------------------------------------------------------------------
    # Toggle / Set
    # ------------------------------------------------------------------

    def _on_sa_toggle(self):
        """Toggle unlock: enable/disable the bin entry and Set button."""
        enabled = self._sa_enabled_var.get()
        state = 'normal' if enabled else 'disabled'
        self._sa_bin_entry.config(state=state)
        self._sa_set_btn.config(state=state)
        if not enabled:
            self._sa_spike_pairs = []
            self._sa_selected_idx = -1
            if hasattr(self._ui, 'left_container'):
                self._ui.left_container.spike_pairs.clear()
            self._ui.update_plot()

    def _on_sa_set(self):
        """Query spike pairs for the current CCG pair + bin offset."""
        ui = self._ui
        if not self._sa_enabled_var.get() or ui.neurons is None:
            return
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        try:
            self._sa_bin_ms = float(self._sa_bin_var.get())
        except ValueError:
            if hasattr(ui, 'left_container'):
                ui.left_container.spike_pairs._spike_pairs_count.set("Invalid bin")
            return
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        self._compute_spike_pairs(ref, tgt, self._sa_bin_ms)
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
            et = ui.ccg_pointer.edge_times
            t0 = float(et.iloc[seg]['start']) if 'start' in et.columns else None
            t1 = float(et.iloc[seg]['stop']) if 'stop' in et.columns else None
        pairs = SpikeAttributionEngine.compute_pairs(
            ui.neurons.spiketrains[ref],
            ui.neurons.spiketrains[tgt],
            bin_ms, bin_size, t0=t0, t1=t1,
        )
        self._sa_spike_pairs = pairs
        self._sa_selected_idx = -1
        if hasattr(ui, 'left_container'):
            ui.left_container.spike_pairs.populate(pairs)
        else:
            ui._sa_listbox.delete(0, tk.END)
            for i, (rt, tt) in enumerate(pairs):
                lag_ms = (tt - rt) * 1000.0
                ui._sa_listbox.insert(
                    tk.END,
                    f"{i+1:>5}  ref {rt:10.4f}  tgt {tt:10.4f}  lag {lag_ms:+6.2f}ms")
            ui._sa_count_var.set(f"{len(pairs)} spike pairs")

    # ------------------------------------------------------------------
    # Raster drawing
    # ------------------------------------------------------------------

    def _exit_spike_attribution_view(self):
        """Exit spike attribution raster and restore normal CCG view."""
        if self._sa_selected_idx < 0:
            return
        self._sa_selected_idx = -1
        self._ui.update_plot()

    def _on_sa_pair_click(self, _event=None):
        """Handle click on a spike pair — show raster in center panel."""
        ui = self._ui
        sel = ui._sa_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._sa_spike_pairs):
            return
        self._sa_selected_idx = idx
        self._draw_sa_raster(idx)

    def _draw_sa_raster(self, idx: int):
        """Draw a 2-row raster of ref/tgt spike trains around the selected pair."""
        ui = self._ui
        ref_t, tgt_t = self._sa_spike_pairs[idx]
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        center = ref_t
        win = self._sa_raster_window
        t0, t1 = center - win, center + win
        ref_spikes = ui.neurons.spiketrains[ref]
        tgt_spikes = ui.neurons.spiketrains[tgt]
        ref_win = ref_spikes[(ref_spikes >= t0) & (ref_spikes <= t1)]
        tgt_win = tgt_spikes[(tgt_spikes >= t0) & (tgt_spikes <= t1)]
        ref_label = f"Ref {ui.network_panel._shank_label(ref)}"
        tgt_label = f"Tgt {ui.network_panel._shank_label(tgt)}"
        ui.fig.clear()
        ax_ref = ui.fig.add_subplot(211)
        ax_tgt = ui.fig.add_subplot(212, sharex=ax_ref)
        if len(ref_win):
            ax_ref.eventplot([ref_win - center], lineoffsets=0,
                             linelengths=0.8, colors='#1565C0')
        ax_ref.axvline(0, color='#E53935', lw=1.5, ls='--', alpha=0.7)
        ax_ref.set_ylabel(ref_label, fontsize=9)
        ax_ref.set_yticks([])
        ax_ref.set_title(
            f"Spike pair #{idx+1}: ref={ref_t:.4f}s  tgt={tgt_t:.4f}s  "
            f"lag={(tgt_t - ref_t)*1000:.2f}ms",
            fontsize=9)
        if len(tgt_win):
            ax_tgt.eventplot([tgt_win - center], lineoffsets=0,
                             linelengths=0.8, colors='#2E7D32')
        ax_tgt.axvline(tgt_t - center, color='#E53935', lw=1.5, ls='--', alpha=0.7)
        ax_tgt.set_ylabel(tgt_label, fontsize=9)
        ax_tgt.set_yticks([])
        ax_tgt.set_xlabel("Time relative to ref spike (s)", fontsize=9)
        ax_tgt.set_xlim(-win, win)
        ui.fig.tight_layout()
        ui.canvas.draw()

    def _draw_spike_pairs_raster(self, idx: int, spike_pairs: list):
        """Called from SpikePairsPanel when a spike pair is clicked."""
        self._sa_spike_pairs = spike_pairs
        self._sa_selected_idx = idx
        self._draw_sa_raster(idx)


# ---------------------------------------------------------------------------
# SegmentChipsPanel
# ---------------------------------------------------------------------------

class SegmentChipsPanel:
    """Segment chip row and sbs / stacked-segment controls.

    Owns ``frame`` (the chip-row tk.Frame).  Mutable state
    (_sbs_mode, _stacked_segments, seg_sig_labels) stays on CCGReviewUI so
    the render path can read it without an extra indirection.
    """

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self.frame = ttk.Frame(parent)
        self.frame.pack(side=tk.BOTTOM, pady=2, fill=tk.X)

    def rebuild(self):
        """Rebuild all chip widgets — call whenever segment list changes."""
        ui = self._ui
        for widget in self.frame.winfo_children():
            widget.destroy()
        ui.seg_sig_labels = []

        _fs = max(8, ui._min_font_size())

        # CS chip (right-aligned)
        cs_panel = ui.center_container.cs_panel
        cs_on = cs_panel._conn_str_show_var.get()
        cs_btn = tk.Label(
            self.frame, text="CS",
            relief=tk.SUNKEN if cs_on else tk.RAISED,
            font=('Arial', _fs, 'bold'),
            bg='#B0C4FF' if cs_on else '#E0E0E0',
            padx=4, pady=2, cursor='hand2')
        cs_btn.pack(side=tk.RIGHT, padx=(2, 2))
        cs_btn.bind('<Button-1>', lambda e: (
            cs_panel._conn_str_show_var.set(not cs_panel._conn_str_show_var.get()),
            ui._on_conn_str_toggle(),
            self.rebuild(),
        ))

        # lo|hi side-by-side toggle (right-aligned)
        sbs_bg     = '#B0C4FF' if ui._sbs_mode else '#E0E0E0'
        sbs_relief = tk.SUNKEN  if ui._sbs_mode else tk.RAISED
        sbs_btn = tk.Label(
            self.frame, text="lo|hi",
            relief=sbs_relief, font=('Arial', _fs, 'bold'),
            bg=sbs_bg, padx=4, pady=2, cursor='hand2')
        sbs_btn.pack(side=tk.RIGHT, padx=(2, 6))
        sbs_btn.bind('<Button-1>', lambda e: self._toggle_sbs_mode())

        ttk.Label(self.frame, text="Segments:").pack(side=tk.LEFT, padx=(4, 2))

        # "All" chip — appended to seg_sig_labels AFTER real segs (index = n_segments)
        lbl_all = tk.Label(
            self.frame, text="All",
            relief=tk.RAISED, font=('Arial', _fs, 'bold'),
            bg='#E0E0E0', padx=4, pady=2, cursor='hand2')
        lbl_all.pack(side=tk.LEFT, padx=(2, 0))
        lbl_all.bind(
            '<Button-1>',
            lambda e: self._on_segment_chip_primary_click(ui.n_segments))
        for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
            lbl_all.bind(
                _seq,
                lambda e: (self._toggle_segment_chip_multi(ui.n_segments), 'break')[1])
        for _btn in ('<Button-2>', '<Button-3>'):
            lbl_all.bind(
                _btn,
                lambda e: self._segment_chip_ctx_menu(e, ui.n_segments))

        # Separator
        tk.Frame(self.frame, width=1, bg='#AAAAAA').pack(
            side=tk.LEFT, fill=tk.Y, padx=3, pady=2)

        # Real segment chips
        for i, name in enumerate(ui.segment_names):
            lbl = tk.Label(
                self.frame, text=name,
                relief=tk.RAISED, font=('Arial', _fs),
                bg='#E0E0E0', padx=4, pady=2)
            lbl.pack(side=tk.LEFT, padx=2)
            lbl.bind('<Button-1>',
                     lambda e, idx=i: self._on_segment_chip_primary_click(idx))
            for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
                lbl.bind(_seq,
                         lambda e, idx=i: (self._toggle_segment_chip_multi(idx), 'break')[1])
            for _btn in ('<Button-2>', '<Button-3>'):
                lbl.bind(_btn, lambda e, idx=i: self._segment_chip_ctx_menu(e, idx))
            ui.seg_sig_labels.append(lbl)

        # Append All chip at index n_segments
        ui.seg_sig_labels.append(lbl_all)

        # Custom segment chips
        for ci, cs in enumerate(getattr(ui, '_custom_segments', [])):
            seg_idx = ui.n_segments + 1 + ci
            lbl_cust = tk.Label(
                self.frame, text=cs['name'],
                relief=tk.SUNKEN, font=('Arial', _fs, 'italic'),
                bg='#FFF9C4', fg='#5D4037', padx=4, pady=2)
            lbl_cust.pack(side=tk.LEFT, padx=(4, 2))
            lbl_cust.bind('<Button-1>',
                          lambda e, idx=seg_idx: self._on_segment_chip_primary_click(idx))
            for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
                lbl_cust.bind(_seq,
                              lambda e, idx=seg_idx: (self._toggle_segment_chip_multi(idx), 'break')[1])
            for _btn in ('<Button-2>', '<Button-3>'):
                lbl_cust.bind(_btn,
                              lambda e, idx=seg_idx: self._segment_chip_ctx_menu(e, idx))
            lbl_cust.bind('<Double-Button-1>',
                          lambda e, idx=ci: ui._remove_custom_segment(idx))
            ui.seg_sig_labels.append(lbl_cust)

    def _toggle_sbs_mode(self):
        ui = self._ui
        ui._sbs_mode = not ui._sbs_mode
        self.rebuild()
        ui.update_plot()

    def _on_segment_chip_primary_click(self, idx: int):
        ui = self._ui
        try:
            ui._stacked_segments.clear()
        except Exception:
            ui._stacked_segments = set()
        ui._jump_to_segment(idx)

    def _toggle_segment_chip_multi(self, idx: int):
        ui = self._ui
        sel = getattr(ui, '_stacked_segments', set()) or set()
        if idx in sel:
            sel.discard(idx)
        else:
            sel.add(idx)
        ui._stacked_segments = sel
        inds = ui._current_inds()
        if inds is not None:
            ui._update_sig_indicators(inds)

    def _segment_chip_ctx_menu(self, event, idx: int):
        ui = self._ui
        menu = tk.Menu(ui.root, tearoff=0)
        menu.add_command(label="Stack selected segments",
                         command=self._show_stacked_segments)
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
            ui._update_sig_indicators(inds)

    def _clear_segment_selection(self):
        ui = self._ui
        try:
            ui._stacked_segments.clear()
        except Exception:
            ui._stacked_segments = set()
        inds = ui._current_inds()
        if inds is not None:
            ui._update_sig_indicators(inds)

    def _show_stacked_segments(self):
        ui = self._ui
        if not getattr(ui, '_stacked_segments', None):
            return
        ui.update_plot()


# ---------------------------------------------------------------------------
# CenterPanelContainer
# ---------------------------------------------------------------------------

class CenterPanelContainer:
    """Assembles all center-panel sections and provides get_display_state().

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

    def get_display_state(self) -> DisplayState:
        """Compose the current display configuration into a frozen DisplayState."""
        return DisplayState(
            norm=self.norm_panel.get_config(),
            style=self.correlogram_panel.get_config(),
            baseline=self.baseline_panel.get_config(),
            cs=self.cs_panel.get_config(),
            jitter=JitterOverlayConfig(
                line_mode=bool(self.correlogram_panel._line_jitter_var.get())),
            spike_attribution=self.spike_attribution_panel.get_config(),
            highres=bool(getattr(self._ui, '_highres_mode', False)),
            alpha=float(getattr(self._ui, 'active_alpha', 0.05)),
        )
