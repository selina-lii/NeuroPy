"""
CCG Manual Review UI

Interactive GUI for reviewing and selecting significant CCG pairs.

Layout (3-column PanedWindow):
  Left   — pair selection lists + alpha control
  Center — CCG plot with normalization toggles, segment navigation, optional
            waveforms sub-panel
  Right  — probe network (neuron positions + connection edges)

Top bar:  menubar (Panels menu) + tool strip (session/type/resolution/segment filter)
Time slider: full-width panel above the 3-column area, hidden by default.
Bottom bar: pair statistics + Save / Cancel buttons.

Keyboard shortcuts
------------------
  ←  /  →      previous / next segment
  Ctrl+R        toggle lo-res ↔ hi-res (cycles datasets when > 2)
  Ctrl+E        toggle waveforms sub-panel
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from neuropy.plotting import ccg as plot_ccg
import imageio

try:
    from neuropy.analyses.ms_connectivity import NormalizeBy
except ImportError:
    NormalizeBy = None

# Sentinel value for the virtual "All segments" view
_ALL_SEGS = "All segments"


class CCGReviewUI:
    """
    GUI for reviewing and manually selecting CCG pairs.

    Parameters
    ----------
    cd : CCGDataset
        Dataset to review.  If ``cd._ccg_highres`` is populated
        (via ``cd.load_highres()``), a resolution toggle button is shown.
    key : Key
        Identifies which CCGPointer to review.
    """

    def __init__(self, cd, key):
        self.cd = cd

        self.key = key
        self.ccg_pointer = self.cd.data.get(key)
        self.ccg_data = self.cd._ccg.get(key.nd())

        if self.ccg_pointer is None or self.ccg_pointer.inds is None:
            raise ValueError(f"No CCG data found for key: {key}")

        # Neurons (normalization + network + waveforms)
        self.neurons = (self.cd.nd.data[key.nd()]
                        if getattr(self.cd, 'nd', None) is not None
                        else None)

        # Pair / segment state
        self.all_inds = self.ccg_pointer.inds2          # unique (ref, tgt)
        self.n_segments = self.ccg_pointer.n_segments   # real segment count
        self.segment_names = list(self.ccg_pointer.edge_times['label'].values)
        self.current_segment = 0   # 0..n_segments-1 = real; n_segments = All
        self.current_pair_idx = 0

        # Manual selection state
        if (hasattr(self.ccg_pointer, 'manually_selected_inds') and
                self.ccg_pointer.manually_selected_inds is not None):
            self.selected_inds = set(
                map(tuple, self.ccg_pointer.manually_selected_inds))
        else:
            self.selected_inds = set()
        self.unselected_inds = set(map(tuple, self.all_inds)) - self.selected_inds

        # Active config
        self.active_alpha = getattr(getattr(self.cd, 'conf', None), 'alpha', 0.05)
        self.active_norms: set = set()
        self.norm_vars: dict = {}
        self.active_segment_filter = None

        # Resolution state
        self._highres_mode = False           # True when showing _ccg_highres

        # Same-scale state (Task 1)
        self._same_scale_mode: str = None    # None | 'pair' | 'session'
        self._pair_scale_cache: dict = {}    # (ref, tgt) -> (ymin, ymax)
        self._session_scale_cache = None     # (ymin, ymax)

        # Jitter state (Task 2)
        self._jitter_cache: dict = {}        # (ref, tgt) -> (j_avg, j_pval)

        # Panel visibility state
        self._panel_vars: dict = {}          # populated in setup_panels_menu
        self._waveforms_visible = False

        # Time-slider / custom-window state
        self._slider_t_start: float = None
        self._slider_t_end: float = None
        self._custom_window_active: bool = False
        self._custom_ccg_cache: dict = {}    # (ref, tgt) → np.ndarray
        self._slider_dragging: str = None    # 'start' | 'end' | None
        self._ts_epoch_bounds = []           # [(t0_sec, t1_sec, label), …]
        self._ts_total_sec: float = 0.0

        # PNG cache
        self.tmp_dir = os.path.expanduser(
            "~/Documents/ms_synchrony/NeuroPy/images/tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Build UI
        self.root = tk.Tk()
        self.root.title("CCG Manual Review")
        self.setup_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.root.geometry("1800x950")

        # ── Menubar (Panels menu) ──────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.setup_panels_menu(menubar)

        # ── Tool-strip row ─────────────────────────────────────────────
        self.setup_menu()

        # ── Bottom bar (packed before main so it gets space first) ─────
        self.setup_bottom_panel()

        # ── Main area ──────────────────────────────────────────────────
        self._main_frame = ttk.Frame(self.root)
        self._main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Time slider (full-width, hidden by default; packed before paned)
        self.setup_time_slider_panel(self._main_frame)

        # Three-column PanedWindow
        self._paned = ttk.PanedWindow(self._main_frame, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        self._left_frame = ttk.Frame(self._paned, width=350)
        self._center_frame = ttk.Frame(self._paned)
        self._right_frame = ttk.Frame(self._paned, width=340)

        self._paned.add(self._left_frame, weight=0)
        self._paned.add(self._center_frame, weight=1)
        self._paned.add(self._right_frame, weight=0)

        self.setup_left_panel(self._left_frame)
        self.setup_center_panel(self._center_frame)
        self.setup_network_panel(self._right_frame)

        # Keyboard bindings (Control for Linux/Windows; Command for macOS)
        self.root.bind('<Left>',      lambda e: self.change_segment(-1))
        self.root.bind('<Right>',     lambda e: self.change_segment(1))
        for _key in ('<Control-r>', '<Command-r>'):
            self.root.bind(_key, lambda e: self._toggle_resolution())
        for _key in ('<Control-e>', '<Command-e>'):
            self.root.bind(_key, lambda e: self._on_ctrl_e())

    # ── Menubar ────────────────────────────────────────────────────────

    def setup_panels_menu(self, menubar):
        """Panels menu with checkbuttons for each panel."""
        panels_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Panels", menu=panels_menu)
        panel_defaults = [
            ('Pair Selection', True),
            ('CCG',            True),
            ('Probe Network',  True),
            ('Waveforms',      False),
            ('Time Slider',    False),
        ]
        for name, default in panel_defaults:
            var = tk.BooleanVar(value=default)
            self._panel_vars[name] = var
            panels_menu.add_checkbutton(
                label=name, variable=var,
                command=lambda n=name: self._toggle_panel(n))

    # ── Tool-strip row ─────────────────────────────────────────────────

    def setup_menu(self):
        menu_frame = ttk.Frame(self.root, relief=tk.RAISED, borderwidth=2)
        menu_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Session
        ttk.Label(menu_frame, text="Session:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(8, 2))
        nd_keys = self._all_nd_keys()
        self._nd_keys_list = nd_keys
        session_labels = [self._session_label(k) for k in nd_keys]
        self._session_var = tk.StringVar(value=self._session_label(self.key.nd()))
        self._session_combo = ttk.Combobox(
            menu_frame, textvariable=self._session_var,
            values=session_labels, width=22, state='readonly')
        self._session_combo.pack(side=tk.LEFT, padx=2)
        self._session_combo.bind('<<ComboboxSelected>>', self._on_session_change)

        # Type
        ttk.Label(menu_frame, text="Type:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(10, 2))
        type_keys = self._available_type_keys(self.key.nd())
        self._type_keys_list = type_keys
        type_labels = [self._type_label(k) for k in type_keys]
        self._type_var = tk.StringVar(value=self._type_label(self.key))
        self._type_combo = ttk.Combobox(
            menu_frame, textvariable=self._type_var,
            values=type_labels, width=18, state='readonly')
        self._type_combo.pack(side=tk.LEFT, padx=2)
        self._type_combo.bind('<<ComboboxSelected>>', self._on_type_change)



    # ── Left panel ─────────────────────────────────────────────────────

    def setup_left_panel(self, parent):
        ttk.Label(parent, text="Pair Selection",
                  font=('Arial', 11, 'bold')).pack()

        columns_frame = ttk.Frame(parent)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        # Unselected list
        unsel_frame = ttk.Frame(columns_frame)
        unsel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self._avail_label_var = tk.StringVar(
            value=f"Available ({len(self.unselected_inds)})")
        ttk.Label(unsel_frame, textvariable=self._avail_label_var,
                  font=('Arial', 10)).pack()
        unsel_scroll = ttk.Scrollbar(unsel_frame)
        unsel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.unselected_list = tk.Listbox(
            unsel_frame, yscrollcommand=unsel_scroll.set,
            selectmode=tk.EXTENDED, font=('Courier', 9))
        self.unselected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        unsel_scroll.config(command=self.unselected_list.yview)
        self.unselected_list.bind('<Double-Button-1>', self.move_to_selected)
        self.unselected_list.bind('<Return>',          self.move_to_selected)
        self.unselected_list.bind('<<ListboxSelect>>', self.on_pair_select)
        self.unselected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))

        # Selected list
        sel_frame = ttk.Frame(columns_frame)
        sel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self._sel_label_var = tk.StringVar(
            value=f"Selected ({len(self.selected_inds)})")
        ttk.Label(sel_frame, textvariable=self._sel_label_var,
                  font=('Arial', 10)).pack()
        sel_scroll = ttk.Scrollbar(sel_frame)
        sel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_list = tk.Listbox(
            sel_frame, yscrollcommand=sel_scroll.set,
            selectmode=tk.EXTENDED, font=('Courier', 9))
        self.selected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scroll.config(command=self.selected_list.yview)
        self.selected_list.bind('<Double-Button-1>', self.move_to_unselected)
        self.selected_list.bind('<Return>',          self.move_to_unselected)
        self.selected_list.bind('<<ListboxSelect>>', self.on_pair_select)
        self.selected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))

        self.refresh_lists()

        # Buttons row: Select All/Deselect All / resolution toggle
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(4, 0))
        self._select_all_btn = ttk.Button(btn_frame, text="Select All",
                                          command=self._select_all)
        self._select_all_btn.pack(side=tk.LEFT, padx=(0, 4))

        # Resolution toggle (disabled when _ccg_highres not loaded)
        _has_highres = (hasattr(self.cd, '_ccg_highres')
                        and bool(self.cd._ccg_highres))
        init_label = 'Res: lowres' if _has_highres else 'Res: default'
        self._res_btn_text = tk.StringVar(value=init_label)
        res_btn = ttk.Button(btn_frame, textvariable=self._res_btn_text,
                             command=self._toggle_resolution)
        res_btn.pack(side=tk.LEFT, padx=2)
        if not _has_highres:
            res_btn.state(['disabled'])

    # ── Center panel ───────────────────────────────────────────────────

    def setup_center_panel(self, parent):
        self.plot_title_var = tk.StringVar(value=self.get_plot_title())
        ttk.Label(parent, textvariable=self.plot_title_var,
                  font=('Arial', 11, 'bold')).pack(side=tk.TOP)

        # Bottom controls packed before the canvas (Tkinter pack order)
        self.setup_norm_panel(parent)
        self.setup_jitter_panel(parent)
        self.setup_waveforms_panel(parent)   # hidden by default

        # Significance chips
        self.sig_frame = ttk.Frame(parent)
        self.sig_frame.pack(side=tk.BOTTOM, pady=2, fill=tk.X)
        self._build_sig_chips()

        # Segment navigation: ← / [combobox] / → (includes "All segments")
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(side=tk.BOTTOM, pady=4)
        ttk.Button(nav_frame, text="←",
                   command=lambda: self.change_segment(-1)).pack(
            side=tk.LEFT, padx=4)
        self.segment_var = tk.StringVar(
            value=self.segment_names[self.current_segment])
        self.segment_combo = ttk.Combobox(
            nav_frame, textvariable=self.segment_var,
            values=self.segment_names + [_ALL_SEGS], width=14,
            state='readonly', font=('Arial', 10, 'bold'))
        self.segment_combo.pack(side=tk.LEFT, padx=4)
        self.segment_combo.bind('<<ComboboxSelected>>', self._on_segment_change)
        ttk.Button(nav_frame, text="→",
                   command=lambda: self.change_segment(1)).pack(
            side=tk.LEFT, padx=4)

        # CCG figure
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.root.after(100, self._deferred_initial_draw)

    def setup_norm_panel(self, parent):
        if NormalizeBy is None:
            return
        norm_frame = ttk.LabelFrame(parent, text="Normalization", padding=4)
        norm_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        options = [
            (NormalizeBy.REF_FRATE,    "Ref firing rate"),
            (NormalizeBy.TARGET_FRATE, "Tgt firing rate"),
            (NormalizeBy.TIME_SPAN,    "Time span"),
        ]
        btn_frame = ttk.Frame(norm_frame)
        btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for nm, label in options:
            if self.neurons is None and nm in (
                    NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE):
                continue
            var = tk.BooleanVar(value=False)
            self.norm_vars[nm] = var
            cb = ttk.Checkbutton(btn_frame, text=label, variable=var,
                                 command=self._on_norm_toggle)
            cb.pack(side=tk.LEFT, padx=4)
        # Scale checkbuttons
        scale_frame = ttk.Frame(norm_frame)
        scale_frame.pack(side=tk.LEFT, padx=(12, 0))
        self._pair_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scale_frame, text="Same scale (pair)",
                        variable=self._pair_scale_var,
                        command=self._on_pair_scale_toggle).pack(side=tk.LEFT, padx=4)
        self._sess_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scale_frame, text="Same scale (session)",
                        variable=self._sess_scale_var,
                        command=self._on_session_scale_toggle).pack(side=tk.LEFT, padx=4)

        ttk.Button(norm_frame, text="Normalize All",
                   command=self._finalize_normalization).pack(
            side=tk.RIGHT, padx=6)

    def setup_jitter_panel(self, parent):
        """Jitter controls: run jitter on demand for the current pair."""
        jitter_frame = ttk.LabelFrame(parent, text="Jitter", padding=4)
        jitter_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        ttk.Label(jitter_frame, text="n=").pack(side=tk.LEFT)
        self._njitter_var = tk.IntVar(value=100)
        ttk.Spinbox(jitter_frame, from_=10, to=5000, increment=50,
                    textvariable=self._njitter_var, width=6).pack(
            side=tk.LEFT, padx=2)
        self._jitter_btn_text = tk.StringVar(value="Run Jitter")
        ttk.Button(jitter_frame, textvariable=self._jitter_btn_text,
                   command=self._on_run_jitter).pack(side=tk.LEFT, padx=6)
        ttk.Button(jitter_frame, text="Clear",
                   command=self._on_clear_jitter).pack(side=tk.LEFT)

    def setup_waveforms_panel(self, parent):
        """Waveform sub-panel in center column — hidden by default."""
        self.wave_frame = ttk.LabelFrame(parent, text="Waveforms")
        # Not packed — toggled via Panels menu / Ctrl+E
        self.wave_fig = Figure(figsize=(4, 5), tight_layout=True)
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.wave_frame)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Right panel ────────────────────────────────────────────────────

    def setup_network_panel(self, parent):
        ttk.Label(parent, text="Probe Network",
                  font=('Arial', 10, 'bold')).pack(pady=(0, 2))
        self.net_fig = Figure(figsize=(2.8, 6.5))
        self.net_ax = self.net_fig.add_subplot(111)
        self.net_canvas = FigureCanvasTkAgg(self.net_fig, master=parent)
        self.net_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._net_pick_cid = self.net_canvas.mpl_connect(
            'pick_event', self._on_network_pick)
        self.root.after(200, self._draw_network)

    # ── Time slider panel ──────────────────────────────────────────────

    def setup_time_slider_panel(self, parent):
        """Full-width time-window selector — hidden by default."""
        self.time_slider_frame = ttk.LabelFrame(
            parent, text="Time Window (Custom CCG)")
        # Not packed — shown when 'Time Slider' panel is enabled

        top = ttk.Frame(self.time_slider_frame)
        top.pack(fill=tk.X, padx=4, pady=(2, 0))

        self.ts_canvas = tk.Canvas(top, height=44, bg='#F5F5F5', cursor='crosshair')
        self.ts_canvas.pack(fill=tk.X, expand=True)
        self.ts_canvas.bind('<Configure>',      self._ts_redraw)
        self.ts_canvas.bind('<Button-1>',        self._ts_mouse_press)
        self.ts_canvas.bind('<B1-Motion>',       self._ts_mouse_drag)
        self.ts_canvas.bind('<ButtonRelease-1>', self._ts_mouse_release)

        ctrl = ttk.Frame(self.time_slider_frame)
        ctrl.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(ctrl, text="Start:").pack(side=tk.LEFT)
        self._ts_start_var = tk.StringVar(value="00:00:00")
        ttk.Entry(ctrl, textvariable=self._ts_start_var,
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="End:").pack(side=tk.LEFT, padx=(6, 0))
        self._ts_end_var = tk.StringVar(value="00:00:00")
        ttk.Entry(ctrl, textvariable=self._ts_end_var,
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Set",
                   command=self._on_time_slider_set).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Clear",
                   command=self._on_time_slider_clear).pack(side=tk.LEFT, padx=2)
        self._ts_status_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._ts_status_var,
                  font=('Courier', 8), foreground='#555').pack(
            side=tk.LEFT, padx=8)

    # ── Bottom bar ─────────────────────────────────────────────────────

    def setup_bottom_panel(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        ttk.Button(bottom_frame, text="Save Selections",
                   command=self.save_selections).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Cancel",
                   command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
        self.stats_var = tk.StringVar(value=self._compute_stats_str())
        ttk.Label(bottom_frame, textvariable=self.stats_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)

    # ------------------------------------------------------------------
    # Panel toggle helpers
    # ------------------------------------------------------------------

    def _toggle_panel(self, name):
        """Show or hide a panel based on its BooleanVar."""
        show = self._panel_vars[name].get()
        if name in ('Pair Selection', 'CCG', 'Probe Network'):
            frame_map = {
                'Pair Selection': self._left_frame,
                'CCG':            self._center_frame,
                'Probe Network':  self._right_frame,
            }
            order   = ['Pair Selection', 'CCG', 'Probe Network']
            weights = {'Pair Selection': 0, 'CCG': 1, 'Probe Network': 0}
            frame = frame_map[name]
            if show:
                pos = sum(1 for n in order[:order.index(name)]
                          if self._panel_vars[n].get())
                self._paned.insert(pos, frame, weight=weights[name])
            else:
                self._paned.forget(frame)
        elif name == 'Waveforms':
            self._toggle_waveforms_panel()
        elif name == 'Time Slider':
            self._toggle_time_slider()

    def _toggle_waveforms_panel(self):
        self._waveforms_visible = self._panel_vars['Waveforms'].get()
        if self._waveforms_visible:
            self.wave_frame.pack(
                in_=self._center_frame, side=tk.BOTTOM,
                fill=tk.X, before=self.sig_frame, pady=(2, 0))
            self._draw_waveforms()
        else:
            self.wave_frame.pack_forget()

    def _toggle_time_slider(self):
        show = self._panel_vars['Time Slider'].get()
        if show:
            self.time_slider_frame.pack(
                in_=self._main_frame, side=tk.TOP,
                fill=tk.X, before=self._paned, pady=(0, 4))
            self._ts_init_times()
            self._ts_redraw()
        else:
            self.time_slider_frame.pack_forget()
            self._on_time_slider_clear()

    def _toggle_resolution(self):
        """Toggle between low-res ``_ccg`` and high-res ``_ccg_highres``."""
        if not (hasattr(self.cd, '_ccg_highres') and self.cd._ccg_highres):
            return
        # Resolution change invalidates scale caches and jitter (different bin_size)
        self._pair_scale_cache.clear()
        self._session_scale_cache = None
        self._jitter_cache.clear()
        self._highres_mode = not self._highres_mode
        mode_label = 'highres' if self._highres_mode else 'lowres'
        if hasattr(self, '_res_btn_text'):
            self._res_btn_text.set(f"Res: {mode_label}")
        nd_key = self.key.nd()
        if self._highres_mode:
            self.ccg_data = self.cd._ccg_highres.get(nd_key)
        else:
            self.ccg_data = self.cd._ccg.get(nd_key)
        self._clear_all_png_cache()
        self._custom_ccg_cache.clear()
        self.update_plot()

    def _on_ctrl_e(self):
        var = self._panel_vars.get('Waveforms')
        if var:
            var.set(not var.get())
            self._toggle_panel('Waveforms')

    # ------------------------------------------------------------------
    # Significance helpers
    # ------------------------------------------------------------------

    def _is_significant(self, ref: int, tgt: int, seg: int) -> bool:
        """Return True if (ref, tgt) is significant in segment *seg*.

        When seg == self.n_segments ("All segments"), returns True if
        the pair is significant in ANY real segment.

        Priority: jitter (segment-aware) → pval_corrected + active_alpha
        → stored significant array.
        """
        if seg == self.n_segments:
            return any(self._is_significant(ref, tgt, s)
                       for s in range(self.n_segments))

        j = self.cd._jitter.get(self.key) if hasattr(self.cd, '_jitter') else None
        if j is not None:
            inds = j.ccg_pointer.inds
            if inds is not None:
                if getattr(j.ccg_pointer, 'stored_by_segment', False):
                    mask = ((inds[:, 0] == seg) &
                            (inds[:, -2] == ref) & (inds[:, -1] == tgt))
                else:
                    mask = (inds[:, -2] == ref) & (inds[:, -1] == tgt)
                if mask.any():
                    return bool(j.j_sig[mask].any())
                return False

        cd = self.ccg_data
        if cd is not None and cd.pval_corrected is not None:
            conf = cd.conf
            lb = getattr(conf, 'min_lag_bin', None)
            ub = getattr(conf, 'max_lag_bin', None)
            if lb is not None and ub is not None:
                return bool(
                    cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                    <= self.active_alpha)
        if cd is not None and cd.significant is not None:
            return bool(cd.significant[seg, ref, tgt])
        return False

    # ------------------------------------------------------------------
    # Alpha / normalization callbacks
    # ------------------------------------------------------------------

    def _on_alpha_change(self, val):
        self.active_alpha = round(float(val), 4)
        self._alpha_label.set(f"{self.active_alpha:.3f}")
        if hasattr(self.cd, 'conf') and self.cd.conf is not None:
            self.cd.conf.alpha = self.active_alpha
        if self.current_pair_idx < len(self.all_inds):
            self._update_sig_indicators(self.all_inds[self.current_pair_idx])

    def _on_norm_toggle(self):
        if NormalizeBy is None:
            return
        self.active_norms = {nm for nm, var in self.norm_vars.items()
                             if var.get()}
        # Norms change the y-axis values → invalidate scale caches
        self._pair_scale_cache.clear()
        self._session_scale_cache = None
        if self.current_pair_idx < len(self.all_inds):
            inds = self.all_inds[self.current_pair_idx]
            for seg in range(self.n_segments + 1):
                p = self._png_path(inds, seg)
                if os.path.exists(p):
                    os.remove(p)
        self.update_plot()

    # ------------------------------------------------------------------
    # Same-scale helpers (Task 1)
    # ------------------------------------------------------------------

    def _effective_bin_size(self) -> float:
        """Infer true bin_size from CCG array shape (robust to conf mutation)."""
        conf = self.ccg_data.conf
        n_bins = self.ccg_data.ccg.shape[-1]
        return conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

    def _compute_pair_scale(self, ref: int, tgt: int):
        """Return (ymin, ymax) unified across all segments for this pair."""
        cd = self.ccg_data
        ymin, ymax = 0.0, 0.0
        for seg in range(self.n_segments):
            ccg_raw = cd.ccg[seg, ref, tgt, :]
            null_raw = cd.ccg_null[seg, ref, tgt, :] if cd.ccg_null is not None else None
            ccg, ccg_null = self._apply_norms_to_ccg(ccg_raw, null_raw, ref, tgt, seg)
            ymin = min(ymin, float(ccg.min()))
            ymax = max(ymax, float(ccg.max()))
            if ccg_null is not None:
                ymax = max(ymax, float(ccg_null.max()))
        return (ymin, ymax * 1.1 if ymax > 0 else 1.0)

    def _compute_session_scale(self):
        """Return (ymin, ymax) unified across all pairs and segments in this key."""
        cd = self.ccg_data
        ymin, ymax = 0.0, 0.0
        for ref_tgt in self.all_inds:
            ref, tgt = int(ref_tgt[0]), int(ref_tgt[1])
            for seg in range(self.n_segments):
                ccg_raw = cd.ccg[seg, ref, tgt, :]
                null_raw = (cd.ccg_null[seg, ref, tgt, :]
                            if cd.ccg_null is not None else None)
                ccg, ccg_null = self._apply_norms_to_ccg(
                    ccg_raw, null_raw, ref, tgt, seg)
                ymin = min(ymin, float(ccg.min()))
                ymax = max(ymax, float(ccg.max()))
                if ccg_null is not None:
                    ymax = max(ymax, float(ccg_null.max()))
        return (ymin, ymax * 1.1 if ymax > 0 else 1.0)

    def _get_current_scale_ylim(self, ref: int, tgt: int):
        """Return (ymin, ymax) for the active scale mode, or None."""
        if self._same_scale_mode == 'pair':
            if (ref, tgt) not in self._pair_scale_cache:
                self._pair_scale_cache[(ref, tgt)] = self._compute_pair_scale(ref, tgt)
            return self._pair_scale_cache[(ref, tgt)]
        if self._same_scale_mode == 'session':
            if self._session_scale_cache is None:
                self._session_scale_cache = self._compute_session_scale()
            return self._session_scale_cache
        return None

    def _on_pair_scale_toggle(self):
        if self._pair_scale_var.get():
            self._same_scale_mode = 'pair'
            self._sess_scale_var.set(False)   # mutual exclusion
        else:
            self._same_scale_mode = None
        self._pair_scale_cache.clear()
        self._clear_all_png_cache()
        self.update_plot()

    def _on_session_scale_toggle(self):
        if self._sess_scale_var.get():
            self._same_scale_mode = 'session'
            self._pair_scale_var.set(False)   # mutual exclusion
            self._session_scale_cache = None  # force recompute
        else:
            self._same_scale_mode = None
        self._clear_all_png_cache()
        self.update_plot()

    # ------------------------------------------------------------------
    # On-demand jitter (Task 2)
    # ------------------------------------------------------------------

    def _run_jitter_for_pair(self, ref: int, tgt: int, njitter: int):
        """Run jitter significance test for a single pair.

        Returns (j_avg [n_bins], j_pval float) or (None, None) on error.
        """
        from neuropy.analyses.jitter import Jitter, JitterConfig
        import copy, types

        if self.neurons is None:
            messagebox.showerror("Jitter", "No neuron data attached.")
            return None, None

        # Build a CCGConfig copy with the correct (possibly inferred) bin_size
        conf = self.ccg_data.conf
        conf_eff = copy.copy(conf)
        conf_eff.bin_size = self._effective_bin_size()
        jconf = JitterConfig(ccg=conf_eff, njitter=njitter)

        # Minimal CCGPointer-like namespace (avoids importing CCGPointer)
        ptr = types.SimpleNamespace(
            inds=np.array([[ref, tgt]]),
            stored_by_segment=False,
            edge_times=self.ccg_pointer.edge_times,
            n_pairs=1,
        )
        try:
            j = Jitter(
                key=self.key,
                neurons=self.neurons,
                conf=jconf,
                ccg_pointer=ptr,
                ccg_data=self.ccg_data,
            )
            j.run()
        except Exception as ex:
            messagebox.showerror("Jitter", f"Jitter failed:\n{ex}")
            return None, None

        j_avg, _, _ = j._j_ccg_cache.get(0, (None, None, None))
        j_pval = float(j.pval[0]) if j.pval is not None and len(j.pval) else None
        return j_avg, j_pval

    def _on_run_jitter(self):
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        njitter = int(self._njitter_var.get())
        self._jitter_btn_text.set(f"Running ({njitter})…")
        self.root.update_idletasks()
        j_avg, j_pval = self._run_jitter_for_pair(ref, tgt, njitter)
        self._jitter_btn_text.set("Run Jitter")
        if j_avg is None:
            return
        self._jitter_cache[(ref, tgt)] = (j_avg, j_pval)
        # Invalidate PNGs for this pair (all segments) so they are rerendered
        for seg in range(self.n_segments + 1):
            p = self._png_path(inds, seg)
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        self.update_plot()

    def _on_clear_jitter(self):
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        self._jitter_cache.pop((ref, tgt), None)
        self._clear_all_png_cache()
        self.update_plot()

    def _finalize_normalization(self):
        if not self.active_norms:
            messagebox.showinfo("Normalize All", "No normalizations are toggled on.")
            return
        if self.neurons is None and any(
                nm in (NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE)
                for nm in self.active_norms):
            messagebox.showerror(
                "Normalize All",
                "Neuron data is unavailable — cannot normalize by firing rate.")
            return
        norm_names = ', '.join(nm.name for nm in self.active_norms)
        if not messagebox.askyesno(
                "Normalize All",
                f"Apply {norm_names} normalization to the stored CCG data?\n\n"
                "This modifies the data in place and cannot be undone."):
            return
        cd = self.ccg_data
        nd_key = self.key.nd()
        edge_times = (self.cd.nd.edge_times[nd_key]
                      if self.cd.nd is not None else None)
        frates = (self.cd.nd.segment_firing_rates[nd_key]
                  if self.cd.nd is not None else None)
        for seg in range(cd.n_segment):
            for nm in self.active_norms:
                if nm == NormalizeBy.REF_FRATE and frates is not None:
                    fr = frates[seg]
                    cd.ccg[seg] /= np.maximum(fr[np.newaxis, :, np.newaxis], 1e-12)
                    if cd.ccg_null is not None:
                        cd.ccg_null[seg] /= np.maximum(
                            fr[np.newaxis, :, np.newaxis], 1e-12)
                elif nm == NormalizeBy.TARGET_FRATE and frates is not None:
                    fr = frates[seg]
                    cd.ccg[seg] /= np.maximum(fr[:, np.newaxis, np.newaxis], 1e-12)
                    if cd.ccg_null is not None:
                        cd.ccg_null[seg] /= np.maximum(
                            fr[:, np.newaxis, np.newaxis], 1e-12)
                elif nm == NormalizeBy.TIME_SPAN and edge_times is not None:
                    et = float(edge_times.iloc[seg]['effective_time_hours'])
                    cd.ccg[seg] /= max(et, 1e-12)
                    if cd.ccg_null is not None:
                        cd.ccg_null[seg] /= max(et, 1e-12)
        existing = list(getattr(cd.conf, 'normalize_methods', []) or [])
        for nm in self.active_norms:
            if nm not in existing:
                existing.append(nm)
        cd.conf.normalize_methods = existing
        self._clear_all_png_cache()
        for var in self.norm_vars.values():
            var.set(False)
        self.active_norms = set()
        self.update_plot()
        messagebox.showinfo("Normalize All",
                            "Normalization applied. Don't forget to Save Selections.")

    def _clear_all_png_cache(self):
        for f in os.listdir(self.tmp_dir):
            if f.endswith('.png'):
                try:
                    os.remove(os.path.join(self.tmp_dir, f))
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Segment filter
    # ------------------------------------------------------------------

    def _apply_seg_filter(self):
        name = self.seg_filter_var.get().strip()
        if not name:
            self.active_segment_filter = None
            self._draw_network()
            return
        try:
            idx = self.segment_names.index(name)
        except ValueError:
            matches = [i for i, n in enumerate(self.segment_names)
                       if n.startswith(name)]
            if len(matches) == 1:
                idx = matches[0]
            else:
                messagebox.showwarning(
                    "Segment filter",
                    f"'{name}' not found:\n" + ', '.join(self.segment_names))
                return
        self.active_segment_filter = idx
        self.current_segment = idx
        self.segment_var.set(self.segment_names[idx])
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()
        self._draw_network()

    def _clear_seg_filter(self):
        self.seg_filter_var.set('')
        self.active_segment_filter = None
        self._draw_network()

    # ------------------------------------------------------------------
    # Sig-chip builder
    # ------------------------------------------------------------------

    def _build_sig_chips(self):
        for widget in self.sig_frame.winfo_children():
            widget.destroy()
        self.seg_sig_labels = []
        ttk.Label(self.sig_frame, text="Segs:").pack(side=tk.LEFT, padx=(4, 2))
        # Real segment chips
        for i, name in enumerate(self.segment_names):
            lbl = tk.Label(
                self.sig_frame, text=name,
                relief=tk.RAISED, font=('Arial', 8),
                bg='#E0E0E0', padx=4, pady=2)
            lbl.pack(side=tk.LEFT, padx=2)
            lbl.bind('<Button-1>', lambda e, idx=i: self._jump_to_segment(idx))
            self.seg_sig_labels.append(lbl)
        # "All" chip (index n_segments)
        lbl_all = tk.Label(
            self.sig_frame, text="All",
            relief=tk.RAISED, font=('Arial', 8, 'bold'),
            bg='#E0E0E0', padx=4, pady=2)
        lbl_all.pack(side=tk.LEFT, padx=2)
        lbl_all.bind('<Button-1>',
                     lambda e: self._jump_to_segment(self.n_segments))
        self.seg_sig_labels.append(lbl_all)

    # ------------------------------------------------------------------
    # Key / dropdown helpers
    # ------------------------------------------------------------------

    def _all_nd_keys(self) -> list:
        seen, seen_str = [], set()
        for k in self.cd.data.keys():
            nk = k.nd()
            s = str(nk)
            if s not in seen_str:
                seen.append(nk)
                seen_str.add(s)
        return seen

    def _session_label(self, nd_key) -> str:
        return str(nd_key.session) if nd_key.session else str(nd_key)

    def _available_type_keys(self, nd_key) -> list:
        nd_session = nd_key.session
        return [k for k in self.cd.data.keys() if k.nd().session == nd_session]

    def _type_label(self, key) -> str:
        parts = []
        if getattr(key, 'excitability', None):
            parts.append(key.excitability)
        if getattr(key, 'conn_type', None):
            ref, tgt = key.conn_type
            parts.append(f"{ref}→{tgt}")
        if getattr(key, 'epoch', None):
            parts.append(f"[{key.epoch}]")
        return ' '.join(parts) if parts else str(key)

    def _switch_key(self, new_key) -> bool:
        ptr = self.cd.data.get(new_key)
        if ptr is None or ptr.inds is None:
            messagebox.showwarning("Switch key", f"No data for key:\n{new_key}")
            return False
        self.key = new_key
        self.ccg_pointer = ptr
        nd_key = new_key.nd()
        if (getattr(self, '_highres_mode', False)
                and hasattr(self.cd, '_ccg_highres')
                and self.cd._ccg_highres.get(nd_key) is not None):
            self.ccg_data = self.cd._ccg_highres[nd_key]
        else:
            self._highres_mode = False   # reset if highres not available
            self.ccg_data = self.cd._ccg.get(nd_key)
        self.neurons = (self.cd.nd.data[new_key.nd()]
                        if getattr(self.cd, 'nd', None) is not None else None)
        self.all_inds = self.ccg_pointer.inds2
        self.n_segments = self.ccg_pointer.n_segments
        self.segment_names = list(self.ccg_pointer.edge_times['label'].values)
        self.current_segment = 0
        self.current_pair_idx = 0
        if (hasattr(self.ccg_pointer, 'manually_selected_inds') and
                self.ccg_pointer.manually_selected_inds is not None):
            self.selected_inds = set(
                map(tuple, self.ccg_pointer.manually_selected_inds))
        else:
            self.selected_inds = set()
        self.unselected_inds = set(map(tuple, self.all_inds)) - self.selected_inds
        self.active_norms = set()
        for var in self.norm_vars.values():
            var.set(False)
        self.active_segment_filter = None
        self._custom_window_active = False
        self._custom_ccg_cache.clear()
        return True

    def _refresh_after_key_switch(self):
        self.segment_combo['values'] = self.segment_names + [_ALL_SEGS]
        self.segment_var.set(self.segment_names[0])
        self._build_sig_chips()
        self.refresh_lists()
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()
        self._draw_network()

    # ------------------------------------------------------------------
    # Dropdown callbacks
    # ------------------------------------------------------------------

    def _on_session_change(self, event):
        idx = self._session_combo.current()
        if idx < 0 or idx >= len(self._nd_keys_list):
            return
        nd_key = self._nd_keys_list[idx]
        type_keys = self._available_type_keys(nd_key)
        self._type_keys_list = type_keys
        type_labels = [self._type_label(k) for k in type_keys]
        self._type_combo['values'] = type_labels
        if not type_keys:
            return
        current_lbl = self._type_label(self.key)
        if current_lbl in type_labels:
            new_key = type_keys[type_labels.index(current_lbl)]
        else:
            new_key = type_keys[0]
            self._type_var.set(type_labels[0])
        if self._switch_key(new_key):
            self._refresh_after_key_switch()

    def _on_type_change(self, event):
        idx = self._type_combo.current()
        if idx < 0 or idx >= len(self._type_keys_list):
            return
        new_key = self._type_keys_list[idx]
        if self._switch_key(new_key):
            self._refresh_after_key_switch()

    def _on_segment_change(self, event):
        name = self.segment_var.get()
        if name == _ALL_SEGS:
            self.current_segment = self.n_segments
        elif name in self.segment_names:
            self.current_segment = self.segment_names.index(name)
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats_str(self) -> str:
        n_sig = len(self.all_inds)
        n_sel = len(self.selected_inds)
        n_poss = None
        if (self.neurons is not None and
                self.key is not None and
                self.key.conn_type is not None):
            ref_t, tgt_t = self.key.conn_type
            try:
                n_ref = self.neurons.get_neuron_type(ref_t).n_neurons
                n_tgt = self.neurons.get_neuron_type(tgt_t).n_neurons
                # Exclude self-self when ref and target types are the same
                n_poss = n_ref * (n_ref - 1) if ref_t == tgt_t else n_ref * n_tgt
            except Exception:
                pass

        s = f"Sig: {n_sig:4d}  Sel: {n_sel:4d}"
        if n_poss is not None and n_poss > 0:
            sel_over_sig  = n_sel / max(n_sig,  1)
            sel_over_poss = n_sel / n_poss
            sig_over_poss = n_sig / n_poss
            s += (f"  Poss: {n_poss:4d}"
                  f"  Sel/Sig: {sel_over_sig:.0%}"
                  f"  Sel/Poss: {sel_over_poss:.0%}"
                  f"  Sig/Poss: {sig_over_poss:.0%}")
        return s

    def _refresh_stats(self):
        if hasattr(self, 'stats_var'):
            self.stats_var.set(self._compute_stats_str())

    # ------------------------------------------------------------------
    # Pair lists
    # ------------------------------------------------------------------

    def refresh_lists(self):
        self.unselected_list.delete(0, tk.END)
        self.selected_list.delete(0, tk.END)
        for inds in sorted(self.unselected_inds):
            self.unselected_list.insert(tk.END, f"[{inds[0]:3d}, {inds[1]:3d}]")
        for inds in sorted(self.selected_inds):
            self.selected_list.insert(tk.END, f"[{inds[0]:3d}, {inds[1]:3d}]")
        self._avail_label_var.set(f"Available ({len(self.unselected_inds)})")
        self._sel_label_var.set(f"Selected ({len(self.selected_inds)})")
        if hasattr(self, '_select_all_btn'):
            self._select_all_btn.config(
                text="Deselect All" if not self.unselected_inds else "Select All")
        self._refresh_stats()

    def move_to_selected(self, event=None):
        sel = self.unselected_list.curselection()
        if not sel:
            return
        sorted_unsel = sorted(self.unselected_inds)
        to_move = [sorted_unsel[i] for i in sel]
        for inds in to_move:
            self.unselected_inds.discard(inds)
            self.selected_inds.add(inds)
        self.refresh_lists()
        last = to_move[-1]
        self.current_pair_idx = self.get_pair_index(last)
        self.update_plot()
        self._draw_network()

    def move_to_unselected(self, event=None):
        sel = self.selected_list.curselection()
        if not sel:
            return
        sorted_sel = sorted(self.selected_inds)
        to_move = [sorted_sel[i] for i in sel]
        for inds in to_move:
            self.selected_inds.discard(inds)
            self.unselected_inds.add(inds)
        self.refresh_lists()
        last = to_move[-1]
        self.current_pair_idx = self.get_pair_index(last)
        self.update_plot()
        self._draw_network()

    def on_pair_select(self, event):
        widget = event.widget
        sel = widget.curselection()
        if not sel:
            return
        # For multi-selection, navigate to the last clicked item
        idx = sel[-1]
        if widget == self.unselected_list:
            inds = sorted(self.unselected_inds)[idx]
        else:
            inds = sorted(self.selected_inds)[idx]
        self.current_pair_idx = self.get_pair_index(inds)
        self.update_plot()
        self._draw_network()

    def _select_all(self):
        """Toggle between Select All and Deselect All."""
        if self.unselected_inds:
            for inds in list(self.unselected_inds):
                self.selected_inds.add(inds)
            self.unselected_inds.clear()
        else:
            for inds in list(self.selected_inds):
                self.unselected_inds.add(inds)
            self.selected_inds.clear()
        self.refresh_lists()
        self._draw_network()

    def _ctx_menu(self, event, widget, action):
        """Right-click context menu for the pair lists."""
        # Select the item under the cursor first
        idx = widget.nearest(event.y)
        if idx >= 0:
            widget.selection_clear(0, tk.END)
            widget.selection_set(idx)
            widget.activate(idx)
        menu = tk.Menu(self.root, tearoff=0)
        if action == 'add':
            menu.add_command(label="Add to Selected",
                             command=self.move_to_selected)
            menu.add_command(label="Select All",
                             command=self._select_all)
        else:
            menu.add_command(label="Remove from Selected",
                             command=self.move_to_unselected)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def get_pair_index(self, inds):
        for i, pair in enumerate(self.all_inds):
            if tuple(pair) == tuple(inds):
                return i
        return 0

    # ------------------------------------------------------------------
    # Segment navigation
    # ------------------------------------------------------------------

    def change_segment(self, delta):
        # Navigate over n_segments real segments + 1 virtual "All segments"
        self.current_segment = (self.current_segment + delta) % (self.n_segments + 1)
        if self.current_segment == self.n_segments:
            self.segment_var.set(_ALL_SEGS)
        else:
            self.segment_var.set(self.segment_names[self.current_segment])
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()

    def _jump_to_segment(self, idx):
        self.current_segment = idx
        if idx == self.n_segments:
            self.segment_var.set(_ALL_SEGS)
        else:
            self.segment_var.set(self.segment_names[idx])
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()

    def _update_sig_indicators(self, inds):
        """Color each segment chip green/gray based on significance."""
        if not hasattr(self, 'seg_sig_labels'):
            return
        ref, tgt = int(inds[0]), int(inds[1])
        cd = self.ccg_data
        conf = cd.conf if cd is not None else None
        lb = getattr(conf, 'min_lag_bin', None)
        ub = getattr(conf, 'max_lag_bin', None)

        for chip_idx, lbl in enumerate(self.seg_sig_labels):
            is_all_chip = (chip_idx == self.n_segments)
            if is_all_chip:
                sig = any(self._is_significant(ref, tgt, s)
                          for s in range(self.n_segments))
                active = (self.current_segment == self.n_segments)
            else:
                seg = chip_idx
                if (cd is not None and cd.pval_corrected is not None and
                        lb is not None and ub is not None):
                    sig = bool(
                        cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                        <= self.active_alpha)
                else:
                    sig = self._is_significant(ref, tgt, seg)
                active = (seg == self.current_segment)

            if sig:
                bg = '#4CAF50' if active else '#90EE90'
            else:
                bg = '#9E9E9E' if active else '#E0E0E0'
            lbl.config(bg=bg)

    def get_plot_title(self):
        if self.current_pair_idx < len(self.all_inds):
            inds = self.all_inds[self.current_pair_idx]
            seg_label = (_ALL_SEGS if self.current_segment == self.n_segments
                         else self.segment_names[self.current_segment])
            return f"Pair [{inds[0]}, {inds[1]}] — {seg_label}"
        return "No pair selected"

    # ------------------------------------------------------------------
    # PNG rendering
    # ------------------------------------------------------------------

    def _png_path(self, inds, segment) -> str:
        seg_name = (_ALL_SEGS.replace(' ', '_') if segment == self.n_segments
                    else self.segment_names[segment])
        norm_key = ('_'.join(sorted(n.name for n in self.active_norms))
                    if self.active_norms else 'raw')
        alpha_key = ''
        if self.ccg_data is not None and self.ccg_data.pval_corrected is not None:
            alpha_key = f'_a{self.active_alpha:.3f}'
        res_key = '_hi' if getattr(self, '_highres_mode', False) else '_lo'
        scale_key = {'pair': '_ssp', 'session': '_sss'}.get(
            getattr(self, '_same_scale_mode', None), '')
        j_key = '_j' if self._jitter_cache.get(
            (int(inds[0]), int(inds[1]))) is not None else ''
        return os.path.join(
            self.tmp_dir,
            f"pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
            f"{alpha_key}{res_key}{scale_key}{j_key}.png")

    def _apply_norms_to_ccg(self, ccg_raw, ccg_null_raw, ref: int, tgt: int,
                             seg: int):
        """Return (ccg, ccg_null) with active normalizations applied (copies)."""
        if not self.active_norms or NormalizeBy is None:
            return ccg_raw, ccg_null_raw
        ccg = ccg_raw.copy().astype(float)
        ccg_null = (ccg_null_raw.copy().astype(float)
                    if ccg_null_raw is not None else None)
        nd_key = self.key.nd()
        if NormalizeBy.REF_FRATE in self.active_norms and self.neurons is not None:
            fr = float(self.neurons.firing_rate[ref])
            ccg /= max(fr, 1e-12)
            if ccg_null is not None:
                ccg_null /= max(fr, 1e-12)
        if NormalizeBy.TARGET_FRATE in self.active_norms and self.neurons is not None:
            fr = float(self.neurons.firing_rate[tgt])
            ccg /= max(fr, 1e-12)
            if ccg_null is not None:
                ccg_null /= max(fr, 1e-12)
        if (NormalizeBy.TIME_SPAN in self.active_norms and
                self.cd.nd is not None and
                seg != self.n_segments):
            et = float(self.cd.nd.edge_times[nd_key].iloc[seg]
                       ['effective_time_hours'])
            ccg /= max(et, 1e-12)
            if ccg_null is not None:
                ccg_null /= max(et, 1e-12)
        return ccg, ccg_null

    def _render_png(self, inds, segment) -> str:
        ref, tgt = int(inds[0]), int(inds[1])
        cd = self.ccg_data
        conf = cd.conf

        is_all = (segment == self.n_segments)
        if is_all:
            ccg_raw = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            ccg_null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                            if cd.ccg_null is not None else None)
            pval_arg = None
            pval_c_arg = None
            seg_label = _ALL_SEGS
        else:
            loc = (segment, ref, tgt)
            ccg_raw = cd.ccg[loc]
            ccg_null_raw = cd.ccg_null[loc] if cd.ccg_null is not None else None
            pval_arg = cd.pval[loc] if cd.pval is not None else None
            pval_c_arg = (cd.pval_corrected[loc]
                          if cd.pval_corrected is not None else None)
            seg_label = self.segment_names[segment]

        ccg, ccg_null = self._apply_norms_to_ccg(
            ccg_raw, ccg_null_raw, ref, tgt, segment)

        norm_info = (', '.join(nm.name for nm in self.active_norms)
                     if self.active_norms and NormalizeBy is not None else None)

        # Infer bin_size from the actual CCG length so rendering is
        # correct even if conf.bin_size was mutated (e.g. by load_highres).
        n_bins = len(ccg)
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

        # Jitter overlay for this pair (if computed)
        j_data = self._jitter_cache.get((ref, tgt))
        j_ccg_arg = j_data[0] if j_data is not None else None
        j_pval_arg = j_data[1] if j_data is not None else None

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg, ids=inds, inds=inds,
            window_size=conf.duration, bin_size=bin_size_eff,
            pval=pval_arg, pval_corrected=pval_c_arg,
            alpha=self.active_alpha, ccg_null=ccg_null,
            j_ccg=j_ccg_arg, j_pval=j_pval_arg,
            segment_id=seg_label,
            is_significant_pair=self._is_significant(ref, tgt, segment),
            min_lag=conf.min_lag, max_lag=conf.max_lag,
            normalize_info=norm_info,
        )
        # Same-scale y-axis override
        ylim = self._get_current_scale_ylim(ref, tgt)
        if ylim is not None:
            ax.set_ylim(ylim)

        png_path = self._png_path(inds, segment)
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return png_path

    # ------------------------------------------------------------------
    # Plot update
    # ------------------------------------------------------------------

    def _deferred_initial_draw(self):
        self.update_plot()

    def update_plot(self):
        try:
            if self.current_pair_idx >= len(self.all_inds):
                return

            # Custom time-window view overrides regular rendering
            if getattr(self, '_custom_window_active', False):
                self._render_custom_ccg()
                self.plot_title_var.set(
                    self.get_plot_title() + " [custom window]")
                self._update_sig_indicators(self.all_inds[self.current_pair_idx])
                self._draw_waveforms()
                return

            inds = self.all_inds[self.current_pair_idx]
            png_path = self._png_path(inds, self.current_segment)
            if not os.path.exists(png_path):
                png_path = self._render_png(inds, self.current_segment)

            img = mpimg.imread(png_path)
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            self.fig.tight_layout(pad=0)
            self.canvas.draw()

            self.plot_title_var.set(self.get_plot_title())
            self._update_sig_indicators(inds)
            self._draw_waveforms()

        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Waveforms
    # ------------------------------------------------------------------

    def _draw_waveforms(self):
        if not self._waveforms_visible or self.neurons is None:
            return
        if self.current_pair_idx >= len(self.all_inds):
            return
        from neuropy.plotting.probe import plot_waveform_on_channel
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        wf = getattr(self.neurons, 'waveforms', None)
        shank_ids = getattr(self.neurons, 'shank_ids', None)
        self.wave_ax.clear()
        if wf is None or shank_ids is None:
            self.wave_ax.text(0.5, 0.5, "No waveform data",
                              ha='center', va='center',
                              transform=self.wave_ax.transAxes, fontsize=8)
            self.wave_ax.axis('off')
        else:
            # Resolve probe geometry from NeuronsDatasetConfig so discarded
            # channels are mapped correctly (mirrors plot_ccg_figure logic).
            nd_conf = getattr(getattr(self.cd, 'nd', None), '_conf', None)
            ch_per_shank = getattr(nd_conf, 'ch_per_shank', 16)
            recinfo = getattr(nd_conf, 'recinfo', None)
            skipped = getattr(recinfo, "skipped_channels", None)
            discarded = None if skipped is None else np.asarray(skipped, dtype=int)
            
            def get_filled_waveform(shank_id, wf_neuron):
                """Map compact waveform to full (ch_per_shank, n_samples) grid."""
                if wf_neuron.ndim == 1:
                    return np.tile(wf_neuron, (ch_per_shank, 1))
                channel_ids = ch_per_shank * shank_id + np.arange(ch_per_shank)
                mask = ~np.isin(channel_ids, discarded)
                start = int(ch_per_shank * shank_id
                            - np.sum(discarded < ch_per_shank * shank_id))
                length = int(np.sum(mask))
                clean = np.full((ch_per_shank, wf_neuron.shape[-1]), np.nan)
                clean[mask] = wf_neuron[start:start + length]
                return clean

            ref_shank = int(shank_ids[ref])
            tgt_shank = int(shank_ids[tgt])
            ref_waveform = get_filled_waveform(ref_shank, wf[ref])
            tgt_waveform = get_filled_waveform(tgt_shank, wf[tgt])
            color = 'green' if ref_shank != tgt_shank else 'orange'
            plot_waveform_on_channel(
                ref_waveform=ref_waveform,
                ref_shank=ref_shank,
                target_waveform=tgt_waveform,
                target_shank=tgt_shank,
                color=color,
                amplitude_limit=True,
                ax=self.wave_ax,
            )
        self.wave_canvas.draw()

    # ------------------------------------------------------------------
    # Time slider
    # ------------------------------------------------------------------

    def _ts_init_times(self):
        """Populate epoch bounds from edge_times; detect column names."""
        et = self.ccg_pointer.edge_times
        cols = et.columns.tolist()

        # Find start/stop columns by common name conventions
        def _find_col(candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None

        start_col = _find_col(['start', 't_start', 'start_time', 'start_s'])
        stop_col  = _find_col(['stop',  't_end',   'end_time',   'stop_s', 'end'])

        self._ts_epoch_bounds = []
        if start_col and stop_col:
            for _, row in et.iterrows():
                t0 = float(row[start_col])
                t1 = float(row[stop_col])
                self._ts_epoch_bounds.append((t0, t1, str(row['label'])))
            self._ts_total_sec = (self._ts_epoch_bounds[-1][1]
                                  if self._ts_epoch_bounds else 1.0)
        else:
            # Fall back: reconstruct from cumulative effective_time_hours
            t = 0.0
            for _, row in et.iterrows():
                dur = float(row['effective_time_hours']) * 3600.0
                self._ts_epoch_bounds.append((t, t + dur, str(row['label'])))
                t += dur
            self._ts_total_sec = t if t > 0 else 1.0

    def _ts_t_to_x(self, t: float) -> int:
        w = max(self.ts_canvas.winfo_width(), 20)
        return int((t / max(self._ts_total_sec, 1)) * (w - 20) + 10)

    def _ts_x_to_t(self, x: int) -> float:
        w = max(self.ts_canvas.winfo_width(), 20)
        return max(0.0, min(self._ts_total_sec,
                            (x - 10) / max(w - 20, 1) * self._ts_total_sec))

    def _ts_redraw(self, event=None):
        c = self.ts_canvas
        c.delete('all')
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or not self._ts_epoch_bounds:
            return
        colors = ['#BBDEFB', '#C8E6C9', '#FFF9C4', '#FFE0B2', '#E1BEE7',
                  '#F8BBD0', '#D7CCC8']
        for i, (t0, t1, lbl) in enumerate(self._ts_epoch_bounds):
            x0, x1 = self._ts_t_to_x(t0), self._ts_t_to_x(t1)
            c.create_rectangle(x0, 8, x1, h - 8,
                                fill=colors[i % len(colors)],
                                outline='#90A4AE')
            if x1 - x0 > 22:
                c.create_text((x0 + x1) // 2, h // 2,
                              text=lbl, font=('Arial', 7), fill='#333')
        # Handles
        for t, color in [(self._slider_t_start, '#1565C0'),
                         (self._slider_t_end,   '#B71C1C')]:
            if t is not None:
                x = self._ts_t_to_x(t)
                c.create_line(x, 2, x, h - 2, fill=color, width=2)
                c.create_polygon(x - 5, 2, x + 5, 2, x, 10,
                                 fill=color, outline='')
        # Selected range shading
        if self._slider_t_start is not None and self._slider_t_end is not None:
            x0 = self._ts_t_to_x(self._slider_t_start)
            x1 = self._ts_t_to_x(self._slider_t_end)
            c.create_rectangle(x0, 8, x1, h - 8,
                                fill='', outline='#1565C0',
                                width=2, dash=(4, 2))

    def _ts_mouse_press(self, event):
        if self._slider_t_start is None:
            self._slider_dragging = 'start'
        elif self._slider_t_end is None:
            self._slider_dragging = 'end'
        else:
            xs = self._ts_t_to_x(self._slider_t_start)
            xe = self._ts_t_to_x(self._slider_t_end)
            self._slider_dragging = ('start'
                                     if abs(event.x - xs) <= abs(event.x - xe)
                                     else 'end')
        self._ts_update_handle(event.x)

    def _ts_mouse_drag(self, event):
        self._ts_update_handle(event.x)

    def _ts_mouse_release(self, event):
        self._ts_update_handle(event.x, snap=True)
        self._slider_dragging = None

    def _ts_update_handle(self, canvas_x: int, snap: bool = False):
        t = self._ts_x_to_t(canvas_x)
        if snap and self._ts_epoch_bounds:
            bounds_t = [b for (t0, t1, _) in self._ts_epoch_bounds
                        for b in (t0, t1)]
            for bt in bounds_t:
                if abs(self._ts_t_to_x(bt) - canvas_x) <= 25:
                    t = bt
                    break
        if self._slider_dragging == 'start':
            cap = self._slider_t_end if self._slider_t_end is not None else self._ts_total_sec
            self._slider_t_start = min(t, cap)
            self._ts_start_var.set(self._ts_sec_to_hms(self._slider_t_start))
        elif self._slider_dragging == 'end':
            floor = self._slider_t_start if self._slider_t_start is not None else 0.0
            self._slider_t_end = max(t, floor)
            self._ts_end_var.set(self._ts_sec_to_hms(self._slider_t_end))
        self._ts_redraw()

    @staticmethod
    def _ts_hms_to_sec(hms: str) -> float:
        parts = hms.strip().split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    @staticmethod
    def _ts_sec_to_hms(sec: float) -> str:
        sec = int(sec)
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    def _on_time_slider_set(self):
        try:
            t0 = self._ts_hms_to_sec(self._ts_start_var.get())
            t1 = self._ts_hms_to_sec(self._ts_end_var.get())
        except (ValueError, IndexError):
            messagebox.showerror("Time window",
                                 "Invalid time format. Use HH:MM:SS.")
            return
        if t1 <= t0:
            messagebox.showerror("Time window",
                                 "End time must be after start time.")
            return
        self._slider_t_start = t0
        self._slider_t_end = t1
        self._custom_window_active = True
        self._custom_ccg_cache.clear()
        self._ts_status_var.set(
            f"Active: {self._ts_sec_to_hms(t0)} – {self._ts_sec_to_hms(t1)}")
        self._ts_redraw()
        self.update_plot()

    def _on_time_slider_clear(self):
        self._custom_window_active = False
        self._custom_ccg_cache.clear()
        if hasattr(self, '_ts_status_var'):
            self._ts_status_var.set("")
        self.update_plot()

    # ------------------------------------------------------------------
    # Custom-window CCG
    # ------------------------------------------------------------------

    def _compute_custom_ccg(self, ref: int, tgt: int):
        """Compute CCG for the custom time window; returns 1-D array or None."""
        if self.neurons is None:
            messagebox.showerror("Custom CCG", "No neuron data available.")
            return None
        t0, t1 = self._slider_t_start, self._slider_t_end
        if t0 is None or t1 is None or t1 <= t0:
            return None
        try:
            from neuropy.analyses.correlations import (
                np_spike_correlations_2groups)
            neurons_slice = self.neurons.time_slice(t0, t1)
            conf = self.ccg_data.conf
            ccg = np_spike_correlations_2groups(
                neurons=neurons_slice,
                ref_inds=[ref],
                target_inds=[tgt],
                bin_size=conf.bin_size,
                window_size=conf.duration,
                symmetrize=False,
            )   # shape (1, 1, n_bins)
            return ccg[0, 0, :]
        except Exception as ex:
            messagebox.showerror("Custom CCG",
                                 f"Error computing CCG:\n{ex}")
            return None

    def _render_custom_ccg(self):
        """Render custom-window CCG directly into self.fig (no PNG cache)."""
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        cache_key = (ref, tgt)
        if cache_key not in self._custom_ccg_cache:
            data = self._compute_custom_ccg(ref, tgt)
            if data is None:
                return
            self._custom_ccg_cache[cache_key] = data
        ccg = self._custom_ccg_cache[cache_key]
        conf = self.ccg_data.conf
        seg_id = (f"custom [{self._ts_sec_to_hms(self._slider_t_start)}"
                  f" – {self._ts_sec_to_hms(self._slider_t_end)}]")
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg, ids=inds, inds=inds,
            window_size=conf.duration, bin_size=conf.bin_size,
            pval=None, pval_corrected=None, alpha=self.active_alpha,
            ccg_null=None, segment_id=seg_id,
            is_significant_pair=None,
            min_lag=conf.min_lag, max_lag=conf.max_lag,
            normalize_info="Custom window",
        )
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Probe network
    # ------------------------------------------------------------------

    def _get_neuron_positions(self):
        neurons = self.neurons
        if neurons is None:
            return None
        if neurons.shank_ids is None or neurons.peak_channels is None:
            return None
        ch_per_shank = 16
        nd_conf = getattr(getattr(self.cd, 'nd', None), 'conf', None)
        if nd_conf is not None and hasattr(nd_conf, 'ch_per_shank'):
            ch_per_shank = nd_conf.ch_per_shank
        x = np.asarray(neurons.shank_ids, dtype=float) * 150.0
        y = (np.asarray(neurons.peak_channels, dtype=float) % ch_per_shank) * 20.0
        return x, y

    def _pairs_for_segment_filter(self):
        if self.active_segment_filter is None:
            return set(map(tuple, self.all_inds))
        pt = self.ccg_pointer
        if pt.stored_by_segment:
            seg_i = self.active_segment_filter
            mask = pt.inds[:, 0] == seg_i
            return set(map(tuple, pt.inds[mask, -2:]))
        return set(map(tuple, self.all_inds))

    def _draw_network(self):
        ax = self.net_ax
        ax.clear()
        pos = self._get_neuron_positions()
        if pos is None:
            ax.text(0.5, 0.5, "No probe\nposition data",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.axis('off')
            self.net_canvas.draw()
            return
        x_pos, y_pos = pos
        ei = getattr(self.key, 'excitability', 'E')
        base_color = '#E57373' if ei == 'E' else '#64B5F6'
        visible_pairs = self._pairs_for_segment_filter()
        all_pair_arr = self.ccg_pointer.inds[:, -2:]
        involved = np.unique(all_pair_arr)
        types = (self.neurons.neuron_type[involved]
                 if self.neurons is not None and
                 self.neurons.neuron_type is not None else None)
        for idx in involved:
            ntype = (types[np.where(involved == idx)[0][0]]
                     if types is not None else None)
            marker = 's' if ntype == 'inter' else 'o'
            ax.scatter(x_pos[idx], y_pos[idx], s=30, marker=marker,
                       color='#1565C0' if ntype == 'inter' else '#2E7D32',
                       zorder=3, linewidths=0)
        for ref, tgt in sorted(set(map(tuple, self.all_inds))):
            in_filter = (ref, tgt) in visible_pairs
            is_selected = (ref, tgt) in self.selected_inds
            if not in_filter:
                ax.plot([x_pos[ref], x_pos[tgt]], [y_pos[ref], y_pos[tgt]],
                        color='#EEEEEE', lw=0.5, alpha=0.3, zorder=1)
                continue
            color = base_color if is_selected else '#BDBDBD'
            lw    = 2.0 if is_selected else 0.8
            alpha = 0.85 if is_selected else 0.45
            line, = ax.plot(
                [x_pos[ref], x_pos[tgt]], [y_pos[ref], y_pos[tgt]],
                color=color, lw=lw, alpha=alpha, picker=6, zorder=2)
            line.set_gid(f"{ref}_{tgt}")
        if self.current_pair_idx < len(self.all_inds):
            ref, tgt = self.all_inds[self.current_pair_idx]
            ax.scatter([x_pos[ref], x_pos[tgt]],
                       [y_pos[ref], y_pos[tgt]],
                       s=90, zorder=5, edgecolors='black',
                       facecolors=base_color, linewidths=1.5)
            ax.plot([x_pos[ref], x_pos[tgt]], [y_pos[ref], y_pos[tgt]],
                    color=base_color, lw=2.5, zorder=4)
        ax.axis('off')
        ax.set_aspect('equal')
        self.net_fig.tight_layout(pad=0.5)
        self.net_canvas.draw()

    def _on_network_pick(self, event):
        gid = getattr(event.artist, 'get_gid', lambda: None)()
        if not gid:
            return
        try:
            ref, tgt = map(int, gid.split('_'))
        except (ValueError, AttributeError):
            return
        self.current_pair_idx = self.get_pair_index((ref, tgt))
        self.update_plot()
        self._draw_network()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_selections(self):
        selected_array = np.array(sorted(self.selected_inds))
        self.ccg_pointer.manually_selected_inds = selected_array
        if not hasattr(self.cd, 'manual_selections'):
            self.cd.manual_selections = {}
        self.cd.manual_selections[self.key] = self.ccg_pointer
        print(f"Saved {len(selected_array)} manually selected pairs for {self.key}")
        self.root.destroy()

    def save_gifs(self):
        """Generate per-pair animated GIFs cycling over segments.

        Kept separate from save_selections so it can be called explicitly
        after all PNGs have been rendered at a consistent figure size.
        Each frame is resized to the shape of the first frame so imageio
        can stack them into a GIF.
        """
        from datetime import datetime
        selected_array = np.array(sorted(self.selected_inds))
        if self.n_segments <= 1 or len(selected_array) == 0:
            print("Nothing to GIF (need >1 segment and ≥1 selected pair).")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_folder = os.path.join(self.tmp_dir, f"review_{timestamp}")
        os.makedirs(gif_folder, exist_ok=True)
        print(f"Generating GIFs for {len(selected_array)} selected pairs...")
        for inds in selected_array:
            raw_frames = []
            for seg in range(self.n_segments):
                png_path = self._png_path(inds, seg)
                if not os.path.exists(png_path):
                    png_path = self._render_png(inds, seg)
                raw_frames.append(mpimg.imread(png_path))
            # Normalise to uint8 and resize to first frame's shape
            h, w = raw_frames[0].shape[:2]
            frames_u8 = []
            for f in raw_frames:
                arr = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
                if arr.shape[:2] != (h, w):
                    from PIL import Image as _Image
                    arr = np.array(
                        _Image.fromarray(arr).resize((w, h), _Image.LANCZOS))
                frames_u8.append(arr)
            gif_path = os.path.join(gif_folder,
                                    f"pair_{inds[0]}_{inds[1]}.gif")
            imageio.mimsave(gif_path, frames_u8, duration=0.8)
        print(f"GIFs saved to: {gif_folder}")
        return gif_folder

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------

def launch_ccg_review(cd, key):
    """
    Launch CCG review UI.

    Parameters
    ----------
    cd : CCGDataset
        Dataset to review.  Call ``cd.load_highres()`` beforehand to enable
        the resolution toggle button.
    key : Key
        Key identifying which CCGPointer to review.

    Examples
    --------
    >>> ui = launch_ccg_review(cd, key)
    >>> cd.load_highres(); ui = launch_ccg_review(cd, key)
    """
    ui = CCGReviewUI(cd, key)
    ui.run()
    return ui
