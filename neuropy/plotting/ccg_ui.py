"""
CCG Manual Review UI

Interactive GUI for reviewing and selecting significant CCG pairs.

Layout (3-column PanedWindow):
  Left   — pair selection lists + alpha control
  Center — CCG plot with normalization toggles, segment navigation, optional
            waveforms sub-panel
  Right  — probe network (neuron positions + connection edges)

Top bar:  menubar (Panels / Groups / Selections menus) + tool strip
Time slider: full-width panel above the 3-column area, hidden by default.
Bottom bar: pair statistics + Save / Cancel buttons.

Keyboard shortcuts
------------------
  ←  /  →      previous / next segment
  Ctrl+R        toggle lo-res ↔ hi-res
  Ctrl+E        toggle waveforms sub-panel
  m             move current pair between Available / Selected
  Ctrl+S        save selection + export groups
  Ctrl+1..0     assign / jump to group 1–10
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import json
import datetime
from neuropy.plotting import ccg as plot_ccg
import imageio

try:
    from neuropy.analyses.ms_connectivity import NormalizeBy
except ImportError:
    NormalizeBy = None

# Sentinel value for the virtual "All segments" view
_ALL_SEGS = "All segments"
_ADMITTED_GROUP = "__admitted__"


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

        if self.ccg_pointer is None:
            raise ValueError(f"No CCG data found for key: {key}")

        # Neurons (normalization + network + waveforms)
        self.neurons = (self.cd.nd.data[key.nd()]
                        if getattr(self.cd, 'nd', None) is not None
                        else None)

        # Group state  {group_name -> set((ref,tgt))}
        # Initialized early because all_inds @property reads _groups
        self._groups: dict = {}
        self._groups.setdefault(_ADMITTED_GROUP, set())
        self._group_hotkeys: dict = {}       # group_name -> hotkey str e.g. 'Control-1'
        self._group_notes: dict = {}         # group_name -> notes string

        # Pair / segment state  (all_inds is a @property — see below)
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

        # Undo/redo stack for pair selection changes
        self._undo_stack: list = []  # list of (selected_inds_copy, unselected_inds_copy)
        self._redo_stack: list = []
        self._UNDO_LIMIT = 30

        # Active config
        self.active_alpha = getattr(getattr(self.cd, 'conf', None), 'alpha', 0.05)
        self.active_norms: set = set()
        self.norm_vars: dict = {}
        self.active_segment_filter = None

        # Resolution state
        self._highres_mode = False           # True when showing _ccg_highres

        # Same-scale state
        self._same_scale_mode: str = None    # None | 'pair' | 'session'
        self._pair_scale_cache: dict = {}    # (ref, tgt) -> (ymin, ymax)
        self._session_scale_cache = None     # (ymin, ymax)

        # Jitter state
        self._jitter_cache: dict = {}        # (ref, tgt) -> (j_avg, j_pval)

        # Significance display toggles (what to show on CCG panel)
        self._sig_show_conv_baseline = True   # convolution baseline (ccg_null)
        self._sig_show_conv_p = False         # convolution p-value line
        self._sig_show_conv_pc = False        # convolution corrected-p line
        self._sig_show_test_window = True     # green test-window highlight
        self._sig_show_jitter_p = False       # jitter p-value line
        self._sig_show_jitter_pc = False      # jitter corrected-p (horizontal line)

        # Double-click debounce
        self._select_after: int = None       # after() id for deferred pair update

        # Probe-network neuron/pair focus
        self._focused_neuron: int = None     # neuron index to highlight (None = off)
        self._focused_pair: tuple = None     # (ref, tgt) pair to highlight (None = off)
        self._net_show_arrows: bool = True   # show/hide connection arrows
        self._net_hide_unconnected: bool = False  # hide neurons with no connections
        self._net_group_filter: str = None       # group name shown in probe network

        # Versioned selections save dir
        self._sel_save_dir = os.path.expanduser(
            "~/Documents/ms_synchrony/NeuroPy/data/selections")
        os.makedirs(self._sel_save_dir, exist_ok=True)

        # Panel visibility state
        self._panel_vars: dict = {}          # populated in setup_panels_menu
        self._waveforms_visible = False

        # Time-slider / custom-segment state
        self._slider_t_start: float = None
        self._slider_t_end: float = None
        self._slider_dragging: str = None    # 'start' | 'end' | None
        self._ts_epoch_bounds = []           # [(t0_sec, t1_sec, label), …]
        self._ts_total_sec: float = 0.0
        self._ts_segment_name: str = ""      # name for the next custom segment
        # Custom segments: each entry =
        #   {'name':str, 't0':float, 't1':float,
        #    'ccg': [1,N,N,bins], 'ccg_null', 'pval', 'pval_corrected',
        #    (optional hi-res): 'ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'}
        self._custom_segments: list = []

        # PNG cache
        self.tmp_dir = os.path.expanduser(
            "~/Documents/ms_synchrony/NeuroPy/images/tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Build UI
        self.root = tk.Tk()
        self.root.title("CCG Manual Review")
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.setup_ui()

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def all_inds(self):
        """Significant pairs + manually admitted pairs, as Nx2 numpy array."""
        base = self.ccg_pointer.inds2
        admitted = self._groups.get(_ADMITTED_GROUP, set())
        if not admitted:
            return base
        base_set = set(map(tuple, base))
        extra = sorted(admitted - base_set)
        if not extra:
            return base
        return np.vstack([base, np.array(extra, dtype=base.dtype)])

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.root.geometry("1800x950")

        # ── Menubar ────────────────────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.setup_panels_menu(menubar)
        self.setup_groups_menu(menubar)
        self.setup_file_menu(menubar)

        # ── Tool-strip row ─────────────────────────────────────────────
        self.setup_menu()

        # ── Group hotkeys bar (below tool-strip, hidden by default) ────
        self.setup_group_hotkeys_bar()

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
        for _key in ('<Control-s>', '<Command-s>'):
            self.root.bind(_key, lambda e: self._quick_save())
        for _key in ('<Control-z>', '<Command-z>'):
            self.root.bind(_key, self._undo)
        for _key in ('<Control-y>', '<Command-y>',
                      '<Control-Shift-z>', '<Command-Shift-z>',
                      '<Control-Shift-Z>', '<Command-Shift-Z>'):
            self.root.bind(_key, self._redo)
        # 'm' key moves current pair between Available / Selected
        # Guard: don't fire when typing in an Entry or Spinbox widget
        def _m_key_handler(e):
            if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
                return
            self._move_current_pair()
        self.root.bind('<m>', _m_key_handler)
        # Ctrl/Cmd+1..0 for groups — use global KeyPress handler because
        # macOS Tkinter doesn't deliver <Command-digit> events reliably.
        def _global_key_handler(e):
            if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox,
                                     tk.Text)):
                return
            # Check for Ctrl (bit 2) or Cmd (bit 3) modifier
            has_mod = e.state & (0x4 | 0x8)
            if not has_mod:
                return
            if e.keysym in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                self._group_hotkey_handler(int(e.keysym) - 1)
            elif e.keysym == '0':
                self._group_hotkey_handler(9)
        self.root.bind('<KeyPress>', _global_key_handler)

    # ── Menubar ────────────────────────────────────────────────────────

    def setup_panels_menu(self, menubar):
        """Panels menu with checkbuttons for each panel."""
        panels_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Panels", menu=panels_menu)
        panel_defaults = [
            ('Pair Selection',  True),
            ('CCG',             True),
            ('Probe Network',   True),
            ('Waveforms',       False),
            ('Time Slider',     False),
            ('Group Hotkeys',   True),
        ]
        for name, default in panel_defaults:
            var = tk.BooleanVar(value=default)
            self._panel_vars[name] = var
            panels_menu.add_checkbutton(
                label=name, variable=var,
                command=lambda n=name: self._toggle_panel(n))

    def setup_groups_menu(self, menubar):
        """Groups menu: create / manage pair groups."""
        self._groups_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Groups", menu=self._groups_menu)
        self._groups_menu.add_command(label="Create group…",
                                      command=self._create_group_dialog)
        self._groups_menu.add_command(label="Manage groups…",
                                      command=self._manage_groups_dialog)
        self._groups_menu.add_command(label="Merge groups…",
                                      command=self._merge_groups_dialog)
        self._groups_menu.add_command(label="Export groups…",
                                      command=self._export_groups)
        self._groups_menu.add_command(label="Import groups…",
                                      command=self._import_groups)
        self._groups_menu.add_separator()
        # Dynamic group entries added in _rebuild_groups_menu()

    def setup_file_menu(self, menubar):
        """Selections menu: save / load selection versions."""
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Selections", menu=file_menu)
        file_menu.add_command(label="Save selection…",
                              command=self._quick_save)
        file_menu.add_command(label="Load selection…",
                              command=self._load_selection_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.root.destroy)

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

    # ── Group hotkeys bar ──────────────────────────────────────────────

    def setup_group_hotkeys_bar(self):
        """Horizontal bar showing Ctrl+1…0 → group-name mappings."""
        self._hotkeys_bar = ttk.Frame(self.root, relief=tk.GROOVE, borderwidth=1)
        self._hotkeys_bar_labels: list[tk.Label] = []
        self._refresh_hotkeys_bar()
        # Pack immediately if default is visible
        if self._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get():
            self._hotkeys_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2))

    def _refresh_hotkeys_bar(self):
        """Rebuild the labels inside the hotkeys bar."""
        for w in self._hotkeys_bar.winfo_children():
            w.destroy()
        self._hotkeys_bar_labels.clear()

        ttk.Label(self._hotkeys_bar, text="Groups:",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(6, 4))

        # Build ordered slots: Ctrl+1 … Ctrl+9, Ctrl+0
        slot_order = [str(i) for i in range(1, 10)] + ['0']
        # Invert: hotkey_str → group_name
        hk_to_group = {v: k for k, v in self._group_hotkeys.items()}

        # Current session pairs for counting
        current_pairs = set(map(tuple, self.all_inds)) if len(self.all_inds) else set()

        for key_str in slot_order:
            gname = hk_to_group.get(key_str)
            if gname is None:
                continue
            # Count only pairs belonging to current session/type
            n_in_session = len(self._groups.get(gname, set()) & current_pairs)
            display = f"⌘{key_str}: {gname} ({n_in_session})"
            lbl = tk.Label(self._hotkeys_bar, text=display,
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1)
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            # Click label → jump to group
            lbl.bind('<Button-1>',
                     lambda e, g=gname: self._select_group(g))
            self._hotkeys_bar_labels.append(lbl)

        if not hk_to_group:
            ttk.Label(self._hotkeys_bar, text="(no hotkeys assigned)",
                      font=('Arial', 9), foreground='#888').pack(
                side=tk.LEFT, padx=4)

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
            selectmode=tk.EXTENDED, font=('Courier', 9), activestyle='none')
        self.unselected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        unsel_scroll.config(command=self.unselected_list.yview)
        # Single click → navigate (debounced); double-click → move
        self.unselected_list.bind('<ButtonRelease-1>', self.on_pair_select)
        self.unselected_list.bind('<Double-Button-1>', self.move_to_selected)
        self.unselected_list.bind('<Return>',          self.move_to_selected)
        # Bind right-click (Button-3) and macOS two-finger/ctrl-click (Button-2)
        self.unselected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))
        self.unselected_list.bind('<Button-2>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))
        self.unselected_list.bind('<KeyRelease-Up>',   self._on_arrow_key)
        self.unselected_list.bind('<KeyRelease-Down>', self._on_arrow_key)

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
            selectmode=tk.EXTENDED, font=('Courier', 9), activestyle='none')
        self.selected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scroll.config(command=self.selected_list.yview)
        self.selected_list.bind('<ButtonRelease-1>', self.on_pair_select)
        self.selected_list.bind('<Double-Button-1>', self.move_to_unselected)
        self.selected_list.bind('<Return>',          self.move_to_unselected)
        self.selected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))
        self.selected_list.bind('<Button-2>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))
        self.selected_list.bind('<KeyRelease-Up>',   self._on_arrow_key)
        self.selected_list.bind('<KeyRelease-Down>', self._on_arrow_key)

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
        self.setup_sig_display_panel(parent)
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

    def setup_sig_display_panel(self, parent):
        """Significance display toggles: conv baseline/p/p-corrected, jitter."""
        sig_frame = ttk.LabelFrame(parent, text="Significance", padding=4)
        sig_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        # Convolution group
        conv_lbl = ttk.Label(sig_frame, text="Convolution:", font=('Arial', 8))
        conv_lbl.pack(side=tk.LEFT, padx=(0, 2))
        self._sig_conv_baseline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sig_frame, text="baseline",
                        variable=self._sig_conv_baseline_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        self._sig_conv_p_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sig_frame, text="p",
                        variable=self._sig_conv_p_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        self._sig_conv_pc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sig_frame, text="p-corrected",
                        variable=self._sig_conv_pc_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        self._sig_test_window_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sig_frame, text="test window",
                        variable=self._sig_test_window_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)

        sep = ttk.Separator(sig_frame, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        # Jitter group
        j_lbl = ttk.Label(sig_frame, text="Jitter:", font=('Arial', 8))
        j_lbl.pack(side=tk.LEFT, padx=(0, 2))
        self._sig_jitter_p_var = tk.BooleanVar(value=False)
        self._sig_jitter_p_cb = ttk.Checkbutton(
            sig_frame, text="p",
            variable=self._sig_jitter_p_var,
            command=self._on_sig_toggle)
        self._sig_jitter_p_cb.pack(side=tk.LEFT, padx=2)
        self._sig_jitter_pc_var = tk.BooleanVar(value=False)
        self._sig_jitter_pc_cb = ttk.Checkbutton(
            sig_frame, text="p-corrected",
            variable=self._sig_jitter_pc_var,
            command=self._on_sig_toggle)
        self._sig_jitter_pc_cb.pack(side=tk.LEFT, padx=2)
        # Disable jitter buttons initially (enabled when jitter data exists)
        self._sig_jitter_p_cb.state(['disabled'])
        self._sig_jitter_pc_cb.state(['disabled'])

        # ── ACG row (separate from significance toggles) ──────────────
        acg_frame = ttk.LabelFrame(parent, text="Auto-correlograms", padding=4)
        acg_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))

        self._acg_ref_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(acg_frame, text="ref",
                        variable=self._acg_ref_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        self._acg_tgt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(acg_frame, text="tgt",
                        variable=self._acg_tgt_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)

        ttk.Separator(acg_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(acg_frame, text="Y scale:", font=('Arial', 8)).pack(
            side=tk.LEFT, padx=(0, 2))
        self._acg_yscale_var = tk.DoubleVar(value=1.0)
        acg_slider = ttk.Scale(acg_frame, from_=0.1, to=1.5,
                               variable=self._acg_yscale_var,
                               orient=tk.HORIZONTAL, length=100,
                               command=lambda v: self._on_acg_scale_change())
        acg_slider.pack(side=tk.LEFT, padx=2)
        self._acg_scale_label = ttk.Label(acg_frame, text="1.0x",
                                           font=('Courier', 8), width=4)
        self._acg_scale_label.pack(side=tk.LEFT, padx=(0, 4))

        self._acg_match_ccg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(acg_frame, text="Match CCG scale",
                        variable=self._acg_match_ccg_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=4)

    def _on_acg_scale_change(self):
        """ACG Y-scale slider changed."""
        val = self._acg_yscale_var.get()
        self._acg_scale_label.config(text=f"{val:.1f}x")
        if self._acg_ref_var.get() or self._acg_tgt_var.get():
            self._clear_all_png_cache()
            self.update_plot()

    def _on_sig_toggle(self):
        """Update significance display flags and redraw."""
        self._sig_show_conv_baseline = self._sig_conv_baseline_var.get()
        self._sig_show_conv_p = self._sig_conv_p_var.get()
        self._sig_show_conv_pc = self._sig_conv_pc_var.get()
        self._sig_show_test_window = self._sig_test_window_var.get()
        self._sig_show_jitter_p = self._sig_jitter_p_var.get()
        self._sig_show_jitter_pc = self._sig_jitter_pc_var.get()
        self._clear_all_png_cache()
        self.update_plot()

    def _update_jitter_sig_buttons(self):
        """Enable/disable jitter significance buttons based on cache."""
        if not hasattr(self, '_sig_jitter_p_cb'):
            return
        inds = self.all_inds[self.current_pair_idx] if self.current_pair_idx < len(self.all_inds) else None
        has_jitter = False
        if inds is not None:
            has_jitter = self._jitter_cache.get(
                (int(inds[0]), int(inds[1]))) is not None
        state = ['!disabled'] if has_jitter else ['disabled']
        self._sig_jitter_p_cb.state(state)
        self._sig_jitter_pc_cb.state(state)

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

        # ── Neuron-focus row ───────────────────────────────────────────
        focus_frame = ttk.Frame(parent)
        focus_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(focus_frame, text="Focus neuron:").pack(side=tk.LEFT)
        self._focus_var = tk.StringVar()
        focus_entry = ttk.Entry(focus_frame, textvariable=self._focus_var, width=6)
        focus_entry.pack(side=tk.LEFT, padx=2)
        focus_entry.bind('<Return>', lambda e: self._on_neuron_focus())
        ttk.Button(focus_frame, text="Clear",
                   command=self._on_neuron_focus_clear).pack(side=tk.LEFT, padx=2)
        self._focus_info_var = tk.StringVar(value="")
        ttk.Label(focus_frame, textvariable=self._focus_info_var,
                  font=('Arial', 8), foreground='#555').pack(
            side=tk.LEFT, padx=4)

        # ── Pair-focus row ─────────────────────────────────────────────
        pair_focus_frame = ttk.Frame(parent)
        pair_focus_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(pair_focus_frame, text="Focus pair:").pack(side=tk.LEFT)
        self._focus_pair_var = tk.StringVar()
        pair_focus_entry = ttk.Entry(pair_focus_frame,
                                     textvariable=self._focus_pair_var, width=8)
        pair_focus_entry.pack(side=tk.LEFT, padx=2)
        pair_focus_entry.bind('<Return>', lambda e: self._on_pair_focus())
        ttk.Button(pair_focus_frame, text="Clear",
                   command=self._on_pair_focus_clear).pack(side=tk.LEFT, padx=2)
        self._focus_pair_info_var = tk.StringVar(value="")
        ttk.Label(pair_focus_frame, textvariable=self._focus_pair_info_var,
                  font=('Arial', 8), foreground='#555').pack(
            side=tk.LEFT, padx=4)
        self._add_pair_btn = ttk.Button(pair_focus_frame, text="Add to available",
                                         command=self._on_add_focused_pair,
                                         state=tk.DISABLED)
        self._add_pair_btn.pack(side=tk.LEFT, padx=4)

        # ── Connection type toggles ──────────────────────────────────────
        ct_frame = ttk.Frame(parent)
        ct_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        _ct_labels = {
            ('pyr', 'pyr'):     'P→P',
            ('pyr', 'inter'):   'P→I',
            ('inter', 'inter'): 'I→I',
            ('inter', 'pyr'):   'I→P',
        }
        self._net_ct_vars = {}
        for ct in [('pyr', 'pyr'), ('pyr', 'inter'),
                   ('inter', 'inter'), ('inter', 'pyr')]:
            var = tk.BooleanVar(value=True)
            self._net_ct_vars[ct] = var
            ttk.Checkbutton(ct_frame, text=_ct_labels[ct], variable=var,
                            command=self._draw_network).pack(side=tk.LEFT, padx=2)

        # ── Network display toggles ──────────────────────────────────────
        toggle_frame = ttk.Frame(parent)
        toggle_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._net_arrows_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Lines",
                        variable=self._net_arrows_var,
                        command=self._on_net_toggle_arrows
                        ).pack(side=tk.LEFT, padx=(0, 6))
        self._net_hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Hide unconnected",
                        variable=self._net_hide_var,
                        command=self._on_net_toggle_hide
                        ).pack(side=tk.LEFT)
        self._net_hide_same_shank_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Hide same shank",
                        variable=self._net_hide_same_shank_var,
                        command=self._on_net_toggle_hide_same_shank
                        ).pack(side=tk.LEFT, padx=(6, 0))

        # ── Group filter dropdown ─────────────────────────────────────────
        group_frame = ttk.Frame(parent)
        group_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(group_frame, text="Group:").pack(side=tk.LEFT)
        self._net_group_var = tk.StringVar(value="(none)")
        self._net_group_combo = ttk.Combobox(
            group_frame, textvariable=self._net_group_var,
            state='readonly', width=14)
        self._net_group_combo['values'] = ['(none)']
        self._net_group_combo.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self._net_group_combo.bind('<<ComboboxSelected>>',
                                   self._on_net_group_select)

        # ── Zoom sliders ─────────────────────────────────────────────────
        zoom_frame = ttk.Frame(parent)
        zoom_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(zoom_frame, text="H:", font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_hzoom_var = tk.DoubleVar(value=1.0)
        self._net_hzoom = ttk.Scale(
            zoom_frame, from_=0.2, to=5.0, orient=tk.HORIZONTAL,
            variable=self._net_hzoom_var, command=self._on_net_zoom)
        self._net_hzoom.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Label(zoom_frame, text="V:", font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_vzoom_var = tk.DoubleVar(value=1.0)
        self._net_vzoom = ttk.Scale(
            zoom_frame, from_=0.2, to=5.0, orient=tk.HORIZONTAL,
            variable=self._net_vzoom_var, command=self._on_net_zoom)
        self._net_vzoom.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.net_fig = Figure(figsize=(2.8, 6.5))
        self.net_ax = self.net_fig.add_subplot(111)
        self.net_canvas = FigureCanvasTkAgg(self.net_fig, master=parent)
        self.net_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._net_pick_cid = self.net_canvas.mpl_connect(
            'pick_event', self._on_network_pick)
        self._net_scroll_cid = self.net_canvas.mpl_connect(
            'scroll_event', self._on_net_scroll)
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
        ttk.Label(ctrl, text="Name:").pack(side=tk.LEFT, padx=(8, 0))
        self._ts_name_var = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self._ts_name_var,
                  width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Clear",
                   command=self._on_time_slider_clear).pack(side=tk.LEFT, padx=(8, 2))
        self._ts_status_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._ts_status_var,
                  font=('Courier', 8), foreground='#555').pack(
            side=tk.LEFT, padx=8)

    # ── Bottom bar ─────────────────────────────────────────────────────

    def setup_bottom_panel(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        ttk.Button(bottom_frame, text="Save Selections",
                   command=self._quick_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Quit",
                   command=self._on_close).pack(side=tk.RIGHT, padx=5)
        self.stats_var = tk.StringVar(value=self._compute_stats_str())
        ttk.Label(bottom_frame, textvariable=self.stats_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)
        self._pair_info_var = tk.StringVar(value="")
        ttk.Label(bottom_frame, textvariable=self._pair_info_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)

    # ------------------------------------------------------------------
    # Panel toggle helpers
    # ------------------------------------------------------------------

    def _toggle_panel(self, name):
        """Show or hide a panel based on its BooleanVar."""
        print(f"[CCGReviewUI] _toggle_panel({name!r})")
        try:
            self._toggle_panel_impl(name)
        except Exception as ex:
            print(f"[CCGReviewUI] _toggle_panel error for {name!r}: {ex}")
            import traceback; traceback.print_exc()

    def _toggle_panel_impl(self, name):
        """Inner implementation of panel toggling."""
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
        elif name == 'Group Hotkeys':
            if show:
                self._refresh_hotkeys_bar()
                # Pack after the tool-strip (setup_menu frame), before the main area
                self._hotkeys_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2),
                                       before=self._main_frame)
            else:
                self._hotkeys_bar.pack_forget()

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
        print(f"[CCGReviewUI] _toggle_time_slider show={show}")
        if show:
            try:
                self.time_slider_frame.pack(
                    in_=self._main_frame, side=tk.TOP,
                    fill=tk.X, before=self._paned, pady=(0, 4))
                self._ts_init_times()
                print(f"[CCGReviewUI]   epoch_bounds={len(self._ts_epoch_bounds)} "
                      f"total_sec={self._ts_total_sec:.1f}")
                self._ts_redraw()
            except Exception as ex:
                print(f"[CCGReviewUI]   ERROR in _toggle_time_slider: {ex}")
                import traceback; traceback.print_exc()
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
        self._build_sig_chips()
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

        Significance is always derived from the **low-res** CCGData so that
        segment chips stay green when the user switches to high-res mode.
        Priority: jitter (segment-aware) → pval_corrected + active_alpha
        → stored significant array.
        """
        # Custom segments: use stored pval_corrected
        if self._is_custom_segment(seg):
            ci = self._custom_seg_index(seg)
            cs_list = getattr(self, '_custom_segments', [])
            if 0 <= ci < len(cs_list):
                cs = cs_list[ci]
                pc = cs.get('pval_corrected')
                if pc is not None:
                    conf = self.ccg_data.conf
                    lb = getattr(conf, 'min_lag_bin', None)
                    ub = getattr(conf, 'max_lag_bin', None)
                    if lb is not None and ub is not None:
                        try:
                            return bool(pc[0, ref, tgt, lb:ub].min()
                                        <= self.active_alpha)
                        except (IndexError, ValueError):
                            pass
            return False

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

        # Always use low-res data for significance (high-res may lack pval arrays)
        cd = self.cd._ccg.get(self.key.nd()) if hasattr(self.cd, '_ccg') else None
        if cd is None:
            cd = self.ccg_data   # fallback
        if cd is not None and cd.pval_corrected is not None:
            conf = cd.conf
            lb = getattr(conf, 'min_lag_bin', None)
            ub = getattr(conf, 'max_lag_bin', None)
            if lb is not None and ub is not None:
                try:
                    return bool(
                        cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                        <= self.active_alpha)
                except (IndexError, ValueError):
                    pass
        if cd is not None and cd.significant is not None:
            try:
                return bool(cd.significant[seg, ref, tgt])
            except (IndexError, ValueError):
                pass
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
        self._update_jitter_sig_buttons()
        self.update_plot()

    def _on_clear_jitter(self):
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        self._jitter_cache.pop((ref, tgt), None)
        self._clear_all_png_cache()
        self._update_jitter_sig_buttons()
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
        ttk.Label(self.sig_frame, text="Segments:").pack(side=tk.LEFT, padx=(4, 2))
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
        # Custom segment chips
        for ci, cs in enumerate(getattr(self, '_custom_segments', [])):
            seg_idx = self.n_segments + 1 + ci
            lbl_cust = tk.Label(
                self.sig_frame, text=cs['name'],
                relief=tk.SUNKEN, font=('Arial', 8, 'italic'),
                bg='#FFF9C4', fg='#5D4037', padx=4, pady=2)
            lbl_cust.pack(side=tk.LEFT, padx=(4, 2))
            lbl_cust.bind('<Button-1>',
                          lambda e, idx=seg_idx: self._jump_to_segment(idx))
            lbl_cust.bind('<Double-Button-1>',
                          lambda e, idx=ci: self._remove_custom_segment(idx))
            self.seg_sig_labels.append(lbl_cust)

    # ------------------------------------------------------------------
    # Key / dropdown helpers
    # ------------------------------------------------------------------

    def _all_nd_keys(self) -> list:
        """Return unique nd_keys (one per session) across the dataset.

        Prefers ``cd._ccg`` (keyed by nd_keys directly) so that ALL sessions
        are represented even when some have no significant pairs in ``cd.data``.
        Falls back to enumerating ``cd.data`` if ``_ccg`` is unavailable.
        """
        seen, seen_str = [], set()
        # Primary source: _ccg is keyed by nd_keys, one per session
        ccg_source = getattr(self.cd, '_ccg', None) or {}
        for nk in ccg_source.keys():
            s = str(nk)
            if s not in seen_str:
                seen.append(nk)
                seen_str.add(s)
        # Secondary: pick up any sessions present only in cd.data
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
        # Persist in-session selections to the current pointer before switching,
        # so they survive type/session changes and can be restored on return.
        self.ccg_pointer.manually_selected_inds = (
            np.array(sorted(self.selected_inds), dtype=int)
            if self.selected_inds else None
        )

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
        # all_inds is a @property — no assignment needed
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
        self._custom_segments.clear()
        return True

    def _refresh_after_key_switch(self):
        self.segment_combo['values'] = self.segment_names + [_ALL_SEGS]
        self.segment_var.set(self.segment_names[0])
        self._build_sig_chips()
        self.refresh_lists()
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()
        if self._focused_neuron is not None:
            self._update_focus_info(self._focused_neuron)
        self._draw_network()

    # ------------------------------------------------------------------
    # Dropdown callbacks
    # ------------------------------------------------------------------

    def _on_session_change(self, event):
        idx = self._session_combo.current()
        if idx < 0 or idx >= len(self._nd_keys_list):
            return
        # Auto-save current session before switching away
        self._autosave_current()
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
            self._autoload_session_latest()

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
        n_ref = None
        n_tgt = None
        ref_t = tgt_t = None
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
        # Show per-type neuron counts
        if n_ref is not None:
            if ref_t == tgt_t:
                s += f"  {ref_t}: {n_ref}"
            else:
                s += f"  {ref_t}: {n_ref}  {tgt_t}: {n_tgt}"
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

    def _update_pair_info(self, inds):
        """Update per-pair info (firing rates) in the stats bar."""
        if not hasattr(self, '_pair_info_var'):
            return
        if self.neurons is None:
            self._pair_info_var.set("")
            return
        ref, tgt = int(inds[0]), int(inds[1])
        fr = getattr(self.neurons, 'firing_rate', None)
        if fr is None:
            self._pair_info_var.set("")
            return
        ref_fr = float(fr[ref])
        tgt_fr = float(fr[tgt])

        # Segment-level firing rate
        seg_str = ""
        seg = self.current_segment
        nd_key = self.key.nd() if self.key else None
        seg_frates = None
        if (nd_key is not None and self.cd.nd is not None
                and not self._is_custom_segment(seg)
                and seg != self.n_segments):
            seg_frates = self.cd.nd.segment_firing_rates.get(nd_key)
        if seg_frates is not None and seg < seg_frates.shape[0]:
            sr = float(seg_frates[seg, ref])
            st = float(seg_frates[seg, tgt])
            seg_str = f"  Seg FR: ref={sr:.1f}  tgt={st:.1f}"

        self._pair_info_var.set(
            f"FR: ref={ref_fr:.1f}Hz  tgt={tgt_fr:.1f}Hz{seg_str}")

    # ------------------------------------------------------------------
    # Undo / redo for pair selection
    # ------------------------------------------------------------------

    def _push_undo(self):
        """Snapshot current selection state before a mutation."""
        self._undo_stack.append((set(self.selected_inds),
                                 set(self.unselected_inds)))
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self, event=None):
        if not self._undo_stack:
            return
        # Save current state to redo
        self._redo_stack.append((set(self.selected_inds),
                                 set(self.unselected_inds)))
        sel, unsel = self._undo_stack.pop()
        self.selected_inds = sel
        self.unselected_inds = unsel
        self.refresh_lists()
        self.update_plot()
        self._draw_network()
        self._refresh_stats()

    def _redo(self, event=None):
        if not self._redo_stack:
            return
        self._undo_stack.append((set(self.selected_inds),
                                 set(self.unselected_inds)))
        sel, unsel = self._redo_stack.pop()
        self.selected_inds = sel
        self.unselected_inds = unsel
        self.refresh_lists()
        self.update_plot()
        self._draw_network()
        self._refresh_stats()

    # ------------------------------------------------------------------
    # Pair lists
    # ------------------------------------------------------------------

    def refresh_lists(self):
        self.unselected_list.delete(0, tk.END)
        self.selected_list.delete(0, tk.END)

        # Build gray-out set based on active focus mode
        gray_out = None
        fn = self._focused_neuron
        fp = self._focused_pair
        if fn is not None:
            focus_connected = {(ref, tgt) for ref, tgt in
                               map(tuple, self.all_inds)
                               if ref == fn or tgt == fn}
            gray_out = lambda inds: inds not in focus_connected
        elif fp is not None:
            gray_out = lambda inds: inds != fp

        # Same-shank graying
        hide_same_shank = (hasattr(self, '_net_hide_same_shank_var')
                           and self._net_hide_same_shank_var.get())
        shank_ids = (getattr(self.neurons, 'shank_ids', None)
                     if self.neurons is not None else None)

        def _should_gray(inds):
            if gray_out is not None and gray_out(inds):
                return True
            if hide_same_shank and shank_ids is not None:
                if int(shank_ids[inds[0]]) == int(shank_ids[inds[1]]):
                    return True
            return False

        for inds in sorted(self.unselected_inds):
            label = f"[{inds[0]:3d}, {inds[1]:3d}]"
            grp = self._pair_group_label(inds)
            if grp:
                label += f" {grp}"
            self.unselected_list.insert(tk.END, label)
            if _should_gray(inds):
                idx = self.unselected_list.size() - 1
                self.unselected_list.itemconfig(idx, foreground='#AAAAAA')

        for inds in sorted(self.selected_inds):
            label = f"[{inds[0]:3d}, {inds[1]:3d}]"
            grp = self._pair_group_label(inds)
            if grp:
                label += f" {grp}"
            self.selected_list.insert(tk.END, label)
            if _should_gray(inds):
                idx = self.selected_list.size() - 1
                self.selected_list.itemconfig(idx, foreground='#AAAAAA')

        self._avail_label_var.set(f"Available ({len(self.unselected_inds)})")
        self._sel_label_var.set(f"Selected ({len(self.selected_inds)})")
        if hasattr(self, '_select_all_btn'):
            self._select_all_btn.config(
                text="Deselect All" if not self.unselected_inds else "Select All")
        self._refresh_stats()

    def move_to_selected(self, event=None):
        """Move the item under the cursor (double-click) or current pair (keyboard)."""
        # Cancel any pending single-click deferred update
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
            self._select_after = None
        # Determine which item to move
        if event is not None:
            # Double-click: use position under cursor — independent of selection state
            idx = self.unselected_list.nearest(event.y)
        else:
            # Keyboard / context menu: use currently highlighted item
            sel = self.unselected_list.curselection()
            idx = sel[-1] if sel else None
        if idx is None or idx < 0:
            return
        sorted_unsel = sorted(self.unselected_inds)
        if idx >= len(sorted_unsel):
            return
        inds = sorted_unsel[idx]
        # Preserve scroll position so the list doesn't jump to the top
        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        self.unselected_inds.discard(inds)
        self.selected_inds.add(inds)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self.current_pair_idx = self.get_pair_index(inds)
        self.update_plot()
        self._draw_network()

    def move_to_unselected(self, event=None):
        """Move the item under the cursor (double-click) or current pair (keyboard)."""
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
            self._select_after = None
        if event is not None:
            idx = self.selected_list.nearest(event.y)
        else:
            sel = self.selected_list.curselection()
            idx = sel[-1] if sel else None
        if idx is None or idx < 0:
            return
        sorted_sel = sorted(self.selected_inds)
        if idx >= len(sorted_sel):
            return
        inds = sorted_sel[idx]
        # Preserve scroll position so the list doesn't jump to the top
        scroll_top = self.selected_list.yview()[0]
        self._push_undo()
        self.selected_inds.discard(inds)
        self.unselected_inds.add(inds)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self.current_pair_idx = self.get_pair_index(inds)
        self.update_plot()
        self._draw_network()

    def _move_current_pair(self):
        """Hotkey 'm': toggle the current pair between Available and Selected."""
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = tuple(self.all_inds[self.current_pair_idx])
        self._push_undo()
        if inds in self.unselected_inds:
            self.unselected_inds.discard(inds)
            self.selected_inds.add(inds)
        else:
            self.selected_inds.discard(inds)
            self.unselected_inds.add(inds)
        self.refresh_lists()
        self._draw_network()

    def on_pair_select(self, event):
        """Single-click: navigate to the clicked pair (debounced to avoid blocking double-click)."""
        widget = event.widget
        # In BROWSE mode curselection() is reliably set by the time ButtonRelease fires
        sel = widget.curselection()
        if not sel:
            # Fallback: use nearest()
            idx = widget.nearest(event.y)
        else:
            idx = sel[-1]
        if idx < 0:
            return
        if widget == self.unselected_list:
            sorted_items = sorted(self.unselected_inds)
        else:
            sorted_items = sorted(self.selected_inds)
        if idx >= len(sorted_items):
            return
        inds = sorted_items[idx]
        self.current_pair_idx = self.get_pair_index(inds)
        # Debounce: defer the heavy update so double-click can fire first
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
        self._select_after = self.root.after(180, self._do_pair_select_update)

    def _do_pair_select_update(self):
        """Execute the deferred pair-select update (after debounce timeout)."""
        self._select_after = None
        self.update_plot()
        self._draw_network()

    def _on_arrow_key(self, event):
        """Up/Down arrow key in a pair list: update CCG to the newly selected pair."""
        widget = event.widget
        sel = widget.curselection()
        if not sel:
            return
        idx = sel[-1]
        if widget == self.unselected_list:
            sorted_items = sorted(self.unselected_inds)
        else:
            sorted_items = sorted(self.selected_inds)
        if idx >= len(sorted_items):
            return
        inds = sorted_items[idx]
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
        """Right-click context menu for the pair lists.

        Supports multi-selection: if multiple items are selected (via
        Ctrl+click or Shift+click), the group-tag action applies to all.
        """
        # If right-clicked item is not already part of the selection,
        # replace selection with just that item (standard behavior).
        click_idx = widget.nearest(event.y)
        if click_idx not in widget.curselection():
            widget.selection_clear(0, tk.END)
            widget.selection_set(click_idx)
            widget.activate(click_idx)

        # Build list of selected pairs
        if action == 'add':
            sorted_items = sorted(self.unselected_inds)
        else:
            sorted_items = sorted(self.selected_inds)
        sel_indices = list(widget.curselection())
        pairs = [sorted_items[i] for i in sel_indices
                 if 0 <= i < len(sorted_items)]
        n = len(pairs)

        menu = tk.Menu(self.root, tearoff=0)
        if action == 'add':
            menu.add_command(
                label=f"Move to Selected ({n})" if n > 1 else "Move to Selected",
                command=lambda pp=pairs: self._ctx_move_multi_to_selected(pp))
            menu.add_command(label="Select All",
                             command=self._select_all)
        else:
            menu.add_command(
                label=f"Move to Available ({n})" if n > 1 else "Move to Available",
                command=lambda pp=pairs: self._ctx_move_multi_to_unselected(pp))

        # Group tag submenu
        menu.add_separator()
        grp_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Group tag", menu=grp_menu)
        grp_menu.add_command(label="Create new group…",
                             command=self._create_group_dialog)
        if self._groups:
            grp_menu.add_separator()
        for gname in sorted(self._groups):
            if gname.startswith('__'):
                continue
            if pairs:
                # Show ✓ if ALL selected pairs are in the group
                all_in = all(p in self._groups[gname] for p in pairs)
                label = f"{'✓ ' if all_in else ''}  {gname}"
                grp_menu.add_command(
                    label=label,
                    command=lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))
        menu.tk_popup(event.x_root, event.y_root)

    def _ctx_move_to_selected(self, pair):
        """Context-menu: move a specific pair from Available → Selected."""
        if pair is None:
            return
        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        self.unselected_inds.discard(pair)
        self.selected_inds.add(pair)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self.current_pair_idx = self.get_pair_index(pair)
        self.update_plot()
        self._draw_network()

    def _ctx_move_multi_to_selected(self, pairs):
        """Context-menu: move multiple pairs from Available → Selected."""
        if not pairs:
            return
        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            self.unselected_inds.discard(p)
            self.selected_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._draw_network()

    def _ctx_move_to_unselected(self, pair):
        """Context-menu: move a specific pair from Selected → Available."""
        if pair is None:
            return
        scroll_top = self.selected_list.yview()[0]
        self._push_undo()
        self.selected_inds.discard(pair)
        self.unselected_inds.add(pair)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self.current_pair_idx = self.get_pair_index(pair)
        self.update_plot()
        self._draw_network()

    def _ctx_move_multi_to_unselected(self, pairs):
        """Context-menu: move multiple pairs from Selected → Available."""
        if not pairs:
            return
        scroll_top = self.selected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            self.selected_inds.discard(p)
            self.unselected_inds.add(p)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self._draw_network()

    def get_pair_index(self, inds):
        for i, pair in enumerate(self.all_inds):
            if tuple(pair) == tuple(inds):
                return i
        return 0

    # ------------------------------------------------------------------
    # Segment navigation
    # ------------------------------------------------------------------

    def _n_total_segments(self):
        """Total navigable segments: real + All + custom."""
        return self.n_segments + 1 + len(self._custom_segments)

    def _is_custom_segment(self, seg=None):
        if seg is None:
            seg = self.current_segment
        return seg > self.n_segments

    def _custom_seg_index(self, seg=None):
        """Return the index into _custom_segments for the given segment id."""
        if seg is None:
            seg = self.current_segment
        return seg - self.n_segments - 1

    def change_segment(self, delta):
        total = self._n_total_segments()
        self.current_segment = (self.current_segment + delta) % total
        self._update_segment_label()
        self.update_plot()

    def _jump_to_segment(self, idx):
        self.current_segment = idx
        self._update_segment_label()
        self.update_plot()

    def _update_segment_label(self):
        seg = self.current_segment
        if seg == self.n_segments:
            self.segment_var.set(_ALL_SEGS)
        elif self._is_custom_segment(seg):
            ci = self._custom_seg_index(seg)
            self.segment_var.set(self._custom_segments[ci]['name'])
        else:
            self.segment_var.set(self.segment_names[seg])
        self.plot_title_var.set(self.get_plot_title())

    def _remove_custom_segment(self, ci):
        """Remove a custom segment by its index in _custom_segments."""
        if 0 <= ci < len(self._custom_segments):
            self._custom_segments.pop(ci)
            # If we were viewing the removed (or a later) custom segment, reset
            if self.current_segment > self.n_segments:
                self.current_segment = min(self.current_segment,
                                           self._n_total_segments() - 1)
            self._build_sig_chips()
            self._update_segment_label()
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
            # Custom segment chips (after All chip)
            if chip_idx > self.n_segments:
                ci = chip_idx - self.n_segments - 1
                cs_list = getattr(self, '_custom_segments', [])
                if 0 <= ci < len(cs_list):
                    cs = cs_list[ci]
                    pc = cs.get('pval_corrected')
                    if (pc is not None and lb is not None and ub is not None
                            and ref < pc.shape[1] and tgt < pc.shape[2]):
                        sig = bool(pc[0, ref, tgt, lb:ub].min()
                                   <= self.active_alpha)
                    else:
                        sig = False
                    active = (self.current_segment == chip_idx)
                    bg = ('#4CAF50' if active else '#90EE90') if sig \
                        else ('#FFD54F' if active else '#FFF9C4')
                else:
                    bg = '#FFF9C4'
                lbl.config(bg=bg)
                continue

            is_all_chip = (chip_idx == self.n_segments)
            if is_all_chip:
                sig = any(self._is_significant(ref, tgt, s)
                          for s in range(self.n_segments))
                active = (self.current_segment == self.n_segments)
            else:
                seg = chip_idx
                if (cd is not None and cd.pval_corrected is not None and
                        lb is not None and ub is not None and
                        seg < cd.pval_corrected.shape[0]):
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
            if self._is_custom_segment():
                ci = self._custom_seg_index()
                seg_label = self._custom_segments[ci]['name']
            elif self.current_segment == self.n_segments:
                seg_label = _ALL_SEGS
            else:
                seg_label = self.segment_names[self.current_segment]
            # Include session and connectivity type
            sess = self.key.session if self.key else ''
            ct = (f"{self.key.conn_type[0]}-{self.key.conn_type[1]}"
                  if self.key and self.key.conn_type else '')
            return f"{sess} | {ct} | Pair [{inds[0]}, {inds[1]}] — {seg_label}"
        return "No pair selected"

    # ------------------------------------------------------------------
    # PNG rendering
    # ------------------------------------------------------------------

    def _png_path(self, inds, segment) -> str:
        if self._is_custom_segment(segment):
            ci = self._custom_seg_index(segment)
            cs = self._custom_segments[ci]
            # Use name + time range for uniqueness
            seg_name = f"custom_{cs['name']}_{cs['t0']:.0f}_{cs['t1']:.0f}"
            seg_name = seg_name.replace(' ', '_').replace(':', '-')
        elif segment == self.n_segments:
            seg_name = _ALL_SEGS.replace(' ', '_')
        else:
            seg_name = self.segment_names[segment]
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
        # Significance display state
        sig_key = ''
        sig_bits = (
            ('b' if self._sig_show_conv_baseline else '') +
            ('p' if self._sig_show_conv_p else '') +
            ('c' if self._sig_show_conv_pc else '') +
            ('tw' if self._sig_show_test_window else '') +
            ('jp' if self._sig_show_jitter_p else '') +
            ('jc' if self._sig_show_jitter_pc else '') +
            ('ar' if getattr(self, '_acg_ref_var', None) and self._acg_ref_var.get() else '') +
            ('at' if getattr(self, '_acg_tgt_var', None) and self._acg_tgt_var.get() else '') +
            (f'as{self._acg_yscale_var.get():.1f}' if getattr(self, '_acg_yscale_var', None) and (
                (getattr(self, '_acg_ref_var', None) and self._acg_ref_var.get()) or
                (getattr(self, '_acg_tgt_var', None) and self._acg_tgt_var.get())) else '') +
            ('am' if getattr(self, '_acg_match_ccg_var', None) and self._acg_match_ccg_var.get() else ''))
        if sig_bits:
            sig_key = f'_s{sig_bits}'
        return os.path.join(
            self.tmp_dir,
            f"pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
            f"{alpha_key}{res_key}{scale_key}{j_key}{sig_key}.png")

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
                seg != self.n_segments and
                not self._is_custom_segment(seg)):
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

        is_custom = self._is_custom_segment(segment)
        is_all = (segment == self.n_segments)

        if is_custom:
            ci = self._custom_seg_index(segment)
            cs = self._custom_segments[ci]
            # Pick hi-res or lo-res data based on current resolution mode
            if self._highres_mode and 'ccg_hi' in cs:
                ccg_raw = cs['ccg_hi'][0, ref, tgt, :]
                ccg_null_raw = (cs['ccg_null_hi'][0, ref, tgt, :]
                                if cs.get('ccg_null_hi') is not None else None)
                pval_arg = (cs['pval_hi'][0, ref, tgt, :]
                            if cs.get('pval_hi') is not None else None)
                pval_c_arg = (cs['pval_corrected_hi'][0, ref, tgt, :]
                              if cs.get('pval_corrected_hi') is not None else None)
            else:
                ccg_raw = cs['ccg'][0, ref, tgt, :]
                ccg_null_raw = (cs['ccg_null'][0, ref, tgt, :]
                                if cs['ccg_null'] is not None else None)
                pval_arg = (cs['pval'][0, ref, tgt, :]
                            if cs.get('pval') is not None else None)
                pval_c_arg = (cs['pval_corrected'][0, ref, tgt, :]
                              if cs.get('pval_corrected') is not None else None)
            seg_label = cs['name']
        elif is_all:
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

        # Apply significance display toggles
        show_null = ccg_null if self._sig_show_conv_baseline else None
        show_pval = pval_arg if self._sig_show_conv_p else None
        show_pval_c = pval_c_arg if self._sig_show_conv_pc else None
        show_j_ccg = j_ccg_arg if self._sig_show_jitter_p else None
        show_j_pval = j_pval_arg if self._sig_show_jitter_pc else None

        # Auto-correlogram overlays (diagonal of CCG matrix)
        acg_ref = acg_tgt = None
        show_acg_ref = getattr(self, '_acg_ref_var', None) and self._acg_ref_var.get()
        show_acg_tgt = getattr(self, '_acg_tgt_var', None) and self._acg_tgt_var.get()
        if show_acg_ref or show_acg_tgt:
            if is_custom:
                src = cs.get('ccg_hi' if (self._highres_mode and 'ccg_hi' in cs) else 'ccg')
            elif is_all:
                src = cd.ccg
            else:
                src = cd.ccg
            if src is not None:
                if show_acg_ref:
                    if is_all:
                        acg_ref = np.sum(src[:, ref, ref, :], axis=0)
                    elif is_custom:
                        acg_ref = src[0, ref, ref, :]
                    else:
                        acg_ref = src[segment, ref, ref, :]
                if show_acg_tgt:
                    if is_all:
                        acg_tgt = np.sum(src[:, tgt, tgt, :], axis=0)
                    elif is_custom:
                        acg_tgt = src[0, tgt, tgt, :]
                    else:
                        acg_tgt = src[segment, tgt, tgt, :]

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg, ids=inds, inds=inds,
            window_size=conf.duration, bin_size=bin_size_eff,
            pval=show_pval, pval_corrected=show_pval_c,
            alpha=self.active_alpha, ccg_null=show_null,
            j_ccg=show_j_ccg, j_pval=show_j_pval,
            segment_id=seg_label,
            is_significant_pair=self._is_significant(ref, tgt, segment),
            min_lag=conf.min_lag if self._sig_show_test_window else None,
            max_lag=conf.max_lag if self._sig_show_test_window else None,
            normalize_info=norm_info,
            acg_ref=acg_ref, acg_tgt=acg_tgt,
            acg_yscale=getattr(self, '_acg_yscale_var', None) and self._acg_yscale_var.get() or 1.0,
            acg_match_ccg=getattr(self, '_acg_match_ccg_var', None) and self._acg_match_ccg_var.get(),
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

    def _autosave_current(self):
        """Silently save current session's selections + groups as 'latest'.

        Called before any operation that would overwrite self._groups or
        self.selected_inds (session switch, GUI close).
        """
        try:
            self._save_selection_version('latest')
        except Exception as exc:
            print(f"[CCGReviewUI] autosave failed: {exc}")

    def _autoload_session_latest(self, restore_groups: bool = False):
        """Load the 'latest' selection file for the current session, if it exists.

        By default only restores pair selections — groups are shared across
        sessions and should not be overwritten on session switch.  Pass
        restore_groups=True on first launch to seed groups from the file.
        """
        latest_path = self._sel_version_path('latest')
        if not os.path.isfile(latest_path):
            return
        try:
            self._load_selection_from_file(latest_path,
                                           restore_groups=restore_groups)
        except Exception as exc:
            print(f"[CCGReviewUI] failed to autoload latest: {exc}")

    def _deferred_initial_draw(self):
        # On first launch, restore groups from file (subsequent session
        # switches will keep groups intact via restore_groups=False)
        self._autoload_session_latest(restore_groups=True)
        self.update_plot()

    def update_plot(self):
        try:
            # Focus-pair override: show focused pair's CCG directly
            if self._focused_pair is not None:
                inds = np.array(self._focused_pair)
            elif self.current_pair_idx >= len(self.all_inds):
                return
            else:
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
            self._update_jitter_sig_buttons()
            self._update_pair_info(inds)
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
        # Use entered name, falling back to the time-range string
        name = getattr(self, '_ts_name_var', None)
        name = name.get().strip() if name is not None else ""
        if not name:
            name = f"{self._ts_sec_to_hms(t0)}–{self._ts_sec_to_hms(t1)}"

        # Compute full CCG pipeline for this time window (all neurons)
        seg_data = self._compute_custom_segment(t0, t1, name)
        if seg_data is None:
            return
        self._custom_segments.append(seg_data)

        # Navigate to the new custom segment
        self.current_segment = self.n_segments + len(self._custom_segments)
        self._ts_status_var.set(
            f"Active: {self._ts_sec_to_hms(t0)} – {self._ts_sec_to_hms(t1)}")
        self._ts_redraw()
        self._build_sig_chips()
        self._update_segment_label()
        self.update_plot()

    def _on_time_slider_clear(self):
        self._custom_segments.clear()
        if hasattr(self, '_ts_status_var'):
            self._ts_status_var.set("")
        # Reset to first real segment
        self.current_segment = 0
        self._build_sig_chips()
        self._update_segment_label()
        self.update_plot()

    # ------------------------------------------------------------------
    # Custom-window CCG
    # ------------------------------------------------------------------

    def _compute_custom_segment(self, t0: float, t1: float, name: str):
        """Compute full CCG pipeline for a custom time window.

        Runs spike_correlations → EranConv._conv → multiple_correction
        at **both** low-res (1 ms) and high-res (0.1 ms) bin sizes so
        that Ctrl+R resolution toggle works on custom segments too.

        Returns a dict with keys: name, t0, t1, ccg, ccg_null, pval,
        pval_corrected (low-res), and optionally ccg_hi, ccg_null_hi,
        pval_hi, pval_corrected_hi — or None on failure.
        """
        if self.neurons is None:
            messagebox.showerror("Custom CCG", "No neuron data available.")
            return None
        try:
            from neuropy.analyses.correlations import spike_correlations
            from neuropy.analyses.ms_connectivity import EranConv, _CCG_RESOLUTION

            neurons_slice = self.neurons.time_slice(t0, t1)
            conf = self.ccg_data.conf
            n_neurons = self.neurons.n_neurons
            neuron_inds = np.arange(n_neurons)
            method = conf.mc_method if conf.mc_method is not None else 'bonferroni'
            ei = getattr(self.key, 'excitability', 'E')

            def _run_pipeline(bin_size, label):
                print(f"[CustomSegment] computing {label} CCG for {name} "
                      f"({t1-t0:.1f}s, {n_neurons} neurons, "
                      f"bin={bin_size*1e3:.2f}ms) ...")
                ccg = spike_correlations(
                    neurons=neurons_slice,
                    neuron_inds=neuron_inds,
                    bin_size=bin_size,
                    window_size=conf.duration,
                    symmetrize=conf.symmetrize_ccg,
                    use_acceleration=conf.use_acceleration,
                )
                ccg = ccg[np.newaxis, ...]
                # Compute W from the actual bin_size used, not from conf
                # (conf.bin_size may be mutated to high-res)
                W = conf.conv_window / bin_size
                pvals, pred, qvals = EranConv._conv(
                    ccg, W=W, wintype="gauss",
                    hollow_frac=None)
                p_raw = pvals if ei == 'E' else qvals
                _, pval_corrected = EranConv.multiple_correction(
                    p_raw, conf.alpha, method=method)
                print(f"[CustomSegment] {label} done. shape={ccg.shape}")
                return ccg, pred, p_raw, pval_corrected

            # Low-res (always)
            lo_bs = _CCG_RESOLUTION['lowres']
            ccg_lo, pred_lo, pval_lo, pvalc_lo = _run_pipeline(lo_bs, 'lowres')

            result = {
                'name':           name,
                't0':             t0,
                't1':             t1,
                'ccg':            ccg_lo,
                'ccg_null':       pred_lo,
                'pval':           pval_lo,
                'pval_corrected': pvalc_lo,
            }

            # High-res (only if highres data is available for this session)
            has_highres = (hasattr(self.cd, '_ccg_highres') and
                           bool(self.cd._ccg_highres))
            if has_highres:
                hi_bs = _CCG_RESOLUTION['highres']
                ccg_hi, pred_hi, pval_hi, pvalc_hi = _run_pipeline(
                    hi_bs, 'highres')
                result['ccg_hi'] = ccg_hi
                result['ccg_null_hi'] = pred_hi
                result['pval_hi'] = pval_hi
                result['pval_corrected_hi'] = pvalc_hi

            return result
        except Exception as ex:
            print(f"[CustomSegment] ERROR: {ex}")
            import traceback; traceback.print_exc()
            messagebox.showerror("Custom CCG",
                                 f"Error computing CCG:\n{ex}")
            return None

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
        # Base spacing scaled by H/V sliders
        h_scale = getattr(self, '_net_hzoom_var', None)
        v_scale = getattr(self, '_net_vzoom_var', None)
        shank_gap = 150.0 * (h_scale.get() if h_scale else 1.0)
        chan_gap  = 20.0  * (v_scale.get() if v_scale else 1.0)
        x = np.asarray(neurons.shank_ids, dtype=float) * shank_gap
        y = (np.asarray(neurons.peak_channels, dtype=float) % ch_per_shank) * chan_gap
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

    # Connection-type color palette — high-saturation, distinct
    _NET_TYPE_COLOR = {
        ('pyr', 'pyr'):     '#D32F2F',   # red
        ('pyr', 'inter'):   '#DAA520',   # gold
        ('inter', 'pyr'):   '#2E7D32',   # green
        ('inter', 'inter'): '#1565C0',   # blue
    }
    _NET_DEFAULT_E = '#D32F2F'
    _NET_DEFAULT_I = '#1565C0'

    def _draw_network(self):
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.lines import Line2D

        try:
            self._draw_network_impl()
        except Exception as ex:
            print(f"[CCGReviewUI] ERROR in _draw_network: {ex}")
            import traceback; traceback.print_exc()

    def _draw_network_impl(self):
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.lines import Line2D

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
        n_neurons = len(x_pos)
        fn = self._focused_neuron  # focused neuron id or None
        fp = self._focused_pair    # focused (ref, tgt) pair or None
        gf = self._net_group_filter  # group filter name or None
        shank_ids = (getattr(self.neurons, 'shank_ids', None)
                     if self.neurons is not None else None)

        # ── Gather pairs: current type only by default; all types when focused ──
        if gf is not None or fn is not None or fp is not None:
            type_keys_show = self._available_type_keys(self.key.nd())
        else:
            type_keys_show = [self.key]

        group_pairs = self._groups.get(gf, set()) if gf is not None else None
        visible_pairs_current = self._pairs_for_segment_filter()
        current_pair = (tuple(self.all_inds[self.current_pair_idx])
                        if self.current_pair_idx < len(self.all_inds) else None)

        pair_entries: dict = {}
        for tk_ in type_keys_show:
            pt = self.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            ct = getattr(tk_, 'conn_type', None)
            ei = getattr(tk_, 'excitability', 'E')
            is_cur = (tk_ == self.key)
            arr = pt.inds[:, -2:]
            for ref, tgt in map(tuple, arr):
                # Group filter: only include pairs in the selected group
                if gf is not None and (ref, tgt) not in group_pairs:
                    continue
                # In focus mode keep only pairs that connect to the focused neuron
                if gf is None and fn is not None and ref != fn and tgt != fn:
                    continue
                key_t = (ref, tgt)
                if key_t not in pair_entries:
                    pair_entries[key_t] = []
                pair_entries[key_t].append({
                    'key':       tk_,
                    'conn_type': ct,
                    'ei':        ei,
                    'is_current': is_cur,
                    'in_filter': (ref, tgt) in visible_pairs_current if is_cur
                                 else True,
                    'is_selected': (ref, tgt) in self.selected_inds if is_cur
                                   else False,
                })

        # ── Neuron sets ──────────────────────────────────────────────────
        cur_arr = self.ccg_pointer.inds[:, -2:]
        cluster_neurons = set(int(v) for v in np.unique(cur_arr))
        all_involved = set()
        for ref, tgt in pair_entries:
            all_involved.add(ref)
            all_involved.add(tgt)

        nt = (self.neurons.neuron_type
              if self.neurons is not None and
              self.neurons.neuron_type is not None else None)

        # ── Build per-neuron connection-type color map (focus mode) ──────
        # In focus mode, color each connected neuron by its conn_type to the
        # focused neuron.  Map neuron → best conn_type color.
        neuron_ct_color: dict = {}
        if fn is not None:
            for (ref, tgt), entries in pair_entries.items():
                partner = tgt if ref == fn else (ref if tgt == fn else None)
                if partner is None or partner == fn:
                    continue
                for entry in entries:
                    ct = entry['conn_type']
                    c = self._NET_TYPE_COLOR.get(
                        ct, self._NET_DEFAULT_E if entry['ei'] == 'E'
                        else self._NET_DEFAULT_I)
                    neuron_ct_color[partner] = c  # last wins; usually one type

        # ── Draw neurons ──────────────────────────────────────────────────
        # When hiding unconnected, only show neurons that have an actual edge
        # in pair_entries (all_involved).  Otherwise also include cluster_neurons.
        unconnected_o, unconnected_s = [], []  # pyr / inter with no connections
        connected_by_color: dict = {}  # color → (list_o, list_s)
        # In focus-pair mode, the two neurons in the pair are drawn separately
        fp_neurons = set()
        if fp is not None:
            fp_neurons = {fp[0], fp[1]}

        for idx in range(n_neurons):
            if fn is not None and idx == fn:
                continue    # focused neuron drawn individually below
            if idx in fp_neurons:
                continue    # focused pair neurons drawn individually below
            ntype = nt[idx] if nt is not None else None
            is_inter = (ntype == 'inter')
            if self._net_hide_unconnected:
                in_any = idx in all_involved
            else:
                in_any = idx in all_involved or idx in cluster_neurons
            if in_any:
                # Pick color: conn-type color in focus mode, default otherwise
                if fn is not None and idx in neuron_ct_color:
                    c = neuron_ct_color[idx]
                else:
                    c = '#1565C0' if is_inter else '#2E7D32'
                if fp is not None:
                    # Dim all other neurons when pair is focused
                    c = '#9E9E9E'
                if c not in connected_by_color:
                    connected_by_color[c] = ([], [])
                (connected_by_color[c][1] if is_inter
                 else connected_by_color[c][0]).append(idx)
            else:
                (unconnected_s if is_inter else unconnected_o).append(idx)

        def _scatter(indices, marker, color, size, zo, alpha=1.0):
            if indices:
                ax.scatter(x_pos[indices], y_pos[indices],
                           s=size, marker=marker, color=color,
                           zorder=zo, linewidths=0, edgecolors='none',
                           alpha=alpha)

        # Unconnected neurons: transparent gray (hidden when toggle is on)
        if not self._net_hide_unconnected:
            _scatter(unconnected_o, 'o', '#9E9E9E', 14, 1, alpha=0.25)
            _scatter(unconnected_s, 's', '#9E9E9E', 14, 1, alpha=0.25)
        # Connected neurons: colored by connection type (focus) or neuron type
        for color, (o_list, s_list) in connected_by_color.items():
            a = 0.3 if fp is not None else 1.0
            _scatter(o_list, 'o', color, 50 if fp is None else 20, 4, alpha=a)
            _scatter(s_list, 's', color, 50 if fp is None else 20, 4, alpha=a)
        # Focused neuron (single neuron mode)
        if fn is not None and 0 <= fn < n_neurons:
            fn_ntype = nt[fn] if nt is not None else None
            fn_marker = 's' if fn_ntype == 'inter' else 'o'
            ax.scatter([x_pos[fn]], [y_pos[fn]], s=140, marker=fn_marker,
                       color='#FF6F00', zorder=6, linewidths=2.0,
                       edgecolors='black')
        # Focused pair neurons
        if fp is not None:
            for i, (nid, clr) in enumerate([(fp[0], '#FF6F00'), (fp[1], '#1E88E5')]):
                if 0 <= nid < n_neurons:
                    ntype = nt[nid] if nt is not None else None
                    m = 's' if ntype == 'inter' else 'o'
                    ax.scatter([x_pos[nid]], [y_pos[nid]], s=140, marker=m,
                               color=clr, zorder=6, linewidths=2.0,
                               edgecolors='black')

        # ── Draw edges (arrows) ──────────────────────────────────────────
        # Build a set of all (ref,tgt) so we know if a reverse edge exists
        # (for arc-offset to keep both arrows visible)
        all_pair_set = set(pair_entries.keys())

        for (ref, tgt), entries in pair_entries.items():
            if not self._net_show_arrows:
                break
            has_reverse = (tgt, ref) in all_pair_set
            # Slight curve when a pair goes both directions
            rad = 0.18 if has_reverse else 0.0

            for entry in entries:
                ct       = entry['conn_type']
                ei       = entry['ei']
                is_cur   = entry['is_current']
                in_filt  = entry['in_filter']
                is_sel   = entry['is_selected']
                is_cpair = (is_cur and (ref, tgt) == current_pair)

                # Skip if this connection type is toggled off
                if ct is not None and not self._net_ct_vars.get(ct, tk.BooleanVar(value=True)).get():
                    continue
                # Skip same-shank pairs when toggle is on
                if (self._net_hide_same_shank_var.get()
                        and shank_ids is not None
                        and int(shank_ids[ref]) == int(shank_ids[tgt])):
                    continue

                # Determine color — always use connection-type palette
                ec = self._NET_TYPE_COLOR.get(
                    ct, self._NET_DEFAULT_E if ei == 'E'
                    else self._NET_DEFAULT_I)

                is_fp = (fp is not None and (ref, tgt) == fp)

                if is_fp:
                    alpha, lw, zo = 1.00, 3.0, 7
                elif fp is not None:
                    # Dim all other arrows when pair is focused
                    alpha, lw, zo = 0.12, 0.3, 1
                    ec = '#CCCCCC'
                elif not in_filt:
                    alpha, lw, zo = 0.20, 0.4, 1
                    ec = '#CCCCCC'
                elif not is_cur:
                    alpha, lw, zo = 0.35, 0.6, 2
                elif is_cpair:
                    alpha, lw, zo = 1.00, 3.0, 7
                    ec = 'black'
                elif is_sel:
                    alpha, lw, zo = 0.90, 1.8, 4
                else:
                    alpha, lw, zo = 0.55, 0.9, 3

                mutation = 10 if is_cpair else 7
                # In focus mode, all visible arrows are pickable (for type-jump)
                pickable = (in_filt and
                            (fn is not None or is_cur) and
                            (fn is None or ref == fn or tgt == fn))

                arrow = FancyArrowPatch(
                    (x_pos[ref], y_pos[ref]),
                    (x_pos[tgt], y_pos[tgt]),
                    arrowstyle='->', color=ec,
                    linewidth=lw, alpha=alpha,
                    mutation_scale=mutation,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=5, shrinkB=5,
                    zorder=zo,
                    picker=6 if pickable else False,
                )
                # Encode ref_tgt_keystr so pick handler can switch type
                arrow.set_gid(f"{ref}_{tgt}_{entry['key']}")
                ax.add_patch(arrow)

        # ── Dashed arrow for non-existent focused pair ────────────────────
        if (fp is not None and fp not in pair_entries
                and self._net_show_arrows
                and 0 <= fp[0] < n_neurons and 0 <= fp[1] < n_neurons):
            from matplotlib.patches import FancyArrowPatch as _FAP
            dashed = _FAP(
                (x_pos[fp[0]], y_pos[fp[0]]),
                (x_pos[fp[1]], y_pos[fp[1]]),
                arrowstyle='->', color='#888888',
                linewidth=1.5, alpha=0.7,
                linestyle='--',
                mutation_scale=8,
                connectionstyle='arc3,rad=0',
                shrinkA=5, shrinkB=5,
                zorder=7,
            )
            ax.add_patch(dashed)

        # ── Legend ───────────────────────────────────────────────────────
        shown_types = set()
        for (ref, tgt), entries in pair_entries.items():
            for entry in entries:
                if entry['conn_type'] is not None:
                    shown_types.add(entry['conn_type'])

        _ct_label = {
            ('pyr', 'pyr'):     'pyr→pyr',
            ('pyr', 'inter'):   'pyr→int',
            ('inter', 'inter'): 'int→int',
            ('inter', 'pyr'):   'int→pyr',
        }
        legend_handles = []
        for ct in [('pyr', 'pyr'), ('pyr', 'inter'),
                   ('inter', 'inter'), ('inter', 'pyr')]:
            if ct in shown_types and self._net_ct_vars.get(ct, tk.BooleanVar(value=True)).get():
                legend_handles.append(
                    Line2D([0], [0], color=self._NET_TYPE_COLOR[ct],
                           lw=2, label=_ct_label[ct]))
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=6, loc='lower left',
                      framealpha=0.75, handlelength=1.4)

        # ── Shank labels at top ──────────────────────────────────────────
        if shank_ids is not None:
            unique_shanks = np.unique(shank_ids)
            h_scale = getattr(self, '_net_hzoom_var', None)
            shank_gap = 150.0 * (h_scale.get() if h_scale else 1.0)
            y_top = np.max(y_pos) + 20
            for sk in unique_shanks:
                sx = float(sk) * shank_gap
                ax.text(sx, y_top, f"S{int(sk)}",
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='#555555')

        ax.axis('off')
        ax.set_aspect('equal')
        self.net_fig.tight_layout(pad=0.5)
        self.net_canvas.draw()

    def _on_network_pick(self, event):
        gid = getattr(event.artist, 'get_gid', lambda: None)()
        if not gid:
            return
        try:
            parts = gid.split('_', 2)
            ref, tgt = int(parts[0]), int(parts[1])
            key_str = parts[2] if len(parts) > 2 else None
        except (ValueError, AttributeError, IndexError):
            return

        # If the pair belongs to a different type key, switch to it
        if key_str is not None and self._focused_neuron is not None:
            # Find the matching key from available type keys
            for tk_ in self._available_type_keys(self.key.nd()):
                if str(tk_) == key_str and tk_ != self.key:
                    self._switch_key(tk_)
                    self._refresh_after_key_switch()
                    # Update type combo if present
                    if hasattr(self, 'type_combo'):
                        self.type_combo.set(self._type_label(tk_))
                    break

        self.current_pair_idx = self.get_pair_index((ref, tgt))
        self.update_plot()
        self._draw_network()

    def _on_net_scroll(self, event):
        """Scroll wheel: adjust both spacing sliders and redraw."""
        if event.inaxes != self.net_ax:
            return
        factor = 1.15 if event.step > 0 else 0.87
        cur_vz = self._net_vzoom_var.get()
        cur_hz = self._net_hzoom_var.get()
        self._net_vzoom_var.set(min(max(cur_vz * factor, 0.2), 5.0))
        self._net_hzoom_var.set(min(max(cur_hz * factor, 0.2), 5.0))
        self._draw_network()

    # ------------------------------------------------------------------
    # Neuron focus (Part II.1)
    # ------------------------------------------------------------------

    def _on_neuron_focus(self):
        val = self._focus_var.get().strip()
        print(f"[CCGReviewUI] _on_neuron_focus called, val={val!r}")
        if not val:
            self._on_neuron_focus_clear()
            return
        try:
            nid = int(val)
        except ValueError:
            messagebox.showerror("Neuron focus", f"Invalid neuron id: {val!r}")
            return
        if self.neurons is not None:
            if nid < 0 or nid >= self.neurons.n_neurons:
                messagebox.showerror("Neuron focus",
                                     f"Neuron {nid} out of range "
                                     f"[0, {self.neurons.n_neurons-1}]")
                return
        self._focused_neuron = nid
        self._focused_pair = None
        self._focus_pair_var.set("")
        self._focus_pair_info_var.set("")
        self._update_focus_info(nid)
        self.refresh_lists()
        self._draw_network()

    def _update_focus_info(self, nid):
        """Update focus info label with current-type and total connection counts."""
        # Current type counts
        cur_out = sum(1 for r, t in map(tuple, self.all_inds) if r == nid)
        cur_in  = sum(1 for r, t in map(tuple, self.all_inds) if t == nid)
        # Total across all types
        tot_out, tot_in = 0, 0
        for tk_ in self._available_type_keys(self.key.nd()):
            pt = self.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            arr = pt.inds[:, -2:]
            tot_out += sum(1 for r, t in set(map(tuple, arr)) if r == nid)
            tot_in  += sum(1 for r, t in set(map(tuple, arr)) if t == nid)
        ct_label = self._type_label(self.key)
        self._focus_info_var.set(
            f"{ct_label}: in={cur_in} out={cur_out}  |  all: in={tot_in} out={tot_out}")

    def _on_neuron_focus_clear(self):
        self._focused_neuron = None
        self._focus_var.set("")
        self._focus_info_var.set("")
        self.refresh_lists()
        self._draw_network()

    def _on_pair_focus(self):
        """Set focus to a specific (ref, tgt) pair. Clears neuron focus."""
        val = self._focus_pair_var.get().strip()
        if not val:
            self._on_pair_focus_clear()
            return
        try:
            parts = val.replace(' ', '').split(',')
            ref, tgt = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            messagebox.showerror("Pair focus",
                                 f"Invalid pair format: {val!r}\nUse ref,tgt (e.g. 1,170)")
            return
        # Validate neuron indices exist
        if self.neurons is not None:
            n = self.neurons.n_neurons
            if ref < 0 or ref >= n or tgt < 0 or tgt >= n:
                messagebox.showerror("Pair focus",
                                     f"Neuron index out of range [0, {n-1}]")
                return
        # Check if pair exists in any type
        pair = (ref, tgt)
        pair_exists = False
        for tk_ in self._available_type_keys(self.key.nd()):
            pt = self.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            if pair in set(map(tuple, pt.inds2)):
                pair_exists = True
                break
        if not pair_exists:
            self._show_temp_warning(f"Pair ({ref},{tgt}) not significant — showing position")
        self._focused_pair = pair
        self._focused_neuron = None
        self._focus_var.set("")
        self._focus_info_var.set("")
        # Update pair info
        self._update_pair_focus_info(pair, pair_exists)
        self.refresh_lists()
        self._draw_network()
        self.update_plot()

    def _update_pair_focus_info(self, pair, exists):
        """Update the pair focus info label and 'Add to available' button."""
        ref, tgt = pair
        if self.neurons is not None:
            nt = self.neurons.neuron_type
            ref_type = nt[ref] if nt is not None and ref < len(nt) else '?'
            tgt_type = nt[tgt] if nt is not None and tgt < len(nt) else '?'
            # Check if already in all_inds (sig or admitted)
            in_available = pair in set(map(tuple, self.all_inds))
            status = "sig" if exists else ("admitted" if in_available else "not sig")
            self._focus_pair_info_var.set(
                f"{ref}({ref_type})→{tgt}({tgt_type}) [{status}]")
        else:
            in_available = pair in set(map(tuple, self.all_inds))
            self._focus_pair_info_var.set(f"{ref}→{tgt}")
        # Enable "Add to available" only for pairs not already available
        if hasattr(self, '_add_pair_btn'):
            self._add_pair_btn.config(
                state=tk.NORMAL if not in_available else tk.DISABLED)

    def _on_pair_focus_clear(self):
        self._focused_pair = None
        self._focus_pair_var.set("")
        self._focus_pair_info_var.set("")
        if hasattr(self, '_add_pair_btn'):
            self._add_pair_btn.config(state=tk.DISABLED)
        self.refresh_lists()
        self._draw_network()
        self.update_plot()

    def _on_add_focused_pair(self):
        """Add the currently focused non-significant pair to available pairs."""
        pair = self._focused_pair
        if pair is None:
            return
        ref, tgt = pair
        # Validate indices are within CCG data shape
        cd = self.ccg_data
        if cd is None or cd.ccg is None:
            messagebox.showerror("Add pair", "No CCG data loaded.")
            return
        if ref >= cd.ccg.shape[1] or tgt >= cd.ccg.shape[2]:
            messagebox.showerror("Add pair",
                                 f"Pair ({ref},{tgt}) out of CCG data range "
                                 f"({cd.ccg.shape[1]}x{cd.ccg.shape[2]}).")
            return
        # Add to admitted group
        self._push_undo()
        self._groups.setdefault(_ADMITTED_GROUP, set()).add(pair)
        # Add to unselected (now a valid available pair)
        self.unselected_inds.add(pair)
        # Navigate to the pair
        self.current_pair_idx = self.get_pair_index(pair)
        # Clear focus and update UI
        self._focused_pair = None
        self._focus_pair_var.set("")
        self._update_pair_focus_info(pair, exists=False)
        self._add_pair_btn.config(state=tk.DISABLED)
        self.refresh_lists()
        self._draw_network()
        self.update_plot()

    def _on_net_toggle_arrows(self):
        self._net_show_arrows = self._net_arrows_var.get()
        self._draw_network()

    def _on_net_toggle_hide(self):
        self._net_hide_unconnected = self._net_hide_var.get()
        self._draw_network()

    def _on_net_toggle_hide_same_shank(self):
        self.refresh_lists()
        self._draw_network()

    def _on_net_group_select(self, _=None):
        """Called when user picks a group from the probe-network dropdown."""
        val = self._net_group_var.get()
        if val == '(none)':
            self._net_group_filter = None
        else:
            self._net_group_filter = val
            # Clear focus — group filter overrides focus
            self._focused_neuron = None
            self._focus_var.set("")
            self._focus_info_var.set("")
        self._draw_network()

    def _refresh_net_group_combo(self):
        """Update the group combobox values from self._groups."""
        if not hasattr(self, '_net_group_combo'):
            return
        names = ['(none)'] + sorted(k for k in self._groups if not k.startswith('__'))
        self._net_group_combo['values'] = names
        # If the current selection was deleted, reset
        if self._net_group_var.get() not in names:
            self._net_group_var.set('(none)')
            self._net_group_filter = None

    def _on_net_zoom(self, _=None):
        """Called when H or V zoom slider changes — redraws with new spacing."""
        self._draw_network()

    # ------------------------------------------------------------------
    # Modified _draw_network (neuron focus support)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Group helpers (Part II.2)
    # ------------------------------------------------------------------

    def _pair_group_label(self, inds) -> str:
        """Return a short label like '[G1,G2]' for the groups this pair belongs to."""
        pair = tuple(inds)
        tags = [gname for gname, pairs in self._groups.items()
                if pair in pairs and not gname.startswith('__')]
        return f"[{','.join(tags)}]" if tags else ""

    def _toggle_pair_group(self, pair, group_name):
        if group_name not in self._groups:
            self._groups[group_name] = set()
        if pair in self._groups[group_name]:
            self._groups[group_name].discard(pair)
        else:
            self._groups[group_name].add(pair)
        # Preserve scroll positions so the list doesn't jump to the top
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    def _toggle_pairs_group(self, pairs, group_name):
        """Toggle multiple pairs in/out of a group.

        If ALL pairs are already in the group, remove them all.
        Otherwise, add all pairs to the group.
        """
        if group_name not in self._groups:
            self._groups[group_name] = set()
        all_in = all(p in self._groups[group_name] for p in pairs)
        if all_in:
            for p in pairs:
                self._groups[group_name].discard(p)
        else:
            for p in pairs:
                self._groups[group_name].add(p)
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    def _create_group_dialog(self):
        name = simpledialog.askstring(
            "Create group", "Group name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self._groups:
            messagebox.showinfo("Create group", f"Group '{name}' already exists.")
            return
        self._groups[name] = set()
        self._rebuild_groups_menu()
        self.refresh_lists()

    def _manage_groups_dialog(self):
        """Pop-up window to rename groups, view pairs, assign hotkeys."""
        if not self._groups:
            messagebox.showinfo("Manage groups", "No groups yet. Create one first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Manage Groups")
        win.geometry("480x420")
        win.transient(self.root)
        win.grab_set()

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        for gname in sorted(self._groups):
            if gname.startswith('__'):
                continue
            frame = ttk.Frame(nb)
            nb.add(frame, text=gname)

            top = ttk.Frame(frame)
            top.pack(fill=tk.X, padx=6, pady=4)
            ttk.Label(top, text="Name:").pack(side=tk.LEFT)
            name_var = tk.StringVar(value=gname)
            ttk.Entry(top, textvariable=name_var, width=18).pack(
                side=tk.LEFT, padx=4)
            ttk.Button(top, text="Rename",
                       command=lambda old=gname, nv=name_var: self._rename_group(old, nv.get(), win)).pack(
                side=tk.LEFT)

            hk_frame = ttk.Frame(frame)
            hk_frame.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(hk_frame, text="Hotkey (Ctrl+1…0):").pack(side=tk.LEFT)
            hk_var = tk.StringVar(value=self._group_hotkeys.get(gname, ''))
            hk_entry = ttk.Entry(hk_frame, textvariable=hk_var, width=6)
            hk_entry.pack(side=tk.LEFT, padx=4)
            ttk.Button(hk_frame, text="Set",
                       command=lambda g=gname, hv=hk_var: self._set_group_hotkey(g, hv.get())).pack(
                side=tk.LEFT)

            # Notes
            ttk.Label(frame, text="Notes:").pack(anchor='w', padx=6, pady=(4, 0))
            notes_text = tk.Text(frame, height=3, width=40, font=('Arial', 9),
                                 wrap=tk.WORD)
            notes_text.pack(fill=tk.X, padx=6, pady=2)
            notes_text.insert('1.0', self._group_notes.get(gname, ''))
            # Auto-save notes on every keystroke
            notes_text.bind('<KeyRelease>',
                            lambda e, g=gname, t=notes_text:
                            self._group_notes.__setitem__(g, t.get('1.0', 'end-1c')))

            ttk.Label(frame, text="Pairs in group:").pack(anchor='w', padx=6)
            lb_frame = ttk.Frame(frame)
            lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
            sb = ttk.Scrollbar(lb_frame)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb = tk.Listbox(lb_frame, yscrollcommand=sb.set, font=('Courier', 9))
            lb.pack(fill=tk.BOTH, expand=True)
            sb.config(command=lb.yview)
            for pair in sorted(self._groups.get(gname, [])):
                lb.insert(tk.END, f"[{pair[0]:3d}, {pair[1]:3d}]")
            ttk.Button(frame, text=f"Delete group '{gname}'",
                       command=lambda g=gname: self._delete_group(g, win)).pack(
                pady=4)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)

    def _rename_group(self, old_name, new_name, win=None):
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if new_name in self._groups:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")
            return
        self._groups[new_name] = self._groups.pop(old_name)
        if old_name in self._group_hotkeys:
            self._group_hotkeys[new_name] = self._group_hotkeys.pop(old_name)
        if old_name in self._group_notes:
            self._group_notes[new_name] = self._group_notes.pop(old_name)
        self._rebuild_groups_menu()
        self.refresh_lists()
        if win:
            win.destroy()
            self._manage_groups_dialog()

    def _delete_group(self, name, win=None):
        if not messagebox.askyesno("Delete group",
                                   f"Delete group '{name}'?"):
            return
        self._groups.pop(name, None)
        self._group_hotkeys.pop(name, None)
        self._group_notes.pop(name, None)
        self._rebuild_groups_menu()
        self.refresh_lists()
        if win:
            win.destroy()

    def _merge_groups_dialog(self):
        """Dialog to merge two or more groups into one."""
        if len(self._groups) < 2:
            messagebox.showinfo("Merge groups",
                                "Need at least 2 groups to merge.")
            return
        win = tk.Toplevel(self.root)
        win.title("Merge Groups")
        win.geometry("340x320")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select groups to merge:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)
        check_vars = {}
        for gname in sorted(self._groups):
            if gname.startswith('__'):
                continue
            var = tk.BooleanVar(value=False)
            check_vars[gname] = var
            ttk.Checkbutton(frame, text=gname, variable=var).pack(anchor='w')

        name_frame = ttk.Frame(win)
        name_frame.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(name_frame, text="Merged group name:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, width=20)
        name_entry.pack(side=tk.LEFT, padx=4)

        def do_merge():
            selected = [g for g, v in check_vars.items() if v.get()]
            if len(selected) < 2:
                messagebox.showwarning("Merge", "Select at least 2 groups.")
                return
            target = name_entry.get().strip() or selected[0]
            if not messagebox.askokcancel(
                    "Merge groups",
                    f"This will merge {len(selected)} groups into '{target}'.\n"
                    "This cannot be undone. Proceed?"):
                return
            merged_pairs = set()
            for g in selected:
                merged_pairs |= self._groups.get(g, set())
                if g != target:
                    self._groups.pop(g, None)
                    self._group_hotkeys.pop(g, None)
                    self._group_notes.pop(g, None)
            self._groups[target] = merged_pairs
            self._rebuild_groups_menu()
            self.refresh_lists()
            win.destroy()

        ttk.Button(win, text="Merge", command=do_merge).pack(pady=8)

    def _set_group_hotkey(self, group_name, key_str):
        """Assign hotkey string like '1', '2' … '0' (maps to Ctrl+n)."""
        key_str = key_str.strip()
        if not key_str:
            self._group_hotkeys.pop(group_name, None)
            return
        if key_str not in [str(i) for i in range(1, 10)] + ['0']:
            messagebox.showwarning("Hotkey", "Enter a digit 1–9 or 0 only.")
            return
        # Remove from any other group that had this key
        for g, k in list(self._group_hotkeys.items()):
            if k == key_str and g != group_name:
                del self._group_hotkeys[g]
        self._group_hotkeys[group_name] = key_str
        print(f"[_set_group_hotkey] assigned '{group_name}' → Ctrl+{key_str}, "
              f"hotkeys now={self._group_hotkeys}, id={id(self._group_hotkeys)}")
        self._rebuild_groups_menu()

    def _rebuild_groups_menu(self):
        """Refresh the dynamic part of the Groups menu."""
        if not hasattr(self, '_groups_menu'):
            return
        # Remove items after the separator (index 6+)
        try:
            while self._groups_menu.index('end') >= 6:
                self._groups_menu.delete(6)
        except tk.TclError:
            pass
        current_pairs = set(map(tuple, self.all_inds)) if len(self.all_inds) else set()
        for gname in sorted(self._groups):
            if gname.startswith('__'):
                continue  # hide internal groups like __admitted__
            hk = self._group_hotkeys.get(gname, '')
            n_in_session = len(self._groups.get(gname, set()) & current_pairs)
            label = f"{gname} ({n_in_session})" + (f" [Ctrl+{hk}]" if hk else "")
            self._groups_menu.add_command(
                label=label,
                command=lambda g=gname: self._select_group(g))
        # Also refresh the probe-network group dropdown and hotkeys bar
        self._refresh_net_group_combo()
        if (hasattr(self, '_hotkeys_bar') and
                self._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get()):
            self._refresh_hotkeys_bar()

    def _select_group(self, group_name):
        """Navigate to the first pair in the group."""
        pairs = self._groups.get(group_name, set())
        if not pairs:
            return
        first = sorted(pairs)[0]
        self.current_pair_idx = self.get_pair_index(first)
        self.update_plot()
        self._draw_network()

    def _group_hotkey_handler(self, n: int):
        """Called when Ctrl+n is pressed (n=0..9).

        Toggles the current pair in/out of the group assigned to that hotkey.
        """
        key_str = str(n + 1) if n < 9 else '0'
        if self.current_pair_idx >= len(self.all_inds):
            return
        pair = tuple(self.all_inds[self.current_pair_idx])
        for gname, k in self._group_hotkeys.items():
            if k == key_str:
                self._toggle_pair_group(pair, gname)
                # Also move to Selected side if currently in Available
                if pair in self.unselected_inds:
                    self._push_undo()
                    self.unselected_inds.discard(pair)
                    self.selected_inds.add(pair)
                    unsel_scroll = self.unselected_list.yview()[0]
                    sel_scroll = self.selected_list.yview()[0]
                    self.refresh_lists()
                    self.unselected_list.yview_moveto(unsel_scroll)
                    self.selected_list.yview_moveto(sel_scroll)
                    self._draw_network()
                return
        # No group assigned to this hotkey — show temporary warning
        self._show_temp_warning(f"No group assigned to Ctrl+{key_str}")

    def _show_temp_warning(self, msg: str, duration_ms: int = 2000):
        """Show a temporary warning label at the top of the window that auto-disappears."""
        lbl = tk.Label(self.root, text=msg, bg='#FFF3CD', fg='#856404',
                       font=('Arial', 10, 'bold'), padx=8, pady=4)
        lbl.place(relx=0.5, y=4, anchor='n')
        self.root.after(duration_ms, lbl.destroy)

    # ------------------------------------------------------------------
    # Group export / import
    # ------------------------------------------------------------------

    def _export_groups(self):
        """Export all group definitions to a standalone JSON file."""
        if not self._groups:
            messagebox.showinfo("Export groups", "No groups to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export groups",
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
            initialfile='groups_export.json',
            initialdir=self._sel_save_dir,
        )
        if not path:
            return
        groups = {g: [[int(r), int(c)] for r, c in sorted(pairs)]
                  for g, pairs in self._groups.items()}
        data = {
            'groups': groups,
            'hotkeys': dict(self._group_hotkeys),
            'notes': dict(self._group_notes),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)
        print(f"[CCGReviewUI] groups exported → {path}")

    def _import_groups(self):
        """Import group definitions from a JSON file, merging with existing."""
        path = filedialog.askopenfilename(
            title="Import groups",
            filetypes=[('JSON files', '*.json')],
            initialdir=self._sel_save_dir,
        )
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            messagebox.showerror("Import groups", f"Failed to read file:\n{exc}")
            return
        imported_groups = data.get('groups', {})
        for gname, pairs in imported_groups.items():
            merged = self._groups.get(gname, set())
            merged |= set(tuple(int(v) for v in p) for p in pairs)
            self._groups[gname] = merged
        for gname, hk in data.get('hotkeys', {}).items():
            if gname not in self._group_hotkeys:
                self._group_hotkeys[gname] = hk
        for gname, note in data.get('notes', {}).items():
            if gname not in self._group_notes:
                self._group_notes[gname] = note
        self._rebuild_groups_menu()
        self.refresh_lists()
        print(f"[CCGReviewUI] groups imported from {path}")

    # ------------------------------------------------------------------
    # Versioning helpers (Part II.3)
    # ------------------------------------------------------------------

    def _sel_version_path(self, name: str) -> str:
        safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        session_tag = getattr(self.key, 'session', 'sess')
        return os.path.join(self._sel_save_dir, f"{session_tag}__{safe}.json")

    @staticmethod
    def _json_default(obj):
        """JSON encoder that converts numpy integer/float types to Python scalars."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    def _save_selection_version(self, name: str):
        """Persist selections for ALL types in the session to a single JSON."""
        # Flush current type's selections to its pointer
        self.ccg_pointer.manually_selected_inds = (
            np.array(sorted(self.selected_inds), dtype=int)
            if self.selected_inds else None
        )

        # Collect selections for every type key in this session
        type_keys = self._available_type_keys(self.key.nd())
        selections_by_type = {}
        for tk_ in type_keys:
            ptr = self.cd.data.get(tk_)
            if ptr is None:
                continue
            sel = getattr(ptr, 'manually_selected_inds', None)
            if sel is not None and len(sel) > 0:
                selections_by_type[str(tk_)] = [[int(r), int(c)]
                                                 for r, c in sorted(map(tuple, sel))]
            else:
                selections_by_type[str(tk_)] = []

        groups = {g: [[int(r), int(c)] for r, c in sorted(pairs)]
                  for g, pairs in self._groups.items()}
        data = {
            'version': '2.0',
            'name': name,
            'saved_at': datetime.datetime.now().isoformat(),
            'session': getattr(self.key, 'session', 'sess'),
            'selections': selections_by_type,
            'groups': groups,
            'hotkeys': dict(self._group_hotkeys),
            'notes': dict(self._group_notes),
        }
        path = self._sel_version_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)
        print(f"[CCGReviewUI] selection saved → {path}")
        return path

    def _list_selection_versions(self) -> list:
        """Return list of (name, path, saved_at, is_valid) for all matching versions."""
        session_tag = getattr(self.key, 'session', 'sess')
        prefix = session_tag + '__'
        versions = []
        if not os.path.isdir(self._sel_save_dir):
            return versions
        all_files = sorted(os.listdir(self._sel_save_dir))
        print(f"[list_versions] session_tag={session_tag!r}, prefix={prefix!r}, "
              f"files={[f for f in all_files if f.endswith('.json')]}")
        for fname in all_files:
            if not fname.startswith(prefix) or not fname.endswith('.json'):
                continue
            path = os.path.join(self._sel_save_dir, fname)
            try:
                with open(path, encoding='utf-8') as f:
                    meta = json.load(f)
                versions.append((meta.get('name', fname), path,
                                 meta.get('saved_at', ''), True))
            except Exception:
                versions.append((fname, path, '⚠ corrupted', False))
        return versions

    def _load_selection_from_file(self, path: str, restore_groups: bool = True):
        """Load selection from a JSON file (v1.0 or v2.0).

        If restore_groups is False, only pair selections are loaded — groups,
        hotkeys, and notes are left untouched.  Used by autoload on session
        switch (groups are shared across sessions).
        """
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON parse error in {os.path.basename(path)}.\n\n"
                f"The file is likely corrupted from a previous save attempt.\n"
                f"Technical detail: {exc}\n\n"
                f"Please delete this file and re-save your selection."
            ) from exc

        version = data.get('version', '1.0')
        if version >= '2.0' and 'selections' in data:
            # v2.0: session-level — restore each type's selections
            selections_by_type = data.get('selections', {})
            type_keys = self._available_type_keys(self.key.nd())
            total_loaded = 0
            for tk_ in type_keys:
                ptr = self.cd.data.get(tk_)
                if ptr is None:
                    continue
                pairs = selections_by_type.get(str(tk_), [])
                if pairs:
                    ptr.manually_selected_inds = np.array(
                        [[int(r), int(c)] for r, c in pairs], dtype=int)
                    total_loaded += len(pairs)
                else:
                    ptr.manually_selected_inds = None
            # Apply current type's selections to UI
            cur_sel = selections_by_type.get(str(self.key), [])
            selected = set(tuple(int(v) for v in p) for p in cur_sel)
        else:
            # v1.0 backward compat: single-type selections
            # Only apply selections if the file's key matches the current type
            file_key = data.get('key', '')
            if file_key == str(self.key):
                selected = set(tuple(int(v) for v in p)
                               for p in data.get('selected', []))
            else:
                selected = set()

        self._push_undo()
        # Check for pairs that are no longer available (epoch/CCG change)
        current_available = set(map(tuple, self.all_inds))
        missing = selected - current_available
        if missing and restore_groups:
            # Only show dialog on explicit load (not autoload)
            action = self._show_missing_pairs_dialog(missing)
            if action == 'cancel':
                return
            elif action == 'partial':
                selected = selected & current_available
            elif action == 'admit_all':
                self._groups.setdefault(_ADMITTED_GROUP, set()).update(missing)
                # Recompute current_available now that admitted pairs are added
                current_available = set(map(tuple, self.all_inds))
        elif missing:
            # Autoload: silently keep only available pairs
            selected = selected & current_available

        self.selected_inds = selected
        self.unselected_inds = current_available - selected
        if restore_groups:
            self._groups = {g: set(tuple(int(v) for v in p) for p in pairs)
                            for g, pairs in data.get('groups', {}).items()}
            self._groups.setdefault(_ADMITTED_GROUP, set())
            self._group_hotkeys = data.get('hotkeys', {})
            self._group_notes = data.get('notes', {})
            self._rebuild_groups_menu()
        self.refresh_lists()
        self._draw_network()

    def _show_missing_pairs_dialog(self, missing: set) -> str:
        """Dialog when loaded selection has pairs not in current available set.

        Returns 'partial', 'admit_all', or 'cancel'.
        """
        win = tk.Toplevel(self.root)
        win.title("Missing Pairs")
        win.geometry("450x320")
        win.transient(self.root)
        win.grab_set()

        result = {'action': 'cancel'}
        n = len(missing)

        ttk.Label(win, text=f"{n} selected pair(s) are no longer in available pairs:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        ttk.Label(win,
                  text="These pairs may have lost significance after CCG/epoch changes.",
                  font=('Arial', 9), foreground='#666').pack(pady=(0, 4))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        lb = tk.Listbox(frame, font=('Courier', 9))
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        for ref, tgt in sorted(missing):
            lb.insert(tk.END, f"  ({ref:3d}, {tgt:3d})")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        def partial():
            result['action'] = 'partial'
            win.destroy()

        def admit_all():
            result['action'] = 'admit_all'
            win.destroy()

        def cancel():
            result['action'] = 'cancel'
            win.destroy()

        ttk.Button(btn_frame, text="Keep only available",
                   command=partial).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Admit all missing",
                   command=admit_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=cancel).pack(side=tk.RIGHT, padx=4)

        win.protocol('WM_DELETE_WINDOW', cancel)
        win.wait_window()
        return result['action']

    def _load_selection_dialog(self):
        """Show a dialog listing all saved versions; user picks one to load."""
        versions = self._list_selection_versions()
        if not versions:
            messagebox.showinfo("Load selection",
                                "No saved selections found for this key.")
            return
        win = tk.Toplevel(self.root)
        win.title("Load Selection")
        win.geometry("620x340")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select a version to load:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        sb = ttk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(frame, yscrollcommand=sb.set, font=('Courier', 9),
                        selectmode=tk.BROWSE)
        lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=lb.yview)
        for name, path, saved_at, is_valid in versions:
            prefix = '   ' if is_valid else '⚠  '
            lb.insert(tk.END, f"{prefix}{name:28s}  {saved_at[:19]}")
            if not is_valid:
                lb.itemconfig(lb.size() - 1, foreground='#CC4444')

        def do_load():
            sel = lb.curselection()
            if not sel:
                return
            name, path, saved_at, is_valid = versions[sel[0]]
            if not is_valid:
                if not messagebox.askyesno(
                        "Corrupted file",
                        f"'{name}' appears to be corrupted.\n"
                        "Delete it and continue?"):
                    return
                try:
                    os.remove(path)
                except OSError as ex:
                    messagebox.showerror("Delete failed", str(ex))
                win.destroy()
                self._load_selection_dialog()   # reopen with updated list
                return
            try:
                self._load_selection_from_file(path)
                win.destroy()
            except Exception as ex:
                messagebox.showerror("Load selection",
                                     f"Failed to load:\n{ex}")

        def do_delete(event=None):
            sel = lb.curselection()
            if not sel:
                # Select item under cursor for right-click
                idx = lb.nearest(event.y) if event else None
                if idx is not None:
                    lb.selection_clear(0, tk.END)
                    lb.selection_set(idx)
                    sel = (idx,)
                else:
                    return
            name, path, saved_at, is_valid = versions[sel[0]]
            if not messagebox.askyesno(
                    "Delete selection",
                    f"Move '{name}' to deleted folder?",
                    parent=win):
                return
            deleted_dir = os.path.join(self._sel_save_dir, 'deleted')
            os.makedirs(deleted_dir, exist_ok=True)
            import shutil
            try:
                shutil.move(path, os.path.join(deleted_dir, os.path.basename(path)))
            except OSError as ex:
                messagebox.showerror("Delete failed", str(ex), parent=win)
                return
            win.destroy()
            self._load_selection_dialog()  # reopen with updated list

        def _ctx_menu_load(event):
            menu = tk.Menu(win, tearoff=0)
            menu.add_command(label="Delete", command=lambda: do_delete(event))
            menu.tk_popup(event.x_root, event.y_root)

        lb.bind('<Button-2>', _ctx_menu_load)
        lb.bind('<Button-3>', _ctx_menu_load)
        lb.bind('<Double-Button-1>', lambda e: do_load())
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(btn_frame, text="Load", command=do_load).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Cancel",
                   command=win.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Label(btn_frame, text="⚠ = corrupted file (can be deleted)",
                  font=('Arial', 8), foreground='#CC4444').pack(
            side=tk.RIGHT, padx=6)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _do_save(self, name: str):
        """Core save logic: persist all types' selections + groups."""
        try:
            self._save_selection_version(name)
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Save error",
                                 f"Failed to save selection:\n{exc}")
            return

        # Count total selections across all types
        type_keys = self._available_type_keys(self.key.nd())
        total = sum(
            len(self.cd.data[tk_].manually_selected_inds)
            for tk_ in type_keys
            if self.cd.data.get(tk_) is not None
            and getattr(self.cd.data[tk_], 'manually_selected_inds', None) is not None
        )

        # Auto-export groups alongside the selection
        groups_msg = ""
        if self._groups:
            groups = {g: [[int(r), int(c)] for r, c in sorted(pairs)]
                      for g, pairs in self._groups.items()}
            data = {
                'groups': groups,
                'hotkeys': dict(self._group_hotkeys),
                'notes': dict(self._group_notes),
            }
            path = os.path.join(self._sel_save_dir, 'groups_export.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_default)
            groups_msg = f"\nGroups exported ({len(self._groups)} groups)."
            print(f"[CCGReviewUI] groups auto-exported → {path}")

        messagebox.showinfo(
            "Saved",
            f"Saved {total} pairs across {len(type_keys)} types as '{name}'.{groups_msg}",
            parent=self.root)

    def _quick_save(self):
        """Ctrl+S / Save button: custom dialog with name entry + Latest button."""
        default_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        win = tk.Toplevel(self.root)
        win.title("Save selection")
        win.geometry("360x130")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Version name:").pack(pady=(10, 2))
        name_var = tk.StringVar(value=default_name)
        entry = ttk.Entry(win, textvariable=name_var, width=32)
        entry.pack(padx=10)
        entry.select_range(0, tk.END)
        entry.focus_set()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)

        def _save_named():
            name = name_var.get().strip() or default_name
            win.destroy()
            self._do_save(name)

        def _save_latest():
            win.destroy()
            self._do_save('latest')

        ttk.Button(btn_frame, text="Save", command=_save_named).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Save as Latest", command=_save_latest).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(
            side=tk.LEFT, padx=6)

        entry.bind('<Return>', lambda e: _save_named())

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

    def _on_close(self):
        """Prompt user before closing — optionally skip autosave."""
        answer = messagebox.askyesnocancel(
            "Quit",
            "Save current selections before quitting?",
            default=messagebox.YES)
        if answer is None:
            return  # Cancel — don't quit
        if answer:
            self._autosave_current()
        self.root.destroy()

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
