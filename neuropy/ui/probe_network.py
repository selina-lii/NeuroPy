"""Network panel for CCGReviewUI — probe layout with connection arrows."""

import traceback
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

_SPECIAL_PREFIX = "__special_"


class NetworkPanel:
    """Probe network panel — neuron positions + connection arrows.

    Parameters
    ----------
    parent : tk.Widget
        The parent frame (right column of the 3-pane layout).
    ui : CCGReviewUI
        The owning UI instance.  All shared state (neurons, groups,
        focused neuron/pair, etc.) is accessed via ``self.ui``.
    """

    # Connection-type color palette — high-saturation, distinct
    _NET_TYPE_COLOR = {
        ('pyr', 'pyr'):     '#D32F2F',   # red
        ('pyr', 'inter'):   '#DAA520',   # gold
        ('inter', 'pyr'):   '#2E7D32',   # green
        ('inter', 'inter'): '#1565C0',   # blue
    }
    _NET_DEFAULT_E = '#D32F2F'
    _NET_DEFAULT_I = '#1565C0'

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self.ui = ui
        ui._focused_neuron: int = None
        ui._focused_pair: tuple = None
        ui._net_show_arrows: bool = True
        ui._net_hide_unconnected: bool = False
        ui._net_group_filter_vars: dict = {}
        ui._net_grp_items: list = []
        self._net_any_idx: int = 0
        self._net_focused: bool = False
        self._net_any_sessions_cache: list = []
        self._last_shank_ids = None  # kept for backward compat; prefer ui.neurons.shank_ids
        self._setup(parent)

    def _highlighted_ct_labels(self) -> set[str]:
        """Return conn-type labels currently highlighted in this panel."""
        labels = set()
        for (a, b), var in getattr(self, '_net_ct_vars', {}).items():
            if var.get():
                labels.add(self.ui._conn_type_label((a, b)))
        return labels

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self, parent):
        """Build all widgets — body of the old setup_network_panel."""
        ui = self.ui

        # ── Scrollable controls area ────────────────────────────────────
        # All control widgets go into `ctrl` (inside a scrollable canvas).
        # The matplotlib figure canvas stays in `parent` below.
        _ctrl_outer = tk.Frame(parent)
        _ctrl_outer.pack(side=tk.TOP, fill=tk.X)
        _ctrl_outer.configure(height=240)
        _ctrl_outer.pack_propagate(False)

        _ctrl_sb = ttk.Scrollbar(_ctrl_outer, orient=tk.VERTICAL)
        _ctrl_sb.pack(side=tk.RIGHT, fill=tk.Y)

        _ctrl_canvas = tk.Canvas(_ctrl_outer, highlightthickness=0,
                                  yscrollcommand=_ctrl_sb.set)
        _ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _ctrl_sb.config(command=_ctrl_canvas.yview)

        ctrl = ttk.Frame(_ctrl_canvas)
        _ctrl_win = _ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')

        def _on_ctrl_frame_configure(e):
            _ctrl_canvas.configure(scrollregion=_ctrl_canvas.bbox('all'))
        ctrl.bind('<Configure>', _on_ctrl_frame_configure)

        def _on_ctrl_canvas_configure(e):
            _ctrl_canvas.itemconfig(_ctrl_win, width=e.width)
        _ctrl_canvas.bind('<Configure>', _on_ctrl_canvas_configure)

        def _on_mousewheel(e):
            _ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        _ctrl_canvas.bind('<MouseWheel>', _on_mousewheel)
        ctrl.bind('<MouseWheel>', _on_mousewheel)
        # ── (end scrollable wrapper) ────────────────────────────────────

        ttk.Label(ctrl, text="Probe Network",
                  font=('Arial', 10, 'bold')).pack(pady=(0, 2))

        # ── Neuron-focus (foldable) — packed after Probes ──────────────
        _fn_outer = ttk.Frame(ctrl)
        _fn_hdr = ttk.Frame(_fn_outer)
        _fn_hdr.pack(fill=tk.X)
        self._fn_fold_var = tk.BooleanVar(value=True)
        _fn_cb = ttk.Checkbutton(_fn_hdr, text="Focus neuron",
                                  variable=self._fn_fold_var)
        _fn_cb.pack(side=tk.LEFT)
        _fn_inner = ttk.Frame(_fn_outer)
        _fn_inner.pack(fill=tk.X)
        self._focus_var = tk.StringVar()
        focus_entry = ttk.Entry(_fn_inner, textvariable=self._focus_var, width=6)
        focus_entry.pack(side=tk.LEFT, padx=2)
        focus_entry.bind('<Return>', lambda e: self._on_neuron_focus())
        ttk.Button(_fn_inner, text="Clear",
                   command=self._on_neuron_focus_clear).pack(side=tk.LEFT, padx=2)
        self._focus_info_var = tk.StringVar(value="")
        ttk.Label(_fn_inner, textvariable=self._focus_info_var,
                  font=('Arial', 8), foreground='#555').pack(side=tk.LEFT, padx=4)

        def _toggle_fn(_cb=_fn_cb, _inner=_fn_inner, _var=self._fn_fold_var):
            if _var.get():
                _inner.pack(fill=tk.X)
                _cb.config(text="Focus neuron")
            else:
                _inner.pack_forget()
                _cb.config(text="▸ Focus neuron")
        _fn_cb.config(command=_toggle_fn)

        # ── Pair-focus (foldable) — packed after Probes ────────────────
        _fp_outer = ttk.Frame(ctrl)
        _fp_hdr = ttk.Frame(_fp_outer)
        _fp_hdr.pack(fill=tk.X)
        self._fp_fold_var = tk.BooleanVar(value=True)
        _fp_cb = ttk.Checkbutton(_fp_hdr, text="Focus pair",
                                  variable=self._fp_fold_var)
        _fp_cb.pack(side=tk.LEFT)
        _fp_inner = ttk.Frame(_fp_outer)
        _fp_inner.pack(fill=tk.X)
        self._focus_pair_var = tk.StringVar()
        pair_focus_entry = ttk.Entry(_fp_inner,
                                     textvariable=self._focus_pair_var, width=8)
        pair_focus_entry.pack(side=tk.LEFT, padx=2)
        pair_focus_entry.bind('<Return>', lambda e: self._on_pair_focus())
        ttk.Button(_fp_inner, text="Clear",
                   command=self._on_pair_focus_clear).pack(side=tk.LEFT, padx=2)
        self._focus_pair_info_var = tk.StringVar(value="")
        ttk.Label(_fp_inner, textvariable=self._focus_pair_info_var,
                  font=('Arial', 8), foreground='#555').pack(side=tk.LEFT, padx=4)
        self._add_pair_btn = ttk.Button(_fp_inner, text="Add to available",
                                         command=ui._on_add_focused_pair,
                                         state=tk.DISABLED)
        self._add_pair_btn.pack(side=tk.LEFT, padx=4)

        def _toggle_fp(_cb=_fp_cb, _inner=_fp_inner, _var=self._fp_fold_var):
            if _var.get():
                _inner.pack(fill=tk.X)
                _cb.config(text="Focus pair")
            else:
                _inner.pack_forget()
                _cb.config(text="▸ Focus pair")
        _fp_cb.config(command=_toggle_fp)

        # ── Connection type toggles ──────────────────────────────────────
        ct_frame = ttk.Frame(ctrl)
        ct_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        # "Current pair" toggle — highlights only the current pair
        self._net_cur_pair_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ct_frame, text="Current pair",
                        variable=self._net_cur_pair_var,
                        command=self.draw).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(ct_frame, text="|", foreground='#BBB').pack(side=tk.LEFT, padx=2)
        ttk.Label(ct_frame, text="Conn type:").pack(side=tk.LEFT, padx=(0, 2))
        _ct_labels = {
            ('pyr', 'pyr'):     'P→P',
            ('pyr', 'inter'):   'P→I',
            ('inter', 'inter'): 'I→I',
            ('inter', 'pyr'):   'I→P',
        }
        cur_ct = getattr(ui.key, 'conn_type', None)
        self._net_ct_vars = {}
        for ct in [('pyr', 'pyr'), ('pyr', 'inter'),
                   ('inter', 'inter'), ('inter', 'pyr')]:
            var = tk.BooleanVar(value=(ct == cur_ct))
            self._net_ct_vars[ct] = var
            ttk.Checkbutton(ct_frame, text=_ct_labels[ct], variable=var,
                            command=self.draw).pack(side=tk.LEFT, padx=2)

        # ── Network display toggles ──────────────────────────────────────
        toggle_frame = ttk.Frame(ctrl)
        toggle_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._net_arrows_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Lines",
                        variable=self._net_arrows_var,
                        command=self._on_toggle_arrows
                        ).pack(side=tk.LEFT, padx=(0, 6))
        self._net_hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Hide unconnected",
                        variable=self._net_hide_var,
                        command=self._on_toggle_hide
                        ).pack(side=tk.LEFT)
        self._net_hide_same_channel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Hide same ch",
                        variable=self._net_hide_same_channel_var,
                        command=self._on_toggle_hide_same_channel
                        ).pack(side=tk.LEFT, padx=(6, 0))
        self._net_hide_same_shank_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Hide same shank",
                        variable=self._net_hide_same_shank_var,
                        command=self._on_toggle_hide_same_shank
                        ).pack(side=tk.LEFT, padx=(6, 0))
        self._net_show_chid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Ch IDs",
                        variable=self._net_show_chid_var,
                        command=self.draw
                        ).pack(side=tk.LEFT, padx=(6, 0))

        # ── Group filter — wrapping toggle checkbuttons (multi-select / merge) ──
        group_lf = ttk.Frame(ctrl, relief='groove', borderwidth=1)
        group_lf.pack(fill=tk.X, padx=4, pady=(0, 2))
        # Header row: fold arrow + "Groups" label + "Clear all" button
        grp_hdr = ttk.Frame(group_lf)
        grp_hdr.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._net_grp_fold_var = tk.BooleanVar(value=True)
        self._net_grp_arrow = tk.Label(grp_hdr, text='▾', cursor='hand2',
                                       font=('Arial', 9))
        self._net_grp_arrow.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(grp_hdr, text="Groups (highlighted/session/all)",
                  font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        ttk.Button(grp_hdr, text="Clear all", width=7,
                   command=self._on_group_clear
                   ).pack(side=tk.RIGHT, padx=(2, 0))
        self._net_grp_counts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp_hdr, text="counts",
                        variable=self._net_grp_counts_var,
                        command=self.refresh_group_buttons
                        ).pack(side=tk.RIGHT, padx=(2, 0))
        # Wrapping body — buttons are laid out by refresh_group_buttons
        self._net_grp_body = ttk.Frame(group_lf)
        self._net_grp_body.pack(fill=tk.X, expand=False, padx=2, pady=(0, 2))
        self._net_special_collapsed = False

        def _toggle_grp_body():
            if self._net_grp_fold_var.get():
                self._net_grp_body.pack_forget()
                self._net_grp_fold_var.set(False)
                self._net_grp_arrow.config(text='▸')
            else:
                self._net_grp_body.pack(fill=tk.X, expand=False,
                                        padx=2, pady=(0, 2))
                self._net_grp_fold_var.set(True)
                self._net_grp_arrow.config(text='▾')

        self._net_grp_arrow.bind('<Button-1>', lambda e: _toggle_grp_body())
        # Rewrap when the panel is resized
        self._net_grp_body.bind(
            '<Configure>',
            lambda e: ui.root.after_idle(self.rewrap_group_buttons)
        )

        # ── Probe (shank) visibility ──────────────────────────────────────
        probe_lf = ttk.Frame(ctrl, relief='groove', borderwidth=1)
        probe_lf.pack(fill=tk.X, padx=4, pady=(0, 2))
        probe_hdr = ttk.Frame(probe_lf)
        probe_hdr.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._net_probe_fold_var = tk.BooleanVar(value=True)
        self._net_probe_arrow = tk.Label(probe_hdr, text='▾', cursor='hand2',
                                         font=('Arial', 9))
        self._net_probe_arrow.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(probe_hdr, text="Probes", font=('Arial', 8, 'bold')).pack(side=tk.LEFT)

        def _probe_set_all(val):
            for v in self._net_shank_vars.values():
                v.set(val)
            self.draw()

        ttk.Button(probe_hdr, text="All", width=3,
                   command=lambda: _probe_set_all(True)).pack(side=tk.RIGHT, padx=(2, 0))
        ttk.Button(probe_hdr, text="None", width=4,
                   command=lambda: _probe_set_all(False)).pack(side=tk.RIGHT, padx=(2, 0))
        self._net_probe_body = ttk.Frame(probe_lf)
        self._net_probe_body.pack(fill=tk.X, padx=2, pady=(0, 2))
        self._net_shank_vars: dict = {}  # shank_id (int) → BooleanVar

        def _toggle_probe_body():
            if self._net_probe_fold_var.get():
                self._net_probe_body.pack_forget()
                self._net_probe_fold_var.set(False)
                self._net_probe_arrow.config(text='▸')
            else:
                self._net_probe_body.pack(fill=tk.X, padx=2, pady=(0, 2))
                self._net_probe_fold_var.set(True)
                self._net_probe_arrow.config(text='▾')

        self._net_probe_arrow.bind('<Button-1>', lambda e: _toggle_probe_body())

        # Focus neuron / Focus pair appear below the Probes section
        _fn_outer.pack(fill=tk.X, padx=4, pady=(0, 2))
        _fp_outer.pack(fill=tk.X, padx=4, pady=(0, 2))

        # ── Zoom sliders ─────────────────────────────────────────────────
        zoom_frame = ttk.Frame(ctrl)
        zoom_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(zoom_frame, text="H:", font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_hzoom_var = tk.DoubleVar(value=1.0)
        self._net_hzoom = ttk.Scale(
            zoom_frame, from_=0.2, to=1.5, orient=tk.HORIZONTAL,
            variable=self._net_hzoom_var, command=self._on_zoom)
        self._net_hzoom.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Label(zoom_frame, text="V:", font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_vzoom_var = tk.DoubleVar(value=1.0)
        self._net_vzoom = ttk.Scale(
            zoom_frame, from_=0.2, to=1.5, orient=tk.HORIZONTAL,
            variable=self._net_vzoom_var, command=self._on_zoom)
        self._net_vzoom.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        alpha_frame = ttk.Frame(ctrl)
        alpha_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(alpha_frame, text="α:", font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_line_alpha_var = tk.DoubleVar(value=1.0)
        self._net_line_alpha = ttk.Scale(
            alpha_frame, from_=0.05, to=1.0, orient=tk.HORIZONTAL,
            variable=self._net_line_alpha_var, command=self._on_zoom)
        self._net_line_alpha.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Canvas area: three-column layout so vertical nav bars flank the plot
        _canvas_row = ttk.Frame(parent)
        _canvas_row.pack(fill=tk.BOTH, expand=True)

        # Left nav bar (◀) — hidden until any-mode
        self._net_nav_left_bar = tk.Label(
            _canvas_row, text='◀', font=('Arial', 14, 'bold'),
            fg='#555', bg='#DDEEFF', cursor='hand2', width=2,
            relief=tk.FLAT)
        self._net_nav_left_bar.bind('<Button-1>', self._on_net_arrow_left)

        # Right nav bar (▶) — hidden until any-mode
        self._net_nav_right_bar = tk.Label(
            _canvas_row, text='▶', font=('Arial', 14, 'bold'),
            fg='#555', bg='#DDEEFF', cursor='hand2', width=2,
            relief=tk.FLAT)
        self._net_nav_right_bar.bind('<Button-1>', self._on_net_arrow_right)

        self.net_fig = Figure(figsize=(2.8, 6.5))
        self.net_ax = self.net_fig.add_subplot(111)
        self.net_canvas = FigureCanvasTkAgg(self.net_fig, master=_canvas_row)
        self.net_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom label showing session index (hidden until any-mode)
        self._net_nav_frame = ttk.Frame(parent)
        self._net_nav_label = ttk.Label(self._net_nav_frame, text='', font=('Arial', 8))
        self._net_nav_label.pack(fill=tk.X, expand=True)
        self._net_pick_cid = self.net_canvas.mpl_connect(
            'pick_event', self._on_network_pick)
        self._net_scroll_cid = self.net_canvas.mpl_connect(
            'scroll_event', self._on_net_scroll)
        self._scale_initialized = False

        ui.root.after(200, self.draw)
        _cw = self.net_canvas.get_tk_widget()
        _cw.bind('<Button-1>', self._on_canvas_click, add='+')
        _cw.bind('<Left>',  self._on_net_arrow_left)
        _cw.bind('<Right>', self._on_net_arrow_right)

    # ------------------------------------------------------------------
    # Any-mode session navigation
    # ------------------------------------------------------------------

    def _net_any_sessions(self) -> list:
        """Session nd_keys to display in any-mode probe navigation.

        If a group is toggled on in the network filter, only sessions with
        ≥1 pair in that group are returned; otherwise all real sessions.
        """
        ui = self.ui
        sessions = ui._real_nd_keys_ordered()
        active_groups = {g for g, var in ui._net_group_filter_vars.items()
                         if var.get()}
        if not active_groups:
            return sessions
        filtered = []
        for nk in sessions:
            ckey = ui._type_key_for_nd(nk)
            if ckey is None:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.data.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            for g in active_groups:
                if any((int(a), int(b)) in valid
                       for a, b in ui._group_pairs(g, session=sess)):
                    filtered.append(nk)
                    break
        return filtered

    def _on_canvas_click(self, event):
        """Give the canvas keyboard focus so arrow keys work."""
        self.net_canvas.get_tk_widget().focus_set()
        self._net_focused = True

    def _on_net_arrow_left(self, event):
        if not getattr(self.ui, '_session_any_mode', False):
            return
        if self._net_any_idx > 0:
            self._net_any_idx -= 1
            self.draw()
            self.refresh_group_buttons()

    def _on_net_arrow_right(self, event):
        if not getattr(self.ui, '_session_any_mode', False):
            return
        sessions = self._net_any_sessions_cache or self._net_any_sessions()
        if self._net_any_idx < len(sessions) - 1:
            self._net_any_idx += 1
            self.draw()
            self.refresh_group_buttons()

    # ------------------------------------------------------------------
    # Public draw entry point
    # ------------------------------------------------------------------

    def draw(self):
        """Redraw the network (safe wrapper — prints traceback on error)."""
        if not self.ui._panel_vars.get('Probe Network', tk.BooleanVar(value=True)).get():
            return
        try:
            self._draw_impl()
        except Exception as ex:
            print(f"[NetworkPanel] ERROR in draw: {ex}")
            traceback.print_exc()
        self._update_nav_bar()

    def _update_nav_bar(self):
        any_mode = getattr(self.ui, '_session_any_mode', False)
        was_visible = getattr(self, '_nav_bar_visible', False)
        if any_mode:
            sessions = self._net_any_sessions_cache or self._net_any_sessions()
            n = len(sessions)
            idx = max(0, min(self._net_any_idx, n - 1)) if n else 0
            lbl = self.ui._session_label(sessions[idx]) if sessions else ''
            self._net_nav_label.config(text=f'{idx + 1}/{n}  {lbl}')
            _dim = '#cccccc'
            self._net_nav_left_bar.config(fg='#333' if idx > 0 else _dim)
            self._net_nav_right_bar.config(fg='#333' if idx < n - 1 else _dim)
            if not was_visible:
                self._net_nav_left_bar.pack(side=tk.LEFT, fill=tk.Y)
                self._net_nav_right_bar.pack(side=tk.RIGHT, fill=tk.Y)
                self._net_nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
                self._nav_bar_visible = True
        else:
            if was_visible:
                self._net_nav_left_bar.pack_forget()
                self._net_nav_right_bar.pack_forget()
                self._net_nav_frame.pack_forget()
                self._nav_bar_visible = False

    # ------------------------------------------------------------------
    # Internal draw implementation
    # ------------------------------------------------------------------

    def _draw_impl(self):
        ui = self.ui
        ax = self.net_ax
        ax.clear()
        for txt in self.net_fig.texts[:]:
            txt.remove()

        # ── Any-mode: resolve per-session overrides ──────────────────────
        any_mode = getattr(ui, '_session_any_mode', False)
        if any_mode:
            _sessions = self._net_any_sessions()
            if not _sessions:
                ax.text(0.5, 0.5, "No sessions\n(active group filter)",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.axis('off')
                self.net_canvas.draw()
                return
            self._net_any_idx = max(0, min(self._net_any_idx, len(_sessions) - 1))
            self._net_any_sessions_cache = _sessions
            _sess_nk = _sessions[self._net_any_idx]
            _nd_key = _sess_nk
            _neurons = ui.cd.nd.data.get(_sess_nk) if ui.cd.nd is not None else None
            _type_key = ui._type_key_for_nd(_sess_nk)
            _ptr = ui.cd.data.get(_type_key) if _type_key is not None else None
            _deleted_inds = (ui._pair_deleted_store.get(str(_type_key), set())
                             if _type_key is not None else set())
        else:
            _nd_key = ui.key.nd()
            _neurons = ui.neurons
            _type_key = ui.key
            _ptr = ui.ccg_ptr
            _deleted_inds = ui.deleted_inds
            _sess_nk = None
            _sessions = None

        # Read zoom sliders (H = shank spacing, V = channel spacing)
        h_scale = self._net_hzoom_var.get()
        v_scale = self._net_vzoom_var.get()
        line_alpha = max(0.05, min(1.0, self._net_line_alpha_var.get()))

        pos = self._get_neuron_positions(x_scale=h_scale, y_scale=v_scale,
                                         nd_key=_nd_key, neurons=_neurons)
        if pos is None:
            ax.text(0.5, 0.5, "No probe\nposition data",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.axis('off')
            self.net_canvas.draw()
            return

        x_pos, y_pos, peak_ch = pos
        n_neurons = len(x_pos)

        shank_ids = (getattr(_neurons, 'shank_ids', None)
                     if _neurons is not None else None)
        peak_channels = (getattr(_neurons, 'peak_channels', None)
                         if _neurons is not None else None)

        # ── Draw probe background via plot_probe ──────────────────────────
        from neuropy.plotting.probe import plot_probe
        pg = getattr(getattr(ui.cd, 'nd', None), 'probegroups', {}).get(_nd_key)

        self._last_shank_ids = shank_ids  # kept for backward compat

        hidden_shanks = {s for s, v in self._net_shank_vars.items() if not v.get()}
        if pg is not None:
            show_chid = self._net_show_chid_var.get()
            plot_probe(pg, channel_id=show_chid, disconnected=True,
                       x_scale=h_scale, y_scale=v_scale,
                       hidden_shanks=hidden_shanks or None, ax=ax)
            ax.set_title('')

        fn = ui._focused_neuron
        fp = ui._focused_pair
        # Multi-group filter: union of all toggled groups' pairs
        active_groups = {g for g, var in ui._net_group_filter_vars.items()
                         if var.get()}
        _gp_sess = (str(getattr(_sess_nk, 'session', _sess_nk))
                    if any_mode and _sess_nk is not None
                    else ui._current_session_str())
        group_pairs = set()
        for g in active_groups:
            group_pairs |= ui._group_pairs(g, session=_gp_sess)
        gf_active = bool(active_groups)
        fp_neurons = {fp[0], fp[1]} if fp is not None else set()

        # ── Gather pairs: all types available, filtered by ct checkboxes ─
        type_keys_show = ui._available_type_keys(_nd_key)

        if any_mode:
            visible_pairs_current = (set(map(tuple, _ptr.inds[:, -2:]))
                                     if _ptr is not None and _ptr.inds is not None
                                     else set())
            current_pair = None
        else:
            visible_pairs_current = self._pairs_for_segment_filter()
            current_pair = (tuple(ui.all_inds[ui.current_pair_idx])
                            if ui.current_pair_idx < len(ui.all_inds) else None)

        pair_entries: dict = {}
        for tk_ in type_keys_show:
            pt = ui.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            ct = getattr(tk_, 'conn_type', None)
            ei = getattr(tk_, 'excitability', 'E')
            is_cur = (tk_ == _type_key)
            arr = pt.inds[:, -2:]
            # Build selected-pair set for this type key
            _ptr_sel = set()
            if is_cur and not any_mode:
                _ptr_sel = ui.selected_inds
            else:
                _other_ptr = ui.cd.data.get(tk_)
                if (_other_ptr is not None
                        and hasattr(_other_ptr, 'selected_inds')
                        and _other_ptr.selected_inds is not None):
                    _ptr_sel = set(map(tuple, _other_ptr.selected_inds))

            for ref, tgt in map(tuple, arr):
                # Focus neuron filter (always applied when set)
                if fn is not None and ref != fn and tgt != fn:
                    continue
                # Focus-pair mode: override group filter — show all pairs involving
                # either focused neuron; otherwise apply group filter normally
                if fp is not None:
                    if ref not in fp_neurons and tgt not in fp_neurons:
                        continue
                elif gf_active and (ref, tgt) not in group_pairs:
                    continue
                # Shank visibility filter
                if hidden_shanks and shank_ids is not None:
                    if (ref < len(shank_ids) and tgt < len(shank_ids)
                            and (int(shank_ids[ref]) in hidden_shanks
                                 or int(shank_ids[tgt]) in hidden_shanks)):
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
                    'is_selected': (ref, tgt) in _ptr_sel,
                })

        # ── Inject deleted pairs as a separate layer ─────────────────────
        # Deleted pairs that are not already in pair_entries get their own slot
        # so they can be drawn as flat light-gray lines.
        deleted_pair_entries: dict = {}  # (ref, tgt) -> True
        for (ref, tgt) in _deleted_inds:
            if (ref, tgt) in pair_entries:
                continue  # already shown via normal path
            if fn is not None and ref != fn and tgt != fn:
                continue
            if (hidden_shanks and shank_ids is not None
                    and ref < len(shank_ids) and tgt < len(shank_ids)
                    and (int(shank_ids[ref]) in hidden_shanks
                         or int(shank_ids[tgt]) in hidden_shanks)):
                continue
            if 0 <= ref < n_neurons and 0 <= tgt < n_neurons:
                deleted_pair_entries[(ref, tgt)] = True



        # ── Neuron sets ──────────────────────────────────────────────────
        cur_arr = (_ptr.inds[:, -2:] if _ptr is not None and _ptr.inds is not None
                   else np.empty((0, 2), dtype=int))
        cluster_neurons = set(int(v) for v in np.unique(cur_arr))
        # Build all_involved from pairs that actually pass the focus/filter checks,
        # so hide-unconnected correctly hides neurons with no *visible* arrow.
        all_involved = set()
        for (ref, tgt), entries in pair_entries.items():
            for entry in entries:
                ct_e = entry['conn_type']
                if ct_e is not None and not self._net_ct_vars.get(
                        ct_e, tk.BooleanVar(value=True)).get():
                    continue
                if (self._net_hide_same_shank_var.get()
                        and shank_ids is not None
                        and ref < len(shank_ids) and tgt < len(shank_ids)
                        and int(shank_ids[ref]) == int(shank_ids[tgt])):
                    continue
                if (self._net_hide_same_channel_var.get()
                        and peak_channels is not None
                        and ref < len(peak_channels) and tgt < len(peak_channels)
                        and int(peak_channels[ref]) == int(peak_channels[tgt])):
                    continue
                all_involved.add(ref)
                all_involved.add(tgt)
                break  # one passing entry is enough

        nt = (_neurons.neuron_type
              if _neurons is not None and
              _neurons.neuron_type is not None else None)

        # ── Build per-neuron connection-type color map (focus mode) ──────
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
        unconnected_o, unconnected_s = [], []  # pyr / inter with no connections
        connected_by_color: dict = {}  # color → (list_o, list_s)
        fp_neurons = set()
        if fp is not None:
            fp_neurons = {fp[0], fp[1]}

        for idx in range(n_neurons):
            if fn is not None and idx == fn:
                continue    # focused neuron drawn individually below
            if idx in fp_neurons:
                continue    # focused pair neurons drawn individually below
            # Shank visibility filter
            if hidden_shanks and shank_ids is not None and idx < len(shank_ids):
                if int(shank_ids[idx]) in hidden_shanks:
                    continue
            ntype = nt[idx] if nt is not None else None
            is_inter = (ntype == 'inter')
            if ui._net_hide_unconnected:
                in_any = idx in all_involved
            else:
                in_any = idx in all_involved or idx in cluster_neurons
            if in_any:
                c = '#9E9E9E' if fp is not None else 'black'
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
        if not ui._net_hide_unconnected:
            _scatter(unconnected_o, '^', '#9E9E9E', 14, 1, alpha=0.25)
            _scatter(unconnected_s, 'o', '#9E9E9E', 14, 1, alpha=0.25)
        # Connected neurons: colored by connection type (focus) or neuron type
        for color, (o_list, s_list) in connected_by_color.items():
            a = 0.3 if fp is not None else 1.0
            _scatter(o_list, '^', color, 50 if fp is None else 20, 4, alpha=a)
            _scatter(s_list, 'o', color, 50 if fp is None else 20, 4, alpha=a)
        # Focused neuron (single neuron mode)
        if fn is not None and 0 <= fn < n_neurons:
            fn_ntype = nt[fn] if nt is not None else None
            fn_marker = 'o' if fn_ntype == 'inter' else '^'
            ax.scatter([x_pos[fn]], [y_pos[fn]], s=140, marker=fn_marker,
                       color='#FF6F00', zorder=6, linewidths=2.0,
                       edgecolors='black')
        # Focused pair neurons
        if fp is not None:
            for i, (nid, clr) in enumerate([(fp[0], '#FF6F00'), (fp[1], '#1E88E5')]):
                if 0 <= nid < n_neurons:
                    ntype = nt[nid] if nt is not None else None
                    m = 'o' if ntype == 'inter' else '^'
                    ax.scatter([x_pos[nid]], [y_pos[nid]], s=140, marker=m,
                               color=clr, zorder=6, linewidths=2.0,
                               edgecolors='black')

        # ── Draw edges (arrows) ──────────────────────────────────────────
        # Build a set of all (ref,tgt) so we know if a reverse edge exists
        # (for arc-offset to keep both arrows visible)
        all_pair_set = set(pair_entries.keys())

        for (ref, tgt), entries in pair_entries.items():
            if not ui._net_show_arrows:
                break
            if not (0 <= ref < n_neurons and 0 <= tgt < n_neurons):
                continue
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
                if ct is not None and not self._net_ct_vars.get(
                        ct, tk.BooleanVar(value=True)).get():
                    continue
                # Skip same-shank pairs (subsumes same-channel)
                if (self._net_hide_same_shank_var.get()
                        and shank_ids is not None
                        and int(shank_ids[ref]) == int(shank_ids[tgt])):
                    continue
                # Skip same-channel pairs when toggle is on
                if (self._net_hide_same_channel_var.get()
                        and peak_channels is not None
                        and ref < len(peak_channels) and tgt < len(peak_channels)
                        and int(peak_channels[ref]) == int(peak_channels[tgt])):
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
                elif not is_cur and is_sel:
                    alpha, lw, zo = 0.70, 1.4, 3   # selected in another type — thicker
                elif not is_cur:
                    alpha, lw, zo = 0.35, 0.6, 2
                elif is_cpair:
                    alpha, lw, zo = 1.00, 3.0, 7
                    # ec stays as type color; black overlay drawn separately below
                elif is_sel:
                    alpha, lw, zo = 0.90, 1.8, 4
                else:
                    alpha, lw, zo = 0.55, 0.9, 3
                alpha *= line_alpha

                mutation = 10 if is_cpair else 7
                # All visible, non-heavily-dimmed arrows are pickable so that
                # clicking any connection (including other types) can jump to it.
                pickable = in_filt and alpha >= 0.30

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
                # Current pair: black overlay at narrower lw to give type-color edge
                if is_cpair:
                    _black = FancyArrowPatch(
                        (x_pos[ref], y_pos[ref]),
                        (x_pos[tgt], y_pos[tgt]),
                        arrowstyle='->', color='black',
                        linewidth=1.5, alpha=alpha,
                        mutation_scale=mutation,
                        connectionstyle=f'arc3,rad={rad}',
                        shrinkA=5, shrinkB=5,
                        zorder=zo + 1,
                        picker=False,
                    )
                    ax.add_patch(_black)

        # ── Deleted pairs — flat light-gray lines (20% opacity) ─────────────
        if ui._net_show_arrows:
            for (ref, tgt) in deleted_pair_entries:
                has_reverse = (tgt, ref) in deleted_pair_entries
                rad = 0.18 if has_reverse else 0.0
                del_arrow = FancyArrowPatch(
                    (x_pos[ref], y_pos[ref]),
                    (x_pos[tgt], y_pos[tgt]),
                    arrowstyle='->', color='#333333',
                    linewidth=0.5, alpha=0.20 * line_alpha,
                    mutation_scale=5,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=5, shrinkB=5,
                    zorder=1,
                    picker=False,
                )
                ax.add_patch(del_arrow)

        # ── Dashed arrow for non-existent focused pair ────────────────────
        if (fp is not None and fp not in pair_entries
                and ui._net_show_arrows
                and 0 <= fp[0] < n_neurons and 0 <= fp[1] < n_neurons):
            dashed = FancyArrowPatch(
                (x_pos[fp[0]], y_pos[fp[0]]),
                (x_pos[fp[1]], y_pos[fp[1]]),
                arrowstyle='->', color='#888888',
                linewidth=1.5, alpha=0.7 * line_alpha,
                linestyle='--',
                mutation_scale=8,
                connectionstyle='arc3,rad=0',
                shrinkA=5, shrinkB=5,
                zorder=7,
            )
            ax.add_patch(dashed)

        # ── Current pair arrow (additive, drawn on top) ──────────────────
        cur_pair_on = self._net_cur_pair_var.get()
        if (cur_pair_on and current_pair is not None
                and ui._net_show_arrows
                and 0 <= current_pair[0] < n_neurons
                and 0 <= current_pair[1] < n_neurons):
            _cp_ct = None
            if current_pair in pair_entries:
                _cp_ents = pair_entries[current_pair]
                _cp_e = next((e for e in _cp_ents if e.get('is_current')), _cp_ents[0])
                _cp_ct = _cp_e.get('conn_type')
            _cp_type_col = self._NET_TYPE_COLOR.get(_cp_ct, '#888888')
            cp_color_arrow = FancyArrowPatch(
                (x_pos[current_pair[0]], y_pos[current_pair[0]]),
                (x_pos[current_pair[1]], y_pos[current_pair[1]]),
                arrowstyle='->', color=_cp_type_col,
                linewidth=5.0, alpha=1.0 * line_alpha,
                mutation_scale=10,
                connectionstyle='arc3,rad=0',
                shrinkA=5, shrinkB=5,
                zorder=8,
            )
            ax.add_patch(cp_color_arrow)
            cp_black_arrow = FancyArrowPatch(
                (x_pos[current_pair[0]], y_pos[current_pair[0]]),
                (x_pos[current_pair[1]], y_pos[current_pair[1]]),
                arrowstyle='->', color='black',
                linewidth=2.5, alpha=1.0 * line_alpha,
                mutation_scale=10,
                connectionstyle='arc3,rad=0',
                shrinkA=5, shrinkB=5,
                zorder=9,
            )
            ax.add_patch(cp_black_arrow)

        # ── Same-channel circle loops ─────────────────────────────────────
        # Pairs where ref and tgt share the same peak_channel are drawn as
        # concentric arc-circles at the channel's probe position.  All entries
        # for that channel (across all pairs and all conn types) are stacked as
        # concentric rings growing outward, each colored by connection type.
        _hide_same_ch    = self._net_hide_same_channel_var.get()
        _hide_same_shank = self._net_hide_same_shank_var.get()
        if peak_channels is not None and ui._net_show_arrows and not _hide_same_ch:
            from matplotlib.patches import Arc as _Arc
            import math as _math

            BASE_R = 7    # innermost ring radius (data units)
            R_STEP = 5    # radius increment per additional ring
            GAP    = BASE_R + 4  # offset from channel x so circles don't cover neuron dot

            # Deduplicate pair_entries: one arc entry per (ref,tgt) direction.
            # Prefer the is_current entry; otherwise take the first.
            _arc_entry_for: dict = {}
            for (ref, tgt), entries in pair_entries.items():
                cur = next((e for e in entries if e.get('is_current')), entries[0])
                _arc_entry_for[(ref, tgt)] = cur

            # Collect same-channel entries grouped by channel value
            _chan_entries: dict = {}  # ch → [(ref, tgt, entry)]
            for (ref, tgt), entry in _arc_entry_for.items():
                if ref >= n_neurons or tgt >= n_neurons:
                    continue
                try:
                    if peak_channels[ref] != peak_channels[tgt]:
                        continue
                except (IndexError, TypeError):
                    continue
                # Filter by conn_type checkbox
                ct_e = entry.get('conn_type')
                if ct_e is not None and not self._net_ct_vars.get(ct_e, tk.BooleanVar(value=True)).get():
                    continue
                # Also skip when same-shank is hidden and they share a shank
                if (_hide_same_shank and shank_ids is not None
                        and ref < len(shank_ids) and tgt < len(shank_ids)
                        and int(shank_ids[ref]) == int(shank_ids[tgt])):
                    continue
                ch = int(peak_channels[ref])
                _chan_entries.setdefault(ch, []).append((ref, tgt, entry))

            for ch, ch_ents in _chan_entries.items():
                ref0 = ch_ents[0][0]
                cx = x_pos[ref0] + GAP
                cy = y_pos[ref0]
                for k, (ref, tgt, entry) in enumerate(ch_ents):
                    ct  = entry.get('conn_type')
                    is_cpair_arc = ((ref, tgt) == current_pair)
                    if is_cpair_arc:
                        arc_alpha = 1.0
                        lw = 2.5
                    elif fp is not None and (ref, tgt) != fp:
                        arc_alpha = 0.12
                        lw = 0.5
                    elif fn is not None and ref != fn and tgt != fn:
                        arc_alpha = 0.12
                        lw = 0.5
                    else:
                        arc_alpha = 0.85
                        lw = 1.4
                    arc_alpha *= line_alpha
                    col = self._NET_TYPE_COLOR.get(ct, '#888888')
                    r   = BASE_R + k * R_STEP
                    arc = _Arc((cx, cy), 2 * r, 2 * r,
                               angle=0, theta1=20, theta2=340,
                               color=col, linewidth=lw,
                               alpha=arc_alpha, zorder=4)
                    arc.set_gid(f"{ref}_{tgt}_{entry.get('key', '')}")
                    arc.set_picker(3)
                    ax.add_patch(arc)
                    # Clockwise arrowhead at theta=20
                    t_r = _math.radians(20)
                    px = cx + r * _math.cos(t_r)
                    py = cy + r * _math.sin(t_r)
                    eps = r * 0.22
                    ax.annotate('', xy=(px + _math.sin(t_r) * eps,
                                        py - _math.cos(t_r) * eps),
                                xytext=(px, py),
                                arrowprops=dict(arrowstyle='->', color=col,
                                                lw=1.0, mutation_scale=6),
                                zorder=5)
                    # Index label at top of arc so user knows which ring = which pair
                    lbl_x = cx + r * _math.cos(_math.radians(90))
                    lbl_y = cy + r * _math.sin(_math.radians(90))
                    ax.text(lbl_x, lbl_y, str(k + 1),
                            ha='center', va='center', fontsize=5,
                            color=col, zorder=6,
                            bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                      ec='none', alpha=0.6))

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
            if ct in shown_types and self._net_ct_vars.get(
                    ct, tk.BooleanVar(value=True)).get():
                legend_handles.append(
                    Line2D([0], [0], color=self._NET_TYPE_COLOR[ct],
                           lw=2, label=_ct_label[ct]))
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=6, loc='lower left',
                      framealpha=0.75, handlelength=1.4)

        # ── Shank labels (visible shanks only) ──────────────────────────
        if pg is not None:
            y_top = np.max(pg.y) * v_scale + 20
            for sk in pg._data['shank_id'].unique():
                if int(sk) in hidden_shanks:
                    continue
                shank_data = pg._data[pg._data['shank_id'] == sk]
                sx = shank_data['x'].mean() * h_scale
                ax.text(sx, y_top, f"S{int(sk)}",
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='#555555')

        # ── Zoom to visible shanks ───────────────────────────────────────
        xs_all, ys_all = [], []
        if pg is not None:
            df = pg._data
            if hidden_shanks:
                df = df[~df['shank_id'].apply(lambda s: int(s) in hidden_shanks)]
            if not df.empty:
                xs_all.extend((df['x'] * h_scale).tolist())
                ys_all.extend((df['y'] * v_scale).tolist())
        # Also include visible neuron positions
        if shank_ids is not None:
            for idx in range(n_neurons):
                if idx < len(shank_ids) and int(shank_ids[idx]) not in hidden_shanks:
                    xs_all.append(x_pos[idx])
                    ys_all.append(y_pos[idx])
        if xs_all and ys_all:
            pad_x = max((max(xs_all) - min(xs_all)) * 0.08, 20)
            pad_y = max((max(ys_all) - min(ys_all)) * 0.06, 20)
            ax.set_xlim(min(xs_all) - pad_x, max(xs_all) + pad_x)
            ax.set_ylim(min(ys_all) - pad_y, max(ys_all) + pad_y)
            if not self._scale_initialized:
                self._scale_initialized = True
                data_w = (max(xs_all) - min(xs_all)) + 2 * pad_x
                data_h = (max(ys_all) - min(ys_all)) + 2 * pad_y
                if data_w > 0 and data_h > 0:
                    canvas_aspect = 6.5 / 2.8  # fig height / fig width
                    data_aspect = data_h / data_w
                    if data_aspect > canvas_aspect:
                        # data taller than canvas → shrink V or expand H
                        target_h = self._net_hzoom_var.get()
                        target_v = self._net_vzoom_var.get() * canvas_aspect / data_aspect
                        target_v = max(0.2, min(1.5, target_v))
                        self._net_vzoom_var.set(target_v)
                    else:
                        # data wider than canvas → shrink H or expand V
                        target_h = self._net_hzoom_var.get() * data_aspect / canvas_aspect
                        target_h = max(0.2, min(1.5, target_h))
                        self._net_hzoom_var.set(target_h)

        ax.axis('off')
        ax.set_aspect('equal')

        # Any-mode: session title + pair title drawn in figure space (avoids overlap with shank labels)
        if any_mode and _sess_nk is not None:
            n_sess = len(_sessions)
            sess_lbl = f"{ui._session_label(_sess_nk)}  {self._net_any_idx + 1}/{n_sess}"
            self.net_fig.text(0.5, 0.985, sess_lbl,
                              fontsize=8, ha='center', va='top', color='#222')
            try:
                pair_title = ui.get_plot_title()
            except Exception:
                pair_title = ''
            if pair_title:
                self.net_fig.text(0.5, 0.002, pair_title,
                                  fontsize=6, ha='center', va='bottom', color='#444')

        if getattr(ui, '_dark', False):
            _bg, _fg = '#2b2b2b', 'white'
            self.net_fig.set_facecolor(_bg)
            ax.set_facecolor(_bg)
            ax.tick_params(colors=_fg)
            ax.xaxis.label.set_color(_fg)
            ax.yaxis.label.set_color(_fg)
            for txt in self.net_fig.texts:
                txt.set_color(_fg)
            for txt in ax.texts:
                txt.set_color(_fg)
            for sp in ax.spines.values():
                sp.set_edgecolor('#666666')
        self.net_fig.tight_layout(pad=0.5)
        self.net_canvas.draw()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_network_pick(self, event):
        # Deduplicate: only process the first pick per mouse event
        me = getattr(event, 'mouseevent', None)
        if me is not None:
            if getattr(self, '_last_pick_mouseevent', None) is me:
                return
            self._last_pick_mouseevent = me
        gid = getattr(event.artist, 'get_gid', lambda: None)()
        if not gid:
            return
        try:
            parts = gid.split('_', 2)
            ref, tgt = int(parts[0]), int(parts[1])
            key_str = parts[2] if len(parts) > 2 else None
        except (ValueError, AttributeError, IndexError):
            return

        ui = self.ui
        pair = (ref, tgt)

        # ── Any-mode: navigate probe to the session that owns this pair ─────
        if getattr(ui, '_session_any_mode', False):
            sessions = self._net_any_sessions_cache or self._net_any_sessions()
            # Find the session index in the probe nav list that matches key_str
            target_idx = self._net_any_idx
            if key_str is not None:
                for si, nk in enumerate(sessions):
                    ckey = ui._type_key_for_nd(nk)
                    if ckey is not None:
                        avail = ui._available_type_keys(nk)
                        if any(str(k) == key_str for k in avail):
                            target_idx = si
                            break
            if target_idx != self._net_any_idx:
                self._net_any_idx = target_idx
            # Select the pair in the any-mode list
            pidx = ui.get_pair_index(pair)
            if pidx < len(ui.all_inds):
                ui.current_pair_idx = pidx
                ui._select_pair_in_list(pair)
            self.draw()
            ui.update_plot()
            return

        # ── Normal mode: optionally switch type key, then select pair ─────
        if key_str is not None:
            clicked_key = next(
                (k for k in ui._available_type_keys(ui.key.nd())
                 if str(k) == key_str),
                None)
            if clicked_key is not None and clicked_key != ui.key:
                if ui._switch_key(clicked_key):
                    type_labels = [ui._type_label(k)
                                   for k in ui._type_keys_list]
                    new_label = ui._type_label(clicked_key)
                    if new_label in type_labels:
                        ui._type_var.set(new_label)
                    ui._refresh_after_key_switch()

        idx = ui.get_pair_index(pair)
        if idx < len(ui.all_inds):
            ui.current_pair_idx = idx
            ui._select_pair_in_list(pair)
        self.draw()
        ui.update_plot()

    def _on_net_scroll(self, event):
        """Scroll wheel: adjust both spacing sliders and redraw."""
        if event.inaxes != self.net_ax:
            return
        factor = 1.15 if event.step > 0 else 0.87
        cur_vz = self._net_vzoom_var.get()
        cur_hz = self._net_hzoom_var.get()
        self._net_vzoom_var.set(min(max(cur_vz * factor, 0.2), 5.0))
        self._net_hzoom_var.set(min(max(cur_hz * factor, 0.2), 5.0))
        self.draw()

    def _on_toggle_arrows(self):
        self.ui._net_show_arrows = self._net_arrows_var.get()
        self.draw()

    def _on_toggle_hide(self):
        self.ui._net_hide_unconnected = self._net_hide_var.get()
        self.draw()

    def _on_toggle_hide_same_channel(self):
        self.draw()
        self.ui.refresh_lists()

    def _on_toggle_hide_same_shank(self):
        self.draw()
        self.ui.refresh_lists()

    def _on_group_toggle(self, group_name):
        """Called when a group toggle checkbutton changes state."""
        self.draw()

    def _on_group_clear(self):
        """Clear all group filter toggles."""
        for var in self.ui._net_group_filter_vars.values():
            var.set(False)
        self.draw()

    def _on_save_selections_to_group(self):
        """Save currently visible pairs (Lines view) to a new group."""
        ui = self.ui
        pairs = self._pairs_for_segment_filter()
        if not pairs:
            import tkinter.messagebox as _mb
            _mb.showinfo("Save selections", "No pairs visible to save.")
            return
        name = tk.simpledialog.askstring(
            "Save selections to group",
            f"Group name ({len(pairs)} pairs):",
            parent=ui.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        import tkinter.messagebox as _mb
        if name in ui._sel_data._groups:
            if not _mb.askyesno(
                    "Save selections",
                    f"Group '{name}' already exists. Replace its pairs for this session?",
                    parent=ui.root):
                return
            # Clear existing session pairs before overwriting
            sess = ui._current_session_str()
            ui._sel_data._groups[name][sess] = set()
        else:
            ui._sel_data._groups[name] = {}
        for pair in pairs:
            ui._group_add_pair(name, pair)
        ui._rebuild_groups_menu()
        ui.refresh_lists()
        _mb.showinfo("Save selections",
                     f"Saved {len(pairs)} pairs to group '{name}'.",
                     parent=ui.root)

    def _on_zoom(self, _=None):
        """Called when H or V zoom slider changes — redraws with new spacing."""
        self.draw()

    # ------------------------------------------------------------------
    # Group / shank button refresh
    # ------------------------------------------------------------------

    def refresh_group_buttons(self):
        """Rebuild the group toggle checkbuttons in the probe network panel."""
        ui = self.ui
        if not hasattr(self, '_net_grp_body'):
            return
        for w in self._net_grp_body.winfo_children():
            w.destroy()
        ui._net_grp_items = []

        regular = sorted(k for k in ui._sel_data._groups
                         if not k.startswith('__'))
        special = sorted(k for k in ui._sel_data._groups
                         if k.startswith(_SPECIAL_PREFIX))
        # Remove vars for groups that no longer exist
        gone = set(ui._net_group_filter_vars) - set(ui._sel_data._groups)
        for g in gone:
            del ui._net_group_filter_vars[g]

        if not regular and not special:
            lbl = ttk.Label(self._net_grp_body, text="(no groups)",
                            foreground='#888')
            ui._net_grp_items.append((lbl, False))
            self.rewrap_group_buttons()
            return

        # Resolve which session to use for per-session counts.
        # In any-mode the browsed session is the right context; elsewhere use current.
        if getattr(ui, '_session_any_mode', False) and self._net_any_sessions_cache:
            _idx = max(0, min(self._net_any_idx, len(self._net_any_sessions_cache) - 1))
            _nk  = self._net_any_sessions_cache[_idx]
            _count_sess = str(getattr(_nk, 'session', _nk))
        else:
            _count_sess = ui._current_session_str()

        for gname in regular:
            if gname not in ui._net_group_filter_vars:
                ui._net_group_filter_vars[gname] = tk.BooleanVar(
                    master=ui.root, value=False)
            sess = _count_sess
            pairs_sess = ui._group_pairs(gname, session=sess)
            n_sess = len(pairs_sess)
            n_all  = len(ui._group_pairs_all_sessions(gname))
            n_hl = len(ui._filter_pairs_to_conn_types(
                sess, pairs_sess, self._highlighted_ct_labels()))
            count = f"{n_hl}/{n_sess}/{n_all}"
            show_counts = (getattr(self, '_net_grp_counts_var', None) is None
                           or self._net_grp_counts_var.get())
            btn = ttk.Checkbutton(
                self._net_grp_body, text=(f"{gname} ({count})" if show_counts else gname),
                variable=ui._net_group_filter_vars[gname],
                command=lambda g=gname: self._on_group_toggle(g))
            ui._net_grp_items.append((btn, False))

        if special:
            sep = ttk.Separator(self._net_grp_body, orient='horizontal')
            ui._net_grp_items.append((sep, True))
            arrow = '▸' if self._net_special_collapsed else '▾'
            lbl = tk.Label(self._net_grp_body,
                           text=f"{arrow} Special:",
                           foreground='#666', cursor='hand2',
                           font=('Arial', 8))

            def _toggle_special(e=None, _lbl=lbl):
                self._net_special_collapsed = not self._net_special_collapsed
                _lbl.config(text=('▸' if self._net_special_collapsed else '▾') + ' Special:')
                self.rewrap_group_buttons()

            lbl.bind('<Button-1>', _toggle_special)
            ui._net_grp_items.append((lbl, True))
            for gname in special:
                display = gname[len(_SPECIAL_PREFIX):]
                if gname not in ui._net_group_filter_vars:
                    ui._net_group_filter_vars[gname] = tk.BooleanVar(
                        master=ui.root, value=False)
                sess = _count_sess
                pairs_sess = ui._group_pairs(gname, session=sess)
                n_sess = len(pairs_sess)
                n_all  = len(ui._group_pairs_all_sessions(gname))
                n_hl = len(ui._filter_pairs_to_conn_types(
                    sess, pairs_sess, self._highlighted_ct_labels()))
                count = f"{n_hl}/{n_sess}/{n_all}"
                show_counts = (getattr(self, '_net_grp_counts_var', None) is None
                               or self._net_grp_counts_var.get())
                btn = ttk.Checkbutton(
                    self._net_grp_body, text=(f"{display} ({count})" if show_counts else display),
                    variable=ui._net_group_filter_vars[gname],
                    command=lambda g=gname: self._on_group_toggle(g))
                ui._net_grp_items.append((btn, False))

        self.rewrap_group_buttons()

    def rewrap_group_buttons(self):
        """Place group buttons in wrapping rows inside _net_grp_body."""
        ui = self.ui
        if not hasattr(self, '_net_grp_body') or not ui._net_grp_items:
            return
        body = self._net_grp_body
        body.update_idletasks()
        avail_w = body.winfo_width()
        if avail_w <= 1:
            # Not yet realized — try again shortly
            body.after(100, self.rewrap_group_buttons)
            return

        # Forget all current geometry
        for w, _ in ui._net_grp_items:
            w.place_forget()

        PAD_X, PAD_Y = 2, 1
        x, y, row_h = PAD_X, PAD_Y, 0
        in_special = False   # True once we've passed the separator
        for w, is_sep in ui._net_grp_items:
            if is_sep:
                in_special = True  # separator marks start of special section
            # Skip special buttons (not headers) when special section collapsed
            if in_special and self._net_special_collapsed and not is_sep:
                continue
            w.update_idletasks()
            ww = w.winfo_reqwidth()
            wh = w.winfo_reqheight()
            if is_sep:
                # Separators / "Special:" label always take a full line
                if x > PAD_X:
                    y += row_h + PAD_Y
                w.place(x=0, y=y, relwidth=1.0, height=2)
                y += 4 + PAD_Y
                x, row_h = PAD_X, 0
            else:
                if x + ww > avail_w - PAD_X and x > PAD_X:
                    y += row_h + PAD_Y
                    x = PAD_X
                    row_h = 0
                w.place(x=x, y=y)
                x += ww + PAD_X
                row_h = max(row_h, wh)

        total_h = y + row_h + PAD_Y
        body.configure(height=max(total_h, 4))

    def refresh_shank_buttons(self):
        """Populate per-shank checkbuttons in the Probes section."""
        ui = self.ui
        if not hasattr(self, '_net_probe_body'):
            return
        body = self._net_probe_body
        for w in body.winfo_children():
            w.destroy()
        # Prefer ProbeGroup (has all physical shanks incl. those without neurons)
        pg = getattr(getattr(ui.cd, 'nd', None), 'probegroups', {}).get(ui.key.nd())
        if pg is not None:
            unique_shanks = sorted(int(s) for s in pg._data['shank_id'].unique())
        else:
            shank_ids = (getattr(ui.neurons, 'shank_ids', None)
                         if ui.neurons is not None else None)
            if shank_ids is None or len(shank_ids) == 0:
                ttk.Label(body, text="No shank data", font=('Arial', 8),
                          foreground='gray').pack(side=tk.LEFT, padx=2)
                self._net_shank_vars = {}
                return
            unique_shanks = sorted(set(int(s) for s in shank_ids))
        # Preserve existing var states if shanks haven't changed
        new_vars = {}
        for shank in unique_shanks:
            existing = self._net_shank_vars.get(shank)
            var = existing if existing is not None else tk.BooleanVar(value=True)
            new_vars[shank] = var
            cb = ttk.Checkbutton(body, text=f"S{shank}", variable=var,
                                 command=self.draw)
            cb.pack(side=tk.LEFT, padx=2)
        self._net_shank_vars = new_vars

    # ------------------------------------------------------------------
    # Shank / pair labels
    # ------------------------------------------------------------------

    def _shank_label(self, idx: int) -> str:
        """Return the shank number for neuron at position idx, or str(idx) as fallback."""
        shank_ids = getattr(self.ui.neurons, 'shank_ids', None)
        if shank_ids is not None:
            try:
                return str(int(shank_ids[idx]))
            except Exception:
                pass
        return str(idx)

    def _pair_label(self, inds) -> str:
        """Short display label for a (ref, tgt) pair using shank numbers."""
        return f"{self._shank_label(inds[0])}→{self._shank_label(inds[1])}"

    # ------------------------------------------------------------------
    # Neuron / pair focus
    # ------------------------------------------------------------------

    def _on_neuron_focus(self):
        val = self._focus_var.get().strip()
        if not val:
            self._on_neuron_focus_clear()
            return
        try:
            nid = int(val)
        except ValueError:
            messagebox.showerror("Neuron focus", f"Invalid neuron id: {val!r}")
            return
        if self.ui.neurons is not None:
            if nid < 0 or nid >= self.ui.neurons.n_neurons:
                messagebox.showerror("Neuron focus",
                                     f"Neuron {nid} out of range "
                                     f"[0, {self.ui.neurons.n_neurons-1}]")
                return
        self.ui._focused_neuron = nid
        self.ui._focused_pair = None
        self._focus_pair_var.set("")
        self._focus_pair_info_var.set("")
        self._update_focus_info(nid)
        self.ui.refresh_lists()
        self.draw()

    def _update_focus_info(self, nid):
        """Update focus info label with current-type and total connection counts."""
        ui = self.ui
        cur_out = sum(1 for r, t in map(tuple, ui.all_inds) if r == nid)
        cur_in  = sum(1 for r, t in map(tuple, ui.all_inds) if t == nid)
        tot_out, tot_in = 0, 0
        for tk_ in ui._available_type_keys(ui.key.nd()):
            pt = ui.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            arr = pt.inds[:, -2:]
            tot_out += sum(1 for r, t in set(map(tuple, arr)) if r == nid)
            tot_in  += sum(1 for r, t in set(map(tuple, arr)) if t == nid)
        ct_label = ui._type_label(ui.key)
        self._focus_info_var.set(
            f"{ct_label}: in={cur_in} out={cur_out}  |  all: in={tot_in} out={tot_out}")

    def _on_neuron_focus_clear(self):
        self.ui._focused_neuron = None
        self._focus_var.set("")
        self._focus_info_var.set("")
        self.ui.refresh_lists()
        self.draw()

    def _on_pair_focus(self):
        """Set focus to a specific (ref, tgt) pair. Clears neuron focus."""
        ui = self.ui
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
        if ui.neurons is not None:
            n = ui.neurons.n_neurons
            if ref < 0 or ref >= n or tgt < 0 or tgt >= n:
                messagebox.showerror("Pair focus",
                                     f"Neuron index out of range [0, {n-1}]")
                return
        pair = (ref, tgt)
        pair_exists = False
        for tk_ in ui._available_type_keys(ui.key.nd()):
            pt = ui.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            if pair in set(map(tuple, pt.inds2)):
                pair_exists = True
                break
        if not pair_exists:
            ui._show_temp_warning(f"Pair ({ref},{tgt}) not significant — showing position")
        ui._focused_pair = pair
        ui._focused_neuron = None
        self._focus_var.set("")
        self._focus_info_var.set("")
        self._update_pair_focus_info(pair, pair_exists)
        ui.refresh_lists()
        self.draw()
        ui.update_plot()

    def _update_pair_focus_info(self, pair, exists):
        """Update the pair focus info label and 'Add to available' button."""
        ui = self.ui
        ref, tgt = pair
        if ui.neurons is not None:
            nt = ui.neurons.neuron_type
            ref_type = nt[ref] if nt is not None and ref < len(nt) else '?'
            tgt_type = nt[tgt] if nt is not None and tgt < len(nt) else '?'
            in_available = pair in set(map(tuple, ui.all_inds))
            status = "sig" if exists else ("admitted" if in_available else "not sig")
            self._focus_pair_info_var.set(
                f"{ref}({ref_type})→{tgt}({tgt_type}) [{status}]")
        else:
            in_available = pair in set(map(tuple, ui.all_inds))
            self._focus_pair_info_var.set(f"{ref}→{tgt}")
        self._add_pair_btn.config(
            state=tk.NORMAL if not in_available else tk.DISABLED)

    def _on_pair_focus_clear(self):
        self.ui._focused_pair = None
        self._focus_pair_var.set("")
        self._focus_pair_info_var.set("")
        self._add_pair_btn.config(state=tk.DISABLED)
        self.ui.refresh_lists()
        self.draw()
        self.ui.update_plot()

    # ------------------------------------------------------------------
    # Helpers (formerly on CCGReviewUI)
    # ------------------------------------------------------------------

    def _get_neuron_positions(self, x_scale=1.0, y_scale=1.0, nd_key=None, neurons=None):
        ui = self.ui
        if neurons is None:
            neurons = ui.neurons
        if neurons is None or neurons.peak_channels is None:
            return None
        pgs = getattr(getattr(ui.cd, 'nd', None), 'probegroups', {})
        if nd_key is None:
            nd_key = ui.key.nd()
        pg = pgs.get(nd_key)
        if pg is None:
            print(f"[ProbeNetwork] No ProbeGroup for key={nd_key}. "
                  f"Available keys: {list(pgs.keys())}")
            return None
        peak_ch = np.asarray(neurons.peak_channels)
        pg_df = pg.to_dataframe()
        ch_to_xy = {int(row['channel_id']): (float(row['x']), float(row['y']))
                    for _, row in pg_df.iterrows()}
        x = np.zeros(len(peak_ch))
        y = np.zeros(len(peak_ch))
        n_miss = 0
        for i, ch in enumerate(peak_ch):
            xy = ch_to_xy.get(int(ch))
            if xy is not None:
                x[i] = xy[0] * x_scale
                y[i] = xy[1] * y_scale
            else:
                n_miss += 1
        if n_miss:
            print(f"[ProbeNetwork] WARNING: {n_miss}/{len(peak_ch)} neurons "
                  f"have peak_channels not found in ProbeGroup.channel_id. "
                  f"peak_ch sample={peak_ch[:5]}, "
                  f"pg_ch sample={list(ch_to_xy.keys())[:5]}")
        return x, y, peak_ch

    def _pairs_for_segment_filter(self):
        ui = self.ui
        if ui.active_segment_filter is None:
            return set(map(tuple, ui.all_inds))
        pt = ui.ccg_ptr
        if pt.stored_by_segment:
            seg_i = ui.active_segment_filter
            mask = pt.inds[:, 0] == seg_i
            return set(map(tuple, pt.inds[mask, -2:]))
        return set(map(tuple, ui.all_inds))
