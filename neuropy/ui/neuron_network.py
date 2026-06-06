"""Probe network panel — PySide6 + pyqtgraph implementation.

Replaces the matplotlib/tkinter NetworkPanel from probe_network.py.
Toolbox controls stay in a tkinter frame (parent); the plot opens as
a floating Qt window whose event loop is pumped from tkinter via root.after.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

import numpy as np

# ── PySide6 / pyqtgraph ───────────────────────────────────────────────────────
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    _HAS_PG = True
except ImportError:
    pg = None
    _HAS_PG = False

from neuropy.ui.utils import LRUCache, _SPECIAL_PREFIX

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

# ── Qt bootstrap ──────────────────────────────────────────────────────────────

def _ensure_qt_app():
    if not _HAS_PG:
        return None
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app

# ── Color helpers ─────────────────────────────────────────────────────────────

_CT_RGBA: dict = {
    ('pyr',   'pyr'):   (211,  47,  47, 255),
    ('pyr',   'inter'): (218, 165,  32, 255),
    ('inter', 'pyr'):   ( 46, 125,  50, 255),
    ('inter', 'inter'): ( 21, 101, 192, 255),
}

def _ct_rgba(ct, ei: str = 'E') -> tuple:
    return _CT_RGBA.get(ct, (211, 47, 47, 255) if ei == 'E' else (21, 101, 192, 255))

def _with_alpha(rgba: tuple, alpha: float) -> tuple:
    r, g, b, _ = rgba
    return (r, g, b, max(0, min(255, int(255 * alpha))))

def _perp_offset(x1: float, y1: float, x2: float, y2: float,
                 d: float = 4.0) -> tuple:
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, d
    return -dy / length * d, dx / length * d

# ── Custom pyqtgraph items ────────────────────────────────────────────────────

if _HAS_PG:

    class ProbeConnectionItem(pg.GraphicsObject):
        """Clickable directed edge: shaft line + filled triangle arrowhead."""

        sigClicked = QtCore.Signal(str)  # gid

        def __init__(self, x1: float, y1: float, x2: float, y2: float,
                     rgba: tuple, lw: float = 1.0, gid: str = ''):
            super().__init__()
            self._x1, self._y1 = x1, y1
            self._x2, self._y2 = x2, y2
            self._rgba = rgba
            self._lw   = lw
            self._gid  = gid
            self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            pad = 12.0
            self._br = QtCore.QRectF(
                min(x1, x2) - pad, min(y1, y2) - pad,
                abs(x2 - x1) + 2 * pad, abs(y2 - y1) + 2 * pad,
            )

        def boundingRect(self) -> QtCore.QRectF:
            return self._br

        def paint(self, painter: QtGui.QPainter, *_args):
            r, g, b, a = self._rgba
            color = QtGui.QColor(r, g, b, a)
            pen = QtGui.QPen(color, self._lw)
            pen.setCapStyle(QtCore.Qt.FlatCap)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(self._x1, self._y1),
                             QtCore.QPointF(self._x2, self._y2))

            dx, dy = self._x2 - self._x1, self._y2 - self._y1
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return
            ux, uy = dx / dist, dy / dist
            px, py = -uy, ux
            arrow_len = min(dist * 0.25, 8.0)
            arrow_w   = arrow_len * 0.4
            bx = self._x2 - ux * arrow_len
            by = self._y2 - uy * arrow_len
            tri = QtGui.QPolygonF([
                QtCore.QPointF(self._x2,          self._y2),
                QtCore.QPointF(bx + px * arrow_w, by + py * arrow_w),
                QtCore.QPointF(bx - px * arrow_w, by - py * arrow_w),
            ])
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            painter.setBrush(QtGui.QBrush(color))
            painter.drawPolygon(tri)

        def mouseClickEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                self.sigClicked.emit(self._gid)
                ev.accept()


    class SameChannelArcItem(pg.GraphicsObject):
        """Clickable open-arc for same-channel pairs (concentric circles)."""

        sigClicked = QtCore.Signal(str)

        def __init__(self, cx: float, cy: float, radius: float,
                     rgba: tuple, lw: float = 1.0, gid: str = ''):
            super().__init__()
            self._cx, self._cy = cx, cy
            self._r    = radius
            self._rgba = rgba
            self._lw   = lw
            self._gid  = gid
            self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            r = radius + 4
            self._br = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)

        def boundingRect(self) -> QtCore.QRectF:
            return self._br

        def paint(self, painter: QtGui.QPainter, *_args):
            r, g, b, a = self._rgba
            color = QtGui.QColor(r, g, b, a)
            painter.setPen(QtGui.QPen(color, self._lw))
            painter.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            rect = QtCore.QRectF(
                self._cx - self._r, self._cy - self._r,
                2 * self._r, 2 * self._r,
            )
            # Qt arc in 1/16° units, CCW from 3 o'clock
            painter.drawArc(rect, 20 * 16, 320 * 16)

        def mouseClickEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                self.sigClicked.emit(self._gid)
                ev.accept()

# ── Data snapshot (identical to probe_network.ProbeNetworkData) ───────────────

@dataclass
class ProbeNetworkData:
    nd_key:       object
    session_label: str

    pos:          np.ndarray        # (n_neurons, 2) unscaled [x, y]
    peak_channels: np.ndarray       # (n_neurons,) int
    shank_ids:    np.ndarray | None
    neuron_type:  np.ndarray | None
    n_neurons:    int

    pg_info:      object | None     # ProbeGroup | None

    pair_entries: dict = field(default_factory=dict)
    # (ref, tgt) -> list[{key, conn_type, ei, is_current, in_filter}]

    sel_data:     object = field(default=None)
    # SelectionData — selected_inds / deleted_inds read live at render time

    current_pair: tuple | None = None

# ── NetworkPanel ──────────────────────────────────────────────────────────────

class NetworkPanel:
    """Probe network using pyqtgraph.

    Toolbox lives in the tkinter ``parent`` frame.
    Plot opens as a standalone floating Qt window.
    """

    _CT_LABELS = {
        ('pyr',   'pyr'):   'P→P',
        ('pyr',   'inter'): 'P→I',
        ('inter', 'inter'): 'I→I',
        ('inter', 'pyr'):   'I→P',
    }

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self.ui = ui
        self._focused_neuron: int | None   = None
        self._focused_pair:   tuple | None = None
        self._net_group_filter_vars: dict  = {}
        self._net_grp_items:         list  = []
        self._data_cache: LRUCache         = LRUCache(max_size=8)
        self._net_any_idx:             int = 0
        self._net_any_sessions_cache: list = []
        self._net_shank_vars:         dict = {}
        self._net_special_collapsed        = False

        self._qt_app   = None
        self._plot_win = None
        self._pg_items: list = []    # strong refs to prevent GC

        self._setup_controls(parent)

        if _HAS_PG:
            self._qt_app = _ensure_qt_app()
            self._setup_plot_window()
            self._pump_qt()
        else:
            ttk.Label(parent,
                      text='[pyqtgraph not installed — probe network disabled]',
                      foreground='gray').pack(pady=10)

        ui.root.after(200, self.draw)

    # ── Qt event-loop integration ─────────────────────────────────────────────

    def _pump_qt(self):
        if self._qt_app is not None:
            try:
                self._qt_app.processEvents()
            except Exception:
                pass
        self.ui.root.after(16, self._pump_qt)

    def _setup_plot_window(self):
        pw = pg.PlotWidget(title='Probe Network')
        pw.setWindowTitle('Probe Network')
        pw.resize(380, 720)
        pw.setBackground('w')
        pw.getViewBox().setAspectLocked(False)
        pw.hideAxis('bottom')
        pw.hideAxis('left')
        pw.show()
        self._plot_win = pw

    # ── Toolbox (tkinter) ─────────────────────────────────────────────────────

    def _setup_controls(self, parent):
        ctrl = self._setup_scroll_controls(parent)
        ttk.Label(ctrl, text='Probe Network',
                  font=('Arial', 10, 'bold')).pack(pady=(0, 2))
        self._setup_focus_panels(ctrl)
        self._setup_toggles_and_groups(ctrl)

    def _setup_scroll_controls(self, parent) -> ttk.Frame:
        outer = tk.Frame(parent)
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        outer.configure(height=240)
        outer.pack_propagate(False)

        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv = tk.Canvas(outer, highlightthickness=0, yscrollcommand=sb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=cv.yview)

        ctrl = ttk.Frame(cv)
        win  = cv.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>', lambda e: cv.configure(
            scrollregion=cv.bbox('all')))
        cv.bind('<Configure>', lambda e: cv.itemconfig(win, width=e.width))
        for w in (cv, ctrl):
            w.bind('<MouseWheel>',
                   lambda e: cv.yview_scroll(int(-1*(e.delta/120)), 'units'))
        return ctrl

    def _setup_focus_panels(self, ctrl):
        ui = self.ui
        for attr_pfx, label, is_pair in [
            ('_fn', 'Focus neuron', False),
            ('_fp', 'Focus pair',   True),
        ]:
            outer = ttk.Frame(ctrl)
            hdr   = ttk.Frame(outer)
            hdr.pack(fill=tk.X)
            fvar = tk.BooleanVar(value=True)
            setattr(self, f'{attr_pfx}_fold_var', fvar)
            cb    = ttk.Checkbutton(hdr, text=label, variable=fvar)
            cb.pack(side=tk.LEFT)
            inner = ttk.Frame(outer)
            inner.pack(fill=tk.X)
            evar  = tk.StringVar()
            ivar  = tk.StringVar(value='')
            setattr(self, f'{attr_pfx}_var',      evar)
            setattr(self, f'{attr_pfx}_info_var', ivar)
            ent = ttk.Entry(inner, textvariable=evar,
                            width=(8 if is_pair else 6))
            ent.pack(side=tk.LEFT, padx=2)
            if is_pair:
                ent.bind('<Return>', lambda e: self._on_pair_focus())
                ttk.Button(inner, text='Clear',
                           command=self._on_pair_focus_clear
                           ).pack(side=tk.LEFT, padx=2)
                self._focus_pair_var      = evar
                self._focus_pair_info_var = ivar
                self._add_pair_btn = ttk.Button(
                    inner, text='Add to available',
                    command=ui._misc_mgr._on_add_focused_pair,
                    state=tk.DISABLED)
                self._add_pair_btn.pack(side=tk.LEFT, padx=4)
            else:
                ent.bind('<Return>', lambda e: self._on_neuron_focus())
                ttk.Button(inner, text='Clear',
                           command=self._on_neuron_focus_clear
                           ).pack(side=tk.LEFT, padx=2)
                self._focus_var      = evar
                self._focus_info_var = ivar
            ttk.Label(inner, textvariable=ivar,
                      font=('Arial', 8), foreground='#555'
                      ).pack(side=tk.LEFT, padx=4)

            def _toggle(cb=cb, inner=inner, var=fvar, base=label):
                if var.get():
                    inner.pack(fill=tk.X)
                    cb.config(text=base)
                else:
                    inner.pack_forget()
                    cb.config(text=f'▸ {base}')
            cb.config(command=_toggle)
            outer.pack(fill=tk.X, padx=4, pady=(0, 2))

    def _setup_toggles_and_groups(self, ctrl):
        ui = self.ui

        # ── Lines collapsible section ───────────────────────────────────────
        lines_lf = ttk.Frame(ctrl, relief='groove', borderwidth=1)
        lines_lf.pack(fill=tk.X, padx=4, pady=(0, 2))
        lines_hdr = ttk.Frame(lines_lf)
        lines_hdr.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._net_lines_fold_var = tk.BooleanVar(value=True)
        self._net_lines_arrow = tk.Label(lines_hdr, text='▾', cursor='hand2',
                                         font=('Arial', 9))
        self._net_lines_arrow.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(lines_hdr, text='Lines',
                  font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        lines_body = ttk.Frame(lines_lf)
        lines_body.pack(fill=tk.X, padx=2, pady=(0, 2))

        def _toggle_lines():
            if self._net_lines_fold_var.get():
                lines_body.pack_forget()
                self._net_lines_fold_var.set(False)
                self._net_lines_arrow.config(text='▸')
            else:
                lines_body.pack(fill=tk.X, padx=2, pady=(0, 2))
                self._net_lines_fold_var.set(True)
                self._net_lines_arrow.config(text='▾')
        self._net_lines_arrow.bind('<Button-1>', lambda e: _toggle_lines())

        # Row 1: ON/OFF | Current pair | Conn type chips
        ct_f = ttk.Frame(lines_body)
        ct_f.pack(fill=tk.X, pady=(2, 0))
        self._net_arrows_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ct_f, text='ON/OFF', variable=self._net_arrows_var,
                        command=self.draw).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(ct_f, text='|', foreground='#BBB').pack(side=tk.LEFT, padx=2)
        self._net_cur_pair_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ct_f, text='Current pair',
                        variable=self._net_cur_pair_var,
                        command=self.draw).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(ct_f, text='|', foreground='#BBB').pack(side=tk.LEFT, padx=2)
        ttk.Label(ct_f, text='Conn type:', font=('Arial', 7)
                  ).pack(side=tk.LEFT, padx=(0, 2))
        self._net_ct_vars = {}
        self._net_ct_frame = ct_f   # kept for refresh_ct_buttons
        self._net_ct_widgets: list = []
        self.refresh_ct_buttons()

        # Row 2: Hide | Same channel, Same shank
        tog = ttk.Frame(lines_body)
        tog.pack(fill=tk.X, pady=(2, 2))
        self._net_hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tog, text='Hide unconnected',
                        variable=self._net_hide_var,
                        command=self.draw).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(tog, text='|', foreground='#BBB').pack(side=tk.LEFT, padx=2)
        self._net_hide_same_channel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tog, text='Same ch',
                        variable=self._net_hide_same_channel_var,
                        command=self._on_toggle_hide_same_channel
                        ).pack(side=tk.LEFT, padx=2)
        self._net_hide_same_shank_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tog, text='Same shank',
                        variable=self._net_hide_same_shank_var,
                        command=self._on_toggle_hide_same_shank
                        ).pack(side=tk.LEFT, padx=2)

        # Groups collapsible
        grp_lf = ttk.Frame(ctrl, relief='groove', borderwidth=1)
        grp_lf.pack(fill=tk.X, padx=4, pady=(0, 2))
        grp_hdr = ttk.Frame(grp_lf)
        grp_hdr.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._net_grp_fold_var = tk.BooleanVar(value=True)
        self._net_grp_arrow = tk.Label(grp_hdr, text='▾', cursor='hand2',
                                       font=('Arial', 9))
        self._net_grp_arrow.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(grp_hdr, text='Groups (highlighted/session/all)',
                  font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        ttk.Button(grp_hdr, text='Clear all', width=7,
                   command=self._on_group_clear).pack(side=tk.RIGHT, padx=(2, 0))
        self._net_grp_counts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp_hdr, text='counts',
                        variable=self._net_grp_counts_var,
                        command=self.refresh_group_buttons
                        ).pack(side=tk.RIGHT, padx=(2, 0))
        self._net_grp_body = ttk.Frame(grp_lf)
        self._net_grp_body.pack(fill=tk.X, expand=False, padx=2, pady=(0, 2))

        def _toggle_grp():
            if self._net_grp_fold_var.get():
                self._net_grp_body.pack_forget()
                self._net_grp_fold_var.set(False)
                self._net_grp_arrow.config(text='▸')
            else:
                self._net_grp_body.pack(fill=tk.X, expand=False,
                                        padx=2, pady=(0, 2))
                self._net_grp_fold_var.set(True)
                self._net_grp_arrow.config(text='▾')
        self._net_grp_arrow.bind('<Button-1>', lambda e: _toggle_grp())
        self._net_grp_body.bind(
            '<Configure>',
            lambda e: ui.root.after_idle(self.rewrap_group_buttons))

        # ── Annotations collapsible section ────────────────────────────────
        ann_lf = ttk.Frame(ctrl, relief='groove', borderwidth=1)
        ann_lf.pack(fill=tk.X, padx=4, pady=(0, 2))
        ann_hdr = ttk.Frame(ann_lf)
        ann_hdr.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._net_ann_fold_var = tk.BooleanVar(value=True)
        self._net_ann_arrow = tk.Label(ann_hdr, text='▾', cursor='hand2',
                                       font=('Arial', 9))
        self._net_ann_arrow.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(ann_hdr, text='Annotations',
                  font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        ann_body = ttk.Frame(ann_lf)
        ann_body.pack(fill=tk.X, padx=2, pady=(0, 2))

        def _toggle_ann():
            if self._net_ann_fold_var.get():
                ann_body.pack_forget()
                self._net_ann_fold_var.set(False)
                self._net_ann_arrow.config(text='▸')
            else:
                ann_body.pack(fill=tk.X, padx=2, pady=(0, 2))
                self._net_ann_fold_var.set(True)
                self._net_ann_arrow.config(text='▾')
        self._net_ann_arrow.bind('<Button-1>', lambda e: _toggle_ann())

        # Probe shanks row
        prb_row = ttk.Frame(ann_body)
        prb_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(prb_row, text='Probe shanks',
                  font=('Arial', 7, 'bold')).pack(side=tk.LEFT)

        def _probe_all(val):
            for v in self._net_shank_vars.values():
                v.set(val)
            self.draw()
        ttk.Button(prb_row, text='All',  width=3,
                   command=lambda: _probe_all(True)).pack(side=tk.RIGHT, padx=(2, 0))
        ttk.Button(prb_row, text='None', width=4,
                   command=lambda: _probe_all(False)).pack(side=tk.RIGHT, padx=(2, 0))
        self._net_probe_body = ttk.Frame(ann_body)
        self._net_probe_body.pack(fill=tk.X, padx=2, pady=(0, 2))

        # Separator
        ttk.Separator(ann_body, orient='horizontal').pack(fill=tk.X, pady=2)

        # Zoom row
        ttk.Label(ann_body, text='Zoom', font=('Arial', 7, 'bold')
                  ).pack(anchor='w')
        zm = ttk.Frame(ann_body)
        zm.pack(fill=tk.X)
        self._net_hzoom_var = tk.DoubleVar(value=1.0)
        self._net_vzoom_var = tk.DoubleVar(value=1.0)
        for lbl, var in [('H:', self._net_hzoom_var), ('V:', self._net_vzoom_var)]:
            ttk.Label(zm, text=lbl, font=('Arial', 7)).pack(side=tk.LEFT)
            ttk.Scale(zm, from_=0.2, to=1.5, orient=tk.HORIZONTAL,
                      variable=var, command=self._on_zoom
                      ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        al = ttk.Frame(ann_body)
        al.pack(fill=tk.X)
        ttk.Label(al, text='α:', font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_line_alpha_var = tk.DoubleVar(value=1.0)
        ttk.Scale(al, from_=0.05, to=1.0, orient=tk.HORIZONTAL,
                  variable=self._net_line_alpha_var, command=self._on_zoom
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Label(al, text='spread:', font=('Arial', 7)).pack(side=tk.LEFT)
        self._net_spread_var = tk.DoubleVar(value=1.0)
        ttk.Scale(al, from_=0.0, to=3.0, orient=tk.HORIZONTAL,
                  variable=self._net_spread_var, command=self._on_zoom
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Annotations chip row
        ttk.Label(ann_body, text='Annotations', font=('Arial', 7, 'bold')
                  ).pack(anchor='w', pady=(4, 0))
        ann_chips = ttk.Frame(ann_body)
        ann_chips.pack(fill=tk.X)
        self._net_show_chid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ann_chips, text='channel ids',
                        variable=self._net_show_chid_var,
                        command=self.draw).pack(side=tk.LEFT, padx=2)
        self._net_show_nid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ann_chips, text='neuron ids',
                        variable=self._net_show_nid_var,
                        command=self.draw).pack(side=tk.LEFT, padx=2)
        self._net_show_pair_ind_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ann_chips, text='pair inds',
                        variable=self._net_show_pair_ind_var,
                        command=self.draw).pack(side=tk.LEFT, padx=2)

        # Any-mode nav
        self._net_nav_frame = ttk.Frame(ctrl)
        self._net_nav_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._net_nav_left  = ttk.Button(self._net_nav_frame, text='◀', width=2,
                                          command=self._on_net_arrow_left)
        self._net_nav_right = ttk.Button(self._net_nav_frame, text='▶', width=2,
                                          command=self._on_net_arrow_right)
        self._net_nav_label_var = tk.StringVar(value='')
        self._net_nav_label = ttk.Label(self._net_nav_frame,
                                        textvariable=self._net_nav_label_var,
                                        font=('Arial', 8))
        self._nav_bar_visible = False

    def refresh_ct_buttons(self):
        """Rebuild conn-type checkbuttons from actual type keys in the dataset."""
        ui = self.ui
        frame = getattr(self, '_net_ct_frame', None)
        if frame is None:
            return
        # Remove existing ct widgets
        for w in getattr(self, '_net_ct_widgets', []):
            w.destroy()
        self._net_ct_widgets = []

        cur_ct = getattr(ui.key, 'conn_type', None)
        # Derive conn types from available type keys; fall back to hardcoded if none
        try:
            avail_keys = ui._sess_mgr._available_type_keys(ui.key.nd())
            ct_set = {getattr(k, 'conn_type', None) for k in avail_keys
                      if getattr(k, 'conn_type', None) is not None}
        except Exception:
            ct_set = set()
        if not ct_set:
            ct_set = set(self._CT_LABELS.keys())

        for ct in sorted(ct_set, key=lambda x: str(x)):
            # Label: first letter of each type, e.g. ('pyr','inter') -> 'P→I'
            a, b = ct if isinstance(ct, tuple) and len(ct) == 2 else (str(ct), '')
            lbl  = f'{str(a)[0].upper()}→{str(b)[0].upper()}' if b else str(a)[:3]
            if ct not in self._net_ct_vars:
                self._net_ct_vars[ct] = tk.BooleanVar(value=(ct == cur_ct))
            cb = ttk.Checkbutton(frame, text=lbl,
                                 variable=self._net_ct_vars[ct],
                                 command=self.draw)
            cb.pack(side=tk.LEFT, padx=2)
            self._net_ct_widgets.append(cb)
        # Remove vars for types no longer present
        for ct in set(self._net_ct_vars) - ct_set:
            del self._net_ct_vars[ct]

    # ── Draw entry points ─────────────────────────────────────────────────────

    def draw(self):
        if not _HAS_PG or self._plot_win is None:
            return
        if not self.ui._panel_vars.get(
                'Probe Network', tk.BooleanVar(value=True)).get():
            return
        try:
            self._draw_impl()
        except Exception as ex:
            print(f'[NetworkPanel] ERROR: {ex}')
            traceback.print_exc()
        self._update_nav_bar()

    def _draw_impl(self):
        ui = self.ui
        pw = self._plot_win

        any_mode = getattr(ui, '_session_any_mode', False)
        if any_mode:
            _sessions = self._net_any_sessions()
            if not _sessions:
                pw.clear()
                t = pg.TextItem('No sessions\n(active group filter)',
                                color='gray', anchor=(0.5, 0.5))
                pw.addItem(t)
                return
            self._net_any_idx = max(0, min(self._net_any_idx,
                                           len(_sessions) - 1))
            self._net_any_sessions_cache = _sessions
            _sess_nk  = _sessions[self._net_any_idx]
            _nd_key   = _sess_nk
            _type_key = ui._sess_mgr._type_key_for_nd(_sess_nk)
            segment_filter = None
        else:
            _nd_key        = ui.key.nd()
            _sess_nk       = None
            _sessions      = None
            _type_key      = ui.key
            segment_filter = ui.active_segment_filter

        cache_key = self._make_cache_key(_nd_key, _type_key,
                                         segment_filter, any_mode)
        data = self._data_cache.get(cache_key)
        if data is None:
            data = self._assemble_data(_nd_key, _type_key,
                                       segment_filter, any_mode, _sess_nk)
            if data is None:
                pw.clear()
                t = pg.TextItem('No probe\nposition data',
                                color='gray', anchor=(0.5, 0.5))
                pw.addItem(t)
                return
            self._data_cache.put(cache_key, data)

        self._render(data, any_mode, _sess_nk, _sessions)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _make_cache_key(self, nd_key, type_key,
                        segment_filter, any_mode: bool) -> tuple:
        ui  = self.ui
        sel = (frozenset(ui._sel_data.selected_inds)
               if ui._sel_data is not None else frozenset())
        dlt = (frozenset(ui._sel_data.deleted_inds)
               if ui._sel_data is not None else frozenset())
        return (nd_key, str(type_key), segment_filter, sel, dlt)

    def _assemble_data(self, nd_key, type_key, segment_filter,
                       any_mode: bool, sess_nk) -> ProbeNetworkData | None:
        ui = self.ui

        if any_mode:
            _neurons = (ui.cd.nd.data.get(sess_nk)
                        if ui.cd.nd is not None else None)
            _ptr = (ui.cd.ptr.get(type_key)
                    if type_key is not None else None)
        else:
            _neurons = ui.neurons
            _ptr     = ui.ccg_ptr

        pos = self._get_neuron_positions(nd_key=nd_key, neurons=_neurons)
        if pos is None:
            return None
        xy, peak_ch = pos
        n_neurons   = len(xy)

        shank_ids   = getattr(_neurons, 'shank_ids',   None) if _neurons else None
        neuron_type = _neurons.neuron_type               if _neurons else None
        pg_info     = self._probegroups().get(nd_key)

        if any_mode:
            visible_pairs = (set(map(tuple, _ptr.inds[:, -2:]))
                             if _ptr is not None and _ptr.inds is not None
                             else set())
            current_pair  = None
        else:
            visible_pairs = self._pairs_for_segment_filter()
            current_pair  = (tuple(ui.all_inds[ui.current_pair_idx])
                             if ui.current_pair_idx < len(ui.all_inds)
                             else None)

        pair_entries: dict = {}
        for tk_ in ui._sess_mgr._available_type_keys(nd_key):
            pt = ui.cd.ptr.get(tk_)
            if pt is None or pt.inds is None:
                continue
            ct     = getattr(tk_, 'conn_type',    None)
            ei     = getattr(tk_, 'excitability', 'E')
            is_cur = (tk_ == type_key)
            for ref, tgt in map(tuple, pt.inds[:, -2:]):
                key_t = (ref, tgt)
                if key_t not in pair_entries:
                    pair_entries[key_t] = []
                pair_entries[key_t].append({
                    'key':        tk_,
                    'conn_type':  ct,
                    'ei':         ei,
                    'is_current': is_cur,
                    'in_filter':  (ref, tgt) in visible_pairs if is_cur else True,
                })

        session_label = (ui._sess_mgr._session_label(sess_nk)
                         if any_mode and sess_nk is not None else '')

        return ProbeNetworkData(
            nd_key=nd_key, session_label=session_label,
            pos=xy, peak_channels=peak_ch,
            shank_ids=shank_ids, neuron_type=neuron_type,
            n_neurons=n_neurons, pg_info=pg_info,
            pair_entries=pair_entries, sel_data=ui._sel_data,
            current_pair=current_pair,
        )

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self, data: ProbeNetworkData,
                any_mode: bool, sess_nk, sessions):
        pw = self._plot_win
        pw.clear()
        self._pg_items.clear()

        ui          = self.ui
        h_scale     = self._net_hzoom_var.get()
        v_scale     = self._net_vzoom_var.get()
        x_pos       = data.pos[:, 0] * h_scale
        y_pos       = data.pos[:, 1] * v_scale
        line_alpha  = max(0.05, min(1.0, self._net_line_alpha_var.get()))
        hidden_shanks = frozenset(
            s for s, v in self._net_shank_vars.items() if not v.get())

        # Live current pair (not in cache)
        current_pair = (None if any_mode else
                        (tuple(ui.all_inds[ui.current_pair_idx])
                         if ui.current_pair_idx < len(ui.all_inds) else None))

        # Group filter
        fp            = self._focused_pair
        active_groups = {g for g, var in self._net_group_filter_vars.items()
                         if var.get()}
        if active_groups and fp is None:
            _gp_sess = (str(getattr(data.nd_key, 'session', data.nd_key))
                        if any_mode
                        else ui._setup_mgr._current_session_str())
            group_pairs: set = set()
            for g in active_groups:
                group_pairs |= ui._sel_data.pairs_in_group(g, _gp_sess)
            gf_active = True
        else:
            group_pairs, gf_active = set(), False

        sel_data      = data.sel_data
        selected_inds = getattr(sel_data, 'selected_inds', set()) if sel_data else set()
        deleted_inds  = getattr(sel_data, 'deleted_inds',  set()) if sel_data else set()

        enabled_cts = frozenset(ct for ct, v in self._net_ct_vars.items()
                                if v.get())
        if len(enabled_cts) == len(self._net_ct_vars):
            enabled_cts = None   # all enabled

        fn         = self._focused_neuron
        fp_neurons = set(fp) if fp is not None else set()

        self._draw_probe_bg(data, x_pos, y_pos, h_scale, v_scale, hidden_shanks)

        if self._net_arrows_var.get():
            self._draw_connections(
                data, x_pos, y_pos, current_pair,
                selected_inds, deleted_inds, enabled_cts,
                hidden_shanks, gf_active, group_pairs,
                self._net_hide_same_channel_var.get(),
                self._net_hide_same_shank_var.get(),
                fn, fp, fp_neurons, line_alpha,
            )

        self._draw_neurons(data, x_pos, y_pos,
                           fn, fp, fp_neurons, hidden_shanks)

        self._draw_legend(data, enabled_cts)

        if data.session_label:
            ti = pg.TextItem(data.session_label, color='#444444',
                             anchor=(0.5, 1.0))
            ti.setPos(float(np.mean(x_pos)), float(np.max(y_pos)) + 20)
            pw.addItem(ti)
            self._pg_items.append(ti)

    # ── Probe background ──────────────────────────────────────────────────────

    def _draw_probe_bg(self, data, x_pos, y_pos,
                       h_scale, v_scale, hidden_shanks):
        pw      = self._plot_win
        pg_info = data.pg_info
        if pg_info is None:
            return
        try:
            df = pg_info.to_dataframe()
        except Exception:
            return

        # Identify bad/skipped channels
        ui = self.ui
        try:
            nd_conf  = getattr(getattr(ui.cd, 'nd', None), '_conf', None)
            recinfo  = getattr(nd_conf, 'recinfo', None) if nd_conf else None
            skipped  = set(getattr(recinfo, 'skipped_channels', None) or [])
        except Exception:
            skipped = set()

        show_chid = self._net_show_chid_var.get()
        for shank_id in df['shank_id'].unique():
            if int(shank_id) in hidden_shanks:
                continue
            sub    = df[df['shank_id'] == shank_id]
            xs     = sub['x'].to_numpy(dtype=float) * h_scale
            ys     = sub['y'].to_numpy(dtype=float) * v_scale
            ch_ids = sub['channel_id'].to_numpy()

            # Normal channels
            good_mask = np.array([int(c) not in skipped for c in ch_ids])
            bad_mask  = ~good_mask
            if good_mask.any():
                sc = pg.ScatterPlotItem(
                    x=xs[good_mask], y=ys[good_mask], size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(180, 180, 180, 160), symbol='o')
                sc.setZValue(1)
                pw.addItem(sc)
                self._pg_items.append(sc)
            if bad_mask.any():
                sc_bad = pg.ScatterPlotItem(
                    x=xs[bad_mask], y=ys[bad_mask], size=5,
                    pen=pg.mkPen('#CC0000', width=1),
                    brush=pg.mkBrush(180, 180, 180, 160), symbol='o')
                sc_bad.setZValue(1)
                pw.addItem(sc_bad)
                self._pg_items.append(sc_bad)

            lbl = pg.TextItem(f'S{int(shank_id)}', color='#888888',
                              anchor=(0.5, 1.0))
            lbl.setPos(float(np.mean(xs)), float(np.min(ys)) - 10)
            pw.addItem(lbl)
            self._pg_items.append(lbl)
            if show_chid:
                for xi, yi, cid in zip(xs, ys, ch_ids):
                    t = pg.TextItem(str(cid), color='#888', anchor=(0.0, 0.5))
                    t.setPos(float(xi) + 3, float(yi))
                    t.setFont(QtGui.QFont('Arial', 6))
                    pw.addItem(t)
                    self._pg_items.append(t)

    # ── Connections ───────────────────────────────────────────────────────────

    def _draw_connections(self, data, x_pos, y_pos, current_pair,
                          selected_inds, deleted_inds, enabled_cts,
                          hidden_shanks, gf_active, group_pairs,
                          hide_same_ch, hide_same_shk,
                          fn, fp, fp_neurons, line_alpha):
        pw           = self._plot_win
        n            = data.n_neurons
        all_pairs    = set(data.pair_entries.keys())
        same_ch_map: dict = {}
        show_pair_ind = getattr(self, '_net_show_pair_ind_var', None)
        show_pair_ind = show_pair_ind.get() if show_pair_ind is not None else False

        for (ref, tgt), entries in data.pair_entries.items():
            if not (0 <= ref < n and 0 <= tgt < n):
                continue
            if fn is not None and ref != fn and tgt != fn:
                continue
            if fp is not None and ref not in fp_neurons and tgt not in fp_neurons:
                continue
            if gf_active and (ref, tgt) not in group_pairs:
                continue
            if (hidden_shanks and data.shank_ids is not None
                    and ref < len(data.shank_ids) and tgt < len(data.shank_ids)
                    and (int(data.shank_ids[ref]) in hidden_shanks
                         or int(data.shank_ids[tgt]) in hidden_shanks)):
                continue

            for entry in entries:
                ct = entry['conn_type']
                if enabled_cts is not None and ct not in enabled_cts:
                    continue
                if (hide_same_shk and data.shank_ids is not None
                        and ref < len(data.shank_ids) and tgt < len(data.shank_ids)
                        and int(data.shank_ids[ref]) == int(data.shank_ids[tgt])):
                    continue

                is_same_ch = (data.peak_channels is not None
                              and data.peak_channels[ref] == data.peak_channels[tgt])
                if hide_same_ch and is_same_ch:
                    continue

                is_cur   = entry['is_current']
                in_filt  = entry['in_filter']
                is_sel   = (ref, tgt) in selected_inds
                is_fp    = (fp is not None and (ref, tgt) == fp)
                is_cpair = is_cur and (ref, tgt) == current_pair

                rgba = _ct_rgba(ct, entry.get('ei', 'E'))

                # Arrow style — mirrors _arrow_style in probe.py
                if is_fp:
                    alpha, lw = 1.00, 3.0
                elif fp is not None:
                    alpha, lw, rgba = 0.12, 0.3, (204, 204, 204, 255)
                elif not in_filt:
                    alpha, lw, rgba = 0.20, 0.4, (204, 204, 204, 255)
                elif not is_cur and is_sel:
                    alpha, lw = 0.70, 1.4
                elif not is_cur:
                    alpha, lw = 0.35, 0.6
                elif is_cpair:
                    alpha, lw = 1.00, 3.0
                elif is_sel:
                    alpha, lw = 0.90, 1.8
                else:
                    alpha, lw = 0.55, 0.9

                final_rgba = _with_alpha(rgba, alpha * line_alpha)

                if is_same_ch:
                    ch = int(data.peak_channels[ref])
                    same_ch_map.setdefault(ch, []).append(
                        (ref, tgt, entry, is_cpair, final_rgba, lw))
                    continue

                ox, oy = (0.0, 0.0)
                if (tgt, ref) in all_pairs:
                    ox, oy = _perp_offset(
                        x_pos[ref], y_pos[ref], x_pos[tgt], y_pos[tgt], d=3.0)

                gid  = f'{ref}_{tgt}_{entry["key"]}'
                item = ProbeConnectionItem(
                    x_pos[ref] + ox, y_pos[ref] + oy,
                    x_pos[tgt] + ox, y_pos[tgt] + oy,
                    final_rgba, lw, gid)
                item.setZValue(3)
                item.sigClicked.connect(self._on_arrow_click)
                pw.addItem(item)
                self._pg_items.append(item)

                if show_pair_ind and entry.get('is_current'):
                    mx = (x_pos[ref] + ox + x_pos[tgt] + ox) / 2
                    my = (y_pos[ref] + oy + y_pos[tgt] + oy) / 2
                    t  = pg.TextItem(f'{ref},{tgt}', color='#555',
                                     anchor=(0.5, 1.0))
                    t.setPos(mx, my)
                    t.setFont(QtGui.QFont('Arial', 6))
                    pw.addItem(t)
                    self._pg_items.append(t)

                if is_cpair:
                    outline = ProbeConnectionItem(
                        x_pos[ref] + ox, y_pos[ref] + oy,
                        x_pos[tgt] + ox, y_pos[tgt] + oy,
                        _with_alpha((0, 0, 0, 255), alpha * line_alpha),
                        lw + 1.5, '')
                    outline.setZValue(2)
                    pw.addItem(outline)
                    self._pg_items.append(outline)

        # Same-channel arcs
        BASE_R, R_STEP, GAP = 7, 5, 11
        for ch, ch_ents in same_ch_map.items():
            ref0 = ch_ents[0][0]
            cx   = float(x_pos[ref0]) + GAP
            cy   = float(y_pos[ref0])
            for k, (ref, tgt, entry, is_cpair, rgba, lw) in enumerate(ch_ents):
                r   = BASE_R + k * R_STEP
                gid = f'{ref}_{tgt}_{entry["key"]}'
                arc = SameChannelArcItem(cx, cy, r, rgba,
                                         lw=(2.0 if is_cpair else 1.0), gid=gid)
                arc.setZValue(4)
                arc.sigClicked.connect(self._on_arrow_click)
                pw.addItem(arc)
                self._pg_items.append(arc)

        # Deleted pairs — faded gray
        for (ref, tgt) in deleted_inds:
            if (ref, tgt) in data.pair_entries:
                continue
            if not (0 <= ref < n and 0 <= tgt < n):
                continue
            if fn is not None and ref != fn and tgt != fn:
                continue
            ox, oy = (0.0, 0.0)
            if (tgt, ref) in deleted_inds:
                ox, oy = _perp_offset(
                    x_pos[ref], y_pos[ref], x_pos[tgt], y_pos[tgt], d=3.0)
            rgba = _with_alpha((51, 51, 51, 255), 0.20 * line_alpha)
            item = ProbeConnectionItem(
                x_pos[ref] + ox, y_pos[ref] + oy,
                x_pos[tgt] + ox, y_pos[tgt] + oy,
                rgba, 0.5, f'{ref}_{tgt}_deleted')
            item.setZValue(1)
            item.sigClicked.connect(self._on_arrow_click)
            pw.addItem(item)
            self._pg_items.append(item)

    # ── Neurons ───────────────────────────────────────────────────────────────

    def _draw_neurons(self, data, x_pos, y_pos,
                      fn, fp, fp_neurons, hidden_shanks):
        pw      = self._plot_win
        cluster_neurons = {n for r, t in data.pair_entries for n in (r, t)}

        # Per-channel grayscale + spread
        _GRAYS = [(0,0,0),(51,51,51),(102,102,102),(153,153,153),(187,187,187)]
        _ch_slots: dict = {}
        for i, ch in enumerate(data.peak_channels):
            _ch_slots.setdefault(int(ch), []).append(i)
        neuron_rgb    = [(0, 0, 0)] * data.n_neurons
        x_spread      = x_pos.copy()
        spread_factor = getattr(self, '_net_spread_var', None)
        spread        = spread_factor.get() if spread_factor is not None else 1.0
        _step         = 5.0 * spread   # pixels between same-ch neurons
        for ch, idxs in _ch_slots.items():
            n_ch    = len(idxs)
            offsets = (np.arange(n_ch) - (n_ch - 1) / 2.0) * _step
            for slot, ni in enumerate(idxs):
                neuron_rgb[ni]  = _GRAYS[slot % len(_GRAYS)]
                x_spread[ni]   += offsets[slot]

        show_nid  = getattr(self, '_net_show_nid_var',  None)
        show_nid  = show_nid.get()  if show_nid  is not None else False
        hide_unc  = self._net_hide_var.get()
        spots: list = []

        for idx in range(data.n_neurons):
            if idx == fn or idx in fp_neurons:
                continue
            if (hidden_shanks and data.shank_ids is not None
                    and idx < len(data.shank_ids)
                    and int(data.shank_ids[idx]) in hidden_shanks):
                continue
            in_any = idx in cluster_neurons
            if hide_unc and not in_any:
                continue
            ntype  = (data.neuron_type[idx]
                      if data.neuron_type is not None
                      and idx < len(data.neuron_type) else None)
            symbol = 'o' if ntype == 'inter' else 't'
            a      = 200 if in_any else 64
            r, g, b = neuron_rgb[idx]
            xi, yi  = float(x_spread[idx]), float(y_pos[idx])
            spots.append({
                'pos': (xi, yi), 'size': 8, 'symbol': symbol,
                'pen': None, 'brush': pg.mkBrush(r, g, b, a), 'data': idx,
            })
            if show_nid:
                t = pg.TextItem(str(idx), color='#555', anchor=(0.0, 1.0))
                t.setPos(xi + 4, yi)
                t.setFont(QtGui.QFont('Arial', 6))
                pw.addItem(t)
                self._pg_items.append(t)

        if fn is not None and 0 <= fn < data.n_neurons:
            ntype  = (data.neuron_type[fn]
                      if data.neuron_type is not None else None)
            symbol = 'o' if ntype == 'inter' else 't'
            spots.append({
                'pos': (float(x_pos[fn]), float(y_pos[fn])),
                'size': 12, 'symbol': symbol,
                'pen': pg.mkPen('black', width=2),
                'brush': pg.mkBrush(255, 111, 0, 255),
                'data': fn,
            })

        for nid, rgb in ([(fp[0], (255,111,0)), (fp[1], (30,136,229))]
                         if fp is not None else []):
            if 0 <= nid < data.n_neurons:
                ntype  = (data.neuron_type[nid]
                          if data.neuron_type is not None else None)
                symbol = 'o' if ntype == 'inter' else 't'
                spots.append({
                    'pos': (float(x_pos[nid]), float(y_pos[nid])),
                    'size': 12, 'symbol': symbol,
                    'pen': pg.mkPen('black', width=2),
                    'brush': pg.mkBrush(*rgb, 255),
                    'data': nid,
                })

        sc = pg.ScatterPlotItem(size=8)
        sc.addPoints(spots)
        sc.sigClicked.connect(self._on_neuron_scatter_click)
        sc.setZValue(5)
        pw.addItem(sc)
        self._pg_items.append(sc)

    # ── Legend ────────────────────────────────────────────────────────────────

    def _draw_legend(self, data, enabled_cts):
        pw     = self._plot_win
        shown: set = set()
        for entries in data.pair_entries.values():
            for e in entries:
                ct = e['conn_type']
                if ct is not None and (enabled_cts is None or ct in enabled_cts):
                    shown.add(ct)
        _lbl = {('pyr','pyr'):'pyr→pyr', ('pyr','inter'):'pyr→int',
                ('inter','pyr'):'int→pyr', ('inter','inter'):'int→int'}
        vr    = pw.getViewBox().viewRect()
        x0    = vr.left()  + 5 if not vr.isNull() else 0.0
        y0    = vr.bottom() - 5 if not vr.isNull() else 0.0
        for i, ct in enumerate(sorted(shown, key=str)):
            rgba = _ct_rgba(ct)
            ti   = pg.TextItem(f'● {_lbl.get(ct, str(ct))}',
                               color=QtGui.QColor(*rgba), anchor=(0.0, 1.0))
            ti.setPos(x0, y0 - i * 14)
            pw.addItem(ti)
            self._pg_items.append(ti)

    # ── Click handlers ────────────────────────────────────────────────────────

    def _on_arrow_click(self, gid: str):
        parts = gid.split('_', 2)
        try:
            ref, tgt = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return
        key_str = parts[2] if len(parts) > 2 else None
        if key_str == 'deleted':
            return
        self._navigate_to_pair((ref, tgt), key_str)

    def _on_neuron_scatter_click(self, scatter, points, ev):
        if not points:
            return
        nid = points[0].data()
        if not isinstance(nid, int):
            return
        self._focused_neuron = nid
        self._focused_pair   = None
        self._focus_var.set(str(nid))
        self._focus_pair_var.set('')
        self._update_focus_info(nid)
        self.ui.refresh_lists()
        self.draw()

    def _navigate_to_pair(self, pair: tuple, key_str: str | None):
        ui = self.ui
        ref, tgt = pair
        if getattr(ui, '_session_any_mode', False):
            sessions   = self._net_any_sessions_cache or self._net_any_sessions()
            target_idx = self._net_any_idx
            if key_str is not None:
                for si, nk in enumerate(sessions):
                    ckey = ui._sess_mgr._type_key_for_nd(nk)
                    if ckey is not None:
                        avail = ui._sess_mgr._available_type_keys(nk)
                        if any(str(k) == key_str for k in avail):
                            target_idx = si
                            break
            if target_idx != self._net_any_idx:
                self._net_any_idx = target_idx
            pidx = ui._plot_mgr.get_pair_index(pair)
            if pidx < len(ui.all_inds):
                ui.current_pair_idx = pidx
                ui._select_pair_in_list(pair)
            self.draw()
            ui._plot_mgr.update_plot()
            return

        if key_str is not None:
            clicked_key = next(
                (k for k in ui._sess_mgr._available_type_keys(ui.key.nd())
                 if str(k) == key_str), None)
            if clicked_key is not None and clicked_key != ui.key:
                if ui._switch_key(clicked_key):
                    type_labels = [ui._type_label(k) for k in ui._type_keys_list]
                    new_label   = ui._type_label(clicked_key)
                    if new_label in type_labels:
                        ui._type_var.set(new_label)
                    ui._refresh_after_key_switch()

        idx = ui._plot_mgr.get_pair_index(pair)
        if idx < len(ui.all_inds):
            ui.current_pair_idx = idx
            ui._select_pair_in_list(pair)
        self.draw()
        ui._plot_mgr.update_plot()

    # ── Slider / toggle handlers ──────────────────────────────────────────────

    def _on_zoom(self, _=None):
        self.draw()

    def _on_toggle_hide_same_channel(self):
        self.draw()
        self.ui.refresh_lists()

    def _on_toggle_hide_same_shank(self):
        self.draw()
        self.ui.refresh_lists()

    def _on_group_toggle(self, _g):
        self.draw()

    def _on_group_clear(self):
        for var in self._net_group_filter_vars.values():
            var.set(False)
        self.draw()

    def _on_save_selections_to_group(self):
        ui    = self.ui
        pairs = self._pairs_for_segment_filter()
        if not pairs:
            messagebox.showinfo('Save selections', 'No pairs visible to save.')
            return
        name = tk.simpledialog.askstring(
            'Save selections to group',
            f'Group name ({len(pairs)} pairs):', parent=ui.root)
        if not name or not name.strip():
            return
        name = name.strip()
        sess = ui._setup_mgr._current_session_str()
        if name in ui._sel_data.groups:
            if not messagebox.askyesno(
                    'Save selections',
                    f"Group '{name}' exists. Replace pairs for this session?",
                    parent=ui.root):
                return
            for _, r, t in list(ui._sel_data.forward(name)):
                if _ == sess:
                    ui._sel_data.discard_from_group(name, sess, (r, t))
        for pair in pairs:
            ui._sel_data.add_to_group(name, sess, pair)
        ui._group_mgr._rebuild_groups_menu()
        ui.refresh_lists()
        messagebox.showinfo('Save selections',
                            f"Saved {len(pairs)} pairs to group '{name}'.",
                            parent=ui.root)

    # ── Any-mode navigation ───────────────────────────────────────────────────

    def _net_any_sessions(self) -> list:
        ui            = self.ui
        sessions      = ui._sess_mgr._real_nd_keys_ordered()
        active_groups = {g for g, var in self._net_group_filter_vars.items()
                         if var.get()}
        if not active_groups:
            return sessions
        filtered = []
        for nk in sessions:
            ckey = ui._sess_mgr._type_key_for_nd(nk)
            if ckey is None:
                continue
            sess  = str(ckey.session)
            ptr   = ui.cd.ptr.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            for g in active_groups:
                if any((int(a), int(b)) in valid
                       for a, b in ui._sel_data.pairs_in_group(g, sess)):
                    filtered.append(nk)
                    break
        return filtered

    def _on_net_arrow_left(self):
        if not getattr(self.ui, '_session_any_mode', False):
            return
        if self._net_any_idx > 0:
            self._net_any_idx -= 1
            self.draw()
            self.refresh_group_buttons()

    def _on_net_arrow_right(self):
        if not getattr(self.ui, '_session_any_mode', False):
            return
        sessions = self._net_any_sessions_cache or self._net_any_sessions()
        if self._net_any_idx < len(sessions) - 1:
            self._net_any_idx += 1
            self.draw()
            self.refresh_group_buttons()

    def _update_nav_bar(self):
        any_mode   = getattr(self.ui, '_session_any_mode', False)
        was_vis    = self._nav_bar_visible
        if any_mode:
            sessions = self._net_any_sessions_cache or self._net_any_sessions()
            n   = len(sessions)
            idx = max(0, min(self._net_any_idx, n - 1)) if n else 0
            lbl = (self.ui._sess_mgr._session_label(sessions[idx])
                   if sessions else '')
            self._net_nav_label_var.set(f'{idx + 1}/{n}  {lbl}')
            if not was_vis:
                self._net_nav_left.pack(side=tk.LEFT, padx=2)
                self._net_nav_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self._net_nav_right.pack(side=tk.RIGHT, padx=2)
                self._nav_bar_visible = True
            self._net_nav_left.config(
                state='normal' if idx > 0       else 'disabled')
            self._net_nav_right.config(
                state='normal' if idx < n - 1   else 'disabled')
        else:
            if was_vis:
                self._net_nav_left.pack_forget()
                self._net_nav_label.pack_forget()
                self._net_nav_right.pack_forget()
                self._nav_bar_visible = False

    # ── Group / shank buttons ─────────────────────────────────────────────────

    def _highlighted_ct_labels(self) -> set:
        labels = set()
        for (a, b), var in getattr(self, '_net_ct_vars', {}).items():
            if var.get():
                labels.add(self.ui._conn_type_label((a, b)))
        return labels

    def _make_group_button(self, gname: str, display: str,
                           count_sess: str) -> ttk.Checkbutton:
        ui = self.ui
        if gname not in self._net_group_filter_vars:
            self._net_group_filter_vars[gname] = tk.BooleanVar(
                master=ui.root, value=False)
        if self._net_grp_counts_var.get():
            pairs_sess = ui._sel_data.pairs_in_group(gname, count_sess)
            n_hl  = len(ui._filter_pairs_to_conn_types(
                count_sess, pairs_sess, self._highlighted_ct_labels()))
            n_all = len({(r, t) for _, r, t in ui._sel_data.forward(gname)})
            label = f'{display} ({n_hl}/{len(pairs_sess)}/{n_all})'
        else:
            label = display
        return ttk.Checkbutton(
            self._net_grp_body, text=label,
            variable=self._net_group_filter_vars[gname],
            command=lambda g=gname: self._on_group_toggle(g))

    def refresh_group_buttons(self):
        ui = self.ui
        if not hasattr(self, '_net_grp_body'):
            return
        for w in self._net_grp_body.winfo_children():
            w.destroy()
        self._net_grp_items = []

        regular = sorted(k for k in ui._sel_data._groups
                         if not k.startswith('__'))
        special = sorted(k for k in ui._sel_data._groups
                         if k.startswith(_SPECIAL_PREFIX))
        for g in set(self._net_group_filter_vars) - set(ui._sel_data._groups):
            del self._net_group_filter_vars[g]

        if not regular and not special:
            self._net_grp_items.append(
                (ttk.Label(self._net_grp_body, text='(no groups)',
                           foreground='#888'), False))
            self.rewrap_group_buttons()
            return

        if getattr(ui, '_session_any_mode', False) and self._net_any_sessions_cache:
            _idx = max(0, min(self._net_any_idx,
                              len(self._net_any_sessions_cache) - 1))
            _nk  = self._net_any_sessions_cache[_idx]
            count_sess = str(getattr(_nk, 'session', _nk))
        else:
            count_sess = ui._setup_mgr._current_session_str()

        for gname in regular:
            self._net_grp_items.append(
                (self._make_group_button(gname, gname, count_sess), False))

        if special:
            self._net_grp_items.append(
                (ttk.Separator(self._net_grp_body, orient='horizontal'), True))
            arrow = '▸' if self._net_special_collapsed else '▾'
            lbl   = tk.Label(self._net_grp_body,
                             text=f'{arrow} Special:',
                             foreground='#666', cursor='hand2',
                             font=('Arial', 8))

            def _toggle(e=None, _lbl=lbl):
                self._net_special_collapsed = not self._net_special_collapsed
                _lbl.config(
                    text=('▸' if self._net_special_collapsed else '▾')
                    + ' Special:')
                self.rewrap_group_buttons()
            lbl.bind('<Button-1>', _toggle)
            self._net_grp_items.append((lbl, True))
            for gname in special:
                display = gname[len(_SPECIAL_PREFIX):]
                self._net_grp_items.append(
                    (self._make_group_button(gname, display, count_sess), False))

        self.rewrap_group_buttons()

    def rewrap_group_buttons(self):
        if not hasattr(self, '_net_grp_body') or not self._net_grp_items:
            return
        body = self._net_grp_body
        body.update_idletasks()
        avail_w = body.winfo_width()
        if avail_w <= 1:
            body.after(100, self.rewrap_group_buttons)
            return
        for w, _ in self._net_grp_items:
            w.place_forget()
        PAD_X, PAD_Y = 2, 1
        x, y, row_h  = PAD_X, PAD_Y, 0
        in_special   = False
        for w, is_sep in self._net_grp_items:
            if is_sep:
                in_special = True
            if in_special and self._net_special_collapsed and not is_sep:
                continue
            w.update_idletasks()
            ww, wh = w.winfo_reqwidth(), w.winfo_reqheight()
            if is_sep:
                if x > PAD_X:
                    y += row_h + PAD_Y
                w.place(x=0, y=y, relwidth=1.0, height=2)
                y += 4 + PAD_Y
                x, row_h = PAD_X, 0
            else:
                if x + ww > avail_w - PAD_X and x > PAD_X:
                    y += row_h + PAD_Y
                    x, row_h = PAD_X, 0
                w.place(x=x, y=y)
                x    += ww + PAD_X
                row_h = max(row_h, wh)
        body.configure(height=max(y + row_h + PAD_Y, 4))

    def refresh_shank_buttons(self):
        ui = self.ui
        if not hasattr(self, '_net_probe_body'):
            return
        body = self._net_probe_body
        for w in body.winfo_children():
            w.destroy()
        pg_info = self._probegroups().get(ui.key.nd())
        if pg_info is not None:
            unique_shanks = sorted(
                int(s) for s in pg_info._data['shank_id'].unique())
        else:
            shank_ids = (getattr(ui.neurons, 'shank_ids', None)
                         if ui.neurons is not None else None)
            if shank_ids is None or len(shank_ids) == 0:
                ttk.Label(body, text='No shank data',
                          font=('Arial', 8), foreground='gray'
                          ).pack(side=tk.LEFT, padx=2)
                self._net_shank_vars = {}
                return
            unique_shanks = sorted(set(int(s) for s in shank_ids))
        new_vars = {}
        for shank in unique_shanks:
            existing = self._net_shank_vars.get(shank)
            var      = existing if existing is not None else tk.BooleanVar(value=True)
            new_vars[shank] = var
            ttk.Checkbutton(body, text=f'S{shank}', variable=var,
                            command=self.draw).pack(side=tk.LEFT, padx=2)
        self._net_shank_vars = new_vars

    # ── Focus handlers ────────────────────────────────────────────────────────

    def _on_neuron_focus(self):
        val = self._focus_var.get().strip()
        if not val:
            self._on_neuron_focus_clear()
            return
        try:
            nid = int(val)
        except ValueError:
            messagebox.showerror('Neuron focus', f'Invalid id: {val!r}')
            return
        if self.ui.neurons is not None:
            if nid < 0 or nid >= self.ui.neurons.n_neurons:
                messagebox.showerror(
                    'Neuron focus',
                    f'Out of range [0, {self.ui.neurons.n_neurons-1}]')
                return
        self._focused_neuron = nid
        self._focused_pair   = None
        self._focus_pair_var.set('')
        self._focus_pair_info_var.set('')
        self._update_focus_info(nid)
        self.ui.refresh_lists()
        self.draw()

    def _update_focus_info(self, nid: int):
        ui      = self.ui
        cur_out = sum(1 for r, t in map(tuple, ui.all_inds) if r == nid)
        cur_in  = sum(1 for r, t in map(tuple, ui.all_inds) if t == nid)
        tot_out = tot_in = 0
        for tk_ in ui._sess_mgr._available_type_keys(ui.key.nd()):
            pt = ui.cd.ptr.get(tk_)
            if pt is None or pt.inds is None:
                continue
            arr = pt.inds[:, -2:]
            tot_out += sum(1 for r, t in set(map(tuple, arr)) if r == nid)
            tot_in  += sum(1 for r, t in set(map(tuple, arr)) if t == nid)
        self._focus_info_var.set(
            f'{ui._type_label(ui.key)}: in={cur_in} out={cur_out}'
            f'  |  all: in={tot_in} out={tot_out}')

    def _on_neuron_focus_clear(self):
        self._focused_neuron = None
        self._focus_var.set('')
        self._focus_info_var.set('')
        self.ui.refresh_lists()
        self.draw()

    def _on_pair_focus(self):
        ui  = self.ui
        val = self._focus_pair_var.get().strip()
        if not val:
            self._on_pair_focus_clear()
            return
        try:
            parts    = val.replace(' ', '').split(',')
            ref, tgt = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            messagebox.showerror('Pair focus',
                                 f'Invalid format: {val!r}\nUse ref,tgt')
            return
        if ui.neurons is not None:
            n = ui.neurons.n_neurons
            if not (0 <= ref < n and 0 <= tgt < n):
                messagebox.showerror('Pair focus',
                                     f'Out of range [0, {n-1}]')
                return
        pair        = (ref, tgt)
        pair_exists = any(
            pair in set(map(tuple, ui.cd.ptr.get(tk_).inds2))
            for tk_ in ui._sess_mgr._available_type_keys(ui.key.nd())
            if ui.cd.ptr.get(tk_) is not None
            and ui.cd.ptr.get(tk_).inds is not None)
        if not pair_exists:
            ui._group_mgr._show_temp_warning(
                f'Pair ({ref},{tgt}) not significant — showing position')
        self._focused_pair   = pair
        self._focused_neuron = None
        self._focus_var.set('')
        self._focus_info_var.set('')
        self._update_pair_focus_info(pair, pair_exists)
        ui.refresh_lists()
        self.draw()
        ui._plot_mgr.update_plot()

    def _update_pair_focus_info(self, pair: tuple, exists: bool):
        ui       = self.ui
        ref, tgt = pair
        if ui.neurons is not None:
            nt       = ui.neurons.neuron_type
            ref_type = nt[ref] if nt is not None and ref < len(nt) else '?'
            tgt_type = nt[tgt] if nt is not None and tgt < len(nt) else '?'
            in_avail = pair in set(map(tuple, ui.all_inds))
            status   = ('sig' if exists
                        else ('admitted' if in_avail else 'not sig'))
            self._focus_pair_info_var.set(
                f'{ref}({ref_type})→{tgt}({tgt_type}) [{status}]')
        else:
            self._focus_pair_info_var.set(f'{ref}→{tgt}')
        in_avail = pair in set(map(tuple, ui.all_inds))
        self._add_pair_btn.config(
            state=tk.NORMAL if not in_avail else tk.DISABLED)

    def _on_pair_focus_clear(self):
        self._focused_pair = None
        self._focus_pair_var.set('')
        self._focus_pair_info_var.set('')
        self._add_pair_btn.config(state=tk.DISABLED)
        self.ui.refresh_lists()
        self.draw()
        self.ui._plot_mgr.update_plot()

    # ── Probe helpers ─────────────────────────────────────────────────────────

    def _probegroups(self) -> dict:
        return getattr(getattr(self.ui.cd, 'nd', None), 'probe_info', {})

    def _get_neuron_positions(self, nd_key=None, neurons=None):
        ui = self.ui
        if neurons is None:
            neurons = ui.neurons
        if neurons is None or neurons.peak_channels is None:
            return None
        if nd_key is None:
            nd_key = ui.key.nd()
        pg_info = self._probegroups().get(nd_key)
        if pg_info is None:
            return None
        peak_ch = np.asarray(neurons.peak_channels, dtype=int)
        pg_df   = pg_info.to_dataframe().set_index('channel_id')
        x = pg_df['x'].reindex(peak_ch).fillna(0.0).to_numpy(dtype=float)
        y = pg_df['y'].reindex(peak_ch).fillna(0.0).to_numpy(dtype=float)
        return np.stack([x, y], axis=1), peak_ch

    def _pairs_for_segment_filter(self):
        ui = self.ui
        if ui.active_segment_filter is None:
            return set(map(tuple, ui.all_inds))
        pt = ui.ccg_ptr
        if pt.stored_by_segment:
            seg_i = ui.active_segment_filter
            mask  = pt.inds[:, 0] == seg_i
            return set(map(tuple, pt.inds[mask, -2:]))
        return set(map(tuple, ui.all_inds))

    def _shank_label(self, idx: int) -> str:
        shank_ids = getattr(self.ui.neurons, 'shank_ids', None)
        if shank_ids is not None:
            try:
                return str(int(shank_ids[idx]))
            except Exception:
                pass
        return str(idx)

    def _pair_label(self, inds) -> str:
        return f'{self._shank_label(inds[0])}→{self._shank_label(inds[1])}'

"""Network panel for CCGReviewUI — probe layout with connection arrows."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from neuropy.ui.utils import (LRUCache, fit_axes_to_data,
                              _SPECIAL_PREFIX, is_special_group)
from neuropy.plotting.probe import (
    ProbeNetworkConfig, _draw_neurons, _draw_connections, _draw_labels,
    plot_probe, _compute_positions,
)

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI


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
        # --- State owned by NetworkPanel (formerly scattered on ui) ---
        self._focused_neuron: int | None = None
        self._focused_pair: tuple | None = None
        self._net_group_filter_vars: dict = {}
        self._net_grp_items: list = []
        # --- Cache: key=(nd_key, type_key, seg_filter, sel_hash, del_hash) ---
        self._data_cache: LRUCache = LRUCache(max_size=8)
        # --- Other state ---
        self._net_any_idx: int = 0
        self._net_focused: bool = False
        self._net_any_sessions_cache: list = []
        self._setup(parent)

    def _highlighted_ct_labels(self) -> set[str]:
        """Return conn-type labels currently highlighted in this panel."""
        labels = set()
        for (a, b), var in getattr(self, '_net_ct_vars', {}).items():
            if var.get():
                labels.add(self.ui._conn_type_label((a, b)))
        return labels

    def _probegroups(self) -> dict:
        """Return probe_info dict from cd.nd (empty dict if absent)."""
        return getattr(getattr(self.ui.cd, 'nd', None), 'probe_info', {})

    def _setup(self, parent):
        """Build all widgets."""
        ctrl = self._setup_scroll_controls(parent)
        ttk.Label(ctrl, text="Probe Network",
                  font=('Arial', 10, 'bold')).pack(pady=(0, 2))
        self._setup_focus_panels(ctrl)
        self._setup_toggles_and_groups(ctrl)
        self._setup_canvas(parent)
        self.ui.root.after(200, self.draw)

    def _setup_scroll_controls(self, parent) -> ttk.Frame:
        """Build scrollable controls wrapper; return inner ``ctrl`` frame."""
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
        return ctrl

    def _setup_focus_panels(self, ctrl):
        """Build foldable neuron-focus and pair-focus sections."""
        ui = self.ui

        # ── Neuron-focus (foldable) ─────────────────────────────────────
        _fn_outer = ttk.Frame(ctrl)
        _fn_hdr   = ttk.Frame(_fn_outer)
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

        # ── Pair-focus (foldable) ───────────────────────────────────────
        _fp_outer = ttk.Frame(ctrl)
        _fp_hdr   = ttk.Frame(_fp_outer)
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
                                         command=ui._misc_mgr._on_add_focused_pair,
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

        # Pack after Probes section (order matters for layout)
        _fn_outer.pack(fill=tk.X, padx=4, pady=(0, 2))
        _fp_outer.pack(fill=tk.X, padx=4, pady=(0, 2))

    def _setup_toggles_and_groups(self, ctrl):
        """Build ct toggles, display toggles, group filter, probe buttons, zoom sliders."""
        ui = self.ui

        # ── Connection type toggles ──────────────────────────────────────
        ct_frame = ttk.Frame(ctrl)
        ct_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
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
                        command=self.draw).pack(side=tk.LEFT, padx=(0, 6))
        self._net_hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Hide unconnected",
                        variable=self._net_hide_var,
                        command=self.draw).pack(side=tk.LEFT)
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
                        command=self.draw).pack(side=tk.LEFT, padx=(6, 0))

        # ── Group filter ─────────────────────────────────────────────────
        group_lf = ttk.Frame(ctrl, relief='groove', borderwidth=1)
        group_lf.pack(fill=tk.X, padx=4, pady=(0, 2))
        grp_hdr = ttk.Frame(group_lf)
        grp_hdr.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._net_grp_fold_var = tk.BooleanVar(value=True)
        self._net_grp_arrow = tk.Label(grp_hdr, text='▾', cursor='hand2',
                                       font=('Arial', 9))
        self._net_grp_arrow.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(grp_hdr, text="Groups (highlighted/session/all)",
                  font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        ttk.Button(grp_hdr, text="Clear all", width=7,
                   command=self._on_group_clear).pack(side=tk.RIGHT, padx=(2, 0))
        self._net_grp_counts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp_hdr, text="counts",
                        variable=self._net_grp_counts_var,
                        command=self.refresh_group_buttons
                        ).pack(side=tk.RIGHT, padx=(2, 0))
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
        self._net_shank_vars: dict = {}

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

    def _setup_canvas(self, parent):
        """Build matplotlib canvas, nav bar labels, and event bindings."""
        _canvas_row = ttk.Frame(parent)
        _canvas_row.pack(fill=tk.BOTH, expand=True)
        _canvas_row.columnconfigure(0, weight=0)
        _canvas_row.columnconfigure(1, weight=1)
        _canvas_row.columnconfigure(2, weight=0)
        _canvas_row.rowconfigure(0, weight=1)

        self._net_nav_left_bar = tk.Label(
            _canvas_row, text='◀', font=('Arial', 14, 'bold'),
            fg='#555', bg='#DDEEFF', cursor='hand2', width=2,
            relief=tk.FLAT)
        self._net_nav_left_bar.bind('<Button-1>', self._on_net_arrow_left)

        self._net_nav_right_bar = tk.Label(
            _canvas_row, text='▶', font=('Arial', 14, 'bold'),
            fg='#555', bg='#DDEEFF', cursor='hand2', width=2,
            relief=tk.FLAT)
        self._net_nav_right_bar.bind('<Button-1>', self._on_net_arrow_right)

        self.net_fig = Figure(figsize=(2.8, 6.5))
        self.net_ax = self.net_fig.add_subplot(111)
        self.net_canvas = FigureCanvasTkAgg(self.net_fig, master=_canvas_row)
        self.net_canvas.get_tk_widget().grid(row=0, column=1, sticky='nsew')

        self._net_nav_frame = ttk.Frame(parent)
        self._net_nav_label = ttk.Label(self._net_nav_frame, text='', font=('Arial', 8))
        self._net_nav_label.pack(fill=tk.X, expand=True)
        self._net_pick_cid = self.net_canvas.mpl_connect(
            'pick_event', self._on_network_pick)
        self._net_scroll_cid = self.net_canvas.mpl_connect(
            'scroll_event', self._on_net_scroll)
        self._scale_initialized = False

    def _net_any_sessions(self) -> list:
        """Session nd_keys to display in any-mode probe navigation.

        If a group is toggled on in the network filter, only sessions with
        ≥1 pair in that group are returned; otherwise all real sessions.
        """
        ui = self.ui
        sessions = ui._sess_mgr._real_nd_keys_ordered()
        active_groups = {g for g, var in self._net_group_filter_vars.items()
                         if var.get()}
        if not active_groups:
            return sessions
        filtered = []
        for nk in sessions:
            ckey = ui._sess_mgr._type_key_for_nd(nk)
            if ckey is None:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.ptr.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            for g in active_groups:
                if any((int(a), int(b)) in valid
                       for a, b in ui._sel_data.pairs_in_group(g, sess)):
                    filtered.append(nk)
                    break
        return filtered

    def _on_net_arrow_left(self, event=None):
        if not getattr(self.ui, '_session_any_mode', False):
            return
        if self._net_any_idx > 0:
            self._net_any_idx -= 1
            self.draw()
            self.refresh_group_buttons()

    def _on_net_arrow_right(self, event=None):
        if not getattr(self.ui, '_session_any_mode', False):
            return
        sessions = self._net_any_sessions_cache or self._net_any_sessions()
        if self._net_any_idx < len(sessions) - 1:
            self._net_any_idx += 1
            self.draw()
            self.refresh_group_buttons()

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
            lbl = self.ui._sess_mgr._session_label(sessions[idx]) if sessions else ''
            self._net_nav_label.config(text=f'{idx + 1}/{n}  {lbl}')
            _dim = '#cccccc'
            self._net_nav_left_bar.config(fg='#333' if idx > 0 else _dim)
            self._net_nav_right_bar.config(fg='#333' if idx < n - 1 else _dim)
            if not was_visible:
                self._net_nav_left_bar.grid(row=0, column=0, sticky='ns')
                self._net_nav_right_bar.grid(row=0, column=2, sticky='ns')
                self._net_nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
                self._nav_bar_visible = True
        else:
            if was_visible:
                self._net_nav_left_bar.grid_remove()
                self._net_nav_right_bar.grid_remove()
                self._net_nav_frame.pack_forget()
                self._nav_bar_visible = False

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
            _type_key = ui._sess_mgr._type_key_for_nd(_sess_nk)
            segment_filter = None
        else:
            _nd_key = ui.key.nd()
            _sess_nk = None
            _sessions = None
            _type_key = ui.key
            segment_filter = ui.active_segment_filter

        # ── Cache lookup ─────────────────────────────────────────────────
        cache_key = self._make_cache_key(_nd_key, _type_key, segment_filter, any_mode)
        data = self._data_cache.get(cache_key)
        if data is None:
            data = self._assemble_data(_nd_key, _type_key, segment_filter, any_mode, _sess_nk)
            if data is None:
                ax.text(0.5, 0.5, "No probe\nposition data",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                ax.axis('off')
                self.net_canvas.draw()
                return
            self._data_cache.put(cache_key, data)

        self._render(data, any_mode, _sess_nk, _sessions)

    def _make_cache_key(self, nd_key, type_key, segment_filter, any_mode: bool) -> tuple:
        """Compute cache key: (nd_key, type_key_str, segment_filter, sel_hash, del_hash)."""
        ui = self.ui
        sel_hash = frozenset(ui._sel_data.selected_inds) if ui._sel_data is not None else frozenset()
        del_hash = frozenset(ui._sel_data.deleted_inds)  if ui._sel_data is not None else frozenset()
        return (nd_key, str(type_key), segment_filter, sel_hash, del_hash)

    def _assemble_data(self, nd_key, type_key, segment_filter, any_mode: bool,
                       sess_nk) -> 'ProbeNetworkData | None':
        """Assemble session-specific draw inputs (no matplotlib, no render-time state).

        Group filter IS applied here (it changes which pairs are assembled).
        Focus neuron/pair, zoom, ct-checkboxes, hidden_shanks are intentionally NOT applied
        so that toggling them re-renders from cache without a data rebuild.
        """
        ui = self.ui

        if any_mode:
            _neurons = ui.cd.nd.data.get(sess_nk) if ui.cd.nd is not None else None
            _ptr = ui.cd.ptr.get(type_key) if type_key is not None else None
        else:
            _neurons = ui.neurons
            _ptr = ui.ccg_ptr

        # Unscaled positions — scale applied at render time
        pos = self._get_neuron_positions(x_scale=1.0, y_scale=1.0,
                                         nd_key=nd_key, neurons=_neurons)
        if pos is None:
            return None
        xy, peak_ch = pos
        n_neurons = len(xy)

        shank_ids   = getattr(_neurons, 'shank_ids', None) if _neurons is not None else None
        neuron_type = _neurons.neuron_type if _neurons is not None else None
        pg = self._probegroups().get(nd_key)

        # Segment filter → visible pairs for current type
        if any_mode:
            visible_pairs_current = (set(map(tuple, _ptr.inds[:, -2:]))
                                     if _ptr is not None and _ptr.inds is not None
                                     else set())
            current_pair = None
        else:
            visible_pairs_current = self._pairs_for_segment_filter()
            current_pair = (tuple(ui.all_inds[ui.current_pair_idx])
                            if ui.current_pair_idx < len(ui.all_inds) else None)

        type_keys_show = ui._sess_mgr._available_type_keys(nd_key)
        pair_entries: dict = {}
        for tk_ in type_keys_show:
            pt = ui.cd.ptr.get(tk_)
            if pt is None or pt.inds is None:
                continue
            ct  = getattr(tk_, 'conn_type', None)
            ei  = getattr(tk_, 'excitability', 'E')
            is_cur = (tk_ == type_key)
            arr = pt.inds[:, -2:]
            for ref, tgt in map(tuple, arr):
                key_t = (ref, tgt)
                if key_t not in pair_entries:
                    pair_entries[key_t] = []
                pair_entries[key_t].append({
                    'key':        tk_,
                    'conn_type':  ct,
                    'ei':         ei,
                    'is_current': is_cur,
                    'in_filter':  (ref, tgt) in visible_pairs_current if is_cur else True,
                })

        session_label = ui._sess_mgr._session_label(sess_nk) if (any_mode and sess_nk is not None) else ''

        return ProbeNetworkData(
            nd_key=nd_key,
            session_label=session_label,
            pos=xy,
            peak_channels=peak_ch,
            shank_ids=shank_ids,
            neuron_type=neuron_type,
            n_neurons=n_neurons,

            pg=pg,
            pair_entries=pair_entries,
            sel_data=ui._sel_data,
            current_pair=current_pair,
        )

    def _render(self, data: 'ProbeNetworkData', any_mode: bool, sess_nk, sessions):
        """Clear axes, draw probe background, call _draw_* helpers, then canvas.draw_idle."""
        ui            = self.ui
        ax            = self.net_ax
        h_scale       = self._net_hzoom_var.get()
        v_scale       = self._net_vzoom_var.get()
        line_alpha    = max(0.05, min(1.0, self._net_line_alpha_var.get()))
        hidden_shanks = {s for s, v in self._net_shank_vars.items() if not v.get()}

        x_pos = data.pos[:, 0] * h_scale
        y_pos = data.pos[:, 1] * v_scale

        # current_pair is render-time state — read live so cache hits still reflect it.
        if any_mode:
            current_pair = None
        else:
            current_pair = (tuple(ui.all_inds[ui.current_pair_idx])
                            if ui.current_pair_idx < len(ui.all_inds) else None)

        # ── Group filter ──────────────────────────────────────────────────────
        fp = self._focused_pair
        active_groups = {g for g, var in self._net_group_filter_vars.items() if var.get()}
        if active_groups and fp is None:
            _gp_sess = (str(getattr(data.nd_key, 'session', data.nd_key))
                        if any_mode else ui._setup_mgr._current_session_str())
            group_pairs: set = set()
            for g in active_groups:
                gp = ui._sel_data.pairs_in_group(g, _gp_sess)
                if not gp:
                    stored_keys = list(ui._sel_data.forward(g))[:5]
                    print(f"[probe_network] group={g!r} _gp_sess={_gp_sess!r} stored_keys={stored_keys} → empty")
                group_pairs |= gp
            gf_active = True
        else:
            group_pairs = set()
            gf_active = False

        if data.pg is not None:
            show_chid = self._net_show_chid_var.get()
            plot_probe(data.pg, channel_id=show_chid, disconnected=True,
                       x_scale=h_scale, y_scale=v_scale,
                       hidden_shanks=hidden_shanks or None, ax=ax)
            ax.set_title('')

        # ── Build config from Tk vars ─────────────────────────────────────────
        if any_mode and sess_nk is not None:
            session_label = ui._sess_mgr._session_label(sess_nk)
            try:
                pair_title = ui._plot_mgr.get_plot_title()
            except Exception:
                pair_title = ''
        else:
            session_label, pair_title = '', ''

        cfg = ProbeNetworkConfig(
            focused_neuron    = self._focused_neuron,
            focused_pair      = fp,
            current_pair      = current_pair,
            show_current_pair = self._net_cur_pair_var.get(),
            show_arrows       = self._net_arrows_var.get(),
            hide_unconnected  = self._net_hide_var.get(),
            hide_same_channel = self._net_hide_same_channel_var.get(),
            hide_same_shank   = self._net_hide_same_shank_var.get(),
            line_alpha        = line_alpha,
            h_scale           = h_scale,
            v_scale           = v_scale,
            hidden_shanks     = frozenset(hidden_shanks),
            enabled_conn_types = (
                frozenset(ct for ct, v in self._net_ct_vars.items() if v.get())
                if any(not v.get() for v in self._net_ct_vars.values()) else None),
            group_pairs       = frozenset(group_pairs),
            gf_active         = gf_active,
            dark_mode         = getattr(ui, '_dark', False),
            show_ch_ids       = self._net_show_chid_var.get(),
            session_label     = session_label,
            any_mode          = any_mode,
            n_sessions        = len(sessions) if sessions else 1,
            sess_idx          = self._net_any_idx,
            pair_title        = pair_title,
        )

        _draw_neurons(ax, x_pos, y_pos, data.peak_channels, data.shank_ids,
                      data.neuron_type, data.n_neurons,
                      data.pair_entries, cfg)
        _draw_connections(ax, x_pos, y_pos, data.peak_channels, data.shank_ids,
                          data.n_neurons, data.pair_entries, data.sel_data, cfg)
        xs_all, ys_all = _draw_labels(ax, x_pos, y_pos, data.pg, data.shank_ids,
                                      data.n_neurons, data.pair_entries, cfg)

        # ── First-render zoom init ────────────────────────────────────────────
        if xs_all and ys_all and not self._scale_initialized:
            self._scale_initialized = True
            pad_x = max((max(xs_all) - min(xs_all)) * 0.08, 20)
            pad_y = max((max(ys_all) - min(ys_all)) * 0.06, 20)
            data_w = (max(xs_all) - min(xs_all)) + 2 * pad_x
            data_h = (max(ys_all) - min(ys_all)) + 2 * pad_y
            if data_w > 0 and data_h > 0:
                canvas_aspect = 6.5 / 2.8
                data_aspect   = data_h / data_w
                if data_aspect > canvas_aspect:
                    target_v = self._net_vzoom_var.get() * canvas_aspect / data_aspect
                    self._net_vzoom_var.set(max(0.2, min(1.5, target_v)))
                else:
                    target_h = self._net_hzoom_var.get() * data_aspect / canvas_aspect
                    self._net_hzoom_var.set(max(0.2, min(1.5, target_h)))

        self.net_fig.tight_layout(pad=0.5)
        self.net_canvas.draw_idle()

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
            target_idx = self._net_any_idx
            if key_str is not None:
                for si, nk in enumerate(sessions):
                    ckey = ui._sess_mgr._type_key_for_nd(nk)
                    if ckey is not None:
                        avail = ui._sess_mgr._available_type_keys(nk)
                        if any(str(k) == key_str for k in avail):
                            target_idx = si
                            break
            if target_idx != self._net_any_idx:
                self._net_any_idx = target_idx
            pidx = ui._plot_mgr.get_pair_index(pair)
            if pidx < len(ui.all_inds):
                ui.current_pair_idx = pidx
                ui._select_pair_in_list(pair)
            self.draw()
            ui._plot_mgr.update_plot()
            return

        # ── Normal mode: optionally switch type key, then select pair ─────
        if key_str is not None:
            clicked_key = next(
                (k for k in ui._sess_mgr._available_type_keys(ui.key.nd())
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

        idx = ui._plot_mgr.get_pair_index(pair)
        if idx < len(ui.all_inds):
            ui.current_pair_idx = idx
            ui._select_pair_in_list(pair)
        self.draw()
        ui._plot_mgr.update_plot()

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
        for var in self._net_group_filter_vars.values():
            var.set(False)
        self.draw()

    def _on_save_selections_to_group(self):
        """Save currently visible pairs (Lines view) to a new group."""
        ui = self.ui
        pairs = self._pairs_for_segment_filter()
        if not pairs:
            messagebox.showinfo("Save selections", "No pairs visible to save.")
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
        sess = ui._setup_mgr._current_session_str()
        if name in ui._sel_data.groups:
            if not messagebox.askyesno(
                    "Save selections",
                    f"Group '{name}' already exists. Replace its pairs for this session?",
                    parent=ui.root):
                return
            for _, r, t in list(ui._sel_data.forward(name)):
                if _ == sess:
                    ui._sel_data.discard_from_group(name, sess, (r, t))
        for pair in pairs:
            ui._sel_data.add_to_group(name, sess, pair)
        ui._group_mgr._rebuild_groups_menu()
        ui.refresh_lists()
        messagebox.showinfo("Save selections",
                            f"Saved {len(pairs)} pairs to group '{name}'.",
                            parent=ui.root)

    def _on_zoom(self, _=None):
        """Called when H or V zoom slider changes — redraws with new spacing."""
        self.draw()

    def _make_group_button(self, gname: str, display: str, count_sess: str) -> ttk.Checkbutton:
        """Create and return a group checkbutton with optional pair-count label."""
        ui = self.ui
        if gname not in self._net_group_filter_vars:
            self._net_group_filter_vars[gname] = tk.BooleanVar(master=ui.root, value=False)
        if self._net_grp_counts_var.get():
            pairs_sess = ui._sel_data.pairs_in_group(gname, count_sess)
            n_hl  = len(ui._filter_pairs_to_conn_types(
                count_sess, pairs_sess, self._highlighted_ct_labels()))
            n_all = len({(r, t) for _, r, t in ui._sel_data.forward(gname)})
            label = f"{display} ({n_hl}/{len(pairs_sess)}/{n_all})"
        else:
            label = display
        return ttk.Checkbutton(
            self._net_grp_body, text=label,
            variable=self._net_group_filter_vars[gname],
            command=lambda g=gname: self._on_group_toggle(g))

    def refresh_group_buttons(self):
        """Rebuild the group toggle checkbuttons in the probe network panel."""
        ui = self.ui
        if not hasattr(self, '_net_grp_body'):
            return
        for w in self._net_grp_body.winfo_children():
            w.destroy()
        self._net_grp_items = []

        regular = sorted(k for k in ui._sel_data._groups if not k.startswith('__'))
        special = sorted(k for k in ui._sel_data._groups if k.startswith(_SPECIAL_PREFIX))
        # Remove vars for groups that no longer exist
        for g in set(self._net_group_filter_vars) - set(ui._sel_data._groups):
            del self._net_group_filter_vars[g]

        if not regular and not special:
            lbl = ttk.Label(self._net_grp_body, text="(no groups)", foreground='#888')
            self._net_grp_items.append((lbl, False))
            self.rewrap_group_buttons()
            return

        # Resolve which session to use for per-session counts.
        if getattr(ui, '_session_any_mode', False) and self._net_any_sessions_cache:
            _idx = max(0, min(self._net_any_idx, len(self._net_any_sessions_cache) - 1))
            _nk  = self._net_any_sessions_cache[_idx]
            count_sess = str(getattr(_nk, 'session', _nk))
        else:
            count_sess = ui._setup_mgr._current_session_str()

        for gname in regular:
            btn = self._make_group_button(gname, gname, count_sess)
            self._net_grp_items.append((btn, False))

        if special:
            sep = ttk.Separator(self._net_grp_body, orient='horizontal')
            self._net_grp_items.append((sep, True))
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
            self._net_grp_items.append((lbl, True))
            for gname in special:
                display = gname[len(_SPECIAL_PREFIX):]
                btn = self._make_group_button(gname, display, count_sess)
                self._net_grp_items.append((btn, False))

        self.rewrap_group_buttons()

    def rewrap_group_buttons(self):
        """Place group buttons in wrapping rows inside _net_grp_body."""
        if not hasattr(self, '_net_grp_body') or not self._net_grp_items:
            return
        body = self._net_grp_body
        body.update_idletasks()
        avail_w = body.winfo_width()
        if avail_w <= 1:
            body.after(100, self.rewrap_group_buttons)
            return

        for w, _ in self._net_grp_items:
            w.place_forget()

        PAD_X, PAD_Y = 2, 1
        x, y, row_h = PAD_X, PAD_Y, 0
        in_special = False
        for w, is_sep in self._net_grp_items:
            if is_sep:
                in_special = True
            if in_special and self._net_special_collapsed and not is_sep:
                continue
            w.update_idletasks()
            ww = w.winfo_reqwidth()
            wh = w.winfo_reqheight()
            if is_sep:
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
        pg = self._probegroups().get(ui.key.nd())
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
        self._focused_neuron = nid
        self._focused_pair = None
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
        for tk_ in ui._sess_mgr._available_type_keys(ui.key.nd()):
            pt = ui.cd.ptr.get(tk_)
            if pt is None or pt.inds is None:
                continue
            arr = pt.inds[:, -2:]
            tot_out += sum(1 for r, t in set(map(tuple, arr)) if r == nid)
            tot_in  += sum(1 for r, t in set(map(tuple, arr)) if t == nid)
        ct_label = ui._type_label(ui.key)
        self._focus_info_var.set(
            f"{ct_label}: in={cur_in} out={cur_out}  |  all: in={tot_in} out={tot_out}")

    def _on_neuron_focus_clear(self):
        self._focused_neuron = None
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
        for tk_ in ui._sess_mgr._available_type_keys(ui.key.nd()):
            pt = ui.cd.ptr.get(tk_)
            if pt is None or pt.inds is None:
                continue
            if pair in set(map(tuple, pt.inds2)):
                pair_exists = True
                break
        if not pair_exists:
            ui._group_mgr._show_temp_warning(f"Pair ({ref},{tgt}) not significant — showing position")
        self._focused_pair = pair
        self._focused_neuron = None
        self._focus_var.set("")
        self._focus_info_var.set("")
        self._update_pair_focus_info(pair, pair_exists)
        ui.refresh_lists()
        self.draw()
        ui._plot_mgr.update_plot()

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
        self._focused_pair = None
        self._focus_pair_var.set("")
        self._focus_pair_info_var.set("")
        self._add_pair_btn.config(state=tk.DISABLED)
        self.ui.refresh_lists()
        self.draw()
        self.ui._plot_mgr.update_plot()

    def _get_neuron_positions(self, x_scale=1.0, y_scale=1.0, nd_key=None, neurons=None):
        ui = self.ui
        if neurons is None:
            neurons = ui.neurons
        if neurons is None or neurons.peak_channels is None:
            return None
        if nd_key is None:
            nd_key = ui.key.nd()
        pg = self._probegroups().get(nd_key)
        if pg is None:
            print(f"[ProbeNetwork] No ProbeGroup for key={nd_key}. "
                  f"Available keys: {list(self._probegroups().keys())}")
            return None
        peak_ch = np.asarray(neurons.peak_channels, dtype=int)
        pg_df = pg.to_dataframe().set_index('channel_id')
        x_raw = pg_df['x'].reindex(peak_ch)
        y_raw = pg_df['y'].reindex(peak_ch)
        n_miss = int(x_raw.isna().sum())
        if n_miss:
            valid_chs = list(pg_df.index[:5])
            print(f"[ProbeNetwork] WARNING: {n_miss}/{len(peak_ch)} neurons "
                  f"have peak_channels not found in ProbeGroup.channel_id. "
                  f"peak_ch sample={peak_ch[:5].tolist()}, "
                  f"pg_ch sample={valid_chs}")
        x = x_raw.fillna(0.0).to_numpy(dtype=float) * x_scale
        y = y_raw.fillna(0.0).to_numpy(dtype=float) * y_scale
        return np.stack([x, y], axis=1), peak_ch

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
