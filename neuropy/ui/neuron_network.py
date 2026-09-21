"""Probe network panel — PySide6 + pyqtgraph implementation."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QSplitter,
    QLabel, QPushButton, QCheckBox, QLineEdit, QSlider, QFrame,
    QMessageBox, QInputDialog, QSizePolicy,
)
from pyqtgraph.Qt.QtCore import Qt, QTimer
from neuropy.ui.ui_common import LRUCache, area_rgb, cell_areas, _SPECIAL_PREFIX
from neuropy.ui.app_state import _ALL_SEGS
from neuropy.ui.utils import (chip_button, FlowLayout, SliderWithInput,
                              small_font_pt, widget_row)
from neuropy.analyses.neurons_dataset import Key

def _UI_FS() -> str:
    return f'font-size: {small_font_pt()}pt;'

# Legend glyphs: the same marks pyqtgraph draws for 't' and 'o'.
TRIANGLE = '&#9650;'
CIRCLE = '&#9679;'

def _css_rgb(rgb) -> str:
    """CSS colour from any int-like triple — a numpy tuple's repr is not valid CSS."""
    return 'rgb({},{},{})'.format(*(int(c) for c in rgb))

def _muted(html: str) -> str:
    return f'<span style="color:#999">{html}</span>'

def _dot(rgb) -> str:
    return f'<span style="color:{_css_rgb(rgb)}">{CIRCLE}</span>'

_DEFAULT_GRAYS = ((0, 0, 0), (51, 51, 51), (102, 102, 102),
                  (153, 153, 153), (187, 187, 187))


def _ramp(rgb, step: int, n: int) -> tuple:
    """Later slots lift toward white, keeping the hue readable at every step."""
    f = step / max(1, n - 1) * 0.55
    return tuple(int(c + (255 - c) * f) for c in rgb)


def _block(rgb) -> str:
    """One swatch of the gradient ramp — padding gives it width, so it needs no glyph."""
    return f'<span style="background:{_css_rgb(rgb)}; padding:0 5px">&#8202;</span>'

def _INPUT_SS() -> str:
    return (
        f'QLineEdit {{ background: #fff; color: #222; border: 1px solid #aaa; '
        f'font-size: {small_font_pt()}pt; padding: 2px 4px; }}'
    )

def _OUTLINE_BTN_SS() -> str:
    return (
        f'QPushButton {{ border: 1px solid #aaa; border-radius: 3px; '
        f'padding: 1px 8px; font-size: {small_font_pt()}pt; background: #fafafa; color: #222; }}'
        f'QPushButton:hover {{ background: #eef; }}'
    )


def _outline_button(label: str, width: int | None = None) -> QPushButton:
    btn = QPushButton(label)
    btn.setStyleSheet(_OUTLINE_BTN_SS())
    if width is not None:
        btn.setFixedWidth(width)
    return btn

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI as CCGReviewUI


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
        p1 = QtCore.QPointF(self._x1, self._y1)
        p2 = QtCore.QPointF(self._x2, self._y2)
        # Thin constant-width shaft: cosmetic pen = fixed pixels regardless of zoom.
        pen = QtGui.QPen(color, self._lw)
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.FlatCap)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        dx, dy = self._x2 - self._x1, self._y2 - self._y1
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        # Fixed-pixel, isotropic arrowhead: build it in pixel space via the device scale,
        # then map vertex offsets back to data units so unequal H/V zoom can't stretch it.
        dt = painter.deviceTransform()
        sx = math.hypot(dt.m11(), dt.m21()) or 1.0
        sy = math.hypot(dt.m12(), dt.m22()) or 1.0
        pdx, pdy = dx * sx, dy * sy
        pdist = math.hypot(pdx, pdy)
        if pdist < 1e-9:
            return
        ux, uy = pdx / pdist, pdy / pdist       # unit direction, in pixels
        nx, ny = -uy, ux
        HEAD_L, HEAD_W = 9.0, 4.5                # pixels
        base_x, base_y = -ux * HEAD_L, -uy * HEAD_L
        tri = QtGui.QPolygonF([
            p2,
            QtCore.QPointF(self._x2 + (base_x + nx * HEAD_W) / sx,
                           self._y2 + (base_y + ny * HEAD_W) / sy),
            QtCore.QPointF(self._x2 + (base_x - nx * HEAD_W) / sx,
                           self._y2 + (base_y - ny * HEAD_W) / sy),
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


@dataclass
class ProbeNetworkData:
    nd_key:       object
    session_label: str

    pos:          np.ndarray        # (n_neurons, 2) unscaled [x, y]
    peak_channels: np.ndarray       # (n_neurons,) int
    shank_ids:    np.ndarray | None
    neuron_type:  np.ndarray | None
    cell_area:    np.ndarray | None
    n_neurons:    int

    pg_info:      object | None     # ProbeGroup | None

    pair_entries: dict = field(default_factory=dict)
    # (ref, tgt) -> list[{key, conn_type, ei, is_current, in_filter}]

    sel_data:     object = field(default=None)
    # SelectionData — selected_inds / deleted_inds read live at render time

    current_pair: tuple | None = None


@dataclass
class NetworkView:
    """Everything one draw needs beyond the data: geometry, filters and highlights.

    Resolved once in _render from widget state, then passed whole so the draw
    methods take one argument instead of fifteen.
    """
    x: np.ndarray                 # neuron x, zoomed and spread
    y: np.ndarray                 # neuron y, zoomed
    slot_of: list                 # neuron -> slot on its channel, drives grayscale
    h_scale: float
    v_scale: float
    line_alpha: float
    hidden_shanks: frozenset
    enabled_cts: frozenset | None  # None means every connection type
    current_pair: tuple | None
    focused_neuron: int | None
    focused_pair: tuple | None
    selected_inds: set
    deleted_inds: set
    group_pairs: set
    group_filter_active: bool
    hide_same_channel: bool
    hide_same_shank: bool
    dark: bool
    fg_muted: str

    @property
    def focused_pair_neurons(self) -> set:
        return set(self.focused_pair) if self.focused_pair is not None else set()

    def hides_shank_of(self, shank_ids, ref: int, tgt: int) -> bool:
        """True when either endpoint sits on a shank the user switched off."""
        if not self.hidden_shanks or shank_ids is None:
            return False
        if ref >= len(shank_ids) or tgt >= len(shank_ids):
            return False
        return (int(shank_ids[ref]) in self.hidden_shanks
                or int(shank_ids[tgt]) in self.hidden_shanks)

    def same_shank(self, shank_ids, ref: int, tgt: int) -> bool:
        if shank_ids is None or ref >= len(shank_ids) or tgt >= len(shank_ids):
            return False
        return int(shank_ids[ref]) == int(shank_ids[tgt])

    def excluded_by_focus(self, ref: int, tgt: int) -> bool:
        """True when a neuron or pair focus is active and this pair is not in it."""
        if self.focused_neuron is not None and ref != self.focused_neuron \
                and tgt != self.focused_neuron:
            return True
        if self.focused_pair is not None:
            neurons = self.focused_pair_neurons
            if ref not in neurons and tgt not in neurons:
                return True
        return False


class _NavPlotWidget(pg.PlotWidget):
    """PlotWidget that forwards Up/Down to a callback for edge-to-edge navigation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._nav_key_cb = None

    _NAV_KEYS = (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Return, Qt.Key.Key_Enter)

    def keyPressEvent(self, ev):
        if (self._nav_key_cb is not None
                and ev.key() in self._NAV_KEYS
                and self._nav_key_cb(ev.key())):
            ev.accept()
            return
        super().keyPressEvent(ev)


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

    def __init__(self, parent: QWidget, ui: 'CCGReviewUI'):
        self.ui = ui
        self._focused_neuron: int | None = None
        self._focused_pair:   tuple | None = None
        self._font_scaled_widgets: list = []   # [(widget, lambda: stylesheet_str), ...]

        self._net_arrows:           bool  = True
        self._net_cur_pair:         bool  = True
        self._net_hide:             bool  = False
        self._net_hide_same_channel: bool = False
        self._net_hide_same_shank:  bool  = False
        self._net_grp_counts:       bool  = True
        self._net_hzoom:            float = 1.0
        self._net_vzoom:            float = 1.0
        self._net_line_alpha:       float = 1.0
        self._net_spread:           float = 1.0
        self._net_show_chid:        bool  = False
        self._net_show_nid:         bool  = False
        self._net_style_color:      bool  = True    # fill by brain region
        self._net_style_shape:      bool  = True    # marker by cell class
        self._net_style_gradient:   bool  = True    # lightness by channel slot
        self._net_show_pair_ind:    bool  = False
        self._net_special_collapsed: bool = False

        # Dicts: key → bool  (QCheckBox refs kept in *_cbs dicts)
        self._net_ct_vars:           dict = {}   # {conn_type: bool}
        self._net_ct_cbs:            dict = {}   # {conn_type: QCheckBox}
        self._net_group_filter_vars: dict = {}   # {gname: bool}
        self._net_grp_cbs:           dict = {}   # {gname: QCheckBox}
        self._net_shank_vars:        dict = {}   # {shank_id: bool}
        self._net_shank_cbs:         dict = {}   # {shank_id: QCheckBox}
        self._net_grp_items:         list = []

        self._data_cache: LRUCache = LRUCache(max_size=8)
        self._net_any_idx:             int  = 0
        self._net_any_sessions_cache:  list = []

        self._qt_app   = None
        self._plot_win = None
        self._draw_pending = False
        self._draw_timer = QTimer(singleShot=True)
        self._draw_timer.timeout.connect(self._draw_now)
        self._pg_items: list = []
        self._nav_edges: list = []   # visible edges (ref,tgt,key_str) for Up/Down stepping

        self._qt_app = QtWidgets.QApplication.instance()
        if self._qt_app is None:
            self._qt_app = QtWidgets.QApplication([])
        self._setup_controls(parent)
        self._setup_plot_window()

        QTimer.singleShot(200, self.draw)

    def _setup_plot_window(self):
        container = QWidget()
        container_lay = QVBoxLayout(container)
        container_lay.setContentsMargins(0, 0, 0, 0)
        container_lay.setSpacing(0)

        # Session caption (any-session mode only) — sits above the flanked plot.
        self._net_nav_label = QLabel('')
        self._net_nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scale_font(self._net_nav_label, lambda: f'font-size: {small_font_pt()}pt; color: #444; padding: 2px;')
        self._net_nav_label.setVisible(False)
        container_lay.addWidget(self._net_nav_label)

        # ◀ | plot | ▶ — arrows flank the plot edges (any-session session paging).
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(0)
        self._net_nav_left = QPushButton('◀')
        self._net_nav_left.setFixedWidth(24)
        self._net_nav_left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._net_nav_left.setVisible(False)
        self._net_nav_left.clicked.connect(self._on_net_arrow_left)
        row_lay.addWidget(self._net_nav_left)

        pw = _NavPlotWidget()
        pw._nav_key_cb = self._on_plot_nav_key
        pw.setBackground('w')
        pw.getViewBox().setAspectLocked(False)
        pw.hideAxis('bottom')
        pw.hideAxis('left')
        self._plot_win = pw
        row_lay.addWidget(pw, stretch=1)

        self._net_nav_right = QPushButton('▶')
        self._net_nav_right.setFixedWidth(24)
        self._net_nav_right.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._net_nav_right.setVisible(False)
        self._net_nav_right.clicked.connect(self._on_net_arrow_right)
        row_lay.addWidget(self._net_nav_right)

        # Legend floats over the plot itself (parented to pw, not the container) so the
        # top caption and flanking arrows never overlap it.
        self._legend_overlay = QLabel(pw)
        self._scale_font(self._legend_overlay, lambda: (
            'background: rgba(255,255,255,0.9); padding: 3px 6px; '
            f'border: 1px solid #ccc; font-size: {small_font_pt()}pt;'))
        self._legend_overlay.move(6, 6)
        self._legend_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._legend_overlay.raise_()

        container_lay.addWidget(row, stretch=1)
        # Slot 1 of the vertical splitter (below config panel)
        if hasattr(self, '_main_splitter'):
            self._main_splitter.addWidget(container)
            self._main_splitter.setStretchFactor(0, 0)  # config: fixed
            self._main_splitter.setStretchFactor(1, 1)  # plot: expands
            self._main_splitter.setCollapsible(0, False)
            self._main_splitter.setCollapsible(1, False)
            self._main_splitter.setSizes([180, 600])
            pw.setMinimumHeight(200)
            pw.getViewBox().setMouseEnabled(x=True, y=True)

    def _scale_font(self, widget, stylesheet_fn):
        """Apply stylesheet_fn() now and remember it so refresh_font() can reapply it live."""
        widget.setStyleSheet(stylesheet_fn())
        self._font_scaled_widgets.append((widget, stylesheet_fn))

    def refresh_font(self):
        for widget, stylesheet_fn in self._font_scaled_widgets:
            widget.setStyleSheet(stylesheet_fn())
        self.refresh_group_buttons()
        self.refresh_shank_buttons()

    def _make_section(self, parent_layout, title: str) -> tuple:
        """Return (section_widget, header_btn, body_widget, body_layout)."""
        sec = QWidget()
        sec.setStyleSheet('QWidget { border: 1px solid #ccc; }')
        sec_lay = QVBoxLayout(sec)
        sec_lay.setContentsMargins(2, 2, 2, 2)
        sec_lay.setSpacing(1)

        hdr = QWidget()
        hdr.setStyleSheet('QWidget { border: none; }')
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(2)
        arrow_btn = QPushButton('▾ ' + title)
        arrow_btn.setFlat(True)
        self._scale_font(arrow_btn, lambda: f'font-weight: bold; {_UI_FS()} text-align: left;')
        hdr_lay.addWidget(arrow_btn)
        hdr_lay.addStretch()
        sec_lay.addWidget(hdr)

        body = QWidget()
        body.setStyleSheet('QWidget { border: none; }')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(4, 2, 4, 2)
        body_lay.setSpacing(2)
        sec_lay.addWidget(body)

        def _toggle():
            visible = body.isVisible()
            body.setVisible(not visible)
            arrow_btn.setText(('▾ ' if not visible else '▸ ') + title)
        arrow_btn.clicked.connect(_toggle)

        parent_layout.addWidget(sec)
        return sec, arrow_btn, body, body_lay, hdr_lay

    def _setup_controls(self, parent: QWidget):
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter: config panel (top) | plot (bottom)
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(self._main_splitter)

        # Config panel — scrollable, sits above the plot
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._config_scroll = scroll

        ctrl = QWidget()
        self._ctrl_layout = QVBoxLayout(ctrl)
        self._ctrl_layout.setContentsMargins(4, 2, 4, 2)
        self._ctrl_layout.setSpacing(2)
        scroll.setWidget(ctrl)
        self._main_splitter.addWidget(scroll)   # slot 0 = config (top)

        self._setup_toggles_and_groups(self._ctrl_layout)
        self._setup_focus_panels(self._ctrl_layout)
        self._ctrl_layout.addStretch()

    def _setup_focus_panels(self, layout):
        _, _, body, body_lay, _ = self._make_section(layout, 'Focus')

        n_row = QWidget()
        n_lay = QHBoxLayout(n_row)
        n_lay.setContentsMargins(0, 0, 0, 0)
        n_lay.setSpacing(4)
        n_chip = chip_button('Neuron', checkable=False)
        n_lay.addWidget(n_chip)
        self._focus_entry = QLineEdit()
        self._focus_entry.setFixedWidth(64)
        self._scale_font(self._focus_entry, _INPUT_SS)
        n_lay.addWidget(self._focus_entry)
        n_clr = chip_button('✕', checkable=False)
        n_clr.setFixedWidth(28)
        n_lay.addWidget(n_clr)
        self._same_ch_btn = chip_button('Next on ch', checkable=False)
        self._same_ch_btn.setToolTip(
            'Focus the next neuron sharing this one\'s peak channel (Enter on the network)')
        self._same_ch_btn.clicked.connect(lambda: self.focus_next_on_channel())
        n_lay.addWidget(self._same_ch_btn)
        n_lay.addStretch()
        body_lay.addWidget(n_row)

        self._focus_info_label = QLabel('')
        self._scale_font(self._focus_info_label, lambda: f'color: #555; {_UI_FS()} border: none;')
        body_lay.addWidget(self._focus_info_label)

        p_row = QWidget()
        p_lay = QHBoxLayout(p_row)
        p_lay.setContentsMargins(0, 0, 0, 0)
        p_lay.setSpacing(4)
        sep = QLabel('|')
        self._scale_font(sep, lambda: f'color: #bbb; font-size: {small_font_pt()}pt;')
        p_lay.addWidget(sep)
        p_chip = chip_button('Pair', checkable=False)
        p_lay.addWidget(p_chip)
        self._focus_pair_entry = QLineEdit()
        self._focus_pair_entry.setFixedWidth(80)
        self._scale_font(self._focus_pair_entry, _INPUT_SS)
        p_lay.addWidget(self._focus_pair_entry)
        p_clr = chip_button('✕', checkable=False)
        p_clr.setFixedWidth(28)
        p_lay.addWidget(p_clr)
        self._add_pair_btn = chip_button('Add to available', checkable=False)
        self._add_pair_btn.setEnabled(False)
        self._add_pair_btn.clicked.connect(
            lambda: getattr(self.ui.root, "_misc_mgr", None)._on_add_focused_pair()
            if hasattr(self.ui, '_misc_mgr') else None)
        p_lay.addWidget(self._add_pair_btn)
        p_lay.addStretch()
        body_lay.addWidget(p_row)

        self._focus_pair_info_label = QLabel('')
        self._scale_font(self._focus_pair_info_label, lambda: f'color: #555; {_UI_FS()} border: none;')
        body_lay.addWidget(self._focus_pair_info_label)

        self._focus_entry.returnPressed.connect(self._on_neuron_focus)
        n_clr.clicked.connect(self._on_neuron_focus_clear)
        self._focus_pair_entry.returnPressed.connect(self._on_pair_focus)
        p_clr.clicked.connect(self._on_pair_focus_clear)

    @staticmethod
    def _row_widget(row_layout) -> QWidget:
        """Wrap a row layout in a widget, for layouts that take widgets only."""
        holder = QWidget()
        holder.setLayout(row_layout)
        return holder

    @staticmethod
    def _pipe() -> QLabel:
        """The thin gray '|' this panel uses between chip clusters."""
        sep = QLabel('|')
        sep.setStyleSheet('color: #bbb;')
        return sep

    def _toggle_chip(self, label: str, attr: str, checked: bool,
                     also=None) -> QPushButton:
        """A chip that writes one bool onto the panel and redraws."""
        chip = chip_button(label, checked=checked)
        if also is None:
            chip.toggled.connect(lambda v: self._set_bool(attr, v))
        else:
            chip.toggled.connect(lambda v: (self._set_bool(attr, v), also()))
        return chip

    def _setup_toggles_and_groups(self, layout):
        _, _, lines_body, lines_lay, _ = self._make_section(layout, 'Lines')

        conn_type_label = QLabel('Conn type:')
        self._scale_font(conn_type_label, _UI_FS)
        self._net_ct_frame = QWidget()
        self._net_ct_frame_lay = QHBoxLayout(self._net_ct_frame)
        self._net_ct_frame_lay.setContentsMargins(0, 0, 0, 0)
        self._net_ct_frame_lay.setSpacing(2)
        lines_lay.addWidget(self._row_widget(widget_row(
            self._toggle_chip('ON/OFF', '_net_arrows', True), self._pipe(),
            self._toggle_chip('Current pair', '_net_cur_pair', True), self._pipe(),
            conn_type_label, self._net_ct_frame, spacing=4)))
        self.refresh_ct_buttons()

        lines_lay.addWidget(self._row_widget(widget_row(
            self._toggle_chip('Hide', '_net_hide', False), self._pipe(),
            self._toggle_chip('Same channel', '_net_hide_same_channel', False,
                              self._on_toggle_hide_same_channel),
            self._toggle_chip('Same shank', '_net_hide_same_shank', False,
                              self._on_toggle_hide_same_shank), spacing=4)))

        _, _, grp_body, grp_lay, grp_hdr_lay = self._make_section(layout, 'Groups')
        cb_counts = QCheckBox('Show counts')
        cb_counts.setChecked(True)
        cb_counts.toggled.connect(lambda v: (self._set_bool('_net_grp_counts', v),
                                             self.refresh_group_buttons()))
        grp_hdr_lay.addWidget(cb_counts)
        btn_grp_sync = _outline_button('Sync', 48)
        btn_grp_sync.clicked.connect(
            lambda: (self.ui.ensure_groups_loaded_for([str(self.ui.key.session)]),
                     self.refresh_group_buttons(), self.draw()))
        grp_hdr_lay.addWidget(btn_grp_sync)
        btn_grp_clr = _outline_button('Clear all', 72)
        btn_grp_clr.clicked.connect(self._on_group_clear)
        grp_hdr_lay.addWidget(btn_grp_clr)
        self._net_grp_body_widget = grp_body
        self._net_grp_body_lay    = grp_lay
        self.refresh_group_buttons()

        _, _, ann_body, ann_lay, _ = self._make_section(layout, 'Annotations')

        # Probe shanks
        btn_none = _outline_button('None', 48)
        btn_none.clicked.connect(lambda: self._probe_all_shanks(False))
        btn_all = _outline_button('All', 40)
        btn_all.clicked.connect(lambda: self._probe_all_shanks(True))
        ann_lay.addWidget(self._row_widget(widget_row(
            self._heading('Probe shanks'), btn_none, btn_all, spacing=2)))

        self._net_probe_flow = QWidget()
        self._net_probe_flow_lay = FlowLayout(self._net_probe_flow, spacing=4)
        ann_lay.addWidget(self._net_probe_flow)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet('color: #ccc;')
        ann_lay.addWidget(separator)

        ann_lay.addWidget(self._heading('Zoom / Appearance'))
        ann_lay.addWidget(self._zoom_grid())

        ann_lay.addWidget(self._heading('Annotations'))
        ann_lay.addWidget(self._row_widget(widget_row(*(
            self._toggle_chip(text, attr, getattr(self, attr, False))
            for text, attr in (('channel ids', '_net_show_chid'),
                               ('neuron ids', '_net_show_nid'),
                               ('pair inds', '_net_show_pair_ind'))))))

        # each style toggle sits beside the key it switches on
        ann_lay.addWidget(self._heading('Neuron style'))
        self._net_style_row = QWidget()
        self._net_style_lay = FlowLayout(self._net_style_row, spacing=6)
        ann_lay.addWidget(self._net_style_row)
        self.refresh_style_row()

    def _heading(self, text: str) -> QLabel:
        """A bold section heading that tracks the UI font size."""
        label = QLabel(text)
        self._scale_font(label, lambda: f'font-weight: bold; {_UI_FS()}')
        return label

    _ZOOM_SLIDERS = (('H:', '_net_hzoom', 20, 1500, 100, 0.01),
                     ('V:', '_net_vzoom', 20, 1500, 100, 0.01),
                     ('α:', '_net_line_alpha', 5, 100, 100, 0.01),
                     ('spread:', '_net_spread', 0, 1500, 100, 0.01))

    def _zoom_grid(self) -> QWidget:
        """The zoom and appearance sliders, two per row."""
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        for i, (text, attr, lo, hi, init, scale) in enumerate(self._ZOOM_SLIDERS):
            row, col = divmod(i, 2)
            label = QLabel(text)
            self._scale_font(label, _UI_FS)
            slider = SliderWithInput(lo, hi, init, scale=scale)
            slider.value_changed.connect(
                lambda v, a=attr: (setattr(self, a, v), self._on_zoom()))
            grid.addWidget(label, row, col * 2)
            grid.addWidget(slider, row, col * 2 + 1)
        return holder

    _STYLE_TOGGLES = (('color',    '_net_style_color',    '_area_legend_html'),
                      ('shape',    '_net_style_shape',    '_shape_legend_html'),
                      ('gradient', '_net_style_gradient', '_gradient_legend_html'))

    def refresh_style_row(self):
        """Build the style toggles and their keys — areas differ per session."""
        lay = self._net_style_lay
        self._style_keys = {}
        for text, attr, html in self._STYLE_TOGGLES:
            cb = chip_button(text, checked=getattr(self, attr))
            cb.toggled.connect(lambda v, a=attr: (self._set_bool(a, v),
                                                  self.refresh_style_keys()))
            label = self._legend_label(getattr(self, html)())
            self._style_keys[attr] = label
            lay.addWidget(cb)
            lay.addWidget(label)

    def refresh_style_keys(self):
        """Retext the keys; rebuilding the row would delete the chip mid-signal."""
        for _, attr, html in self._STYLE_TOGGLES:
            self._style_keys[attr].setText(getattr(self, html)())

    def _area_legend_html(self) -> str:
        """A dot per region present, so the key never lists what is not on screen."""
        areas = self._areas_present()
        if not areas:
            return _muted('no region labels')
        palette = self._area_palette()
        return ' '.join(f'{_dot(area_rgb(a, palette))} {a}' for a in areas)

    def _shape_legend_html(self) -> str:
        if not self._net_style_shape:
            return _muted(f'{CIRCLE} all round')
        return f'{TRIANGLE} pyr &#8195; {CIRCLE} inter'

    def _gradient_legend_html(self) -> str:
        """The ramp itself, drawn as the blocks a neuron can actually take."""
        grays = self._grays()
        areas = self._areas_present()
        base = (area_rgb(areas[0], self._area_palette())
                if self._net_style_color and areas else None)
        blocks = [_block(grays[step] if base is None else _ramp(base, step, len(grays)))
                  for step in range(len(grays))]
        return f'{"".join(blocks)} by channel'

    def _legend_label(self, html: str) -> QLabel:
        lbl = QLabel(html)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        self._scale_font(lbl, lambda: f'color: #666; font-size: {small_font_pt()}pt;')
        return lbl

    def _symbol_for(self, ntype) -> str:
        """Triangle for pyramidal, circle for interneuron; all round when off."""
        if not self._net_style_shape:
            return 'o'
        return 'o' if ntype == 'inter' else 't'

    def _grays(self) -> tuple:
        return getattr(getattr(self, '_plot_theme', None), 'neuron_grays', None) or _DEFAULT_GRAYS

    def _neuron_rgb(self, data, idx: int, slot_of, grays, palette) -> tuple:
        """One neuron's fill: region hue, shaded by channel slot."""
        step = slot_of[idx] % len(grays)
        if not self._net_style_color:
            return grays[step] if self._net_style_gradient else grays[0]
        area = (data.cell_area[idx]
                if data.cell_area is not None and idx < len(data.cell_area) else None)
        rgb = area_rgb(area, palette)
        return _ramp(rgb, step, len(grays)) if self._net_style_gradient else rgb

    def _area_palette(self) -> dict:
        return self.ui.root.settings.area_colors

    def _areas_present(self) -> list:
        """Region names in the loaded session, in palette order then alphabetical."""
        found = cell_areas(getattr(self.ui, 'neurons', None))
        if found is None:
            return []
        names = {str(a) for a in np.asarray(found).ravel().tolist()}
        order = list(self._area_palette())
        return sorted(names, key=lambda n: (order.index(n) if n in order else len(order), n))

    def _set_bool(self, attr: str, value: bool):
        setattr(self, attr, value)
        self.draw()

    def refresh_ct_buttons(self):
        ui = self.ui
        lay = getattr(self, '_net_ct_frame_lay', None)
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
        self._net_ct_cbs.clear()

        cur_ct = getattr(ui.key, 'conn_type', None)
        ct_set = set(self._CT_LABELS.keys())
        conf = getattr(ui.cd, 'conf', None)
        if conf is not None:
            for _, ct in getattr(conf, 'conn_types_labeled', []):
                ct_set.add(ct)
        try:
            nd = ui.key.nd()
            for k in ui.cd.ptr.keys():
                if k.nd() == nd:
                    ct = getattr(k, 'conn_type', None)
                    if ct:
                        ct_set.add(ct)
        except Exception:
            pass

        for ct in sorted(ct_set, key=lambda x: str(x)):
            lbl = self._CT_LABELS.get(ct)
            if lbl is None:
                a, b = ct if isinstance(ct, tuple) and len(ct) == 2 else (str(ct), '')
                lbl = f'{str(a)[0].upper()}→{str(b)[0].upper()}' if b else str(a)[:3]
            if ct not in self._net_ct_vars:
                self._net_ct_vars[ct] = True
            cb = chip_button(lbl, checked=self._net_ct_vars[ct])
            cb.toggled.connect(lambda v, c=ct: (self._net_ct_vars.__setitem__(c, v), self.draw()))
            # Tinted with the colour this type's arrows are drawn in, so the
            # filter row doubles as the connection legend.
            r, g, b, _ = _ct_rgba(ct)
            cb.setStyleSheet(cb.styleSheet() + (
                f'QPushButton {{ background: rgba({r},{g},{b},40); '
                f'border: 1px solid rgb({r},{g},{b}); }}'
                f'QPushButton:checked {{ background: rgba({r},{g},{b},170); color: white; }}'))
            lay.addWidget(cb)
            self._net_ct_cbs[ct] = cb
        for ct in set(self._net_ct_vars) - ct_set:
            del self._net_ct_vars[ct]

    def _probe_all_shanks(self, val: bool):
        for k in self._net_shank_vars:
            self._net_shank_vars[k] = val
            if k in self._net_shank_cbs:
                self._net_shank_cbs[k].setChecked(val)
        self.draw()

    def draw(self):
        """Coalesce the bursts of draw() a session switch fires into one repaint."""
        if self._plot_win is None:
            return
        self._draw_pending = True
        if self._plot_win.isVisible():
            self._draw_timer.start(0)

    def _draw_now(self):
        if self._plot_win is None or not self._plot_win.isVisible():
            return
        self._draw_pending = False
        self._draw_impl()
        self._update_nav_bar()

    def draw_if_pending(self):
        if self._draw_pending:
            self.draw()

    def _draw_impl(self):
        ui = self.ui
        pw = self._plot_win

        any_mode = ui.session_any_mode
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
            _type_key = ui.type_key_for_nd(_sess_nk)
            segment_filter = None
        else:
            _nd_key        = ui.key.nd()
            _sess_nk       = None
            _sessions      = None
            _type_key      = ui.key
            segment_filter = None if ui.current_segment in (None, _ALL_SEGS) else ui.segment_index(ui.current_segment)

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

    def _make_cache_key(self, nd_key, type_key,
                        segment_filter, any_mode: bool) -> tuple:
        ui    = self.ui
        b     = ui.active_selections if ui.sel_data is not None else None
        unsel = frozenset(b.unselected) if b is not None else frozenset()
        sel   = frozenset(b.selected)   if b is not None else frozenset()
        dlt   = frozenset(b.deleted)    if b is not None else frozenset()
        return (nd_key, str(type_key), segment_filter, unsel, sel, dlt)

    def _assemble_data(self, nd_key, type_key, segment_filter,
                       any_mode: bool, sess_nk) -> ProbeNetworkData | None:
        ui = self.ui

        if any_mode:
            _neurons = (ui.cd.nd.neurons_for(sess_nk)
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
        cell_area   = cell_areas(_neurons)
        pg_info     = self._probegroups().get(nd_key)

        if any_mode:
            visible_pairs = _ptr.pair_set if _ptr is not None else set()
            current_pair  = None
        else:
            visible_pairs = ui.all_pairs_set
            current_pair  = (tuple(ui.all_pairs_np[ui.current_pair_idx])
                             if ui.current_pair_idx < len(ui.all_pairs_np)
                             else None)

        pair_entries: dict = {}
        for tk_ in ui.available_type_keys(nd_key):
            pt = ui.cd.ptr.get(tk_)
            inds = pt.pairs
            if inds is None or len(inds) == 0:
                continue
            ct     = getattr(tk_, 'conn_type',    None)
            ei     = getattr(tk_, 'excitability', 'E')
            is_cur = (tk_ == type_key)
            for ref, tgt in map(tuple, inds):
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

        session_label = (ui.session_label(sess_nk)
                         if any_mode and sess_nk is not None else '')

        return ProbeNetworkData(
            nd_key=nd_key, session_label=session_label,
            pos=xy, peak_channels=peak_ch,
            shank_ids=shank_ids, neuron_type=neuron_type, cell_area=cell_area,
            n_neurons=n_neurons, pg_info=pg_info,
            pair_entries=pair_entries, sel_data=ui.sel_data,
            current_pair=current_pair,
        )

    def _render(self, data: ProbeNetworkData,
                any_mode: bool, sess_nk, sessions):
        pw = self._plot_win
        pw.clear()
        self._pg_items.clear()

        ui = self.ui
        theme = getattr(ui, 'theme', None)
        if theme is not None:
            pw.setBackground(theme.plot_bg)
        self._plot_theme = theme

        # authoritative here: probe df and neuron shank_ids are populated by now,
        # unlike at panel build time
        self._sync_shank_chips(self._shanks_from_data(data))

        view = self._resolve_view(data, any_mode, sess_nk)
        x_pos = data.pos[:, 0] * view.h_scale
        y_pos = data.pos[:, 1] * view.v_scale

        self._draw_probe_bg(data, x_pos, y_pos, view)

        if self._net_arrows:
            self._draw_connections(data, view)
        else:
            self._nav_edges = []

        self._draw_neurons(data, view)
        self._draw_legend(data, view.enabled_cts)

        if data.session_label:
            lbl_color = (getattr(self._plot_theme, 'fg_muted', None)
                         or '#444444')
            ti = pg.TextItem(data.session_label, color=lbl_color,
                             anchor=(0.5, 1.0))
            ti.setPos(float(np.mean(x_pos)), float(np.max(y_pos)) + 20)
            pw.addItem(ti)
            self._pg_items.append(ti)

    def _resolve_view(self, data, any_mode: bool, sess_nk) -> NetworkView:
        """Read the panel's widget state into the NetworkView one draw runs on."""
        ui = self.ui
        h_scale, v_scale = self._net_hzoom, self._net_vzoom
        x_pos = data.pos[:, 0] * h_scale
        # spread once: arrows must land on the spread neuron, not the channel column
        x_spread, slot_of = self._compute_spread(data, x_pos)

        enabled_cts = frozenset(ct for ct, v in self._net_ct_vars.items() if v)
        if len(enabled_cts) == len(self._net_ct_vars):
            enabled_cts = None

        focused_pair = self._focused_pair
        group_pairs, group_filter_active = self._resolve_group_filter(
            data, any_mode, focused_pair)
        selections = ui.active_selections if ui.sel_data is not None else None

        return NetworkView(
            x=x_spread, y=data.pos[:, 1] * v_scale, slot_of=slot_of,
            h_scale=h_scale, v_scale=v_scale,
            line_alpha=max(0.05, min(1.0, self._net_line_alpha)),
            hidden_shanks=frozenset(s for s, v in self._net_shank_vars.items() if not v),
            enabled_cts=enabled_cts,
            current_pair=self._resolve_current_pair(any_mode, sess_nk),
            focused_neuron=self._focused_neuron, focused_pair=focused_pair,
            selected_inds=set(selections.selected) if selections else set(),
            deleted_inds=set(selections.deleted) if selections else set(),
            group_pairs=group_pairs, group_filter_active=group_filter_active,
            hide_same_channel=self._net_hide_same_channel,
            hide_same_shank=self._net_hide_same_shank,
            dark=bool(getattr(self._plot_theme, 'dark', False)),
            fg_muted=getattr(self._plot_theme, 'fg_muted', '#555'),
        )

    def _resolve_current_pair(self, any_mode: bool, sess_nk) -> tuple | None:
        """The pair under review, or None. Live state, so never cached with the data."""
        ui = self.ui
        if not any_mode:
            if ui.current_pair_idx >= len(ui.all_pairs_np):
                return None
            return tuple(ui.all_pairs_np[ui.current_pair_idx])
        # all-session mode: highlight it only on the session that owns it
        handle = ui.current_pair
        if handle is None:
            return None
        key, ref, tgt = handle
        if str(key.session) != str(getattr(sess_nk, 'session', sess_nk)):
            return None
        return (int(ref), int(tgt))

    def _resolve_group_filter(self, data, any_mode: bool, focused_pair):
        """(pairs, active) for the checked groups; a pair focus overrides the filter."""
        active_groups = {g for g, v in self._net_group_filter_vars.items() if v}
        if not active_groups or focused_pair is not None:
            return set(), False
        session = (str(getattr(data.nd_key, 'session', data.nd_key)) if any_mode
                   else self.ui.current_session_str)
        pairs = set()
        for group in active_groups:
            pairs |= self.ui.groups.pairs_in_group(group, session)
        return pairs, True

    def _draw_probe_bg(self, data, x_pos, y_pos, view: NetworkView):
        pw      = self._plot_win
        pg_info = data.pg_info
        if pg_info is None:
            return
        try:
            df = pg_info.to_dataframe()
        except Exception:
            return

        skipped = self._skipped_channels()
        show_chid = self._net_show_chid
        ch_brush = (140, 140, 140, 160) if view.dark else (180, 180, 180, 160)
        lbl_muted = getattr(self._plot_theme, 'fg_muted', '#888888')
        for shank_id in df['shank_id'].unique():
            if int(shank_id) in view.hidden_shanks:
                continue
            sub    = df[df['shank_id'] == shank_id]
            xs     = sub['x'].to_numpy(dtype=float) * view.h_scale
            ys     = sub['y'].to_numpy(dtype=float) * view.v_scale
            ch_ids = sub['channel_id'].to_numpy()

            # Normal channels
            good_mask = np.array([int(c) not in skipped for c in ch_ids])
            bad_mask  = ~good_mask
            if good_mask.any():
                sc = pg.ScatterPlotItem(
                    x=xs[good_mask], y=ys[good_mask], size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*ch_brush), symbol='o')
                sc.setZValue(1)
                pw.addItem(sc)
                self._pg_items.append(sc)
            if bad_mask.any():
                sc_bad = pg.ScatterPlotItem(
                    x=xs[bad_mask], y=ys[bad_mask], size=5,
                    pen=pg.mkPen('#CC0000', width=1),
                    brush=pg.mkBrush(*ch_brush), symbol='o')
                sc_bad.setZValue(1)
                pw.addItem(sc_bad)
                self._pg_items.append(sc_bad)

            lbl = pg.TextItem(f'S{int(shank_id)}', color=lbl_muted,
                              anchor=(0.5, 0.0))
            lbl.setPos(float(np.mean(xs)), float(np.max(ys)) + 8)
            pw.addItem(lbl)
            self._pg_items.append(lbl)
            if show_chid:
                for xi, yi, cid in zip(xs, ys, ch_ids):
                    t = pg.TextItem(str(cid), color=lbl_muted, anchor=(0.0, 0.5))
                    t.setPos(float(xi) + 3, float(yi))
                    t.setFont(QtGui.QFont('Arial', 6))
                    pw.addItem(t)
                    self._pg_items.append(t)

    def _skipped_channels(self) -> set:
        """Channels the recording marked bad or discarded; drawn with a red ring."""
        nd_conf = getattr(getattr(self.ui.cd, 'nd', None), '_conf', None)
        recinfo = getattr(nd_conf, 'recinfo', None) if nd_conf else None
        return (set(getattr(recinfo, 'skipped_channels', None) or [])
                | set(getattr(recinfo, 'discarded_channels', None) or []))

    def _compute_spread(self, data, x_pos):
        """Per-channel horizontal fan-out so same-channel neurons don't overlap.
        Returns (x_spread, slot_of); slot_of[i] is neuron i's slot on its channel
        (drives grayscale). Shared by neuron dots and connection endpoints."""
        ch_slots: dict = {}
        for i, ch in enumerate(data.peak_channels):
            ch_slots.setdefault(int(ch), []).append(i)
        x_spread = x_pos.copy()
        slot_of  = [0] * data.n_neurons
        step = 5.0 * self._net_spread
        for idxs in ch_slots.values():
            offsets = (np.arange(len(idxs)) - (len(idxs) - 1) / 2.0) * step
            for slot, ni in enumerate(idxs):
                slot_of[ni]   = slot
                x_spread[ni] += offsets[slot]
        return x_spread, slot_of

    def _draw_connections(self, data, view: NetworkView):
        """Draw every visible edge, then the current pair, arcs and deleted pairs."""
        nav_edges: list = []
        same_channel: dict = {}   # channel -> arcs to fan out around it

        for (ref, tgt), entries in data.pair_entries.items():
            if self._skip_pair(data, view, ref, tgt):
                continue
            for entry in entries:
                self._draw_one_entry(data, view, ref, tgt, entry,
                                     nav_edges, same_channel)

        self._draw_current_pair(data, view, nav_edges)
        self._draw_same_channel_arcs(view, same_channel, nav_edges)
        self._draw_deleted_pairs(data, view)

        # one entry per pair, ordered, for Up/Down stepping; deleted are excluded
        first_key: dict = {}
        for ref, tgt, key in nav_edges:
            first_key.setdefault((ref, tgt), key)
        self._nav_edges = [(ref, tgt, key)
                           for (ref, tgt), key in sorted(first_key.items())]

    def _skip_pair(self, data, view: NetworkView, ref: int, tgt: int) -> bool:
        """Filters that reject a whole pair before any of its entries are styled."""
        if not (0 <= ref < data.n_neurons and 0 <= tgt < data.n_neurons):
            return True
        # the current pair gets its own highlight edge past every filter, so
        # skipping it here just avoids drawing it twice
        if view.current_pair is not None and (ref, tgt) == view.current_pair:
            return True
        if view.excluded_by_focus(ref, tgt):
            return True
        if view.group_filter_active and (ref, tgt) not in view.group_pairs:
            return True
        return view.hides_shank_of(data.shank_ids, ref, tgt)

    def _entry_style(self, view: NetworkView, entry, ref: int, tgt: int,
                     is_current_pair: bool) -> tuple:
        """(rgba, alpha) for one edge: colour by type, alpha by focus and selection."""
        muted_gray = (120, 120, 120, 255) if view.dark else (204, 204, 204, 255)
        rgba = _ct_rgba(entry['conn_type'], entry.get('ei', 'E'))
        focused_pair = view.focused_pair
        is_selected = (ref, tgt) in view.selected_inds
        is_current = entry['is_current']

        if is_current_pair or (focused_pair is not None and (ref, tgt) == focused_pair):
            return rgba, 1.00
        if focused_pair is not None:
            return muted_gray, 0.12
        if not entry['in_filter']:
            return muted_gray, 0.20
        if not is_current:
            return rgba, 0.70 if is_selected else 0.35
        return rgba, 0.90 if is_selected else 0.55

    def _pair_offset(self, view: NetworkView, pairs, ref: int, tgt: int) -> tuple:
        """Perpendicular nudge so a reciprocal pair's two arrows don't overlap."""
        if (tgt, ref) not in pairs:
            return 0.0, 0.0
        return _perp_offset(view.x[ref], view.y[ref], view.x[tgt], view.y[tgt], d=3.0)

    def _draw_one_entry(self, data, view: NetworkView, ref: int, tgt: int, entry,
                        nav_edges: list, same_channel: dict) -> None:
        if view.enabled_cts is not None and entry['conn_type'] not in view.enabled_cts:
            return
        if view.hide_same_shank and view.same_shank(data.shank_ids, ref, tgt):
            return
        is_same_channel = (data.peak_channels is not None
                           and data.peak_channels[ref] == data.peak_channels[tgt])
        if view.hide_same_channel and is_same_channel:
            return

        is_current_pair = entry['is_current'] and (ref, tgt) == view.current_pair
        rgba, alpha = self._entry_style(view, entry, ref, tgt, is_current_pair)
        rgba = _with_alpha(rgba, alpha * view.line_alpha)
        width = 2.6   # cosmetic px, so the shaft stays constant at any zoom

        if is_same_channel:
            channel = int(data.peak_channels[ref])
            same_channel.setdefault(channel, []).append(
                (ref, tgt, entry, is_current_pair, rgba, width))
            return

        ox, oy = self._pair_offset(view, data.pair_entries, ref, tgt)
        self._add_arrow(view, ref, tgt, ox, oy, rgba, width,
                        f'{ref}_{tgt}_{entry["key"]}', z=3)
        nav_edges.append((ref, tgt, str(entry['key'])))

        if self._net_show_pair_ind and entry['is_current']:
            self._add_pair_label(view, ref, tgt, ox, oy)
        if is_current_pair:
            outline_rgba = (255, 255, 255, 255) if view.dark else (0, 0, 0, 255)
            self._add_arrow(view, ref, tgt, ox, oy,
                            _with_alpha(outline_rgba, alpha * view.line_alpha),
                            width + 1.5, '', z=2, clickable=False)

    def _add_arrow(self, view: NetworkView, ref: int, tgt: int,
                   ox: float, oy: float, rgba, width: float, gid: str,
                   z: int, clickable: bool = True) -> None:
        item = ProbeConnectionItem(view.x[ref] + ox, view.y[ref] + oy,
                                   view.x[tgt] + ox, view.y[tgt] + oy,
                                   rgba, width, gid)
        item.setZValue(z)
        if clickable:
            item.sigClicked.connect(self._on_arrow_click)
        self._plot_win.addItem(item)
        self._pg_items.append(item)

    def _add_pair_label(self, view: NetworkView, ref: int, tgt: int,
                        ox: float, oy: float) -> None:
        label = pg.TextItem(f'{ref},{tgt}', color=view.fg_muted, anchor=(0.5, 1.0))
        label.setPos((view.x[ref] + view.x[tgt]) / 2 + ox,
                     (view.y[ref] + view.y[tgt]) / 2 + oy)
        label.setFont(QtGui.QFont('Arial', 6))
        self._plot_win.addItem(label)
        self._pg_items.append(label)

    def _draw_current_pair(self, data, view: NetworkView, nav_edges: list) -> None:
        """Drawn last and unfiltered, so no conn-type filter can hide the pair under review."""
        if view.current_pair is None or not self._net_cur_pair:
            return
        ref, tgt = int(view.current_pair[0]), int(view.current_pair[1])
        if not (0 <= ref < data.n_neurons and 0 <= tgt < data.n_neurons):
            return
        ox, oy = self._pair_offset(view, data.pair_entries, ref, tgt)
        self._add_arrow(view, ref, tgt, ox, oy, (255, 20, 147, 255), 4.4,
                        f'{ref}_{tgt}_cur', z=20)
        nav_edges.append((ref, tgt, None))   # already the current type, no switch needed
        if self._net_show_pair_ind:
            self._add_pair_label(view, ref, tgt, ox, oy)

    def _draw_same_channel_arcs(self, view: NetworkView, same_channel: dict,
                                nav_edges: list) -> None:
        """Same-channel pairs have no length, so they become nested arcs instead."""
        base_radius, radius_step, gap = 7, 5, 11
        for arcs in same_channel.values():
            anchor = arcs[0][0]
            cx, cy = float(view.x[anchor]) + gap, float(view.y[anchor])
            for i, (ref, tgt, entry, is_current_pair, rgba, _w) in enumerate(arcs):
                arc = SameChannelArcItem(cx, cy, base_radius + i * radius_step, rgba,
                                         lw=(2.0 if is_current_pair else 1.0),
                                         gid=f'{ref}_{tgt}_{entry["key"]}')
                arc.setZValue(4)
                arc.sigClicked.connect(self._on_arrow_click)
                self._plot_win.addItem(arc)
                self._pg_items.append(arc)
                nav_edges.append((ref, tgt, str(entry['key'])))

    def _draw_deleted_pairs(self, data, view: NetworkView) -> None:
        """Deleted pairs still show, faded, unless they also exist as a live pair."""
        gray = (180, 180, 180, 255) if view.dark else (51, 51, 51, 255)
        rgba = _with_alpha(gray, 0.20 * view.line_alpha)
        for ref, tgt in view.deleted_inds:
            if (ref, tgt) in data.pair_entries:
                continue
            if not (0 <= ref < data.n_neurons and 0 <= tgt < data.n_neurons):
                continue
            if view.focused_neuron is not None \
                    and ref != view.focused_neuron and tgt != view.focused_neuron:
                continue
            ox, oy = self._pair_offset(view, view.deleted_inds, ref, tgt)
            self._add_arrow(view, ref, tgt, ox, oy, rgba, 1.0,
                            f'{ref}_{tgt}_deleted', z=1)

    # pxMode markers: fixed pixels, so icons stay round at any H/V zoom
    _NEURON_SIZE, _NEURON_SIZE_HL = 9, 13

    def _draw_neurons(self, data, view: NetworkView):
        """One scatter for every neuron, plus larger markers for the focused ones."""
        spots = self._plain_neuron_spots(data, view)
        spots += self._highlight_neuron_spots(data, view)

        scatter = pg.ScatterPlotItem(pxMode=True)
        scatter.addPoints(spots)
        scatter.sigClicked.connect(self._on_neuron_scatter_click)
        scatter.setZValue(5)
        self._plot_win.addItem(scatter)
        self._pg_items.append(scatter)

    def _plain_neuron_spots(self, data, view: NetworkView) -> list:
        """Spots for every neuron that is neither focused nor hidden; labels drawn here."""
        connected = {n for pair in data.pair_entries for n in pair}
        palette = self._area_palette()      # resolved once, not per neuron
        grays = self._grays()
        focused = view.focused_pair_neurons | (
            {view.focused_neuron} if view.focused_neuron is not None else set())

        spots = []
        for idx in range(data.n_neurons):
            if idx in focused:
                continue
            if (view.hidden_shanks and data.shank_ids is not None
                    and idx < len(data.shank_ids)
                    and int(data.shank_ids[idx]) in view.hidden_shanks):
                continue
            is_connected = idx in connected
            if self._net_hide and not is_connected:
                continue
            red, green, blue = self._neuron_rgb(data, idx, view.slot_of, grays, palette)
            x, y = float(view.x[idx]), float(view.y[idx])
            spots.append({
                'pos': (x, y), 'size': self._NEURON_SIZE,
                'symbol': self._symbol_for(self._neuron_type_of(data, idx)),
                'pen': None, 'data': idx,
                'brush': pg.mkBrush(red, green, blue, 200 if is_connected else 64),
            })
            if self._net_show_nid:
                label = pg.TextItem(str(idx), color=view.fg_muted, anchor=(0.0, 1.0))
                label.setPos(x + 4, y)
                label.setFont(QtGui.QFont('Arial', 6))
                self._plot_win.addItem(label)
                self._pg_items.append(label)
        return spots

    def _highlight_neuron_spots(self, data, view: NetworkView) -> list:
        """Larger outlined spots: orange for a focused neuron, orange/blue for a pair."""
        highlights = []
        if view.focused_neuron is not None:
            highlights.append((view.focused_neuron, (255, 111, 0)))
        if view.focused_pair is not None:
            highlights.append((view.focused_pair[0], (255, 111, 0)))
            highlights.append((view.focused_pair[1], (30, 136, 229)))

        pen = pg.mkPen('white' if view.dark else 'black', width=2)
        spots = []
        for idx, rgb in highlights:
            if not (0 <= idx < data.n_neurons):
                continue
            spots.append({
                'pos': (float(view.x[idx]), float(view.y[idx])),
                'size': self._NEURON_SIZE_HL,
                'symbol': self._symbol_for(self._neuron_type_of(data, idx)),
                'pen': pen, 'brush': pg.mkBrush(*rgb, 255), 'data': idx,
            })
        return spots

    @staticmethod
    def _neuron_type_of(data, idx: int):
        if data.neuron_type is None or idx >= len(data.neuron_type):
            return None
        return data.neuron_type[idx]

    def _draw_legend(self, data, enabled_cts):
        shown: set = set()
        for entries in data.pair_entries.values():
            for e in entries:
                ct = e['conn_type']
                if ct is not None and (enabled_cts is None or ct in enabled_cts):
                    shown.add(ct)
        _lbl = {('pyr','pyr'):'pyr→pyr', ('pyr','inter'):'pyr→int',
                ('inter','pyr'):'int→pyr', ('inter','inter'):'int→int'}
        lines = []
        for ct in sorted(shown, key=str):
            rgba = _ct_rgba(ct)
            hex_c = f'#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}'
            lines.append(f'<span style="color:{hex_c}">●</span> '
                         f'{_lbl.get(ct, str(ct))}')
        overlay = getattr(self, '_legend_overlay', None)
        if overlay is not None:
            if lines:
                overlay.setText('<br>'.join(lines))
                overlay.setVisible(True)
                overlay.adjustSize()
                overlay.raise_()
            else:
                overlay.setVisible(False)

    def _on_arrow_click(self, gid: str):
        self._plot_win.setFocus()   # grab keyboard so Up/Down step between edges
        parts = gid.split('_', 2)
        try:
            ref, tgt = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return
        key_str = parts[2] if len(parts) > 2 else None
        if key_str == 'deleted':
            return
        # 'cur'/'deleted' etc. are not real type keys → treat as no-switch.
        if key_str is not None and not key_str.startswith('sess_'):
            key_str = None
        self._navigate_to_pair((ref, tgt), key_str)

    def _on_neuron_scatter_click(self, scatter, points, ev):
        self._plot_win.setFocus()
        if points is None or len(points) == 0:
            return
        # overlapping neurons all land in `points`: cycle through them on repeated clicks
        ids = [p.data() for p in points if isinstance(p.data(), int)]
        if not ids:
            return
        nid = ids[(ids.index(self._focused_neuron) + 1) % len(ids)] \
            if self._focused_neuron in ids else ids[0]
        self._focused_neuron = nid
        self._focused_pair   = None
        self._focus_entry.setText(str(nid))
        self._focus_pair_entry.setText('')
        self._update_focus_info(nid)
        self.ui.refresh_lists()
        self.draw()

    def _navigate_to_pair(self, pair: tuple, key_str: str | None):
        ui = self.ui
        ref, tgt = pair
        if ui.session_any_mode:
            sessions   = self._net_any_sessions_cache or self._net_any_sessions()
            target_idx = self._net_any_idx
            if key_str is not None:
                for si, nk in enumerate(sessions):
                    ckey = ui.type_key_for_nd(nk)
                    if ckey is not None:
                        avail = ui.available_type_keys(nk)
                        if any(str(k) == key_str for k in avail):
                            target_idx = si
                            break
            if target_idx != self._net_any_idx:
                self._net_any_idx = target_idx
            pidx = ui.get_pair_index(pair)
            if pidx < len(ui.all_pairs_np):
                ui.set_current_pair(pidx)
                ui.root.pairs_view.pair_selection._select_pair_in_list(pair)
            self.draw()
            ui.root.mainview.request_render()
            return

        if key_str is not None:
            clicked_key = next(
                (k for k in ui.available_type_keys(ui.key.nd())
                 if str(k) == key_str), None)
            if clicked_key is not None and clicked_key != ui.key:
                root = ui.root
                root._ensure_loaded(
                    clicked_key.nd(), 'lowres',
                    lambda ck=clicked_key: root._switch_session(ck))

        idx = ui.get_pair_index(pair)
        if idx < len(ui.all_pairs_np):
            ui.set_current_pair(idx)
            ui.root.pairs_view.pair_selection._select_pair_in_list(pair)
        self.draw()
        ui.root.mainview.request_render()

    def neurons_on_channel(self, nid: int) -> list:
        """Every neuron sharing *nid*'s peak channel, ascending; empty without channels."""
        neurons = self.ui.neurons
        channels = getattr(neurons, 'peak_channels', None) if neurons is not None else None
        if channels is None or nid >= len(channels):
            return []
        return [i for i, ch in enumerate(channels) if int(ch) == int(channels[nid])]

    def focus_next_on_channel(self) -> bool:
        """Move the neuron focus to the next neuron on the same peak channel."""
        nid = self._focused_neuron
        if nid is None:
            return False
        peers = self.neurons_on_channel(nid)
        if len(peers) < 2:
            return False
        self._focus_entry.setText(str(peers[(peers.index(nid) + 1) % len(peers)]))
        self._on_neuron_focus()
        return True

    def _on_plot_nav_key(self, key) -> bool:
        """Up/Down step through the visible edges (as drawn), navigating each in turn.
        Position is derived from the current pair, so it survives the redraw.
        Enter cycles the focused neuron's channel-mates instead."""
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self.focus_next_on_channel()
        edges = self._nav_edges
        if not edges:
            return False
        ui  = self.ui
        cur = (tuple(ui.all_pairs_np[ui.current_pair_idx])
               if ui.current_pair_idx < len(ui.all_pairs_np) else None)
        at  = next((i for i, (r, t, _) in enumerate(edges) if (r, t) == cur), None)
        if at is None:
            nxt = 0 if key == Qt.Key.Key_Down else len(edges) - 1
        else:
            nxt = (at + (1 if key == Qt.Key.Key_Down else -1)) % len(edges)
        ref, tgt, key_str = edges[nxt]
        self._navigate_to_pair((ref, tgt), key_str)
        return True

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
        for g in self._net_group_filter_vars:
            self._net_group_filter_vars[g] = False
            if g in self._net_grp_cbs:
                self._net_grp_cbs[g].setChecked(False)
        self.draw()

    def _on_save_selections_to_group(self):
        ui    = self.ui
        pairs = ui.all_pairs_set
        if not pairs:
            QMessageBox.information(None, 'Save selections', 'No pairs visible to save.')
            return
        name, _ok = QInputDialog.getText(None, 'Save selections to group', f'Group name ({len(pairs)} pairs):')
        if not _ok:
            return
        if not name or not name.strip():
            return
        name = name.strip()
        sess = ui.current_session_str
        if name in ui.groups:
            if QMessageBox.question(None,
                    'Save selections',
                    f"Group '{name}' exists. Replace pairs for this session?",
                    parent=ui.root):
                return
            for _, r, t in list(ui.groups.forward(name)):
                if _ == sess:
                    ui.groups.discard_from_group(name, sess, (r, t))
        for pair in pairs:
            ui.groups.add_to_group(name, sess, pair)
        ui.groups.changed.emit()
        ui.refresh_lists()
        QMessageBox.information(None, 'Save selections',
                            f"Saved {len(pairs)} pairs to group '{name}'.")

    def _net_any_sessions(self) -> list:
        ui            = self.ui
        sessions      = ui.real_nd_keys()
        active_groups = {g for g, v in self._net_group_filter_vars.items() if v}
        if not active_groups:
            return sessions
        filtered = []
        for nk in sessions:
            ckey = ui.type_key_for_nd(nk)
            if ckey is None:
                continue
            sess  = str(ckey.session)
            ptr   = ui.cd.ptr.get(ckey)
            valid = ptr.pair_set
            for g in active_groups:
                if any((int(a), int(b)) in valid
                       for a, b in ui.groups.pairs_in_group(g, sess)):
                    filtered.append(nk)
                    break
        return filtered

    def follow_current_pair(self):
        """Redraw; in any-session mode first page the plot to the current pair's own
        session so its highlighted edge is on screen (mirrors edge-click navigation)."""
        ui = self.ui
        if ui.session_any_mode:
            handle = ui.current_pair   # (ckey, ref, tgt) | None
            if handle is not None:
                sess     = str(handle[0].session)
                sessions = self._net_any_sessions_cache or self._net_any_sessions()
                for i, nk in enumerate(sessions):
                    if str(getattr(nk, 'session', nk)) == sess:
                        if i != self._net_any_idx:
                            self._net_any_idx = i
                            self.refresh_group_buttons()
                        break
        self.draw()

    def _on_net_arrow_left(self):
        if not self.ui.session_any_mode:
            return
        if self._net_any_idx > 0:
            self._net_any_idx -= 1
            self.draw()
            self.refresh_group_buttons()

    def _on_net_arrow_right(self):
        if not self.ui.session_any_mode:
            return
        sessions = self._net_any_sessions_cache or self._net_any_sessions()
        if self._net_any_idx < len(sessions) - 1:
            self._net_any_idx += 1
            self.draw()
            self.refresh_group_buttons()

    def _update_nav_bar(self):
        any_mode = self.ui.session_any_mode
        if any_mode:
            sessions = self._net_any_sessions_cache or self._net_any_sessions()
            n   = len(sessions)
            idx = max(0, min(self._net_any_idx, n - 1)) if n else 0
            lbl = self.ui.session_label(sessions[idx]) if sessions else ''
            self._net_nav_label.setText(f'{idx + 1}/{n} · {lbl}')
            self._net_nav_left.setEnabled(idx > 0)
            self._net_nav_right.setEnabled(idx < n - 1)
        for w in (self._net_nav_label, self._net_nav_left, self._net_nav_right):
            w.setVisible(any_mode)

    def _highlighted_ct_labels(self) -> set:
        labels = set()
        for (a, b), enabled in getattr(self, '_net_ct_vars', {}).items():
            if enabled:
                labels.add(Key.format_conn_type((a, b)))
        return labels

    def refresh_group_buttons(self):
        ui = self.ui
        if not hasattr(self, '_net_grp_body_lay'):
            return
        while self._net_grp_body_lay.count():
            item = self._net_grp_body_lay.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                inner = w.layout()
                if isinstance(inner, FlowLayout):
                    inner.clear_widgets()
                w.setParent(None)
        self._net_grp_flow = None
        self._net_grp_flow_lay = None
        self._net_grp_items = []
        for cb in self._net_grp_cbs.values():
            cb.setParent(None)
        self._net_grp_cbs.clear()

        group_names = set(ui.groups.defined_groups)
        group_names |= {k for k in ui.groups if k.startswith(_SPECIAL_PREFIX)}
        regular = sorted(k for k in group_names if not k.startswith('__'))
        special = sorted(k for k in group_names if k.startswith(_SPECIAL_PREFIX))
        for g in set(self._net_group_filter_vars) - group_names:
            del self._net_group_filter_vars[g]

        if not regular and not special:
            lbl = QLabel('(no groups)')
            lbl.setStyleSheet(f'color: #888; font-size: {small_font_pt()}pt;')
            self._net_grp_body_lay.addWidget(lbl)
            return

        if ui.session_any_mode and self._net_any_sessions_cache:
            _idx = max(0, min(self._net_any_idx, len(self._net_any_sessions_cache) - 1))
            _nk  = self._net_any_sessions_cache[_idx]
            count_sess = str(getattr(_nk, 'session', _nk))
        else:
            count_sess = ui.current_session_str

        flow = getattr(self, '_net_grp_flow', None)
        if flow is None:
            flow = QWidget()
            flow_lay = FlowLayout(flow, spacing=4)
            self._net_grp_flow = flow
            self._net_grp_flow_lay = flow_lay
            self._net_grp_body_lay.addWidget(flow)
        else:
            flow_lay = self._net_grp_flow_lay
        for gname in regular:
            cb = self._make_group_button(gname, gname, count_sess)
            flow_lay.addWidget(cb)

        if special:
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            flow_lay.addWidget(sep)
            arrow_lbl = '▸' if self._net_special_collapsed else '▾'
            toggle_btn = QPushButton(f'{arrow_lbl} Special:')
            toggle_btn.setFlat(True)
            toggle_btn.setStyleSheet(f'color: #666; font-size: {small_font_pt()}pt; text-align: left;')
            def _toggle_special(_checked=False):   # clicked() passes checked; the rebuild relabels
                self._net_special_collapsed = not self._net_special_collapsed
                self.refresh_group_buttons()
            toggle_btn.clicked.connect(_toggle_special)
            flow_lay.addWidget(toggle_btn)
            if not self._net_special_collapsed:
                for gname in special:
                    display = ui.groups.get_group_metadata(gname).display_name
                    cb = self._make_group_button(gname, display, count_sess)
                    flow_lay.addWidget(cb)
        if flow is not None:
            flow.updateGeometry()

    def rewrap_group_buttons(self):
        # In Qt, layout handles wrapping; no-op.
        pass

    @staticmethod
    def _shanks_from_data(data) -> list:
        # Same source that draws the network: probe columns (pg_info df) are the full set of
        # shanks rendered (12-16); neuron shank_ids are the subset fallback.
        if data.pg_info is not None:
            try:
                return sorted(int(s) for s in data.pg_info.to_dataframe()['shank_id'].unique())
            except Exception:
                pass
        if data.shank_ids is not None and len(data.shank_ids):
            return sorted(set(int(s) for s in data.shank_ids))
        return []

    def refresh_shank_buttons(self):
        # Best-effort pre-render pass (panel build before first draw); the authoritative
        # population runs from _render on the built data.
        ui = self.ui
        shank_ids = getattr(ui.neurons, 'shank_ids', None) if ui.neurons is not None else None
        if shank_ids is not None and len(shank_ids):
            shanks = sorted(set(int(s) for s in shank_ids))
        else:
            pg_info = self._probegroups().get(ui.key.nd())
            shanks = (sorted(int(s) for s in pg_info.to_dataframe()['shank_id'].unique())
                      if pg_info is not None else [])
        self._sync_shank_chips(shanks)

    def _sync_shank_chips(self, shanks: list):
        flow_lay = getattr(self, '_net_probe_flow_lay', None)
        if flow_lay is None:
            return
        # Rebuild only when the shank set changes; toggles just flip vars (avoid churn/loop).
        if set(shanks) == set(self._net_shank_cbs):
            return
        while flow_lay.count():
            item = flow_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for cb in self._net_shank_cbs.values():
            cb.setParent(None)
        self._net_shank_cbs.clear()

        if not shanks:
            lbl = QLabel('No shank data')
            lbl.setStyleSheet(f'color: gray; {_UI_FS()}')
            flow_lay.addWidget(lbl)
            self._net_shank_vars = {}
            return

        new_vars = {}
        for shank in shanks:
            val = self._net_shank_vars.get(shank, True)
            new_vars[shank] = val
            cb = chip_button(f'S{shank}', checked=val)
            cb.toggled.connect(lambda v, s=shank: (
                self._net_shank_vars.__setitem__(s, v), self.draw()))
            flow_lay.addWidget(cb)
            self._net_shank_cbs[shank] = cb
        self._net_shank_vars = new_vars

    def _make_group_button(self, gname: str, display: str, count_sess: str) -> QCheckBox:
        ui = self.ui
        if gname not in self._net_group_filter_vars:
            self._net_group_filter_vars[gname] = False
        if self._net_grp_counts:
            pairs_sess = ui.groups.pairs_in_group(gname, count_sess)
            _admitted = set(self._highlighted_ct_labels()) or None
            _by_ct = ui._pairs_by_conn_type(count_sess, pairs_sess)
            n_hl = len([p for lbl, ps in _by_ct.items()
                        if not _admitted or lbl in _admitted for p in ps])
            n_all = len({(r, t) for _, r, t in ui.groups.forward(gname)})
            label = f'{display} ({n_hl}/{len(pairs_sess)}/{n_all})'
        else:
            label = display
        cb = chip_button(label, checked=self._net_group_filter_vars[gname])
        cb.toggled.connect(lambda v, g=gname: (
            self._net_group_filter_vars.__setitem__(g, v),
            self._on_group_toggle(g)))
        self._net_grp_cbs[gname] = cb
        return cb

    def _on_neuron_focus(self):
        val = self._focus_entry.text().strip()
        if not val:
            self._on_neuron_focus_clear()
            return
        try:
            nid = int(val)
        except ValueError:
            QMessageBox.critical(None, 'Neuron focus', f'Invalid id: {val!r}')
            return
        if self.ui.neurons is not None:
            if nid < 0 or nid >= self.ui.neurons.n_neurons:
                QMessageBox.critical(None, 'Neuron focus',
                    f'Out of range [0, {self.ui.neurons.n_neurons-1}]')
                return
        self._focused_neuron = nid
        self._focused_pair   = None
        self._focus_pair_entry.setText('')
        self._focus_pair_info_label.setText('')
        self._update_focus_info(nid)
        self.ui.refresh_lists()
        self.draw()

    def _update_focus_info(self, nid: int):
        ui      = self.ui
        cur_out = sum(1 for r, t in map(tuple, ui.all_pairs_np) if r == nid)
        cur_in  = sum(1 for r, t in map(tuple, ui.all_pairs_np) if t == nid)
        tot_out = tot_in = 0
        for tk_ in ui.available_type_keys(ui.key.nd()):
            pt = ui.cd.ptr.get(tk_)
            if pt is None or pt.inds is None:
                continue
            tot_out += sum(1 for r, t in pt.pair_set if r == nid)
            tot_in  += sum(1 for r, t in pt.pair_set if t == nid)
        self._focus_info_label.setText(
            f'{ui.key.type_label()}: in={cur_in} out={cur_out}'
            f'  |  all: in={tot_in} out={tot_out}')

    def _on_neuron_focus_clear(self):
        self._focused_neuron = None
        self._focus_entry.setText('')
        self._focus_info_label.setText('')
        self.ui.refresh_lists()
        self.draw()

    def _on_pair_focus(self):
        ui  = self.ui
        val = self._focus_pair_entry.text().strip()
        if not val:
            self._on_pair_focus_clear()
            return
        try:
            parts    = val.replace(' ', '').split(',')
            ref, tgt = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            QMessageBox.critical(None, 'Pair focus',
                                 f'Invalid format: {val!r}\nUse ref,tgt')
            return
        if ui.neurons is not None:
            n = ui.neurons.n_neurons
            if not (0 <= ref < n and 0 <= tgt < n):
                QMessageBox.critical(None, 'Pair focus',
                                     f'Out of range [0, {n-1}]')
                return
        pair        = (ref, tgt)
        pair_exists = any(
            pair in ui.cd.ptr.get(tk_).pair_set
            for tk_ in ui.available_type_keys(ui.key.nd())
            if ui.cd.ptr.get(tk_) is not None
            and ui.cd.ptr.get(tk_).inds is not None)
        if not pair_exists:
            ui.root._show_transient_banner(
                f'Pair ({ref},{tgt}) not significant — showing position')
        self._focused_pair   = pair
        self._focused_neuron = None
        self._focus_entry.setText('')
        self._focus_info_label.setText('')
        self._update_pair_focus_info(pair, pair_exists)
        ui.refresh_lists()
        ui.root.pairs_view.pair_selection._select_pair_in_list(pair)
        self.draw()
        ui.root.mainview.request_render()

    def _update_pair_focus_info(self, pair: tuple, exists: bool):
        ui       = self.ui
        ref, tgt = pair
        if ui.neurons is not None:
            nt       = ui.neurons.neuron_type
            ref_type = nt[ref] if nt is not None and ref < len(nt) else '?'
            tgt_type = nt[tgt] if nt is not None and tgt < len(nt) else '?'
            in_avail = pair in ui.all_pairs_set
            status   = ('sig' if exists
                        else ('admitted' if in_avail else 'not sig'))
            self._focus_pair_info_label.setText(
                f'{ref}({ref_type})→{tgt}({tgt_type}) [{status}]')
        else:
            self._focus_pair_info_label.setText(f'{ref}→{tgt}')
        in_avail = pair in ui.all_pairs_set
        self._add_pair_btn.setEnabled(not in_avail)

    def _on_pair_focus_clear(self):
        self._focused_pair = None
        self._focus_pair_entry.setText('')
        self._focus_pair_info_label.setText('')
        self._add_pair_btn.setEnabled(False)
        self.ui.refresh_lists()
        self.draw()
        self.ui.root.mainview.request_render()

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
        peak_ch = np.asarray(neurons.peak_channels, dtype=int)
        # A dataset shipping real per-unit coordinates needs no probe layout to
        # place its neurons; the probegroup only draws the empty-channel backdrop.
        xy = (neurons.metadata or {}).get('positions')
        if xy is not None:
            return np.asarray(xy, dtype=float), peak_ch
        pg_info = self._probegroups().get(nd_key)
        if pg_info is None:
            return None
        pg_df   = pg_info.to_dataframe().set_index('channel_id')
        x = pg_df['x'].reindex(peak_ch).fillna(0.0).to_numpy(dtype=float)
        y = pg_df['y'].reindex(peak_ch).fillna(0.0).to_numpy(dtype=float)
        return np.stack([x, y], axis=1), peak_ch

    def _shank_label(self, idx: int) -> str:
        """The neuron's shank number, falling back to its index when unknown."""
        shank_ids = getattr(self.ui.neurons, 'shank_ids', None)
        if shank_ids is None or idx >= len(shank_ids):
            return str(idx)
        return str(int(shank_ids[idx]))

    def _pair_label(self, inds) -> str:
        return f'{self._shank_label(inds[0])}→{self._shank_label(inds[1])}'

    def build_should_gray(self, any_mode: bool) -> callable:
        fn = self._focused_neuron
        fp = self._focused_pair
        hide_shank   = self._net_hide_same_shank
        hide_channel = self._net_hide_same_channel

        def _meta(neurons):
            return (getattr(neurons, 'peak_channels', None) if neurons else None,
                    getattr(neurons, 'shank_ids',     None) if neurons else None)

        cur_peak, cur_shank = _meta(self.ui.neurons)
        cache: dict = {}   # nd_key → (peak_channels, shank_ids), for any_mode

        def _gray(inds) -> bool:
            if any_mode:
                nd_key = inds[0].nd()
                meta = cache.get(nd_key)
                if meta is None:
                    cache[nd_key] = meta = _meta(self.ui.cd.nd.neurons_for(nd_key))
                peak_channels, shank_ids = meta
                ref_i, tgt_i = int(inds[1]), int(inds[2])
            else:
                peak_channels, shank_ids = cur_peak, cur_shank
                ref_i, tgt_i = int(inds[0]), int(inds[1])
            pair  = (ref_i, tgt_i)
            if fn is not None and ref_i != fn and tgt_i != fn:
                return True
            if fp is not None and pair != fp:
                return True
            if hide_shank and shank_ids is not None:
                if int(shank_ids[ref_i]) == int(shank_ids[tgt_i]):
                    return True
            if hide_channel and peak_channels is not None:
                if int(peak_channels[ref_i]) == int(peak_channels[tgt_i]):
                    return True
            return False

        return _gray
