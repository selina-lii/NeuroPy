"""Time slider UI: epoch timeline, zoom, custom CCG via signals."""
from __future__ import annotations

import datetime
import json
import os
import threading
import traceback
import dataclasses
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Literal

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, Signal, QObject, QPointF, QPoint, QRectF, QTimer
from pyqtgraph.Qt.QtWidgets import (
    QAbstractItemView, QDialog, QListWidget, QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QCheckBox, QComboBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QScrollArea,
    QGraphicsRectItem, QToolButton,
)
from pyqtgraph.Qt.QtGui import QFont
from pyqtgraph.Qt.QtGui import QPainter, QPen, QColor, QBrush
from neuropy.analyses.ms_connectivity import CCGData, CCGDataset, CCGSourceConfig, CCGBatchRequest
from neuropy.analyses.neurons_dataset import Key

_FULL_SEG = 'all'   # reserved label for the permanent whole-session segment (dim0[0])
from neuropy.analyses.utils import JsonSavable, Savable
from neuropy.core.intervals import IntervalOp as _SetOp
from neuropy.ui.ui_common import BackgroundTaskRunner
from neuropy.ui.utils import (
    AddableDropdown, chip_button, CollapsibleSection, ListPickerButton, MetricInput,
    ResultsDialog, small_font_pt, regular_font_pt)
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState
    from neuropy.ui.ccg_ui import CCGReviewUI

_TS_COLORS = [
    '#BBDEFB', '#C8E6C9', '#FFF9C4', '#FFE0B2', '#E1BEE7',
    '#F8BBD0', '#D7CCC8', '#B2EBF2', '#DCEDC8', '#F0F4C3',
]
_TS_NONE_COLOR = '#E0E0E0'

_ALL_SEGS = "all"  # whole-session view == permanent dim0[0]='all' (must match ccg_ui._ALL_SEGS)

class EpochPlotWidget(pg.PlotWidget):
    """Epoch timeline and draggable timing cursors."""

    handle_moved = Signal(float, float)

    _CURSOR_COLOR = '#1565C0'
    _DRAG_COLOR   = '#C62828'
    _BAR_Y0       = 0.0    # epoch bars occupy y=0..1.0; no axis zone
    _DOT_Y        = 0.0    # cursor dots at bar bottom

    def __init__(self, parent=None):
        from neuropy.ui.ui_common import qt_dark_mode
        _axis = pg.AxisItem(orientation='bottom')
        _axis.setStyle(tickLength=10, tickTextOffset=1)
        _axis.tickStrings = lambda values, *_: [
            f"{int(max(0.0,float(v))//3600):02d}:"
            f"{int((max(0.0,float(v))%3600)//60):02d}:"
            f"{int(max(0.0,float(v))%60):02d}"
            for v in values]
        _app = QtWidgets.QApplication.instance()
        _axis.setStyle(tickFont=_app.font() if _app is not None else QFont())
        super().__init__(parent, axisItems={'bottom': _axis})
        self.getPlotItem().layout.setContentsMargins(0, 1, 0, 1)
        bg = '#2b2b2b' if qt_dark_mode() else None
        self.setBackground(bg)
        if bg is None:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.viewport().setAutoFillBackground(False)
        self.hideAxis('left')
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)
        self.setYRange(0, 1, padding=0)
        self.setFixedHeight(42)

        self._t_min:    float = 0.0
        self._t_max:    float = 1.0
        self._start_t:  float = 0.0
        self._end_t:    float = 0.0
        self._epoch_rects: list = []
        self._snap_times:  list[float] = []
        self._snap_enabled: bool = True
        self._has_start:    bool = False
        self._has_end:      bool = False

        _dot_brush = pg.mkBrush(QColor(self._CURSOR_COLOR))
        _dot_pen   = pg.mkPen(QColor('#ffffff'), width=1)
        self._start_dot = pg.ScatterPlotItem(
            symbol='o', size=20, brush=_dot_brush, pen=_dot_pen)
        self._end_dot = pg.ScatterPlotItem(
            symbol='o', size=20, brush=_dot_brush, pen=_dot_pen)
        for dot in (self._start_dot, self._end_dot):
            dot.setZValue(10)
            dot.setVisible(False)
            self.addItem(dot)

        _box_pen = pg.mkPen(QColor(self._CURSOR_COLOR), width=1, style=Qt.PenStyle.DashLine)
        self._box_top    = pg.PlotDataItem(pen=_box_pen)
        self._box_bottom = pg.PlotDataItem(pen=_box_pen)
        self._box_left   = pg.PlotDataItem(pen=_box_pen)
        self._box_right  = pg.PlotDataItem(pen=_box_pen)
        for item in (self._box_top, self._box_bottom, self._box_left, self._box_right):
            item.setZValue(5)
            item.setVisible(False)
            self.addItem(item)

        self._drag_start_t:  float | None = None
        self._grabbed_dot = None   # _start_dot | _end_dot | None
        self._drag_rect = pg.LinearRegionItem(
            movable=False,
            brush=pg.mkBrush(QColor(self._DRAG_COLOR)),
            pen=pg.mkPen(QColor(self._DRAG_COLOR), width=1))
        self._drag_rect.setZValue(20)
        self._drag_rect.setVisible(False)
        self.addItem(self._drag_rect)

        vb = self.getViewBox()
        _orig_press   = vb.mousePressEvent
        _orig_move    = vb.mouseMoveEvent
        _orig_release = vb.mouseReleaseEvent

        def _dot_px(dot, t):
            scene_pt = vb.mapToScene(QPointF(t, self._DOT_Y))
            return float(self.mapFromScene(scene_pt).x())

        def _vb_press(ev):
            if ev.button() != Qt.MouseButton.LeftButton:
                _orig_press(ev); return
            px = ev.pos().x()
            grabbed = None
            if self._has_start:
                if abs(px - _dot_px(self._start_dot, self._start_t)) < 10:
                    grabbed = 'start'
            if grabbed is None and self._has_end:
                if abs(px - _dot_px(self._end_dot, self._end_t)) < 10:
                    grabbed = 'end'
            if grabbed:
                self._grabbed_dot = grabbed
            else:
                self._drag_start_t = vb.mapToView(ev.pos()).x()
                self._drag_rect.setRegion([self._drag_start_t, self._drag_start_t])
                self._drag_rect.setVisible(True)
            ev.accept()

        def _vb_move(ev):
            if self._grabbed_dot:
                t = self._clamp_t(vb.mapToView(ev.pos()).x())
                if self._grabbed_dot == 'start':
                    self._start_t = t
                    self._start_dot.setData([t], [self._DOT_Y])
                else:
                    self._end_t = t
                    self._end_dot.setData([t], [self._DOT_Y])
                self._on_cursor_moved()
                ev.accept()
            elif self._drag_start_t is not None:
                t = vb.mapToView(ev.pos()).x()
                lo, hi = sorted([self._drag_start_t, t])
                self._drag_rect.setRegion([lo, hi])
                ev.accept()
            else:
                _orig_move(ev)

        def _vb_release(ev):
            if ev.button() != Qt.MouseButton.LeftButton:
                _orig_release(ev); return
            if self._grabbed_dot:
                self._grabbed_dot = None
                ev.accept()
            elif self._drag_start_t is not None:
                t = vb.mapToView(ev.pos()).x()
                lo, hi = sorted([self._snap_near(self._clamp_t(self._drag_start_t)),
                                  self._snap_near(self._clamp_t(t))])
                self._drag_rect.setVisible(False)
                self._drag_start_t = None
                if hi - lo > (self._t_max - self._t_min) * 0.01:
                    self.setXRange(lo, hi, padding=0.01)
                else:
                    self._place_cursor_at(lo)
                ev.accept()
            else:
                _orig_release(ev)

        vb.mousePressEvent   = _vb_press
        vb.mouseMoveEvent    = _vb_move
        vb.mouseReleaseEvent = _vb_release

    def update_epochs(self, bounds: list[tuple], label_colors: dict,
                      t_min: float, t_max: float):
        vb = self.getViewBox()
        for item in self._epoch_rects:
            vb.removeItem(item)
        self._epoch_rects.clear()

        self._t_min = t_min
        self._t_max = max(t_max, t_min + 1.0)
        self.setXRange(t_min, self._t_max, padding=0.01)

        snap = {t_min, t_max}
        bar_h = 1.0 - self._BAR_Y0
        for t0, t1, lbl in bounds:
            color = label_colors.get(lbl, _TS_NONE_COLOR)
            rect = QGraphicsRectItem(t0, self._BAR_Y0, t1 - t0, bar_h)
            rect.setBrush(QBrush(QColor(color)))
            rect.setPen(QPen(QColor(color), 0))
            rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            rect.setZValue(-1)
            vb.addItem(rect)
            self._epoch_rects.append(rect)
            snap.add(t0)
            snap.add(t1)
        self._snap_times = sorted(snap)
        self._update_box()

    def clear_selection(self):
        self._has_start = False
        self._has_end = False
        self._start_dot.setVisible(False)
        self._end_dot.setVisible(False)
        self._set_box_visible(False)

    def set_selection(self, t_start: float, t_end: float, *, show: bool = True):
        t_start = max(self._t_min, min(self._t_max, float(t_start)))
        t_end   = max(self._t_min, min(self._t_max, float(t_end)))
        if t_start > t_end:
            t_start, t_end = t_end, t_start
        self._start_t, self._end_t = t_start, t_end
        if show:
            self._has_start = True
            self._has_end   = True
            self._start_dot.setData([t_start], [self._DOT_Y])
            self._end_dot.setData(  [t_end],   [self._DOT_Y])
            self._start_dot.setVisible(True)
            self._end_dot.setVisible(True)
        self._update_box()

    def get_selection(self) -> tuple[float, float]:
        if not self._has_start and not self._has_end:
            return 0.0, 0.0
        t0 = self._start_t if self._has_start else self._t_min
        t1 = self._end_t   if self._has_end   else self._t_max
        if t0 > t1:
            t0, t1 = t1, t0
        return t0, t1

    def has_full_selection(self) -> bool:
        return self._has_start and self._has_end

    def reset_zoom(self):
        self.setXRange(self._t_min, self._t_max, padding=0.01)

    def _clamp_t(self, t: float) -> float:
        return max(self._t_min, min(self._t_max, float(t)))

    def _snap_threshold(self) -> float:
        return max((self._t_max - self._t_min) * 0.05, 1.0)

    def _snap_near(self, v: float) -> float:
        if not self._snap_enabled or not self._snap_times:
            return v
        thresh = self._snap_threshold()
        best, best_d = None, thresh + 1.0
        for t in self._snap_times:
            d = abs(t - v)
            if d <= thresh and d < best_d:
                best_d, best = d, t
        return best if best is not None else v

    def _set_box_visible(self, visible: bool):
        for item in (self._box_top, self._box_bottom, self._box_left, self._box_right):
            item.setVisible(visible)

    def _update_box(self):
        if self._has_start and self._has_end:
            t0, t1 = self.get_selection()
            y0, y1 = self._BAR_Y0, 1.0
            self._box_top.setData(   [t0, t1], [y1, y1])
            self._box_bottom.setData([t0, t1], [y0, y0])
            self._box_left.setData(  [t0, t0], [y0, y1])
            self._box_right.setData( [t1, t1], [y0, y1])
            self._set_box_visible(True)
        else:
            self._set_box_visible(False)

    def _on_cursor_moved(self):
        t0, t1 = self.get_selection()
        self._update_box()
        self.handle_moved.emit(t0, t1)

    def _snap_start(self):
        if not self._has_start:
            return
        v = self._snap_near(self._start_t)
        if self._has_end:
            v = min(v, self._end_t - 1.0)
        self._start_t = self._clamp_t(v)
        self._start_dot.setData([self._start_t], [self._DOT_Y])
        self._on_cursor_moved()

    def _snap_end(self):
        if not self._has_end:
            return
        v = self._snap_near(self._end_t)
        if self._has_start:
            v = max(v, self._start_t + 1.0)
        self._end_t = self._clamp_t(v)
        self._end_dot.setData([self._end_t], [self._DOT_Y])
        self._on_cursor_moved()

    def _place_cursor_at(self, t: float):
        t = self._snap_near(self._clamp_t(t))
        if not self._has_start:
            self._start_t = t
            self._start_dot.setData([t], [self._DOT_Y])
            self._start_dot.setVisible(True)
            self._has_start = True
        elif not self._has_end:
            if t <= self._start_t:
                self._start_t = t
                self._start_dot.setData([t], [self._DOT_Y])
            else:
                self._end_t = t
                self._end_dot.setData([t], [self._DOT_Y])
                self._end_dot.setVisible(True)
                self._has_end = True
        else:
            self._end_dot.setVisible(False)
            self._has_end = False
            self._start_t = t
            self._start_dot.setData([t], [self._DOT_Y])
            self._start_dot.setVisible(True)
            self._has_start = True
        self._on_cursor_moved()


class TimeSliderPanel(QWidget):
    """Time slider panel; custom CCG work is emitted to the parent."""

    queue_ccg_requested = Signal(object)   # CCGSourceConfig
    save_requested        = Signal()
    load_requested        = Signal()

    def __init__(self, nav: 'AppState', cd: 'CCGDataset', parent=None):
        super().__init__(parent)
        self.nav = nav
        self.cd  = cd

        # Epoch state
        self._epoch_bounds:    list = []
        self._total_sec:       float = 0.0
        self._all_theme_bounds: dict = {}    # theme_name → [(s,e,label)]
        self._current_theme:   str  = 'segments'
        self._label_colors:    dict | None = None
        self._per_theme_label_state: dict = {}  # theme → {label: bool}
        self._legend_toggles:  dict = {}     # label → bool (current theme)
        self._batch_counts:    dict = {}     # batch_id → tasks remaining
        self._batch_totals:    dict = {}     # batch_id → total tasks (lo + hi)
        self._batch_names:     dict = {}
        self._batch_meta:      dict = {}     # batch_id → {spec_name, skipped, rows}
        self._batch_next_id:   int = 1

        self._build()
        self._connect_nav()
        self._refresh_theme_ui(self.nav.cd.nd.get_themes(self.nav.key))

    def reload_themes(self):
        """Theme combo and bounds only (no timeline reset)."""
        self._discover_themes(self.nav.cd.nd.get_themes(self.nav.key))

    def _refresh_theme_ui(self, themes: dict):
        """Refresh combo, bounds, timeline, and legend."""
        self._label_colors = None
        self._discover_themes(themes)
        self._init_times()
        self._update_legend()

    @staticmethod
    def _fixed_line_edit(text: str, width: int) -> QLineEdit:
        le = QLineEdit(text); le.setFixedWidth(width)
        return le

    def _build(self):
        root = QVBoxLayout(self)
        title_lbl = QLabel("Time Slider - Behavioral Epochs")
        title_lbl.setStyleSheet(f"font-weight: bold; font-size: {regular_font_pt()}pt;")
        root.addWidget(title_lbl)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Theme:"))
        self._theme_combo = AddableDropdown('theme', self.add_theme)
        self._theme_combo.set_items(['segments'])
        self._theme_combo.setFixedWidth(140)
        self._theme_combo.currentTextChanged.connect(self._on_theme_change)
        row1.addWidget(self._theme_combo)
        self._theme_info_lbl = QLabel("")
        self._theme_info_lbl.setStyleSheet(f"color:#888; font-size:{small_font_pt()}pt;")
        row1.addWidget(self._theme_info_lbl)

        self._filter_checks: dict[str, bool] = {}   # per-theme "include in filter" (cross-theme AND; see FUTURE_STEPS)
        self._filter_check = chip_button("Include in filter", checked=False)
        self._filter_check.toggled.connect(lambda on: self._filter_check.setText(
            "✓ Include in filter" if on else "Include in filter"))
        self._filter_check.toggled.connect(self._on_filter_toggle)
        row1.addWidget(self._filter_check)
        row1.addSpacing(12)
        for text, width, slot in (("All", 40, self._on_label_reset), ("None", 40, self._on_label_none)):
            btn = QPushButton(text); btn.setFixedWidth(width); btn.clicked.connect(slot)
            row1.addWidget(btn)
        row1.addStretch()

        for icon, signal in (("💾", self.save_requested), ("📂", self.load_requested)):
            tb = QToolButton(); tb.setText(icon); tb.clicked.connect(signal)
            row1.addWidget(tb)
        sep_tb = QFrame(); sep_tb.setFrameShape(QFrame.VLine); sep_tb.setStyleSheet('color: #ccc;')
        row1.addWidget(sep_tb)
        self._snap_check = QCheckBox("Snap")
        self._snap_check.setChecked(True); self._snap_check.toggled.connect(self._on_snap_toggle)
        row1.addWidget(self._snap_check)
        self._reset_zoom_btn = QPushButton("Reset")
        self._reset_zoom_btn.setFixedWidth(50); self._reset_zoom_btn.clicked.connect(self._on_reset_zoom)
        row1.addWidget(self._reset_zoom_btn)
        row1_widget = QWidget(); row1_widget.setLayout(row1)
        root.addWidget(row1_widget)

        self._legend_widget = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.addStretch()
        root.addWidget(self._legend_widget)

        self._any_mode_lbl = QLabel(
            "All-sessions view: no single behavioral timeline — "
            "type Start/End below to run custom CCG across the selected sessions.")
        self._any_mode_lbl.setWordWrap(True)
        self._any_mode_lbl.setStyleSheet(f'color:#666; font-size:{small_font_pt()}pt; padding:4px;')
        self._any_mode_lbl.setVisible(False)
        root.addWidget(self._any_mode_lbl)

        self._main_plot = EpochPlotWidget()
        self._main_plot.handle_moved.connect(self._on_main_handle_moved)
        root.addWidget(self._main_plot)
        self._on_snap_toggle(self._snap_check.isChecked())

        self._timing_section = CollapsibleSection("CCG time range", expanded=True)
        root.addWidget(self._timing_section)
        timing_row = QHBoxLayout()
        self._timing_section.body_layout.addLayout(timing_row)
        for lbl, which, default in (("Start:", 'start', "00:00:00"), ("End:", 'end', "end")):
            timing_row.addWidget(QLabel(lbl))
            entry = self._fixed_line_edit(default, 72)
            entry.editingFinished.connect(lambda w=which: self._validate_timing_entry(w))
            timing_row.addWidget(entry)
            setattr(self, f'_{which}_entry', entry)
        set_btn = QPushButton("Set"); set_btn.clicked.connect(self._on_set)
        timing_row.addWidget(set_btn)

        self._ccg_extra_widget = QWidget()
        extra_lay = QHBoxLayout(self._ccg_extra_widget)
        clr_btn = QPushButton("Clear"); clr_btn.clicked.connect(self._on_clear)
        extra_lay.addWidget(clr_btn)
        _sessions = [str(k.session) for k in self.nav.real_nd_keys()]
        self._sessions_picker = ListPickerButton("Sessions", items=_sessions, plural="sessions")
        self._sessions_picker.set_selected([str(self.nav.key.session)])
        self._sessions_picker.setFixedWidth(120)
        extra_lay.addWidget(self._sessions_picker)
        extra_lay.addWidget(QLabel("Name:"))
        self._name_entry = self._fixed_line_edit("", 100)
        self._name_is_auto = True   # False once the user types their own name
        self._name_entry.textEdited.connect(lambda _t: setattr(self, '_name_is_auto', False))
        extra_lay.addWidget(self._name_entry)
        extra_lay.addWidget(QLabel("Splits:"))
        self._splits_spin = QSpinBox()
        self._splits_spin.setRange(1, 99); self._splits_spin.setValue(1); self._splits_spin.setFixedWidth(45)
        extra_lay.addWidget(self._splits_spin)
        self._overlap_metric = MetricInput(
            "Overlap:", ('%', 'hr', 'min', 's'), default="0",
            suggestions=(0, 10, 25, 50), input_width=45, unit_width=60)
        extra_lay.addWidget(self._overlap_metric)
        self._equal_effective_check = QCheckBox("Equal duration")
        self._equal_effective_check.setToolTip(
            "Splits share equal effective (filtered) time; real-time edges may differ.")
        extra_lay.addWidget(self._equal_effective_check)
        timing_row.addWidget(self._ccg_extra_widget)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color:#555; font-size:{small_font_pt()}pt;")
        timing_row.addWidget(self._status_lbl)
        timing_row.addStretch()

        for lyt in (row1, self._legend_layout, timing_row, extra_lay):
            lyt.setContentsMargins(0, 0, 0, 0)
            lyt.setSpacing(4)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        for w in (title_lbl, row1_widget, self._legend_widget, self._any_mode_lbl,
                  self._timing_section):
            w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root.addStretch()

    def _connect_nav(self):
        nav = self.nav
        nav.themes_changed.connect(self._on_themes_changed)
        nav.session_mode_changed.connect(self._on_session_mode_changed)

    def _on_themes_changed(self, themes: dict):
        self._refresh_theme_ui(themes)

    def _on_session_mode_changed(self, any_mode: bool):
        self._any_mode_lbl.setVisible(any_mode)
        self._main_plot.setVisible(not any_mode)
        self._legend_widget.setVisible(True)
        self._timing_section.setEnabled(True)
        if any_mode:
            self._sessions_picker.set_selected(
                [str(k.session) for k in self.nav.real_nd_keys()])
        else:
            self._sessions_picker.set_selected([str(self.nav.key.session)])
        themes = (self.nav.cd.nd.get_themes_any() if any_mode
                  else self.nav.cd.nd.get_themes(self.nav.key))
        self._refresh_theme_ui(themes)

    def _discover_themes(self, themes: dict):
        bounds_by_theme: dict = {}
        for attr, obj in themes.items():
            labs = [str(x).strip() for x in obj.labels]
            bounds = [(float(s), float(e), lb)
                      for s, e, lb in zip(obj.starts, obj.stops, labs)]
            unique = {lb for lb in labs if lb}
            if len(unique) <= 1:
                bounds = [(s, e, attr) for s, e, _ in bounds]
            bounds_by_theme[attr] = bounds
        self._all_theme_bounds = bounds_by_theme
        theme_names = ['segments'] + sorted(themes)
        cur = self._theme_combo.currentText()
        default = cur if cur in theme_names else (theme_names[1] if len(theme_names) > 1 else 'segments')
        self._theme_combo.blockSignals(True)
        self._theme_combo.set_items(theme_names)
        self._theme_combo.setCurrentText(default)
        self._theme_combo.blockSignals(False)
        self._current_theme = default
        n = len(themes)
        self._theme_info_lbl.setText(f"{n} theme{'s' if n != 1 else ''}")

    def _init_times(self):
        theme = self._current_theme
        if theme != 'segments' and theme in self._all_theme_bounds:
            self._epoch_bounds = list(self._all_theme_bounds[theme])
            self._total_sec = max((b[1] for b in self._epoch_bounds), default=1.0)
        else:  # 'segments' = no label filter: empty legend, timeline spans the whole session
            self._epoch_bounds = []
            _, t_stop = self.nav.cd.nd.session_bounds(self.nav.key)
            self._total_sec = float(t_stop) or 1.0

        # Initialise overlap from source config if available
        source = getattr(self.cd, 'source', None)
        if isinstance(source, CCGSourceConfig):
            self._overlap_metric.set_value(source.overlap_sec, 's')

        self._filter_check.blockSignals(True)
        self._filter_check.setChecked(self._filter_checks.get(theme, False))
        self._filter_check.blockSignals(False)
        self._update_legend()
        self._reset_handles()

    def _reset_handles(self):
        self._main_plot.clear_selection()
        self._start_entry.setText("00:00:00")
        self._end_entry.setText("end")

    def _on_snap_toggle(self, checked: bool):
        self._main_plot._snap_enabled = checked

    def _on_reset_zoom(self):
        self._main_plot.reset_zoom()

    def _update_legend(self):
        lyt = self._legend_layout
        # Clear existing chips (leave stretch)
        while lyt.count() > 1:
            item = lyt.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        cmap = self._label_color_map()
        saved = self._per_theme_label_state.get(self._current_theme, {})
        self._legend_toggles = {}

        for lbl, color in cmap.items():
            active = saved.get(lbl, True)
            self._legend_toggles[lbl] = active
            self._add_legend_chip(lbl, color, active)

        none_active = saved.get('NONE', True)
        self._legend_toggles['NONE'] = none_active
        self._add_legend_chip('NONE', _TS_NONE_COLOR, none_active, none_style=True)

        self._sync_name_to_labels()
        self._redraw_main()

    def _add_legend_chip(self, label: str, color: str, active: bool, *,
                         none_style: bool = False):
        chip = chip_button(label, checked=active)
        fs = small_font_pt()
        if none_style:
            ss = (f"QPushButton {{ border: 1px solid #888; border-radius: 2px; "
                  f"padding: 1px 6px; font-size: {fs}pt; background: {color}; "
                  f"color: #444; }}"
                  f"QPushButton:checked {{ font-weight: bold; }}"
                  f"QPushButton:!checked {{ color: #aaa; background: #f0f0f0; }}")
        else:
            ss = (f"QPushButton {{ border: 1px solid #888; border-radius: 2px; "
                  f"padding: 1px 6px; font-size: {fs}pt; background: {color}; }}"
                  f"QPushButton:checked {{ font-weight: bold; }}"
                  f"QPushButton:!checked {{ color: #888; background: #f0f0f0; }}")
        chip.setStyleSheet(ss)
        chip.toggled.connect(lambda on, lb=label: self._on_legend_toggle(lb, on))
        self._legend_layout.insertWidget(self._legend_layout.count() - 1, chip)

    def _on_legend_toggle(self, label: str, active: bool):
        self._legend_toggles[label] = active
        state = self._per_theme_label_state.setdefault(self._current_theme, {})
        state[label] = active
        self._sync_name_to_labels()
        self._redraw_main()

    def _sync_name_to_labels(self):
        """Name mirrors a lone selected label; clears when that stops holding (typed names kept)."""
        if self._name_entry.text().strip() and not self._name_is_auto:
            return
        picked = [lb for lb in self._theme_whitelist(self._current_theme) if lb != 'NONE']
        self._name_entry.setText(picked[0] if len(picked) == 1 else '')
        self._name_is_auto = True

    def _label_color_map(self) -> dict[str, str]:
        if self._label_colors is not None:
            return self._label_colors
        labels = sorted({lb for _, _, lb in self._epoch_bounds})
        cmap = {}
        ci = 0
        for lb in labels:
            if lb == 'NONE':
                cmap[lb] = _TS_NONE_COLOR
            else:
                cmap[lb] = _TS_COLORS[ci % len(_TS_COLORS)]
                ci += 1
        self._label_colors = cmap
        return cmap

    def _redraw_main(self):
        if self.nav.session_any_mode:
            self._main_plot.update_epochs([], {}, 0, 1)
            return
        if not self._epoch_bounds:
            return
        cmap = self._label_color_map()
        visible = [b for b in self._epoch_bounds
                   if self._legend_toggles.get(b[2], True)]
        self._main_plot.update_epochs(visible, cmap, 0.0, self._total_sec)

    def add_theme(self):
        """Add an epoch theme: pick source + format, attach to the session, add to the combo."""
        pass

    def _on_theme_change(self, theme: str):
        if theme == self._current_theme:
            return
        if self._theme_combo.is_add_row(self._theme_combo.currentIndex()):
            return   # AddableDropdown reverts the index and calls add_theme
        self._filter_checks[self._current_theme] = self._filter_check.isChecked()
        self._current_theme = theme
        self._label_colors = None
        self._init_times()
        self._filter_check.blockSignals(True)
        self._filter_check.setChecked(self._filter_checks.get(theme, False))
        self._filter_check.blockSignals(False)

    def _on_label_reset(self):
        self._per_theme_label_state.pop(self._current_theme, None)
        self._update_legend()

    def _on_label_none(self):
        labels = sorted({lb for _, _, lb in self._epoch_bounds})
        off = {lb: False for lb in labels}
        off['NONE'] = False
        self._per_theme_label_state[self._current_theme] = off
        self._update_legend()

    def _on_filter_toggle(self, checked: bool):
        self._filter_checks[self._current_theme] = checked

    def _parse_time_text(self, text: str) -> float:
        s = text.strip().lower()
        if s in ('start', 'end'):
            return 0.0 if s == 'start' else self._total_sec
        return self._hms_to_sec(text)

    def _sync_timing_entries(self, t0: float, t1: float):
        self._start_entry.setText(self._sec_to_hms(t0))
        self._end_entry.setText(self._sec_to_hms(t1))

    def _apply_timing_cursors(self, t0: float, t1: float):
        if t0 > t1:
            t0, t1 = t1, t0
        self._main_plot.set_selection(t0, t1)
        self._sync_timing_entries(t0, t1)

    def _on_main_handle_moved(self, t0: float, t1: float):
        self._sync_timing_entries(t0, t1)

    def _validate_timing_entry(self, which: Literal['start', 'end']):
        entry = self._start_entry if which == 'start' else self._end_entry
        txt = entry.text().strip()
        symbolic = txt.lower() in ('start', 'end')
        try:
            v = self._parse_time_text(txt)
        except ValueError:
            return
        if self.nav.session_any_mode:   # no timeline to sync — text is source of truth
            entry.setText(txt.lower() if symbolic else self._sec_to_hms(v))
            return
        if which == 'start':
            _, t1 = self._main_plot.get_selection()
            self._apply_timing_cursors(v, t1)
        else:
            t0, _ = self._main_plot.get_selection()
            self._apply_timing_cursors(t0, v)
        if symbolic:
            entry.setText(txt.lower())   # cursor moved, but keep the per-session symbol

    def _read_timing(self, any_mode: bool):
        """Timing fields from the UI, or None if invalid."""
        t0_txt = self._start_entry.text().strip()
        t1_txt = self._end_entry.text().strip()
        try:
            t0 = self._parse_time_text(t0_txt)
            t1 = self._parse_time_text(t1_txt)
        except ValueError:
            return None
        if t1 <= t0:
            return None
        if not any_mode:
            self._apply_timing_cursors(t0, t1)
        # keep 'start'/'end' symbolic so each session resolves them against its own bounds
        t0_spec = t0_txt.lower() if t0_txt.lower() in ('start', 'end') else t0
        t1_spec = t1_txt.lower() if t1_txt.lower() in ('start', 'end') else t1
        return (t0_spec, t1_spec, self._splits_spin.value(), *self._overlap_metric.value())

    def _theme_whitelist(self, theme: str) -> list:
        """Labels checked in the legend for *theme* (unrecorded = checked, as the chips show)."""
        saved = self._per_theme_label_state.get(theme, {})
        labels = sorted({lb for _, _, lb in self._all_theme_bounds.get(theme, [])})
        return [lb for lb in labels if saved.get(lb, True)]

    def _read_filter(self) -> dict:
        """Filter state: AND-list of themes. Include-checked themes if any; else current theme."""
        checked = [t for t, on in self._filter_checks.items() if on]
        names = checked or [self._current_theme]
        return [{'name': t, 'labels': self._theme_whitelist(t)} for t in names]

    def _on_set(self):
        any_mode = self.nav.session_any_mode
        if not any_mode and not self._main_plot.has_full_selection():
            return
        if (self._name_entry.text().strip() or 'custom').lower() == _FULL_SEG:
            QMessageBox.warning(None, "Custom CCG",
                                f"'{_FULL_SEG}' is a reserved name — choose another.")
            return
        timing = self._read_timing(any_mode)
        if timing is None:
            return
        t0_spec, t1_spec, n_splits, overlap_raw, overlap_unit = timing
        request = CCGBatchRequest(
            name=self._name_entry.text() or 'custom',
            t0=t0_spec, t1=t1_spec,
            scope=('all' if self.nav.session_any_mode   # scope = session-mode marker
                   else str(getattr(self.nav.key, 'session', ''))),
            sessions=self._sessions_picker.selected,
            n_splits=n_splits, overlap_raw=overlap_raw, overlap_unit=overlap_unit,
            split_mode=('equal_effective' if self._equal_effective_check.isChecked() else 'raw_span'),
            filter_state=self._read_filter())
        self._status_lbl.setText(f"Queued: {request.name}")
        self.queue_ccg_requested.emit(request)

    def _on_clear(self):
        self._reset_handles()
        self._name_entry.clear()
        self._name_is_auto = True   # hand the name back to the chips
        self._sync_name_to_labels()
        self._status_lbl.setText("")

    def save_state(self, path: str):
        import json
        state = {
            'theme': self._current_theme,
            'legend_toggles': dict(self._legend_toggles),
            'per_theme_label_state': {k: dict(v) for k, v in self._per_theme_label_state.items()},
            'include_in_filter': self._filter_checks.get(
                self._current_theme, self._filter_check.isChecked()),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        import json
        with open(path) as f:
            state = json.load(f)
        if 'theme' in state:
            idx = self._theme_combo.findText(state['theme'])
            if idx >= 0:
                self._theme_combo.setCurrentIndex(idx)
        self._per_theme_label_state = state.get('per_theme_label_state', {})
        inc = state.get('include_in_filter', False)
        self._filter_checks[self._current_theme] = inc
        self._filter_check.setChecked(inc)
        self._update_legend()

    @staticmethod
    def _hms_to_sec(hms: str) -> float:
        parts = hms.strip().split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    @staticmethod
    def _sec_to_hms(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class CCGTask:
    """One queued CCG compute.

    With a *spec* this is an appended segment; with *session_key* instead it is
    a whole-session compute (dim0 index 0), which owns its own storage.
    """
    spec: CCGSourceConfig | None
    load_into_ui: bool
    batch_id: int | None = None
    resolution: str = 'lowres'
    session_key: Key | None = None

    @property
    def whole_session(self) -> bool:
        return self.spec is None

    def ccg_key(self) -> Key:
        base = self.session_key if self.whole_session else self.spec.key
        return base.change(resolution=self.resolution)


@dataclass
class CCGTaskResult:
    value: object           # CCGDataset on success
    error: str | None       # error message if failed
    session: str            # session label for routing

    @property
    def ok(self) -> bool:
        return self.error is None


class CustomCCGWorker:
    """Background CCG compute worker."""

    def __init__(self, mgr: 'CustomCCGManager'):
        self._mgr = mgr
        self._ui = mgr._ui
        self._runner = BackgroundTaskRunner(
            max_queue=self._ui.nav.max_ccg_queue, use_result_queue=False)
        self._thread_result: list = []

    def enqueue_task(self, *, spec: CCGSourceConfig = None, load_into_ui: bool = False,
                     batch_id: int | None = None, resolution: str = 'lowres',
                     session_key: Key = None) -> bool:
        """Queue a segment compute (*spec*) or a whole-session one (*session_key*)."""
        task = CCGTask(spec=spec, load_into_ui=bool(load_into_ui),
                       batch_id=batch_id, resolution=resolution,
                       session_key=session_key)
        return self._runner.enqueue(task)

    def on_done(self, completed_task, _result):
        ui, mgr = self._ui, self._mgr
        r: CCGTaskResult = self._thread_result.pop() if self._thread_result else None
        name = ('whole session' if completed_task.whole_session
                else completed_task.spec.name)
        print(f"[CCGq] on_done {name} {completed_task.resolution} "
              f"result={'none' if r is None else ('ok' if r.ok else r.error)}", flush=True)
        bid = completed_task.batch_id
        meta = ui.time_slider._batch_meta.get(bid) if bid is not None else None

        if r is None or not r.ok:
            err = r.error if r else 'unknown error'
            if meta is not None:
                meta['rows'].append(
                    (r.session if r else '?', name, 'fail', err))
            else:
                QMessageBox.critical(None, "Custom CCG",
                    f"Computation failed:\n{err}")
        elif completed_task.whole_session:
            # get_ccg already stored and saved dim0[0]; nothing to splice.
            if meta is not None:
                meta['rows'].append((r.session, name, 'ok', str(r.session)))
        else:
            src, seg_data = r.value           # (CCGSourceConfig, single-segment CCGData)
            nm = src.name
            res_hi = completed_task.resolution == 'highres'

            cd = ui.nav.cd
            try:
                cd.attach_segment(completed_task.ccg_key(), src, seg_data)
            except Exception as exc:
                print(f"[CustomCCG] attach failed '{nm}'/{r.session}: {exc}")
            try:
                mgr.state._emit_inventory_event()
                if r.session == str(ui.nav.key.session):
                    ui.nav.custom_segs_changed.emit()
                    ui.mainview.request_render()
            except Exception:
                import traceback; traceback.print_exc()   # a render error must not stall the queue

            if meta is not None and not res_hi:
                meta['rows'].append((r.session, nm, 'ok', str(r.session)))

        try:
            self._on_chunk_done(completed_task)
        except Exception:
            import traceback; traceback.print_exc()   # never let bookkeeping stall the queue
        self._custom_ccg_start_next()

    def _custom_ccg_start_next(self):
        ui = self._ui
        nav = ui.nav

        def _launch(task: CCGTask, _q):
            self._thread_result.clear()
            seg_key = task.ccg_key()

            name = 'whole session' if task.whole_session else str(task.spec.name)

            def _ccg_worker():
                sess = str(seg_key.session)
                import time as _t; _t0 = _t.time()
                print(f"[CCGq] START {sess} {task.resolution} '{name}'", flush=True)
                try:
                    if task.whole_session:
                        # get_ccg computes dim0 index 0 and writes its own file;
                        # there is no parent array to splice it onto.
                        nav.cd.get_ccg(seg_key)
                        print(f"[CCGq] DONE  {sess} {task.resolution} {_t.time()-_t0:.1f}s", flush=True)
                        self._thread_result.append(CCGTaskResult(None, None, sess))
                        return
                    nav.cd.ccg_for(seg_key.cd())   # base array only; the segment is what we are about to compute
                    print(f"[CCGq]   base ready {sess} {task.resolution} {_t.time()-_t0:.1f}s", flush=True)
                    sliced = nav.cd.nd.sliced_neurons_for(task.spec)
                    if sliced is None:
                        print(f"[CCGq]   NO OVERLAP {sess}", flush=True)
                        self._thread_result.append(CCGTaskResult(None, 'no interval overlap', sess))
                        return
                    neurons_slice, _active_dur = sliced
                    seg_data = nav.cd.compute_segment(seg_key, task.spec, neurons_slice)
                    print(f"[CCGq] DONE  {sess} {task.resolution} {_t.time()-_t0:.1f}s", flush=True)
                    self._thread_result.append(CCGTaskResult((task.spec, seg_data), None, sess))
                except Exception as ex:
                    import traceback; traceback.print_exc()
                    print(f"[CCGq] FAIL  {sess} {task.resolution}: {ex}", flush=True)
                    self._thread_result.append(CCGTaskResult(None, str(ex), sess))

            t = threading.Thread(target=_ccg_worker, daemon=True)
            t.start()
            return t

        started = self._runner.start_next(_launch)
        print(f"[CCGq] start_next -> {started}, pending={len(self._runner._pending)}, "
              f"running={self._runner.is_running()}", flush=True)
        if started:
            self._runner.start_polling_qt(300, self.on_done)

    def _on_chunk_done(self, task: CCGTask):
        bid = task.batch_id
        if bid is None:
            return
        ts = self._ui.time_slider
        if bid not in ts._batch_counts:
            return
        ts._batch_counts[bid] -= 1
        total = ts._batch_totals.get(bid, 0)
        if ts._batch_counts[bid] > 0:
            done = total - ts._batch_counts[bid]
            ts._status_lbl.setText(f"Computing custom CCG… {done}/{total}")
            return
        del ts._batch_counts[bid]
        ts._batch_totals.pop(bid, None)
        self._ui.nav.cd.nd.clear_slice_cache()
        spec_name = (ts._batch_meta.get(bid) or {}).get('spec_name', '')
        ts._status_lbl.setText(f"Done: {spec_name} — {total} CCG(s)")
        names = list(ts._batch_names.pop(bid, []))
        meta = ts._batch_meta.pop(bid, None)
        if meta is not None:
            if meta.get('on_done') is not None:
                failed = [s for s, _n, st, _v in meta.get('rows', []) if st == 'fail']
                QTimer.singleShot(0, lambda f=failed: meta['on_done'](f))
            QTimer.singleShot(100, lambda m=meta: self._show_batch_report(m))
        QTimer.singleShot(100, lambda n=names: self._prompt_save_chunks(n))

    def _show_batch_report(self, meta: dict):
        rows = meta.get('rows', [])
        ok   = [f"  {s}  ->  {v}" for s, _n, st, v in rows if st == 'ok']
        fail = [f"  {s}: {v}"     for s, _n, st, v in rows if st == 'fail']
        skip = [f"  {s}: {w}"     for s, w in meta.get('skipped', [])]
        if len(ok) <= 1 and not fail and not skip:
            return
        lines = [f"Custom CCG: {meta.get('spec_name', '')}"]
        for title, items in (("Computed", ok), ("Failed", fail), ("Skipped", skip)):
            if items or title == "Computed":
                lines += ["", f"{title} ({len(items)}):", *(items or ["  (none)"])]
        ResultsDialog.show_report("Custom CCG results", "\n".join(lines))

    def _prompt_save_chunks(self, names: list[str]):
        return


class CustomCCGState(JsonSavable):
    """Custom CCG session state and suggestions."""

    def __init__(self, ui: 'CCGReviewUI', mgr: 'CustomCCGManager'):
        super().__init__()
        self._mgr = mgr
        self._ui = ui
        self.active_sess: str = ''
        self.inventory_sig: tuple = ()
        self._stacked_segments: set = set()

    def save_path(self, **kwargs) -> str:
        return os.path.join(self._mgr.save_path(), "suggested_custom_ccgs")

    def _emit_inventory_event(self):
        specs = self.load_suggestions()
        sig = tuple(sorted(s._key() for s in specs))
        if sig != self.inventory_sig:
            self.inventory_sig = sig
            self.refresh_suggestions(silent=True)

    def load_suggestions(self) -> list:
        path = self.save_path() + ".json"
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding='utf-8') as f:
                raw = json.load(f)
            out = [CCGBatchRequest.deserialize(x)
                   for x in (raw.get('items') or []) if isinstance(x, dict)]
            return out
        except Exception as ex:
            print(f"[CustomCCG] suggestion list load failed: {ex}")
            return []

    def save_suggestions(self, specs: list) -> None:
        payload = {'version': 1, 'items': [s.serialize() for s in specs]}
        atomic_write_json(self.save_path() + ".json", payload)

    def refresh_suggestions(self, silent: bool = False):
        specs = self.load_suggestions()
        if not silent:
            QMessageBox.information(None, "Custom CCG suggestions",
                                    f"Updated suggestion list with {len(specs)} item(s).")

    def update_suggestion(self, spec: 'CCGBatchRequest'):
        specs = self.load_suggestions()
        if spec not in specs:
            specs.append(spec)
            self.save_suggestions(specs)

    def show_dialog(self):
        ui = self._ui
        specs = self.load_suggestions()
        def _on_run(selected_specs):
            queued = sum(self._mgr._queue_custom_ccgs(s) for s in selected_specs)
            if queued:
                if getattr(ui, "time_slider", None) is not None: ui.time_slider._status_lbl.setText(f"Queued {queued} suggested custom CCG task(s)")
                self._mgr.worker._custom_ccg_start_next()
            else:
                if getattr(ui, "time_slider", None) is not None: ui.time_slider._status_lbl.setText("All suggested custom CCGs already exist")
        SuggestedCCGDialog.show(ui, specs, _on_run)


class CustomCCGManager(Savable):
    """Custom CCG queue coordinator (arrays live on ``cd``)."""

    def __init__(self, ui: 'CCGReviewUI'):
        super().__init__()
        self._ui = ui
        os.makedirs(self.save_path(), exist_ok=True)
        self.state = CustomCCGState(ui, self)
        self.worker = CustomCCGWorker(self)
        ui._custom_ccg_pending = self.worker._runner._pending
        self.state.active_sess = str(ui.nav.key.session)

    def save_path(self, **kwargs) -> str:
        return self._ui.nav.cd.custom_dir

    def _appended_labels(self) -> list:
        """Custom segment labels (dim0 after ``full``)."""
        return [lb for lb in self._ui.nav.available_segments() if lb != _FULL_SEG]

    def _is_custom_segment(self, seg: str = None) -> bool:
        seg = self._ui.current_segment if seg is None else seg
        return seg in self._appended_labels()

    def _custom_seg_index(self, seg: str = None) -> int:
        seg = self._ui.current_segment if seg is None else seg
        labels = self._appended_labels()
        return labels.index(seg) if seg in labels else -1

    def _remove_custom_segment(self, name: str):
        if name not in self._appended_labels():
            return
        cd = self._ui.nav.cd
        cd.drop_segment([self._ui.nav.key.nd().change(segment=name)])  # all resolutions
        if self._ui.current_segment == name:
            self._ui.current_segment = _ALL_SEGS
        self._ui.nav.custom_segs_changed.emit()
        self._ui._build_sig_chips()
        self._ui._update_segment_label()
        self._ui.mainview.request_render()

    def queue_whole_session(self, sessions: list, resolution: str, on_done=None) -> int:
        """Queue whole-session CCG computes; ``on_done(failed_sessions)`` when all land.

        Shares the batch counter and status label with segment computes, so the
        queue reports one x/total regardless of what kind of work is in it.
        """
        ts = self._ui.time_slider
        bid = ts._batch_next_id
        ts._batch_next_id += 1
        queued = 0
        for sess in sessions:
            if self.worker.enqueue_task(session_key=Key(session=str(sess)),
                                        batch_id=bid, resolution=resolution):
                queued += 1
        if queued:
            ts._batch_counts[bid] = queued
            ts._batch_totals[bid] = queued
            ts._batch_meta[bid] = {'spec_name': f'{resolution} CCG',
                                   'skipped': [], 'rows': [], 'on_done': on_done}
            self.worker._custom_ccg_start_next()
        return queued

    def _generate_suggested_custom_ccgs(self):
        self.state.show_dialog()

    def _queue_custom_ccgs(self, spec: 'CCGBatchRequest') -> int:
        nav = self._ui.nav
        _any = nav.session_any_mode
        ts = self._ui.time_slider
        bid = ts._batch_next_id
        ts._batch_next_id += 1
        work, skipped = nav.cd.parse_ccg_batch_request(spec)
        split_names = [s.name for s in work] if len(work) > 1 else []
        # PATCH: always queue both resolutions; the worker computes a missing base CCG in background
        ordered = [(s, res) for s in work for res in ('lowres', 'highres')]
        queued = dropped = 0
        for src, res in ordered:
            if self.worker.enqueue_task(
                    spec=src,
                    load_into_ui=(_any or str(src.key.session) == str(nav.key.session)),
                    batch_id=bid, resolution=res):
                queued += 1
            else:
                dropped += 1
        if dropped:
            runner = self.worker._runner
            QMessageBox.warning(None, "Task queue full",
                f"Custom CCG queue full — {dropped} task(s) not queued "
                f"({len(runner._pending)}/{runner._max_queue}). "
                "Wait for running tasks to complete, then retry.")
        if queued:
            ts._batch_counts[bid] = queued
            ts._batch_totals[bid] = queued
            ts._batch_names[bid] = split_names
            ts._batch_meta[bid] = {'spec_name': str(spec.name),
                                   'skipped': skipped, 'rows': []}
        elif skipped:
            self.worker._show_batch_report({'spec_name': str(spec.name),
                                            'skipped': skipped, 'rows': []})
        return queued


class SuggestedCCGDialog:
    """Dialog to pick suggested custom CCG specs to run."""

    def __init__(self, specs: list, n_total: int, on_run, parent=None):
        self._specs = specs
        self._on_run = on_run

        self._dlg = QDialog(parent)
        self._dlg.setWindowTitle("Suggested custom CCGs")
        self._dlg.resize(640, 380)

        lay = QVBoxLayout(self._dlg)
        lay.addWidget(QLabel("Generate custom CCGs from availability list:"))

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        _mf = QFont(); _mf.setStyleHint(QFont.StyleHint.Monospace); _mf.setPointSize(9)
        self._list.setFont(_mf)
        for i, spec in enumerate(specs):
            name  = str(spec.name)
            t0    = self._format_time(spec.t0)
            t1    = self._format_time(spec.t1)
            scope = str(spec.scope)
            n_have = len(spec.sessions or [])
            label = (f"[{name} | {t0}–{t1}] "
                     f"{'ALL' if scope == 'All' else scope} "
                     f"({n_have}/{n_total})")
            self._list.addItem(label)
            self._list.item(i).setSelected(True)
        lay.addWidget(self._list)

        btn_row = QHBoxLayout()
        for label, slot in [("Generate selected", self._run_selected),
                             ("Generate all",      self._run_all),
                             ("Cancel",            self._dlg.reject)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

    @staticmethod
    def _format_time(v) -> str:
        if isinstance(v, str) and v.lower() in ('start', 'end'):
            return v
        try:
            return str(datetime.timedelta(seconds=int(float(v))))
        except Exception:
            return str(v)

    def _run_selected(self):
        idxs = [self._list.row(it) for it in self._list.selectedItems()]
        self._dlg.accept()
        self._on_run([self._specs[i] for i in idxs])

    def _run_all(self):
        self._dlg.accept()
        self._on_run(list(self._specs))

    @classmethod
    def show(cls, ui, specs: list, on_run):
        if not specs:
            QMessageBox.information(None, "Suggested custom CCGs",
                                    "No suggested entries found. Use 'Refresh' first.")
            return
        n_total = max(1, len(ui.nav.real_nd_keys()))
        cls(specs, n_total, on_run)._dlg.exec()
