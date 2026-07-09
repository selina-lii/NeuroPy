"""Qt time-slider panel: epoch display, zoom, custom-segment CCG creation.

Replaces the tkinter TimeSliderPanel. Uses pyqtgraph for epoch timeline and
emits a spec dict instead of directly calling CustomCCGManager, so it can be
wired to any parent that handles CCG computation.

Architecture
------------
HHMMSTicker          — pyqtgraph AxisItem with HH:MM:SS tick labels
EpochPlotWidget      — pg.PlotWidget: colored epoch blocks + draggable cursors; drag/wheel to zoom x
TimeSliderPanelQt    — top-level QWidget; owns plot + all controls

Signals
-------
ccg_enqueue_requested(dict)  — parent calls enqueue_task(spec)
save_requested()             — parent opens CCG file-management dialog
load_requested()             — parent opens load-custom-CCG dialog
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import TYPE_CHECKING

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
from neuropy.analyses.epoch_filter import EpochFilter
import glob as _glob
from neuropy.analyses.ms_connectivity import CCGData, CCGDataset, CCGSourceConfig
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.utils import JsonSavable, Savable
from neuropy.core.intervals import IntervalOp as _SetOp
from neuropy.ui.ui_common import BackgroundTaskRunner
from neuropy.ui.utils import chip_button, ListPickerButton
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState
    from neuropy.ui.ccg_ui import CCGReviewUI

_TS_COLORS = [
    '#BBDEFB', '#C8E6C9', '#FFF9C4', '#FFE0B2', '#E1BEE7',
    '#F8BBD0', '#D7CCC8', '#B2EBF2', '#DCEDC8', '#F0F4C3',
]
_TS_NONE_COLOR = '#E0E0E0'

_ALL_SEGS = "All"  # must match ccg_ui._ALL_SEGS
_MAX_QUEUE = 50

class EpochPlotWidget(pg.PlotWidget):
    """Epoch timeline with click-to-place timing cursors."""

    handle_moved = Signal(float, float)

    _CURSOR_COLOR = '#1565C0'
    _DRAG_COLOR   = '#C62828'
    _BAR_Y0       = 0.2    # epoch bars occupy y=0.2..1.0 (2x height); axis zone=0..0.2
    _DOT_Y        = 0.1    # cursor dots in axis zone

    def __init__(self, parent=None):
        from neuropy.ui.ui_common import qt_dark_mode
        _axis = pg.AxisItem(orientation='bottom')
        _axis.setStyle(tickLength=10)
        _axis.tickStrings = lambda values, *_: [
            f"{int(max(0.0,float(v))//3600):02d}:"
            f"{int((max(0.0,float(v))%3600)//60):02d}:"
            f"{int(max(0.0,float(v))%60):02d}"
            for v in values]
        super().__init__(parent, axisItems={'bottom': _axis})
        bg = '#2b2b2b' if qt_dark_mode() else None
        self.setBackground(bg)
        if bg is None:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.viewport().setAutoFillBackground(False)
        self.hideAxis('left')
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)
        self.setYRange(0, 1, padding=0)
        self.setFixedHeight(60)

        self._t_min:    float = 0.0
        self._t_max:    float = 1.0
        self._start_t:  float = 0.0
        self._end_t:    float = 0.0
        self._epoch_rects: list = []
        self._snap_times:  list[float] = []
        self._snap_enabled: bool = True
        self._locked:       bool = False
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
            if self._locked or ev.button() != Qt.MouseButton.LeftButton:
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


class TimeSliderPanelQt(QWidget):
    """Qt replacement for TimeSliderPanel.

    Reads session/key from AppState, raw data from CCGDataset.
    Custom CCG creation is delegated to the parent via ccg_enqueue_requested.
    """

    ccg_enqueue_requested = Signal(object)   # spec dict
    save_requested        = Signal()
    load_requested        = Signal()

    def __init__(self, nav: 'AppState', cd: 'CCGDataset', parent=None):
        super().__init__(parent)
        self._nav = nav
        self._cd  = cd

        # Epoch state
        self._epoch_bounds:    list = []
        self._total_sec:       float = 0.0
        self._themes:          dict = {}     # theme_name → Epoch object
        self._all_theme_bounds: dict = {}    # theme_name → [(s,e,label)]
        self._current_theme:   str  = 'segments'
        self._label_colors:    dict | None = None
        self._per_theme_label_state: dict = {}  # theme → {label: bool}
        self._legend_toggles:  dict = {}     # label → bool (current theme)
        self._batch_counts:    dict = {}
        self._batch_names:     dict = {}
        self._batch_next_id:   int = 1

        self._build()
        self._connect_nav()
        self._discover_themes(self._nav._build_themes(self._nav.key))

    def reload_themes(self):
        self._discover_themes(self._nav._build_themes(self._nav.key))

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 1, 4, 1)
        root.setSpacing(1)

        title_lbl = QLabel("Time Slider - Behavioral Epochs")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
        root.addWidget(title_lbl)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Theme:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItem('segments')
        self._theme_combo.setFixedWidth(140)
        self._theme_combo.currentTextChanged.connect(self._on_theme_change)
        row1.addWidget(self._theme_combo)

        self._theme_info_lbl = QLabel("")
        self._theme_info_lbl.setStyleSheet("color:#888; font-size:8pt;")
        row1.addWidget(self._theme_info_lbl)

        # "Include in filter" chip (per-theme flag)
        self._filter_checks: dict[str, bool] = {}
        self._filter_check = chip_button("Include in filter", checked=False)
        self._filter_check.toggled.connect(lambda on: self._filter_check.setText(
            "✓ Include in filter" if on else "Include in filter"))
        self._filter_check.toggled.connect(self._on_filter_toggle)
        row1.addWidget(self._filter_check)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Show:"))
        self._label_combo = QComboBox()
        self._label_combo.setFixedWidth(110)
        self._label_combo.currentTextChanged.connect(self._on_label_change)
        row1.addWidget(self._label_combo)
        reset_btn = QPushButton("All")
        reset_btn.setFixedWidth(40)
        reset_btn.clicked.connect(self._on_label_reset)
        row1.addWidget(reset_btn)

        row1.addStretch()

        save_btn = QToolButton(); save_btn.setText("💾")
        save_btn.clicked.connect(self.save_requested)
        row1.addWidget(save_btn)
        load_btn = QToolButton(); load_btn.setText("📂")
        load_btn.clicked.connect(self.load_requested)
        row1.addWidget(load_btn)

        sep_tb = QFrame()
        sep_tb.setFrameShape(QFrame.VLine)
        sep_tb.setStyleSheet('color: #ccc;')
        row1.addWidget(sep_tb)

        self._snap_check = QCheckBox("Snap")
        self._snap_check.setChecked(True)
        self._snap_check.toggled.connect(self._on_snap_toggle)
        row1.addWidget(self._snap_check)
        self._reset_zoom_btn = QPushButton("Reset")
        self._reset_zoom_btn.setFixedWidth(50)
        self._reset_zoom_btn.clicked.connect(self._on_reset_zoom)
        row1.addWidget(self._reset_zoom_btn)
        self._lock_check = QCheckBox("Lock")
        self._lock_check.toggled.connect(self._on_lock_toggle)
        row1.addWidget(self._lock_check)
        root.addLayout(row1)

        self._legend_widget = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(4)
        self._legend_layout.addStretch()
        root.addSpacing(12)
        root.addWidget(self._legend_widget)
        root.addSpacing(12)

        self._any_mode_lbl = QLabel(
            "All-sessions view: no single behavioral timeline — "
            "switch to one session to use the time slider.")
        self._any_mode_lbl.setWordWrap(True)
        self._any_mode_lbl.setStyleSheet('color:#666; font-size:9pt; padding:4px;')
        self._any_mode_lbl.setVisible(False)
        root.addWidget(self._any_mode_lbl)

        self._main_plot = EpochPlotWidget()
        self._main_plot.handle_moved.connect(self._on_main_handle_moved)
        root.addWidget(self._main_plot)

        self._on_snap_toggle(self._snap_check.isChecked())

        self._timing_row_widget = QWidget()
        timing_row = QHBoxLayout(self._timing_row_widget)
        timing_row.setContentsMargins(0, 0, 0, 0)
        self._timing_lbl = QLabel("CCG time range")
        timing_row.addWidget(self._timing_lbl)
        timing_row.addWidget(QLabel("Start:"))
        self._start_entry = QLineEdit("00:00:00")
        self._start_entry.setFixedWidth(72)
        self._start_entry.editingFinished.connect(self._validate_start)
        timing_row.addWidget(self._start_entry)
        timing_row.addWidget(QLabel("End:"))
        self._end_entry = QLineEdit("00:00:00")
        self._end_entry.setFixedWidth(72)
        self._end_entry.editingFinished.connect(self._validate_end)
        timing_row.addWidget(self._end_entry)

        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._on_set)
        timing_row.addWidget(set_btn)

        self._ccg_extra_widget = QWidget()
        extra_lay = QHBoxLayout(self._ccg_extra_widget)
        extra_lay.setContentsMargins(0, 0, 0, 0)
        extra_lay.setSpacing(4)
        clr_btn = QPushButton("Clear")
        clr_btn.clicked.connect(self._on_clear)
        extra_lay.addWidget(clr_btn)
        _sessions = [str(k.session) for k in self._nav.real_nd_keys()]
        self._sessions_picker = ListPickerButton("Sessions", items=_sessions,
                                                 plural="sessions")
        self._sessions_picker.set_selected([str(self._nav.key.session)])
        self._sessions_picker.setFixedWidth(120)
        extra_lay.addWidget(self._sessions_picker)
        extra_lay.addWidget(QLabel("Name:"))
        self._name_entry = QLineEdit("")
        self._name_entry.setFixedWidth(100)
        extra_lay.addWidget(self._name_entry)
        extra_lay.addWidget(QLabel("Splits:"))
        self._splits_spin = QSpinBox()
        self._splits_spin.setRange(1, 99)
        self._splits_spin.setValue(1)
        self._splits_spin.setFixedWidth(45)
        extra_lay.addWidget(self._splits_spin)
        extra_lay.addWidget(QLabel("Overlap:"))
        self._overlap_entry = QLineEdit("0")
        self._overlap_entry.setFixedWidth(45)
        extra_lay.addWidget(self._overlap_entry)
        self._overlap_unit = QComboBox()
        for u in ('%', 'hr', 'min', 's'):
            self._overlap_unit.addItem(u)
        self._overlap_unit.setFixedWidth(45)
        extra_lay.addWidget(self._overlap_unit)
        timing_row.addWidget(self._ccg_extra_widget)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#555; font-size:8pt;")
        timing_row.addWidget(self._status_lbl)
        timing_row.addStretch()
        root.addWidget(self._timing_row_widget)

    def _connect_nav(self):
        nav = self._nav
        nav.themes_changed.connect(self._on_themes_changed)
        nav.session_mode_changed.connect(self._on_session_mode_changed)

    def _on_themes_changed(self, themes: dict):
        self._discover_themes(themes)
        self._label_colors = None
        self._init_times()
        self._update_legend()
        self._redraw_main()

    def _on_session_mode_changed(self, any_mode: bool):
        self._any_mode_lbl.setVisible(any_mode)
        self._main_plot.setVisible(not any_mode)
        self._legend_widget.setVisible(not any_mode)
        self._timing_row_widget.setEnabled(not any_mode)
        if any_mode:
            self._main_plot.update_epochs([], {}, 0, 1)
        else:
            self._redraw_main()

    def _discover_themes(self, themes: dict):
        self._themes = themes
        self._all_theme_bounds = {
            attr: [(float(s), float(e), str(lb).strip())
                   for s, e, lb in zip(obj.starts, obj.stops, obj.labels)]
            for attr, obj in themes.items()
        }
        theme_names = ['segments'] + sorted(themes)
        cur = self._theme_combo.currentText()
        self._theme_combo.blockSignals(True)
        self._theme_combo.clear()
        self._theme_combo.addItems(theme_names)
        if cur in theme_names:
            self._theme_combo.setCurrentText(cur)
        self._theme_combo.blockSignals(False)
        n = len(themes)
        self._theme_info_lbl.setText(f"{n} theme{'s' if n != 1 else ''}")

    def _init_times(self):
        theme = self._current_theme
        if theme != 'segments' and theme in self._themes:
            epoch = self._themes[theme]
            labs = [str(x).strip() for x in epoch.labels]
            bounds = [(float(s), float(e), lb)
                      for s, e, lb in zip(epoch.starts, epoch.stops, labs)]
            unique = {lb for lb in labs if lb}
            if len(unique) <= 1:
                bounds = [(s, e, theme) for s, e, _ in bounds]
            self._epoch_bounds = bounds
            self._total_sec = float(epoch.stops.max()) if len(epoch.stops) else 1.0
        else:
            ptr = self._nav.ccg_ptr
            if ptr is None:
                self._epoch_bounds = []
                return
            et = self._cd.edge_times_for(self._nav.key)
            if et is None:
                fallback = self._all_theme_bounds.get('paradigm')
                self._epoch_bounds = list(fallback) if fallback else []
                if self._epoch_bounds:
                    self._total_sec = max(b[1] for b in self._epoch_bounds)
                return
            cols = et.columns.tolist()
            def _col(*names):
                return next((c for c in names if c in cols), None)
            sc = _col('start', 't_start', 'start_time')
            ec = _col('stop',  't_end',   'end_time', 'stop_s', 'end')
            self._epoch_bounds = []
            if sc and ec:
                for _, row in et.iterrows():
                    self._epoch_bounds.append(
                        (float(row[sc]), float(row[ec]), str(row['label'])))
                self._total_sec = max(
                    (b[1] for b in self._epoch_bounds), default=1.0)
            else:
                t = 0.0
                for _, row in et.iterrows():
                    dur = float(row['effective_time_hours']) * 3600.0
                    self._epoch_bounds.append((t, t + dur, str(row['label'])))
                    t += dur
                self._total_sec = t or 1.0

        # Initialise overlap from source config if available
        source = getattr(self._cd, 'source', None)
        if isinstance(source, CCGSourceConfig):
            self._overlap_entry.setText(str(source.overlap_sec))
            self._overlap_unit.setCurrentText('s')

        self._filter_check.blockSignals(True)
        self._filter_check.setChecked(self._filter_checks.get(theme, False))
        self._filter_check.blockSignals(False)
        self._update_legend()
        self._reset_handles()

    def _reset_handles(self):
        self._main_plot.clear_selection()
        self._start_entry.setText("00:00:00")
        self._end_entry.setText("00:00:00")

    def _on_snap_toggle(self, checked: bool):
        self._main_plot._snap_enabled = checked

    def _on_lock_toggle(self, checked: bool):
        self._main_plot._locked = checked

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
            chip = chip_button(lbl, checked=active)
            chip.setStyleSheet(
                f"QPushButton {{ border: 1px solid #888; border-radius: 2px; "
                f"padding: 1px 6px; font-size: 8pt; background: {color}; }}"
                f"QPushButton:checked {{ font-weight: bold; }}"
                f"QPushButton:!checked {{ color: #888; background: #f0f0f0; }}")
            chip.toggled.connect(lambda on, lb=lbl: self._on_legend_toggle(lb, on))
            lyt.insertWidget(lyt.count() - 1, chip)

        none_active = saved.get('NONE', True)
        self._legend_toggles['NONE'] = none_active
        none_chip = chip_button("NONE", checked=none_active)
        none_chip.setStyleSheet(
            f"QPushButton {{ border: 1px solid #888; border-radius: 2px; "
            f"padding: 1px 6px; font-size: 8pt; background: {_TS_NONE_COLOR}; "
            f"color: #444; }}"
            f"QPushButton:checked {{ font-weight: bold; }}"
            f"QPushButton:!checked {{ color: #aaa; background: #f0f0f0; }}")
        none_chip.toggled.connect(lambda on: self._on_legend_toggle('NONE', on))
        lyt.insertWidget(lyt.count() - 1, none_chip)

        labels = sorted({lb for _, _, lb in self._epoch_bounds})
        self._label_combo.blockSignals(True)
        cur = self._label_combo.currentText()
        self._label_combo.clear()
        self._label_combo.addItem('')
        for lb in labels:
            self._label_combo.addItem(lb)
        if cur in labels:
            self._label_combo.setCurrentText(cur)
        else:
            self._label_combo.setCurrentIndex(0)
        self._label_combo.blockSignals(False)

        self._redraw_main()

    def _on_legend_toggle(self, label: str, active: bool):
        self._legend_toggles[label] = active
        state = self._per_theme_label_state.setdefault(self._current_theme, {})
        state[label] = active
        self._redraw_main()

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
        if self._nav.session_any_mode:
            self._main_plot.update_epochs([], {}, 0, 1)
            return
        if not self._epoch_bounds:
            return
        cmap = self._label_color_map()
        visible = [b for b in self._epoch_bounds
                   if self._legend_toggles.get(b[2], True)]
        self._main_plot.update_epochs(visible, cmap, 0.0, self._total_sec)

    def _on_theme_change(self, theme: str):
        if theme == self._current_theme:
            return
        self._filter_checks[self._current_theme] = self._filter_check.isChecked()
        self._current_theme = theme
        self._label_colors = None
        self._init_times()
        self._filter_check.blockSignals(True)
        self._filter_check.setChecked(self._filter_checks.get(theme, False))
        self._filter_check.blockSignals(False)
        self._redraw_main()

    def _on_label_change(self, label: str):
        label = (label or '').strip()
        if not self._legend_toggles:
            return
        if not label:
            for lb in self._legend_toggles:
                self._legend_toggles[lb] = True
        else:
            for lb in self._legend_toggles:
                self._legend_toggles[lb] = (lb == label)
        state = self._per_theme_label_state.setdefault(self._current_theme, {})
        for lb, on in self._legend_toggles.items():
            state[lb] = on
        self._redraw_main()

    def _on_label_reset(self):
        self._label_combo.blockSignals(True)
        self._label_combo.setCurrentIndex(0)
        self._label_combo.blockSignals(False)
        for lb in list(self._legend_toggles):
            self._legend_toggles[lb] = True
        self._per_theme_label_state.pop(self._current_theme, None)
        self._update_legend()

    def _on_filter_toggle(self, checked: bool):
        self._filter_checks[self._current_theme] = checked

    def _parse_time_text(self, text: str) -> float:
        s = text.strip().lower()
        if self._nav.session_any_mode:
            if s == 'start':
                return 0.0
            if s == 'end':
                return self._total_sec
        else:
            if s == 'start':
                return 0.0
            if s == 'end':
                return self._total_sec
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

    def _validate_start(self):
        try:
            v = self._parse_time_text(self._start_entry.text())
            _, t1 = self._main_plot.get_selection()
            self._apply_timing_cursors(v, t1)
        except ValueError:
            pass

    def _validate_end(self):
        try:
            v = self._parse_time_text(self._end_entry.text())
            t0, _ = self._main_plot.get_selection()
            self._apply_timing_cursors(t0, v)
        except ValueError:
            pass

    def _on_set(self):
        if self._lock_check.isChecked():
            return
        if not self._main_plot.has_full_selection():
            return
        try:
            t0 = self._parse_time_text(self._start_entry.text())
            t1 = self._parse_time_text(self._end_entry.text())
        except ValueError:
            return
        if t1 <= t0:
            return
        self._apply_timing_cursors(t0, t1)

        overlap_raw = float(self._overlap_entry.text() or 0)
        unit = self._overlap_unit.currentText()
        dur = t1 - t0
        if unit == '%':
            overlap_sec = overlap_raw / 100.0 * dur
        elif unit == 'min':
            overlap_sec = overlap_raw * 60.0
        elif unit == 'hr':
            overlap_sec = overlap_raw * 3600.0
        else:
            overlap_sec = overlap_raw

        active_labels = {lb: on for lb, on in self._legend_toggles.items()}
        filter_state = {
            'theme':  self._current_theme,
            'labels': active_labels,
            'flags':  {'include_in_filter': self._filter_checks.get(
                self._current_theme, self._filter_check.isChecked())},
        }

        spec = {
            't0':         t0,
            't1':         t1,
            'name':       self._name_entry.text() or 'custom',
            'n_splits':   self._splits_spin.value(),
            'overlap_sec': max(0.0, overlap_sec),
            'filter_state': filter_state,
            'sessions':   self._sessions_picker.selected,
            'scope':      str(getattr(self._nav.key, 'session', '')),
        }
        self._status_lbl.setText(f"Queued: {spec.get('name', 'custom')}")
        self.ccg_enqueue_requested.emit(spec)

    def _on_clear(self):
        self._reset_handles()
        self._name_entry.clear()
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

    def _resolve_ts_time(self, raw, t_start: float, t_end: float) -> float:
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s == 'start':
                return float(t_start)
            if s == 'end':
                return float(t_end)
            try:
                return float(self._hms_to_sec(raw))
            except (ValueError, TypeError):
                pass
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(t_start)

    @staticmethod
    def single_exclusive_segment_filter_label(filter_state: dict):
        """Return the one active label from filter_state, or None if 0 or 2+."""
        labels = (filter_state or {}).get('labels') or {}
        on = [lb for lb, v in labels.items() if v and lb != 'NONE']
        return on[0] if len(on) == 1 else None

    def _union_span_for_segment_label(self, key, label: str):
        et = self._cd.edge_times_for(key)
        if et is None:
            return None
        cols = et.columns.tolist()
        sc = next((c for c in ('start', 't_start', 'start_time') if c in cols), None)
        ec = next((c for c in ('stop', 't_end', 'end_time', 'stop_s', 'end') if c in cols), None)
        if not sc or not ec:
            return None
        spans = [(float(row[sc]), float(row[ec]))
                 for _, row in et.iterrows() if str(row['label']) == label]
        if not spans:
            return None
        return min(s for s, _ in spans), max(e for _, e in spans)

    def _intervals_for_spec_on_key(self, spec, key):
        if hasattr(spec, 'filter_state'):
            fs = spec.filter_state or {}
            t0_raw = spec.t0
            t1_raw = spec.t1
        else:
            fs = spec.get('filter_state') or {}
            t0_raw = spec.get('t0', 0.0)
            t1_raw = spec.get('t1', 'end')
        theme = fs.get('theme', 'segments')
        labels = fs.get('labels') or {}
        flags = fs.get('flags') or {}
        include = bool(flags.get('include_in_filter', False))
        wall_extent = self._session_wall_clock_extent_for_key(key)
        t0 = self._resolve_ts_time(t0_raw, *wall_extent)
        t1 = self._resolve_ts_time(t1_raw, *wall_extent)
        if t1 <= t0:
            return None, None
        bounds = []
        if theme == 'segments':
            et = self._cd.edge_times_for(key)
            if et is not None:
                cols = et.columns.tolist()
                sc = next((c for c in ('start', 't_start', 'start_time') if c in cols), None)
                ec = next((c for c in ('stop', 't_end', 'end_time', 'stop_s', 'end') if c in cols), None)
                if sc and ec:
                    for _, row in et.iterrows():
                        bounds.append((float(row[sc]), float(row[ec]), str(row['label'])))
        elif theme in self._themes:
            ep = self._themes[theme]
            labs = [str(x).strip() for x in ep.labels]
            bounds = [(float(s), float(e), lb)
                      for s, e, lb in zip(ep.starts, ep.stops, labs)]
        active = {lb for lb, on in labels.items() if on}
        if include and active:
            bounds = [b for b in bounds if b[2] in active]
        elif labels:
            bounds = [b for b in bounds if labels.get(b[2], True)]
        if not bounds:
            return [(t0, t1)], t1 - t0
        active = ({lb for lb, on in labels.items() if on} if labels
                  else set(lb for _, _, lb in bounds))
        if include and not active:
            active = set(lb for _, _, lb in bounds)
        ef = EpochFilter(bounds)
        result = ef.filter(active or set(lb for _, _, lb in bounds), t0, t1)
        if result is False:
            return [], 0.0
        intervals, active_dur = result
        if intervals is None:
            return [(t0, t1)], active_dur
        return intervals, active_dur

    def _session_wall_clock_extent_for_key(self, key) -> tuple:
        if self._epoch_bounds and str(getattr(key, 'session', '')) == str(getattr(self._nav.key, 'session', '')):
            t = self._total_sec
            return (0.0, t if t > 0 else 1.0)
        return (0.0, 1.0)

    @staticmethod
    def _sec_to_hms(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class CCGTask:
    spec: CCGSourceConfig   # name, t0, t1, filter_state, active_duration, etc.
    key: object             # routing TypeKey
    intervals: object       # resolved epoch intervals or None
    auto_save: bool
    load_into_ui: bool
    batch_id: int | None = None


@dataclass
class CCGTaskResult:
    value: object           # CCGDataset on success
    error: str | None       # error message if failed
    session: str            # session label for routing

    @property
    def ok(self) -> bool:
        return self.error is None


class CustomCCGWorker:
    """Wraps BackgroundTaskRunner + thread result buffer for CCG computation."""

    def __init__(self, mgr: 'CustomCCGManager'):
        self._mgr = mgr
        self._ui = mgr._ui
        self._runner = BackgroundTaskRunner(max_queue=_MAX_QUEUE, use_result_queue=False)
        self._thread_result: list = []

    def enqueue_task(self, *, spec: CCGSourceConfig, key, intervals,
                     auto_save: bool, load_into_ui: bool,
                     batch_id: int | None = None) -> bool:
        task = CCGTask(
            spec=spec, key=key, intervals=intervals,
            auto_save=bool(auto_save), load_into_ui=bool(load_into_ui),
            batch_id=batch_id,
        )
        if not self._runner.enqueue(task):
            n = len(self._runner._pending)
            QMessageBox.warning(None, "Task queue full",
                f"Custom CCG queue full ({n}/{_MAX_QUEUE}). "
                "Wait for running tasks to complete.")
            return False
        return True

    def on_done(self, completed_task, _result):
        ui, mgr = self._ui, self._mgr
        r: CCGTaskResult = self._thread_result.pop() if self._thread_result else None

        if r is None or not r.ok:
            QMessageBox.critical(None, "Custom CCG",
                f"Computation failed:\n{r.error if r else 'unknown error'}")
        else:
            nm = r.value.src_conf.name

            if completed_task.auto_save:
                self._save_custom_to_dir(r.value)
                mgr.state._emit_inventory_event()

            lst = mgr._by_session.setdefault(r.session, [])
            idx = next((i for i, cd in enumerate(lst) if cd.src_conf.name == nm), -1)
            if idx >= 0:
                lst[idx] = r.value
            else:
                lst.append(r.value)
                idx = len(lst) - 1

            if r.session == str(ui._nav.key.session):
                ui._nav._custom_seg_index[nm] = r.value
                ui._nav.custom_segs_changed.emit()
                ui.request_redraw()

            ui.time_slider._status_lbl.setText(f"Done: {nm}")

        self._on_chunk_done(completed_task)
        self._custom_ccg_start_next()

    def _save_custom_to_dir(self, cd) -> None:
        """Persist a custom-segment CCGDataset's CCGData items into the project's
        custom_ccg/ dir via explicit paths. Never calls CCGDataset.save(), so it
        never spawns a project_<src_conf.name> folder."""
        save_dir = self._ui.paths.custom_ccg_dir
        os.makedirs(save_dir, exist_ok=True)
        nm = cd.src_conf.name
        for k, cdata in cd.ccg.items():
            res = getattr(k, 'resolution', 'lowres') or 'lowres'
            suffix = '_highres' if res == 'highres' else ''
            cdata.save(path=os.path.join(save_dir, f"{nm}{suffix}"))

    def _custom_ccg_start_next(self):
        ui = self._ui
        nav = ui._nav

        def _launch(task: CCGTask, _q):
            nd_key = task.key.nd()
            def _read():
                return (
                    nav.cd.ccg_for(nd_key, 'lowres'),
                    (nav.cd.nd.data[nd_key] if getattr(nav.cd, 'nd', None) is not None else None),
                )

            ccg_data_obj = neurons_obj = None
            for attempt in range(2):
                ccg_data_obj, neurons_obj = _read()
                if ccg_data_obj is not None and neurons_obj is not None:
                    break
                if attempt == 0:
                    try:
                        nav.cd.get_ccg()
                    except Exception as ex:
                        print(f"[CustomCCG] ERROR: session load failed for {task.key.session}: {ex}")
                        self._on_chunk_done(task)
                        return None
            if ccg_data_obj is None or neurons_obj is None:
                self._on_chunk_done(task)
                return None
            self._thread_result.clear()

            def _ccg_worker():
                sess = str(task.key.session)
                try:
                    neurons_override = (
                        neurons_obj.time_multislices(*zip(*task.intervals))
                        if task.intervals is not None else None)
                    value = self._compute_custom_segment(
                        task.spec.t0, task.spec.t1, task.spec.name,
                        neurons_override=neurons_override,
                        active_duration=task.spec.active_duration,
                        key_override=task.key,
                        neurons_obj=neurons_obj,
                        ccg_data_obj=ccg_data_obj)
                    if value is None:
                        self._thread_result.append(CCGTaskResult(None, 'compute returned None', sess))
                        return
                    value.src_conf.filter_state = task.spec.filter_state
                    self._thread_result.append(CCGTaskResult(value, None, sess))
                except Exception as ex:
                    self._thread_result.append(CCGTaskResult(None, str(ex), sess))

            t = threading.Thread(target=_ccg_worker, daemon=True)
            t.start()
            return t

        started = self._runner.start_next(_launch)
        if started:
            self._runner.start_polling_qt(300, self.on_done)

    def _compute_custom_segment(self, t0: float, t1: float, name: str,
                                 neurons_override=None, active_duration=None,
                                 key_override=None, neurons_obj=None,
                                 ccg_data_obj=None):
        ui = self._ui
        nav = ui._nav
        key_eff = key_override or nav.key
        neurons_eff = neurons_obj if neurons_obj is not None else nav.neurons
        cd_eff = ccg_data_obj if ccg_data_obj is not None else nav.ccg_data
        if neurons_eff is None:
            print(f"[CustomCCG] ERROR: No neuron data available for {key_eff}")
            return None
        try:
            neurons_slice = (neurons_override if neurons_override is not None
                             else neurons_eff.time_slice(t0, t1))
            _full = sum(len(st) for st in neurons_eff.spiketrains)
            _win  = sum(len(st) for st in neurons_slice.spiketrains)
            print(f"[CustomCCG] '{name}' window=[{t0:.1f},{t1:.1f}]s "
                  f"spikes windowed={_win}/{_full} "
                  f"(override={neurons_override is not None})")
            has_highres = nav.cd.ccg_for(nav.key.nd(), 'highres') is not None
            cd = CCGDataset(conf=cd_eff.conf, nd=nav.cd.nd,
                            save_path=str(ui.paths.data_root),
                            src_conf=CCGSourceConfig(
                                name=name, t0=t0, t1=t1, active_duration=active_duration,
                            ))
            cd.get_ccg_custom(neurons_slice, has_highres=has_highres,
                              excitability=getattr(key_eff, 'excitability', 'E'))
            return cd
        except Exception as ex:
            print(f"[CustomCCG] ERROR: {ex}")
            traceback.print_exc()
            return None

    def _on_chunk_done(self, task: CCGTask):
        bid = task.batch_id
        if bid is None:
            return
        ts = self._ui.time_slider
        if bid not in ts._batch_counts:
            return
        ts._batch_counts[bid] -= 1
        if ts._batch_counts[bid] > 0:
            return
        del ts._batch_counts[bid]
        names = list(ts._batch_names.pop(bid, []))
        QTimer.singleShot(100, lambda n=names: self._prompt_save_chunks(n))

    def _prompt_save_chunks(self, names: list[str]):
        name_set = set(names)
        if not name_set:
            return
        unsaved: list = []
        for lst in self._mgr._by_session.values():
            for cd in lst or []:
                lo_key = next((k for k in cd.ccg if getattr(k, 'resolution', None) == 'lowres'), None)
                if cd.src_conf.name in name_set and (lo_key is None or not cd.ccg[lo_key].is_saved):
                    unsaved.append(cd)
        if not unsaved:
            return
        n = len(unsaved)
        reply = QMessageBox.question(None, "Save split windows",
            f"{n} split window(s) finished computing but are not saved to disk yet.\n\n"
            "Save them as .npz files now? (You can reload them later from the cache.)")
        if reply != QMessageBox.StandardButton.Yes:
            return
        saved = []
        for cd in unsaved:
            try:
                self._save_custom_to_dir(cd)
                saved.append(cd.src_conf.name)
            except Exception as exc:
                print(f"[CustomCCG] save failed '{cd.src_conf.name}': {exc}")
        if saved:
            QMessageBox.information(None, "Saved", "Saved:\n" + "\n".join(saved))


class CustomCCGState(JsonSavable):
    """Active session, inventory change-detection, and suggestion list management."""

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
        sig = tuple(sorted((s._key(), s.scope) for s in specs))
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
            sess = self._ui._nav.current_session_str
            out = [CCGSourceConfig.deserialize(x, default_session=sess)
                   for x in (raw.get('items') or []) if isinstance(x, dict)]
            return out
        except Exception as ex:
            print(f"[CustomCCG] suggestion list load failed: {ex}")
            return []

    def save_suggestions(self, specs: list) -> None:
        payload = {'version': 1, 'items': [{k: v for k, v in s.__dict__.items() if k in s._JSON_KEYS} for s in specs]}
        atomic_write_json(self.save_path() + ".json", payload)

    def refresh_suggestions(self, silent: bool = False):
        specs = self.load_suggestions()
        if not silent:
            QMessageBox.information(None, "Custom CCG suggestions",
                                    f"Updated suggestion list with {len(specs)} item(s).")

    def update_suggestion(self, spec: 'CCGSourceConfig'):
        specs = self.load_suggestions()
        if spec not in specs:
            specs.append(spec)
            self.save_suggestions(specs)

    def show_dialog(self):
        ui = self._ui
        specs = self.load_suggestions()
        def _on_run(selected_specs):
            queued = sum(
                self._mgr._queue_custom_ccgs(
                    s, for_all=(str(s.scope).lower() == 'all'),
                    auto_save=True)
                for s in selected_specs)
            if queued:
                if getattr(ui, "time_slider", None) is not None: ui.time_slider._status_lbl.setText(f"Queued {queued} suggested custom CCG task(s)")
                self._mgr.worker._custom_ccg_start_next()
            else:
                if getattr(ui, "time_slider", None) is not None: ui.time_slider._status_lbl.setText("All suggested custom CCGs already exist")
        SuggestedCCGDialog.show(ui, specs, _on_run)


class CustomCCGManager(Savable):
    """Coordinator: owns worker/state and _by_session; houses cross-cutting UI methods."""

    def __init__(self, ui: 'CCGReviewUI'):
        super().__init__()
        self._ui = ui
        self._by_session: dict = {}
        os.makedirs(self.save_path(), exist_ok=True)
        self.state = CustomCCGState(ui, self)
        self.worker = CustomCCGWorker(self)
        ui._custom_ccg_pending = self.worker._runner._pending
        self.state.active_sess = str(ui._nav.key.session)
        self._by_session.setdefault(self.state.active_sess, [])

    def save_path(self, **kwargs) -> str:
        return self._ui.paths.custom_ccg_dir

    @property
    def _active_list(self) -> list:
        return self._by_session.setdefault(self.state.active_sess, [])

    def _is_custom_segment(self, seg: str = None) -> bool:
        seg = self._ui.current_segment if seg is None else seg
        return seg in self._ui._nav.custom_seg_index

    def _custom_seg_index(self, seg: str = None) -> int:
        seg = self._ui.current_segment if seg is None else seg
        for ci, cs in enumerate(self._active_list):
            if cs.src_conf.name == seg:
                return ci
        return -1

    def _remove_custom_segment(self, name: str):
        ci = self._custom_seg_index(name)
        if ci < 0:
            return
        self._active_list.pop(ci)
        if self._ui.current_segment == name:
            self._ui.current_segment = _ALL_SEGS
        self._ui._build_sig_chips()
        self._ui._update_segment_label()
        self._ui.request_redraw()

    def load_saved_from_disk(self) -> int:
        """Register saved .npz custom CCGs from custom_ccg_dir that aren't already in memory.
        The npz stores only arrays (no conf/key/src_conf), so t0/t1 aren't recoverable and
        default to 0; the segment key uses session=<name> to match freshly-computed customs
        (get_ccg_custom keys on src.name)."""
        save_dir = self.save_path()
        if not os.path.isdir(save_dir):
            return 0
        conf = self._ui._nav.cd.conf
        root = str(self._ui.paths.data_root)
        known = {cd.src_conf.name for lst in self._by_session.values()
                 for cd in lst if getattr(cd, 'src_conf', None)}
        bases: dict = {}
        for p in _glob.glob(os.path.join(save_dir, '*.npz')):
            stem = os.path.splitext(os.path.basename(p))[0]
            base, res = ((stem[:-8], 'highres') if stem.endswith('_highres')
                         else (stem, 'lowres'))
            bases.setdefault(base, {})[res] = os.path.splitext(p)[0]
        sess = self.state.active_sess
        added = 0
        for base, files in sorted(bases.items()):
            if base in known:
                continue
            try:
                ds = CCGDataset(conf=conf, nd=None, save_path=root,
                                src_conf=CCGSourceConfig(name=base, t0=0.0, t1=0.0))
                for res, path in files.items():
                    key = Key(session=base, resolution=res)
                    cd = CCGData(key=key, conf=conf, ccg=None, ccg_null=None,
                                 pval=None, qval=None, root=root)
                    cd.load(path=path)
                    ds.ccg[key] = cd
            except Exception as exc:
                print(f"[CustomCCG] skip {base}: {exc}")
                continue
            self._by_session.setdefault(sess, []).append(ds)
            added += 1
        if added:
            print(f"[CustomCCG] loaded {added} saved custom CCG(s) from {save_dir}")
        return added

    def _generate_suggested_custom_ccgs(self):
        self.state.show_dialog()

    def _queue_custom_ccgs(self, spec: 'CCGSourceConfig', *, for_all: bool, auto_save: bool,
                           target_sessions: list | None = None) -> int:
        ui = self._ui
        nav = ui._nav
        queued = 0
        if for_all and target_sessions is None:
            targets = nav.available_type_keys_any()
        else:
            sess_set = ({str(s) for s in target_sessions}
                        if target_sessions is not None
                        else {str(s) for s in (spec.sessions or []) if s != 'All'})
            targets = ([tk_ for nk in nav.real_nd_keys()
                        if str(nk.session) in sess_set
                        for tk_ in (nav.type_key_for_nd(nk),) if tk_ is not None]
                       if sess_set else [nav.key])
        n_splits = max(1, int(spec.n_splits or 1))
        overlap_sec = max(0.0, float(spec.overlap_sec or 0.0))
        _any = nav.session_any_mode
        for tk_ in targets:
            t_sess_start, t_sess_end = ui.time_slider._session_wall_clock_extent_for_key(tk_)
            t0_r = ui.time_slider._resolve_ts_time(spec.t0, t_sess_start, t_sess_end)
            t1_r = ui.time_slider._resolve_ts_time(spec.t1, t_sess_start, t_sess_end)
            lone = ui.time_slider.single_exclusive_segment_filter_label(spec.filter_state)
            if lone is not None:
                span = ui.time_slider._union_span_for_segment_label(tk_, lone)
                if span is not None:
                    t0_r, t1_r = span[0], span[1]
                    t0_r = ui.time_slider._resolve_ts_time(t0_r, t_sess_start, t_sess_end)
                    t1_r = ui.time_slider._resolve_ts_time(t1_r, t_sess_start, t_sess_end)
            chunks = _SetOp.partition(t0_r, t1_r, n_splits, overlap_sec, str(spec.name))
            split_bid = None
            if len(chunks) > 1 and (_any or str(tk_.session) == str(nav.key.session)):
                split_bid = ui.time_slider._batch_next_id
                ui.time_slider._batch_next_id += 1
            split_names: list[str] = []
            for chunk_t0, chunk_t1, chunk_name in chunks:
                lo = min(t_sess_start, t_sess_end)
                hi = max(t_sess_start, t_sess_end)
                cs = min(max(float(chunk_t0), lo), hi)
                ce = min(max(float(chunk_t1), lo), hi)
                if ce <= cs:
                    continue
                chunk_t0, chunk_t1 = cs, ce
                chunk_spec = CCGSourceConfig(
                    name=chunk_name, t0=chunk_t0, t1=chunk_t1,
                    filter_state=spec.filter_state,
                    scope=spec.scope, sessions=spec.sessions,
                )
                iv = ui.time_slider._intervals_for_spec_on_key(chunk_spec, tk_)
                if iv is None or iv is False:
                    continue
                intervals, active_duration = iv
                if (isinstance(intervals, list) and len(intervals) == 0
                        and (active_duration is None or float(active_duration) <= 0.0)):
                    print(f"[CustomCCG] skip chunk (no overlap with filter): "
                          f"{chunk_name} session={tk_.session}")
                    continue
                chunk_source = CCGSourceConfig(
                    name=chunk_name, t0=chunk_t0, t1=chunk_t1,
                    active_duration=active_duration,
                    filter_state=spec.filter_state,
                    scope=spec.scope, sessions=spec.sessions,
                )
                ok = self.worker.enqueue_task(
                    spec=chunk_source,
                    key=tk_,
                    intervals=intervals,
                    auto_save=auto_save,
                    load_into_ui=(_any or str(tk_.session) == str(nav.key.session)),
                    batch_id=split_bid,
                )
                if ok:
                    queued += 1
                    if split_bid is not None:
                        split_names.append(chunk_name)
            if split_bid is not None and split_names:
                ui.time_slider._batch_counts[split_bid] = len(split_names)
                ui.time_slider._batch_names[split_bid] = split_names
        return queued


class SuggestedCCGDialog:
    """List suggested custom CCG specs; user picks which to generate."""

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
        n_total = max(1, len(ui._nav.real_nd_keys()))
        cls(specs, n_total, on_run)._dlg.exec()
