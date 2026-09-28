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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, Signal, QObject, QPointF, QPoint, QRectF, QTimer
from pyqtgraph.Qt.QtWidgets import (
    QAbstractItemView, QDialog, QListWidget, QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QCheckBox, QComboBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QScrollArea,
    QGraphicsRectItem, QToolButton, QTableWidget, QTableWidgetItem,
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
    ResultsDialog, small_font_pt, regular_font_pt, radio_group, prompt_name)
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState
    from neuropy.ui.ccg_ui import CCGReviewUI

_TS_COLORS = [
    '#BBDEFB', '#C8E6C9', '#FFF9C4', '#FFE0B2', '#E1BEE7',
    '#F8BBD0', '#D7CCC8', '#B2EBF2', '#DCEDC8', '#F0F4C3',
]
_TS_NONE_COLOR = '#E0E0E0'
_CUSTOM_COLORS = [
    '#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4',
    '#42D4F4', '#F032E6', '#9A6324', '#469990', '#800000',
    '#808000', '#000075', '#BFEF45', '#DCBEFF', '#FFD8B1',
    '#AAFFC3', '#FABED4', '#FFE119', '#A9A9A9', '#E6BEFF',
]
_CUSTOM_ALPHA = 110
_FAMILY_PREFIX = '⊞ '
_CUSTOM_COLS = ["name", "start", "end", "active", "filters", "overlap"]
_MODE_ICONS = {'epoch': ('⏱', "Behavioral epochs"), 'custom': ('◆', "Custom CCG windows")}

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
                      t_min: float, t_max: float, overlays: list = ()):
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
        for t0, t1, color in overlays:
            fill = QColor(color); fill.setAlpha(_CUSTOM_ALPHA)
            rect = QGraphicsRectItem(t0, self._BAR_Y0, t1 - t0, bar_h)
            rect.setBrush(QBrush(fill))
            rect.setPen(QPen(QColor(color), 0))
            rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            rect.setZValue(-0.5)
            vb.addItem(rect)
            self._epoch_rects.append(rect)
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


class TimeSliderBackend:
    """Headless time-slider state: themes, label filters, window, custom-CCG batches."""

    def __init__(self, nav: 'AppState', cd: 'CCGDataset'):
        self.nav = nav
        self.cd = cd
        self.epoch_bounds:  list = []
        self.total_sec:     float = 0.0
        self.all_theme_bounds: dict = {}    # theme → [(s, e, label)]
        self.current_theme: str = 'segments'
        self.per_theme_label_state: dict = {}   # theme → {label: bool}
        self.legend_toggles: dict = {}      # label → bool (current theme)
        self.filter_checks:  dict = {}      # theme → included in the cross-theme AND
        self.start: float = 0.0
        self.end:   float = 0.0
        self.name:  str = ''
        self.name_is_auto: bool = True
        self.n_splits: int = 1
        self.seg_len_sec: float = 0.0
        self.discard_last: bool = False
        self.overlap: tuple = (0.0, '%')
        self.equal_effective: bool = False
        self.sessions: list = []
        self.batch_counts:  dict = {}       # batch_id → tasks remaining
        self.batch_totals:  dict = {}
        self.batch_names:   dict = {}
        self.batch_meta:    dict = {}       # batch_id → {spec_name, skipped, rows}
        self.batch_next_id: int = 1
        self._label_colors: dict | None = None
        self.mode: Literal['epoch', 'custom'] = 'epoch'
        self.custom_selected: list = []
        self.custom_visible:  dict = {}
        self._custom_windows: tuple = (None, [])

    def current_session(self) -> str | None:
        """The session the slider is scoped to, or None in all-session mode."""
        return None if self.nav.session_any_mode else str(self.nav.key.session)

    def set_current_session(self, key) -> None:
        """Scope the slider to one session, or to all when *key* is None."""
        self.nav.set_session_any_mode(key is None)
        if key is not None:
            self.nav.set_key(self.nav.key.change(session=str(key)))

    def list_themes(self) -> list:
        """Discovered themes, 'segments' first."""
        return ['segments'] + sorted(self.all_theme_bounds)

    def discover_themes(self, themes: dict) -> str:
        """Store each theme's bounds; returns the theme that should be current."""
        self.all_theme_bounds = {
            attr: self._theme_bounds(obj, attr) for attr, obj in themes.items()}
        names = self.list_themes()
        cur = self.current_theme
        return cur if cur in names else (names[1] if len(names) > 1 else 'segments')

    @staticmethod
    def _theme_bounds(obj, attr: str) -> list:
        labs = [str(x).strip() for x in obj.labels]
        bounds = [(float(s), float(e), lb)
                  for s, e, lb in zip(obj.starts, obj.stops, labs)]
        if len({lb for lb in labs if lb}) <= 1:   # unlabelled theme: the theme is the label
            bounds = [(s, e, attr) for s, e, _ in bounds]
        return bounds

    def set_theme(self, theme: str) -> None:
        """Select *theme* and reload its bounds and total span."""
        self.current_theme = theme
        self._label_colors = None
        if theme != 'segments' and theme in self.all_theme_bounds:
            self.epoch_bounds = list(self.all_theme_bounds[theme])
            self.total_sec = max((b[1] for b in self.epoch_bounds), default=1.0)
        else:   # 'segments' = no label filter: timeline spans the whole session
            self.epoch_bounds = []
            _, t_stop = self.nav.cd.nd.session_bounds(self.nav.key)
            self.total_sec = float(t_stop) or 1.0

    @property
    def active_bounds(self) -> list:
        """Current theme's epochs whose label is toggled on."""
        return [b for b in self.epoch_bounds if self.legend_toggles.get(b[2], True)]

    @staticmethod
    def label_colors(labels) -> dict[str, str]:
        """The slider's palette over *labels* in sorted order; NONE is always grey."""
        cmap, ci = {}, 0
        for lb in sorted(set(labels)):
            if lb == 'NONE':
                cmap[lb] = _TS_NONE_COLOR
            else:
                cmap[lb] = _TS_COLORS[ci % len(_TS_COLORS)]
                ci += 1
        return cmap

    def label_color_map(self) -> dict[str, str]:
        if self._label_colors is None:
            self._label_colors = self.label_colors(lb for _, _, lb in self.epoch_bounds)
        return self._label_colors

    def rebuild_legend(self) -> dict:
        """Refresh ``legend_toggles`` from the saved per-theme state; returns it."""
        saved = self.per_theme_label_state.get(self.current_theme, {})
        self.legend_toggles = {lb: saved.get(lb, True)
                               for lb in self.label_color_map()}
        self.legend_toggles['NONE'] = saved.get('NONE', True)
        return self.legend_toggles

    def set_label(self, label: str, active: bool) -> None:
        self.legend_toggles[label] = active
        self.per_theme_label_state.setdefault(self.current_theme, {})[label] = active

    def theme_whitelist(self, theme: str) -> list:
        """Labels checked in the legend for *theme* (unrecorded = checked)."""
        saved = self.per_theme_label_state.get(theme, {})
        labels = sorted({lb for _, _, lb in self.all_theme_bounds.get(theme, [])})
        return [lb for lb in labels if saved.get(lb, True)]

    def custom_windows(self) -> dict:
        """Current session's custom CCGs: segment → (effective intervals, config)."""
        sess = str(self.nav.key.session)
        if self._custom_windows[0] != sess:
            self._custom_windows = (sess, {seg: (iv, cfg) for seg, iv, cfg
                                           in self.cd.custom_windows(self.nav.key)})
        return self._custom_windows[1]

    def custom_families(self) -> dict:
        return self.cd.custom_families(list(self.custom_windows()))

    def custom_items(self) -> list:
        """Picker rows: families first, then every custom CCG of the session."""
        return [_FAMILY_PREFIX + f for f in sorted(self.custom_families())] + list(self.custom_windows())

    def custom_color_map(self) -> dict[str, str]:
        return {seg: _CUSTOM_COLORS[i % len(_CUSTOM_COLORS)]
                for i, seg in enumerate(self.custom_selected)}

    def set_custom(self, picked: list) -> None:
        """Chips = picked custom CCGs, a family expanding to its members, in pick order."""
        fams = self.custom_families()
        segs = [m for p in picked for m in fams.get(p.removeprefix(_FAMILY_PREFIX), [p])]
        self.custom_selected = [s for i, s in enumerate(segs)
                                if s in self.custom_windows() and s not in segs[:i]]

    def custom_overlays(self) -> list:
        """(t0, t1, color) per effective interval of each visible chip."""
        wins, cmap = self.custom_windows(), self.custom_color_map()
        return [(t0, t1, cmap[seg]) for seg in self.custom_selected
                if seg in wins and self.custom_visible.get(seg, True) for t0, t1 in wins[seg][0]]

    def filter_snapshot(self) -> dict:
        """Theme and filter state for UIStates; it outlives the window and must survive a restart."""
        return {'mode': self.mode,
                'custom_selected': list(self.custom_selected),
                'custom_visible': dict(self.custom_visible),
                'current_theme': self.current_theme,
                'filter_checks': dict(self.filter_checks),
                'per_theme_label_state': {t: dict(d)
                                          for t, d in self.per_theme_label_state.items()}}

    def restore_filters(self, state: dict) -> None:
        self.mode = state.get('mode', 'epoch')
        self.custom_selected = list(state.get('custom_selected') or [])
        self.custom_visible = dict(state.get('custom_visible') or {})
        self.filter_checks = dict(state.get('filter_checks') or {})
        self.per_theme_label_state = {t: dict(d) for t, d
                                      in (state.get('per_theme_label_state') or {}).items()}
        theme = state.get('current_theme')
        if theme in self.list_themes():
            self.current_theme = theme

    def filter_state(self) -> list:
        """AND-list of themes: include-checked ones if any, else the current theme."""
        checked = [t for t, on in self.filter_checks.items() if on]
        return [{'name': t, 'labels': self.theme_whitelist(t)}
                for t in (checked or [self.current_theme])]

    def auto_name(self) -> str:
        """A lone selected label names the segment; otherwise the name clears."""
        picked = [lb for lb in self.theme_whitelist(self.current_theme) if lb != 'NONE']
        return picked[0] if len(picked) == 1 else ''

    def parse_time(self, text: str) -> float:
        s = text.strip().lower()
        if s in ('start', 'end'):
            return 0.0 if s == 'start' else self.total_sec
        return self._hms_to_sec(text)

    def build_request(self, t0_spec, t1_spec) -> CCGBatchRequest:
        """The batch request the current backend state describes."""
        return CCGBatchRequest(
            name=self.name or 'custom', t0=t0_spec, t1=t1_spec,
            scope=('all' if self.nav.session_any_mode
                   else str(getattr(self.nav.key, 'session', ''))),
            sessions=self.sessions, n_splits=self.n_splits,
            seg_len_sec=self.seg_len_sec, discard_last=self.discard_last,
            overlap_raw=self.overlap[0], overlap_unit=self.overlap[1],
            split_mode='equal_effective' if self.equal_effective else 'raw_span',
            filter_state=self.filter_state())

    def saved_custom_ccgs(self, session: str = None, name: str = None) -> list:
        """Saved batch requests, newest first, optionally filtered by session or name."""
        specs = self.nav.root.custom_mgr.state.load_suggestions()
        if session is not None:
            specs = [s for s in specs if str(session) in (s.sessions or [s.scope])]
        if name is not None:
            specs = [s for s in specs if str(s.name) == str(name)]
        return sorted(specs, key=lambda s: str(s.name))

    def run_custom_ccg(self, request: CCGBatchRequest = None) -> int:
        """Queue *request* (or the one the current state describes); returns tasks queued."""
        if request is None:
            request = self.build_request(self.start, self.end)
        mgr = self.nav.root.custom_mgr
        n = mgr._queue_custom_ccgs(request)
        if n:
            mgr.worker._custom_ccg_start_next()
        return n

    def plot(self, path: str, *, start: float = None, end: float = None,
             pin_start: bool = False, pin_end: bool = False,
             px_w: int = 1200, px_h: int = 220, dpi: int = 100) -> str:
        """Write a static PNG of the epoch timeline with optional window markers."""
        fig = Figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        cmap = self.label_color_map()
        for s, e, lb in self.active_bounds:
            ax.axvspan(s, e, color=cmap.get(lb, _TS_NONE_COLOR), lw=0)
        for t, pin in ((start, pin_start), (end, pin_end)):
            if t is not None:
                ax.axvline(float(t), color='#C62828' if pin else '#1565C0',
                           lw=2.0 if pin else 1.2)
        ax.set_xlim(0.0, self.total_sec or 1.0)
        ax.set_yticks([])
        ax.set_xlabel('time (s)')
        ax.set_title(f"{self.current_session() or 'all sessions'} — {self.current_theme}")
        fig.tight_layout()
        fig.savefig(path)
        return path

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
        return f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}"


class TimeSliderPanel(QWidget):
    """Time slider panel; custom CCG work is emitted to the parent."""

    queue_ccg_requested = Signal(object)   # CCGSourceConfig
    load_requested        = Signal()
    window_changed        = Signal(float, float)
    theme_changed         = Signal()

    def __init__(self, nav: 'AppState', cd: 'CCGDataset', parent=None):
        super().__init__(parent)
        self.nav = nav
        self.cd  = cd
        self.backend = TimeSliderBackend(nav, cd)

        self._build()
        self._connect_nav()
        self._refresh_theme_ui(self.nav.cd.nd.get_themes(self.nav.key))

    def reload_themes(self):
        """Theme combo and bounds only (no timeline reset)."""
        self._discover_themes(self.nav.cd.nd.get_themes(self.nav.key))

    @property
    def active_bounds(self) -> list:
        """(start, stop, label) of the current theme's epochs whose label is toggled on."""
        return self.backend.active_bounds

    @property
    def theme_names(self) -> list:
        """Themes the slider has discovered, 'segments' first."""
        return self.backend.list_themes()

    def _refresh_theme_ui(self, themes: dict):
        """Refresh combo, bounds, timeline, and legend."""
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
        self._epoch_ctrls = QWidget()
        epoch_lay = QHBoxLayout(self._epoch_ctrls)
        epoch_lay.addWidget(QLabel("Theme:"))
        self._theme_combo = AddableDropdown('theme', self.add_theme)
        self._theme_combo.set_items(['segments'])
        self._theme_combo.setFixedWidth(140)
        self._theme_combo.currentTextChanged.connect(self._on_theme_change)
        epoch_lay.addWidget(self._theme_combo)
        self._theme_info_lbl = QLabel("")
        self._theme_info_lbl.setStyleSheet(f"color:#888; font-size:{small_font_pt()}pt;")
        epoch_lay.addWidget(self._theme_info_lbl)

        self._filter_check = chip_button("Include in filter", checked=False)
        self._filter_check.toggled.connect(self._on_filter_toggle)
        epoch_lay.addWidget(self._filter_check)
        epoch_lay.addSpacing(12)
        for text, width, slot in (("All", 40, self._on_label_reset), ("None", 40, self._on_label_none)):
            btn = QPushButton(text); btn.setFixedWidth(width); btn.clicked.connect(slot)
            epoch_lay.addWidget(btn)
        row1.addWidget(self._epoch_ctrls)

        self._custom_ctrls = QWidget()
        custom_lay = QHBoxLayout(self._custom_ctrls)
        custom_lay.addWidget(QLabel("Custom CCGs:"))
        self._custom_picker = ListPickerButton(
            "Custom CCGs", plural="custom CCGs", select_all_when_empty=False, ordered=True,
            refresh_provider=self.backend.custom_items)
        self._custom_picker.setFixedWidth(180)
        custom_lay.addWidget(self._custom_picker)
        custom_set_btn = QPushButton("Set"); custom_set_btn.clicked.connect(self._on_custom_set_btn)
        custom_lay.addWidget(custom_set_btn)
        family_btn = QPushButton("Save family…"); family_btn.clicked.connect(self._on_family_save_btn)
        custom_lay.addWidget(family_btn)
        row1.addWidget(self._custom_ctrls)
        row1.addStretch()

        self._mode_btn = QToolButton(); self._mode_btn.clicked.connect(self._on_mode_btn)
        row1.addWidget(self._mode_btn)
        tb = QToolButton(); tb.setText("📂"); tb.clicked.connect(self.load_requested)
        row1.addWidget(tb)
        sep_tb = QFrame(); sep_tb.setFrameShape(QFrame.VLine); sep_tb.setStyleSheet('color: #ccc;')
        row1.addWidget(sep_tb)
        self._snap_check = QCheckBox("Snap")
        self._snap_check.setChecked(True); self._snap_check.toggled.connect(self._on_snap_toggle)
        row1.addWidget(self._snap_check)
        row1.addWidget(QLabel("Reset:"))
        self._reset_zoom_btn = QPushButton("scale")
        self._reset_zoom_btn.setFixedWidth(50); self._reset_zoom_btn.clicked.connect(self._on_reset_zoom)
        row1.addWidget(self._reset_zoom_btn)
        self._reset_pins_btn = QPushButton("pins")
        self._reset_pins_btn.setFixedWidth(50); self._reset_pins_btn.clicked.connect(self._reset_handles)
        row1.addWidget(self._reset_pins_btn)
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
        self._main_plot.handle_moved.connect(self.window_changed)
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
        self._name_entry.textEdited.connect(self._on_name_entry)
        extra_lay.addWidget(self._name_entry)
        self._split_mode_group, self._split_mode_btns = radio_group(
            [('splits', "Splits:"), ('seg_len', "Segment length:")], 'splits',
            on_click=lambda *_: self._on_split_mode_btn())
        timing_row.addWidget(self._ccg_extra_widget)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color:#555; font-size:{small_font_pt()}pt;")
        timing_row.addWidget(self._status_lbl)
        timing_row.addStretch()

        split_row = QHBoxLayout()
        self._timing_section.body_layout.addLayout(split_row)
        split_row.addWidget(self._split_mode_btns['splits'])
        self._splits_spin = QSpinBox()
        self._splits_spin.setRange(1, 99); self._splits_spin.setValue(1); self._splits_spin.setFixedWidth(45)
        split_row.addWidget(self._splits_spin)
        split_row.addWidget(self._split_mode_btns['seg_len'])
        self._seg_len_metric = MetricInput(
            "", ('hr', 'sec', 'ms'), default="1",
            suggestions=(1, 2, 6), input_width=45, unit_width=55)
        split_row.addWidget(self._seg_len_metric)
        self._discard_last_check = QCheckBox("discard last")
        self._discard_last_check.setToolTip("Drop the short tail rather than keeping a partial segment")
        split_row.addWidget(self._discard_last_check)
        self._on_split_mode_btn()
        self._overlap_metric = MetricInput(
            "Overlap:", ('%', 'hr', 'min', 'sec'), default="0",
            suggestions=(0, 10, 25, 50), input_width=45, unit_width=60)
        split_row.addWidget(self._overlap_metric)
        self._equal_effective_check = QCheckBox("Equal duration")
        self._equal_effective_check.setToolTip(
            "Splits share equal effective (filtered) time; real-time edges may differ.")
        split_row.addWidget(self._equal_effective_check)
        split_row.addStretch()

        self._custom_table = QTableWidget(0, len(_CUSTOM_COLS))
        self._custom_table.setHorizontalHeaderLabels(_CUSTOM_COLS)
        self._custom_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._custom_table.verticalHeader().setVisible(False)
        self._custom_table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._custom_table)

        for lyt in (row1, epoch_lay, custom_lay, self._legend_layout, timing_row, extra_lay,
                    split_row):
            lyt.setContentsMargins(0, 0, 0, 0)
            lyt.setSpacing(4)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        for w in (title_lbl, row1_widget, self._legend_widget, self._any_mode_lbl,
                  self._timing_section):
            w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root.addStretch()
        self._apply_mode()

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
        sessions = [str(k.session) for k in self.nav.real_nd_keys()]
        self._sessions_picker.set_items(sessions)   # a project switch replaces the roster
        self._sessions_picker.set_selected(sessions if any_mode
                                           else [str(self.nav.key.session)])
        themes = (self.nav.cd.nd.get_themes_any() if any_mode
                  else self.nav.cd.nd.get_themes(self.nav.key))
        self._refresh_theme_ui(themes)

    def _discover_themes(self, themes: dict):
        b = self.backend
        default = b.discover_themes(themes)
        self._theme_combo.blockSignals(True)
        self._theme_combo.set_items(b.list_themes())
        self._theme_combo.setCurrentText(default)
        self._theme_combo.blockSignals(False)
        b.current_theme = default
        n = len(themes)
        self._theme_info_lbl.setText(f"{n} theme{'s' if n != 1 else ''}")

    def _init_times(self):
        b = self.backend
        b.set_theme(b.current_theme)

        # Initialise overlap from source config if available
        source = getattr(self.cd, 'source', None)
        if isinstance(source, CCGSourceConfig):
            self._overlap_metric.set_value(source.overlap_sec, 'sec')

        self._filter_check.blockSignals(True)
        self._filter_check.setChecked(b.filter_checks.get(b.current_theme, False))
        self._filter_check.blockSignals(False)
        self._sync_filter_check()
        self._update_legend()

    def restore_filters(self, state: dict) -> None:
        """Re-apply saved theme and filters, then resync combo, checkbox and legend to them."""
        self.backend.restore_filters(state)
        self._theme_combo.blockSignals(True)
        self._theme_combo.setCurrentText(self.backend.current_theme)
        self._theme_combo.blockSignals(False)
        self._init_times()
        self._apply_mode()

    def _on_mode_btn(self):
        self.backend.mode = 'custom' if self.backend.mode == 'epoch' else 'epoch'
        self._apply_mode()

    def _apply_mode(self):
        """Swap row-1 controls and the time-range section; neither view's state is touched."""
        custom = self.backend.mode == 'custom'
        icon, tip = _MODE_ICONS[self.backend.mode]
        self._mode_btn.setText(icon); self._mode_btn.setToolTip(tip)
        self._epoch_ctrls.setVisible(not custom)
        self._custom_ctrls.setVisible(custom)
        self._timing_section.setVisible(not custom)
        self._custom_table.setVisible(custom)
        if custom:
            self._custom_picker.set_items(self.backend.custom_items())
        self._update_legend()

    def _on_custom_set_btn(self):
        self.backend.set_custom(self._custom_picker.selected)
        self._update_legend()

    def _remove_custom_chip(self, seg: str):
        self.backend.custom_selected.remove(seg)
        self._custom_picker.set_selected(self.backend.custom_selected)
        self._update_legend()

    def _on_family_save_btn(self):
        name = prompt_name(self, "Save family", "Family name:")
        if name:
            self.cd.save_custom_family(name, self.backend.custom_selected)
            self._custom_picker.set_items(self.backend.custom_items())

    def _fill_custom_table(self):
        wins = self.backend.custom_windows()
        rows = [s for s in self.backend.custom_selected if s in wins]
        self._custom_table.setRowCount(len(rows))
        for r, seg in enumerate(rows):
            cfg = wins[seg][1]
            filt = "; ".join(f"{th['name']}: {', '.join(th.get('labels') or [])}"
                             for th in cfg.get('filter_state') or [])
            t0, t1 = (self.cd.nd.resolve_time(self.nav.key, cfg[k]) for k in ('t0', 't1'))
            vals = (seg, self._sec_to_hms(t0), self._sec_to_hms(t1),
                    self._sec_to_hms(cfg.get('active_duration') or 0.0), filt,
                    f"{cfg.get('overlap_sec', 0.0):g} s")
            for c, v in enumerate(vals):
                self._custom_table.setItem(r, c, QTableWidgetItem(v))
        self._custom_table.resizeColumnsToContents()

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

        if self.backend.mode == 'custom':
            wins = self.backend.custom_windows()
            for seg, color in self.backend.custom_color_map().items():
                if seg in wins:
                    chip = self._add_legend_chip(seg, color, self.backend.custom_visible.get(seg, True))
                    chip.mouseDoubleClickEvent = lambda _e, s=seg: self._remove_custom_chip(s)
            self._fill_custom_table()
            self._redraw_main()
            return
        cmap = self.backend.label_color_map()
        toggles = self.backend.rebuild_legend()
        for lbl, color in cmap.items():
            self._add_legend_chip(lbl, color, toggles[lbl])
        self._add_legend_chip('NONE', _TS_NONE_COLOR, toggles['NONE'], none_style=True)

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
        return chip

    def _on_legend_toggle(self, label: str, active: bool):
        if self.backend.mode == 'custom':
            self.backend.custom_visible[label] = active
            self._redraw_main()
            return
        self.backend.set_label(label, active)
        self._sync_name_to_labels()
        self._redraw_main()
        self.theme_changed.emit()

    def _sync_name_to_labels(self):
        """Name mirrors a lone selected label; clears when that stops holding (typed names kept)."""
        if self._name_entry.text().strip() and not self.backend.name_is_auto:
            return
        self._name_entry.setText(self.backend.auto_name())
        self.backend.name_is_auto = True

    def _redraw_main(self):
        if self.nav.session_any_mode:
            self._main_plot.update_epochs([], {}, 0, 1)
            return
        if not self.backend.epoch_bounds and self.backend.mode == 'epoch':
            return
        overlays = self.backend.custom_overlays() if self.backend.mode == 'custom' else ()
        self._main_plot.update_epochs(self.active_bounds, self.backend.label_color_map(),
                                      0.0, self.backend.total_sec, overlays)

    def add_theme(self):
        """Add an epoch theme: pick source + format, attach to the session, add to the combo."""
        pass

    def _on_name_entry(self, _text: str):
        self.backend.name_is_auto = False

    def _on_theme_change(self, theme: str):
        b = self.backend
        if theme == b.current_theme:
            return
        if self._theme_combo.is_add_row(self._theme_combo.currentIndex()):
            return   # AddableDropdown reverts the index and calls add_theme
        b.filter_checks[b.current_theme] = self._filter_check.isChecked()
        b.current_theme = theme
        self._init_times()
        self._filter_check.blockSignals(True)
        self._filter_check.setChecked(b.filter_checks.get(theme, False))
        self._filter_check.blockSignals(False)
        self._sync_filter_check()
        self.theme_changed.emit()

    def _on_label_reset(self):
        self.backend.per_theme_label_state.pop(self.backend.current_theme, None)
        self._update_legend()

    def _on_label_none(self):
        off = {lb: False for lb in {lb for _, _, lb in self.backend.epoch_bounds}}
        off['NONE'] = False
        self.backend.per_theme_label_state[self.backend.current_theme] = off
        self._update_legend()

    def _on_filter_toggle(self, checked: bool):
        self.backend.filter_checks[self.backend.current_theme] = checked
        self._sync_filter_check()

    def _sync_filter_check(self):
        """Label follows filter_checks, never the signal — blockSignals must not let it lie."""
        on = self.backend.filter_checks.get(self.backend.current_theme, False)
        self._filter_check.setText("✓ Include in filter" if on else "Include in filter")

    def _parse_time_text(self, text: str) -> float:
        return self.backend.parse_time(text)

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
        return (t0_spec, t1_spec, self._splits_spin.value(), *self._overlap_metric.value(),
                self._seg_len_sec(), self._discard_last_check.isChecked())

    _SEG_LEN_UNITS = {'hr': 3600.0, 'sec': 1.0, 'ms': 0.001}

    def _on_split_mode_btn(self):
        """Only the picked sizing control stays live; the other cannot contribute."""
        by_len = self._split_mode_btns['seg_len'].isChecked()
        self._splits_spin.setEnabled(not by_len)
        self._seg_len_metric.setEnabled(by_len)
        self._discard_last_check.setEnabled(by_len)

    def _seg_len_sec(self) -> float:
        if not self._split_mode_btns['seg_len'].isChecked():
            return 0.0
        raw, unit = self._seg_len_metric.value()
        return float(raw) * self._SEG_LEN_UNITS[unit]

    def _theme_whitelist(self, theme: str) -> list:
        return self.backend.theme_whitelist(theme)

    def _on_set(self):
        b = self.backend
        if (self._name_entry.text().strip() or 'custom').lower() == _FULL_SEG:
            QMessageBox.warning(None, "Custom CCG",
                                f"'{_FULL_SEG}' is a reserved name — choose another.")
            return
        timing = self._read_timing(self.nav.session_any_mode)
        if timing is None:
            self._status_lbl.setText("Check the start/end times — end must follow start")
            return
        (t0_spec, t1_spec, b.n_splits, overlap_raw, overlap_unit,
         b.seg_len_sec, b.discard_last) = timing
        b.overlap = (overlap_raw, overlap_unit)
        b.name = self._name_entry.text()
        b.sessions = self._sessions_picker.selected
        b.equal_effective = self._equal_effective_check.isChecked()
        request = b.build_request(t0_spec, t1_spec)
        self._status_lbl.setText(f"Queued: {request.name}")
        self.queue_ccg_requested.emit(request)

    def _on_clear(self):
        self._reset_handles()
        self._name_entry.clear()
        self.backend.name_is_auto = True   # hand the name back to the chips
        self._sync_name_to_labels()
        self._status_lbl.setText("")

    _hms_to_sec = staticmethod(TimeSliderBackend._hms_to_sec)
    _sec_to_hms = staticmethod(TimeSliderBackend._sec_to_hms)


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
    extend: dict | None = None      # {name, window_ms, bin_ms}: an extend unit, not a segment
    aux: dict | None = None         # {segment, tests, overrides, pairs}: an aux-rule sweep

    @property
    def whole_session(self) -> bool:
        return self.spec is None and self.extend is None and self.aux is None

    def ccg_key(self) -> Key:
        base = self.session_key if self.spec is None else self.spec.key
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
                     session_key: Key = None, extend: dict = None, aux: dict = None) -> bool:
        """Queue a segment compute (*spec*), a whole-session one, an extend unit, or an aux sweep."""
        task = CCGTask(spec=spec, load_into_ui=bool(load_into_ui),
                       batch_id=batch_id, resolution=resolution,
                       session_key=session_key, extend=extend, aux=aux)
        return self._runner.enqueue(task)

    def on_done(self, completed_task, _result):
        ui, mgr = self._ui, self._mgr
        r: CCGTaskResult = self._thread_result.pop() if self._thread_result else None
        name = ('whole session' if completed_task.whole_session
                else f"aux {completed_task.aux['segment']}" if completed_task.aux
                else f"extend {completed_task.extend['name']}" if completed_task.extend
                else completed_task.spec.name)
        print(f"[CCGq] on_done {name} {completed_task.resolution} "
              f"result={'none' if r is None else ('ok' if r.ok else r.error)}", flush=True)
        bid = completed_task.batch_id
        meta = ui.time_slider.backend.batch_meta.get(bid) if bid is not None else None

        if r is None or not r.ok:
            err = r.error if r else 'unknown error'
            if meta is not None:
                meta['rows'].append(
                    (r.session if r else '?', name, 'fail', err))
            else:
                QMessageBox.critical(None, "Custom CCG",
                    f"Computation failed:\n{err}")
        elif completed_task.aux is not None:
            if meta is not None:
                meta['rows'].append((r.session, name, 'ok', r.value))
            if r.session == str(ui.nav.key.session):
                ui.nav.chip_colors_changed.emit()
        elif completed_task.whole_session or completed_task.extend is not None:
            # both wrote their own storage: dim0[0] via get_ccg, or an extend unit dir.
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

            name = ('whole session' if task.whole_session
                    else f"aux {task.aux['segment']}" if task.aux
                    else f"extend {task.extend['name']}" if task.extend
                    else str(task.spec.name))

            def _ccg_worker():
                sess = str(seg_key.session)
                import time as _t; _t0 = _t.time()
                print(f"[CCGq] START {sess} {task.resolution} '{name}'", flush=True)
                try:
                    if task.aux is not None:
                        a = task.aux
                        results, on = nav.cd.sweep_aux_tests(
                            task.session_key.change(segment=a['segment']),
                            tests=a['tests'], overrides=a['overrides'])
                        store = nav.sd.aux_results(task.session_key, a['segment'])
                        store.set_results(results, on)
                        store.save()
                        idx = tuple(np.array(a['pairs']).T)
                        tally = f"{int(on[idx].sum())}/{len(a['pairs'])} on ({task.session_key.type_label()})"
                        self._thread_result.append(CCGTaskResult(tally, None, sess))
                        return
                    if task.extend is not None:
                        e = task.extend
                        nav.cd.compute_extend(e['name'], task.session_key.change(segment=e['segment']),
                                              e['window_ms'], e['bin_ms'], e['pairs'])
                        print(f"[CCGq] DONE  {sess} extend {_t.time()-_t0:.1f}s", flush=True)
                        self._thread_result.append(CCGTaskResult(None, None, sess))
                        return
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
        if bid not in ts.backend.batch_counts:
            return
        ts.backend.batch_counts[bid] -= 1
        total = ts.backend.batch_totals.get(bid, 0)
        if ts.backend.batch_counts[bid] > 0:
            done = total - ts.backend.batch_counts[bid]
            ts._status_lbl.setText(f"Computing custom CCG… {done}/{total}")
            # A panel that queued this batch may own the screen while it runs, so
            # the slider's own label is not the only place progress has to land.
            on_progress = (ts.backend.batch_meta.get(bid) or {}).get('on_progress')
            if on_progress is not None:
                on_progress(done, total)
            return
        del ts.backend.batch_counts[bid]
        ts.backend.batch_totals.pop(bid, None)
        self._ui.nav.cd.nd.clear_slice_cache()
        spec_name = (ts.backend.batch_meta.get(bid) or {}).get('spec_name', '')
        ts._status_lbl.setText(f"Done: {spec_name} — {total} CCG(s)")
        names = list(ts.backend.batch_names.pop(bid, []))
        meta = ts.backend.batch_meta.pop(bid, None)
        if meta is not None:
            if meta.get('on_done') is not None:
                failed = [s for s, _n, st, _v in meta.get('rows', []) if st == 'fail']
                QTimer.singleShot(0, lambda f=failed: meta['on_done'](f))
            QTimer.singleShot(100, lambda m=meta: self._show_batch_report(m))
        QTimer.singleShot(100, lambda n=names: self._prompt_save_chunks(n))

    def _show_batch_report(self, meta: dict):
        rows, skipped = meta.get('rows', []), meta.get('skipped', [])
        line = lambda s, n, v: f"  {s}  {n}" + (f"  ->  {v}" if v not in (None, s) else "")
        ok   = [line(s, n, v) for s, n, st, v in rows if st == 'ok']
        fail = [line(s, n, v) for s, n, st, v in rows if st == 'fail']
        skip = [f"  {s}: {w}" for s, w in skipped]
        n_sess = len({r[0] for r in rows} | {s for s, _ in skipped})
        lines = [f"Batch: {meta.get('spec_name', '')}",
                 f"Sessions: {n_sess}   done {len(ok)} · failed {len(fail)} · skipped {len(skip)}"]
        for title, items in (("Done", ok), ("Failed", fail), ("Skipped", skip)):
            lines += ["", f"{title} ({len(items)}):", *(items or ["  (none)"])]
        ResultsDialog.show_report("Custom CCG queue", "\n".join(lines))

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

    def queue_whole_session(self, sessions: list, resolution: str, on_done=None,
                            on_progress=None) -> int:
        """Queue whole-session CCG computes; ``on_done(failed_sessions)`` when all land.

        Shares the batch counter and status label with segment computes, so the
        queue reports one x/total regardless of what kind of work is in it.
        ``on_progress(done, total)`` fires per completion, for a caller whose own
        panel is covering the slider's status label.
        """
        ts = self._ui.time_slider
        bid = ts.backend.batch_next_id
        ts.backend.batch_next_id += 1
        queued = 0
        for sess in sessions:
            if self.worker.enqueue_task(session_key=Key(session=str(sess)),
                                        batch_id=bid, resolution=resolution):
                queued += 1
        if queued:
            ts.backend.batch_counts[bid] = queued
            ts.backend.batch_totals[bid] = queued
            ts.backend.batch_meta[bid] = {'spec_name': f'{resolution} CCG',
                                   'skipped': [], 'rows': [], 'on_done': on_done,
                                   'on_progress': on_progress}
            self.worker._custom_ccg_start_next()
        return queued

    def queue_extend(self, name: str, jobs: list, window_ms: float, bin_ms: float,
                     skipped: list = ()) -> int:
        """Queue an extend per ``(ptr_key, segment, pairs)`` job on the custom-CCG queue and counter."""
        ts = self._ui.time_slider
        bid = ts.backend.batch_next_id
        ts.backend.batch_next_id += 1
        queued = 0
        for key, seg, pairs in jobs:
            if self.worker.enqueue_task(
                    session_key=key, batch_id=bid,
                    extend={'name': name, 'window_ms': window_ms, 'bin_ms': bin_ms,
                            'segment': seg, 'pairs': pairs}):
                queued += 1
        if queued:
            ts.backend.batch_counts[bid] = queued
            ts.backend.batch_totals[bid] = queued
            ts.backend.batch_meta[bid] = {'spec_name': f"extend {name}",
                                          'skipped': list(skipped), 'rows': []}
            self.worker._custom_ccg_start_next()
        return queued

    def queue_aux(self, jobs: list, tests: dict, skipped: list, on_done=None,
                  on_progress=None) -> int:
        """Queue aux-rule sweeps, one per ``(ptr_key, segment, overrides, pairs)``, on the custom-CCG queue."""
        ts = self._ui.time_slider
        bid = ts.backend.batch_next_id
        ts.backend.batch_next_id += 1
        queued = 0
        for key, seg, overrides, pairs in jobs:
            if self.worker.enqueue_task(session_key=key, batch_id=bid,
                                        aux=dict(segment=seg, tests=tests,
                                                 overrides=overrides, pairs=pairs)):
                queued += 1
        if queued:
            ts.backend.batch_counts[bid] = queued
            ts.backend.batch_totals[bid] = queued
            meta = ts.backend.batch_meta[bid] = {
                'spec_name': f"aux rules: {', '.join(tests)}", 'skipped': skipped, 'rows': [],
                'on_done': on_done}
            if on_progress is not None:
                meta['on_progress'] = lambda *_: on_progress(meta['rows'])
            self.worker._custom_ccg_start_next()
        return queued

    def _generate_suggested_custom_ccgs(self):
        self.state.show_dialog()

    def _queue_custom_ccgs(self, spec: 'CCGBatchRequest') -> int:
        nav = self._ui.nav
        _any = nav.session_any_mode
        ts = self._ui.time_slider
        bid = ts.backend.batch_next_id
        ts.backend.batch_next_id += 1
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
            ts.backend.batch_counts[bid] = queued
            ts.backend.batch_totals[bid] = queued
            ts.backend.batch_names[bid] = split_names
            ts.backend.batch_meta[bid] = {'spec_name': str(spec.name),
                                   'skipped': skipped, 'rows': []}
        else:
            self.worker._show_batch_report({'spec_name': str(spec.name),
                                            'skipped': skipped or [('(all sessions)', 'no session matched the scope')],
                                            'rows': []})
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
