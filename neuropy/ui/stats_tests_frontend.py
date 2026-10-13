"""Stats-test panel widgets: the row, the plot canvas and the panel that drives them.

Widgets are the source of truth; configs are snapshots taken at run/save.
"""
from __future__ import annotations

import dataclasses
import datetime
from functools import partial
import pathlib
from typing import TYPE_CHECKING

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt.QtCore import QByteArray, QEvent, QObject, QPoint, Qt, QThread, QTimer, Signal
from pyqtgraph.Qt.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QPlainTextEdit,
    QScrollArea, QFrame, QMessageBox, QSizePolicy, QTabWidget, QComboBox, QSpinBox,
    QDockWidget, QMainWindow,
)
from neuropy.ui.utils import (ALL_OPTION, ConfigBound, ListPickerButton, ScopeField, ScopePicker, make_combo, make_button,
                              ColorLabelButton, chip_button, sync_follow_column, widget_row)
from neuropy.ui.ui_common import qt_dark_mode, UITheme
from neuropy.ui.dialogs import RulePickerButton, VersionSaveDialog, VersionLoadDialog, pair_scope_fields
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.pair_selection_data import DerivedRule
from neuropy.ui.app_state import FAMILY_PREFIX, DisplayConfig
from neuropy.ui.ccg_panel import NormSection, BaselineCSSection
from neuropy.ui.stats_tests_backend import (
    _BAR_COLORS, DISABLED_METRICS, METRICS, PICKERS, PICKER_FIELD, SINGLE_PICKERS,
    RowConfig, StatsTestBackend, StatsTestConfig, StatsResult, _ViewConfig)
from neuropy.ui.stats_tests_plot import draw_stats_figure, fit_figure_rect

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState


# ─────────────────────────── plot widget (frontend) ───────────────────────────

class StatsPlotWidget(QWidget, ConfigBound):
    """Owns the matplotlib figure, the view toggles, and all plot rendering."""

    rerun_requested = Signal()   # a toggle that changes the test, not just the view

    _CONFIG = _ViewConfig
    _BIND = {
        'violin':       (lambda w: w._violin_check.isChecked(),   lambda w, v: w._violin_check.setChecked(v)),
        'outliers':     (lambda w: w._outliers_check.isChecked(), lambda w, v: w._outliers_check.setChecked(v)),
        'sig_brackets': (lambda w: w._sig_check.isChecked(),      lambda w, v: w._sig_check.setChecked(v)),
        'show_n':       (lambda w: w._n_check.isChecked(),        lambda w, v: w._n_check.setChecked(v)),
        'wh_ratio':     (lambda w: w._wh_input.text().strip(),    lambda w, v: w._wh_input.setText(v)),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[StatsResult] = []
        self.test_config: StatsTestConfig | None = None

        pc = QVBoxLayout(self)
        pc.setContentsMargins(0, 0, 0, 0)
        ctrl = QHBoxLayout()
        self._violin_check   = QCheckBox("Violin")
        self._outliers_check = QCheckBox("Show outliers")
        self._sig_check      = QCheckBox("Sig. brackets")
        self._n_check        = QCheckBox("n in title")
        self._outliers_check.setChecked(True)
        self._n_check.setChecked(True)
        for chk in (self._violin_check, self._outliers_check, self._sig_check, self._n_check):
            chk.toggled.connect(self._replot)
            ctrl.addWidget(chk)
        self._rm_outliers_check = QCheckBox("Remove outliers")
        self._rm_outliers_check.setToolTip("Exclude >3 SD pairs and re-run the test")
        self._rm_outliers_check.toggled.connect(self.rerun_requested)
        ctrl.addWidget(self._rm_outliers_check)
        ctrl.addWidget(QLabel("W:H"))
        self._wh_input = QLineEdit("3:1")
        self._wh_input.setFixedWidth(45)
        self._wh_input.editingFinished.connect(self._replot)
        ctrl.addWidget(self._wh_input)
        ctrl.addStretch()
        pc.addLayout(ctrl)

        self._fig = Figure(dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _canvas_resize = self._canvas.resizeEvent
        self._canvas.resizeEvent = lambda ev: (self.refresh_plot(), _canvas_resize(ev))
        pc.addWidget(self._canvas, stretch=1)

    # -- config --------------------------------------------------------------

    # -- render --------------------------------------------------------------
    def render(self, results: list[StatsResult]):
        self._results = results
        self._update_plot()

    def _replot(self):
        if self._results:
            self._update_plot()

    def apply_theme(self):
        """Repaint the matplotlib canvas for the current light/dark palette."""
        self._fig.patch.set_facecolor(UITheme.from_dark(qt_dark_mode()).plot_bg)
        if self._results:
            self._update_plot()
        else:
            self._canvas.draw_idle()

    def refresh_plot(self):
        fit_figure_rect(self._fig, self._canvas.width(), self._canvas.height(),
                        self._wh_input.text())
        if self._fig.axes:
            self._canvas.draw_idle()

    def _update_plot(self):
        fit_figure_rect(self._fig, self._canvas.width(), self._canvas.height(),
                        self._wh_input.text())
        draw_stats_figure(self._fig, self._results, self.config,
                          self.test_config, dark=qt_dark_mode())
        self._canvas.draw()


class PairLinesPlot(QWidget):
    """One line per pair across the rows, from the values the test ran on; a zero breaks the line, a lone point is enlarged."""

    _GRAY = (150, 150, 150)

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__(parent)
        self.nav = nav
        self._results: list[StatsResult] = []
        self._pairs, self._curves, self._colors, self._attrs = [], [], [], []
        self._y = np.zeros((0, 0))
        self._drawn = self._off = np.zeros((0, 0), dtype=bool)
        self._members: dict = {}
        self._res_combo = QComboBox()
        self._res_combo.currentIndexChanged.connect(lambda _i: self._draw())
        self._scope = ScopePicker(pair_scope_fields(nav) + [
            ScopeField('nrn', "Neurons", "neurons", self._all_neurons, select_all_when_empty=False,
                       all_option=True, display=lambda n: tuple(n.split(' · ')) if ' · ' in n else ('', n))])
        self._scope.setToolTip("Highlight the pairs matching every pick; Segments picks the columns shown")
        self._scope.selection_changed.connect(self._draw)
        self._on_lo_spin, self._on_hi_spin = QSpinBox(), QSpinBox()
        for spin in (self._on_lo_spin, self._on_hi_spin):
            spin.valueChanged.connect(lambda _v: self._draw())
        self._unthr_check = QCheckBox("unthresholded")
        self._unthr_check.setToolTip("Off segments at their raw CS, as gray dots the line runs through")
        self._unthr_check.toggled.connect(lambda _on: self._draw())
        self._info = QLabel("")
        clear_btn = make_button("Clear selection", self._on_clear_btn)
        clear_btn.setToolTip("Drop the picked and clicked highlights")
        self._plot = pg.PlotWidget()
        self._points = pg.ScatterPlotItem(pen=None)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(widget_row("Resolution:", self._res_combo, self._scope, clear_btn))
        lay.addLayout(widget_row("On in", self._on_lo_spin, "–", self._on_hi_spin, "segments",
                                 self._unthr_check, self._info))
        lay.addWidget(self._plot, stretch=1)

    def _all_neurons(self) -> list:
        nd = self.nav.cd.nd
        return [f"{s} · {i}" for s in sorted({str(k.session) for k in self.nav.cd.ptr})
                for i in nd.neurons_for(Key(session=s)).neuron_ids]

    def render(self, results: list[StatsResult]) -> None:
        self._results = results
        self._scope.refresh()
        self._res_combo.blockSignals(True)
        self._res_combo.clear()
        self._res_combo.addItems([sr.resolution for sr in results])
        self._res_combo.blockSignals(False)
        groups = results[0].groups if results else []
        for spin in (self._on_lo_spin, self._on_hi_spin):
            spin.blockSignals(True)
            spin.setRange(0, len(groups))
        self._on_lo_spin.setValue(0)
        self._on_hi_spin.setValue(len(groups))
        for spin in (self._on_lo_spin, self._on_hi_spin):
            spin.blockSignals(False)
        self._index_pairs()
        self._draw()

    def _index_pairs(self) -> None:
        """Each test pair's session, conn type and neurons, the groups holding it, and what the picker can offer."""
        cd, sd = self.nav.cd, self.nav.sd
        self._pairs = list(dict.fromkeys(p for sr in self._results for g in sr.groups for p in g['pairs']))
        ptr_of = {}
        for label in cd.conf.conn_type_labels:
            ei, ct = cd.conf.parse_conn_type_label(label)
            for s in {str(p.session) for p in self._pairs}:
                if (k := Key(session=s, excitability=ei, conn_type=ct)) in cd.ptr:
                    for ref, tgt in cd.ptr[k].pair_set:
                        ptr_of.setdefault((s, int(ref), int(tgt)), (label, k))
        self._attrs = []
        for p in self._pairs:
            label, key = ptr_of[(str(p.session), int(p.ref), int(p.tgt))]
            self._attrs.append(dict(sess=str(p.session), ct=label, key=key,
                                    nrn={f"{p.session} · {i}" for i in cd.nd.ids_at(p)}))
        keys = {a['key'] for a in self._attrs}
        self._members = {g: {i for i, a in enumerate(self._attrs)
                             if (int(self._pairs[i].ref), int(self._pairs[i].tgt)) in held.get(a['key'], ())}
                         for g in sd.groups.pickable_groups()
                         for held in [{k: sd.pairs_for_group_key(g, k) for k in keys}]}
        segs = {c['segment'] for sr in self._results for g in sr.groups for c in g.get('seg_configs') or []}
        in_range = dict(seg=segs, sess={a['sess'] for a in self._attrs}, ct={a['ct'] for a in self._attrs},
                        grp={g for g, m in self._members.items() if m},
                        nrn=set().union(*(a['nrn'] for a in self._attrs)))
        for k, picker in self._scope.pickers.items():
            picker.set_disabled(set(self._scope._items_of(self._scope._fields[k])()) - in_range[k] - {ALL_OPTION})

    def _picked(self, key: str) -> set | None:
        """A field's picks; None when it constrains nothing."""
        sel = self._scope.pickers[key].selected
        return None if not sel or ALL_OPTION in sel else set(sel)

    def _hits(self, shown: list) -> set:
        """Shown pairs (row indices into *shown*) matching every constraining pick."""
        picks = {k: self._picked(k) for k in ('sess', 'ct', 'grp', 'nrn')}
        if all(v is None for v in picks.values()):
            return set()
        grp = set().union(*(self._members.get(g, set()) for g in picks['grp'])) if picks['grp'] else None
        return {r for r, i in enumerate(shown)
                if (picks['sess'] is None or self._attrs[i]['sess'] in picks['sess'])
                and (picks['ct'] is None or self._attrs[i]['ct'] in picks['ct'])
                and (grp is None or i in grp)
                and (picks['nrn'] is None or self._attrs[i]['nrn'] & picks['nrn'])}

    def _table(self, groups: list, field: str) -> np.ndarray:
        """(pair, row) values of one per-pair field; NaN where a row lacks the pair."""
        row = {p: i for i, p in enumerate(self._pairs)}
        out = np.full((len(self._pairs), len(groups)), np.nan)
        for j, g in enumerate(groups):
            for p, v in zip(g['pairs'], g[field]):
                out[row[p], j] = v
        return out

    def _draw(self) -> None:
        plot = self._plot
        plot.clear()
        plot.setBackground(UITheme.from_dark(qt_dark_mode()).plot_bg)
        if not self._results:
            return
        sr = self._results[self._res_combo.currentIndex()]
        segs = self._picked('seg')
        cols = [j for j, g in enumerate(sr.groups) if segs is None or g['segment'] in segs]
        groups = [sr.groups[j] for j in cols]   # what the test ran on: outliers already dropped
        has_on = all('on' in g for g in groups)
        for w in (self._on_lo_spin, self._on_hi_spin, self._unthr_check):
            w.setEnabled(has_on)
        raw = self._table(groups, 'vals')
        vals = self._table(StatsTestBackend.log_rows(groups, sr.log_shift), 'vals')
        present = np.isfinite(raw)
        on = self._table(groups, 'on') == 1 if has_on else present
        n_on = on.sum(axis=1) if has_on else np.zeros(len(self._pairs))
        keep = ((n_on >= self._on_lo_spin.value()) & (n_on <= self._on_hi_spin.value())) if has_on \
            else np.ones(len(self._pairs), dtype=bool)
        keep &= present.any(axis=1)
        shown = list(np.flatnonzero(keep))
        if has_on and self._unthr_check.isChecked():
            unthr = self._table(StatsTestBackend.log_rows([dict(g, vals=g['raw']) for g in groups], sr.log_shift), 'vals')
            self._y = np.where(on, vals, unthr)[keep]
            self._drawn = np.isfinite(self._y)
            self._off = self._drawn & ~on[keep]
        else:
            self._y = vals[keep]
            self._drawn = present[keep] & (raw[keep] != 0)
            self._off = np.zeros_like(self._drawn)
        self._shown = shown
        plot.setLabel('left', 'value' if sr.log_shift is None else f"log(v + {sr.log_shift:.4g})")
        plot.getAxis('bottom').setTicks([[(x, g.get('name') or chr(65 + j))
                                          for x, (j, g) in enumerate(zip(cols, groups))]])
        rng = np.random.default_rng(0)   # seeded: a pair keeps its colour across redraws
        self._colors = [pg.hsvColor(h, 0.8, 0.85).getRgb()[:3] for h in rng.random(len(self._pairs))]
        x = np.arange(len(groups), dtype=float)
        self._curves = []
        for r, i in enumerate(shown):
            curve = pg.PlotCurveItem(x, np.where(self._drawn[r], self._y[r], np.nan), connect='finite')
            curve.setClickable(True, width=6)
            curve.sigClicked.connect(partial(self._on_curve_clicked, r))
            plot.addItem(curve)
            self._curves.append(curve)
        plot.addItem(self._points)
        hits = self._hits(shown)
        self._highlight(hits)
        self._info.setText(f"{len(shown)} pair(s)" + (f", {len(hits)} highlighted" if hits else ""))

    def _lone(self) -> np.ndarray:
        """Drawn points whose neighbours are both undrawn: no line reaches them."""
        pad = np.pad(self._drawn, ((0, 0), (1, 1)))
        return self._drawn & ~pad[:, :-2] & ~pad[:, 2:]

    def _rgba(self, r: int, hit: set, gray: bool = False) -> tuple:
        """A shown pair's colour: solid when highlighted, faint while others are."""
        rgb = self._GRAY if gray else self._colors[self._shown[r]]
        return (*rgb, 255 if r in hit else 40 if hit else 150)

    def _highlight(self, hit: set) -> None:
        for r, curve in enumerate(self._curves):
            curve.setPen(pg.mkPen(self._rgba(r, hit), width=3 if r in hit else 1))
            curve.setZValue(10 if r in hit else 0)
        lone = self._lone()
        rr, jj = np.nonzero(self._drawn)
        self._points.setData(x=jj.astype(float), y=self._y[rr, jj],
                             size=np.where(lone[rr, jj], 11, 4),
                             brush=[pg.mkBrush(self._rgba(r, hit, self._off[r, j])) for r, j in zip(rr, jj)])
        self._points.setZValue(20)

    def _on_clear_btn(self) -> None:
        for picker in self._scope.pickers.values():
            picker.set_selected([ALL_OPTION])
        self._draw()

    def _on_curve_clicked(self, r: int, *_):
        p = self._pairs[self._shown[r]]
        ref, tgt = self.nav.cd.nd.ids_at(p)
        self._highlight({r})
        self._info.setText(f"{p.session} {ref}→{tgt}")


# ─────────────────────────── row widget (frontend) ───────────────────────────

def _segment_display(item: str) -> tuple:
    """Families above plain segments in the picker."""
    return ('Families', item) if item.startswith(FAMILY_PREFIX) else ('Segments', item)


class StatsRow(QWidget, ConfigBound):
    """One group row: color, name, four multi-select pickers, data-type combo, delete.
    Owns its widgets and maps them to/from a RowConfig."""

    deleted = Signal(object)
    follow_toggled = Signal()
    family_picked = Signal(object, list)   # (row, member segments in order)

    # RowConfig field -> (getter reading the widget, setter writing the widget);
    # the picker fields are generated, so PICKERS stays the only list of them
    _CONFIG = RowConfig
    _BIND = {
        'name':       (lambda r: r._swatch.name,                lambda r, v: r._swatch.set_name(v)),
        'color':      (lambda r: r._swatch.color,               lambda r, v: r._swatch.set_color(v)),
        'data_type':  (lambda r: (r._pickers['data'].selected or [''])[0],
                       lambda r, v: r._pickers['data'].set_selected([v] if v else [])),
        'id':         (lambda r: r.row_id,                      lambda r, v: setattr(r, 'row_id', v)),
        # the widget follows the row above it; the id is filled in by the panel, which knows the order
        'follow_pickers': (lambda r: [k for k, p in r._pickers.items() if p.following],
                           lambda r, v: [p.set_following(k in (v or []))
                                         for k, p in r._pickers.items()]),
        **{fld: (lambda r, k=k: r._pickers[k].selected,
                 lambda r, v, k=k: r._pickers[k].set_selected(v))
           for k, fld in PICKER_FIELD.items() if k not in SINGLE_PICKERS},
    }

    def __init__(self, nav: 'AppState', backend: 'StatsTestBackend',
                 cfg: RowConfig | None = None, idx: int = 0, parent=None):
        super().__init__(parent)
        self.nav = nav
        self.backend = backend
        self.row_id = 0
        rw = QHBoxLayout(self)
        rw.setContentsMargins(0, 0, 0, 0)
        rw.setSpacing(4)

        self._swatch = ColorLabelButton(
            cfg.color if cfg and cfg.color else _BAR_COLORS[idx % len(_BAR_COLORS)],
            cfg.name if cfg and cfg.name else (chr(65 + idx) if idx < 26 else f"G{idx+1}"),
            editable=True, name_width=42)
        rw.addWidget(self._swatch)

        self._scope = ScopePicker(
            [ScopeField(rkey, label, plural, getattr(backend, prov_attr),
                        single=rkey in SINGLE_PICKERS, followable=True,
                        disabled=DISABLED_METRICS if rkey in SINGLE_PICKERS else (),
                        picker=partial(RulePickerButton, nav) if rkey == 'grp' else None,
                        display=_segment_display if rkey == 'seg' else None)
             for rkey, (_fld, label, plural, prov_attr) in PICKERS.items()], labelled=False)
        self._pickers = self._scope.pickers
        for p in self._pickers.values():
            p.follow_toggled.connect(self.follow_toggled)
        self._pickers['seg'].selection_changed.connect(self._on_seg_picker)
        rw.addWidget(self._scope)

        del_btn = QPushButton("x")
        del_btn.setFixedWidth(22)
        del_btn.clicked.connect(lambda: self.deleted.emit(self))
        rw.addWidget(del_btn)

        if cfg is not None:
            self.apply(cfg)
        else:
            self._seed_defaults()

    # -- config --------------------------------------------------------------

    def refresh(self):
        self._scope.refresh()

    def _on_seg_picker(self, picked: list):
        """A picked family hands its members to the panel, which deals them out from this row down."""
        fam = next((p for p in picked if p.startswith(FAMILY_PREFIX)), None)
        if fam is not None:
            self.family_picked.emit(self, self.backend.segment_families()[fam.removeprefix(FAMILY_PREFIX)])

    def set_segment(self, segment: str):
        self._pickers['seg'].set_selected([segment])
        self._swatch.set_name(segment)

    def _seed_defaults(self):
        """New row: default to current session + conn-type + first real group."""
        key_sess = str(self.nav.key.session)
        cur_ct   = self.nav.key.conn_type
        ct_lbl   = f"{cur_ct[0]}-{cur_ct[1]}" if cur_ct else None
        grp_opts = self.backend.available_groups()
        if key_sess in self._pickers['sess']._items:
            self._pickers['sess'].set_selected([key_sess])
        if ct_lbl and ct_lbl in self._pickers['ct']._items:
            self._pickers['ct'].set_selected([ct_lbl])
        if grp_opts:
            self._pickers['grp'].set_selected([DerivedRule(values=[grp_opts[0]])])
        self._pickers['data'].set_selected([next(iter(METRICS))])


# ─────────────────────────── panel (frontend orchestration) ───────────────────────────

def _hscroll(widget: QWidget) -> QScrollArea:
    """*widget* filling a dock's height but scrolling sideways: a narrow dock clips nothing."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return area


class _StatsWorker(QObject):
    """Runs the test off the GUI thread: gathering values may load CCGs for every session."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, backend):
        super().__init__()
        self._backend = backend

    def run(self):
        try:
            self._backend.run()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit()


class StatsTestPanel(QWidget, ConfigBound):
    """One stats test: top controls, group rows, result plot."""

    renamed = Signal(str)

    _CONFIG = StatsTestConfig
    _BIND = {
        'test_type':     (lambda p: p._test_type.currentText(), lambda p, v: p._test_type.setCurrentText(v)),
        'sides':         (lambda p: p._sides.currentText(),     lambda p, v: p._sides.setCurrentText(v)),
        'direction':     (lambda p: p._dir_btn.text().strip(),  lambda p, v: p._dir_btn.setText(v)),
        'nonparametric': (lambda p: p._nonparam.isChecked(),    lambda p, v: p._nonparam.setChecked(v)),
        'log_transform': (lambda p: p._log.isChecked(),         lambda p, v: p._log.setChecked(v)),
        'post_hoc':      (lambda p: p._post_hoc.isChecked(),    lambda p, v: p._post_hoc.setChecked(v)),
        'remove_outliers': (lambda p: p._plot._rm_outliers_check.isChecked(),
                            lambda p, v: p._plot._rm_outliers_check.setChecked(v)),
    }

    def __init__(self, nav: 'AppState', dock_state: str = '', parent=None):
        super().__init__(parent)
        self.nav = nav
        self._dock_state = dock_state   # base64 QMainWindow.saveState: where the docks sit
        self.backend = StatsTestBackend(nav.cd, nav.sd,
                                        display=nav.get_display_config())
        self._mute_missing_warning = False   # per-GUI-instance, never persisted
        self._rows: list[StatsRow] = []
        self.test_config: StatsTestConfig | None = None
        self._loaded_name = ''
        self._thread = self._worker = None

        self._build()
        self._connect_nav()
        self._add_row()
        self._add_row()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        self._test_type = make_combo(
            ["Independent t-test", "Pairwise t-test",
             "One-way ANOVA + Tukey", "Repeated-measures ANOVA"],
            200, current="Pairwise t-test")
        self._sides = make_combo(["Two-sided", "One-sided"], 90)
        self._dir_btn = make_button("A > B", self._toggle_1sided_direction, 60)
        self._nonparam = QCheckBox("nonparametric")
        self._log      = QCheckBox("log-transform")
        self._post_hoc = QCheckBox("post hoc")
        self._post_hoc.setChecked(True)
        self._post_hoc.setToolTip("Run the ANOVA pairwise post-hoc test")
        top = widget_row("Test type:",
                         self._test_type, "Sides:", self._sides, self._dir_btn,
                         self._nonparam, self._log, self._post_hoc)
        root.addLayout(top)

        rows_pane = QWidget()
        rows_root = QVBoxLayout(rows_pane)
        rows_root.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 2, 0, 2)
        _hdr_fixed = {"Group row": 94, "": 22}
        # PICKERS is the only list of picker columns, so the header cannot drift from the row
        self._lock_btns: dict[str, QPushButton] = {}
        for col in ("Group row", *PICKERS, ""):
            lbl = QLabel(PICKERS[col][1] if col in PICKERS else col, styleSheet="font-weight:bold; padding: 4px 4px;")
            if col in _hdr_fixed:
                lbl.setFixedWidth(_hdr_fixed[col])
                hdr.addWidget(lbl)
                continue
            cell = QHBoxLayout()
            cell.addWidget(lbl)
            if col == 'seg':
                self._fold_btn = chip_button('⊞', checked=True)
                self._fold_btn.setToolTip('Fold rows filled from a family into one block')
                self._fold_btn.toggled.connect(lambda _on: self._rebuild_blocks())
                cell.addWidget(self._fold_btn)
            lock = self._lock_btns[col] = chip_button('🔒')
            lock.setToolTip('Lock: every row shares the top row\'s value')
            lock.toggled.connect(partial(self._on_lock_btn, col))
            cell.addWidget(lock)
            cell.addStretch(1)
            box = QWidget()
            box.setLayout(cell)
            cell.setContentsMargins(0, 0, 0, 0)
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            hdr.addWidget(box)
        rows_root.addLayout(hdr)

        self._rows_area = QVBoxLayout()
        self._rows_area.setSpacing(2)
        self._rows_area.addStretch(1)
        self._rows_widget = QWidget()
        self._rows_widget.setLayout(self._rows_area)
        self._rows_widget.installEventFilter(self)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._rows_widget)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.verticalScrollBar().valueChanged.connect(lambda _v: self._place_blocks())
        self._scroll.viewport().installEventFilter(self)
        self._blocks: list[tuple] = []   # (block, column, first row, last row, family or None)
        rows_root.addWidget(self._scroll, stretch=1)

        add_btn = QPushButton("+ Add group")
        add_btn.clicked.connect(lambda: self._add_row())
        add_btn.setFixedWidth(100)
        family_btn = make_button("＋ Family…", self._on_family_btn)
        family_btn.setToolTip("Name an ordered set of segments; picking it on a row fills the rows below in order")
        rows_root.addLayout(widget_row(add_btn, family_btn))

        # same classes as the CCG panel, on the same nav: both copies track each other
        self._norm_mirror = NormSection(self.nav, expanded=False)
        self._cs_mirror = BaselineCSSection(self.nav, expanded=False)
        mirror = QWidget()
        mirror_root = QVBoxLayout(mirror)
        mirror_root.addLayout(widget_row(self._norm_mirror, self._cs_mirror, stretch=False))
        mirror_root.addStretch(1)

        self._result_text = QPlainTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFont(QtWidgets.QApplication.font())
        self._result_text.setMinimumHeight(80)

        self._plot = StatsPlotWidget()
        self._plot.rerun_requested.connect(lambda: self.results and self._run())
        self._lines = PairLinesPlot(self.nav)
        self._plot_tabs = QTabWidget()
        self._plot_tabs.addTab(self._plot, "Distribution")
        self._plot_tabs.addTab(self._lines, "Pair lines")
        self._plot_tabs.setCornerWidget(make_button("⤢", self._on_fullscreen_btn, 30))
        self._full_win: QDialog | None = None
        self._plot_tabs.setMinimumHeight(150)

        # a nested main window, so each part is a dock its header drags to any side
        self._docks = QMainWindow()
        self._docks.setWindowFlags(Qt.WindowType.Widget)
        self._docks.setDockNestingEnabled(True)
        docks = {}
        for name, title, widget, area in (('rows', "Group rows", rows_pane, Qt.DockWidgetArea.LeftDockWidgetArea),
                                          ('norm', "Normalization / Baseline", mirror, Qt.DockWidgetArea.LeftDockWidgetArea),
                                          ('plot', "Plot", self._plot_tabs, Qt.DockWidgetArea.RightDockWidgetArea),
                                          ('text', "Result", self._result_text, Qt.DockWidgetArea.RightDockWidgetArea)):
            dock = docks[name] = QDockWidget(title)
            dock.setObjectName(f"stats_{name}")   # saveState matches docks by name
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            dock.setWidget(_hscroll(widget))
            self._docks.addDockWidget(area, dock)
        self._plot_dock = docks['plot']
        if self._dock_state:
            self._docks.restoreState(QByteArray.fromBase64(self._dock_state.encode()))
        else:   # sizes only take once the window has a size
            QTimer.singleShot(0, lambda: (
                self._docks.resizeDocks([docks['rows'], docks['plot']], [1, 1], Qt.Orientation.Horizontal),
                self._docks.resizeDocks([docks['rows'], docks['norm']], [3, 1], Qt.Orientation.Vertical),
                self._docks.resizeDocks([docks['plot'], docks['text']], [5, 2], Qt.Orientation.Vertical)))
        root.addWidget(self._docks, stretch=1)

        self._export_btn = make_button("Save…", self._export)
        self._export_btn.setEnabled(False)
        btn_row = widget_row(make_button("Run", self._run), self._export_btn,
                             make_button("Load…", self._load_result))
        root.addLayout(btn_row)

    def _connect_nav(self):
        # bound methods, not lambdas: Qt drops them when a closed tab is deleted
        for sig in (self.nav.key_changed, self.nav.custom_segs_changed,
                    self.nav.groups.changed, self.nav.selection_changed):
            sig.connect(self._refresh_rows)
        self.nav.groups_rewired.connect(self._on_groups_rewired)

    def _on_groups_rewired(self):
        """A project switch replaces sd.groups, taking its connections with it."""
        self.nav.groups.changed.connect(self._refresh_rows)

    # -- rows ----------------------------------------------------------------
    def _add_row(self, cfg: RowConfig | None = None):
        row = StatsRow(self.nav, self.backend, cfg, idx=len(self._rows))
        row.row_id = cfg.id if (cfg and cfg.id) else self.backend._next_id
        self.backend._next_id = max(self.backend._next_id, row.row_id) + 1
        row.deleted.connect(self._del_row)
        row.follow_toggled.connect(self._sync_follows)
        row.family_picked.connect(self._on_family_picked)
        row._pickers['seg'].selection_changed.connect(lambda _p: self._rebuild_blocks())
        for p in row._pickers.values():
            p.selection_changed.connect(self._place_blocks)
            sp = p.sizePolicy()
            sp.setRetainSizeWhenHidden(True)   # a block covers the slot, so the columns keep their place
            p.setSizePolicy(sp)
        if cfg is None and self._rows:
            for k in self._locked_cols():
                row._pickers[k].set_following(True)
        self._rows.append(row)
        self._rows_area.insertWidget(len(self._rows) - 1, row)
        self._sync_follows()

    def _on_family_picked(self, row: StatsRow, members: list):
        """Row k gets the k-th member, counting from the picking row; missing rows copy its other pickers."""
        start = self._rows.index(row)
        while len(self._rows) < start + len(members):
            self._add_row(dataclasses.replace(row.config, id=0, name='', color=''))
        for r, seg in zip(self._rows[start:], members):
            r.set_segment(seg)

    def _on_family_btn(self):
        """Name an ordered set of segments; it joins the segment pickers as a family."""
        dlg = QDialog(self)
        dlg.setWindowTitle("New segment family")
        name = QLineEdit()
        picker = ListPickerButton("Segments", self.backend.available_segments(), plural="segments",
                                  select_all_when_empty=False, ordered=True)
        ok = make_button("Save", dlg.accept)
        lay = QVBoxLayout(dlg)
        lay.addLayout(widget_row("Name:", name))
        lay.addLayout(widget_row("Members, in pick order:", picker))
        lay.addLayout(widget_row(None, ok))
        if dlg.exec() and name.text().strip() and picker.selected:
            self.nav.cd.save_custom_family(name.text().strip(), picker.selected)
            self._refresh_rows()

    def _del_row(self, row: StatsRow):
        if row in self._rows:
            self._rows.remove(row)
        row.deleteLater()
        self._rebuild_blocks()   # before re-binding: a re-bound picker re-places blocks that may span the gone row
        self._sync_follows()

    def _sync_follows(self):
        """Re-bind every picker column; row order is the follow order."""
        for rkey in PICKERS:
            sync_follow_column([r._pickers[rkey] for r in self._rows])
        locked = self._locked_cols()
        for k, btn in self._lock_btns.items():
            btn.blockSignals(True)
            btn.setChecked(k in locked)
            btn.blockSignals(False)
        self._rebuild_blocks()

    def _locked_cols(self) -> set:
        """Columns where every row below the top follows it."""
        return {k for k in PICKERS
                if len(self._rows) > 1 and all(r._pickers[k].following for r in self._rows[1:])}

    def _on_lock_btn(self, col: str, on: bool):
        for r in self._rows[1:]:
            r._pickers[col].set_following(on)
        self._sync_follows()

    def _family_runs(self) -> list:
        """(family, first row, last row) for consecutive rows holding a family's members in order."""
        fams = self.backend.segment_families()
        segs = [r._pickers['seg'].selected for r in self._rows]
        runs, i = [], 0
        while i < len(segs):
            best = max(((f, next((n for n in range(len(m)) if i + n >= len(segs) or segs[i + n] != [m[n]]), len(m)))
                        for f, m in fams.items()), key=lambda fn: fn[1], default=(None, 0))
            if best[1] >= 2:
                runs.append((best[0], i, i + best[1] - 1))
            i += max(best[1], 1)
        return runs

    def _rebuild_blocks(self):
        """One block per locked column over every row, and per folded family over its rows."""
        for b, *_ in self._blocks:
            b.deleteLater()
        self._blocks = []
        for r in self._rows:
            for p in r._pickers.values():
                p.setVisible(True)
        spans = [(k, 0, len(self._rows) - 1) for k in PICKERS if k in self._locked_cols()]
        if self._fold_btn.isChecked() and 'seg' not in self._locked_cols():
            spans += [('seg', a, b, f) for f, a, b in self._family_runs()]
        for col, a, b, *fam in spans:
            top = self._rows[a]._pickers[col]
            if fam:
                block = QPushButton(f"{FAMILY_PREFIX}{fam[0]} · {b - a + 1} rows", self._rows_widget)
                block.setToolTip('Click to unfold')
                block.clicked.connect(lambda: self._fold_btn.setChecked(False))
            else:
                block = QPushButton(top.text(), self._rows_widget)
                block.clicked.connect(top._open_dialog)
            block.setStyleSheet("QPushButton { border: 1px solid #4a7fd4; border-radius: 4px; }")
            for r in self._rows[a:b + 1]:
                r._pickers[col].setVisible(False)
            block.show()
            self._blocks.append((block, col, a, b, fam[0] if fam else None))
        QTimer.singleShot(0, self._place_blocks)

    def _place_blocks(self):
        """Fit each block over its cells, clipped to the visible part of the list so its label stays in view."""
        vis0 = -self._rows_widget.y()
        vis1 = vis0 + self._scroll.viewport().height()
        for block, col, a, b, fam in self._blocks:
            cell = self._rows[a]._pickers[col]
            x = cell.mapTo(self._rows_widget, QPoint(0, 0)).x()
            y0 = max(self._rows[a].y(), vis0)
            y1 = min(self._rows[b].geometry().bottom(), vis1)
            block.setGeometry(x, y0, cell.width(), max(y1 - y0, cell.height()))
            if fam is None:
                block.setText(cell.text())
            block.raise_()

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            QTimer.singleShot(0, self._place_blocks)
        return super().eventFilter(obj, ev)

    def _refresh_rows(self, *_):
        for r in self._rows:
            r.refresh()

    @property
    def results(self) -> list[StatsResult]:
        return self.backend.results

    def _toggle_1sided_direction(self):
        cur = self._dir_btn.text().strip()
        self._dir_btn.setText("A < B" if cur == "A > B" else "A > B")

    # -- config --------------------------------------------------------------

    # -- run -----------------------------------------------------------------
    def _sync_backend(self):
        """Hand the backend what only the widgets know; it owns everything else."""
        b = self.backend
        b.display = self.nav.get_display_config()
        b.test_config = self.config
        b.view_config = self._plot.config
        b.rows = [r.config for r in self._rows]

    def _run(self):
        if self._thread is not None:
            return
        self._sync_backend()
        if (err := self.backend.validate()):
            self._show_result(err)
            return
        self.test_config = self.backend.test_config
        self._plot.test_config = self.test_config
        self._show_result("Running...")
        self._thread = QThread(self)
        self._worker = _StatsWorker(self.backend)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_run_done)
        self._worker.failed.connect(self._on_run_failed)
        self._thread.start()

    def _teardown_run(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = self._worker = None

    def _on_run_done(self):
        self._teardown_run()
        self._warn_missing_segments()
        self._warn_tail_fallbacks()
        self._show_result(self.backend.result_text())
        self._show_plot()
        self._export_btn.setEnabled(True)

    def _on_run_failed(self, msg: str):
        self._teardown_run()
        self._show_result(f"Run failed: {msg}")

    def _warn_tail_fallbacks(self):
        """Sessions whose tailed baseline used the native window for some or all pairs."""
        rows = self.backend.tail_fallbacks()
        if rows:
            QMessageBox.information(
                self, "Tail baseline fell back",
                f"These used the native window instead of '{self.backend.display.cs_tail_extend}':\n\n"
                + '\n'.join(f"{res} {s} {seg}: {n} of {total} pairs" for s, seg, res, n, total in rows))

    def _warn_missing_segments(self):
        """A segment never computed for a session silently shrinks the run to the sessions that have it."""
        missing = self.backend.missing_segments
        if not missing or self._mute_missing_warning:
            return
        body = '\n'.join(f"{seg}: {', '.join(sorted(s))}" for seg, s in sorted(missing.items()))
        box = QMessageBox(QMessageBox.Icon.Warning, "Segment missing for some sessions",
                          f"These sessions have no CCG for the selected segment and were "
                          f"left out of the run:\n\n{body}", parent=self)
        box.addButton(QMessageBox.StandardButton.Ok)
        mute = box.addButton("Don't show again", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is mute:
            self._mute_missing_warning = True

    def apply_theme(self):
        self._plot.apply_theme()
        self._lines._draw()

    def _show_plot(self):
        self._plot.render(self.results)
        self._lines.render(self.results)

    def _on_fullscreen_btn(self):
        """Lift the plot tabs into a full-screen window; closing it puts them back."""
        if self._full_win is not None:
            self._full_win.close()
            return
        self._full_win = QDialog(self)
        self._full_win.setWindowTitle("Stats plot")
        QVBoxLayout(self._full_win).addWidget(self._plot_tabs)
        self._full_win.finished.connect(self._on_full_win_finished)
        self._full_win.showFullScreen()

    def _on_full_win_finished(self, _result: int):
        self._plot_dock.setWidget(_hscroll(self._plot_tabs))
        self._full_win.deleteLater()
        self._full_win = None

    def dock_state(self) -> str:
        return bytes(self._docks.saveState().toBase64()).decode()

    def _show_result(self, text: str):
        self._result_text.setPlainText(text)

    # -- save / load ---------------------------------------------------------
    def _widget_state(self) -> dict:
        """The part of a saved result the backend cannot know: live widget geometry and nav display."""
        return dict(display=self.nav.get_display_config().serialize())

    def _apply_bundle(self, d: dict):
        """Drive the widgets from a bundle the backend has already parsed."""
        if d.get('display'):
            dc = DisplayConfig(); dc.__setstate__(d['display'])
            self.nav.apply_display_config(dc)
        for r in list(self._rows):
            self._del_row(r)
        for rc in self.backend.rows:
            self._add_row(rc)
        while len(self._rows) < 2:
            self._add_row()
        # a loaded result is shown as saved; restoring its toggles must not trigger a rerun
        self._plot.blockSignals(True)
        self.apply(self.backend.test_config)
        self._plot.apply(self.backend.view_config)
        self._plot.blockSignals(False)

    def _export(self):
        if not self.results:
            return
        default = self._loaded_name or datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        def _do_save(name):
            self._sync_backend()
            self.backend.save(name, **self._widget_state())
            self._loaded_name = name
            self.renamed.emit(name)
        VersionSaveDialog.show(self, "Save Stats Result", default, on_save=_do_save)

    def _load_result(self):
        def _do_load(path):
            try:
                self._apply_bundle(self.backend.load(path))
                self.test_config = self.config
                self.backend.test_config = self.test_config
                self._plot.test_config = self.test_config
                self._show_result(self.backend.result_text(self.results))
                self._show_plot()
                self._export_btn.setEnabled(True)
                self._loaded_name = pathlib.Path(path).stem
                self.renamed.emit(self._loaded_name)
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", str(exc))
        VersionLoadDialog.show(self, "Load Stats Result", self.backend.saved_results(),
                               on_load=_do_load, empty_msg="No saved stats results found.")


class StatsTestTabs(QTabWidget):
    """Floating window of stats tests as browser-style tabs: '+' opens one, x closes one."""

    def __init__(self, nav: 'AppState', dock_state: str = '', parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.nav = nav
        self._dock_state = dock_state
        self._opened = 0
        self.setWindowTitle("Stats Tests")
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.setCornerWidget(make_button("+", self._on_new_tab_btn, width=28), Qt.Corner.TopRightCorner)
        self.tabCloseRequested.connect(self._on_close_tab)
        self._on_new_tab_btn()

    def _on_new_tab_btn(self) -> None:
        self._opened += 1
        panel = StatsTestPanel(self.nav, self.dock_state())
        panel.renamed.connect(lambda name: self.setTabText(self.indexOf(panel), name))
        self.setCurrentIndex(self.addTab(panel, f"Test {self._opened}"))

    def dock_state(self) -> str:
        """The open tab's arrangement, else the one this window was opened with."""
        return self.currentWidget().dock_state() if self.count() else self._dock_state

    def _on_close_tab(self, i: int) -> None:
        panel = self.widget(i)
        if panel._thread is not None:   # a running QThread must finish before its owner dies
            return
        self.removeTab(i)
        panel.deleteLater()
        if not self.count():
            self._on_new_tab_btn()

    def apply_theme(self) -> None:
        for i in range(self.count()):
            self.widget(i).apply_theme()
