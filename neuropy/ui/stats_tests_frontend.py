"""Stats-test panel widgets: the row, the plot canvas and the panel that drives them.

Widgets are the source of truth; configs are snapshots taken at run/save.
"""
from __future__ import annotations

import datetime
import pathlib
from typing import TYPE_CHECKING

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt.QtCore import QObject, Qt, QThread, Signal
from pyqtgraph.Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QLineEdit,
    QPushButton, QCheckBox, QPlainTextEdit,
    QScrollArea, QFrame, QMessageBox, QSizePolicy,
)
from neuropy.ui.utils import (ConfigBound, ScopeField, ScopePicker, make_combo, make_button,
                              ColorLabelButton, sync_follow_column, widget_row)
from neuropy.analyses.utils import group_display
from neuropy.ui.ui_common import qt_dark_mode
from neuropy.ui.dialogs import VersionSaveDialog, VersionLoadDialog
from neuropy.ui.app_state import DisplayConfig
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
        self._outliers_check.setChecked(True)
        for chk in (self._violin_check, self._outliers_check, self._sig_check):
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
        self._fig.patch.set_facecolor('#2b2b2b' if qt_dark_mode() else 'white')
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


# ─────────────────────────── row widget (frontend) ───────────────────────────

class StatsRow(QWidget, ConfigBound):
    """One group row: color, name, four multi-select pickers, data-type combo, delete.
    Owns its widgets and maps them to/from a RowConfig."""

    deleted = Signal(object)
    follow_toggled = Signal()

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
                        display=group_display if rkey == 'grp' else None)
             for rkey, (_fld, label, plural, prov_attr) in PICKERS.items()], labelled=False)
        self._pickers = self._scope.pickers
        for p in self._pickers.values():
            p.follow_toggled.connect(self.follow_toggled)
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
        if len(grp_opts) > 1:
            self._pickers['grp'].set_selected([grp_opts[1]])
        self._pickers['data'].set_selected([next(iter(METRICS))])


# ─────────────────────────── panel (frontend orchestration) ───────────────────────────

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
    """Persistent floating stats-test panel: top controls, group rows, result plot."""

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

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.nav = nav
        self.backend = StatsTestBackend(nav.cd, nav.sd,
                                        display=nav.get_display_config())
        self._mute_missing_warning = False   # per-GUI-instance, never persisted
        self._rows: list[StatsRow] = []
        self.test_config: StatsTestConfig | None = None
        self._loaded_name = ''
        self._thread = self._worker = None

        self.setWindowTitle("Stats Tests")
        self.resize(1100, 580)
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

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 2, 0, 2)
        _hdr_fixed = {"Group row": 94, "": 22}
        # PICKERS is the only list of picker columns, so the header cannot drift from the row
        for col in ("Group row", *(v[1] for v in PICKERS.values()), ""):
            lbl = QLabel(col, styleSheet="font-weight:bold; padding: 4px 4px;")
            if col in _hdr_fixed:
                lbl.setFixedWidth(_hdr_fixed[col])
            else:
                lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            hdr.addWidget(lbl)
        root.addLayout(hdr)

        self._rows_area = QVBoxLayout()
        self._rows_area.setSpacing(2)
        rows_widget = QWidget()
        rows_widget.setLayout(self._rows_area)
        scroll = QScrollArea()
        scroll.setWidget(rows_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        add_btn = QPushButton("+ Add group")
        add_btn.clicked.connect(lambda: self._add_row())
        add_btn.setFixedWidth(100)
        root.addWidget(add_btn)

        # same classes as the CCG panel, on the same nav: both copies track each other
        self._norm_mirror = NormSection(self.nav, expanded=False)
        self._cs_mirror = BaselineCSSection(self.nav, expanded=False)
        mirror = widget_row(self._norm_mirror, self._cs_mirror, stretch=False)
        root.addLayout(mirror)

        res_frame = QFrame()
        res_frame.setFrameShape(QFrame.Shape.StyledPanel)
        res_root = QVBoxLayout(res_frame)
        res_root.setContentsMargins(4, 4, 4, 4)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._result_text = QPlainTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFont(QtWidgets.QApplication.font())
        self._result_text.setMinimumHeight(80)
        self._splitter.addWidget(self._result_text)
        self._plot = StatsPlotWidget()
        self._plot.rerun_requested.connect(lambda: self.results and self._run())
        self._splitter.addWidget(self._plot)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.splitterMoved.connect(lambda *_: self._plot.refresh_plot())
        res_root.addWidget(self._splitter)
        root.addWidget(res_frame, stretch=1)

        self._export_btn = make_button("Save…", self._export)
        self._export_btn.setEnabled(False)
        btn_row = widget_row(make_button("Run", self._run), self._export_btn,
                             make_button("Load…", self._load_result))
        root.addLayout(btn_row)

    def _connect_nav(self):
        self.nav.key_changed.connect(lambda _: self._refresh_rows())
        self.nav.custom_segs_changed.connect(self._refresh_rows)
        self.nav.groups.changed.connect(self._refresh_rows)
        self.nav.selection_changed.connect(self._refresh_rows)
        # A project switch replaces sd.groups, taking its connections with it.
        self.nav.groups_rewired.connect(
            lambda: self.nav.groups.changed.connect(self._refresh_rows))

    # -- rows ----------------------------------------------------------------
    def _add_row(self, cfg: RowConfig | None = None):
        row = StatsRow(self.nav, self.backend, cfg, idx=len(self._rows))
        row.row_id = cfg.id if (cfg and cfg.id) else self.backend._next_id
        self.backend._next_id = max(self.backend._next_id, row.row_id) + 1
        row.deleted.connect(self._del_row)
        row.follow_toggled.connect(self._sync_follows)
        self._rows.append(row)
        self._rows_area.addWidget(row)
        self._sync_follows()

    def _del_row(self, row: StatsRow):
        if row in self._rows:
            self._rows.remove(row)
        row.deleteLater()
        self._sync_follows()

    def _sync_follows(self):
        """Re-bind every picker column; row order is the follow order."""
        for rkey in PICKERS:
            sync_follow_column([r._pickers[rkey] for r in self._rows])

    def _refresh_rows(self):
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
        self._show_result(self.backend.result_text())
        self._plot.render(self.results)
        self._export_btn.setEnabled(True)

    def _on_run_failed(self, msg: str):
        self._teardown_run()
        self._show_result(f"Run failed: {msg}")

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

    def _show_result(self, text: str):
        self._result_text.setPlainText(text)

    # -- save / load ---------------------------------------------------------
    def _widget_state(self) -> dict:
        """The part of a saved result the backend cannot know: live widget geometry and nav display."""
        return dict(splitter_sizes=list(self._splitter.sizes()),
                    display=self.nav.get_display_config().serialize())

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
        self.apply(self.backend.test_config)
        # a loaded result is shown as saved; restoring its toggles must not trigger a rerun
        self._plot.blockSignals(True)
        self._plot.apply(self.backend.view_config)
        self._plot.blockSignals(False)
        sizes = d.get('splitter_sizes')
        if sizes and len(sizes) == 2:
            self._splitter.setSizes(sizes)

    def _export(self):
        if not self.results:
            return
        default = self._loaded_name or datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        def _do_save(name):
            self._sync_backend()
            self.backend.save(name, **self._widget_state())
            self._loaded_name = name
        VersionSaveDialog.show(self, "Save Stats Result", default, on_save=_do_save)

    def _load_result(self):
        def _do_load(path):
            try:
                self._apply_bundle(self.backend.load(path))
                self.test_config = self.config
                self.backend.test_config = self.test_config
                self._plot.test_config = self.test_config
                self._show_result(self.backend.result_text(self.results))
                self._plot.render(self.results)
                self._export_btn.setEnabled(True)
                self._loaded_name = pathlib.Path(path).stem
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", str(exc))
        VersionLoadDialog.show(self, "Load Stats Result", self.backend.saved_results(),
                               on_load=_do_load, empty_msg="No saved stats results found.")
