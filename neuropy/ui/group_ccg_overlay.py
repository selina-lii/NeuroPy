"""Every CCG in a pair scope drawn over one another, with their mean and sd."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt, QEvent
from pyqtgraph.Qt.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
                                    QLabel, QSlider, QVBoxLayout)

from neuropy.analyses.ccg_transforms import CCGNorm
from neuropy.ui.dialogs import pair_scope_fields, pair_scope_jobs
from neuropy.ui.ui_common import qt_dark_mode
from neuropy.ui.utils import ColorLabelButton, ScopePicker, apply_plot_chrome, make_button, plot_pen, widget_row

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState

SCALINGS = ('GUI norms', 'z-score')


class GroupCCGOverlayDialog(QDialog):
    """Pick a pair scope, Draw its CCGs as outlines; colours, alpha and the mean/sd lines restyle without reloading."""

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__(parent)
        self.nav = nav
        self.setWindowTitle("Group CCG overlay")
        self.resize(1000, 800)
        self._x, self._ys = None, np.empty((0, 0))
        lay = QVBoxLayout(self)
        self._scope = ScopePicker(pair_scope_fields(nav))
        self._scope.set_selected({'seg': nav.visible_segments(), 'sess': [nav.current_session_str]})
        lay.addWidget(self._scope)
        self._res_combo = QComboBox()
        self._res_combo.addItems(nav.cd.complete_resolutions())
        self._res_combo.setCurrentText(nav.data_resolution)
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(SCALINGS)
        self._status = QLabel()
        lay.addLayout(widget_row("resolution:", self._res_combo, "scale:", self._scale_combo,
                                 make_button("Draw", self._on_draw_btn), self._status))
        self._trace_color = ColorLabelButton('#7aafff', "CCGs", title="CCG colour")
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(1, 100)
        self._alpha_slider.setValue(15)
        self._alpha_slider.setFixedWidth(140)
        self._mean_check, self._sd_check = QCheckBox("mean"), QCheckBox("± sd")
        self._mean_check.setChecked(True)
        self._sd_check.setChecked(True)
        self._mean_color = ColorLabelButton('#f0a030', "", title="Mean colour")
        self._sd_color = ColorLabelButton('#f0d080', "", title="SD colour")
        for w in (self._trace_color, self._mean_color, self._sd_color):
            w.color_changed.connect(self._draw)
        for w in (self._mean_check, self._sd_check):
            w.toggled.connect(self._draw)
        self._alpha_slider.valueChanged.connect(self._draw)
        lay.addLayout(widget_row(self._trace_color, "alpha:", self._alpha_slider, None,
                                 self._mean_check, self._mean_color, self._sd_check, self._sd_color, None,
                                 make_button("Save PNG…", self._on_save_btn)))
        self._plot = pg.PlotWidget()
        lay.addWidget(self._plot, stretch=1)

    def _on_draw_btn(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._x, self._ys = self._load()
        finally:
            QApplication.restoreOverrideCursor()
        self._status.setText(f"{len(self._ys)} CCGs")
        self._draw()

    def _load(self) -> tuple:
        """Bin centres (ms) and one scaled CCG per row, for every pair the scope picks."""
        nav, cd = self.nav, self.nav.cd
        zscore = self._scale_combo.currentText() == 'z-score'
        jobs, _skipped = pair_scope_jobs(nav, self._scope.selected)
        x, ys = None, []
        for key, seg, pairs in jobs:
            k = key.change(resolution=self._res_combo.currentText(), segment=seg)
            data = cd.ccg_for(k)
            neurons, hours = cd.nd.neurons_for(k.nd()), cd.time_hours_for(k, seg)
            for ref, tgt in pairs:
                slices = cd.pair_slices(k.change(ref=ref, tgt=tgt))
                if slices is None:
                    continue
                ccg = np.asarray(slices[0], dtype=float)
                if zscore:
                    ys.append((ccg - ccg.mean()) / max(ccg.std(), 1e-12))
                else:
                    ys.append(CCGNorm.apply(ccg, slices[1], ref, tgt, nav.active_norms,
                                            neurons=neurons, custom_time_hours=hours)[0])
                half = data.conf.duration * 1e3 / 2
                x = np.linspace(-half, half, len(ccg))
        return x, np.array(ys)

    def _draw(self, *_) -> None:
        """Restyle from the loaded CCGs in the current theme."""
        dark = qt_dark_mode()
        self._plot.setBackground('#1e1e1e' if dark else 'w')
        p = self._plot.getPlotItem()
        p.clear()
        apply_plot_chrome(p, dark)
        p.setLabel('bottom', 'Lag (ms)')
        if not len(self._ys):
            return
        trace = pg.mkColor(self._trace_color.color)
        trace.setAlphaF(self._alpha_slider.value() / 100)
        bin_ms = self._x[1] - self._x[0]
        edges = np.append(self._x - bin_ms / 2, self._x[-1] + bin_ms / 2)
        for y in self._ys:
            p.addItem(pg.PlotDataItem(edges, y, stepMode='center', pen=pg.mkPen(trace, width=1)))
        mean, sd = self._ys.mean(axis=0), self._ys.std(axis=0)
        if self._sd_check.isChecked():
            for y in (mean - sd, mean + sd):
                p.addItem(pg.PlotDataItem(edges, y, stepMode='center',
                                          pen=plot_pen(self._sd_color.color, Qt.PenStyle.DashLine)))
        if self._mean_check.isChecked():
            p.addItem(pg.PlotDataItem(edges, mean, stepMode='center', pen=pg.mkPen(self._mean_color.color, width=2.5)))

    def _on_save_btn(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay", "group_ccg_overlay.png", "PNG (*.png)")
        if path:
            self._plot.grab().save(path)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:   # the app switched light/dark
            self._draw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._draw()

    @classmethod
    def show_for(cls, nav: 'AppState', parent=None) -> None:
        cls(nav, parent).exec()
