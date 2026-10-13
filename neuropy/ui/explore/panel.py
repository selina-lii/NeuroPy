"""Explore centre column: the current pair's CCG in a window sliding over the session, with its waveforms."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt, QTimer
from pyqtgraph.Qt.QtWidgets import (QDoubleSpinBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
                                    QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
                                    QWidget)

from neuropy.ui.ccg_panel import WaveformPanelQt
from neuropy.ui.explore.events import PairEvents
from neuropy.ui.explore.overlays import NetworkFlash, NeuronMenu, SliderCursor
from neuropy.ui.explore.raster import RasterPlot


def _hms(t: float) -> str:
    s = int(t)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class ExplorePanel(QWidget):
    """Plays spike-pair events into a CCG; the slider cursor and network flashes follow the clock."""

    def __init__(self, ui):
        super().__init__()
        self.ui, self.nav = ui, ui.nav
        self.events = None
        self.loaded = None
        self.t = 0.0
        self.hist = np.zeros(0)
        self.cursor = SliderCursor(ui.time_slider._main_plot)
        self.cursor.seek.connect(self.seek)
        self.flash = NetworkFlash(ui.neuron_network._plot_win)
        self.extras = []
        self.active = False
        self.menu = NeuronMenu(ui.neuron_network._plot_win, lambda: self.extras)
        self.menu.toggled.connect(self._on_menu_toggled)
        self.menu.cleared.connect(lambda: self._set_extras([]))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.play_btn = QPushButton('▶ Play')
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_btn)
        self.restart_btn = QPushButton('⏮')
        self.restart_btn.clicked.connect(lambda: self.seek(self.events.span[0] if self.events else 0.0))
        self.step_back_btn = QPushButton('◀|')
        self.step_back_btn.clicked.connect(lambda: self.seek(self.t - self.step_spin.value()))
        self.step_fwd_btn = QPushButton('|▶')
        self.step_fwd_btn.clicked.connect(self._on_tick)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.001, 3600.0)
        self.step_spin.setDecimals(3)
        self.step_spin.setValue(1.0)
        self.step_spin.setSuffix(' s/frame')
        self.step_spin.valueChanged.connect(lambda _v: self._refresh())
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(' fps')
        self.fps_spin.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(['count', 'area', 'z-score'])
        self.norm_combo.currentTextChanged.connect(lambda _t: self._refresh())
        self.norm_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.time_lbl = QLabel('')
        self.time_lbl.setFixedWidth(self.time_lbl.fontMetrics().horizontalAdvance('00:00:00 – 00:00:00 '))   # a reflow would redraw the waveform canvas

        transport, timing = QHBoxLayout(), QHBoxLayout()
        for w in (self.restart_btn, self.step_back_btn, self.play_btn, self.step_fwd_btn):
            transport.addWidget(w)
        for w in (self.norm_combo,):
            transport.addWidget(w)
        transport.addStretch(1)
        for w in (self.step_spin, self.fps_spin, self.time_lbl):
            timing.addWidget(w)
        timing.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'lag (ms)')
        self.plot.setLabel('left', 'count')
        self.bars = pg.BarGraphItem(x=[], height=[], width=1.0, brush=pg.mkBrush(66, 133, 244, 200), pen=None)
        self.plot.addItem(self.bars)
        self.waveforms = WaveformPanelQt()

        split = QSplitter(Qt.Orientation.Vertical)
        self.raster = RasterPlot()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.raster, 'Spike trains')
        self.tabs.addTab(self.plot, 'CCG')
        self.tabs.setCurrentWidget(self.plot)
        self.tabs.currentChanged.connect(lambda _i: self._refresh())
        split.addWidget(self.tabs)
        split.addWidget(self.waveforms)
        split.setSizes([500, 300])
        root = QVBoxLayout(self)
        root.addLayout(transport)
        root.addLayout(timing)
        # the centre stack takes its widest page's minimum: this page must not widen the others
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root.addWidget(split, stretch=1)

    def activate(self) -> None:
        self.plot.setBackground(self.ui.theme.plot_bg)
        self.raster.setBackground(self.ui.theme.plot_bg)
        self.active = True
        self.cursor.attach()
        self.menu.attach()
        self.flash.attach()
        self.load_pair()

    def deactivate(self) -> None:
        if not self.active:
            return
        self.active = False
        self.play_btn.setChecked(False)
        self.cursor.detach()
        self.menu.detach()
        self.flash.detach()

    def load_pair(self) -> None:
        """Rebuild events for the current pair; a pair already loaded keeps its clock."""
        pair = self.nav.current_pair
        if pair is None:
            return
        ref, tgt = int(pair[-2]), int(pair[-1])
        data = self.nav.ccg_data
        n_bins = data.ccg.shape[-1]
        bin_size = data.conf.duration / (n_bins - 1)
        ident = (str(self.nav.pair_ptr_key.session), ref, tgt, n_bins, bin_size)
        QTimer.singleShot(0, self.flash.locate)   # after the network redraws
        if ident == self.loaded:
            return
        if self.loaded is None or ident[0] != self.loaded[0]:
            self.extras = []
        self.loaded = ident
        neurons = self.nav.neurons
        self.events = PairEvents.build(neurons.spiketrains[ref], neurons.spiketrains[tgt],
                                       bin_size, n_bins, neurons.sampling_rate)
        lags = (np.arange(n_bins) - n_bins // 2) * bin_size * 1e3
        self.bars.setOpts(x=lags, width=bin_size * 1e3 * 0.9)
        self.waveforms.render(neurons, ref, tgt)
        self._load_flash()
        self.t = self.events.span[0]
        self._refresh()

    def _load_flash(self) -> None:
        neurons, (_, ref, tgt, n_bins, bin_size) = self.nav.neurons, self.loaded
        self.flash.load(neurons.spiketrains, neurons.sampling_rate, bin_size, n_bins,
                        (ref, tgt), self.events, self.extras)
        self.raster.load(self.flash)
        self.tabs.setCurrentWidget(self.raster if len(self.flash.neurons) > 2 else self.plot)

    def _set_extras(self, extras: list) -> None:
        self.extras = extras
        self._load_flash()
        self._show()

    def _on_menu_toggled(self, nid: int) -> None:
        self._set_extras([n for n in self.extras if n != nid] if nid in self.extras else self.extras + [nid])

    def seek(self, t: float) -> None:
        lo, hi = self.events.span
        self.t = float(np.clip(t, lo, hi))
        self.flash.reset()
        self._show()

    def _on_play_btn(self, on: bool) -> None:
        self.play_btn.setText('⏸ Pause' if on else '▶ Play')
        if on:
            self.timer.start(int(1000 / self.fps_spin.value()))
        else:
            self.timer.stop()

    def _on_tick(self) -> None:
        t1 = min(self.t + self.step_spin.value(), self.events.span[1])
        self.flash.advance(self.t, t1)
        self.t = t1
        self._show()
        if t1 >= self.events.span[1]:
            self.play_btn.setChecked(False)

    def _norm(self, hist: np.ndarray) -> np.ndarray:
        """Scale one frame by the chosen normalisation."""
        mode = self.norm_combo.currentText()
        if mode == 'area':
            return hist / max(hist.sum(), 1)
        if mode == 'z-score':
            sd = hist.std()
            return (hist - hist.mean()) / sd if sd > 0 else np.zeros(len(hist))
        return hist.astype(float)

    def _refresh(self) -> None:
        if self.events is None:
            return
        self.plot.setLabel('left', self.norm_combo.currentText())
        self._show()

    def _show(self) -> None:
        cumulative = self.norm_combo.currentText() == 'count'   # counts build up; area and z-score read one frame
        t0 = self.events.span[0] if cumulative else self.t - self.step_spin.value()
        self.hist = self.events.hist(-np.inf if cumulative else t0, self.t)
        self.bars.setOpts(height=self._norm(self.hist))
        self.cursor.set_frame(t0, self.t)
        if self.tabs.currentWidget() is self.raster:
            self.raster.show_at(self.t, self.step_spin.value())
        self.time_lbl.setText(f"{_hms(t0)} – {_hms(self.t)}")
