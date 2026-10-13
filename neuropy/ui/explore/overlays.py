"""Items explore lays over panels it does not own: a time cursor, network flashes and a neuron menu."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import QEvent, QObject, QPointF, Qt, QTimer, Signal
from pyqtgraph.Qt.QtGui import QColor, QPainter, QPen
from pyqtgraph.Qt.QtWidgets import QMenu, QWidget

from neuropy.ui.explore.events import PairEvents

_REF_RGB, _TGT_RGB, _EXTRA_RGB = (255, 111, 0), (30, 136, 229), (46, 160, 67)
_PAIR_EDGE_RGB, _EXTRA_EDGE_RGB = (255, 20, 147), (142, 68, 173)


def network_scatters(plot: pg.PlotWidget) -> list:
    """The network's neuron scatters: their spots carry the neuron index as data."""
    return [item for item in plot.getPlotItem().items if isinstance(item, pg.ScatterPlotItem)]


class SliderCursor(QObject):
    """The current frame's window on the time slider; dropping it seeks playback to its right edge."""

    seek = Signal(float)

    def __init__(self, plot: pg.PlotWidget):
        super().__init__()
        self.plot = plot
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(*_PAIR_EDGE_RGB, 70),
                                          pen=pg.mkPen(_PAIR_EDGE_RGB, width=4))
        self.region.setZValue(50)
        self.region.sigRegionChangeFinished.connect(lambda rg: self.seek.emit(float(rg.getRegion()[1])))

    def attach(self) -> None:
        self.plot.addItem(self.region, ignoreBounds=True)

    def detach(self) -> None:
        self.plot.removeItem(self.region)

    def set_frame(self, t0: float, t1: float) -> None:
        self.region.blockSignals(True)
        self.region.setRegion((t0, t1))
        self.region.blockSignals(False)


class NetworkFlash(QWidget):
    """Halos on played neurons and lines on their co-firing pairs, painted over a snapshot of the network."""

    DECAY = 0.2

    def __init__(self, plot: pg.PlotWidget):
        # an opaque sibling layer: lighting items inside the network's scene would repaint every edge
        super().__init__(plot.viewport())
        self.plot = plot
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.snapshot = None
        self.pos = {}
        self.neurons, self.colors, self.spikes = [], [], []
        self.edges = []   # (a, b, rgb, PairEvents)
        self.n_level = np.zeros(0)
        self.e_level = np.zeros(0)
        self.regrab = QTimer(self)
        self.regrab.setSingleShot(True)
        self.regrab.timeout.connect(self.locate)
        self.hide()

    def attach(self) -> None:
        self.plot.scene().changed.connect(self._on_scene_changed)
        self.plot.viewport().installEventFilter(self)
        self.locate()

    def detach(self) -> None:
        self.plot.scene().changed.disconnect(self._on_scene_changed)
        self.plot.viewport().removeEventFilter(self)
        self.hide()

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() == QEvent.Type.Resize:
            self.regrab.start(0)
        return False

    def _on_scene_changed(self, _rects) -> None:
        self.regrab.start(30)

    def load(self, trains, sampling_rate: float, bin_size: float, n_bins: int,
             pair: tuple, pair_events: PairEvents, extras: list) -> None:
        """Play the pair plus *extras*; extra pairs co-fire inside the same CCG window."""
        ref, tgt = pair
        extras = [n for n in extras if n not in pair]
        self.neurons = [ref, tgt, *extras]
        self.colors = [_REF_RGB, _TGT_RGB] + [_EXTRA_RGB] * len(extras)
        self.spikes = [np.sort(np.asarray(trains[n], dtype=float)) for n in self.neurons]
        self.edges = [(ref, tgt, _PAIR_EDGE_RGB, pair_events)] + [
            (a, b, _EXTRA_EDGE_RGB,
             PairEvents.build(trains[a], trains[b], bin_size, n_bins, sampling_rate))
            for a, b in combinations(self.neurons, 2) if {a, b} != {ref, tgt}]
        self.n_level = np.zeros(len(self.neurons))
        self.e_level = np.zeros(len(self.edges))
        self.update()

    def locate(self) -> None:
        """Re-snapshot the network and read neuron positions off its scatter spots."""
        self.pos = {int(s.data()): (s.pos().x(), s.pos().y())
                    for item in network_scatters(self.plot) for s in item.points()
                    if isinstance(s.data(), (int, np.integer))}
        self.hide()
        self.snapshot = self.plot.viewport().grab()
        self.setGeometry(self.plot.viewport().rect())
        self.show()
        self.raise_()

    def reset(self) -> None:
        self.n_level[:] = 0
        self.e_level[:] = 0
        self.update()

    def advance(self, t0: float, t1: float) -> None:
        """Decay, then relight what fired in (t0, t1]."""
        fired_n = [np.searchsorted(s, t1, side='right') > np.searchsorted(s, t0, side='right') for s in self.spikes]
        fired_e = [np.searchsorted(e[3].times, t1, side='right') > np.searchsorted(e[3].times, t0, side='right') for e in self.edges]
        self.n_level = np.where(fired_n, 1.0, np.maximum(self.n_level - self.DECAY, 0))
        self.e_level = np.where(fired_e, 1.0, np.maximum(self.e_level - self.DECAY, 0))
        self.update()

    def paintEvent(self, _ev) -> None:
        if self.snapshot is None:
            return
        vb = self.plot.getPlotItem().vb
        px = {n: self.plot.mapFromScene(vb.mapViewToScene(QPointF(*xy))) for n, xy in self.pos.items()}
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self.snapshot)
        for (a, b, rgb, _), lvl in zip(self.edges, self.e_level):
            if lvl > 0 and a in px and b in px:
                painter.setPen(QPen(QColor(*rgb, int(255 * lvl)), 2 + 4 * lvl))
                painter.drawLine(px[a], px[b])
        for n, rgb, lvl in zip(self.neurons, self.colors, self.n_level):
            if n in px:
                r = 3 + 4 * lvl
                painter.setPen(QPen(QColor(*rgb, int(255 * max(lvl, .3))), 2))
                painter.setBrush(QColor(*rgb, int(200 * lvl)))
                painter.drawEllipse(QPointF(px[n]), r, r)
        painter.end()


class NeuronMenu(QObject):
    """Right-click on a network neuron toggles it in playback; elsewhere the plot keeps its own menu."""

    toggled = Signal(int)
    cleared = Signal()

    def __init__(self, plot: pg.PlotWidget, members):
        super().__init__()
        self.plot = plot
        self.members = members
        self.swallow = False

    def attach(self) -> None:
        self.plot.viewport().installEventFilter(self)

    def detach(self) -> None:
        self.plot.viewport().removeEventFilter(self)

    def neuron_at(self, pos):
        scene_pos = self.plot.mapToScene(pos)
        for item in network_scatters(self.plot):
            hits = [p.data() for p in item.pointsAt(item.mapFromScene(scene_pos))
                    if isinstance(p.data(), (int, np.integer))]
            if hits:
                return int(hits[0])
        return None

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() == QEvent.Type.MouseButtonRelease and self.swallow:
            self.swallow = False
            return True
        if ev.type() != QEvent.Type.MouseButtonPress or ev.button() != Qt.MouseButton.RightButton:
            return False
        nid = self.neuron_at(ev.position().toPoint())
        if nid is None:
            return False
        self.swallow = True
        menu = QMenu(self.plot)
        menu.addAction(f"{'Remove' if nid in self.members() else 'Include'} neuron {nid} "
                       f"{'from' if nid in self.members() else 'in'} playback",
                       lambda: self.toggled.emit(nid))
        if self.members():
            menu.addAction("Clear playback neurons", self.cleared.emit)
        menu.popup(ev.globalPosition().toPoint())
        return True
