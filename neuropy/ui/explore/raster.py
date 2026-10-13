"""A scrolling spike raster of every played neuron, co-fire spikes in their edge's colour."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

_PLAIN_RGB = (110, 110, 110)


class RasterPlot(pg.PlotWidget):
    """One row per neuron; the visible span ends at the clock."""

    def __init__(self):
        super().__init__()
        self.setLabel('bottom', 'time (s)')
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().getViewBox().disableAutoRange()
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(255, 20, 147, width=2))
        self.addItem(self.playhead, ignoreBounds=True)
        self.layers = []   # (curve, per-row spike times)

    def load(self, flash) -> None:
        """Rows and colours from the flash's played neurons and edges."""
        for curve, _ in self.layers:
            self.removeItem(curve)
        n = len(flash.neurons)
        owner = [np.full(len(s), -1) for s in flash.spikes]
        row = {nid: i for i, nid in enumerate(flash.neurons)}
        for e, (a, b, _, ev) in reversed(list(enumerate(flash.edges))):   # the pair's own edge wins
            owner[row[a]][ev.ref_idx] = e
            owner[row[b]][ev.tgt_idx] = e
        styles = [(-1, _PLAIN_RGB, 1)] + [(e, edge[2], 2) for e, edge in enumerate(flash.edges)]
        self.layers = []
        for e, rgb, width in styles:
            curve = pg.PlotCurveItem(connect='pairs', pen=pg.mkPen(*rgb, width=width))
            self.addItem(curve, ignoreBounds=True)
            self.layers.append((curve, [s[o == e] for s, o in zip(flash.spikes, owner)]))
        self.getAxis('left').setTicks([[(n - 1 - i, f"{nid}") for i, nid in enumerate(flash.neurons)]])
        self.setYRange(-0.6, n - 0.4, padding=0)

    def show_at(self, t: float, span: float) -> None:
        n = len(self.layers[0][1]) if self.layers else 0
        for curve, rows in self.layers:
            xs, ys = [], []
            for i, s in enumerate(rows):
                seg = s[np.searchsorted(s, t - span):np.searchsorted(s, t, side='right')]
                xs.append(np.repeat(seg, 2))
                ys.append(np.tile([n - 1 - i - 0.4, n - 1 - i + 0.4], len(seg)))
            curve.setData(np.concatenate(xs) if xs else [], np.concatenate(ys) if ys else [])
        self.playhead.setValue(t)
        self.setXRange(t - span, t, padding=0)
