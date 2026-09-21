"""Panel for assigned neuron tags: list them, read them, and cut a value range into labelled bands."""
from __future__ import annotations

import numpy as np
from pyqtgraph.Qt.QtCore import QPointF, QRectF, Qt, Signal
from pyqtgraph.Qt.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from pyqtgraph.Qt.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout,
                                    QInputDialog, QLabel, QMessageBox,
                                    QPushButton, QSizePolicy, QTableWidget,
                                    QTableWidgetItem, QVBoxLayout, QWidget)

from neuropy.analyses.neuron_tags import (CUT_ABSOLUTE, NeuronTagSpec,
                                          NeuronTagSet, tag_value_source)

HANDLE_W = 9
HANDLE_H = 15
TRACK_H = 74
GRAB_PX = 7


class CutoffSlider(QWidget):
    """Value axis with draggable labelled cutoffs over a histogram of the data.

    Each cutoff owns the band to its right, so a value takes the label of the
    rightmost cutoff at or below it.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(TRACK_H + HANDLE_H + 48)   # band names, value ticks, source
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._values = np.array([])
        self._lo = 0.0
        self._hi = 1.0
        self._cutoffs: list[float] = []
        self._labels: list[str] = ['all']
        self._colors: list[tuple] = [(120, 120, 120)]
        self._source = ''
        self._dragging = None

    def set_values(self, values) -> None:
        """The population being cut; sets the axis range and the histogram.

        Ratio sources divide by zero, so infinities are dropped alongside NaN.
        """
        raw = np.asarray(values, dtype=float)
        self._values = raw[np.isfinite(raw)]
        if len(self._values):
            self._lo = float(self._values.min())
            self._hi = float(self._values.max())
            if self._hi <= self._lo:
                self._hi = self._lo + 1.0
        self.update()

    def set_source(self, name: str) -> None:
        """What is being cut, printed under the value ticks."""
        self._source = name
        self.update()

    def set_bands(self, cutoffs: list, labels: list, colors: list = None) -> None:
        self._cutoffs = [float(c) for c in cutoffs]
        self._labels = list(labels)
        self._colors = list(colors) if colors else _band_colors(len(labels))
        self.update()

    @property
    def cutoffs(self) -> list:
        return list(self._cutoffs)

    @property
    def labels(self) -> list:
        return list(self._labels)

    def add_cutoff(self, value: float) -> None:
        """Split the band containing *value*, naming the new upper band after it."""
        self._cutoffs.append(float(value))
        order = np.argsort(self._cutoffs)
        self._cutoffs = [self._cutoffs[i] for i in order]
        at = list(order).index(len(self._cutoffs) - 1)
        self._labels.insert(at + 1, _unique_label(self._labels))
        self._colors = _band_colors(len(self._labels))
        self.changed.emit()
        self.update()

    def remove_cutoff(self, index: int) -> None:
        if not self._cutoffs:
            return
        self._cutoffs.pop(index)
        self._labels.pop(index + 1)
        self._colors = _band_colors(len(self._labels))
        self.changed.emit()
        self.update()

    def rename_band(self, index: int, name: str) -> None:
        self._labels[index] = name
        self.changed.emit()
        self.update()

    def band_of(self, x: float) -> int:
        """Which band a pixel x falls in."""
        return int(np.digitize([self._value_at(x)], self._cutoffs)[0])

    # ── geometry ────────────────────────────────────────────────────────

    def _x_of(self, value: float) -> float:
        span = self._hi - self._lo
        return 4 + (value - self._lo) / span * max(1, self.width() - 8)

    def _value_at(self, x: float) -> float:
        span = max(1, self.width() - 8)
        return self._lo + (x - 4) / span * (self._hi - self._lo)

    def _handle_at(self, pos) -> int | None:
        for i, cut in enumerate(self._cutoffs):
            if abs(pos.x() - self._x_of(cut)) <= GRAB_PX and pos.y() >= TRACK_H - HANDLE_H:
                return i
        return None

    # ── events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = self._handle_at(event.position())
            if self._dragging is None and event.position().y() < TRACK_H:
                self.add_cutoff(self._value_at(event.position().x()))
        elif event.button() == Qt.RightButton:
            hit = self._handle_at(event.position())
            if hit is not None:
                self.remove_cutoff(hit)

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            over = self._handle_at(event.position())
            self.setCursor(Qt.SizeHorCursor if over is not None else Qt.ArrowCursor)
            return
        value = min(self._hi, max(self._lo, self._value_at(event.position().x())))
        self._cutoffs[self._dragging] = value
        order = list(np.argsort(self._cutoffs))
        if order != sorted(order):
            self._labels = [self._labels[0]] + [self._labels[i + 1] for i in order]
            self._cutoffs = sorted(self._cutoffs)
            self._dragging = order.index(self._dragging)
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, _event):
        self._dragging = None

    def mouseDoubleClickEvent(self, event):
        band = self.band_of(event.position().x())
        name, ok = QInputDialog.getText(self, "Band label", "Label:",
                                        text=self._labels[band])
        if ok and name.strip():
            self.rename_band(band, name.strip())

    # ── painting ────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            self._paint_histogram(painter)
            self._paint_handles(painter)
        finally:
            painter.end()   # a live painter at return breaks every later repaint

    def _paint_histogram(self, painter: QPainter) -> None:
        """Counts per bin, each bar coloured by the band it falls in."""
        if not len(self._values):
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(QRectF(0, 0, self.width(), TRACK_H),
                             Qt.AlignCenter, "no values for this source")
            return
        counts, edges = np.histogram(self._values, bins=48,
                                     range=(self._lo, self._hi))
        tallest = max(1, counts.max())
        painter.setPen(Qt.NoPen)
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            band = int(np.digitize([(left + right) / 2], self._cutoffs)[0])
            height = count / tallest * (TRACK_H - 8)
            painter.setBrush(QBrush(QColor(*self._colors[band])))
            painter.drawRect(QRectF(self._x_of(left), TRACK_H - height,
                                    max(1.0, self._x_of(right) - self._x_of(left) - 1),
                                    height))

    def _paint_handles(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.drawLine(4, TRACK_H, self.width() - 4, TRACK_H)
        for cut in self._cutoffs:
            x = self._x_of(cut)
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            painter.setBrush(QBrush(QColor(250, 250, 250)))
            painter.drawPolygon(QPolygonF([
                QPointF(x, TRACK_H - HANDLE_H),
                QPointF(x + HANDLE_W / 2, TRACK_H - HANDLE_H + 5),
                QPointF(x + HANDLE_W / 2, TRACK_H),
                QPointF(x - HANDLE_W / 2, TRACK_H),
                QPointF(x - HANDLE_W / 2, TRACK_H - HANDLE_H + 5)]))
            painter.drawText(QRectF(x - 28, TRACK_H + 2, 56, 13),
                             Qt.AlignCenter, f"{cut:.4g}")
        self._paint_band_names(painter)
        self._paint_value_axis(painter)

    def _paint_band_names(self, painter: QPainter) -> None:
        bounds = [self._lo] + self._cutoffs + [self._hi]
        for i, label in enumerate(self._labels):
            left, right = self._x_of(bounds[i]), self._x_of(bounds[i + 1])
            painter.setPen(QColor(*self._colors[i]))
            painter.drawText(QRectF(left, TRACK_H + 15, max(10.0, right - left), 14),
                             Qt.AlignCenter, label)

    def _paint_value_axis(self, painter: QPainter) -> None:
        """The value range under the track: min, midpoint, max, and the source cut."""
        row = QRectF(4, TRACK_H + 30, self.width() - 8, 13)
        painter.setPen(QColor(140, 140, 140))
        mid = (self._lo + self._hi) / 2
        painter.drawText(row, Qt.AlignLeft, f"{self._lo:.4g}")
        painter.drawText(row, Qt.AlignCenter, f"{mid:.4g}")
        painter.drawText(row, Qt.AlignRight, f"{self._hi:.4g}")
        if self._source:
            painter.drawText(QRectF(4, TRACK_H + 43, self.width() - 8, 13),
                             Qt.AlignCenter, self._source)


class NeuronTagsPanel(QWidget):
    """Assigned tags: the list, their table, and the cutoff editor that writes them."""

    def __init__(self, tags: NeuronTagSet, nav, parent=None):
        super().__init__(parent)
        self.tags = tags
        self.nav = nav
        self._build()
        self.refresh()

    @property
    def nd(self):
        return self.nav.cd.nd

    def _build(self) -> None:
        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        self._tag_list = QComboBox()
        self._tag_list.setMinimumWidth(180)
        self._tag_list.currentIndexChanged.connect(self._on_tag_combo)
        top.addWidget(QLabel("Tag:"))
        top.addWidget(self._tag_list)

        delete = QPushButton("Delete")
        delete.clicked.connect(self._on_delete_btn)
        top.addWidget(delete)
        top.addStretch(1)

        top.addWidget(QLabel("Scope:"))
        self._scope_combo = QComboBox()
        self._scope_combo.setMinimumWidth(170)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_combo)
        top.addWidget(self._scope_combo)
        outer.addLayout(top)

        source = QHBoxLayout()
        source.addWidget(QLabel("Cut on:"))
        self._source_combo = QComboBox()
        self._source_combo.setMinimumWidth(200)
        self._source_combo.currentIndexChanged.connect(self._on_source_combo)
        source.addWidget(self._source_combo)
        source.addStretch(1)
        self._counts = QLabel("")
        self._counts.setStyleSheet("color: #888;")
        source.addWidget(self._counts)
        outer.addLayout(source)

        self._slider = CutoffSlider()
        self._slider.changed.connect(self._on_slider_changed)
        outer.addWidget(self._slider)

        hint = QLabel("click to add a cutoff · drag to move · right-click to remove "
                      "· double-click a band to rename")
        hint.setStyleSheet("color: #888;")
        outer.addWidget(hint)

        assign = QHBoxLayout()
        assign.addStretch(1)
        self._assign_btn = QPushButton("Assign")
        self._assign_btn.clicked.connect(self._on_assign_btn)
        assign.addWidget(self._assign_btn)
        outer.addLayout(assign)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(['Session', 'Neuron', 'Label', 'Value'])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self._table, stretch=1)

    # ── state ───────────────────────────────────────────────────────────

    @property
    def tag_name(self) -> str:
        return self._tag_list.currentText()

    @property
    def scope_key(self):
        """The nd-key to act on, or None for every session."""
        return self._scope_combo.currentData()

    def _spec_for(self, name: str) -> NeuronTagSpec:
        """The cutoff spec backing *name*, created on first use."""
        spec = self.tags.specs.get(name)
        if spec is None:
            spec = NeuronTagSpec(prefix=name, cut_mode=CUT_ABSOLUTE)
            self.tags.add_spec(spec)
        spec.cut_mode = CUT_ABSOLUTE
        return spec

    # ── refresh ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild every list from the dataset, keeping the current tag if it survives."""
        keep = self.tag_name
        for combo in (self._tag_list, self._scope_combo):
            combo.blockSignals(True)

        self._tag_list.clear()
        self._tag_list.addItems(self.nd.tag_names)
        if keep in self.nd.tag_names:
            self._tag_list.setCurrentText(keep)

        self._scope_combo.clear()
        self._scope_combo.addItem("All sessions", None)
        self._scope_combo.insertSeparator(1)
        for nd_key in self.nd.session_keys:
            self._scope_combo.addItem(str(nd_key.session), nd_key)

        for combo in (self._tag_list, self._scope_combo):
            combo.blockSignals(False)
        self._reload_tag()

    def _reload_tag(self) -> None:
        name = self.tag_name
        has_tag = bool(name)
        for widget in (self._source_combo, self._slider, self._assign_btn):
            widget.setEnabled(has_tag)
        if not has_tag:
            self._table.setRowCount(0)
            self._counts.setText("")
            return
        self._reload_sources()
        self._reload_slider()
        self._reload_table()

    def _reload_sources(self) -> None:
        spec = self._spec_for(self.tag_name)
        key = self.scope_key or self.nd.session_keys[0]
        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        self._source_combo.addItems(self.tags.available_sources(key))
        own = tag_value_source(self.tag_name)
        self._source_combo.setCurrentText(
            spec.source if spec.source in self._sources() else
            own if own in self._sources() else self._source_combo.itemText(0))
        spec.source = self._source_combo.currentText()
        self._source_combo.blockSignals(False)

    def _sources(self) -> list:
        return [self._source_combo.itemText(i)
                for i in range(self._source_combo.count())]

    def _reload_slider(self) -> None:
        spec = self._spec_for(self.tag_name)
        values = self.tags.pooled_values(spec, self.scope_key)
        self._slider.set_values(values)
        self._slider.set_source(spec.source)
        if not spec.cutoffs and len(values):
            spec.cutoffs = [float(np.quantile(values, 0.5))]
            spec.labels = ['lo', 'hi']
        self._slider.set_bands(spec.cutoffs, spec.labels)
        self._update_counts()

    def _update_counts(self) -> None:
        spec = self._spec_for(self.tag_name)
        counts = self.tags.band_counts(spec, self.scope_key)
        self._counts.setText("  ".join(f"{label}: {n}"
                                       for label, n in zip(spec.labels, counts)))

    def _reload_table(self) -> None:
        table = self.nd.tag_table(self.tag_name)
        if self.scope_key is not None:
            table = table[table['session'] == str(self.scope_key.session)]
        self._table.setRowCount(len(table))
        for row, (_, record) in enumerate(table.iterrows()):
            for col, field in enumerate(['session', 'neuron_id', 'label', 'value']):
                self._table.setItem(row, col, QTableWidgetItem(str(record[field])))

    # ── handlers ────────────────────────────────────────────────────────

    def _on_tag_combo(self, _index: int) -> None:
        self._reload_tag()

    def _on_scope_combo(self, _index: int) -> None:
        self._reload_tag()

    def _on_source_combo(self, _index: int) -> None:
        spec = self._spec_for(self.tag_name)
        spec.source = self._source_combo.currentText()
        spec.cutoffs = []
        self.tags.invalidate(spec)
        self._reload_slider()

    def _on_slider_changed(self) -> None:
        spec = self._spec_for(self.tag_name)
        spec.cutoffs = self._slider.cutoffs
        spec.labels = self._slider.labels
        self.tags.invalidate(spec)
        self._update_counts()

    def _on_assign_btn(self) -> None:
        spec = self._spec_for(self.tag_name)
        written = self.tags.apply(spec, self.scope_key)
        if not written:
            QMessageBox.warning(self, "Assign",
                                f"{spec.source!r} has no values in this scope.")
            return
        self.tags.save_if_bound()
        self.nd.save_tags(self.nav.cd.save_path, [self.tag_name], overwrite=True)
        self._reload_table()
        self.nav.selection_changed.emit()

    def _on_delete_btn(self) -> None:
        name = self.tag_name
        if not name:
            return
        confirm = QMessageBox.question(
            self, "Delete tag", f"Delete {name!r} from every session?")
        if confirm != QMessageBox.Yes:
            return
        self.nd.drop_tag(name, self.nav.cd.save_path)
        self.tags.remove_spec(name)
        self.tags.save_if_bound()
        self.refresh()
        self.nav.selection_changed.emit()


def _band_colors(n: int) -> list:
    """One colour per band, cool to warm so order reads off the slider."""
    base = [(56, 108, 176), (127, 160, 200), (191, 191, 140),
            (233, 155, 85), (214, 96, 77), (165, 60, 60)]
    if n <= len(base):
        return base[:n]
    return [base[i * len(base) // n] for i in range(n)]


def _unique_label(labels: list) -> str:
    """A band name not already in use."""
    i = len(labels)
    while f"b{i}" in labels:
        i += 1
    return f"b{i}"
