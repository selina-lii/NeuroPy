"""Panel for assigned neuron tags: list them, read them, and cut a value range into labelled bands."""
from __future__ import annotations

import numpy as np
from pyqtgraph.Qt.QtCore import QPointF, QRectF, Qt, Signal
from pyqtgraph.Qt.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from pyqtgraph.Qt.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                                    QDoubleSpinBox, QFormLayout,
                                    QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox,
                                    QPushButton, QSizePolicy, QTableWidget,
                                    QTableWidgetItem, QVBoxLayout, QWidget)

from neuropy.analyses.neuron_tags import (CUT_ABSOLUTE, NeuronTagSpec,
                                          NeuronTagSet, activity_source, tag_value_source)
from neuropy.ui.utils import (CheckMenuButton, ExplicitCloseDialog, cascade_menu,
                              make_button, pick_color)

NEW_TAG_MENU = {'activity': {'frate': {'during': 'frate_during', 'ratio': 'frate_ratio'}}}

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
        self.overrides: dict = {}   # label -> picked rgb
        self._source = ''
        self._dragging = None
        self._log = False

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

    def set_log(self, on: bool) -> None:
        """Draw the value axis logarithmically; cutoff values themselves stay raw."""
        self._log = bool(on)
        self.update()

    def set_source(self, name: str) -> None:
        """What is being cut, printed under the value ticks."""
        self._source = name
        self.update()

    def set_bands(self, cutoffs: list, labels: list, colors: list = None, overrides: dict = None) -> None:
        self._cutoffs = [float(c) for c in cutoffs]
        self._labels = list(labels)
        self._colors = list(colors) if colors else _band_colors(len(labels))
        self.overrides = dict(overrides or {})
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

    def set_cutoff(self, index: int, value: float) -> int:
        """Move one cutoff, keeping cutoffs sorted and bands with them; returns its new index."""
        self._cutoffs[index] = float(value)
        order = list(np.argsort(self._cutoffs))
        if order != sorted(order):
            self._labels = [self._labels[0]] + [self._labels[i + 1] for i in order]
            self._cutoffs = sorted(self._cutoffs)
            index = order.index(index)
        self.changed.emit()
        self.update()
        return index

    def band_color(self, band: int) -> tuple:
        return tuple(self.overrides.get(self._labels[band], self._colors[band]))

    def rename_band(self, index: int, name: str) -> None:
        if self._labels[index] in self.overrides:
            self.overrides[name] = self.overrides.pop(self._labels[index])
        self._labels[index] = name
        self.changed.emit()
        self.update()

    def widest_band_midpoint(self) -> float:
        """Where a new cutoff does the most good: the middle of the largest band."""
        bounds = [self._lo] + self._cutoffs + [self._hi]
        spans = [(b - a, (a + b) / 2) for a, b in zip(bounds, bounds[1:])]
        return max(spans)[1]

    def band_of(self, x: float) -> int:
        """Which band a pixel x falls in."""
        return int(np.digitize([self._value_at(x)], self._cutoffs)[0])

    # ── geometry ────────────────────────────────────────────────────────

    @property
    def _draw_lo(self) -> float:
        """Log drawing cannot reach zero, so the axis starts at the smallest positive value."""
        if not self._log:
            return self._lo
        pos = self._values[self._values > 0]
        return float(pos.min()) if len(pos) else 1e-12

    def _fwd(self, value: float) -> float:
        return np.log10(max(value, self._draw_lo)) if self._log else value

    def _inv(self, t: float) -> float:
        return float(10.0 ** t) if self._log else t

    def _x_of(self, value: float) -> float:
        lo, hi = self._fwd(self._draw_lo), self._fwd(self._hi)
        span = (hi - lo) or 1.0
        return 4 + (self._fwd(value) - lo) / span * max(1, self.width() - 8)

    def _value_at(self, x: float) -> float:
        lo, hi = self._fwd(self._draw_lo), self._fwd(self._hi)
        span = max(1, self.width() - 8)
        return self._inv(lo + (x - 4) / span * (hi - lo))

    def _handle_at(self, pos) -> int | None:
        for i, cut in enumerate(self._cutoffs):
            if abs(pos.x() - self._x_of(cut)) <= GRAB_PX and pos.y() >= TRACK_H - HANDLE_H:
                return i
        return None

    # ── events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = self._handle_at(event.position())
        elif event.button() == Qt.RightButton:
            hit = self._handle_at(event.position())
            if hit is not None:
                menu = QMenu(self)
                act = menu.addAction("Remove")
                if menu.exec(event.globalPosition().toPoint()) is act:
                    self.remove_cutoff(hit)

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            over = self._handle_at(event.position())
            self.setCursor(Qt.SizeHorCursor if over is not None else Qt.ArrowCursor)
            return
        value = min(self._hi, max(self._lo, self._value_at(event.position().x())))
        self._dragging = self.set_cutoff(self._dragging, value)

    def mouseReleaseEvent(self, _event):
        self._dragging = None

    def mouseDoubleClickEvent(self, event):
        band = self.band_of(event.position().x())
        cut = [band - 1 if band else 0]    # a band is bounded below by the cutoff before it
        dlg, name = ExplicitCloseDialog(self), QLineEdit(self._labels[band])
        form = QFormLayout(dlg)
        form.addRow("Label:", name)
        name.editingFinished.connect(
            lambda: name.text().strip() and self.rename_band(band, name.text().strip()))
        if self._cutoffs:
            value = QDoubleSpinBox(decimals=6, minimum=-1e18, maximum=1e18)
            value.setValue(self._cutoffs[cut[0]])
            value.editingFinished.connect(
                lambda: cut.__setitem__(0, self.set_cutoff(cut[0], value.value())))
            form.addRow("Lower bound:" if band else "Upper bound:", value)
        swatch = QPushButton()
        swatch.setAutoDefault(False)
        swatch.setStyleSheet("background: rgb%s;" % (self.band_color(band),))

        def on_swatch_btn():
            c = pick_color(self.band_color(band), dlg)
            if c is not None:
                self.overrides[self._labels[band]] = c.getRgb()[:3]
                swatch.setStyleSheet("background: rgb%s;" % (self.band_color(band),))
                self.changed.emit()
                self.update()
        swatch.clicked.connect(on_swatch_btn)
        form.addRow("Colour:", swatch)
        close = QPushButton("Close")
        close.setAutoDefault(False)
        close.clicked.connect(dlg.accept)
        form.addRow(close)
        dlg.exec()

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
        lo = self._draw_lo
        bins = (np.logspace(np.log10(lo), np.log10(self._hi), 49) if self._log
                else np.linspace(self._lo, self._hi, 49))
        counts, edges = np.histogram(self._values, bins=bins)
        tallest = max(1, counts.max())
        painter.setPen(Qt.NoPen)
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            band = int(np.digitize([(left + right) / 2], self._cutoffs)[0])
            height = count / tallest * (TRACK_H - 8)
            painter.setBrush(QBrush(QColor(*self.band_color(band))))
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
            painter.setPen(QColor(*self.band_color(i)))
            painter.drawText(QRectF(left, TRACK_H + 15, max(10.0, right - left), 14),
                             Qt.AlignCenter, label)

    def _paint_value_axis(self, painter: QPainter) -> None:
        """The value range under the track: min, midpoint, max, and the source cut."""
        row = QRectF(4, TRACK_H + 30, self.width() - 8, 13)
        painter.setPen(QColor(140, 140, 140))
        lo = self._draw_lo
        painter.drawText(row, Qt.AlignLeft, f"{lo:.4g}")
        painter.drawText(row, Qt.AlignCenter, f"{self._value_at(self.width() / 2):.4g}")
        painter.drawText(row, Qt.AlignRight, f"{self._hi:.4g}")
        if self._source:
            painter.drawText(QRectF(4, TRACK_H + 43, self.width() - 8, 13),
                             Qt.AlignCenter, self._source + (" (log)" if self._log else ""))


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

        self._new_btn = make_button("New tag ▾", self._on_new_btn)
        top.addWidget(self._new_btn)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._on_delete_btn)
        top.addWidget(delete)
        top.addStretch(1)

        top.addWidget(QLabel("Scope:"))
        self._scope_combo = QComboBox()
        self._scope_combo.setMinimumWidth(170)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_combo)
        top.addWidget(self._scope_combo)
        self._type_combo = QComboBox()
        self._type_combo.currentIndexChanged.connect(self._on_type_combo)
        top.addWidget(self._type_combo)
        outer.addLayout(top)

        source = QHBoxLayout()
        source.addStretch(1)
        self._counts = QLabel("")
        self._counts.setStyleSheet("color: #888;")
        source.addWidget(self._counts)
        outer.addLayout(source)

        self._slider = CutoffSlider()
        self._slider.changed.connect(self._on_slider_changed)
        outer.addWidget(self._slider)

        tools = QHBoxLayout()
        add_cut = QPushButton("+ Add cutoff")
        add_cut.clicked.connect(self._on_add_cutoff_btn)
        tools.addWidget(add_cut)
        self._log_btn = QPushButton("log")
        self._log_btn.setCheckable(True)
        self._log_btn.setFixedWidth(46)
        self._log_btn.toggled.connect(self._on_log_btn)
        tools.addWidget(self._log_btn)
        tools.addWidget(make_button("Recompute", self._on_recompute_btn))
        tools.addStretch(1)
        outer.addLayout(tools)

        hint = QLabel("drag to move · right-click a handle to remove "
                      "· double-click a band to rename and set its cutoff")
        hint.setStyleSheet("color: #888;")
        outer.addWidget(hint)

        assign = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888;")
        assign.addWidget(self._status)
        assign.addStretch(1)
        assign.addWidget(make_button("Preview", self._on_preview_btn))
        self._assign_btn = QPushButton("Assign")
        self._assign_btn.clicked.connect(self._on_assign_btn)
        assign.addWidget(self._assign_btn)
        outer.addLayout(assign)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(['Session', 'Neuron', 'Label', 'Value'])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSortingEnabled(True)
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
        if spec is None:   # a tag supplied from outside cuts on its own stored values
            spec = NeuronTagSpec(prefix=name, source=tag_value_source(name), cut_mode=CUT_ABSOLUTE)
            self.tags.add_spec(spec)
        spec.cut_mode = CUT_ABSOLUTE
        return spec

    # ── refresh ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild every list from the dataset, keeping the current tag if it survives."""
        keep = self.tag_name
        for combo in (self._tag_list, self._scope_combo, self._type_combo):
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

        self._type_combo.clear()
        self._type_combo.addItem("All types", '')
        for t in sorted({str(t) for k in self.nd.session_keys
                         for t in self.nd.neurons_for(k).neuron_type}):
            self._type_combo.addItem(t, t)

        for combo in (self._tag_list, self._scope_combo, self._type_combo):
            combo.blockSignals(False)
        self._reload_tag()

    def _reload_tag(self) -> None:
        name = self.tag_name
        has_tag = bool(name)
        for widget in (self._slider, self._assign_btn, self._log_btn):
            widget.setEnabled(has_tag)
        if not has_tag:
            self._table.setRowCount(0)
            self._counts.setText("")
            return
        self._type_combo.blockSignals(True)
        self._type_combo.setCurrentIndex(self._type_combo.findData(self._spec_for(name).cell_type))
        self._type_combo.blockSignals(False)
        self._reload_slider()
        self._reload_table()

    def _reload_slider(self) -> None:
        spec = self._spec_for(self.tag_name)
        values = self.tags.pooled_values(spec, self.scope_key)
        self._slider.set_values(values)
        self._slider.set_source(spec.source)
        if not spec.cutoffs and len(values):
            spec.cutoffs = [float(np.quantile(values, 0.5))]
            spec.labels = ['lo', 'hi']
        self._slider.set_bands(spec.cutoffs, spec.labels, overrides=spec.label_colors)
        self._log_btn.setChecked(spec.log_axis)
        self._slider.set_log(spec.log_axis)
        self._update_counts()

    def _update_counts(self) -> None:
        spec = self._spec_for(self.tag_name)
        counts = self.tags.band_counts(spec, self.scope_key)
        self._counts.setText("  ".join(f"{label}: {n}"
                                       for label, n in zip(spec.labels, counts)))

    def _reload_table(self) -> None:
        table = self.tags.preview(self._spec_for(self.tag_name), self.scope_key)
        self._table.setSortingEnabled(False)    # else each setItem re-sorts mid-fill
        self._table.setRowCount(len(table))
        for row, record in enumerate(table[['session', 'neuron_id', 'label', 'value']].values.tolist()):
            for col, value in enumerate(record):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value if isinstance(value, (int, float)) else str(value))
                self._table.setItem(row, col, item)
        self._table.setSortingEnabled(True)

    # ── handlers ────────────────────────────────────────────────────────

    def _on_tag_combo(self, _index: int) -> None:
        self._reload_tag()

    def _on_scope_combo(self, _index: int) -> None:
        self._reload_tag()

    def _on_type_combo(self, _index: int) -> None:
        spec = self._spec_for(self.tag_name)
        spec.cell_type = self._type_combo.currentData()
        self.tags.invalidate(spec)
        self._reload_slider()
        self._reload_table()

    def _on_preview_btn(self) -> None:
        self._reload_table()

    def _on_recompute_btn(self) -> None:
        self.tags.clear_values()
        self._reload_slider()
        self._warn_missing()

    def _on_new_btn(self) -> None:
        cascade_menu(NEW_TAG_MENU, self._on_new_tag_pick, self).exec(
            self._new_btn.mapToGlobal(self._new_btn.rect().bottomLeft()))

    def _on_new_tag_pick(self, fn: str) -> None:
        dlg = ActivityTagDialog(fn, _segment_tree(self.nav.cd), self)
        if not dlg.exec():
            return
        spec = NeuronTagSpec(prefix=dlg.name.text().strip(), source=dlg.source, cut_mode=CUT_ABSOLUTE)
        self.tags.add_spec(spec)
        values = self.tags.pooled_values(spec)
        spec.cutoffs = [float(np.median(values))] if len(values) else []
        if self._assign(spec):
            self.refresh()
            self._tag_list.setCurrentText(spec.prefix)

    def _warn_missing(self) -> None:
        missing = self.tags.take_missing()
        if missing:
            body = '\n'.join(f"{seg}: {', '.join(sorted(s))}" for seg, s in sorted(missing.items()))
            QMessageBox.warning(self, "Segment missing for some sessions",
                                f"These sessions lack a segment, so their neurons stay untagged:\n\n{body}")

    def _on_log_btn(self, on: bool) -> None:
        self._spec_for(self.tag_name).log_axis = on
        self._slider.set_log(on)
        self.tags.save_if_bound()

    def _on_add_cutoff_btn(self) -> None:
        self._slider.add_cutoff(self._slider.widest_band_midpoint())

    def _on_slider_changed(self) -> None:
        spec = self._spec_for(self.tag_name)
        spec.cutoffs = self._slider.cutoffs
        spec.labels = self._slider.labels
        colors = {lb: c for lb, c in self._slider.overrides.items() if lb in spec.labels}
        recolored, spec.label_colors = colors != spec.label_colors, colors
        self.tags.invalidate(spec)
        if recolored:
            self.nav.refresh_lists()
        self._update_counts()

    def _on_assign_btn(self) -> None:
        self._assign(self._spec_for(self.tag_name), self.scope_key)

    def _assign(self, spec: NeuronTagSpec, key=None) -> bool:
        written = self.tags.apply(spec, key)
        self._warn_missing()
        skipped = [str(k.session) for k in ([key] if key is not None else self.nd.session_keys)
                   if str(k.session) not in written]
        self._status.setText(f"Assigned {spec.prefix} to {len(written)} session(s)"
                             + (f" · no values: {', '.join(skipped)}" if skipped else ''))
        if not written:
            QMessageBox.warning(self, "Assign", f"{spec.source!r} has no values in this scope.")
            return False
        self.tags.save_if_bound()
        self.nd.save_tags(self.nav.cd.save_path, [spec.prefix], overwrite=True)
        self._reload_table()
        self.nav.refresh_lists()
        return True

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
        self.nav.refresh_lists()


class ActivityTagDialog(ExplicitCloseDialog):
    """Name a new activity tag and tick the segments of each group it compares."""

    def __init__(self, fn: str, tree: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"New activity tag — {fn}")
        self.fn = fn
        form = QFormLayout(self)
        self.name = QLineEdit()
        form.addRow("Name:", self.name)
        self.groups = [CheckMenuButton(tree) for _ in range(2 if fn == 'frate_ratio' else 1)]
        for letter, group in zip("AB", self.groups):
            group.changed.connect(self._sync)
            form.addRow(f"Group {letter}:", group)
        self.expr = QLabel()
        form.addRow(self.expr)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(make_button("Cancel", self.reject))
        buttons.addWidget(make_button("Create", self._on_create_btn))
        form.addRow(buttons)
        self._sync()

    @property
    def source(self) -> str:
        return activity_source(self.fn, [g.checked for g in self.groups])

    def _sync(self) -> None:
        self.expr.setText(self.source)

    def _on_create_btn(self) -> None:
        picked = [g.checked for g in self.groups]
        problem = ("Name the tag." if not self.name.text().strip() else
                   "Tick at least one segment in every group." if not all(picked) else
                   "'none' stands alone in its group." if any('none' in g and len(g) > 1 for g in picked) else
                   "'none' needs a real segment group to invert." if all(g == ['none'] for g in picked) else '')
        if problem:
            QMessageBox.warning(self, "New activity tag", problem)
            return
        self.accept()


def _segment_tree(cd) -> dict:
    """Custom CCGs and epoch labels (by theme) for the group menus, plus 'none'."""
    epochs = {theme: {lb: lb for lb in sorted({str(x).strip() for x in ep.labels} - {''}) or [theme]}
              for theme, ep in cd.nd.get_themes_any().items()}
    return {'Custom CCG': {s: s for s in cd.available_segments() if s != 'all'},
            'Epochs': epochs, 'none': 'none'}


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
