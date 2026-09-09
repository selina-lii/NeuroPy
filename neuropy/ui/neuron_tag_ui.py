"""Manage page and painting helpers for per-neuron gradient tags.

Self-contained: the host wires it in with ``install(ui)``, which adds the page to
Manage Groups and the dots to the pair list without either file knowing the
details of a gradient tag.
"""
from __future__ import annotations

from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtGui import QColor
from pyqtgraph.Qt.QtWidgets import (QComboBox, QHBoxLayout,
                                    QInputDialog, QLabel, QMessageBox,
                                    QPushButton, QSpinBox, QTreeWidget,
                                    QTreeWidgetItem, QVBoxLayout, QWidget)

from neuropy.analyses.neuron_tags import (FIRING_RATE, NORMALIZERS, SCOPE_ALL,
                                          SCOPE_SESSION, NeuronTagSet,
                                          NeuronTagSpec)
from neuropy.ui.utils import ColorLabelButton


class NeuronTagPage(QWidget):
    """Tag list with per-label unfold, plus the editor for the selected tag."""

    def __init__(self, tags: NeuronTagSet, nav, parent=None):
        super().__init__(parent)
        self.tags = tags
        self.nav = nav
        self._current: NeuronTagSpec | None = None
        self._build()
        self._reload()

    def _build(self):
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(['Tag', 'n'])
        self._tree.setColumnWidth(0, 170)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        left.addWidget(self._tree, stretch=1)

        btns = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self._on_add_btn)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._on_remove_btn)
        btns.addWidget(add)
        btns.addWidget(rm)
        left.addLayout(btns)
        lay.addLayout(left, stretch=1)

        self._editor = QWidget()
        el = QVBoxLayout(self._editor)

        self._source = QComboBox()
        self._source.currentTextChanged.connect(self._on_definition_changed)
        el.addLayout(_row("Source:", self._source))

        self._segment = QComboBox()
        self._segment.currentTextChanged.connect(self._on_definition_changed)
        el.addLayout(_row("Segment:", self._segment))

        self._norm = QComboBox()
        self._norm.addItems([n or '(none)' for n in NORMALIZERS])
        self._norm.currentTextChanged.connect(self._on_definition_changed)
        el.addLayout(_row("Normalize:", self._norm))

        self._scope = QComboBox()
        self._scope.addItems([SCOPE_ALL, SCOPE_SESSION])
        self._scope.currentTextChanged.connect(self._on_definition_changed)
        el.addLayout(_row("Percentile over:", self._scope))

        self._nbins = QSpinBox()
        self._nbins.setRange(2, 6)
        self._nbins.valueChanged.connect(self._on_bins_changed)
        el.addLayout(_row("Bins:", self._nbins))

        self._color_btn = ColorLabelButton((0, 0, 0), "Base colour", title="Base colour")
        self._color_btn.color_changed.connect(self._on_color_btn)
        el.addWidget(self._color_btn)

        self._notes = QLabel("")
        self._notes.setWordWrap(True)
        self._notes.setStyleSheet("color: #888;")
        el.addWidget(self._notes)
        el.addStretch(1)
        lay.addWidget(self._editor, stretch=1)

    # ── list ────────────────────────────────────────────────────────────

    def _reload(self):
        self._tree.blockSignals(True)
        self._tree.clear()
        key = self.nav.key
        for spec in self.tags.specs.values():
            top = QTreeWidgetItem([spec.prefix, ''])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable)
            top.setCheckState(0, Qt.Checked if spec.enabled else Qt.Unchecked)
            top.setData(0, Qt.UserRole, (spec.prefix, None))
            counts = self.tags.counts(spec, key)
            colors = spec.colors()
            for label in spec.labels:
                child = QTreeWidgetItem([f"{spec.prefix}_{label}",
                                         str(counts.get(label, 0))])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Checked if spec.label_enabled(label)
                                    else Qt.Unchecked)
                child.setForeground(0, QColor(*colors[label]))
                child.setData(0, Qt.UserRole, (spec.prefix, label))
                top.addChild(child)
            self._tree.addTopLevelItem(top)
        self._tree.expandAll()
        self._tree.blockSignals(False)

    def _on_item_changed(self, item, _column):
        prefix, label = item.data(0, Qt.UserRole)
        spec = self.tags.specs.get(prefix)
        if spec is None:
            return
        on = item.checkState(0) == Qt.Checked
        if label is None:
            spec.enabled = on
        else:
            spec.enabled_labels[label] = on
        self._emit_changed()

    def _on_current_changed(self, item, _prev):
        if item is None:
            return
        prefix, _label = item.data(0, Qt.UserRole)
        self._current = self.tags.specs.get(prefix)
        self._load_editor()

    # ── editor ──────────────────────────────────────────────────────────

    def _load_editor(self):
        spec = self._current
        self._editor.setEnabled(spec is not None)
        if spec is None:
            return
        for widget in (self._source, self._segment, self._norm,
                       self._scope, self._nbins):
            widget.blockSignals(True)

        self._source.clear()
        self._source.addItems(self.tags.available_sources(self.nav.key))
        self._source.setCurrentText(spec.source)

        self._segment.clear()
        self._segment.addItem('(whole session)')
        self._segment.addItems([s for s in self.nav.cd.available_segments(self.nav.key)
                                if s != 'all'])
        self._segment.setCurrentText(spec.segment or '(whole session)')
        self._segment.setEnabled(spec.source == FIRING_RATE)

        self._norm.setCurrentText(spec.normalization or '(none)')
        self._scope.setCurrentText(spec.scope)
        self._nbins.setValue(len(spec.labels))

        for widget in (self._source, self._segment, self._norm,
                       self._scope, self._nbins):
            widget.blockSignals(False)

        self._color_btn.set_color(tuple(int(c) for c in spec.base_rgb))
        self._notes.setText(self._notes_text(spec))

    def _notes_text(self, spec: NeuronTagSpec) -> str:
        """Where the cuts landed; session-scope edges are per session, so resolve them here."""
        edges = self.tags.edges_for(spec, self.nav.key)
        if not edges:
            return f"{spec.source}: not available in this session"
        unit = {'zscore': 'z', 'log10': 'log10', 'rank': 'pct'}.get(spec.normalization, 'raw')
        source = spec.source + (f"/{spec.segment}" if spec.segment else '')
        scope = 'this session' if spec.scope == SCOPE_SESSION else 'all sessions'
        cuts = ', '.join(f"{e:.3g}" for e in edges)
        return f"{source} [{unit}], percentiles over {scope} — cuts at {cuts}"

    def _on_definition_changed(self, *_):
        spec = self._current
        if spec is None:
            return
        spec.source = self._source.currentText()
        segment = self._segment.currentText()
        spec.segment = '' if segment.startswith('(') else segment
        norm = self._norm.currentText()
        spec.normalization = '' if norm == '(none)' else norm
        spec.scope = self._scope.currentText()
        self.tags.invalidate(spec)
        self._segment.setEnabled(spec.source == FIRING_RATE)
        self._after_edit()

    def _on_bins_changed(self, n: int):
        spec = self._current
        if spec is None:
            return
        spec.labels = _default_labels(n)
        spec.quantiles = [(i + 1) / n for i in range(n - 1)]
        spec.enabled_labels = {}
        self.tags.invalidate(spec)
        self._after_edit()

    def _on_color_btn(self, _color: str):
        spec = self._current
        if spec is None:
            return
        spec.base_rgb = self._color_btn.rgb
        self._after_edit()

    def _on_add_btn(self):
        name, ok = QInputDialog.getText(self, "New neuron tag", "Prefix (e.g. frate):")
        name = name.strip()
        if not ok or not name:
            return
        if name in self.tags.specs:
            QMessageBox.warning(self, "New neuron tag", f"{name!r} already exists.")
            return
        self.tags.add_spec(NeuronTagSpec(prefix=name))
        self._after_edit()

    def _on_remove_btn(self):
        if self._current is None:
            return
        self.tags.remove_spec(self._current.prefix)
        self._current = None
        self._after_edit()

    def _after_edit(self):
        self._reload()
        self._load_editor()
        self._emit_changed()

    def _emit_changed(self):
        self.tags.save()
        self.nav.selection_changed.emit()


def _row(label: str, widget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(widget, stretch=1)
    return row


def _default_labels(n: int) -> list:
    """Bin names for an n-way cut, coarse enough to read in a chip."""
    return {2: ['lo', 'hi'],
            3: ['lo', 'mid', 'hi'],
            4: ['q1', 'q2', 'q3', 'q4']}.get(n, [f'b{i + 1}' for i in range(n)])
