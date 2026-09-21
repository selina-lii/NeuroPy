"""Neuron view: the session's neurons on the left, their binned firing rates on the right."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtWidgets import (QAbstractItemView, QDoubleSpinBox,
                                    QInputDialog, QListWidget, QListWidgetItem,
                                    QMenu, QSplitter, QVBoxLayout, QWidget)

from neuropy.analyses.utils import UndoRedo
from neuropy.analyses.view_spec import epoch_bin_mask
from neuropy.ui.ccg_panel import WaveformPanelQt
from neuropy.ui.ui_common import (SelectionCommand, cell_areas, qt_dark_mode,
                                  row_dots)
from neuropy.ui.pair_selection_panel import (TagRowDelegate, _ROLE_AREAS,
                                             _ROLE_CHIPS, _ROLE_PAIR)
from neuropy.ui.utils import (AddableDropdown, CycleButton, HotkeyTagFilter,
                              all_groups_dropdown,
                              apply_flat_aspect, apply_plot_chrome, chip_button,
                              plot_pen, row_chips, widget_row,
                              TRACE_COLOR, TRACE_COLOR_DARK)


class NeuronListPanel(QWidget, UndoRedo):
    """The session's neurons, one row each, with group chips and tag dots."""

    def __init__(self, nav, on_select, parent=None):
        super().__init__(parent)
        self.__init_undo__()
        self.nav = nav
        self._on_select = on_select
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setItemDelegate(TagRowDelegate(self))
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.customContextMenuRequested.connect(self._on_context_menu)
        self._hotkeys = HotkeyTagFilter(self.list, self.tag_by_hotkey)
        self._areas_cache: dict = {}
        layout.addWidget(self.list)

    @property
    def groups(self):
        return self.nav.root.neuron_groups

    def selected_keys(self) -> list:
        """Key of every highlighted row, or the current one when nothing is highlighted."""
        view = self.nav.view
        rows = [i.data(_ROLE_PAIR) for i in self.list.selectedItems()]
        if not rows and self.list.currentItem() is not None:
            rows = [self.list.currentItem().data(_ROLE_PAIR)]
        return [view.key_of(item) for item in rows]

    def tag_by_hotkey(self, char: str) -> bool:
        """Toggle the group bound to *char* on the highlighted neurons."""
        gname = self.groups.group_for_hotkey(char)
        if gname is None:
            return False
        self.toggle_group(gname)
        return True

    def toggle_group(self, gname: str) -> None:
        """Add the highlighted neurons to *gname*, or remove them if all are already in."""
        keys = self.selected_keys()
        if not keys:
            return
        inside = all(gname in self.groups.groups_for_member(k) for k in keys)
        action = 'remove' if inside else 'add'
        changes = [(gname, str(k.session), k, action) for k in keys]
        self.push_undo(SelectionCommand(pair_changes={}, group_changes=changes))
        self._apply_group_changes(changes)

    def _apply_group_changes(self, changes: list, reverse: bool = False) -> None:
        """Run each (group, session, key, action), inverted when *reverse*."""
        for gname, _sess, key, action in changes:
            adding = (action == 'add') != reverse
            if adding:
                self.groups.add_member(gname, key)
            else:
                self.groups.discard_member(gname, key)
        self.groups.save()
        self.refresh()
        self.nav.selection_changed.emit()

    def apply_command(self, cmd, reverse: bool = False) -> None:
        """UndoRedo hook: replay or invert one tagging action."""
        self._apply_group_changes(cmd.group_changes, reverse=reverse)

    def _on_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        tag_menu = QMenu("Group tag", menu)
        keys = self.selected_keys()
        all_groups_dropdown(
            self.groups, tag_menu,
            lambda g: all(g in self.groups.groups_for_member(k) for k in keys),
            self.toggle_group)
        menu.addMenu(tag_menu)
        menu.addAction("New group…", self._new_group)
        menu.exec(self.list.mapToGlobal(pos))

    def _new_group(self) -> None:
        name, ok = QInputDialog.getText(self, "New neuron group", "Name:")
        if ok and name.strip():
            self.groups.create_group(name.strip())
            self.toggle_group(name.strip())

    def refresh(self) -> None:
        """Rebuild every row from the view's items, keeping the cursor where it was."""
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        view = self.nav.view
        for item in view.items(self.nav.key):
            key = view.key_of(item)
            entry = QListWidgetItem(view.row_label(item))
            entry.setData(_ROLE_PAIR, item)
            entry.setData(_ROLE_CHIPS, row_chips(self.groups, key))
            entry.setData(_ROLE_AREAS, self._dots(key, item))
            self.list.addItem(entry)
        self.list.setCurrentRow(min(max(row, 0), self.list.count() - 1))
        self.list.blockSignals(False)

    def _dots(self, key, item) -> list:
        nd_key = key.nd()
        if nd_key not in self._areas_cache:
            self._areas_cache[nd_key] = cell_areas(self.nav.cd.nd.neurons_for(nd_key))
        return row_dots(self._areas_cache[nd_key], key,
                        self.nav.view.neurons_of(item),
                        self.nav.root.settings.area_colors,
                        self.nav.root.neuron_tags)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.nav.set_current_pair(row)
            self._on_select()


class NeuronRatePanel(QWidget):
    """Binned firing rate per neuron, with epoch boundaries drawn over it."""

    def __init__(self, nav, parent=None):
        super().__init__(parent)
        self.nav = nav
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.wf_panel = WaveformPanelQt()
        self.wf_panel.setVisible(False)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.plot_widget)
        split.addWidget(self.wf_panel)
        split.setSizes([700, 300])
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._build_controls())

    def _build_controls(self) -> QWidget:
        self.theme_combo = AddableDropdown('theme', width=140)
        self.theme_combo.currentTextChanged.connect(lambda _t: self.render())
        self.bounds_btn = CycleButton('bounds')
        self.bounds_btn.clicked.connect(self.render)
        self.rate_btn = CycleButton('rate')
        self.rate_btn.clicked.connect(self.render)
        self.bin_spin = QDoubleSpinBox()
        self.bin_spin.setRange(0.001, 60.0)
        self.bin_spin.setValue(0.25)
        self.bin_spin.setSingleStep(0.05)
        self.bin_spin.setSuffix(" s")
        self.bin_spin.setKeyboardTracking(False)   # one render per entry, not per digit
        self.bin_spin.valueChanged.connect(lambda _v: self.render())
        self.wf_btn = chip_button("waveform", checkable=True)
        self.wf_btn.toggled.connect(self._on_wf_btn)
        row = QWidget()
        row.setLayout(widget_row("Theme:", self.theme_combo, None,
                                 self.bounds_btn, self.rate_btn, None,
                                 "Bin:", self.bin_spin, None, self.wf_btn))
        return row

    def _on_wf_btn(self, on: bool) -> None:
        self.wf_panel.setVisible(on)
        self.render()

    def reload_themes(self, names: list) -> None:
        """Offer the themes the time slider already discovered."""
        current = self.theme_combo.currentText()
        self.theme_combo.blockSignals(True)
        self.theme_combo.set_items(names)
        self.theme_combo.setCurrentText(current if current in names else names[0])
        self.theme_combo.blockSignals(False)

    def render(self) -> None:
        """Draw the current neuron's rate trace, then everything layered over it."""
        plot = self.plot_widget.getPlotItem()
        plot.clear()
        dark = qt_dark_mode()
        apply_plot_chrome(plot, dark)
        item = self.nav.current_item
        if item is None:
            return
        key = self.nav.view.key_of(item)
        neurons = self.nav.cd.nd.neurons_for(key.nd())
        bin_size = self.bin_spin.value()
        index = int(np.flatnonzero(np.asarray(neurons.neuron_ids) == key.ref)[0])
        counts, times = neurons.binned_counts(index, bin_size)
        self._draw_rates(plot, times, counts, bin_size, dark)
        self._draw_boundaries(plot, dark)
        self._draw_label_highlight(plot, times, counts, bin_size, dark)
        plot.setTitle(self._title(key, neurons, index), size='9pt')
        plot.setLabel('bottom', 'Time (s)')
        plot.setLabel('left', 'Spikes / bin')
        apply_flat_aspect(plot, times[-1] - times[0] if len(times) > 1 else 1.0,
                          float(counts.max()) if len(counts) else 1.0)
        if self.wf_panel.isVisible():
            self.wf_panel.render(neurons, index)

    def _draw_rates(self, plot, times, counts, bin_size: float, dark: bool) -> None:
        if not self.rate_btn.show:
            return
        color = TRACE_COLOR_DARK if dark else TRACE_COLOR
        if self.rate_btn.line:
            plot.plot(times, counts, pen=plot_pen(color))
        else:
            plot.addItem(pg.BarGraphItem(x=times, height=counts, width=bin_size,
                                         brush=pg.mkBrush(color), pen=None))

    def _draw_boundaries(self, plot, dark: bool) -> None:
        """Epoch edges as lines, or whole epochs as blocks, per the tri-state button."""
        if not self.bounds_btn.show:
            return
        bounds = self._bounds()
        if self.bounds_btn.line:
            color = '#e74c3c' if dark else '#c0392b'
            for start, stop, _label in bounds:
                for edge in (start, stop):
                    plot.addItem(pg.InfiniteLine(pos=edge, angle=90,
                                                 pen=plot_pen(color, Qt.PenStyle.DashLine)))
        else:
            brush = (80, 100, 140, 60) if dark else (200, 220, 255, 60)
            for start, stop, _label in bounds:
                plot.addItem(pg.LinearRegionItem(values=[start, stop],
                                                 brush=pg.mkBrush(*brush),
                                                 pen=pg.mkPen(None), movable=False))

    def _draw_label_highlight(self, plot, times, counts, bin_size, dark: bool) -> None:
        """Shade the trace itself over bins inside the selected labels."""
        bounds = self._bounds()
        if not bounds or not self.rate_btn.show:
            return
        mask = epoch_bin_mask(times, bounds)
        if not mask.any():
            return
        color = '#3ecf6e' if dark else '#1a6b2e'
        plot.addItem(pg.BarGraphItem(x=times[mask], height=counts[mask], y0=0,
                                     width=bin_size, brush=pg.mkBrush(color), pen=None))

    def _bounds(self) -> list:
        """The time slider's resolved epochs for the chosen theme; it already filtered labels."""
        if self.theme_combo.currentText() in ('', 'segments'):
            return []
        return self.nav.root.time_slider.active_bounds

    def _title(self, key, neurons, index: int) -> str:
        kind = '' if neurons.neuron_type is None else f" ({neurons.neuron_type[index]})"
        return f"{key.session}  neuron {key.ref}{kind}"


class NeuronViewPanel:
    """The two panels of the neuron view, and the render they share."""

    def __init__(self, nav):
        self.nav = nav
        self.plot_panel = NeuronRatePanel(nav)
        self.list_panel = NeuronListPanel(nav, self.plot_panel.render)
        slider = nav.root.time_slider
        self.plot_panel.reload_themes(slider.theme_names)
        slider.theme_changed.connect(self._on_theme_changed)
        slider.window_changed.connect(lambda _a, _b: self.plot_panel.render())

    def render(self) -> None:
        """Redraw the plot; the list only rebuilds when its rows actually changed."""
        self.plot_panel.render()

    def rebuild(self) -> None:
        self.list_panel.refresh()
        self.plot_panel.render()

    def _on_theme_changed(self) -> None:
        slider = self.nav.root.time_slider
        self.plot_panel.reload_themes(slider.theme_names)
        self.plot_panel.render()
