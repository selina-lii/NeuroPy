"""Neuron view: the session's neurons on the left, their binned firing rates on the right."""
from __future__ import annotations

from collections import defaultdict as _defaultdict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtGui import QBrush
from pyqtgraph.Qt.QtWidgets import (QAbstractItemView, QDoubleSpinBox, QHBoxLayout, QLabel,
                                    QListWidget, QListWidgetItem,
                                    QMenu, QSlider, QSplitter, QVBoxLayout, QWidget)

from neuropy.analyses.utils import UndoRedo
from neuropy.analyses.view_spec import NEURON_VIEW, epoch_bin_mask
from neuropy.ui.ccg_panel import WaveformPanelQt
from neuropy.ui.ui_common import (CollapseState, SelectionCommand, areas_by_id,
                                  group_header_label, is_special_group, qt_dark_mode,
                                  row_dots)
from neuropy.ui.pair_selection_panel import (PairSelectionPanel, TagRowDelegate, _C_GRAY_FG,
                                             _C_HDR_BG, _C_HDR_FG, _ROLE_AREAS,
                                             _ROLE_CHIPS, _ROLE_HKEY, _ROLE_PAIR)
from neuropy.ui.utils import (AddableDropdown, CycleButton, ExclusiveButtonSet, HotkeyTagFilter,
                              all_groups_dropdown, make_button,
                              apply_plot_chrome, chip_button,
                              plot_pen, prompt_name, row_chips, widget_row, TagChip,
                              TRACE_COLOR, TRACE_COLOR_DARK)

OVERLAP_RGB = (62, 207, 110)   # the cross-theme overlap: one colour, not per label


_SORTS = (('group', 'Group'), ('tag', 'Tag'), ('rate', 'Mean rate'), ('ntag', 'Neuron tag'))


class NeuronListPanel(QWidget, UndoRedo):
    """The session's neurons as available and selected lists, with group chips and tag dots."""

    def __init__(self, nav, on_select, parent=None):
        super().__init__(parent)
        self.__init_undo__()
        self.nav = nav
        self._on_select = on_select
        self._index: dict = {}
        self._areas_cache: dict = {}
        self._collapsed = CollapseState()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        delegate = TagRowDelegate(self)
        self.avail_label, self.avail_list = self._add_column(splitter, delegate)
        self.sel_label, self.sel_list = self._add_column(splitter, delegate)
        layout.addWidget(splitter, stretch=1)

        self._sort_btns = ExclusiveButtonSet(on_change=self.refresh)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sort by:"))
        for key, label in _SORTS:
            btn, _var = self._sort_btns.add(key, label)
            row.addWidget(btn)
        row.addStretch()
        row.addWidget(make_button("Admit all", self._on_admit_all_btn))
        layout.addLayout(row)

    def _add_column(self, splitter, delegate) -> tuple:
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setSpacing(1)
        label = QLabel()
        lst = QListWidget()
        lst.setItemDelegate(delegate)
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lst.setContextMenuPolicy(Qt.CustomContextMenu)
        lst.currentItemChanged.connect(self._on_current_item)
        lst.itemDoubleClicked.connect(self._on_item_double_clicked)
        lst.customContextMenuRequested.connect(lambda pos, w=lst: self._on_context_menu(w, pos))
        HotkeyTagFilter(lst, self.tag_by_hotkey, delete=self._on_delete_key,
                        admit=lambda w=lst: self._on_item_double_clicked(w.currentItem()))
        col_layout.addWidget(label)
        col_layout.addWidget(lst)
        splitter.addWidget(col)
        return label, lst

    @property
    def groups(self):
        return self.nav.root.neuron_groups

    @property
    def selections(self):
        return self.nav.root.neuron_selections

    def state_of(self, item) -> str:
        key = self.nav.view.key_of(item)
        b = self.selections.bucket(key)
        return 'sel' if (key.ref,) in b.selected else 'del' if (key.ref,) in b.deleted else 'unsel'

    def _highlighted(self) -> list:
        """Items highlighted in either list, or the current one when none is."""
        items = [it.data(_ROLE_PAIR) for lst in (self.avail_list, self.sel_list)
                 for it in lst.selectedItems() if it.data(_ROLE_PAIR) is not None]
        current = self.nav.current_item
        return items or ([] if current is None else [current])

    def selected_keys(self) -> list:
        return [self.nav.view.key_of(item) for item in self._highlighted()]

    # ── state changes ───────────────────────────────────────────────────

    def transition(self, items, new_state: str) -> None:
        """Move *items* to *new_state* as one undoable action."""
        changes = {self.nav.view.key_of(i): (self.state_of(i), new_state)
                   for i in items if self.state_of(i) != new_state}
        if changes:
            cmd = SelectionCommand(pair_changes=changes, group_changes=[])
            self.push_undo(cmd)
            self.apply_command(cmd)

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
        cmd = SelectionCommand(pair_changes={}, group_changes=[
            (gname, str(k.session), k, action) for k in keys])
        self.push_undo(cmd)
        self.apply_command(cmd)

    def apply_command(self, cmd, reverse: bool = False) -> None:
        """UndoRedo hook: replay or invert one selection or tagging action."""
        for key, (old, new) in cmd.pair_changes.items():
            self.selections.bucket(key).set_pair_state((key.ref,), old if reverse else new)
        for gname, _sess, key, action in cmd.group_changes:
            if (action == 'add') != reverse:
                self.groups.add_member(gname, key)
            else:
                self.groups.discard_member(gname, key)
        self.selections.save()
        if cmd.group_changes:
            self.groups.save()
        self.nav.selection_changed.emit()

    # ── handlers ────────────────────────────────────────────────────────

    def _on_current_item(self, current, _previous) -> None:
        item = None if current is None else current.data(_ROLE_PAIR)
        if item is not None:
            self.nav.set_current_pair(self._index[item])
            self._on_select()

    def _on_item_double_clicked(self, it) -> bool:
        """Shuttle a neuron between the lists, or fold a header."""
        if it is None:
            return False
        if it.data(_ROLE_HKEY) is not None:
            self._collapsed.toggle(it.data(_ROLE_HKEY))
            self.refresh()
            return True
        item = it.data(_ROLE_PAIR)
        if item is not None:
            self.transition([item], 'unsel' if self.state_of(item) == 'sel' else 'sel')
        return True

    def _on_delete_key(self) -> bool:
        items = self._highlighted()
        deleted = all(self.state_of(i) == 'del' for i in items)
        self.transition(items, 'unsel' if deleted else 'del')
        return True

    def _on_admit_all_btn(self) -> None:
        self.transition([i for i in self.nav.view.items(self.nav.key)
                         if self.state_of(i) == 'unsel'], 'sel')

    def _on_context_menu(self, lst, pos) -> None:
        if lst.itemAt(pos) is None:
            return
        items = self._highlighted()
        states = {self.state_of(i) for i in items}
        menu = QMenu(self)
        for label, state in (("Move to Selected", 'sel'), ("Move to Available", 'unsel'),
                             ("Move to Deleted", 'del')):
            if states - {state}:
                menu.addAction(label, lambda s=state: self.transition(items, s))
        menu.addSeparator()
        tag_menu = QMenu("Group tag", menu)
        keys = self.selected_keys()
        all_groups_dropdown(
            self.groups, tag_menu,
            lambda g: all(g in self.groups.groups_for_member(k) for k in keys),
            self.toggle_group)
        menu.addMenu(tag_menu)
        menu.addAction("New group…", self._new_group)
        menu.exec(lst.mapToGlobal(pos))

    def _new_group(self) -> None:
        name = prompt_name(self, "New neuron group", "Name:")
        if name:
            self.groups.create_group(name)
            self.toggle_group(name)

    # ── lists ───────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild both lists from the view's items, keeping the cursor on its neuron."""
        view = self.nav.view
        items = view.items(self.nav.key)
        self._index = {item: i for i, item in enumerate(items)}
        by_state = {'sel': [], 'unsel': [], 'del': []}
        for item in items:
            by_state[self.state_of(item)].append(item)
        for lst in (self.avail_list, self.sel_list):
            lst.blockSignals(True)
            lst.clear()
        for item in by_state['unsel']:
            self.avail_list.addItem(self._row(item))
        if by_state['del']:
            self.avail_list.addItem(self._header('deleted', len(by_state['del'])))
            if not self._collapsed.is_collapsed('deleted'):
                for item in by_state['del']:
                    self.avail_list.addItem(self._row(item, gray=True))
        for header, section in self._sections(by_state['sel']):
            if header is not None:
                self.sel_list.addItem(self._header(header, len(section)))
                if self._collapsed.is_collapsed(header):
                    continue
            for item in section:
                self.sel_list.addItem(self._row(item))
        self.avail_label.setText(f"Available ({len(by_state['unsel'])}"
                                 + (f", {len(by_state['del'])} deleted)" if by_state['del'] else ")"))
        self.sel_label.setText(f"Selected ({len(by_state['sel'])})")
        self._restore_cursor()
        for lst in (self.avail_list, self.sel_list):
            lst.blockSignals(False)

    def _restore_cursor(self) -> None:
        current = self.nav.current_item
        for lst in (self.avail_list, self.sel_list):
            for row in range(lst.count()):
                if lst.item(row).data(_ROLE_PAIR) == current:
                    lst.setCurrentRow(row)
                    lst.scrollToItem(lst.item(row))
                    return

    def _row(self, item, gray: bool = False) -> QListWidgetItem:
        key = self.nav.view.key_of(item)
        entry = QListWidgetItem(self.nav.view.row_label(item))
        entry.setData(_ROLE_PAIR, item)
        entry.setData(_ROLE_CHIPS, row_chips(self.groups, key) + self._tag_words(key, item))
        entry.setData(_ROLE_AREAS, self._dots(key, item))
        if gray:
            entry.setForeground(QBrush(_C_GRAY_FG))
        return entry

    def _header(self, name: str, count: int) -> QListWidgetItem:
        entry = QListWidgetItem(group_header_label(name, count, self._collapsed.is_collapsed(name)))
        entry.setForeground(QBrush(_C_HDR_FG))
        entry.setBackground(QBrush(_C_HDR_BG))
        entry.setFlags(Qt.ItemIsEnabled)
        entry.setData(_ROLE_HKEY, name)
        return entry

    def _sections(self, items: list) -> list:
        """Ordered (header|None, items) for the active sort; a neuron may sit in several tag sections."""
        view = self.nav.view
        if self._sort_btns.is_checked('rate'):
            return [(None, sorted(items, key=self._rate, reverse=True))]
        if self._sort_btns.is_checked('tag'):
            buckets = _defaultdict(list)
            for item in items:
                for g in sorted(self._group_names(view.key_of(item))) or ['(untagged)']:
                    buckets[g].append(item)
            return sorted(buckets.items(), key=lambda kv: (kv[0] == '(untagged)', kv[0]))
        if self._sort_btns.is_checked('group'):
            return self._combo_sections(items, self._group_names)
        if self._sort_btns.is_checked('ntag'):
            return self._combo_sections(items, lambda k: [
                w for w, _rgb in self.nav.root.neuron_tags.neuron_words(k, k.ref)])
        return [(None, items)]

    def _combo_sections(self, items: list, names_of) -> list:
        """One section per distinct combination of names, untagged last."""
        buckets = _defaultdict(list)
        for item in items:
            buckets[tuple(sorted(names_of(self.nav.view.key_of(item))))].append(item)
        return [(', '.join(c) if c else '(untagged)', buckets[c])
                for c in sorted(buckets, key=PairSelectionPanel._COMBO_SORT_KEY)]

    def _group_names(self, key) -> list:
        return [g for g in self.groups.groups_for_member(key) if not is_special_group(g)]

    def _rate(self, item) -> float:
        key = self.nav.view.key_of(item)
        neurons = self.nav.cd.nd.neurons_for(key.nd())
        return float(neurons.firing_rate[list(neurons.neuron_ids).index(key.ref)])

    def _tag_words(self, key, item) -> list:
        """Each neuron tag covering the row's neuron as a named pill, tinted like its dot."""
        return [(word, *TagChip.tint('#%02x%02x%02x' % tuple(rgb)))
                for nid in self.nav.view.neurons_of(item)
                for word, rgb in self.nav.root.neuron_tags.neuron_words(key, int(nid))]

    def _dots(self, key, item) -> list:
        nd_key = key.nd()
        if nd_key not in self._areas_cache:
            self._areas_cache[nd_key] = areas_by_id(self.nav.cd.nd.neurons_for(nd_key))
        return row_dots(self._areas_cache[nd_key], key,
                        self.nav.view.neurons_of(item),
                        self.nav.root.settings.area_colors,
                        self.nav.root.neuron_tags)


class NeuronRatePanel(QWidget):
    """Binned firing rates of a rolling window of neurons, one strip each, with epoch bounds over them."""

    def __init__(self, nav, parent=None):
        super().__init__(parent)
        self.nav = nav
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.wf_panel = WaveformPanelQt()
        self.wf_panel.setVisible(False)
        self._split = QSplitter(Qt.Horizontal)
        self._split.setChildrenCollapsible(False)
        self._split.addWidget(self.plot_widget)
        self._split.addWidget(self.wf_panel)
        layout.addWidget(self._split, stretch=1)
        layout.addWidget(self._build_controls())

    def _build_controls(self) -> QWidget:
        self.theme_combo = AddableDropdown('theme', width=140)
        self.theme_combo.currentTextChanged.connect(lambda _t: self.render())
        self.bounds_btn = CycleButton('bounds')
        self.bounds_btn.clicked.connect(self.render)
        self.rate_btn = CycleButton('rate', start_line=True)
        self.rate_btn.clicked.connect(self.render)
        self.bin_spin = QDoubleSpinBox()
        self.bin_spin.setDecimals(3)          # the validator rejects digits past this
        self.bin_spin.setRange(0.001, 1e9)
        self.bin_spin.setValue(300.0)
        self.bin_spin.setSingleStep(30.0)
        self.bin_spin.setSuffix(" s")
        self.bin_spin.setKeyboardTracking(False)   # one render per entry, not per digit
        self.bin_spin.valueChanged.connect(lambda _v: self.render())
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(1, 100)
        self.y_slider.setValue(100)
        self.y_slider.setFixedWidth(100)
        self.y_slider.setToolTip("Y range as a percent of each strip's peak")
        self.y_slider.valueChanged.connect(lambda _v: self.render())
        self.log_btn = chip_button("log y", checkable=True)
        self.log_btn.toggled.connect(lambda _on: self.render())
        self.wf_btn = chip_button("waveform", checkable=True)
        self.wf_btn.toggled.connect(self._on_wf_btn)
        row = QWidget()
        row.setLayout(widget_row("Theme:", self.theme_combo, None,
                                 self.bounds_btn, self.rate_btn, None,
                                 "Bin:", self.bin_spin, None,
                                 "Y:", self.y_slider, self.log_btn, None,
                                 self.wf_btn))
        return row

    def _on_wf_btn(self, on: bool) -> None:
        self.wf_panel.setVisible(on)
        if on:   # sizes given while hidden land at zero width
            self._split.setSizes([700, 300])
        self.render()

    def reload_themes(self, names: list) -> None:
        """Offer the themes the time slider already discovered, defaulting to a real one."""
        current = self.theme_combo.currentText()
        default = names[1] if len(names) > 1 else names[0]   # names[0] is 'segments': no bounds
        self.theme_combo.blockSignals(True)
        self.theme_combo.set_items(names)
        self.theme_combo.setCurrentText(current if current in names else default)
        self.theme_combo.blockSignals(False)

    def render(self) -> None:
        """A strip per neuron in a window around the current one, all on one time axis."""
        lay = self.plot_widget
        lay.clear()
        items = self.nav.view.items(self.nav.key)
        cur = self.nav.current_pair_idx
        if not 0 <= cur < len(items):
            return
        n = max(1, int(self.nav.neuron_view_count))
        lo = min(max(cur - n // 2, 0), max(len(items) - n, 0))
        window = list(range(lo, min(lo + n, len(items))))
        dark = qt_dark_mode()
        groups = self._bound_groups()
        first = None
        for row, i in enumerate(window):
            plot = lay.addPlot(row=row, col=1)
            apply_plot_chrome(plot, dark)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            if row < len(window) - 1:
                plot.hideAxis('bottom')
            else:
                plot.setLabel('bottom', 'Time (s)')
            lay.addLabel(self._draw_strip(plot, items[i], i == cur, groups, dark),
                         row=row, col=0)

    def _draw_strip(self, plot, item, current: bool, groups: list, dark: bool) -> str:
        """Draw one neuron's strip; returns the HTML label that sits beside it."""
        key = self.nav.view.key_of(item)
        neurons = self.nav.cd.nd.neurons_for(key.nd())
        found = np.flatnonzero(np.asarray(neurons.neuron_ids) == key.ref)
        if not len(found):      # the row outlived the session it was listed for
            return ''
        index = int(found[0])
        bin_size = self.bin_spin.value()
        counts, times = neurons.binned_counts(index, bin_size)
        if self.log_btn.isChecked():
            counts = np.log10(counts + 1.0)
        peak = float(counts.max()) if len(counts) else 1.0
        plot.getViewBox().setYRange(0, (peak or 1.0) * self.y_slider.value() / 100.0, padding=0.05)
        self._draw_rates(plot, times, counts, bin_size, dark)
        self._draw_boundaries(plot, dark, groups)
        self._draw_label_highlight(plot, times, counts, bin_size, dark, groups)
        if current and self.wf_panel.isVisible():
            self.wf_panel.render(neurons, index)
        words = ''.join(
            f' <span style="color:#%02x%02x%02x">{w}</span>' % tuple(rgb)
            for w, rgb in self.nav.root.neuron_tags.neuron_words(key, int(key.ref)))
        title = self._title(key, neurons, index)
        return (f"<b>▶ {title}</b>" if current else title) + (f"<br>{words}" if words else '')

    def _draw_rates(self, plot, times, counts, bin_size: float, dark: bool) -> None:
        if not self.rate_btn.show:
            return
        color = TRACE_COLOR_DARK if dark else TRACE_COLOR
        if self.rate_btn.line or not self._bars_visible(times, bin_size):
            plot.plot(times, counts, pen=plot_pen(color))
        else:
            plot.addItem(pg.BarGraphItem(x=times, height=counts, width=bin_size,
                                         brush=pg.mkBrush(color), pen=None))

    def _bars_visible(self, times, bin_size: float) -> bool:
        """Bars only once one is at least a pixel wide; below that they draw as nothing."""
        span = float(times[-1] - times[0]) if len(times) > 1 else bin_size
        width_px = self.plot_widget.width() or 1
        return span <= 0 or bin_size / span * width_px >= 1.0

    def _draw_boundaries(self, plot, dark: bool, groups: list) -> None:
        """Epoch edges as lines, or whole epochs as blocks, one item per colour group."""
        if not self.bounds_btn.show:
            return
        # one item per group, not per epoch: a theme like ripple has 26k of them and
        # PlotItem.addItem rescans every existing item on each call
        y0, y1 = plot.vb.viewRange()[1]
        for rgb, intervals in groups:
            xs, ys = [], []
            if self.bounds_btn.line:
                for start, stop in intervals:
                    for edge in (start, stop):
                        xs += [edge, edge, np.nan]
                        ys += [y0, y1, np.nan]
                plot.addItem(pg.PlotDataItem(xs, ys, connect='finite',
                                             pen=plot_pen(rgb, Qt.PenStyle.DashLine)))
            else:
                for start, stop in intervals:   # one closed rectangle per epoch
                    xs += [start, stop, stop, start, start, np.nan]
                    ys += [y0, y0, y1, y1, y0, np.nan]
                plot.addItem(pg.PlotDataItem(xs, ys, connect='finite', pen=pg.mkPen(None),
                                             fillLevel=y0, fillBrush=pg.mkBrush(*rgb, 70)))

    def _draw_label_highlight(self, plot, times, counts, bin_size, dark: bool,
                              groups: list) -> None:
        """Shade the trace itself over bins inside the shown bounds."""
        bounds = [(s, e, '') for _rgb, iv in groups for s, e in iv]
        if not bounds or not self.rate_btn.show:
            return
        mask = epoch_bin_mask(times, bounds)
        if not mask.any():
            return
        color = '#3ecf6e' if dark else '#1a6b2e'
        if self.rate_btn.line or not self._bars_visible(times, bin_size):
            masked = np.where(mask, counts, np.nan)   # NaN gaps: no line across unselected bins
            plot.plot(times, masked, pen=plot_pen(color), connect='finite')
        else:
            plot.addItem(pg.BarGraphItem(x=times[mask], height=counts[mask], y0=0,
                                         width=bin_size, brush=pg.mkBrush(color), pen=None))

    def _bound_groups(self) -> list:
        """``[(rgb, [(start, stop)])]``: the cross-theme overlap in one colour when ≥2 themes are
        included in the filter, else this theme's checked labels, each in its slider colour."""
        b = self.nav.root.time_slider.backend
        fs = b.filter_state()
        if len(fs) >= 2:
            key = self.nav.key.nd()
            t0, t1 = self.nav.cd.nd.session_bounds(key)
            iv, _dur = self.nav.cd.nd.resolve_intervals(key, t0, t1, fs)
            return [(OVERLAP_RGB, list(iv or []))]
        theme = self.theme_combo.currentText()
        if theme in ('', 'segments'):
            return []
        every = b.all_theme_bounds.get(theme, [])
        colors = b.label_colors(lb for _s, _e, lb in every)
        allowed = set(b.theme_whitelist(theme))
        by_label: dict = {}
        for start, stop, label in every:
            if label in allowed:
                by_label.setdefault(label, []).append((start, stop))
        return [(pg.mkColor(colors[lb]).getRgb()[:3], iv) for lb, iv in sorted(by_label.items())]

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
        slider.window_changed.connect(lambda _a, _b: self.render())
        for sig in (nav.key_changed, nav.session_mode_changed,
                    nav.selection_changed, nav.custom_segs_changed):
            sig.connect(self._on_rows_changed)

    @property
    def active(self) -> bool:
        """Hidden panels skip nav's signals: nav.view then lists pairs, not neurons."""
        return self.nav.view.name == NEURON_VIEW

    def render(self) -> None:
        """Redraw the plot; the list only rebuilds when its rows actually changed."""
        if self.active:
            self.plot_panel.render()

    def rebuild(self) -> None:
        self.list_panel.refresh()
        self.plot_panel.render()

    def _on_rows_changed(self, *_args) -> None:
        """Any change to which neurons the view lists, from any of nav's signals."""
        if self.active:
            self.rebuild()

    def _on_theme_changed(self) -> None:
        self.plot_panel.reload_themes(self.nav.root.time_slider.theme_names)
        self.render()
