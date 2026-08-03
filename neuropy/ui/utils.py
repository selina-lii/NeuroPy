"""Reusable Qt widgets shared across NeuroPy UI panels.

Chip buttons, cycle buttons, flow layout, collapsible sections — all panels
import from here so identical widgets don't diverge across files.

Also: CheckboxVar / LabelVar / LineEditVar — tk var .get()/.set() shims for Qt widgets.
PairListWidget — QListWidget with tkinter-compat helpers and key forwarding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtGui import QColor, QBrush, QFont
from pyqtgraph.Qt.QtCore import Signal
from pyqtgraph.Qt.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
    QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
    QSizePolicy, QSlider,
    QToolButton, QVBoxLayout, QWidget, QFrame,
    QSplitter, QStackedWidget, QComboBox
)
from neuropy.ui.ui_common import qt_dark_mode

if TYPE_CHECKING:
    from neuropy.ui.pair_selection_panel import LeftPanel


# Platform primary shortcut modifier: Cmd on macOS, Ctrl elsewhere. Qt maps macOS Cmd →
# ControlModifier; we accept Meta too so the physical Cmd key always registers regardless of
# Qt's Ctrl/Meta swap. Single source of truth for every "Ctrl-click" style shortcut.
PRIMARY_MODIFIER = Qt.ControlModifier | Qt.MetaModifier


def has_primary_modifier(mods) -> bool:
    """True if the platform primary modifier (Cmd on macOS, Ctrl elsewhere) is held."""
    return bool(mods & PRIMARY_MODIFIER)


class CheckboxVar:
    """Wrap QCheckBox with .get()/.set() like tk.BooleanVar."""

    def __init__(self, cb: 'QCheckBox'):
        self._cb = cb

    def get(self) -> bool:
        return self._cb.isChecked()

    def set(self, v: bool):
        self._cb.setChecked(bool(v))


class LabelVar:
    """Wrap QLabel with .get()/.set() like tk.StringVar."""

    def __init__(self, label: 'QLabel'):
        self._lbl = label

    def get(self) -> str:
        return self._lbl.text()

    def set(self, v: str):
        self._lbl.setText(v)


class LineEditVar:
    """Wrap QLineEdit with .get()/.set() like tk.StringVar."""

    def __init__(self, entry: 'QLineEdit'):
        self._entry = entry

    def get(self) -> str:
        return self._entry.text()

    def set(self, v: str):
        self._entry.setText(v)

    def trace_add(self, *_):
        pass   # connect to QLineEdit.textChanged directly



class PairListWidget(QListWidget):
    """QListWidget with tkinter-compat helpers and key forwarding to LeftPanel."""

    def __init__(self, panel: 'LeftPanel', parent: QWidget = None):
        super().__init__(parent)
        self._panel = panel
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        _mf = QFont()
        _mf.setStyleHint(QFont.StyleHint.Monospace)
        _mf.setPointSize(9)
        self.setFont(_mf)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setUniformItemSizes(True)
        if qt_dark_mode():
            self.setStyleSheet(
                "QListWidget { color: #ffffff; background: #1e1e1e; }"
                "QListWidget::item:selected { background: #3366cc; color: #ffffff; }"
            )

    def curselection(self) -> tuple:
        return tuple(sorted(self.row(it) for it in self.selectedItems()))

    def size(self) -> int:
        return self.count()

    def get(self, i: int) -> str:
        it = self.item(i)
        return it.text() if it else ''

    def yview(self) -> tuple:
        sb = self.verticalScrollBar()
        frac = sb.value() / sb.maximum() if sb.maximum() else 0.0
        return (frac, frac)

    def yview_moveto(self, frac: float):
        sb = self.verticalScrollBar()
        sb.setValue(int(frac * sb.maximum()))

    def see(self, i: int):
        it = self.item(i)
        if it:
            self.scrollToItem(it)

    def selection_clear(self, _a=None, _b=None):
        self.clearSelection()

    def selection_set(self, i: int):
        it = self.item(i)
        if it:
            it.setSelected(True)

    def activate(self, i: int):
        it = self.item(i)
        if it:
            self.setCurrentItem(it)

    def nearest(self, y: int) -> int:
        it = self.itemAt(0, y)
        return self.row(it) if it else max(0, self.count() - 1)

    def selection_includes(self, i: int) -> bool:
        it = self.item(i)
        return it is not None and it.isSelected()

    def itemconfig(self, i: int, **kw):
        it = self.item(i)
        if it is None:
            return
        bg = kw.get('background') or kw.get('selectbackground')
        fg = kw.get('foreground') or kw.get('selectforeground')
        if bg:
            it.setBackground(QBrush(QColor(bg)))
        elif 'background' in kw:
            it.setBackground(QBrush())
        if fg:
            it.setForeground(QBrush(QColor(fg)))
        elif 'foreground' in kw:
            it.setForeground(QBrush())

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()   # per-event modifiers: reliable on macOS, unlike the global state
        primary = has_primary_modifier(mods)

        if key in (Qt.Key_Up, Qt.Key_Down):
            super().keyPressEvent(event)
            self._panel._on_arrow_key()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._panel._on_enter_key(self)
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace) and not primary:
            self._panel._on_delete_pair()
            return
        self._panel._on_list_key(event)
        event.accept()


def chip_button(label: str, checkable: bool = True, checked: bool = False,
                parent=None) -> 'QPushButton':
    """Flat toggleable chip-style QPushButton."""
    btn = QPushButton(label, parent)
    btn.setCheckable(checkable)
    btn.setChecked(checked)
    btn.setFlat(False)
    btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    btn.setStyleSheet(
        f"QPushButton {{ border: 1px solid #aaa; border-radius: 3px; padding: 1px 6px; "
        f"font-size: {regular_font_pt()}pt; }}"
        "QPushButton:checked { background: #4a7fd4; color: white; border-color: #3366cc; }"
        "QPushButton:hover { background: #dde; }"
    )
    return btn


def make_combo(items: list[str], width: int, current: str = None) -> 'QComboBox':
    """Fixed-width QComboBox pre-filled with items (and optional current selection)."""
    cb = QComboBox()
    cb.addItems(items)
    if current:
        cb.setCurrentText(current)
    cb.setFixedWidth(width)
    return cb


def make_button(text: str, slot, width: int = None) -> 'QPushButton':
    """QPushButton wired to slot, optionally fixed-width."""
    b = QPushButton(text)
    b.clicked.connect(slot)
    if width:
        b.setFixedWidth(width)
    return b


class ListPickerButton(QPushButton):
    """Button that opens a multi-select dialog. Text auto-summarizes selection."""
    selection_changed = Signal(list)

    def __init__(self, title: str, items: list[str] = (), plural: str = "items",
                 refresh_provider=None, parent=None):
        super().__init__(parent)
        self._title  = title
        self._plural = plural
        self._items: list[str] = list(items)
        self._selected: list[str] = list(items)
        self._refresh_provider = refresh_provider   # callable → fresh item list, or None
        self._update_label()
        self.clicked.connect(self._open_dialog)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_items(self, items: list[str], keep_selection: bool = True):
        prev = set(self._selected) if keep_selection else set()
        self._items = list(items)
        self._selected = [x for x in self._items if x in prev] or list(items)
        self._update_label()

    def set_selected(self, selected: list[str]):
        want = set(selected)
        self._selected = [x for x in self._items if x in want] or list(self._items)
        self._update_label()

    @property
    def selected(self) -> list[str]:
        return list(self._selected)

    def _open_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(self._title)
        dlg.resize(260, 320)
        lay = QVBoxLayout(dlg)
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        def _populate():
            keep = {lst.item(i).text() for i in range(lst.count())
                    if lst.item(i).isSelected()} or set(self._selected)
            lst.clear()
            for item in self._items:
                it = QListWidgetItem(item)
                lst.addItem(it)
                if item in keep:
                    it.setSelected(True)

        _populate()
        lay.addWidget(lst)
        btns = QHBoxLayout()
        sel_all = QPushButton("Select all")
        sel_none = QPushButton("Select none")
        sel_all.clicked.connect(lst.selectAll)
        sel_none.clicked.connect(lst.clearSelection)
        btns.addWidget(sel_all); btns.addWidget(sel_none)
        if self._refresh_provider is not None:
            def _refresh():
                self._items = list(self._refresh_provider())
                _populate()
            refresh_btn = QPushButton("Refresh")
            refresh_btn.clicked.connect(_refresh)
            btns.addWidget(refresh_btn)
        btns.addStretch()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        apply_btn.setDefault(True)   # Enter → apply
        apply_btn.setAutoDefault(True)
        btns.addWidget(apply_btn); btns.addWidget(cancel_btn)
        lay.addLayout(btns)
        apply_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._selected = [lst.item(i).text()
                              for i in range(lst.count()) if lst.item(i).isSelected()]
            if not self._selected:
                self._selected = list(self._items)
            self._update_label()
            self.selection_changed.emit(self._selected)

    def _update_label(self):
        n, total = len(self._selected), len(self._items)
        if n == 0 or n == total:
            self.setText(f"{self._title}: All")
        elif n == 1:
            self.setText(self._selected[0])
        else:
            self.setText(f"{self._title}: {n} {self._plural}")


class ResultsDialog(QDialog):
    """Small read-only run-report dialog (Windows-style small monospace text).

    Reused for custom-CCG batch results, jitter save summaries, and any other
    "here is what ran / where it saved / what was skipped" output.
    """

    def __init__(self, title: str, body: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(body)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        mf = QFont()
        mf.setStyleHint(QFont.StyleHint.Monospace)
        mf.setPointSize(8)
        text.setFont(mf)
        lay.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)

    @classmethod
    def show_report(cls, title: str, body: str, parent=None) -> None:
        cls(title, body, parent).exec()


class SliderWithInput(QWidget):
    """Horizontal slider paired with a numeric QLineEdit.

    valueChanged emits the scaled float on slider release or input commit.
    """

    value_changed = Signal(float)

    def __init__(self, lo: int, hi: int, init: int, scale: float = 0.01,
                 fmt: str = "{:.2f}", tracking: bool = False, min_value: float = 0.01,
                 parent=None):
        super().__init__(parent)
        self._scale = scale
        self._fmt   = fmt
        self._min   = min_value
        self._value = init * scale   # typed input may exceed the slider's range
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(init)
        self._slider.setPageStep(0)
        self._slider.setTracking(tracking)
        self._slider.wheelEvent = lambda ev: ev.ignore()
        self._slider.setFixedWidth(60)
        self._input = QLineEdit(fmt.format(init * scale))
        self._input.setFixedWidth(44)
        self._slider.valueChanged.connect(self._on_slider)
        self._input.editingFinished.connect(self._on_input)
        lay.addWidget(self._slider)
        lay.addWidget(self._input)

    def _on_slider(self, v: int):
        self._value = v * self._scale
        self._input.setText(self._fmt.format(self._value))
        self.value_changed.emit(self._value)

    def _on_input(self):
        try:
            v = float(self._input.text())
        except ValueError:
            return
        self.set_value(v)
        self.value_changed.emit(self._value)

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, v: float):
        """Accepts values past the slider's upper range; the slider parks at its end."""
        v = max(self._min, v)
        self._value = v
        raw = int(round(v / self._scale))
        raw = max(self._slider.minimum(), min(self._slider.maximum(), raw))
        self._slider.blockSignals(True)
        self._slider.setValue(raw)
        self._slider.blockSignals(False)
        self._input.setText(self._fmt.format(v))


class ExclusiveButtonSet:
    """Checkable chip buttons with at-most-one checked (or none)."""

    def __init__(self, *, on_change=None, parent=None):
        self._parent = parent
        self._on_change = on_change
        self._buttons: dict[str, QPushButton] = {}
        self._syncing = False

    def add(self, key: str, label: str, *, checked: bool = False
            ) -> tuple['QPushButton', CheckboxVar]:
        btn = chip_button(label, checkable=True, checked=checked,
                          parent=self._parent)
        btn.toggled.connect(lambda on, k=key: self._on_toggled(k, on))
        self._buttons[key] = btn
        return btn, CheckboxVar(btn)

    def button(self, key: str) -> 'QPushButton':
        return self._buttons[key]

    def select(self, key: str) -> None:
        self._buttons[key].setChecked(True)

    def is_checked(self, key: str) -> bool:
        return self._buttons[key].isChecked()

    def _on_toggled(self, key: str, checked: bool) -> None:
        if self._syncing:
            return
        if checked:
            self._syncing = True
            for k, btn in self._buttons.items():
                if k != key:
                    btn.setChecked(False)
            self._syncing = False
        if self._on_change:
            self._on_change()



def collapsible(title: str, parent=None) -> 'tuple[QGroupBox, QVBoxLayout]':
    """Checkable QGroupBox that shows/hides its children on toggle."""
    box = QGroupBox(title, parent)
    box.setCheckable(True)
    box.setChecked(True)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(2)

    def _toggle(checked):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(checked)

    box.toggled.connect(_toggle)
    return box, layout



class CycleButton(QPushButton):
    """Three-state cycle: solid(■) → line(□) → hidden(x) or reverse start.

    Properties:
        show  — whether the item should be rendered at all
        line  — whether to use line style instead of solid bars
    """

    _STATES_FWD = [('■', True, False),  # solid
                   ('□', True, True),   # line
                   ('x', False, False)] # hidden
    _STATES_REV = [('x', False, False), # hidden  (start_hidden=True)
                   ('□', True, True),   # line
                   ('■', True, False)]  # solid

    def __init__(self, name: str, start_hidden: bool = False, parent=None):
        super().__init__(parent)
        self._name   = name
        self._states = self._STATES_REV if start_hidden else self._STATES_FWD
        self._idx    = 0
        self._apply()
        self.clicked.connect(self._cycle)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "QPushButton { border: 1px solid #aaa; border-radius: 3px; padding: 1px 6px; }"
            "QPushButton:hover { background: #dde; }"
        )

    def _apply(self):
        sym, _, _ = self._states[self._idx]
        self.setText(f"{sym} {self._name}")

    def _cycle(self):
        self._idx = (self._idx + 1) % len(self._states)
        self._apply()

    @property
    def show(self) -> bool:
        return self._states[self._idx][1]

    @property
    def line(self) -> bool:
        return self._states[self._idx][2]



class FlowLayout(QtWidgets.QLayout):
    """Left-to-right wrapping layout — chip buttons wrap to next row when full."""

    def __init__(self, parent=None, spacing: int = 4):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def addWidget(self, w):
        self.addItem(QtWidgets.QWidgetItem(w))
        w.setParent(self.parentWidget())

    def clear_widgets(self):
        """Remove all widgets without destroying the layout container."""
        while self._items:
            item = self._items.pop()
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QtCore.QSize()
        for item in self._items:
            s = s.expandedTo(item.minimumSize())
        return s + QtCore.QSize(2, 2)

    def _do_layout(self, rect, test: bool) -> int:
        x, y = rect.x(), rect.y()
        row_h = 0
        for item in self._items:
            w = item.widget()
            if w is None or not w.isVisible():
                continue
            hint = w.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > rect.right() and row_h > 0:
                x = rect.x()
                y += row_h + self._spacing
                row_h = 0
                next_x = x + hint.width() + self._spacing
            if not test:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y()




class SideNavPanel(QWidget):
    """Sidebar-nav shell — shared by the Settings dialog and Manage Groups.

    A draggable sash separates the nav list (sidebar) from the page stack (main).
    The content pane stretches to fill the window; the nav pane is user-resizable
    (no hard width cap, so neither side is squeezed).
    """

    def __init__(self, parent=None, *, min_width: int = 108, nav_width: int = 160):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.nav_list = QListWidget()
        self.nav_list.setMinimumWidth(min_width)
        self.stack = QStackedWidget()
        self.splitter.addWidget(self.nav_list)
        self.splitter.addWidget(self.stack)
        self.splitter.setStretchFactor(0, 0)   # nav keeps its width on resize
        self.splitter.setStretchFactor(1, 1)   # content absorbs the extra space
        self.splitter.setSizes([nav_width, max(nav_width, 360)])
        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        lay.addWidget(self.splitter)

    @property
    def currentChanged(self):
        """Signal(int) emitted when the selected page changes (compat alias)."""
        return self.nav_list.currentRowChanged

    def add_page(self, label: str, widget: QWidget) -> int:
        self.nav_list.addItem(label)
        idx = self.stack.addWidget(widget)
        if self.nav_list.currentRow() < 0:
            self.nav_list.setCurrentRow(0)
        return idx

    def setCurrentIndex(self, index: int) -> None:
        self.nav_list.setCurrentRow(index)

    def count(self) -> int:
        return self.stack.count()




class ArrowChipBar(QWidget):
    """[arrowed chip bar] ◀ [chips] ▶ — horizontal scroll row for chip widgets."""

    def __init__(self, parent=None, height: int = 22, on_left=None, on_right=None):
        super().__init__(parent)
        self._on_left = on_left     # if set, ◀ calls this instead of scrolling
        self._on_right = on_right   # if set, ▶ calls this instead of scrolling
        self.setMaximumHeight(height + 4)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        self._left = QToolButton()
        self._left.setText('◀')
        self._left.setFixedSize(18, height)
        root.addWidget(self._left)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setFixedHeight(height)
        self._host = QWidget()
        self._lay = QHBoxLayout(self._host)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(3)
        self._lay.addStretch()
        self._scroll_area.setWidget(self._host)
        root.addWidget(self._scroll_area, stretch=1)
        self._right = QToolButton()
        self._right.setText('▶')
        root.addWidget(self._right)
        self._left.clicked.connect(
            lambda: self._on_left() if self._on_left else self._scroll(-80))
        self._right.clicked.connect(
            lambda: self._on_right() if self._on_right else self._scroll(80))

    def _scroll(self, delta: int):
        bar = self._scroll_area.horizontalScrollBar()
        bar.setValue(max(0, min(bar.maximum(), bar.value() + delta)))

    def _sync_host_width(self):
        w = 0
        for i in range(self._lay.count() - 1):
            item = self._lay.itemAt(i)
            if item and item.widget():
                w += item.widget().sizeHint().width() + self._lay.spacing()
        self._host.setMinimumWidth(
            max(w + 8, self._scroll_area.viewport().width()))

    def clear(self):
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)   # hide immediately; deleteLater alone leaves stale paints
                w.deleteLater()

    def add_widget(self, w: QWidget):
        self._lay.insertWidget(self._lay.count() - 1, w)
        self._sync_host_width()

    def set_widgets(self, widgets: list):
        self.clear()
        for w in widgets:
            self.add_widget(w)
        self._sync_host_width()


def regular_font_pt() -> int:
    """Body-text point size — the live app font (Settings > min font size)."""
    app = QApplication.instance()
    return app.font().pointSize() if app is not None else 12


def small_font_pt() -> int:
    """Small/caption-text point size, scaled off the live app font."""
    return regular_font_pt() - 2


def _UI_FS() -> str:
    return f'font-size: {small_font_pt()}pt;'

class CollapsibleSection(QWidget):
    """Base class for a titled collapsible panel. Subclasses add widgets to body_layout."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QWidget { border: 1px solid #ccc; }')
        sec_lay = QVBoxLayout(self)
        sec_lay.setContentsMargins(2, 2, 2, 2)
        sec_lay.setSpacing(1)

        hdr = QWidget()
        hdr.setStyleSheet('QWidget { border: none; }')
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(2)
        self._arrow_btn = QPushButton(('▾ ' if expanded else '▸ ') + title)
        self._arrow_btn.setFlat(True)
        self._arrow_btn.setStyleSheet(
            f'font-weight: bold; font-size: {regular_font_pt()}pt; text-align: left; border: none;')
        hdr_lay.addWidget(self._arrow_btn)
        hdr_lay.addStretch()
        sec_lay.addWidget(hdr)

        self._body = QWidget()
        self._body.setStyleSheet('QWidget { border: none; }')
        self._body.setVisible(expanded)
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(4, 2, 4, 2)
        self.body_layout.setSpacing(2)
        sec_lay.addWidget(self._body)

        self._title = title
        self._arrow_btn.clicked.connect(self._toggle)

    def _toggle(self):
        visible = self._body.isVisible()
        self._body.setVisible(not visible)
        self._arrow_btn.setText(('▾ ' if not visible else '▸ ') + self._title)

    @staticmethod
    def make_spin(options, default: str, width: int = 70):
        cb = QComboBox()
        for v in options:
            cb.addItem(str(v), v)
        cb.setCurrentText(default)
        cb.setEditable(True)
        cb.setFixedWidth(width)
        return cb


def make_collapsible_section(parent_layout, title: str,
                             *, expanded: bool = True) -> tuple:
    """Deprecated — use CollapsibleSection base class instead."""
    sec = CollapsibleSection(title, expanded)
    parent_layout.addWidget(sec)
    return sec, sec._arrow_btn, sec._body, sec.body_layout

class GroupHotkeysBar(QWidget):
    """Horizontal group hotkey chip bar below the index bar."""

    _SLOT_ORDER = ([str(i) for i in range(1, 10)] + ['0']
                   + list('abcdefghijklmnopqrstuvwxyz'))

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self._ui = ui
        self.frame = self  # compat alias
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 0, 4, 0)
        root.setSpacing(6)
        title = QLabel('Hotkeys')
        title.setStyleSheet(f'font-size: {small_font_pt()}pt; font-weight: bold;')
        root.addWidget(title)
        del_lbl = QLabel('Del/⌫: deleted')
        del_lbl.setStyleSheet(f'color: #888; font-size: {small_font_pt()}pt;')
        root.addWidget(del_lbl)
        self._bar = ArrowChipBar(self, height=20)
        root.addWidget(self._bar, stretch=1)
        self.refresh()

    def refresh(self):
        import random
        win = self._ui
        nav = win.nav
        gr = nav.groups
        hk_map = {g.hotkey: name
                  for name in gr.defined_groups
                  for g in [gr.get_group_metadata(name)] if g.hotkey}
        dark = qt_dark_mode()
        bg, fg, border = (('#3a3a3a', '#e0e0e0', '#666') if dark
                          else ('#fafafa', '#202020', '#aaa'))
        chips: list = []
        for key_str in self._SLOT_ORDER:
            if key_str not in hk_map:
                continue
            gname = hk_map[key_str]
            chip = QLabel(f'{key_str}: {gname}')
            chip.setStyleSheet(
                f'font-size: {small_font_pt()}pt; padding: 2px 4px; border: 1px solid {border}; '
                f'background: {bg}; color: {fg};')
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            def _select(_, g=gname):
                pairs = gr.pairs_in_group(g, nav.current_session_str)
                if pairs:
                    nav.set_current_pair(nav.get_pair_index(sorted(pairs)[0]))
                    win.mainview.request_render()
            chip.mousePressEvent = _select

            def _dbl(_, g=gname):
                pairs = gr.pairs_in_group(g, nav.current_session_str)
                if not pairs:
                    return
                nav.set_current_pair(nav.get_pair_index(random.choice(sorted(pairs))))
                win.mainview.request_render()
                win.neuron_network.draw()

            chip.mouseDoubleClickEvent = _dbl
            chips.append(chip)
        self._bar.set_widgets(chips)
