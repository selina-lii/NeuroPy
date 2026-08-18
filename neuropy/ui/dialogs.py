"""Qt dialog replacements for dialogs.py."""
from __future__ import annotations

import datetime
import glob
import os
import pkgutil
import shutil
from importlib import import_module
from typing import TYPE_CHECKING, Callable

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPlainTextEdit, QCheckBox, QListWidget, QListWidgetItem,
    QPushButton, QTabWidget, QWidget, QMessageBox, QMenu, QApplication,
    QScrollArea, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
    QAbstractItemView, QFrame, QComboBox, QInputDialog, QFileDialog, QColorDialog,
)
from pyqtgraph.Qt.QtGui import QFont, QColor
from neuropy.ui.ui_common import _SPECIAL_PREFIX
from neuropy.analyses.ms_connectivity import CCGConfig, ProjectConfig, build_project
from neuropy.analyses.neurons_dataset import Key, NeuronsDatasetConfig
from neuropy.core.nwb_session import NWBDataset
from neuropy.io import datasets
from neuropy.io.fieldmap import FieldMap
from neuropy.io.nwbio import NWB_DEFAULT, UNITS_SCHEMA, NWBFile

from neuropy.ui.utils import (ConfigOptionsWidget, FlowLayout, MetricInput, SideNavPanel,
                              ValueMapEditor, chip_button, make_button, small_font_pt)

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState
    from neuropy.ui.pair_selection_panel import PairSelectionPanel, Groups, SelectionData


class PairTagsDialog(QDialog):
    """Tags + notes for the current pair."""

    def __init__(self, ref: int, tgt: int, existing: dict, parent=None):
        super().__init__(parent)
        self._ref, self._tgt = ref, tgt
        self._existing = existing
        self._build()

    def _build(self):
        self.setWindowTitle(f"Pair Tags — [{self._ref}, {self._tgt}]")
        self.resize(420, 340)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Tags (comma-separated):"))
        self._tags_edit = QLineEdit(', '.join(self._existing.get('tags', [])))
        lay.addWidget(self._tags_edit)
        lay.addWidget(QLabel("Notes:"))
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlainText(self._existing.get('notes', ''))
        lay.addWidget(self._notes_edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result(self) -> dict | None:
        tags  = [t.strip() for t in self._tags_edit.text().split(',') if t.strip()]
        notes = self._notes_edit.toPlainText()
        if not tags and not notes:
            return None
        entry = {'tags': tags, 'notes': notes}
        if self._existing.get('groups'):
            entry['groups'] = self._existing['groups']
        return entry


class CreateGroupDialog(QDialog):
    """Create a new named group."""

    def __init__(self, sel_data: 'SelectionData', group_mgr: 'PairSelectionPanel',
                 parent=None):
        super().__init__(parent)
        self.sel_data = sel_data
        self._group_mgr = group_mgr
        self._groups = group_mgr.ui.groups
        self.setWindowTitle("Create Group")
        self.setFixedSize(320, 130)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Group name:"))
        self._name_edit = QLineEdit()
        self._name_edit.returnPressed.connect(self._ok)
        row.addWidget(self._name_edit)
        lay.addLayout(row)
        self._special_cb = QCheckBox("Create as special group")
        lay.addWidget(self._special_cb)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._name_edit.setFocus()

    def _ok(self):
        name = self._name_edit.text().strip()
        if not name:
            return
        full = (_SPECIAL_PREFIX + name) if self._special_cb.isChecked() else name
        if full in self._groups:
            kind = 'special group' if self._special_cb.isChecked() else 'group'
            QMessageBox.information(self, "Create group",
                                    f"{kind.capitalize()} '{name}' already exists.")
            return
        self._groups.get_group_metadata(full)
        self._groups.changed.emit()
        self.accept()

    @classmethod
    def show(cls, sel_data: 'SelectionData', group_mgr: 'PairSelectionPanel',
             parent=None):
        dlg = cls(sel_data, group_mgr, parent)
        dlg.exec()


class ManageGroupsDialog(QDialog):
    """Rename, set hotkey, edit notes, and delete groups.

    Top-tab layout: regular groups alphabetically, "Special" last.
    All edits autosaved (notes to disk) on tab switch and close.
    """

    def __init__(self, sel_data: 'SelectionData', group_mgr: 'PairSelectionPanel',
                 pairs_by_conn_type_fn=None, parent=None):
        super().__init__(parent)
        self.sel_data = sel_data
        self._group_mgr = group_mgr
        self._groups = group_mgr.ui.groups
        self._pairs_by_ct = pairs_by_conn_type_fn
        self._notes_widgets: dict[str, QPlainTextEdit] = {}
        self.setWindowTitle("Manage Groups")
        self.resize(600, 520)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        gr = self._groups
        regular = gr.groups
        special = gr.special_groups()

        nav = SideNavPanel(min_width=100, nav_width=160)
        for gname in regular:
            nav.add_page(gname, self._make_group_tab(gname, is_special=False))

        if special:
            sp_tabs = QTabWidget()
            sp_tabs.setTabPosition(QTabWidget.TabPosition.North)
            for gname in special:
                display = gr.get_group_metadata(gname).display_name
                sp_tabs.addTab(self._make_group_tab(gname, is_special=True), display)
            sp_tabs.currentChanged.connect(self._autosave_notes)
            nav.add_page("Special", sp_tabs)

        nav.currentChanged.connect(self._autosave_notes)
        lay.addWidget(nav, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._autosave_and_close)
        lay.addWidget(close_btn)

    def _make_group_tab(self, gname: str, is_special: bool) -> QWidget:
        sd = self.sel_data
        gr = self._groups
        display = gr.get_group_metadata(gname).display_name

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Group name:"))
        name_edit = QLineEdit(display)
        name_row.addWidget(name_edit, stretch=1)
        rename_btn = QPushButton("Rename")

        def _rename(_checked=False, old=gname, sp=is_special, ne=name_edit):
            new = ne.text().strip()
            if sp:
                new = _SPECIAL_PREFIX + new
            try:
                self._autosave_notes()
                gr.rename_group(old, new)
                gr.changed.emit()
                gr.ui.refresh_lists()
                self._rebuild()
            except ValueError as e:
                QMessageBox.warning(self, "Rename", str(e))

        rename_btn.clicked.connect(_rename)
        name_row.addWidget(rename_btn)
        lay.addLayout(name_row)

        if not is_special:
            hk_row = QHBoxLayout()
            hk_row.addWidget(QLabel("Hotkey (0-9/a-z):"))
            hk_edit = QLineEdit(gr.get_group_metadata(gname).hotkey)
            hk_edit.setMaximumWidth(60)
            hk_row.addWidget(hk_edit)
            set_hk = QPushButton("Set")

            def _set_hk(_checked=False, g=gname, he=hk_edit):
                try:
                    gr.set_hotkey_ui(g, he.text())
                except Exception as e:
                    QMessageBox.warning(self, "Hotkey", str(e))

            set_hk.clicked.connect(_set_hk)
            hk_row.addWidget(set_hk)
            hk_row.addWidget(QLabel("Colour:"))
            meta = gr.get_group_metadata(gname)
            swatch = QPushButton()
            swatch.setFixedWidth(40)
            swatch.setStyleSheet(f"background: {meta.display_color};")

            def _pick(_checked=False, g=gname, sw=swatch):
                m = gr.get_group_metadata(g)
                c = QColorDialog.getColor(QColor(m.display_color), self, "Tag colour")
                if c.isValid():
                    m.ui_color = c.name()
                    sw.setStyleSheet(f"background: {c.name()};")
                    gr.changed.emit()
                    gr.ui.refresh_lists()

            swatch.clicked.connect(_pick)
            hk_row.addWidget(swatch)
            hk_row.addStretch()
            lay.addLayout(hk_row)

        splitter = QSplitter(Qt.Orientation.Vertical)

        notes_host = QWidget()
        notes_lay = QVBoxLayout(notes_host)
        notes_lay.setContentsMargins(0, 0, 0, 0)
        notes_lay.setSpacing(2)
        notes_lay.addWidget(QLabel("Notes:"))
        notes_edit = QPlainTextEdit()
        notes_edit.setPlainText(gr.get_group_metadata(gname).notes)
        notes_lay.addWidget(notes_edit)
        self._notes_widgets[gname] = notes_edit
        splitter.addWidget(notes_host)

        pairs_host = QWidget()
        pairs_lay = QVBoxLayout(pairs_host)
        pairs_lay.setContentsMargins(0, 0, 0, 0)
        pairs_lay.setSpacing(2)
        pairs_lay.addWidget(QLabel("Pairs in group:"))

        sess_tabs = QTabWidget()
        sess_tabs.setTabPosition(QTabWidget.TabPosition.North)
        sessions = sorted(gr.sessions_for_group(gname))
        if sessions:
            for sess in sessions:
                pairs = sorted(gr.pairs_in_group(gname, sess))
                pair_list = QListWidget()
                for r, t in pairs:
                    pair_list.addItem(f"[{r} {t}]")
                sess_tabs.addTab(pair_list, sess)
        else:
            empty = QLabel("(no pairs in this group)")
            empty.setStyleSheet("color: #888; padding: 8px;")
            sess_tabs.addTab(empty, "—")
        pairs_lay.addWidget(sess_tabs)
        splitter.addWidget(pairs_host)

        splitter.setSizes([160, 200])
        lay.addWidget(splitter, stretch=1)

        btn_row = QHBoxLayout()
        if is_special:
            conv_btn = QPushButton("Convert to group")
            def _conv(_checked=False, g=gname, d=display):
                try:
                    self._autosave_notes()
                    gr.rename_group(g, d)
                    gr.changed.emit()
                    gr.ui.refresh_lists()
                    self._rebuild()
                except ValueError as e:
                    QMessageBox.warning(self, "Convert", str(e))
        else:
            conv_btn = QPushButton("Convert to special group")
            def _conv(_checked=False, g=gname, d=display):
                try:
                    self._autosave_notes()
                    gr.rename_group(g, _SPECIAL_PREFIX + d)
                    gr.changed.emit()
                    gr.ui.refresh_lists()
                    self._rebuild()
                except ValueError as e:
                    QMessageBox.warning(self, "Convert", str(e))
        conv_btn.clicked.connect(_conv)
        btn_row.addWidget(conv_btn)

        del_btn = QPushButton(f"Delete group '{display}'")
        def _del(_checked=False, g=gname):
            if QMessageBox.question(self, "Delete group",
                                    f"Delete group '{g}'?") != QMessageBox.StandardButton.Yes:
                return
            self._autosave_notes()
            gr.delete_group(g)
            gr.changed.emit()
            gr.ui.refresh_lists()
            self._rebuild()
        del_btn.clicked.connect(_del)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        return w

    def _autosave_notes(self, *_):
        gr = self._groups
        for gname, widget in self._notes_widgets.items():
            gr.get_group_metadata(gname).notes = widget.toPlainText()
        gr.save()

    def _autosave_and_close(self):
        self._autosave_notes()
        self.accept()

    def _rebuild(self):
        self._notes_widgets = {}
        sd, gm, pct, par = (self.sel_data, self._group_mgr,
                             self._pairs_by_ct, self.parentWidget())
        self.accept()
        ManageGroupsDialog.show(sd, gm, pct, par)

    def closeEvent(self, event):
        self._autosave_notes()
        event.accept()

    @classmethod
    def show(cls, sel_data: 'SelectionData', group_mgr: 'PairSelectionPanel',
             pairs_by_conn_type_fn=None, parent=None):
        gr = group_mgr.ui.groups
        if not gr.registry and not gr:
            QMessageBox.information(parent, "Manage groups",
                                    "No groups yet. Create one first.")
            return
        dlg = cls(sel_data, group_mgr, pairs_by_conn_type_fn, parent)
        dlg.exec()


class VersionSaveDialog(QDialog):
    """Name entry + Save / optional Save-as-default."""

    @classmethod
    def show(cls, parent, title: str, default_name: str,
             on_save: Callable, on_save_default: Callable | None = None):
        dlg = QDialog(parent)
        dlg.setWindowTitle(title)
        dlg.setFixedSize(360, 110)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Version name:"))
        name_edit = QLineEdit(default_name)
        name_edit.selectAll()
        lay.addWidget(name_edit)
        btn_row = QHBoxLayout()

        def _named():
            name = name_edit.text().strip() or default_name
            dlg.accept()
            on_save(name)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(_named)
        name_edit.returnPressed.connect(_named)
        btn_row.addWidget(save_btn)
        if on_save_default is not None:
            def_btn = QPushButton("Save as Default")
            def _save_def():
                dlg.accept()
                on_save_default()
            def_btn.clicked.connect(_save_def)
            btn_row.addWidget(def_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)
        name_edit.setFocus()
        dlg.exec()


class VersionLoadDialog(QDialog):
    """Pick one version from a list of (name, path, saved_at, is_valid, is_history)."""

    @classmethod
    def show(cls, parent, title: str, versions: list,
             on_load: Callable, empty_msg: str = "No saved versions found."):
        if not versions:
            QMessageBox.information(parent, title, empty_msg)
            return
        versions = list(versions)
        dlg = QDialog(parent)
        dlg.setWindowTitle(title)
        dlg.resize(640, 360)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select a version to load (double-click to rename):"))
        lst = QListWidget()

        def _fill():
            lst.clear()
            for name, path, saved_at, is_valid, is_history in versions:
                pfx = '' if is_valid else '⚠  '
                item = QListWidgetItem(f"{pfx}{name:30s}  {saved_at[:19]}")
                if not is_valid:
                    item.setForeground(Qt.GlobalColor.red)
                elif is_history:
                    item.setForeground(Qt.GlobalColor.gray)
                lst.addItem(item)
        _fill()
        lay.addWidget(lst)

        def _selected():
            row = lst.currentRow()
            return (row, versions[row]) if 0 <= row < len(versions) else (-1, None)

        def _do_load():
            row, ver = _selected()
            if ver is None:
                return
            name, path, saved_at, is_valid, is_history = ver
            if not is_valid:
                QMessageBox.warning(dlg, title, f"'{name}' appears corrupted.")
                return
            dlg.accept()
            on_load(path)

        def _do_rename():
            row, ver = _selected()
            if ver is None:
                return
            name, path, saved_at, is_valid, is_history = ver
            new, ok = QInputDialog.getText(dlg, "Rename", "New name:", text=name)
            new = (new or '').strip()
            if not ok or not new or new == name:
                return
            dst = os.path.join(os.path.dirname(path), new + os.path.splitext(path)[1])
            if os.path.exists(dst):
                QMessageBox.warning(dlg, "Rename", f"'{new}' already exists.")
                return
            try:
                os.rename(path, dst)
            except OSError as exc:
                QMessageBox.warning(dlg, "Rename", f"Rename failed:\n{exc}")
                return
            versions[row] = (new, dst, saved_at, is_valid, is_history)
            _fill()
            lst.setCurrentRow(row)

        def _do_delete():
            row, ver = _selected()
            if ver is None:
                return
            name, path, *_ = ver
            if QMessageBox.question(dlg, "Delete", f"Delete '{name}'?") != QMessageBox.StandardButton.Yes:
                return
            try:
                os.remove(path)
            except OSError as exc:
                QMessageBox.warning(dlg, "Delete", f"Delete failed:\n{exc}")
                return
            versions.pop(row)
            _fill()
            if not versions:
                dlg.reject()

        lst.doubleClicked.connect(_do_rename)
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(_do_load)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(_do_rename)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(_do_delete)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(rename_btn)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)
        dlg.exec()


class QuickSaveDialog(QDialog):
    """Ctrl+S save dialog — update latest and optionally name a snapshot."""

    def __init__(self, group_mgr: 'PairSelectionPanel', parent=None):
        super().__init__(parent)
        self._group_mgr = group_mgr
        self.setWindowTitle("Save Selection")
        self.setFixedSize(360, 130)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Snapshot name (optional):"))
        self._name_edit = QLineEdit(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self._name_edit.selectAll()
        self._name_edit.returnPressed.connect(self._save)
        lay.addWidget(self._name_edit)
        btn_row = QHBoxLayout()
        latest_btn = QPushButton("Save to Latest")
        latest_btn.clicked.connect(self._save_latest)
        save_btn = QPushButton("Save Snapshot")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(latest_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)
        self._name_edit.setFocus()

    def _save_latest(self):
        self.accept()
        self._group_mgr._do_save('')

    def _save(self):
        name = self._name_edit.text().strip()
        self.accept()
        self._group_mgr._do_save(name)

    @classmethod
    def show(cls, group_mgr: 'PairSelectionPanel', parent=None):
        cls(group_mgr, parent).exec()


class LoadSelectionDialog(QDialog):
    """List saved selection versions for user to pick."""

    def __init__(self, group_mgr: 'PairSelectionPanel', sel_save_dir: str,
                 parent=None):
        super().__init__(parent)
        self._group_mgr = group_mgr
        self._sel_save_dir = sel_save_dir
        self._versions = group_mgr._list_selection_versions()
        self.setWindowTitle("Load Selection")
        self.resize(640, 360)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Select a version to load:"))
        self._list = QListWidget()
        for name, path, saved_at, is_valid, is_history in self._versions:
            pfx = '' if is_valid else '⚠  '
            item = QListWidgetItem(f"{pfx}{name:30s}  {saved_at[:19]}")
            if not is_valid:
                item.setForeground(Qt.GlobalColor.red)
            elif is_history:
                item.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(item)
        self._list.doubleClicked.connect(self._do_load)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._ctx_menu)
        lay.addWidget(self._list)

        hint = QLabel("gray = backup/autosave  ⚠ = corrupted")
        hint.setStyleSheet(f"color: #888; font-size: {small_font_pt()}pt;")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._do_load)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

    def _do_load(self):
        row = self._list.currentRow()
        if row < 0:
            return
        name, path, saved_at, is_valid, is_history = self._versions[row]
        if not is_valid:
            r = QMessageBox.question(self, "Corrupted file",
                                     f"'{name}' appears to be corrupted.\nDelete it and continue?")
            if r != QMessageBox.StandardButton.Yes:
                return
            try:
                os.remove(path)
            except OSError as ex:
                QMessageBox.critical(self, "Delete failed", str(ex))
            self.reject()
            return
        try:
            self._group_mgr._load_selection_from_file(path)
            self.accept()
        except Exception as ex:
            QMessageBox.critical(self, "Load selection", f"Failed to load:\n{ex}")

    def _ctx_menu(self, pos):
        row = self._list.currentRow()
        if row < 0:
            return
        name, path, *_ = self._versions[row]
        menu = QMenu(self)
        del_act = menu.addAction("Delete")
        act = menu.exec(self._list.viewport().mapToGlobal(pos))
        if act == del_act:
            r = QMessageBox.question(self, "Delete selection",
                                     f"Move '{name}' to deleted folder?")
            if r != QMessageBox.StandardButton.Yes:
                return
            deleted_dir = os.path.join(self._sel_save_dir, 'deleted')
            os.makedirs(deleted_dir, exist_ok=True)
            try:
                shutil.move(path, os.path.join(deleted_dir, os.path.basename(path)))
            except OSError as ex:
                QMessageBox.critical(self, "Delete failed", str(ex))
                return
            self.reject()

    @classmethod
    def show(cls, group_mgr: 'PairSelectionPanel', sel_save_dir: str, parent=None):
        versions = group_mgr._list_selection_versions()
        if not versions:
            QMessageBox.information(parent, "Load selection",
                                    "No saved selections found for this key.")
            return
        dlg = cls(group_mgr, sel_save_dir, parent)
        dlg.exec()


class MissingPairsDialog(QDialog):
    """Shown when a loaded selection contains pairs absent from available set.

    Returns 'partial', 'admit_all', or 'cancel'.
    """

    def __init__(self, missing: set, parent=None):
        super().__init__(parent)
        self._result = 'cancel'
        self.setWindowTitle("Missing Pairs")
        self.resize(420, 300)
        self._build(missing)

    def _build(self, missing: set):
        lay = QVBoxLayout(self)
        n = len(missing)
        lay.addWidget(QLabel(f"<b>{n} selected pair(s) are no longer in available pairs.</b>"))
        lay.addWidget(QLabel("These pairs may have lost significance after CCG/epoch changes."))
        lb = QListWidget()
        _mf = QFont(); _mf.setStyleHint(QFont.StyleHint.Monospace); _mf.setPointSize(9)
        lb.setFont(_mf)
        for ref, tgt in sorted(missing):
            lb.addItem(f"  ({ref:3d}, {tgt:3d})")
        lay.addWidget(lb)
        btn_row = QHBoxLayout()
        partial = QPushButton("Keep only available")
        admit   = QPushButton("Admit all missing")
        cancel  = QPushButton("Cancel")
        partial.clicked.connect(lambda: (setattr(self, '_result', 'partial'),  self.accept()))
        admit.clicked.connect(  lambda: (setattr(self, '_result', 'admit_all'), self.accept()))
        cancel.clicked.connect( self.reject)
        for b in (partial, admit, cancel):
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

    @classmethod
    def show(cls, ui, missing: set) -> str:  # ui unused; kept for compat signature
        dlg = cls(missing)
        dlg.exec()
        return dlg._result


class CustomCCGManageDialog(QDialog):
    """List saved custom CCG entries (by session or name). Load or delete."""

    def __init__(self, custom_mgr, time_slider=None, *, select_mode: bool = False,
                 parent=None):
        super().__init__(parent)
        self._mgr = custom_mgr
        self._ts = time_slider
        self._select_mode = select_mode
        self.setWindowTitle("Load custom CCG" if select_mode else "Custom CCG files")
        self.resize(560, 420)
        self._build()
        self._refresh_list()

    def _scan_entries(self) -> list[dict]:
        """Appended-window segments on disk."""
        cd = self._mgr._ui.nav.cd
        combos = sorted({(str(k.session), str(k.segment)) for k in cd.saved_customs()})
        return [{'name': name, 'session': sess, 'display': f"{sess} · {name}"}
                for sess, name in combos]

    def _build(self):
        lay = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.West)
        lay.addWidget(self._tabs)
        self._by_sess = QListWidget()
        self._by_name = QListWidget()
        self._tabs.addTab(self._by_sess, "By session")
        self._tabs.addTab(self._by_name, "By name")
        for lw in (self._by_sess, self._by_name):
            lw.setSelectionMode(QAbstractItemView.ExtendedSelection)

        btn_row = QHBoxLayout()
        if self._select_mode:
            load_btn = QPushButton("Load selected")
            load_btn.clicked.connect(self._load_selected)
            btn_row.addWidget(load_btn)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_selected)
        ref_btn = QPushButton("Refresh list")
        ref_btn.clicked.connect(self._refresh_list)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        for b in (del_btn, ref_btn, cancel_btn):
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

    def _refresh_list(self):
        entries = self._scan_entries()
        for lw in (self._by_sess, self._by_name):
            lw.clear()
        by_sess: dict[str, list] = {}
        by_name: dict[str, list] = {}
        for e in entries:
            by_sess.setdefault(e['session'] or '(unknown)', []).append(e)
            by_name.setdefault(e['name'], []).append(e)
        for sess in sorted(by_sess):
            for e in sorted(by_sess[sess], key=lambda x: x['name']):
                it = QListWidgetItem(e['display'])
                it.setData(256, e)
                self._by_sess.addItem(it)
        for name in sorted(by_name):
            for e in sorted(by_name[name], key=lambda x: x['session']):
                it = QListWidgetItem(e['display'])
                it.setData(256, e)
                self._by_name.addItem(it)

    def _selected_entries(self) -> list[dict]:
        lw = self._by_sess if self._tabs.currentIndex() == 0 else self._by_name
        return [it.data(256) for it in lw.selectedItems() if it.data(256)]

    def _load_selected(self):
        entries = self._selected_entries()
        if not entries:
            QMessageBox.information(self, "Load", "Select one or more entries.")
            return
        cd = self._mgr._ui.nav.cd
        keys = [Key(session=e['session'], segment=e['name']) for e in entries]
        cd.load_segment(keys)
        self._mgr._ui.nav.custom_segs_changed.emit()
        QMessageBox.information(self, "Load", f"Loaded {len(entries)} custom CCG(s).")
        self.accept()

    def _delete_selected(self):
        entries = self._selected_entries()
        if not entries:
            return
        # Each entry is a distinct (session, name); delete exactly those, not the whole name.
        targets = sorted({(e['session'], e['name']) for e in entries})
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Delete custom CCG")
        box.setText(f"Are you sure you want to delete {len(targets)} custom CCG(s)?")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        cd = self._mgr._ui.nav.cd
        cd.delete_segment([Key(session=s, segment=n) for s, n in targets])
        self._refresh_list()
        nav = self._mgr._ui.nav
        nav.clamp_segment()
        nav.custom_segs_changed.emit()

    @classmethod
    def show(cls, custom_mgr, time_slider=None, *, select_mode: bool = False,
             parent=None):
        dlg = cls(custom_mgr, time_slider, select_mode=select_mode, parent=parent)
        dlg.exec()


def _dual_list(parent_layout, label_avail: str, label_sel: str,
               choices: list, preselected: list) -> tuple:
    """Build a dual-listbox (available | selected) widget. Returns (avail_lb, sel_lb, selected_list)."""
    grp = QGroupBox()
    grp_lay = QHBoxLayout(grp)

    left = QVBoxLayout()
    left.addWidget(QLabel(label_avail))
    avail_lb = QListWidget()
    avail_lb.setSelectionMode(QAbstractItemView.ExtendedSelection)
    left.addWidget(avail_lb)
    grp_lay.addLayout(left)

    mid = QVBoxLayout()
    mid.addStretch()
    add_btn = QPushButton("→")
    rem_btn = QPushButton("←")
    add_btn.setFixedWidth(32)
    rem_btn.setFixedWidth(32)
    mid.addWidget(add_btn)
    mid.addWidget(rem_btn)
    mid.addStretch()
    grp_lay.addLayout(mid)

    right = QVBoxLayout()
    right.addWidget(QLabel(label_sel))
    sel_lb = QListWidget()
    sel_lb.setSelectionMode(QAbstractItemView.ExtendedSelection)
    right.addWidget(sel_lb)
    grp_lay.addLayout(right)

    parent_layout.addWidget(grp)

    selected = list(preselected)
    sel_set  = set(preselected)

    for c in choices:
        if c not in sel_set:
            avail_lb.addItem(c)
    for c in preselected:
        if c in choices or c in ('Current', 'All'):
            sel_lb.addItem(c)

    def _add():
        for it in avail_lb.selectedItems():
            t = it.text()
            if t not in selected:
                selected.append(t)
                sel_lb.addItem(t)
            avail_lb.takeItem(avail_lb.row(it))

    def _rem():
        for it in sel_lb.selectedItems():
            t = it.text()
            if t in selected:
                selected.remove(t)
            sel_lb.takeItem(sel_lb.row(it))
            avail_lb.addItem(t)

    add_btn.clicked.connect(_add)
    rem_btn.clicked.connect(_rem)
    avail_lb.doubleClicked.connect(lambda: _add())
    sel_lb.doubleClicked.connect(lambda: _rem())

    return avail_lb, sel_lb, selected


class ExportOptionsDialog(QDialog):
    """Export options + preview.  Returns opt-dict or None on cancel."""

    def __init__(self, nav: 'AppState', cd, sel_data: 'SelectionData',
                 group_mgr: 'PairSelectionPanel', ui_state: dict,
                 fmt: str = 'png', preview_pair=None, selected_pairs=None,
                 segment_names: list = None, parent=None):
        super().__init__(parent)
        self.nav            = nav
        self.cd             = cd
        self.sel_data       = sel_data
        self._group_mgr      = group_mgr
        self._groups         = group_mgr.ui.groups
        self._ui_state       = ui_state
        self._fmt            = fmt
        self._preview_pair   = preview_pair
        self._selected_pairs = selected_pairs or []
        self._segment_names  = segment_names or ['Current', 'All']
        self._out: dict | None = None
        self._action         = 'current'

        defs = ui_state.get('export_defaults', {}) or {}
        self._defs = defs

        self.setWindowTitle(f"Export options ({fmt.upper()})")
        self.resize(900, 700)
        self._build()


    def _sv(self, key, default='') -> QLineEdit:
        e = QLineEdit(str(self._defs.get(key) or default))
        return e

    def _bv(self, key, default=True) -> QCheckBox:
        cb = QCheckBox()
        cb.setChecked(bool(self._defs.get(key, default)))
        return cb


    def _build(self):
        root = QHBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(480)
        options_widget = QWidget()
        opt_lay = QVBoxLayout(options_widget)
        scroll.setWidget(options_widget)
        root.addWidget(scroll, stretch=3)

        right_col = QVBoxLayout()
        root.addLayout(right_col, stretch=2)

        # Preview plot
        self._preview_plot = pg.PlotWidget()
        self._preview_plot.setMinimumHeight(220)
        self._preview_plot.setBackground('w')
        right_col.addWidget(QLabel("Preview (current segment):"))
        right_col.addWidget(self._preview_plot, stretch=1)

        right_col.addStretch()

        # Scope buttons
        scope_box = QGroupBox("Export scope")
        scope_lay = QVBoxLayout(scope_box)
        scope_lay.addWidget(QPushButton(
            f"Export current as {self._fmt.upper()}",
            clicked=lambda: self._done('current')))
        if len(self._selected_pairs) > 1:
            scope_lay.addWidget(QPushButton(
                f"Export all selected ({len(self._selected_pairs)})…",
                clicked=lambda: self._done('all')))
        n_bm = len(self.nav.bookmarked_pairs)
        if n_bm > 0:
            scope_lay.addWidget(QPushButton(
                f"Export bookmarked ({n_bm})…",
                clicked=lambda: self._done('bookmarked')))
        all_groups = sorted(
            g for g in self._groups.defined_groups
            if g and not str(g).startswith('__'))
        if all_groups:
            scope_lay.addWidget(QPushButton(
                "Export selected group(s)…",
                clicked=lambda: self._done('groups')))
        right_col.addWidget(scope_box)

        save_btn = QPushButton("Save as defaults")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self._save_defaults)
        cancel_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        right_col.addLayout(btn_row)

        color_box = QGroupBox("Colors (name or #hex, blank = default)")
        cl = QVBoxLayout(color_box)
        self._ccg_color    = self._sv('ccg_color')
        self._base_color   = self._sv('baseline_color')
        self._tw_color     = self._sv('test_window_color')
        self._tw_alpha     = self._sv('test_window_alpha', '0.12')
        self._pval_color   = self._sv('pval_line_color')
        self._alpha_color  = self._sv('alpha_line_color')
        self._cs_shade_col = self._sv('cs_shade_color')
        for lbl, w in [
            ("CCG:", self._ccg_color), ("Baseline:", self._base_color),
            ("Test window:", self._tw_color), ("TW alpha:", self._tw_alpha),
            ("P-value line:", self._pval_color), ("Alpha line:", self._alpha_color),
            ("CS shade:", self._cs_shade_col),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl), 1)
            row.addWidget(w, 2)
            cl.addLayout(row)
        opt_lay.addWidget(color_box)

        misc_box = QGroupBox("Appearance")
        ml = QVBoxLayout(misc_box)
        self._ccg_alpha  = self._sv('ccg_alpha',  '0.5')
        self._base_alpha = self._sv('baseline_alpha', '0.3')
        self._min_text   = self._sv('min_text_size', '8')
        self._xticks     = self._sv('xticks_raw')
        self._mirror_cb  = self._bv('mirror_xticks', True)
        self._legend_cb  = self._bv('show_legend', True)
        self._mirror_cb.setText("Mirror x-ticks to negative")
        self._legend_cb.setText("Show legend")
        for lbl, w in [
            ("CCG alpha:", self._ccg_alpha), ("Baseline alpha:", self._base_alpha),
            ("Min text size (pt):", self._min_text),
            ("X-ticks (ms, comma-sep):", self._xticks),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl), 1)
            row.addWidget(w, 2)
            ml.addLayout(row)
        ml.addWidget(self._mirror_cb)
        ml.addWidget(self._legend_cb)
        opt_lay.addWidget(misc_box)

        title_box = QGroupBox("Title")
        tl = QHBoxLayout(title_box)
        self._t_shanks = self._bv('title_show_shanks', True);   self._t_shanks.setText("Shanks")
        self._t_inds   = self._bv('title_show_inds',   True);   self._t_inds.setText("Inds")
        self._t_type   = self._bv('title_show_type',   True);   self._t_type.setText("Type")
        self._t_seg    = self._bv('title_show_seg',    True);   self._t_seg.setText("Segment")
        self._t_norm   = self._bv('title_show_norm_details', True); self._t_norm.setText("Norm")
        self._t_sess   = self._bv('title_show_session', False);  self._t_sess.setText("Session")
        for cb in (self._t_shanks, self._t_inds, self._t_type,
                   self._t_seg, self._t_norm, self._t_sess):
            tl.addWidget(cb)
        opt_lay.addWidget(title_box)

        res_box = QGroupBox("Resolution & CS")
        rl = QHBoxLayout(res_box)
        self._lo_res  = self._bv('export_lores', True);  self._lo_res.setText("Lo-res")
        self._hi_res  = self._bv('export_hires', False); self._hi_res.setText("Hi-res")
        self._stg_cb  = self._bv('print_cs_stg',  False); self._stg_cb.setText("Print STG")
        self._jbsi_cb = self._bv('print_cs_jbsi', False); self._jbsi_cb.setText("Print JBSI")
        for cb in (self._lo_res, self._hi_res, self._stg_cb, self._jbsi_cb):
            rl.addWidget(cb)
        opt_lay.addWidget(res_box)

        seg_default = list(self._defs.get('export_segments') or ['Current'])
        _, _, self._selected_segments = _dual_list(
            opt_lay, "Available segments", "Export segments",
            self._segment_names,
            [s for s in seg_default if s in self._segment_names or s in ('Current', 'All')])

        sf_choices = ["conn type", "excitatory/inhibitory", "session", "animal"]
        sf_default = [x for x in (self._defs.get('subfolder_by') or []) if x in sf_choices]
        _, _, self._selected_subfolders = _dual_list(
            opt_lay, "Available subfolder keys", "Subfolder by (ordered)",
            sf_choices, sf_default)

        _, _, self._selected_groups = _dual_list(
            opt_lay, "Available groups", "Selected groups",
            all_groups, [])

        opt_lay.addStretch()

        # Initial preview
        self._refresh_preview()

        # Re-preview when color fields change
        for w in (self._ccg_color, self._base_color, self._tw_color,
                  self._ccg_alpha, self._base_alpha):
            w.textChanged.connect(lambda _: self._refresh_preview())


    def _refresh_preview(self):
        """Draw a quick pyqtgraph preview of the current pair's CCG."""
        if self._preview_plot is None or self._preview_pair is None:
            return
        try:
            ref, tgt = int(self._preview_pair[0]), int(self._preview_pair[1])
            nav = self.nav
            key = nav.get_complete_key().change(segment=0, ref=ref, tgt=tgt)  # seg 0 = whole session
            slices = nav.cd.pair_slices(key)
            if slices is None:
                return
            ccg_raw, null_raw = slices[0].astype(float), slices[1]
            null_raw = null_raw.astype(float) if null_raw is not None else None

            cd_data = nav.cd.ccg_for(key)
            n  = len(ccg_raw)
            bs = cd_data.conf.bin_size * 1000.0
            ws = cd_data.conf.duration * 1000.0
            xs = np.linspace(-ws / 2, ws / 2, n)

            p = self._preview_plot
            p.clear()
            color = (self._ccg_color.text().strip() or '#4a7fd4')
            try:
                alpha = float(self._ccg_alpha.text() or 0.5)
            except ValueError:
                alpha = 0.5
            p.addItem(pg.BarGraphItem(x=xs, height=ccg_raw,
                                      width=bs * 0.85, brush=color, pen=None))
            if null_raw is not None:
                bc = (self._base_color.text().strip() or '#e88')
                p.plot(xs, null_raw, pen=pg.mkPen(bc, width=1.5))
            p.setLabel('bottom', 'Lag (ms)')
        except Exception:
            pass


    def _collect(self) -> dict:
        def _s(w): return w.text().strip() or None
        def _f(w, default):
            try: return float(w.text())
            except Exception: return default
        o: dict = {
            'ccg_color':          _s(self._ccg_color),
            'baseline_color':     _s(self._base_color),
            'test_window_color':  _s(self._tw_color),
            'test_window_alpha':  _f(self._tw_alpha, 0.12),
            'pval_line_color':    _s(self._pval_color),
            'alpha_line_color':   _s(self._alpha_color),
            'cs_shade_color':     _s(self._cs_shade_col),
            'ccg_alpha':          _f(self._ccg_alpha, 0.5),
            'baseline_alpha':     _f(self._base_alpha, 0.3),
            'min_text_size':      _f(self._min_text, 8),
            'show_legend':        self._legend_cb.isChecked(),
            'mirror_xticks':      self._mirror_cb.isChecked(),
            'xticks_raw':         self._xticks.text().strip(),
            'export_lores':       self._lo_res.isChecked(),
            'export_hires':       self._hi_res.isChecked(),
            'print_cs_stg':       self._stg_cb.isChecked(),
            'print_cs_jbsi':      self._jbsi_cb.isChecked(),
            'title_show_shanks':       self._t_shanks.isChecked(),
            'title_show_inds':         self._t_inds.isChecked(),
            'title_show_type':         self._t_type.isChecked(),
            'title_show_seg':          self._t_seg.isChecked(),
            'title_show_norm_details': self._t_norm.isChecked(),
            'title_show_session':      self._t_sess.isChecked(),
            'export_segments':    list(self._selected_segments) or ['Current'],
            'subfolder_by':       list(self._selected_subfolders),
        }
        raw = o['xticks_raw']
        if raw:
            try:
                o['xticks_ms'] = [float(x.strip()) for x in raw.split(',') if x.strip()]
            except Exception:
                o['xticks_ms'] = None
        else:
            o['xticks_ms'] = None
        return o

    def _save_defaults(self):
        self._ui_state['export_defaults'] = self._collect()
        try:
            self._group_mgr._save_all_state(selection_name=None, silent=True)
        except Exception:
            pass

    def _done(self, action: str):
        self._action = action
        self._out = self._collect()
        self._out['_action']          = action
        self._out['_selected_pairs']  = list(self._selected_pairs)
        self._out['_selected_groups'] = list(self._selected_groups)
        self.accept()

    @classmethod
    def show(cls, nav: 'AppState', fmt: str = 'png', preview_pair=None,
             selected_pairs=None) -> dict | None:
        """Show dialog. Returns opt-dict or None on cancel."""
        cd        = nav.cd
        sel_data  = nav.sel_data
        group_mgr = nav.root.pairs_view.left_panel
        ui_state  = nav.root._load_ui_state()
        seg_names = nav.available_segments()

        parent = nav.root
        dlg = cls(nav, cd, sel_data, group_mgr, ui_state, fmt,
                  preview_pair, selected_pairs, seg_names, parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg._out
        return None


class SettingsTabs:
    """West-tab settings pages, reusable shell.
    """

    def __init__(self, ui, nav: 'SideNavPanel'):
        self._ui = ui
        self.nav = nav
        self._appliers: list = []
        self._autosave_sel_cb = self._autosave_sel_metric = None
        self._autosave_grp_cb = self._autosave_grp_metric = None
        self._save_ui_cb = None
        self._build()

    def _spin_row(self, layout, label, obj, attr, lo, hi, on_live=None):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(getattr(obj, attr))
        if on_live is not None:
            spin.valueChanged.connect(on_live)
        row.addWidget(spin)
        layout.addLayout(row)
        self._appliers.append(lambda: setattr(obj, attr, spin.value()))
        return spin

    def _build(self):
        ui = self._ui
        nav = self.nav

        disp = QWidget()
        dl = QVBoxLayout(disp)
        dl.setSpacing(8)
        self._spin_row(dl, "Max pairs in 'Show Together':",
                       ui.nav, 'max_together_pairs', 2, 20)
        self._spin_row(dl, "Minimum font size:", ui.settings, 'min_font_size', 6, 32,
                       on_live=lambda v: ui._apply_min_font_size(v))
        self._spin_row(dl, "Classifier: min pairs per label:", ui.settings,
                       'classifier_min_count', 5, 500)
        dl.addStretch()
        nav.add_page("Display", disp)

        cache = QWidget()
        cl = QVBoxLayout(cache)
        cl.setSpacing(8)
        self._spin_row(cl, "Max CCG queue size:", ui.nav, 'max_ccg_queue', 1, 500)
        self._spin_row(cl, "Max jitter queue size:", ui.nav, 'max_jitter_queue', 1, 500)
        self._spin_row(cl, "Max jitter cache size:", ui.nav, 'max_jitter_cache', 1, 5000)
        cl.addStretch()
        nav.add_page("Cache", cache)

        auto = QWidget()
        al = QVBoxLayout(auto)
        al.setSpacing(6)

        def _interval_row(label, attr_prefix):
            row = QHBoxLayout()
            s = ui.settings
            cb = QCheckBox(label)
            cb.setChecked(getattr(s, f'{attr_prefix}_on'))
            metric = MetricInput("", ['min', 'hour', 'day'],
                                 suggestions=(1, 5, 10, 15, 30, 60), input_width=60)
            metric.set_value(*getattr(s, f'{attr_prefix}_interval'))
            row.addWidget(cb)
            row.addWidget(metric)
            row.addStretch()
            setattr(self, f'_{attr_prefix}_cb', cb)
            setattr(self, f'_{attr_prefix}_metric', metric)
            al.addLayout(row)

        _interval_row("Autosave selections:", 'autosave_sel')
        _interval_row("Autosave groups:", 'autosave_grp')
        row_ui = QHBoxLayout()
        self._save_ui_cb = QCheckBox("Save UI status")
        self._save_ui_cb.setChecked(ui.settings.save_ui_on_close)
        row_ui.addWidget(self._save_ui_cb)
        al.addLayout(row_ui)
        clear_btn = QPushButton("Clear autosaved data")
        clear_btn.clicked.connect(self._on_clear_autosave)
        al.addWidget(clear_btn)
        al.addStretch()
        nav.add_page("Autosave", auto)

    def apply(self):
        ui = self._ui
        for write in self._appliers:   # each Tunable write fans out via its on_change
            write()
        ui._apply_min_font_size(ui.settings.min_font_size)
        ui.settings.autosave_sel_on = self._autosave_sel_cb.isChecked()
        ui.settings.autosave_sel_interval = self._autosave_sel_metric.value()
        ui.settings.autosave_grp_on = self._autosave_grp_cb.isChecked()
        ui.settings.autosave_grp_interval = self._autosave_grp_metric.value()
        ui.settings.save_ui_on_close = self._save_ui_cb.isChecked()

    def _on_clear_autosave(self):
        ui = self._ui
        history_dir = os.path.join(ui._sel_save_dir, '.history')
        removed = 0
        for f in glob.glob(os.path.join(history_dir, '*.autosaved.json')):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
        QMessageBox.information(self.nav, "Clear autosaved data",
                                f"Removed {removed} autosaved file(s).")


class _TargetBox(QFrame):
    """One schema field's slot: click to select it, holds the chips assigned to it."""

    def __init__(self, field, on_click, on_unassign):
        super().__init__()
        self.field = field
        self.columns: list[str] = []
        self.value_map: dict = {}       # source value -> what the loader should see
        self.needs_map = False          # assigned values don't say what the field wants yet
        self._selected = False
        self._on_unassign = on_unassign
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._flow = FlowLayout(self, spacing=3)
        self.mousePressEvent = lambda _e: on_click(self)
        self.set_selected(False)

    def set_selected(self, on: bool):
        self._selected = on
        self.restyle()

    def restyle(self):
        """Blue while selected, amber while the assigned values still need a map."""
        border = ('2px solid #4a7fd4' if self._selected
                  else '2px solid #d99000' if self.needs_map else '1px solid #bbb')
        self.setStyleSheet(f"QFrame {{ border: {border};"
                           f" border-radius: 3px; min-height: 24px; }}")

    def binding(self):
        """What this box contributes to a mapping — the dict form only when values remap."""
        return {'col': self.columns, 'map': self.value_map} if self.value_map else self.columns

    def set_columns(self, columns: list):
        """Replace the assigned columns and rebuild their chips."""
        self.columns = list(columns)
        self._flow.clear_widgets()
        for col in self.columns:
            chip = chip_button(f"{col} ✕", checkable=False)
            chip.clicked.connect(lambda _c, x=col: self._on_unassign(self, x))
            self._flow.addWidget(chip)
            chip.show()
        self.updateGeometry()


class FieldMapWidget(QWidget):
    """Edit a field map: source columns on the left, one target drop box per schema field."""

    def __init__(self, field_map, available: list, column_values=None, parent=None):
        super().__init__(parent)
        self.field_map = field_map      # FieldMap being edited
        self.available = available      # the dataset's real column names
        self.column_values = column_values   # NWBDataset.column_values, once a source is scanned
        self._boxes = {}                # field name -> _TargetBox
        self._selected: _TargetBox = None
        self._build()

    def _build(self):
        """Source list on the left, one drop box per schema field on the right."""
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Available columns"))
        self._source_host = QWidget()
        self._source_flow = FlowLayout(self._source_host, spacing=3)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self._source_host)
        left.addWidget(area)
        root.addLayout(left, stretch=1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Target fields  (* required)"))
        for field in self.field_map.schema:
            row = QHBoxLayout()
            row.addWidget(QLabel(field.name + (' *' if field.required else '')))
            box = _TargetBox(field, self._on_target_box_click, self._on_unassign_chip)
            box.setToolTip(field.note)
            row.addWidget(box, stretch=1)
            right.addLayout(row)
            self._boxes[field.name] = box
        right.addStretch()
        root.addLayout(right, stretch=2)

        self.set_available(self.available)
        self.set_mapping(self.field_map.mapping)

    def set_available(self, columns: list):
        """Repopulate the source chips from a dataset's real column names."""
        self.available = list(columns)
        self._source_flow.clear_widgets()
        for col in self.available:
            chip = chip_button(col, checkable=True)
            self._source_flow.addWidget(chip)
            chip.show()
        self._source_host.updateGeometry()

    def set_mapping(self, mapping: dict):
        """Fill the target boxes from a {"target": "input"} dict, value maps included."""
        for name, box in self._boxes.items():
            value = mapping.get(name)
            if isinstance(value, dict):
                value, box.value_map = value.get('cols', value.get('col')), value.get('map') or {}
            else:
                box.value_map = {}
            cols = [] if value is None else ([value] if isinstance(value, str) else list(value))
            box.set_columns([c for c in cols if c in self.available])
            self._mark_unmapped(box)

    def _checked_chips(self) -> list:
        """Source chips currently ticked, in display order."""
        return [self._source_flow.itemAt(i).widget()
                for i in range(self._source_flow.count())
                if self._source_flow.itemAt(i).widget().isChecked()]

    def _on_target_box_click(self, box: '_TargetBox'):
        """Select this box, then move any ticked source chips into it."""
        if self._selected is not None:
            self._selected.set_selected(False)
        self._selected = box
        box.set_selected(True)
        picked = self._checked_chips()
        if not picked and box.needs_map:
            self._map_values(box)
            return
        if picked:
            cols = [c.text() for c in picked]
            box.set_columns(box.columns + cols if box.field.multi else cols[:1])
            for chip in picked:
                chip.setChecked(False)
            self._mark_unmapped(box)
            if box.needs_map:
                self._map_values(box)

    def _mark_unmapped(self, box: '_TargetBox'):
        """Amber the box while no value reaches the field — unmapped ones are ignored, not wrong."""
        if box.field.value_map and self.column_values and box.columns:
            values = self.column_values(box.columns[0])
            box.needs_map = not any(box.value_map.get(v, v) in box.field.values
                                    for v in values)
        box.restyle()

    def _map_values(self, box: '_TargetBox'):
        """Open the value editor for the assigned column."""
        column = box.columns[0]
        edited = ValueMapEditor.edit(f"Map {column}", self.column_values(column),
                                     box.field.values, box.value_map, self)
        if edited is not None:
            box.value_map = edited
        self._mark_unmapped(box)

    def _on_unassign_chip(self, box: '_TargetBox', column: str):
        box.set_columns([c for c in box.columns if c != column])
        box.value_map = {}
        self._mark_unmapped(box)

    def mapping(self) -> dict:
        """The {"target": "input"} dict this widget describes, value maps included."""
        return {name: box.binding() for name, box in self._boxes.items() if box.columns}


class AddProjectDialog(QDialog):
    """Add a project: pick the source folder, map the fields its sessions share, load them."""

    def __init__(self, nav: 'AppState', parent=None, draft: ProjectConfig = None):
        super().__init__(parent)
        self.nav = nav
        self._out = None
        self._dataset = None    # NWBDataset, set once a source folder is scanned
        self.setWindowTitle("Add Project")
        self._build()
        if draft is not None:
            self._restore_draft(draft)

    def _build(self):
        """Path row + FieldMapWidget + config options + Ok/Cancel."""
        self.resize(760, 620)
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        row.addWidget(self._name_edit, stretch=1)
        row.addWidget(QLabel("Source:"))
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        row.addWidget(self._path_edit, stretch=2)
        row.addWidget(make_button("Browse…", self._on_browse_btn))
        row.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItem("NWB", NWBDataset)
        row.addWidget(self._format_combo)
        self._dataset_name = None      # the io.datasets module a prefill came from, if any
        row.addWidget(make_button("Prefill from dataset…", self._on_prefill_btn))
        lay.addLayout(row)

        self._map_widget = FieldMapWidget(FieldMap(UNITS_SCHEMA, {}, partial=True), [])
        box = QGroupBox("Fields to load")
        QVBoxLayout(box).addWidget(self._map_widget)

        opts = QScrollArea()
        opts.setWidgetResizable(True)
        host = QWidget()
        host_lay = QVBoxLayout(host)
        self._nd_opts = ConfigOptionsWidget(NeuronsDatasetConfig, "Neurons")
        self._nd_opts.add_row("sampling_rate", self._rate_row())   # how spike times are read
        self._ccg_opts = ConfigOptionsWidget(CCGConfig, "CCG")
        host_lay.addWidget(self._nd_opts)
        host_lay.addWidget(self._ccg_opts)
        opts.setWidget(host)

        sash = QSplitter(Qt.Orientation.Vertical)   # drag to trade field map for options
        sash.addWidget(box)
        sash.addWidget(opts)
        sash.setSizes([300, 300])
        lay.addWidget(sash, stretch=1)

        self._compute_check = QCheckBox("Compute CCGs after building")
        lay.addWidget(self._compute_check)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Build project")
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_browse_btn(self):
        """Pick the project root (or one file), scan it, and refresh the field map widget."""
        path = QFileDialog.getExistingDirectory(self, "Choose project root")
        if not path:
            return
        if not self._name_edit.text().strip():
            self._name_edit.setText(os.path.basename(os.path.normpath(path)))
        self._scan_source(path)

    def _scan_source(self, path: str, mapping: dict = None):
        """Read every session's columns under *path*; *mapping* seeds the field map."""
        scanner = self._format_combo.currentData()
        self.setEnabled(False)
        self._dataset = scanner(path, on_progress=self._on_scan_progress)
        self.setEnabled(True)
        n = len(self._dataset.files)
        self._path_edit.setText(f"{path}   ({n} sessions, "
                                f"{len(self._dataset.input_fields)} fields)")
        self._map_widget.set_available(self._dataset.input_fields)
        self._map_widget.column_values = self._dataset.column_values
        self._map_widget.set_mapping(mapping or NWB_DEFAULT)
        self._nd_opts.set_choices('themes', self._dataset.themes)

    def _restore_draft(self, draft: ProjectConfig):
        """Refill the dialog from a saved-but-unbuilt header so the build can be finished."""
        self._name_edit.setText(draft.name)
        self._format_combo.setCurrentText(draft.format)
        self._dataset_name = draft.dataset
        self._rate_edit.setText('' if draft.sampling_rate is None else str(draft.sampling_rate))
        self._scan_source(draft.source, draft.fields)   # its own mapping, never a default
        for name, value in (draft.nd_conf or {}).items():
            self._nd_opts.set_value(name, value)

    def _rate_row(self) -> QWidget:
        """Sampling-rate box plus the button that recovers it from the spike times."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self._rate_edit = QLineEdit()
        self._rate_edit.setFixedWidth(80)
        self._rate_edit.setPlaceholderText("derived")
        self._rate_edit.setToolTip("Clock the spike times quantize onto; blank derives per file")
        row.addWidget(self._rate_edit)
        row.addWidget(make_button("Derive", self._on_derive_rate_btn))
        row.addStretch()
        return w

    def _on_derive_rate_btn(self):
        """Recover the spike clock from the smallest gap between pooled spike times."""
        if self._dataset is None:
            QMessageBox.information(self, "Sampling rate", "Choose a source folder first.")
            return
        with NWBFile(self._dataset.files[0]) as f:
            rate = f.sampling_rate
        if rate is None:
            QMessageBox.information(self, "Sampling rate",
                                    "This session has too few spikes to derive a rate.")
            return
        self._rate_edit.setText(str(rate))

    def _on_prefill_btn(self):
        """Pick a known dataset and take its field map, value renames included."""
        names = [m.name for m in pkgutil.iter_modules(datasets.__path__)]
        name, ok = QInputDialog.getItem(self, "Prefill from dataset",
                                        "Dataset conventions:", names, 0, False)
        if ok and name:
            self._dataset_name = name
            self._map_widget.set_mapping(import_module(
                f'neuropy.io.datasets.{name}').FIELDS)

    def _on_scan_progress(self, i: int, total: int, path):
        self._path_edit.setText(f"scanning {i + 1}/{total}  {path.name}")
        QApplication.processEvents()

    def _on_accept(self):
        """Check the mapping against the schema, build the project, store it in _out."""
        name = self._name_edit.text().strip()
        if not name or self._dataset is None:
            QMessageBox.information(self, "Add project",
                                    "A project name and a source folder are both required.")
            return
        existing = ProjectConfig(name=name)
        if os.path.isfile(existing.save_path() + '.json'):
            existing.load()
        if existing.built:   # a header with no build behind it is a failed attempt, not a project
            QMessageBox.information(self, "Add project",
                                    f"Project '{name}' already exists "
                                    f"({existing.n_sessions} sessions, built {existing.built_at}).")
            return
        try:
            field_map = FieldMap(UNITS_SCHEMA, self._map_widget.mapping())
        except ValueError as e:
            QMessageBox.information(self, "Fields to load", str(e))
            return
        sessions = self._dataset.sessions(field_map)
        if not sessions:
            QMessageBox.information(self, "Add project",
                                    "No session supplies every required field.\n\n"
                                    + self._dataset.report(field_map))
            return
        if QMessageBox.question(self, "Add project",
                                self._dataset.report(field_map) + "\n\nBuild the project?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        header = ProjectConfig(name=name, source=str(self._dataset.path),
                               format=self._format_combo.currentText(),
                               dataset=self._dataset_name,
                               fields=field_map.mapping,
                               nd_conf=self._nd_opts.values(),
                               sampling_rate=float(self._rate_edit.text() or 0) or None)
        neurons, cd, sd = build_project(header, CCGConfig(name=name, **self._ccg_opts.values()),
                                        compute=self._compute_check.isChecked())
        self._out = (neurons, cd, sd)
        self.accept()

    @classmethod
    def show(cls, nav, parent=None, draft: ProjectConfig = None):
        """Run the dialog; returns the loaded project, or None if cancelled."""
        dlg = cls(nav, parent, draft)
        dlg.exec()
        return dlg._out


class SettingsDialog(QDialog):
    """Settings > Settings — modal wrapper around SettingsTabs."""

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self._ui = ui
        self.setWindowTitle("Settings")
        self.resize(480, 340)
        root = QVBoxLayout(self)
        nav = SideNavPanel(self, min_width=120, nav_width=160)
        self._settings = SettingsTabs(ui, nav)
        root.addWidget(nav)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _on_accept(self):
        self._settings.apply()
        self.accept()

    @classmethod
    def show(cls, ui, parent=None):
        dlg = cls(ui, parent)
        dlg.exec()
