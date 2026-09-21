"""Manage any GroupDataset: rename, hotkey, notes, delete, and list what is tagged.

The registry is the only thing that differs between views, so one page serves pair
groups, neuron groups and whatever a later view stores.
"""
from __future__ import annotations

from pyqtgraph.Qt.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget,
                                    QMessageBox, QPlainTextEdit, QPushButton,
                                    QSplitter, QVBoxLayout, QWidget)
from pyqtgraph.Qt.QtCore import Qt

from neuropy.ui.utils import widget_row


class GroupManagerPage(QWidget):
    """One registry's groups: pick one, edit it, see its members."""

    def __init__(self, groups, member_label=lambda m: str(m),
                 session_of=lambda: '', parent=None):
        super().__init__(parent)
        self._groups = groups
        self._member_label = member_label
        self._session_of = session_of
        self._build()
        self.reload()

    def _build(self) -> None:
        split = QSplitter(Qt.Horizontal, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)

        self._list = QListWidget()
        self._list.currentTextChanged.connect(self._on_group_changed)
        split.addWidget(self._list)

        editor = QWidget()
        form = QVBoxLayout(editor)
        self._name_edit = QLineEdit()
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._on_rename_btn)
        form.addLayout(widget_row("Name:", self._name_edit, rename_btn))

        self._hotkey_edit = QLineEdit()
        self._hotkey_edit.setMaxLength(1)
        self._hotkey_edit.setFixedWidth(40)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._on_hotkey_btn)
        form.addLayout(widget_row("Hotkey:", self._hotkey_edit, set_btn))

        self._notes = QPlainTextEdit()
        self._notes.setPlaceholderText("Notes")
        form.addWidget(self._notes)

        form.addWidget(QLabel("Tagged:"))
        self._members = QListWidget()
        form.addWidget(self._members, stretch=1)

        delete_btn = QPushButton("Delete group")
        delete_btn.clicked.connect(self._on_delete_btn)
        form.addWidget(delete_btn)
        split.addWidget(editor)
        split.setSizes([180, 420])

    @property
    def current_group(self) -> str:
        item = self._list.currentItem()
        return '' if item is None else item.text()

    def reload(self) -> None:
        """Rebuild the group list, keeping the selected group when it survives."""
        keep = self.current_group
        self._list.blockSignals(True)
        self._list.clear()
        self._list.addItems(self._groups.groups)
        if keep:
            matches = self._list.findItems(keep, Qt.MatchExactly)
            if matches:
                self._list.setCurrentItem(matches[0])
        elif self._list.count():
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._load_editor()

    def _load_editor(self) -> None:
        gname = self.current_group
        self.setEnabled(True)
        if not gname:
            self._members.clear()
            return
        meta = self._groups.get_group_metadata(gname)
        self._name_edit.setText(meta.display_name)
        self._hotkey_edit.setText(meta.hotkey or '')
        self._notes.setPlainText(meta.notes or '')
        self._members.clear()
        self._members.addItems(
            [self._member_label(m) for m in
             sorted(self._groups.members_in_group(gname, self._session_of()))])

    def _on_group_changed(self, _name: str) -> None:
        self._load_editor()

    def _mutate(self, what: str, change) -> None:
        """Run one registry change, report a rejection, then save and reload."""
        try:
            change()
        except ValueError as exc:
            QMessageBox.warning(self, what, str(exc))
            return
        self._groups.save()
        self.reload()

    def _on_rename_btn(self) -> None:
        gname, new = self.current_group, self._name_edit.text().strip()
        if gname and new and new != gname:
            self._mutate("Rename", lambda: self._groups.rename_group(gname, new))

    def _on_hotkey_btn(self) -> None:
        gname, typed = self.current_group, self._hotkey_edit.text()
        if gname:
            self._mutate("Hotkey", lambda: self._groups.set_group_hotkey(
                gname, self._groups.valid_hotkey(typed)))

    def _on_delete_btn(self) -> None:
        gname = self.current_group
        if not gname:
            return
        if QMessageBox.question(self, "Delete group",
                                f"Delete {gname!r}?") == QMessageBox.Yes:
            self._mutate("Delete", lambda: self._groups.delete_group(gname))

    def save_notes(self) -> None:
        """Persist the notes field; call before the page loses focus."""
        gname = self.current_group
        if gname:
            self._groups.get_group_metadata(gname).notes = self._notes.toPlainText()
            self._groups.save()
