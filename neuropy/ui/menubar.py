"""Top bar: project/session/type index + main menu."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyqtgraph.Qt.QtCore import QObject, QThread, Signal
from pyqtgraph.Qt.QtWidgets import QWidget, QHBoxLayout, QLabel, QMessageBox
from pyqtgraph.Qt.QtGui import QAction

from neuropy.analyses.ms_connectivity import ProjectConfig, build_project, projects_on_disk
from neuropy.analyses.neurons_dataset import Key
from neuropy.ui.app_state import _ALL_SESSION_MARKER
from neuropy.ui.dialogs import (AddProjectDialog, CreateGroupDialog, ExportOptionsDialog,
                                SettingsDialog)
from neuropy.ui.jitter_ui import JitterQueueDialog
from neuropy.ui.utils import AddableDropdown, chip_button, make_button

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI


class _BuildProjectWorker(QObject):
    """Runs one blocking build_project off the Qt main thread."""
    done  = Signal(object)
    error = Signal(str)

    def __init__(self, header, ccg_conf, compute: bool):
        super().__init__()
        self._header, self._ccg_conf, self._compute = header, ccg_conf, compute

    def run(self):
        try:
            built = build_project(self._header, self._ccg_conf, compute=self._compute)
        except Exception as e:
            self.error.emit(str(e))
            return
        self.done.emit(built)


class IndexBar:
    """Project / session / conn-type dropdowns."""

    def __init__(self, win: CCGReviewUI):
        self._win = win
        self.widget = QWidget()
        row = QHBoxLayout(self.widget)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)

        row.addWidget(QLabel("Project:"))
        self.project_combo = AddableDropdown('project', self.add_project, width=120)
        self.project_combo.currentIndexChanged.connect(self._on_project_changed)
        row.addWidget(self.project_combo)
        self.edit_project_btn = make_button("⚙", self.edit_project, width=26)
        self.edit_project_btn.setToolTip("Edit this project — rescan the source and remap its fields")
        row.addWidget(self.edit_project_btn)
        self.populate_project_combo()

        row.addWidget(QLabel("Session:"))
        self.session_combo = AddableDropdown('session', self.add_session, width=180)
        self.populate_session_combo()
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)
        row.addWidget(self.session_combo)

        row.addWidget(QLabel("Type:"))
        self.type_combo = AddableDropdown('type', width=120)   # fixed enum -> no Add row
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        row.addWidget(self.type_combo)

        row.addWidget(QLabel("Labels:"))
        self.complete_chip = chip_button("incomplete")
        self.complete_chip.setToolTip(
            "Every pair here has been reviewed, so untagged pairs count as\n"
            "negatives when training the classifier.")
        self.complete_chip.setStyleSheet(
            "QPushButton { border: 1px solid #aaa; border-radius: 3px; padding: 1px 6px; "
            "background: #f0b0b0; }"
            "QPushButton:checked { background: #90EE90; border-color: #4a8; }")
        self.complete_chip.toggled.connect(
            lambda on: self.complete_chip.setText("complete" if on else "incomplete"))
        self.complete_chip.clicked.connect(self._on_complete_chip)
        row.addWidget(self.complete_chip)

        win._project_combo = self.project_combo
        win._session_combo = self.session_combo
        win._type_combo = self.type_combo

        row.addStretch()

    def attach(self, root_layout):
        root_layout.addWidget(self.widget)
        self.sync()

    def populate_session_combo(self):
        """Fill the session list from the current project's dataset (call on project switch)."""
        w = self._win
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItem("All sessions", _ALL_SESSION_MARKER)
        self.session_combo.insertSeparator(1)
        for nk in w.nav.real_nd_keys():
            self.session_combo.addItem(w.nav.session_label(nk), nk)
        self.session_combo.append_add_row()
        self.session_combo.blockSignals(False)

    def sync(self):
        """Align session/type combos with nav.key (nav is canonical context)."""
        w = self._win
        nd_key = w.nav.key.nd()
        target_sess = str(getattr(nd_key, 'session', nd_key))
        self.session_combo.blockSignals(True)
        if w.nav.session_any_mode:
            self.session_combo.setCurrentIndex(0)
        else:
            for i in range(self.session_combo.count()):
                nk = self.session_combo.itemData(i)
                if nk is None or nk is _ALL_SESSION_MARKER:
                    continue
                if nk == nd_key or str(getattr(nk, 'session', nk)) == target_sess:
                    self.session_combo.setCurrentIndex(i)
                    break
        self.session_combo.blockSignals(False)
        self.refresh_type_combo()

    def _current_session_nd_key(self):
        if self.session_combo.count():
            idx = self.session_combo.currentIndex()
            if idx >= 0:
                nk = self.session_combo.itemData(idx)
                if nk is not None and nk is not _ALL_SESSION_MARKER:
                    return nk
        return self._win.nav.key.nd()

    def _type_key_at_index(self, idx: int):
        w = self._win
        labeled = getattr(getattr(w.cd, 'conf', None), 'conn_types_labeled', None) or []
        if idx < 0 or idx >= len(labeled):
            return None
        ei, conn_type = labeled[idx]
        nk = self._current_session_nd_key()
        return Key(session=nk.session, excitability=ei, conn_type=conn_type)

    def refresh_type_combo(self):
        w = self._win
        nd_key = self._current_session_nd_key()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for ei, conn_type in w.cd.conf.conn_types_labeled:
            tk = Key(session=nd_key.session, excitability=ei, conn_type=conn_type)
            ref, tgt = conn_type
            _m = {'pyr': 'PYR', 'inter': 'INT'}
            label = f"{ei}: {_m.get(ref, ref)}→{_m.get(tgt, tgt)}"
            self.type_combo.addItem(label, tk)
        cur_lbl = w.nav.key.type_label() if w.nav.key else None
        for i in range(self.type_combo.count()):
            cand = self.type_combo.itemData(i)
            if cand == w.nav.key or (cur_lbl and cand.type_label() == cur_lbl):
                self.type_combo.setCurrentIndex(i)
                break
        self.type_combo.blockSignals(False)
        self.complete_chip.setChecked(w.nav.active_selections.complete)

    def _on_complete_chip(self):
        self._win.nav.active_selections.complete = self.complete_chip.isChecked()

    @staticmethod
    def _project_header(project_dir: str) -> 'ProjectConfig':
        """The project's header; a draft one has never been built."""
        header = ProjectConfig(name=project_dir[len('project_'):])
        if os.path.isfile(header.save_path() + '.json'):
            header.load()
        return header

    def populate_project_combo(self):
        w = self._win
        projects = [f'project_{n}' for n in projects_on_disk(str(w.nav.cd.data_root))]
        current = getattr(w, '_project_dir', None)
        if current and current not in projects:
            projects = [current] + projects
        headers = {p: self._project_header(p) for p in projects}
        labels = [p if headers[p].built else f"{p}  (draft)" for p in projects]
        self.project_combo.blockSignals(True)
        self.project_combo.set_items(labels, data=projects)
        if current:
            idx = self.project_combo.findData(current)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
        self.project_combo.blockSignals(False)
        self.edit_project_btn.setEnabled(bool(current and headers[current].source))

    def _on_session_changed(self, idx):
        w = self._win
        nk = self.session_combo.itemData(idx)
        if nk is None:
            return
        if nk is _ALL_SESSION_MARKER:
            w._enter_all_session_mode()
            return
        if w.nav.session_any_mode:
            w._exit_all_session_mode()
        prev_lbl = w.nav.key.type_label()
        self.refresh_type_combo()
        tk = None
        for i in range(self.type_combo.count()):
            cand = self.type_combo.itemData(i)
            if cand is not None and cand.type_label() == prev_lbl:
                tk = cand
                break
        if tk is None:
            for ei, ct in w.cd.conf.conn_types_labeled:
                cand = Key(session=nk.session, excitability=ei, conn_type=ct)
                if cand.type_label() == prev_lbl:
                    tk = cand
                    break
        if tk is None and self.type_combo.count():
            tk = self.type_combo.itemData(0)
        if tk is not None:
            w._ensure_loaded(tk.nd(), 'lowres', lambda: w._switch_session(tk))

    def _on_type_changed(self, idx):
        w = self._win
        tk = self._type_key_at_index(idx)
        if tk is not None:
            lbl = tk.type_label()
            print(f"[CCGReviewUI] conn type → {lbl} "
                  f"(session={getattr(tk, 'session', '')})", flush=True)
            w._ensure_loaded(tk.nd(), 'lowres', lambda: w._switch_session(tk))

    def _on_project_changed(self, index: int):
        """Switch to the picked project; a draft one reopens the builder instead."""
        w = self._win
        project_dir = self.project_combo.itemData(index)   # the Add row carries no name
        if not isinstance(project_dir, str) or project_dir == getattr(w, '_project_dir', None):
            return
        if self._project_header(project_dir).built:
            w._switch_project(project_dir)
        else:
            self.add_project(self._project_header(project_dir))

    def add_project(self, draft: 'ProjectConfig' = None):
        """Build a project via AddProjectDialog — from scratch, or resuming a saved draft."""
        w = self._win
        spec = AddProjectDialog.show(w.nav, parent=w, draft=draft)
        if spec is None:
            return
        header, ccg_conf, compute = spec
        editing = draft is not None and draft.built
        verb, gerund = ("Update", "Updating") if editing else ("Build", "Building")

        thread = QThread(w)
        worker = _BuildProjectWorker(header, ccg_conf, compute)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self._build_thread, self._build_worker = thread, worker
        w.status_bar.start_task(f"{gerund} {header.name}…")

        def _teardown():
            w.status_bar.end_task()
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            self._build_thread = self._build_worker = None

        def _on_done(built):
            _teardown()
            w._adopt_project(built[1])   # sets _project_dir from the new cd
            self.populate_project_combo()

        def _on_error(msg: str):
            _teardown()
            QMessageBox.critical(w, f"{verb} project",
                                 f"Could not {verb.lower()} '{header.name}':\n{msg}")

        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        thread.start()

    def edit_project(self):
        """Reopen the builder on the current project — rescans the source, remaps, rebuilds."""
        header = self._project_header(self._win._project_dir)
        if not header.source:
            QMessageBox.information(
                self._win, "Edit project",
                f"'{header.name}' is a {header.format} project: it has no source folder "
                "to rescan, so its fields cannot be remapped here.")
            return
        self.add_project(header)

    def add_session(self):
        """Add a session to the current project's dataset and to the combo."""
        pass


class ReviewMenuBar:
    """Main window menu bar."""

    def __init__(self, win: CCGReviewUI):
        self._win = win
        self.panel_actions: dict[str, QAction] = {}

    def build(self):
        w = self._win
        mb = w.menuBar()

        panels_menu = mb.addMenu("Panels")
        for name, attr in [
            ("Pair Selection",  'pairs_view'),
            ("Main View (CCG)", 'mainview'),
            ("Neuron Network",  'neuron_network'),
            ("Waveforms",       '_waveforms_panel'),
            ("Time Slider",     'time_slider'),
            ("Group Hotkeys",   'hotkeys_bar'),
        ]:
            act = panels_menu.addAction(name, lambda a=attr: w._toggle_panel(a))
            act.setCheckable(True)
            act.setChecked(attr != '_waveforms_panel')
            self.panel_actions[attr] = act

        groups_menu = mb.addMenu("Groups")
        groups_menu.addAction("Create group…", lambda: CreateGroupDialog.show(w.nav.sel_data, w.pairs_view.pair_selection, w))
        groups_menu.addAction("Manage groups…", w._manage_groups)
        groups_menu.addSeparator()
        groups_menu.addAction("Export groups…", lambda: w.nav.groups.save())
        groups_menu.addAction("Import groups…", lambda: w.nav.groups.load())

        sel_menu = mb.addMenu("Selections")
        sel_menu.addAction("Save (Ctrl+S)", lambda: QuickSaveDialog.show(w.pairs_view, w))
        sel_menu.addAction("Load…", lambda: LoadSelectionDialog.show(w.pairs_view, w.nav.cd.selections_dir, w))
        sel_menu.addSeparator()
        sel_menu.addAction("Export PNGs…", lambda: ExportOptionsDialog.show(w.nav))
        sel_menu.addSeparator()
        panel = w.pairs_view.pair_selection
        sel_menu.addAction("Bookmark current pair (Ctrl+B)",
                           panel._bookmark_toggle_current)
        sel_menu.addAction("Clear bookmarks", panel._clear_bookmarks)

        mod_menu = mb.addMenu("Modules")
        stats_menu = mod_menu.addMenu("Stats tests")
        stats_menu.addAction("Run stats test…", w._show_stats_panel)
        jitter_menu = mod_menu.addMenu("Jitter")
        jitter_menu.addAction("View queue…", lambda: JitterQueueDialog(w.jitter_mgr, w).exec())
        jitter_menu.addAction("Clear queue", w.jitter_mgr.clear_queue)
        mod_menu.addSeparator()
        classify_menu = mod_menu.addMenu("Classify")
        classify_menu.addAction("Run classifier", w._run_classifier)

        settings_menu = mb.addMenu("Settings")
        settings_menu.addAction("Settings…", lambda: SettingsDialog.show(w, parent=w))

        w._panel_actions = self.panel_actions
