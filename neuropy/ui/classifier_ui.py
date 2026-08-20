"""Modules ▸ Classify — train on the project's tags, then review the predictions.

Predictions never write into the group tags on their own. They land in a
``PredictionStore`` the user reviews pair by pair, so the labels that train the
next model stay the user's own.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyqtgraph.Qt.QtCore import QObject, Qt, QThread, Signal
from pyqtgraph.Qt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from neuropy.classifier.dataset import is_shape_label
from neuropy.classifier.models import MODELS, UNSURE
from neuropy.ui.ui_common import SelectionCommand
from neuropy.ui.utils import BusyButton, ListPickerButton, TagChip
from neuropy.classifier.predictions import (PredictionStore, prior_admissions,
                                            store_from_rows)
from neuropy.classifier.run import (DEFAULT_MODEL, apply_model, delete_model,
                                    list_models, missing_highres, open_projects,
                                    predict_project, rename_model, scope_keys,
                                    scorable_keys, train_project)

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

# Shapes that differ only in degree are easier to separate from each other than
# from all thirteen labels at once, so they train as one family. Labels absent
# from a project are dropped by the support filter, so a family costs nothing.
LABEL_FAMILIES = {
    'quality (best/good/ok)': ['best', 'good', 'ok', 'bad'],
    'fast patterns': ['msconn', '2peakms', 'wideMs', 'rift', 'triple', 'leak'],
    'rhythm': ['rhythm', '0rhythm', 'burst', 'bimodal'],
    'inhibition': ['ppinhib', 'inhib2sides', 'Disinhib', 'I-I-flip'],
}


class _ClassifyWorker(QObject):
    """Trains a new classifier, or applies a saved one, off the Qt main thread."""
    done = Signal(object)
    error = Signal(str)

    def __init__(self, cd, opts: dict):
        super().__init__()
        self._cd = cd
        self._opts = opts
        self._saved = opts.get('saved')

    def run(self):
        try:
            result = (self._apply() if self._saved else self._train())
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.done.emit(result)

    def _train(self) -> dict:
        o = self._opts
        result = train_project(self._cd, o['model_name'], figures=o['figures'],
                               save_as=o['save_as'], extra=o['extra'],
                               highres=o['highres'], min_count=o['min_count'],
                               bias=o['bias'], only_labels=o['only_labels'])
        keys, skipped = scorable_keys(self._cd, result['model'], o['scope'])
        rows = predict_project(self._cd, result['model'], keys)
        result['skipped'] = skipped
        result['store'] = store_from_rows(rows, result['saved_as'],
                                          result['model'].label_names)
        return result

    def _apply(self) -> dict:
        out = apply_model(self._cd, self._saved, self._opts['scope'])
        trained = out['meta'].get('trained_on', {})
        return {'model': out['model'], 'figures': [], 'out_dir': '',
                'saved_as': self._saved, 'skipped': out['skipped'],
                'store': store_from_rows(out['rows'], self._saved,
                                         out['model'].label_names),
                # An applied model reports the scores it earned when trained;
                # this project has no labels to re-score it against.
                'summary': {'model': self._saved, 'scores': {},
                            'n_samples': trained.get('n_samples', 0),
                            'n_rats': len(trained.get('rats', [])),
                            'mean_f1': trained.get('mean_f1', 0.0),
                            'mean_auc': trained.get('mean_auc', 0.0),
                            'trained_on': trained}}


class _AcceptChip(TagChip):
    """One predicted label, clicked to tag the pair with just that label.

    A prediction is often partly right, so each label is its own click target;
    the extra "all" chip takes the whole row when every label is correct.
    """
    clicked = Signal(str)

    def __init__(self, text: str, color: str, label: str):
        super().__init__(text, color)
        self._label = label
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self._label)


class ClassifierDialog(QDialog):
    """Pick a model, train it, then read the per-label scores it achieved."""

    def __init__(self, win: 'CCGReviewUI'):
        super().__init__(win)
        self._win = win
        self._thread = self._worker = None
        self._rows: list[tuple] = []
        self.setWindowTitle("Classify pairs")
        self.resize(620, 520)
        # Non-modal throughout: judging a candidate pair means toggling
        # resolution and panels in the main window while this stays open.
        self.setModal(False)
        self._build()

    def _build(self):
        # Training above, review below: two jobs that compete for height, so the
        # sash lets whichever one is in use take the space.
        outer = QVBoxLayout(self)
        self._splitter = QSplitter(Qt.Vertical)
        outer.addWidget(self._splitter)
        train_box, lay = QWidget(), QVBoxLayout()
        train_box.setLayout(lay)
        self._splitter.addWidget(train_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for name in MODELS:
            self.model_combo.addItem(name)
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        row.addWidget(self.model_combo)
        self.bias_combo = QComboBox()
        for mode, tip in (('discover', "find as many true pairs as possible"),
                          ('balanced', "even trade of coverage against errors"),
                          ('accurate', "only pairs it is confident about")):
            self.bias_combo.addItem(mode)
            self.bias_combo.setItemData(self.bias_combo.count() - 1, tip, Qt.ToolTipRole)
        self.bias_combo.setCurrentText('balanced')
        row.addWidget(self.bias_combo)
        row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(self._win.cd.conf.name)   # default: this project
        self.name_edit.setMinimumWidth(140)
        row.addWidget(self.name_edit)
        self.highres_check = QCheckBox("Use high-res")
        self.highres_check.setChecked(True)
        self.highres_check.setToolTip(
            "Adds the fine-binned CCGs as extra features. Sessions missing them "
            "are computed first, since a model trained on both resolutions "
            "cannot score a low-res-only session.")
        row.addWidget(self.highres_check)
        self.figures_check = QCheckBox("Figures")
        self.figures_check.setChecked(True)
        row.addWidget(self.figures_check)
        row.addStretch()
        lay.addLayout(row)

        # Scope narrows which pairs get scored. Every picker starts with all of
        # its options selected, so out of the box the whole loaded project is used.
        nav = self._win.nav
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.session_picker = ListPickerButton(
            "Session", nav.available_sessions(), plural="sessions",
            refresh_provider=nav.available_sessions)
        scope_row.addWidget(self.session_picker)
        self.type_picker = ListPickerButton(
            "ConnType", nav.available_conn_types(), plural="types",
            refresh_provider=nav.available_conn_types)
        scope_row.addWidget(self.type_picker)
        scope_row.addWidget(QLabel("Also train on:"))
        self.extra_picker = ListPickerButton(
            "Projects", self._other_projects(), plural="projects",
            select_all_when_empty=False)
        scope_row.addWidget(self.extra_picker)
        scope_row.addWidget(QLabel("Labels:"))
        self.label_picker = ListPickerButton(
            "Labels", self._shape_labels(), plural="labels",
            refresh_provider=self._shape_labels)
        scope_row.addWidget(self.label_picker)
        # Training one family at a time is easier than all shapes at once, so the
        # families are presets that fill the picker, which stays freely editable.
        self.family_combo = QComboBox()
        self.family_combo.addItem("all labels", None)
        for name, labels in LABEL_FAMILIES.items():
            self.family_combo.addItem(name, list(labels))
        self.family_combo.currentIndexChanged.connect(self._on_family_combo)
        scope_row.addWidget(self.family_combo)
        scope_row.addStretch()
        lay.addLayout(scope_row)

        self.train_btn = BusyButton("Train and classify")
        self.train_btn.clicked.connect(self._on_train_btn)
        lay.addWidget(self.train_btn)

        # Applying a model trained elsewhere is the cross-project path: one
        # picker and one button, sharing the training run's worker and handlers.
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("Saved:"))
        self.saved_combo = QComboBox()
        self.saved_combo.setMinimumWidth(240)
        saved_row.addWidget(self.saved_combo)
        self.apply_btn = BusyButton("Apply to scope")
        self.apply_btn.clicked.connect(self._on_apply_btn)
        saved_row.addWidget(self.apply_btn)
        manage_btn = QPushButton("Manage…")
        manage_btn.clicked.connect(self._on_manage_btn)
        saved_row.addWidget(manage_btn)
        saved_row.addStretch()
        lay.addLayout(saved_row)

        self.status_label = QLabel(self._describe_existing())
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)

        lay.addStretch()

        # Review lives here rather than in a second dialog: the scores are already
        # in the status line, so the space belongs to the pairs being judged.
        review_box, lay = QWidget(), QVBoxLayout()
        review_box.setLayout(lay)
        self._splitter.addWidget(review_box)
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.currentIndexChanged.connect(self._reload)
        filt.addWidget(self.label_combo)
        self.session_check = QCheckBox("Current session only")
        self.session_check.setChecked(True)
        self.session_check.stateChanged.connect(self._reload)
        filt.addWidget(self.session_check)
        filt.addWidget(QLabel("Min confidence:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.0, 1.0)
        self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.valueChanged.connect(self._reload)
        filt.addWidget(self.cutoff_spin)
        filt.addStretch()
        lay.addLayout(filt)

        # Predicted/accepted are the two halves of an available/selected editor:
        # a chip click moves one label across, and the row stays until it is empty.
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ['pair', 'predicted', 'confidence', 'accepted'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        lay.addWidget(self.table, stretch=1)

        act = QHBoxLayout()
        self.accept_btn = QPushButton("Accept (tag pair)")
        self.accept_btn.clicked.connect(self._on_accept_btn)
        act.addWidget(self.accept_btn)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self._on_reject_btn)
        act.addWidget(self.reject_btn)
        self.accept_all_btn = QPushButton("Accept all shown")
        self.accept_all_btn.setToolTip(
            "Tag every row currently listed — narrow the list with the label\n"
            "filter and the confidence cutoff first.")
        self.accept_all_btn.clicked.connect(self._on_accept_all_btn)
        act.addWidget(self.accept_all_btn)
        act.addStretch()
        lay.addLayout(act)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        self._splitter.setStretchFactor(1, 1)   # review pane absorbs resizes
        self.refresh_saved()
        self._reload()

    @property
    def _store(self) -> PredictionStore | None:
        """The live store: predictions belong to the window, not this dialog."""
        return self._win.prediction_store

    def _describe_existing(self) -> str:
        store = self._store
        if store is None:
            return ("Trains on this project's group tags, then labels every pointer pair.\n"
                    "Predictions stay separate from your tags until you accept them.")
        return f"{len(store)} pairs already classified by '{store.model_name}'."

    def _other_projects(self) -> list[str]:
        """Sibling projects whose selections could be pooled into training."""
        root = os.path.dirname(str(self._win.cd.save_path))
        here = self._win.cd.conf.name
        return sorted(d[len('project_'):] for d in os.listdir(root)
                      if d.startswith('project_') and d[len('project_'):] != here)

    def _shape_labels(self) -> list[str]:
        """Group tags that name a CCG shape — the ones a model can be trained on."""
        return [g for g in self._win.nav.groups.groups if is_shape_label(g)]

    def _on_family_combo(self):
        """A family is a preset: it fills the label picker, which stays editable."""
        family = self.family_combo.currentData()
        self.label_picker.set_selected(family if family else self._shape_labels())

    def _options(self, saved: str | None) -> dict:
        """Everything the worker needs, read once off the widgets."""
        return {'saved': saved,
                'model_name': self.model_combo.currentText(),
                'bias': self.bias_combo.currentText(),
                'save_as': self.name_edit.text().strip() or self._win.cd.conf.name,
                'figures': self.figures_check.isChecked(),
                'highres': self.highres_check.isChecked(),
                'extra': open_projects(self.extra_picker.selected),
                'min_count': int(self._win.settings.classifier_min_count),
                'only_labels': self.label_picker.selected,
                'scope': scope_keys(self._win.cd, self.session_picker.selected,
                                    self.type_picker.selected)}

    def _on_apply_btn(self):
        self._launch(self.saved_combo.currentData())

    def _on_train_btn(self):
        self._launch(None)

    def _launch(self, saved: str | None):
        opts = self._options(saved)
        if saved and not self._confirm_reapply(saved, opts['scope']):
            return
        missing = missing_highres(self._win.cd, opts['scope']) if opts['highres'] else []
        if missing and not self._confirm_highres(missing):
            return
        if missing:
            self._precompute(missing, opts)
            return
        self._start(opts)

    def _confirm_reapply(self, saved: str, scope: list) -> bool:
        """Warn when *saved* has already had admissions accepted in this scope.

        Re-running is allowed — a model re-scores pairs whose tags have changed
        since — but its own past suggestions will come back as predictions, so
        say how many were already judged rather than letting them look new.
        """
        n = prior_admissions(self._win.nav.groups, saved,
                             {str(k.session) for k in scope})
        if not n:
            return True
        return QMessageBox.question(
            self, "Already applied",
            f"'{saved}' has {n} accepted pair(s) in this scope from an earlier run.\n\n"
            f"Applying it again re-proposes them alongside anything new. "
            f"Accepted labels are kept either way. Proceed?"
        ) == QMessageBox.StandardButton.Yes

    def _confirm_highres(self, sessions: list[str]) -> bool:
        """High-res compute is minutes and gigabytes — never start it unasked."""
        shown = ', '.join(sessions[:8]) + ('…' if len(sessions) > 8 else '')
        return QMessageBox.question(
            self, "Compute high-res CCGs",
            f"{len(sessions)} session(s) in scope have no high-res CCGs:\n\n{shown}\n\n"
            f"A classifier trained on both resolutions cannot score a low-res-only "
            f"session, so these must be computed first. Proceed?"
        ) == QMessageBox.StandardButton.Yes

    def _precompute(self, sessions: list[str], opts: dict):
        """Queue the missing high-res computes, then train when they all land."""
        self.train_btn.set_busy(True, f"Computing high-res 0/{len(sessions)}")
        self.apply_btn.setEnabled(False)
        started = self._win.custom_mgr.queue_whole_session(
            sessions, 'highres', on_done=lambda ok: self._on_precomputed(ok, opts))
        if started != len(sessions):
            self.train_btn.set_busy(False)
            self.apply_btn.setEnabled(self.saved_combo.count() > 0)
            QMessageBox.warning(self, "Classify",
                                f"Only {started} of {len(sessions)} session(s) could be "
                                f"queued — the CCG queue is full. Let it drain and retry.")

    def _on_precomputed(self, failed: list[str], opts: dict):
        self.train_btn.set_busy(False)
        self.apply_btn.setEnabled(self.saved_combo.count() > 0)
        if failed:
            # Refuse rather than train on a subset: which sessions are missing
            # changes what the model saw, so that is the user's call.
            QMessageBox.critical(self, "Classify",
                                 f"High-res compute failed for {len(failed)} session(s):\n"
                                 f"{', '.join(failed)}\n\nNothing was trained.")
            return
        self._start(opts)

    def _start(self, opts: dict):
        saved = opts.get('saved')
        busy = self.apply_btn if saved else self.train_btn
        busy.set_busy(True, "Applying classifier…" if saved
                      else "Training classifier… (~2 min)")
        (self.train_btn if saved else self.apply_btn).setEnabled(False)

        thread = QThread(self)
        worker = _ClassifyWorker(self._win.cd, opts)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self._thread, self._worker = thread, worker

        def _teardown():
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            self._thread = self._worker = None
            self.train_btn.set_busy(False)
            self.apply_btn.set_busy(False)
            self.apply_btn.setEnabled(self.saved_combo.count() > 0)

        def _on_done(result):
            _teardown()
            self._show_result(result)

        def _on_error(msg: str):
            _teardown()
            QMessageBox.critical(self, "Classify", msg)

        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        thread.start()

    def _show_result(self, result: dict):
        summary = result['summary']
        self._win.prediction_store = result['store']
        trained = summary.get('trained_on', {})
        origin = (f"trained on {trained['project']}" if trained
                  else f"saved as '{result['saved_as']}'")
        lines = [f"{len(result['store'])} pairs classified "
                 f"({summary['n_samples']} training pairs / "
                 f"{summary['n_rats']} animals, {origin}).",
                 f"Leave-one-animal-out mean F1 {summary['mean_f1']:.2f}, "
                 f"AUC {summary['mean_auc']:.2f}."]
        best = sorted(summary['scores'].items(), key=lambda kv: -kv[1]['f1'])[:4]
        if best:
            lines.append("  ".join(f"{n} p{s['precision']:.2f}/r{s['recall']:.2f}"
                                   for n, s in best))
        if result.get('skipped'):
            lines.append(f"Skipped {len(result['skipped'])} session(s) lacking the "
                         f"CCGs this model needs: {', '.join(result['skipped'][:4])}"
                         + ("…" if len(result['skipped']) > 4 else ""))
        if result['out_dir']:
            lines.append(f"Figures and model in {result['out_dir']}")
            result['store'].save(os.path.join(result['out_dir'], 'predictions.json'))
        self.status_label.setText("\n".join(lines))
        self._reload()

    def _session(self) -> str | None:
        if not self.session_check.isChecked():
            return None
        return str(self._win.nav.key.session)

    def _visible_pairs(self) -> list[tuple]:
        """Rows the table shows: scope- and label-filtered, above the cutoff."""
        if self._store is None:
            return []
        label = self.label_combo.currentText()
        sess = self._session()
        one = None if label in ('all pending', UNSURE) else label
        pairs = (self._store.review_order(sess, self._win.nav.groups)
                 if label == 'all pending'
                 else self._store.pairs_for_label(label, sess))
        cutoff = self.cutoff_spin.value()
        return [pk for pk in pairs
                if self._store.confidence(*pk, label=one) >= cutoff]

    def _sync_labels(self):
        """Refill the label filter from the current store, keeping the choice."""
        want = ['all pending'] + (self._store.labels if self._store else []) + [UNSURE]
        if [self.label_combo.itemText(i)
                for i in range(self.label_combo.count())] == want:
            return
        keep = self.label_combo.currentText()
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItems(want)
        self.label_combo.setCurrentText(keep if keep in want else 'all pending')
        self.label_combo.blockSignals(False)

    def _reload(self):
        self._sync_labels()
        pairs = self._visible_pairs()
        label = self.label_combo.currentText()
        shown = None if label in ('all pending', UNSURE) else label
        self.table.setRowCount(len(pairs))
        self._rows = pairs
        self.accept_all_btn.setText(f"Accept all shown ({len(pairs)})")
        for i, pk in enumerate(pairs):
            pair = f"{pk[0]}  {pk[1]}→{pk[2]}"
            for col, val in ((0, pair + ('  ✗' if pk in self._store.rejected else '')),
                             (2, f"{self._store.confidence(*pk, label=shown):.2f}")):
                self.table.setItem(i, col, QTableWidgetItem(val))
            self.table.setCellWidget(i, 1, self._chip_cell(pk, taken=False))
            self.table.setCellWidget(i, 3, self._chip_cell(pk, taken=True))
        self.table.resizeRowsToContents()

    def _chip_cell(self, pk: tuple, taken: bool) -> QWidget:
        """One side of the row's editor: accepted labels when *taken*, else pending.

        Clicking a chip moves that label to the other side — accepting one label
        leaves the pair's other predictions on the table, still to be judged.
        """
        cell = QWidget()
        row = QHBoxLayout(cell)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(TagChip.GAP)
        groups = self._win.nav.groups
        labels = (self._store.taken(*pk, groups) if taken
                  else self._store.pending(*pk, groups))
        for label in labels + (['all'] if len(labels) > 1 and not taken else []):
            colour = ('' if label == 'all'
                      else groups.get_group_metadata(label).display_color)
            chip = _AcceptChip(label, colour, '' if label == 'all' else label)
            chip.clicked.connect(
                (lambda only, p=pk: self._unaccept_one(p, only)) if taken
                else (lambda only, p=pk: self._accept_one(p, only)))
            row.addWidget(chip)
        row.addStretch()
        return cell

    def _accept(self, pairs: list[tuple], only: str = None):
        """Accept across *pairs* as one undoable step, then refresh both chip sides."""
        added, promoted = [], {}
        for pk in pairs:
            new = self._store.accept(pk[0], pk[1], pk[2], self._win.nav.groups,
                                     only=only)
            if new:
                promoted.update(self._promote(pk))
            added += new
        if added:
            self._push_undo([(g, s, p, 'add') for g, s, p in added], promoted)
        self._after_change()

    def _promote(self, pk: tuple) -> dict:
        """Move an accepted pair into the selected list, as hand tagging does.

        Returns the state change for undo, keyed by pair — empty unless the pair
        was unselected in the key currently on screen, which is the only bucket
        SelectionCommand can restore.
        """
        nav = self._win.nav
        key = self._win.cd.find(pk[0], type_label=self._store.type_label_for(*pk),
                                strict=False)
        if key is None:
            return {}
        pair = (pk[1], pk[2])
        bucket = nav.sd.get_selection_by_session(key).selections[key]
        if pair not in bucket.unselected:
            return {}
        bucket.set_pair_state(pair, 'sel')
        return {pair: ('unsel', 'sel')} if key == nav.key else {}

    def _push_undo(self, group_changes: list, pair_changes: dict = None):
        """Record group edits on the pair panel's stack so Ctrl+Z reaches them.

        Accepting writes real tags — Accept all can write hundreds — so it belongs
        on the same stack as hand tagging rather than being irreversible.
        """
        self._win.pairs_view.pair_selection.push_undo(
            SelectionCommand(pair_changes or {}, group_changes))

    def _accept_one(self, pk: tuple, only: str):
        """Tag *pk* with one predicted label (or all of them when only is empty)."""
        self._accept([pk], only=only or None)

    def _unaccept_one(self, pk: tuple, label: str):
        """Send an accepted label back to pending, untagging the pair."""
        self._win.nav.groups.discard_from_group(label, pk[0], (pk[1], pk[2]))
        self._store.unaccept(pk[0], pk[1], pk[2], label)
        self._push_undo([(label, pk[0], (pk[1], pk[2]), 'remove')])
        self._after_change()

    def _selected(self) -> list[tuple]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return [self._rows[i] for i in sorted(rows) if i < len(self._rows)]

    def _on_table_selection(self):
        """Show the highlighted pair in the main CCG view so it can be judged.

        A prediction may belong to another session or conn type than the one on
        screen, so navigate there first — otherwise the pair index resolves
        against the wrong list and shows an unrelated CCG.
        """
        picked = self._selected()
        if len(picked) != 1:
            return
        sess, ref, tgt = picked[0]
        nav = self._win.nav
        want_type = self._store.type_label_for(sess, ref, tgt)
        if str(nav.key.session) != sess or (want_type
                                            and nav.key.type_label() != want_type):
            target = self._win.cd.find(sess, type_label=want_type, strict=False)
            if target is None:
                return
            self._win._ensure_loaded(target.nd(), 'lowres',
                                     lambda t=target: self._goto(t, ref, tgt))
            return
        self._goto(None, ref, tgt)

    def _goto(self, target, ref: int, tgt: int):
        """Land on (ref, tgt), switching session/type first when given a target."""
        nav = self._win.nav
        if target is not None:
            self._win._switch_session(target)
        nav.set_current_pair(nav.get_pair_index((ref, tgt)))

    def _on_accept_btn(self):
        self._accept(self._selected())

    def _on_reject_btn(self):
        for pk in self._selected():
            self._store.reject(*pk)
        self._after_change()

    def _on_accept_all_btn(self):
        pairs = self._visible_pairs()
        if QMessageBox.question(
                self, "Accept all",
                f"Tag {len(pairs)} pairs with their predicted labels?\n"
                "Ctrl+Z undoes this as one step.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._accept(pairs)

    def _after_change(self):
        self._win.nav.groups.changed.emit()
        self._reload()

    def _on_manage_btn(self):
        ManageModelsDialog(self._win, self).exec()

    def refresh_saved(self):
        """Repopulate the saved-model combo after the library changed on disk."""
        keep = self.saved_combo.currentData()
        self.saved_combo.clear()
        for entry in list_models(self._win.cd):
            trained = entry.get('trained_on', {})
            self.saved_combo.addItem(
                f"{entry['name']}  ({trained.get('n_samples', 0)} pairs from "
                f"{', '.join(trained.get('projects', ['?']))})", entry['name'])
        if keep:
            self.saved_combo.setCurrentIndex(max(0, self.saved_combo.findData(keep)))
        self.apply_btn.setEnabled(self.saved_combo.count() > 0)

    def refresh_scope(self):
        """Re-read every picker from the live project.

        The dialog outlives a project switch, and its options were read once at
        construction — stale session names match no key in the new project, so
        the scope comes out empty and nothing can be scored.
        """
        nav = self._win.nav
        for picker, provider in ((self.session_picker, nav.available_sessions),
                                 (self.type_picker, nav.available_conn_types),
                                 (self.label_picker, self._shape_labels),
                                 (self.extra_picker, self._other_projects)):
            picker.set_items(provider(), keep_selection=True)
        self.name_edit.setText(self._win.cd.conf.name)
        self.status_label.setText(self._describe_existing())
        self.refresh_saved()
        self._reload()

    @classmethod
    def show_for(cls, win: 'CCGReviewUI'):
        """Open (or resurface) the one classifier dialog for *win*."""
        if win.classifier_dialog is None:
            win.classifier_dialog = cls(win)
        else:
            win.classifier_dialog.refresh_scope()
        win.classifier_dialog.show()
        win.classifier_dialog.raise_()
        return win.classifier_dialog


class ManageModelsDialog(QDialog):
    """Rename or delete saved classifiers; the name is the unique identifier."""

    def __init__(self, win: 'CCGReviewUI', parent: ClassifierDialog):
        super().__init__(parent)
        self._win = win
        self._parent = parent
        self.setWindowTitle("Manage classifiers")
        self.resize(420, 420)
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_selected)
        lay.addWidget(self.list)
        self.stats_table = QTableWidget(0, 4)
        self.stats_table.setHorizontalHeaderLabels(['label', 'n', 'prec/rec', 'F1/AUC'])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.stats_table)
        row = QHBoxLayout()
        for text, slot in (("Rename…", self._on_rename_btn),
                           ("Delete", self._on_delete_btn)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch()
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._reload()

    def _reload(self):
        self.list.clear()
        for entry in list_models(self._win.cd):
            item = QListWidgetItem(entry['name'])
            item.setData(Qt.UserRole, entry.get('trained_on', {}).get('scores', {}))
            self.list.addItem(item)
        self._parent.refresh_saved()

    def _current(self) -> str:
        item = self.list.currentItem()
        return item.text() if item else ''

    def _on_selected(self, item: 'QListWidgetItem'):
        """Per-label CV scores for the selected model, recorded at training time."""
        scores = item.data(Qt.UserRole) if item else {}
        self.stats_table.setRowCount(len(scores))
        for i, (label, s) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]['f1'])):
            for col, val in enumerate([label, str(s['n']), f"{s['precision']:.2f}/{s['recall']:.2f}",
                                       f"{s['f1']:.2f}/{s['auc']:.2f}"]):
                self.stats_table.setItem(i, col, QTableWidgetItem(val))

    def _on_rename_btn(self):
        name = self._current()
        if not name:
            return
        new, ok = QInputDialog.getText(self, "Rename classifier", "New name:", text=name)
        if not ok:
            return
        try:
            rename_model(self._win.cd, name, new)
        except ValueError as exc:
            QMessageBox.warning(self, "Rename", str(exc))
            return
        self._reload()

    def _on_delete_btn(self):
        name = self._current()
        if not name or QMessageBox.question(
                self, "Delete classifier",
                f"Delete '{name}' permanently?") != QMessageBox.StandardButton.Yes:
            return
        delete_model(self._win.cd, name)
        self._reload()
