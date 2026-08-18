"""Modules ▸ Classify — train on the project's tags, then review the predictions.

Predictions never write into the group tags on their own. They land in a
``PredictionStore`` the user reviews pair by pair, so the labels that train the
next model stay the user's own.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyqtgraph.Qt.QtCore import QObject, QThread, Signal
from pyqtgraph.Qt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from neuropy.analyses.neurons_dataset import Key
from neuropy.classifier.models import MODELS, UNSURE
from neuropy.ui.utils import BusyButton
from neuropy.classifier.predictions import PredictionStore, store_from_rows
from neuropy.classifier.run import (DEFAULT_MODEL, apply_model, list_models,
                                    missing_highres, open_projects, predict_project,
                                    scope_keys, scorable_keys, train_project)

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI


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
                               highres=o['highres'], min_count=o['min_count'])
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


class ClassifierDialog(QDialog):
    """Pick a model, train it, then read the per-label scores it achieved."""

    def __init__(self, win: 'CCGReviewUI'):
        super().__init__(win)
        self._win = win
        self._thread = self._worker = None
        self._review = None
        self.setWindowTitle("Classify pairs")
        self.resize(620, 520)
        # Non-modal throughout: judging a candidate pair means toggling
        # resolution and panels in the main window while this stays open.
        self.setModal(False)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for name in MODELS:
            self.model_combo.addItem(name)
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        row.addWidget(self.model_combo)
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

        # Scope narrows which pairs get scored. Both dropdowns default to the
        # widest option, so out of the box the whole loaded project is used.
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.session_combo = QComboBox()
        self.session_combo.addItem("all sessions", None)
        for nk in self._win.nav.real_nd_keys():
            self.session_combo.addItem(str(nk.session), str(nk.session))
        scope_row.addWidget(self.session_combo)
        self.type_combo = QComboBox()
        self.type_combo.addItem("all types", None)
        for ei, conn in self._win.cd.conf.conn_types_labeled:
            label = Key(excitability=ei, conn_type=conn).type_label()
            self.type_combo.addItem(label, label)
        scope_row.addWidget(self.type_combo)
        scope_row.addWidget(QLabel("Also train on:"))
        self.extra_combo = QComboBox()
        self.extra_combo.addItem("this project only", [])
        for other in self._other_projects():
            self.extra_combo.addItem(f"+ {other}", [other])
        scope_row.addWidget(self.extra_combo)
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
        for entry in list_models(self._win.cd):
            trained = entry.get('trained_on', {})
            self.saved_combo.addItem(
                f"{entry['name']}  ({trained.get('n_samples', 0)} pairs from "
                f"{', '.join(trained.get('projects', ['?']))})", entry['name'])
        saved_row.addWidget(self.saved_combo)
        self.apply_btn = BusyButton("Apply to scope")
        self.apply_btn.clicked.connect(self._on_apply_btn)
        self.apply_btn.setEnabled(self.saved_combo.count() > 0)
        saved_row.addWidget(self.apply_btn)
        saved_row.addStretch()
        lay.addLayout(saved_row)

        self.status_label = QLabel(self._describe_existing())
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['label', 'n', 'precision', 'recall', 'F1'])
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)

        self.review_btn = QPushButton("Review predictions…")
        self.review_btn.clicked.connect(self._on_review_btn)
        self.review_btn.setEnabled(self._store() is not None)
        lay.addWidget(self.review_btn)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _store(self) -> PredictionStore | None:
        return self._win.prediction_store

    def _describe_existing(self) -> str:
        store = self._store()
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

    def _options(self, saved: str | None) -> dict:
        """Everything the worker needs, read once off the widgets."""
        return {'saved': saved,
                'model_name': self.model_combo.currentText(),
                'save_as': self.name_edit.text().strip() or self._win.cd.conf.name,
                'figures': self.figures_check.isChecked(),
                'highres': self.highres_check.isChecked(),
                'extra': open_projects(self.extra_combo.currentData() or []),
                'min_count': int(self._win.settings.classifier_min_count),
                'scope': scope_keys(self._win.cd,
                                    self.session_combo.currentData(),
                                    self.type_combo.currentData())}

    def _on_apply_btn(self):
        self._launch(self.saved_combo.currentData())

    def _on_train_btn(self):
        self._launch(None)

    def _launch(self, saved: str | None):
        opts = self._options(saved)
        missing = missing_highres(self._win.cd, opts['scope']) if opts['highres'] else []
        if missing and not self._confirm_highres(missing):
            return
        if missing:
            self._precompute(missing, opts)
            return
        self._start(opts)

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
        scores = summary['scores']
        self.table.setRowCount(len(scores))
        order = sorted(scores, key=lambda n: -scores[n]['f1'])
        for i, name in enumerate(order):
            s = scores[name]
            for col, val in enumerate([name, str(s['n']), f"{s['precision']:.2f}",
                                       f"{s['recall']:.2f}", f"{s['f1']:.2f}"]):
                self.table.setItem(i, col, QTableWidgetItem(val))
        trained = summary.get('trained_on', {})
        origin = (f"trained on {trained['project']}" if trained
                  else f"saved as '{result['saved_as']}'")
        lines = [f"{len(result['store'])} pairs classified "
                 f"({summary['n_samples']} training pairs / "
                 f"{summary['n_rats']} animals, {origin}).",
                 f"Leave-one-animal-out mean F1 {summary['mean_f1']:.2f}, "
                 f"AUC {summary['mean_auc']:.2f}."]
        if result.get('skipped'):
            lines.append(f"Skipped {len(result['skipped'])} session(s) lacking the "
                         f"CCGs this model needs: {', '.join(result['skipped'][:4])}"
                         + ("…" if len(result['skipped']) > 4 else ""))
        if result['out_dir']:
            lines.append(f"Figures and model in {result['out_dir']}")
            result['store'].save(os.path.join(result['out_dir'], 'predictions.json'))
        self.status_label.setText("\n".join(lines))
        self.review_btn.setEnabled(True)

    def _on_review_btn(self):
        store = self._store()
        if store is None:
            return
        if self._review is None:
            self._review = PredictionReviewDialog(self._win, store)
        self._review.show()
        self._review.raise_()

    @classmethod
    def show_for(cls, win: 'CCGReviewUI'):
        """Open (or resurface) the one classifier dialog for *win*."""
        if win.classifier_dialog is None:
            win.classifier_dialog = cls(win)
        win.classifier_dialog.show()
        win.classifier_dialog.raise_()
        return win.classifier_dialog


class PredictionReviewDialog(QDialog):
    """Walk the predictions and accept the ones that are right.

    Ordered least-confident first: the confident predictions need no attention,
    so review time is spent where the model is actually unsure.
    """

    def __init__(self, win: 'CCGReviewUI', store: PredictionStore):
        super().__init__(win)
        self._win = win
        self._store = store
        self.setWindowTitle("Review predictions")
        self.resize(660, 500)
        self.setModal(False)
        self._build()
        self._reload()

    def _build(self):
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.addItem("all pending")
        for name in self._store.labels:
            self.label_combo.addItem(name)
        self.label_combo.addItem(UNSURE)
        self.label_combo.currentIndexChanged.connect(self._reload)
        row.addWidget(self.label_combo)
        self.session_check = QCheckBox("Current session only")
        self.session_check.setChecked(True)
        self.session_check.stateChanged.connect(self._reload)
        row.addWidget(self.session_check)
        row.addWidget(QLabel("Min confidence:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.0, 1.0)
        self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.valueChanged.connect(self._reload)
        row.addWidget(self.cutoff_spin)
        row.addStretch()
        lay.addLayout(row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['pair', 'predicted', 'confidence', 'state'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        lay.addWidget(self.table)

        row2 = QHBoxLayout()
        self.accept_btn = QPushButton("Accept (tag pair)")
        self.accept_btn.clicked.connect(self._on_accept_btn)
        row2.addWidget(self.accept_btn)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self._on_reject_btn)
        row2.addWidget(self.reject_btn)
        self.accept_all_btn = QPushButton("Accept all shown")
        self.accept_all_btn.setToolTip(
            "Tag every row currently listed — narrow the list with the label\n"
            "filter and the confidence cutoff first.")
        self.accept_all_btn.clicked.connect(self._on_accept_all_btn)
        row2.addWidget(self.accept_all_btn)
        row2.addStretch()
        lay.addLayout(row2)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _session(self) -> str | None:
        if not self.session_check.isChecked():
            return None
        return str(self._win.nav.key.session)

    def _visible_pairs(self) -> list[tuple]:
        """Rows the table shows: scope- and label-filtered, above the cutoff."""
        label = self.label_combo.currentText()
        sess = self._session()
        one = None if label in ('all pending', UNSURE) else label
        pairs = (self._store.review_order(sess) if label == 'all pending'
                 else self._store.pairs_for_label(label, sess))
        cutoff = self.cutoff_spin.value()
        return [pk for pk in pairs
                if self._store.confidence(*pk, label=one) >= cutoff]

    def _reload(self):
        pairs = self._visible_pairs()
        label = self.label_combo.currentText()
        shown = None if label in ('all pending', UNSURE) else label
        self.table.setRowCount(len(pairs))
        self._rows = pairs
        self.accept_all_btn.setText(f"Accept all shown ({len(pairs)})")
        for i, pk in enumerate(pairs):
            state = ('accepted' if pk in self._store.accepted else
                     'rejected' if pk in self._store.rejected else '')
            cells = [f"{pk[0]}  {pk[1]}→{pk[2]}",
                     ', '.join(self._store.labels_for(*pk)),
                     f"{self._store.confidence(*pk, label=shown):.2f}", state]
            for col, val in enumerate(cells):
                self.table.setItem(i, col, QTableWidgetItem(val))

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
        for pk in self._selected():
            self._store.accept(pk[0], pk[1], pk[2], self._win.nav.groups)
        self._after_change()

    def _on_reject_btn(self):
        for pk in self._selected():
            self._store.reject(*pk)
        self._after_change()

    def _on_accept_all_btn(self):
        pairs = self._visible_pairs()
        if QMessageBox.question(
                self, "Accept all",
                f"Tag {len(pairs)} pairs with their predicted labels?") \
                != QMessageBox.StandardButton.Yes:
            return
        for pk in pairs:
            self._store.accept(pk[0], pk[1], pk[2], self._win.nav.groups)
        self._after_change()

    def _after_change(self):
        self._win.nav.groups.changed.emit()
        self._reload()
