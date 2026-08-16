"""Modules ▸ Classify — train on the project's tags, then review the predictions.

Predictions never write into the group tags on their own. They land in a
``PredictionStore`` the user reviews pair by pair, so the labels that train the
next model stay the user's own.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyqtgraph.Qt.QtCore import Qt, QObject, QThread, Signal
from pyqtgraph.Qt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QMessageBox, QProgressDialog, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from neuropy.classifier.models import MODELS, UNSURE
from neuropy.classifier.predictions import PredictionStore, store_from_rows
from neuropy.classifier.run import DEFAULT_MODEL, predict_project, train_project

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI


class _TrainWorker(QObject):
    """Runs training + prediction off the Qt main thread."""
    done = Signal(object)
    error = Signal(str)

    def __init__(self, cd, model_name: str, figures: bool):
        super().__init__()
        self._cd = cd
        self._model_name = model_name
        self._figures = figures

    def run(self):
        try:
            result = train_project(self._cd, self._model_name, figures=self._figures)
            rows = predict_project(self._cd, result['model'])
            result['store'] = store_from_rows(rows, self._model_name,
                                              result['model'].label_names)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.done.emit(result)


class ClassifierDialog(QDialog):
    """Pick a model, train it, then read the per-label scores it achieved."""

    def __init__(self, win: 'CCGReviewUI'):
        super().__init__(win)
        self._win = win
        self._thread = self._worker = None
        self.setWindowTitle("Classify pairs")
        self.resize(560, 460)
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
        self.figures_check = QCheckBox("Write verification figures")
        self.figures_check.setChecked(True)
        row.addWidget(self.figures_check)
        row.addStretch()
        lay.addLayout(row)

        self.train_btn = QPushButton("Train and classify")
        self.train_btn.clicked.connect(self._on_train_btn)
        lay.addWidget(self.train_btn)

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

    def _on_train_btn(self):
        self.train_btn.setEnabled(False)
        dlg = QProgressDialog("Training classifier…", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.show()

        thread = QThread(self)
        worker = _TrainWorker(self._win.cd, self.model_combo.currentText(),
                              self.figures_check.isChecked())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self._thread, self._worker = thread, worker

        def _teardown():
            dlg.close()
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            self._thread = self._worker = None
            self.train_btn.setEnabled(True)

        def _on_done(result):
            _teardown()
            self._apply(result)

        def _on_error(msg: str):
            _teardown()
            QMessageBox.critical(self, "Classify", f"Training failed:\n{msg}")

        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        thread.start()

    def _apply(self, result: dict):
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
        self.status_label.setText(
            f"{summary['n_samples']} tagged pairs / {summary['n_rats']} animals → "
            f"{len(result['store'])} pairs classified.\n"
            f"Leave-one-animal-out mean F1 {summary['mean_f1']:.2f}, "
            f"AUC {summary['mean_auc']:.2f}.\n"
            f"Figures and model in {result['out_dir']}")
        self.review_btn.setEnabled(True)
        result['store'].save(os.path.join(result['out_dir'], 'predictions.json'))

    def _on_review_btn(self):
        store = self._store()
        if store is None:
            return
        PredictionReviewDialog(self._win, store).exec()

    @classmethod
    def show_for(cls, win: 'CCGReviewUI'):
        cls(win).exec()


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
        self.resize(620, 480)
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
        label = self.label_combo.currentText()
        sess = self._session()
        if label == 'all pending':
            return self._store.review_order(sess)
        return self._store.pairs_for_label(label, sess)

    def _reload(self):
        pairs = self._visible_pairs()
        self.table.setRowCount(len(pairs))
        self._rows = pairs
        for i, pk in enumerate(pairs):
            state = ('accepted' if pk in self._store.accepted else
                     'rejected' if pk in self._store.rejected else '')
            cells = [f"{pk[0]}  {pk[1]}→{pk[2]}",
                     ', '.join(self._store.labels_for(*pk)),
                     f"{self._store.confidence(*pk):.2f}", state]
            for col, val in enumerate(cells):
                self.table.setItem(i, col, QTableWidgetItem(val))

    def _selected(self) -> list[tuple]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return [self._rows[i] for i in sorted(rows) if i < len(self._rows)]

    def _on_table_selection(self):
        """Show the highlighted pair in the main CCG view so it can be judged."""
        picked = self._selected()
        if len(picked) != 1:
            return
        sess, ref, tgt = picked[0]
        if str(self._win.nav.key.session) != sess:
            return
        self._win.nav.set_current_pair(self._win.nav.get_pair_index((ref, tgt)))

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
