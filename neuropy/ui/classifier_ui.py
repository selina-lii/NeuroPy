"""Modules ▸ Classify — train on the project's tags, then review the predictions.

Predictions never write into the group tags on their own. They land in a
``PredictionStore`` the user reviews pair by pair, so the labels that train the
next model stay the user's own.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyqtgraph.Qt.QtCore import QObject, Qt, QThread, Signal
from pyqtgraph.Qt.QtGui import QBrush, QColor
from pyqtgraph.Qt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from neuropy.analyses.utils import NOTHING, is_shape_label
from neuropy.classifier.models import MODELS
from neuropy.ui.cluster_ui import ClusterPanel
from neuropy.ui.ui_common import SelectionCommand
from neuropy.ui.utils import (BusyButton, HotkeyTagFilter, ListPickerButton, TagChip,
                              confirm_overwrite, read_only_cell,
                              read_only_table)
from neuropy.classifier.predictions import (PredictionStore, by_owner,
                                            store_from_rows)
from neuropy.classifier.run import (DEFAULT_MODEL, apply_cascade, apply_model,
                                    delete_model, list_models, missing_highres,
                                    model_path, open_projects, predict_project,
                                    rename_model,
                                    scope_keys, scorable_keys, train_project)

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

# Shapes that differ only in degree are easier to separate from each other than
# from all thirteen labels at once, so they train as one family. Labels absent
# from a project are dropped by the support filter, so a family costs nothing.
MODEL_RANKING = (
    "Measured mean F1 (n>=200 labels, pooled CV, discover):\n"
    "  rule     0.572   kernel + the hand-written shape rules\n"
    "  kernel   0.554   matched filters at named lags\n"
    "  conv     0.537   learned local filters\n"
    "  dualres  0.438   whole-trace PCA per resolution\n"
    "  shape    0.324   14 interpretable descriptors\n\n"
    "Picking several trains a routed model — measured 0.548, worse than\n"
    "'rule' alone, because the strategies overlap rather than complement.")

LABEL_FAMILIES = {
    'quality (best/good/ok)': ['best', 'good', 'ok', 'bad'],
    'fast patterns': ['msconn', '2peakms', 'wideMs', 'rift', 'triple', 'leak'],
    'rhythm': ['rhythm', '0rhythm', 'burst', 'bimodal'],
    'inhibition': ['ppinhib', 'inhib2sides', 'Disinhib', 'I-I-flip'],
}

# This table is a filtered view of the pair list, so it paints a row the same
# three ways: selected, deleted, or neither.
_SELECTED_TINT = '#2f4f3f'
_DELETED_FG = '#AAAAAA'   # the gray the pair panel grays deleted pairs with


def _tight(box):
    """A control row with no padding around it and none between its widgets."""
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(2)
    return box


def saved_names(opts: dict) -> list[str]:
    """Every classifier name a training run will write, cascade heads included."""
    names = opts['model_names']
    if len(names) > 1 and not opts['route']:
        return [f"{opts['save_as']}.{n}" for n in names]
    return [opts['save_as']]


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
        """Train the chosen strategies as one model, then score the scope with it.

        Several strategies train as a routed model: each label is answered by
        whichever strategy cross-validates best for it. Unchecking *route* trains
        them as separate heads and cascades them in the listed order instead.
        """
        o = self._opts
        names = o['model_names']
        if len(names) > 1 and not o['route']:
            return self._train_cascade(names)
        result = self._fit(names, o['save_as'])
        keys, skipped = scorable_keys(self._cd, result['model'], o['scope'])
        rows = predict_project(self._cd, result['model'], keys)
        result['skipped'] = skipped
        result['store'] = store_from_rows(rows, result['saved_as'],
                                          result['model'].label_names)
        return result

    def _fit(self, names: list[str], save_as: str) -> dict:
        """One train_project call, so a new training option is added in one place."""
        o = self._opts
        return train_project(self._cd, names, figures=o['figures'],
                             save_as=save_as, extra=o['extra'],
                             highres=o['highres'], min_count=o['min_count'],
                             bias=o['bias'], only_labels=o['only_labels'])

    def _train_cascade(self, names: list[str]) -> dict:
        """Train one model per strategy, then let each claim what the earlier missed."""
        o = self._opts
        heads = [self._fit([name], save_as)
                 for name, save_as in zip(names, saved_names(o))]
        out = apply_cascade(self._cd, [h['saved_as'] for h in heads], o['scope'])
        result = dict(heads[-1])
        result['saved_as'] = ' → '.join(h['saved_as'] for h in heads)
        result['heads'] = [h['summary'] for h in heads]
        result['skipped'] = out['skipped']
        result['store'] = store_from_rows(out['rows'], result['saved_as'],
                                          out['labels'])
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
    Enter takes the whole row when every label is correct.
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
        self._row_of: dict = {}
        self._key_cache: dict = {}   # (session, type_label) -> nd-key; see _key_of
        self._mark = 0               # undo depth on open; see mark_session
        self.setWindowTitle("Classify pairs")
        self.resize(620, 520)
        # Non-modal throughout: judging a candidate pair means toggling
        # resolution and panels in the main window while this stays open.
        self.setModal(False)
        self._build()

    def _build(self):
        # Training above, review below: two jobs that compete for height, so the
        # sash lets whichever one is in use take the space.
        self.body = QWidget()   # the panel re-parents this; the dialog just holds it
        QVBoxLayout(self).addWidget(self.body)
        outer = QVBoxLayout(self.body)
        outer.setContentsMargins(0, 0, 0, 0)
        # Two ways to reach the same tags: a model proposing them one pair at a
        # time, or a cluster proposing them for a whole shape at once.
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        heads = QWidget()
        self.tabs.addTab(heads, "MLP heads")
        outer = QVBoxLayout(heads)
        outer.setContentsMargins(0, 0, 0, 0)
        self._splitter = QSplitter(Qt.Vertical)
        outer.addWidget(self._splitter)
        train_box, lay = QWidget(), _tight(QVBoxLayout())
        train_box.setLayout(lay)
        self._splitter.addWidget(train_box)

        row = _tight(QHBoxLayout())
        row.addWidget(QLabel("Model:"))
        # Ordered: several strategies train as a cascade of heads, each claiming
        # the labels the earlier ones left, so the order is the priority.
        self.model_picker = ListPickerButton(
            "Model", list(MODELS), plural="models",
            select_all_when_empty=False, ordered=True)
        self.model_picker.setToolTip(MODEL_RANKING)
        self.model_picker.set_selected([DEFAULT_MODEL])
        row.addWidget(self.model_picker)
        self.route_check = QCheckBox("route per label")
        self.route_check.setChecked(True)
        self.route_check.setToolTip(
            "With several strategies: give each label to the one that scores it\n"
            "best in cross-validation. Unchecked, they run as a cascade in the\n"
            "listed order and each claims what the earlier ones missed.")
        row.addWidget(self.route_check)
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
        scope_row = _tight(QHBoxLayout())
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
        saved_row = _tight(QHBoxLayout())
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
        review_box, lay = QWidget(), _tight(QVBoxLayout())
        review_box.setLayout(lay)
        self._splitter.addWidget(review_box)
        filt = _tight(QHBoxLayout())
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
        # a chip click moves one label across, and the row stays either way.
        self.table = read_only_table(
            ['session', 'pair', 'predicted', 'confidence', 'accepted'])
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_menu)
        HotkeyTagFilter(self.table, self._tag_by_hotkey, self._delete_selected,
                        self._admit_selected)
        lay.addWidget(self.table, stretch=1)

        act = _tight(QHBoxLayout())
        self.accept_sel_btn = QPushButton("Accept selected")
        self.accept_sel_btn.setToolTip(
            "Tag the rows picked out with the mouse — with the filtered label\n"
            "only, when the label filter names one — then go back to the pair list.")
        self.accept_sel_btn.clicked.connect(self._on_accept_sel_btn)
        act.addWidget(self.accept_sel_btn)
        self.accept_all_btn = QPushButton("Accept all shown")
        self.accept_all_btn.setToolTip(
            "Tag every row currently listed — narrow the list with the label\n"
            "filter and the confidence cutoff first — then go back to the pair list.")
        self.accept_all_btn.clicked.connect(self._on_accept_all_btn)
        act.addWidget(self.accept_all_btn)
        act.addStretch()
        lay.addLayout(act)

        self.clusters = ClusterPanel(self._win, self)
        self.tabs.addTab(self.clusters, "Clustering")

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close
                              | QDialogButtonBox.StandardButton.Save)
        bb.rejected.connect(self._on_close_btn)
        self.save_btn = bb.button(QDialogButtonBox.StandardButton.Save)
        self.save_btn.setToolTip("Keep the tags accepted here; closing without this undoes them.")
        bb.accepted.connect(self._on_save_btn)
        for b in bb.buttons():   # Enter belongs to the table, not the default button
            b.setAutoDefault(False)
            b.setDefault(False)
        self.body.layout().addWidget(bb)   # below the tabs, shared by both
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
                'model_names': self.model_picker.selected or [DEFAULT_MODEL],
                'route': self.route_check.isChecked(),
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
        if not saved and not all(
                confirm_overwrite(self, model_path(self._win.cd, n), 'classifier')
                for n in saved_names(opts)):
            return
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
            sessions, 'highres', on_done=lambda ok: self._on_precomputed(ok, opts),
            on_progress=lambda done, total: self.train_btn.set_busy(
                True, f"Computing high-res {done}/{total}"))
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
        routes = summary.get('routes') or {}
        if routes:   # which strategy won each label, most-used first
            per = by_owner(routes, sorted(routes))
            lines.append("Routing — " + " | ".join(
                f"{k}: {', '.join(v)}" for k, v
                in sorted(per, key=lambda kv: -len(kv[1]))))
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

    def _filtered_label(self) -> str | None:
        """The one label the listing is narrowed to, or ``None`` for all of them.

        NOTHING is the model's "cleared no threshold" marker, not a label, so it
        scopes the listing but can never be tagged onto a pair.
        """
        label = self.label_combo.currentText()
        return None if label in ('all pending', NOTHING) else label

    def _visible_pairs(self) -> list[tuple]:
        """Rows the table shows: scope- and label-filtered, above the cutoff."""
        if self._store is None:
            return []
        label = self.label_combo.currentText()
        sess = self._session()
        one = self._filtered_label()
        pairs = (self._store.review_order(sess)
                 if label == 'all pending'
                 else self._store.pairs_for_label(label, sess))
        cutoff = self.cutoff_spin.value()
        return [pk for pk in pairs
                if self._store.confidence(*pk, label=one) >= cutoff]

    def _sync_labels(self):
        """Refill the label filter from the current store, keeping the choice."""
        want = ['all pending'] + (self._store.labels if self._store else []) + [NOTHING]
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
        shown = self._filtered_label()
        self.table.setRowCount(len(pairs))
        self._rows = pairs
        self._row_of = {pk: i for i, pk in enumerate(pairs)}
        self.accept_all_btn.setText(f"Accept all shown ({len(pairs)})")
        for i, pk in enumerate(pairs):
            self._paint_row(i, pk, shown)
        self.table.resizeRowsToContents()

    def _paint_row(self, i: int, pk: tuple, shown: str = None):
        """Draw row *i* for pair *pk* in the pair list's three states."""
        state = self._pair_state(pk)
        for col, val in ((0, pk[0]),
                         (1, f"{pk[1]}→{pk[2]}"),
                         (3, f"{self._store.confidence(*pk, label=shown):.2f}")):
            cell = read_only_cell(val)
            if state == 'sel':
                cell.setBackground(QBrush(QColor(_SELECTED_TINT)))
            elif state == 'del':
                cell.setForeground(QBrush(QColor(_DELETED_FG)))
            self.table.setItem(i, col, cell)
        self.table.setCellWidget(i, 2, self._chip_cell(pk, taken=False))
        self.table.setCellWidget(i, 4, self._chip_cell(pk, taken=True))

    def _chip_cell(self, pk: tuple, taken: bool) -> QWidget:
        """One side of the row's editor: accepted labels when *taken*, else pending.

        Clicking a chip moves that label to the other side — accepting one label
        leaves the pair's other predictions on the table, still to be judged.
        """
        cell = QWidget()
        col = QVBoxLayout(cell)
        col.setContentsMargins(2, 1, 2, 1)
        col.setSpacing(1)
        groups = self._win.nav.groups
        labels = (self._store.taken(groups, *pk) if taken
                  else self._store.pending(groups, *pk))
        # One line per head that proposed something, so a cascade reads
        # "conv: [best] [refractory] / kernel: [burst]"; Enter takes the row.
        lines = self._store.by_model(*pk, labels)
        if taken:
            hand = self._store.hand_tags(groups, *pk)
            if hand:
                lines = lines + [('by hand', hand)]
        for model, names in lines:
            row = QHBoxLayout()
            row.setSpacing(TagChip.GAP)
            if len(lines) > 1:
                tag = QLabel(f"{model}:")
                tag.setStyleSheet("color: gray;")
                row.addWidget(tag)
            for label in names:
                chip = _AcceptChip(label, self._colour(label), label)
                chip.clicked.connect(
                    (lambda one, p=pk: self._unaccept_one(p, one)) if taken
                    else (lambda one, p=pk: self._accept_one(p, one)))
                row.addWidget(chip)
            row.addStretch()
            col.addLayout(row)
        if taken and self._pair_state(pk) == 'del':
            col.addLayout(self._deleted_chip_row(pk))
        return cell

    def _deleted_chip_row(self, pk: tuple) -> QHBoxLayout:
        """Deleted shown as its own chip, clicked to undelete as a tag is untagged."""
        row = QHBoxLayout()
        row.setSpacing(TagChip.GAP)
        chip = _AcceptChip('deleted', _DELETED_FG, 'deleted')
        chip.clicked.connect(lambda _, p=pk: self._delete_selected([p]))
        row.addWidget(chip)
        row.addStretch()
        return row

    def _colour(self, label: str) -> str:
        """Chip tint for *label*, taken from its group."""
        return self._win.nav.groups.get_group_metadata(label).display_color

    def _accept(self, pairs: list[tuple], only: str = None):
        """Accept across *pairs* as one undoable step, repainting their rows in place."""
        added, promoted = [], {}
        for pk in pairs:
            new = self._store.accept(pk[0], pk[1], pk[2], self._win.nav.groups,
                                     only=only)
            if new:
                promoted.update(self.promote_pair(pk))
            added += new
        if added:
            self.push_undo([(g, s, p, 'add') for g, s, p in added], promoted)
        self.after_change()
        for pk in pairs:
            self._redraw_row(pk)

    def _key_of(self, pk: tuple):
        """The nd-key a predicted pair belongs to, memoized across a batch.

        ``cd.find`` rescans every pointer key, and Accept-all resolves hundreds
        of pairs that fall into a handful of (session, conn type) buckets.
        """
        ident = (pk[0], self._store.type_label_for(*pk))
        if ident not in self._key_cache:
            self._key_cache[ident] = self._win.cd.find(
                ident[0], type_label=ident[1], strict=False)
        return self._key_cache[ident]

    def _pair_state(self, pk: tuple) -> str | None:
        """The pair's state in its own key's bucket, or ``None`` if it has no key.

        Deleted is tested for, never inferred: an unpopulated bucket has no state.
        """
        key = self._key_of(pk)
        if key is None:
            return None
        bucket = self._win.nav.sd.get_selection_by_session(key).selections[key]
        pair = (pk[1], pk[2])
        if pair in bucket.selected:
            return 'sel'
        if pair in bucket.deleted:
            return 'del'
        return 'unsel' if pair in bucket.unselected else None

    def _set_state(self, pk: tuple, want: str) -> dict:
        """Move a pair to *want* in its own key's bucket.

        Returns the change for undo, keyed by pair — empty unless the pair sits in
        the key on screen, which is the only bucket SelectionCommand can restore.
        """
        was = self._pair_state(pk)
        if was is None or was == want:
            return {}
        key = self._key_of(pk)
        self._win.nav.sd.get_selection_by_session(key).selections[key].set_pair_state(
            (pk[1], pk[2]), want)
        return {(pk[1], pk[2]): (was, want)} if key == self._win.nav.key else {}

    def promote_pair(self, pk: tuple, add: bool = True) -> dict:
        """Select a pair on tagging, and put it back when its last shape tag goes.

        A deleted pair stays deleted: tagging is not what undeletes it, and the
        reviewer's delete would otherwise be undone behind their back.
        """
        if self._pair_state(pk) == 'del':
            return {}
        return self._set_state(pk, 'sel' if add else 'unsel')

    def _still_tagged(self, pk: tuple) -> bool:
        """True while the pair carries any shape tag — an admitted marker is not one."""
        return self._win.nav.sd.has_shape_tag(pk[0], (pk[1], pk[2]))

    def push_undo(self, group_changes: list, pair_changes: dict = None):
        """Record group edits on the pair panel's stack so Ctrl+Z reaches them.

        Accepting writes real tags — Accept all can write hundreds — so it belongs
        on the same stack as hand tagging rather than being irreversible.
        """
        self._win.pairs_view.pair_selection.push_undo(
            SelectionCommand(pair_changes or {}, group_changes))

    def _accept_one(self, pk: tuple, label: str):
        """Tag *pk* with the one predicted label whose chip was clicked."""
        self._accept([pk], only=label)

    def _admit_selected(self) -> bool:
        """Enter: move every predicted label on the picked rows across to accepted."""
        pairs = self._selected()
        if pairs:
            self._accept(pairs)
        return True

    def _unaccept_one(self, pk: tuple, label: str):
        """Send an accepted label back to pending, untagging the pair.

        Losing the last shape tag returns the pair to unselected, so accept and
        unaccept are symmetric rather than leaving it selected with no tags.
        """
        self._win.nav.groups.discard_from_group(label, pk[0], (pk[1], pk[2]))
        demoted = {} if self._still_tagged(pk) else self.promote_pair(pk, add=False)
        self.push_undo([(label, pk[0], (pk[1], pk[2]), 'remove')], demoted)
        self.after_change()
        self._redraw_row(pk)

    def _tag_by_hotkey(self, char: str) -> bool:
        """Tag the selected rows with the hotkey's group, toggling as the pair panel does."""
        groups = self._win.nav.groups
        gname = groups.group_for_hotkey(char)
        if gname is None:
            return False
        pairs = self._selected()
        if not pairs:
            self._win._show_transient_banner("Select a row before using a group hotkey")
            return True
        changes = []
        for pk in pairs:
            pair = (pk[1], pk[2])
            if gname in groups.groups_for_pair(*pk):
                groups.discard_from_group(gname, pk[0], pair)
                changes.append((gname, pk[0], pair, 'remove'))
            else:
                groups.add_to_group(gname, pk[0], pair)
                changes.append((gname, pk[0], pair, 'add'))
        if changes:
            promoted = {}
            for pk in pairs:
                promoted.update(self.promote_pair(pk, add=self._still_tagged(pk)))
            self.push_undo(changes, promoted)
        self.after_change()
        for pk in pairs:
            self._redraw_row(pk)
        return True

    def _redraw_row(self, pk: tuple):
        """Repaint one row's chips in place — a rebuild would re-sort under the cursor."""
        i = self._row_of.get(pk)
        if i is None:
            return
        self._paint_row(i, pk, self._filtered_label())
        self.table.resizeRowToContents(i)

    def _selected(self) -> list[tuple]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return [self._rows[i] for i in sorted(rows) if i < len(self._rows)]

    def _on_table_selection(self):
        picked = self._selected()
        if len(picked) == 1:
            self.goto_pair(*picked[0])

    def goto_pair(self, sess: str, ref: int, tgt: int):
        """Show a pair in the main CCG view so it can be judged.

        A pair may belong to another session or conn type than the one on
        screen, so navigate there first — otherwise the pair index resolves
        against the wrong list and shows an unrelated CCG.
        """
        nav = self._win.nav
        want_type = self._store.type_label_for(sess, ref, tgt) if self._store else None
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

    def _on_accept_sel_btn(self):
        self._accept(self._selected(), only=self._filtered_label())
        self._return_to_pairs()

    def _return_to_pairs(self):
        """Give the left panel back; accepting is a commit, so it survives Close."""
        self.mark_session()
        self._win.pairs_view.pair_selection.refresh_lists()
        self._win.show_pair_selection()

    def _on_accept_all_btn(self):
        pairs = self._visible_pairs()
        only = self._filtered_label()
        what = f"'{only}'" if only else "their predicted labels"
        if QMessageBox.question(
                self, "Accept all",
                f"Tag {len(pairs)} pairs with {what}?\n"
                "Ctrl+Z undoes this as one step.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._accept(pairs, only=only)
        self._return_to_pairs()

    def _on_table_menu(self, pos):
        """Right-click actions on the picked-out rows."""
        pairs = self._selected()
        if not pairs:
            return
        menu = QMenu(self.table)
        n, only = len(pairs), self._filtered_label()
        menu.addAction(f"Accept {'‘' + only + '’ on ' if only else ''}{n} selected",
                       lambda: self._accept(pairs, only=only))
        verb = 'Delete' if self._would_delete(pairs) else 'Undelete'
        menu.addAction(f"{verb} {n} selected",
                       lambda: self._delete_selected(pairs))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _would_delete(self, pairs: list[tuple]) -> bool:
        """True when the toggle should delete rather than undelete.

        Whichever way the pick leans decides the verb, so a mixed selection
        resolves one way instead of doing half of each.
        """
        dead = sum(self._pair_state(pk) == 'del' for pk in pairs)
        return dead * 2 < len(pairs)

    def _delete_selected(self, pairs: list[tuple] = None) -> bool:
        """Toggle deleted on the picked rows, as Delete does in the pair list."""
        pairs = self._selected() if pairs is None else pairs
        if not pairs:
            return False
        want = 'del' if self._would_delete(pairs) else 'unsel'
        changes = {}
        for pk in pairs:
            changes.update(self._set_state(pk, want))
        if changes:
            self.push_undo([], changes)
        self.after_change()
        for pk in pairs:
            self._redraw_row(pk)
        return True

    # shared with the cluster tab, which refreshes its own table
    def after_change(self):
        self._win.nav.groups.changed.emit()

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
        self._key_cache.clear()   # keys belong to the project that resolved them
        for picker, provider in ((self.session_picker, nav.available_sessions),
                                 (self.type_picker, nav.available_conn_types),
                                 (self.label_picker, self._shape_labels),
                                 (self.extra_picker, self._other_projects)):
            picker.set_items(provider(), keep_selection=True)
        self.clusters.refresh_scope()
        self.name_edit.setText(self._win.cd.conf.name)
        self.status_label.setText(self._describe_existing())
        self.refresh_saved()
        self._reload()

    @property
    def _undo_stack(self) -> list:
        """The pair panel's stack, where every edit this dialog makes is recorded."""
        return self._win.pairs_view.pair_selection._undo_stack

    def mark_session(self):
        """Remember the undo depth, so closing can tell this visit's edits apart."""
        self._mark = len(self._undo_stack)

    def _on_save_btn(self):
        """Keep this visit's tags: stop treating them as undoable on close."""
        self.mark_session()
        self._win._show_transient_banner("Accepted tags kept")

    def _on_close_btn(self):
        """Close means 'give the left panel back', dropping tags not kept.

        Everything the dialog wrote since it opened is on the pair panel's stack,
        so unwinding to the opening depth restores the state the reviewer arrived
        with — one restore, no per-edit bookkeeping.
        """
        n = len(self._undo_stack) - self._mark
        if n and QMessageBox.question(
                self, "Discard changes",
                f"Undo {n} change(s) made here?\nSave first to keep them.") \
                == QMessageBox.StandardButton.Yes:
            panel = self._win.pairs_view.pair_selection
            for _ in range(n):
                panel.undo()
            self._reload()
        self.mark_session()
        self._win.show_pair_selection()

    @classmethod
    def show_for(cls, win: 'CCGReviewUI'):
        """Open (or resurface) the one classifier for *win*, in the left panel."""
        if win.classifier_dialog is None:
            win.classifier_dialog = cls(win)
        else:
            win.classifier_dialog.refresh_scope()
        win.classifier_dialog.mark_session()
        win.show_classifier(win.classifier_dialog.body)
        return win.classifier_dialog


def _score_cells(s: dict) -> list[str]:
    return [str(s['n']), f"{s['precision']:.2f}/{s['recall']:.2f}",
            f"{s['f1']:.2f}/{s['auc']:.2f}"]


def _fill_rows(table: QTableWidget, rows: list[list[str]]):
    table.clearContents()   # setRowCount alone leaves stale cell widgets behind
    table.setRowCount(len(rows))
    for i, cells in enumerate(rows):
        for col, val in enumerate(cells):
            table.setItem(i, col, read_only_cell(val))


class StrategyScoresDialog(QDialog):
    """How every strategy scored on one label, best first — why the router chose."""

    def __init__(self, parent, label: str, by_strategy: dict, chosen: str):
        super().__init__(parent)
        self.setWindowTitle(f"{label}: every strategy")
        self.resize(380, 320)
        lay = QVBoxLayout(self)
        rows = sorted(by_strategy.items(), key=lambda ns: -ns[1]['f1'])
        table = read_only_table(['strategy', 'n', 'prec/rec', 'F1/AUC'])
        _fill_rows(table, [[f"{n} ←" if n == chosen else n] + _score_cells(s)
                           for n, s in rows])
        lay.addWidget(table)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


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
        self.stats_table = read_only_table(
            ['label', 'strategy', 'n', 'prec/rec', 'F1/AUC', ''])
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
            item.setData(Qt.UserRole,
                         {'model_key': entry.get('model_key', '—'),
                          **entry.get('trained_on', {})})
            self.list.addItem(item)
        self._parent.refresh_saved()

    def _current(self) -> str:
        item = self.list.currentItem()
        return item.text() if item else ''

    def _on_selected(self, item: 'QListWidgetItem'):
        """Per-label CV scores for the selected model, recorded at training time."""
        meta = item.data(Qt.UserRole) if item else {}
        scores = meta.get('scores', {})
        routes = meta.get('routes', {})
        per_strategy = meta.get('per_strategy', {})
        ranked = sorted(scores.items(), key=lambda kv: -kv[1]['f1'])
        _fill_rows(self.stats_table,
                   [[label, routes.get(label) or meta['model_key']] + _score_cells(s)
                    for label, s in ranked])
        if len(per_strategy) < 2:
            return
        for i, (label, _) in enumerate(ranked):
            btn = QPushButton("more info")
            # Slice per row: the button outlives this call, so it holds one
            # label's scores rather than the whole provenance table.
            mine = {n: sc[label] for n, sc in per_strategy.items() if label in sc}
            btn.clicked.connect(
                lambda _=False, l=label, ps=mine, c=routes.get(label, ''):
                    StrategyScoresDialog(self, l, ps, c).exec())
            self.stats_table.setCellWidget(i, 5, btn)

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
