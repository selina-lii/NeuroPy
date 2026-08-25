"""Modules ▸ Classify ▸ Clusters — group CCGs by shape, then tag a group at once.

Tagging pair by pair is slow and drifts: the same shape gets tagged heavily in
one session and lightly in another, and the next model learns the drift as
signal. Here the grouping is unsupervised — the existing tags are shown against
each cluster but never used to form it — so a cluster is evidence about its own
untagged members, and confirming it is one decision instead of forty.

Assignment writes real group tags through the same path as the pair panel, so
Ctrl+Z reaches them and a tagged pair moves to 'selected' exactly as it would by
hand.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import QObject, Qt, QThread, Signal
from pyqtgraph.Qt.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from neuropy.classifier.cluster import (MIN_CLUSTER, SCOPES, cluster_pairs,
                                        in_scope, rule_space)
from neuropy.classifier.dataset import build_multi
from neuropy.ui.utils import (BusyButton, HotkeyTagFilter, ListPickerButton,
                              read_only_cell, read_only_table)

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

# Distinguishable at small dot sizes and stable across runs, so a cluster keeps
# its colour while the reviewer works through the list.
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860',
           '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
DIM = '#3A3A3A'          # rows in no kept cluster


def _tight(box):
    box.setContentsMargins(4, 2, 4, 2)
    box.setSpacing(4)
    return box


class _ClusterWorker(QObject):
    """Feature extraction and Ward linkage, off the GUI thread.

    Both are seconds of work on a few thousand pairs — long enough that running
    them inline freezes the window mid-click.
    """

    done = Signal(object)
    failed = Signal(str)
    advice = Signal(str)     # a scope that cannot work, not a fault to report

    def __init__(self, cd, n_clusters: int, scope: str, sessions: list[str],
                 conn_types: list[str]):
        super().__init__()
        self._cd, self._n, self._scope = cd, n_clusters, scope
        self._sessions, self._conn_types = sessions, conn_types

    def run(self):
        try:
            # all_pairs: clustering exists to reach untagged pairs, so the
            # population is every pointer pair, not the tagged ones training uses.
            ls = build_multi([self._cd], min_count=1, highres=True,
                             conn_types=self._conn_types,
                             sessions=self._sessions, all_pairs=True)
            mask = in_scope(ls.samples, self._scope)
            if mask.sum() < MIN_CLUSTER:
                self.advice.emit(
                    f"Too few pairs to cluster — {int(mask.sum())} in this scope, "
                    f"and a group needs at least {MIN_CLUSTER}. Pick more sessions "
                    f"or connection types and try again.")
                return
            a = ls.arrays(True)
            hi, hi_null = (a.ccg_hi, a.null_hi) if a.has_highres else (None, None)
            X = rule_space(a.ccg[mask], a.null[mask], self._cd.conf.duration,
                           a.acg_ref[mask], a.acg_tgt[mask],
                           None if hi is None else hi[mask],
                           None if hi is None else hi_null[mask])
            samples = [s for s, keep in zip(ls.samples, mask) if keep]
            clustering = cluster_pairs(X, self._n, [s.labels for s in samples])
            self.done.emit((clustering, samples))
        except Exception as exc:                      # surfaced in the status line
            self.failed.emit(str(exc))


class ClusterPanel(QWidget):
    """Scatter of pairs by shape, a cluster picker, and its members as a table."""

    def __init__(self, win: 'CCGReviewUI', owner):
        super().__init__()
        self._win = win
        self._owner = owner          # ClassifierDialog: reuses its tagging path
        self._thread = self._worker = None
        self._clustering = None
        self._samples: list = []
        self._rows: list[tuple] = []
        self._build()

    def _build(self):
        outer = _tight(QVBoxLayout())
        self.setLayout(outer)
        outer.addWidget(QLabel("<b>CCG shape clusters</b>"))

        top = _tight(QHBoxLayout())
        top.addWidget(QLabel("Clusters:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 200)
        self.n_spin.setValue(40)
        top.addWidget(self.n_spin)
        top.addWidget(QLabel("Over:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(SCOPES)
        self.scope_combo.setToolTip(
            "Which pairs to cluster. 'all pairs' shows the whole shape landscape\n"
            "with existing tags as evidence; 'untagged only' makes this a pure\n"
            "worklist of pairs nobody has reviewed.")
        top.addWidget(self.scope_combo)
        nav = self._win.nav
        self.session_picker = ListPickerButton(
            "Session", nav.available_sessions(), plural="sessions",
            refresh_provider=nav.available_sessions)
        top.addWidget(self.session_picker)
        self.type_picker = ListPickerButton(
            "ConnType", nav.available_conn_types(), plural="types",
            refresh_provider=nav.available_conn_types)
        top.addWidget(self.type_picker)
        self.run_btn = BusyButton("Cluster")
        self.run_btn.clicked.connect(self._on_run_btn)
        top.addWidget(self.run_btn)
        top.addStretch()
        outer.addLayout(top)

        split = QSplitter(Qt.Vertical)
        outer.addWidget(split, stretch=1)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel('bottom', 'shape axis 1')
        self.plot.setLabel('left', 'shape axis 2')
        self._scatter = pg.ScatterPlotItem(size=5, pen=None)
        self._scatter.sigClicked.connect(self._on_scatter_click)
        self.plot.addItem(self._scatter)
        split.addWidget(self.plot)

        lower = QWidget()
        lay = _tight(QVBoxLayout())
        lower.setLayout(lay)
        split.addWidget(lower)

        pick = _tight(QHBoxLayout())
        pick.addWidget(QLabel("Cluster id:"))
        self.cid_spin = QSpinBox()
        self.cid_spin.setRange(0, 999)
        self.cid_spin.valueChanged.connect(self._reload)
        pick.addWidget(self.cid_spin)
        pick.addWidget(QLabel("Assign label:"))
        self.label_combo = QComboBox()
        self.label_combo.setMinimumWidth(130)
        pick.addWidget(self.label_combo)
        self.assign_combo = QComboBox()
        # A pair can legitimately carry several tags, so assignment adds rather
        # than replaces; the choice is only which rows to skip.
        self.assign_combo.addItems(["selected rows", "whole cluster",
                                    "untagged in cluster"])
        pick.addWidget(self.assign_combo)
        self.assign_btn = QPushButton("Assign")
        self.assign_btn.clicked.connect(self._on_assign_btn)
        pick.addWidget(self.assign_btn)
        pick.addStretch()
        lay.addLayout(pick)

        self.table = read_only_table(['session', 'pair', 'tags'])
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        HotkeyTagFilter(self.table, self._tag_by_hotkey)
        lay.addWidget(self.table, stretch=1)

        self.status = QLabel("Cluster to group pairs by shape.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

    def refresh_scope(self):
        """Re-read the pickers after a project switch; stale names match no key."""
        nav = self._win.nav
        self.session_picker.set_items(nav.available_sessions(), keep_selection=True)
        self.type_picker.set_items(nav.available_conn_types(), keep_selection=True)

    def _on_run_btn(self):
        if self._thread is not None:
            return
        self.run_btn.set_busy(True, "Clustering…")
        self.status.setText("Extracting shape features…")
        self._thread = QThread(self)
        self._worker = _ClusterWorker(self._win.cd, self.n_spin.value(),
                                      self.scope_combo.currentText(),
                                      self.session_picker.selected,
                                      self.type_picker.selected)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.advice.connect(self._on_advice)
        self._thread.start()

    def _teardown(self):
        self.run_btn.set_busy(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = self._worker = None

    def _on_failed(self, msg: str):
        self._teardown()
        self.status.setText(f"Clustering failed: {msg}")

    def _on_advice(self, msg: str):
        """A scope that cannot be clustered — the reviewer's next move, not an error."""
        self._teardown()
        self.status.setText(msg)

    def _on_done(self, payload):
        self._teardown()
        self._clustering, self._samples = payload
        # Wind the request down to what this scope supports, so the number on
        # screen is the one that ran rather than one silently overridden.
        if self._clustering.n_used < self._clustering.n_asked:
            self.n_spin.setValue(self._clustering.n_used)
        ids = [c.cid for c in self._clustering]
        self.cid_spin.setRange(min(ids), max(ids))
        self._refresh_labels()
        self._draw()
        self.cid_spin.setValue(ids[0])
        self._reload()
        n_pairs = len(self._samples)
        capped = (f"That's too many clusters for {n_pairs} pairs — used "
                  f"{self._clustering.n_used} instead of {self._clustering.n_asked}. "
                  if self._clustering.n_used < self._clustering.n_asked else "")
        n_cl = len(self._clustering)
        self.status.setText(
            f"{capped}{n_cl} cluster{'' if n_cl == 1 else 's'} over {n_pairs} "
            f"pairs. Largest: {len(self._clustering.clusters[0])}.")

    def _refresh_labels(self):
        """Offer the project's shape tags, keeping whatever was already chosen."""
        want = self.label_combo.currentText()
        names = sorted(self._win.nav.groups.defined_groups)
        self.label_combo.clear()
        self.label_combo.addItems(names)
        if want in names:
            self.label_combo.setCurrentText(want)

    def _draw(self):
        """One dot per pair, coloured by cluster; dropped rows stay dim."""
        xy = self._clustering.coords
        kept = self._clustering.kept
        brushes = [pg.mkBrush(DIM if c < 0 else PALETTE[c % len(PALETTE)])
                   for c in kept]
        self._scatter.setData(x=xy[:, 0], y=xy[:, 1], brush=brushes,
                              data=list(range(len(xy))))
        self.plot.autoRange()

    def _on_scatter_click(self, _plot, points):
        """Clicking a dot loads the cluster it belongs to."""
        if not len(points):
            return
        row = points[0].data()
        cid = int(self._clustering.assign[row])
        if self._clustering.by_id(cid) is not None:
            self.cid_spin.setValue(cid)

    def _cluster(self):
        if self._clustering is None:
            return None
        return self._clustering.by_id(self.cid_spin.value())

    def _reload(self):
        """List the current cluster's members, with whatever tags they carry."""
        cluster = self._cluster()
        self._rows = []
        if cluster is None:
            self.table.setRowCount(0)
            return
        rows = []
        for i in cluster.members:
            s = self._samples[i]
            self._rows.append((s.session, s.ref, s.tgt))
            rows.append([s.session, f"{s.ref}-{s.tgt}", ", ".join(s.labels)])
        self.table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                self.table.setItem(r, c, read_only_cell(text))
        top = cluster.top_tag or '(untagged)'
        self.status.setText(
            f"Cluster {cluster.cid}: {len(cluster)} pairs, "
            f"{cluster.n_labeled} tagged, purity {cluster.purity:.2f} ({top}).")

    def _on_table_selection(self):
        """Show the highlighted pair in the main CCG view, as the review tab does."""
        picked = self._selected()
        if len(picked) == 1:
            self._owner.goto_pair(*picked[0])

    def _selected(self) -> list[tuple]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return [self._rows[i] for i in sorted(rows) if i < len(self._rows)]

    def _tag_by_hotkey(self, char: str) -> bool:
        """Tag the selected rows with the hotkey's group, toggling per pair."""
        groups = self._win.nav.groups
        gname = groups.group_for_hotkey(char)
        if gname is None:
            return False
        pairs = self._selected()
        if not pairs:
            self.status.setText("Select a row before using a group hotkey.")
            return True
        changes, promoted = [], {}
        for sess, ref, tgt in pairs:
            if gname in groups.groups_for_pair(sess, ref, tgt):
                groups.discard_from_group(gname, sess, (ref, tgt))
                changes.append((gname, sess, (ref, tgt), 'remove'))
            else:
                groups.add_to_group(gname, sess, (ref, tgt))
                changes.append((gname, sess, (ref, tgt), 'add'))
            promoted.update(self._owner.promote_pair((sess, ref, tgt)))
        self._owner.push_undo(changes, promoted)
        self._owner.after_change()
        self._refresh_rows_tags()
        self.status.setText(f"Toggled '{gname}' on {len(pairs)} pair(s).")
        return True

    def _targets(self) -> list[tuple]:
        """The pairs Assign will write, per the scope combo."""
        mode = self.assign_combo.currentText()
        if mode == "selected rows":
            return self._selected()
        cluster = self._cluster()
        if cluster is None:
            return []
        members = [(self._samples[i], self._rows[n])
                   for n, i in enumerate(cluster.members)]
        if mode == "untagged in cluster":
            return [pk for s, pk in members if not s.labels]
        return [pk for _, pk in members]

    def _on_assign_btn(self):
        """Tag every target with the chosen label, as one undoable step.

        Adds rather than replaces: a pair carrying other tags keeps them, since
        the labels describe different things about the same shape. A pair that
        already has this label is skipped, so re-assigning a cluster is a no-op
        rather than a duplicate write.
        """
        label = self.label_combo.currentText()
        targets = self._targets()
        if not label or not targets:
            self.status.setText("Pick a label and at least one pair.")
            return
        groups = self._win.nav.groups
        added, promoted = [], {}
        for sess, ref, tgt in targets:
            if label in groups.groups_for_pair(sess, ref, tgt):
                continue
            groups.add_to_group(label, sess, (ref, tgt))
            promoted.update(self._owner.promote_pair((sess, ref, tgt)))
            added.append((label, sess, (ref, tgt), 'add'))
        if added:
            self._owner.push_undo(added, promoted)
            self._owner.after_change()
        self._refresh_rows_tags()
        self.status.setText(
            f"Tagged {len(added)} pairs '{label}'"
            f"{f' ({len(targets) - len(added)} already had it)' if len(added) < len(targets) else ''}.")

    def _refresh_rows_tags(self):
        """Re-read the tag column in place — the cluster membership is unchanged."""
        cluster = self._cluster()
        if cluster is None:
            return
        for r, i in enumerate(cluster.members):
            s = self._samples[i]
            s.labels = sorted(self._win.nav.groups.groups_for_pair(
                s.session, s.ref, s.tgt))
            self.table.setItem(r, 2, read_only_cell(", ".join(s.labels)))
