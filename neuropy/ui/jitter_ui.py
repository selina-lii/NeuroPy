"""
Jitter UI glue for the CCG Review UI.

JitterWorker        — low-level process-pool / cache management (no Tk).
JitterManager  — Qt-native orchestration; uses QTimer + signals.
"""
from __future__ import annotations

import multiprocessing as _mp
from typing import TYPE_CHECKING

from neuropy.analyses._jitter_worker import jitter_worker
from neuropy.analyses.jitter import JitterTask
from neuropy.analyses.jitter_batch import DEFAULT_NJITTER, pairs_for, run_and_store
from neuropy.analyses.utils import group_display
from neuropy.ui.ui_common import BackgroundTaskRunner, LRUCache
from pyqtgraph.Qt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
    QComboBox, QLabel, QCheckBox, QProgressDialog, QApplication)
from pyqtgraph.Qt.QtCore import QObject, QTimer, Qt, Signal as _Signal
from pyqtgraph.Qt.QtWidgets import QMessageBox

from neuropy.ui.utils import ListPickerButton, ResultsDialog

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState


_ALL_SEGS = "all"

class JitterWorker:
    """Manages on-demand jitter computation queue, process lifecycle, and in-memory cache.

    Process/queue/deque lifecycle is delegated to BackgroundTaskRunner.

    Cache key: (ref: int, tgt: int, res_key: str, seg_arg: int | None)
    Cache value: (j_avg, j_pval, j_pval_bins, j_lo, j_hi)
    """

    UNVIEWED_BG = '#FFEE58'
    UNVIEWED_FG = '#333333'
    VIEWED_BG   = '#FFF9C4'
    VIEWED_FG   = '#333333'

    def __init__(self, max_queue: int, max_cache: int):
        self._runner = BackgroundTaskRunner(
            max_queue=max_queue, use_result_queue=True)
        self._cache: LRUCache = LRUCache(max_cache)
        self.unviewed: set = set()

    def enqueue(self, tag: str, ref: int, tgt: int, njitter: int,
                res_key: str, bin_size_eff: float,
                seg_arg=None, t0=None, t1=None, nd_key=None):
        self._runner.enqueue(
            JitterTask(tag, ref, tgt, njitter, res_key, bin_size_eff, seg_arg, t0, t1,
                       nd_key))

    def is_running(self) -> bool:
        return self._runner.is_running()

    def start_next(self, key, neurons, ccg_data_lo, ccg_data_hi, edge_times,
                   ccg_conf) -> bool:
        def _launch(task, q):
            ccg_data = ccg_data_hi if task.res_key == 'hi' else ccg_data_lo
            if ccg_data is None:
                return None  # signals skip to BackgroundTaskRunner
            return _mp.Process(
                target=jitter_worker,
                args=(q, key, neurons, ccg_data, edge_times,
                      task.ref, task.tgt, task.njitter, task.bin_size_eff, ccg_conf),
                kwargs={'segment': task.seg_arg, 't0': task.t0, 't1': task.t1},
                daemon=True,
            )
        return self._runner.start_next(_launch)



class JitterManager(QObject):
    """
    Signals
    -------
    completed(ref, tgt, res_key, seg_key)
        Emitted after a successful jitter run.  Panels should call
        request_render() when they receive this if the pair matches.
    failed(msg)
        Emitted on worker error.
    status_changed(text)
        Button-label text: empty string means "idle / Run Jitter".
    colors_changed(pair_or_None)
        Request list-color refresh.  None = all rows; (ref,tgt) = one row.
    """

    completed = _Signal(int, int, str, object)
    failed    = _Signal(str)
    status_changed   = _Signal(str)
    colors_changed   = _Signal(object)

    def __init__(self, nav: 'AppState', cd):
        super().__init__()
        self.nav = nav
        self.cd  = cd
        self.jitter_worker = JitterWorker(max_queue=nav.max_jitter_queue,
                                          max_cache=nav.max_jitter_cache)
        self._timer = QTimer()
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._poll)

    def run_jitter(self, key, njitter: int, run_lo: bool = True, run_hi: bool = False):
        """Enqueue jitter for *key*'s pair at the current segment and start polling."""
        ref, tgt = key.ref, key.tgt
        nav = self.nav
        if nav.neurons is None:
            QMessageBox.critical(None, "Jitter", "No neuron data attached.")
            return

        total = ((1 if self.jitter_worker.is_running() else 0)
                 + self.jitter_worker._runner.pending_count())
        cap = self.jitter_worker._runner._max_queue
        if total >= cap:
            QMessageBox.warning(None, "Jitter",
                                f"Queue full ({total}/{cap}).")
            return

        seg_arg = self.seg()
        # An appended window jitters over its own extent (source config); 'full' = whole session.
        label = nav.segment_name(seg_arg) if seg_arg is not None else _ALL_SEGS
        src = self.cd.source_config(nav.key, label) if label and label != _ALL_SEGS else None
        if src is not None and not isinstance(src.t0, str) and not isinstance(src.t1, str):
            jitter_t0, jitter_t1 = float(src.t0), float(src.t1)
        else:
            jitter_t0 = jitter_t1 = None

        nd_key = key.nd()
        lo_ccg = self.cd.ccg_for(nd_key.change(resolution='lowres')) or nav.ccg_data
        hi_ccg = self.cd.ccg_for(nd_key.change(resolution='highres'))

        for res_key, ccg_data, should_run in [('lo', lo_ccg, run_lo),
                                               ('hi', hi_ccg, run_hi)]:
            if not should_run:
                continue
            if ccg_data is None:
                if res_key == 'hi':
                    QMessageBox.warning(None, "Jitter",
                                        "High-res CCG not loaded; cannot run hi jitter.")
                continue
            n = ccg_data.ccg.shape[-1]
            bin_size_eff = (self.cd.conf.duration / (n - 1)
                            if n > 1 else self.cd.conf.bin_size)
            self.jitter_worker.enqueue('jitter', ref, tgt, njitter,
                                       res_key, bin_size_eff,
                                       seg_arg, jitter_t0, jitter_t1, nd_key)

        self._update_status()
        self._start_next()

    def _neurons_for_task(self, task, nd_key):
        """Neurons over the segment's active fragments — a contiguous slice would let the
        null see filtered-out time the real CCG never saw."""
        whole = self.cd.nd.neurons_for(nd_key)   # the task's session: the cursor may have moved since enqueue
        if task.seg_arg is None:
            return whole
        label = self.cd.segment_name(nd_key, task.seg_arg)
        src = self.cd.source_config(nd_key, label)
        sliced = self.cd.nd.sliced_neurons_for(src) if src is not None else None
        return sliced[0] if sliced is not None else whole

    def _start_next(self):
        nav = self.nav
        if not self.jitter_worker._runner.pending_count():
            self._update_status()
            return
        task = self.jitter_worker._runner._pending[0]
        nd_key = task.nd_key if task.nd_key is not None else nav.key.nd()
        lo = self.cd.ccg_for(nd_key.change(resolution='lowres')) or nav.ccg_data
        hi = self.cd.ccg_for(nd_key.change(resolution='highres'))
        started = self.jitter_worker.start_next(
            nav.key, self._neurons_for_task(task, nd_key), lo, hi,
            None, self.cd.conf)   # no CCG-derived edge times; window bounds ride on the task
        self._update_status()
        if started and not self._timer.isActive():
            self._timer.start()

    def _poll(self):
        if self.jitter_worker.is_running():
            return
        self._timer.stop()
        completed, result = self.jitter_worker._runner.poll()
        if result is not None and not result.get('error') and result.get('j_avg') is not None:
            res_key  = completed.res_key  if completed is not None else 'lo'
            seg_key  = completed.seg_arg  if completed is not None else None
            ref, tgt = int(result['ref']), int(result['tgt'])
            nd_key   = completed.nd_key if completed is not None else self.nav.key.nd()
            cache_key = (nd_key, ref, tgt, res_key, seg_key)
            jitter_val = (result.get('j_avg'), result.get('j_pval'),
                          result.get('j_pval_bins'), result.get('j_lo'), result.get('j_hi'),
                          result.get('bands'))
            # live cache only: a single-pair run is exploratory, batches are what persist
            self.jitter_worker._cache.put(cache_key, jitter_val)
            self.jitter_worker.unviewed.add((ref, tgt))
            self.completed.emit(ref, tgt, res_key, seg_key)
            self.colors_changed.emit((ref, tgt))
        elif completed is not None:
            self.failed.emit(str((result or {}).get('error') or "jitter worker exited without a result"))
        self._start_next()


    def load_from_cd(self):
        """Pull saved batch results for this session into the live cache."""
        self.cd.load_jitter()
        nd_key = self.nav.key.nd()
        for cache_key, val in self.cd._jitter_results.get(nd_key, {}).items():
            self.jitter_worker._cache.put((nd_key, *cache_key), val)
        self.colors_changed.emit(None)

    def on_save(self):
        try:
            self.cd.save_jitter()
            total = sum(len(v) for v in self.cd._jitter_results.values())
            lines = [f"Saved {total} jitter pair(s).", ""]
            lines += [f"  {str(nd.session)}: {len(v)} pair(s)"
                      for nd, v in self.cd._jitter_results.items() if v]
            ResultsDialog.show_report("Save Jitter", "\n".join(lines))
        except Exception as exc:
            QMessageBox.critical(None, "Save Jitter", f"Save failed:\n{exc}")


    def seg(self, seg=None) -> int | None:
        """Segment key for jitter cache: None for the whole-session 'full' segment, else the
        appended window's dim0 index (so each window keeps a distinct cache entry)."""
        nav = self.nav
        if seg is None:
            seg = nav.current_segment
        if isinstance(seg, str):
            if seg == _ALL_SEGS:                 # 'full' = whole session
                return None
            idx = nav.segment_index(seg)
            return idx if idx > 0 else None
        return None if seg <= 0 else int(seg)

    def clear_queue(self) -> int:
        """Remove all pending (non-running) tasks. Returns count removed."""
        worker = self.jitter_worker
        if worker.is_running() and worker._runner._pending:
            pending_to_clear = list(worker._runner._pending[1:])
        else:
            pending_to_clear = list(worker._runner._pending)
        n = len(pending_to_clear)
        worker._runner._pending.clear()
        return n

    def clear(self, key):
        """Drop *key*'s pair at both resolutions, live and saved."""
        saved = self.cd._jitter_results.get(key.nd(), {})
        for rk in ('lo', 'hi'):
            self.jitter_worker._cache.pop((key.nd(), key.ref, key.tgt, rk, self.seg()), None)
            saved.pop((key.ref, key.tgt, rk, self.seg()), None)
        self.jitter_worker.unviewed.discard((key.ref, key.tgt))
        self.colors_changed.emit((key.ref, key.tgt))

    def apply_list_colors(self):
        """Refresh pair-list jitter highlight colors (no-op if lists handle via signal)."""
        self.colors_changed.emit(None)

    def mark_viewed(self, ref: int = None, tgt: int = None) -> bool:
        """Mark pair as viewed. Returns True if newly marked."""
        if ref is None or tgt is None:
            inds = self.nav.current_pair_inds
            if inds is None:
                return False
            ref, tgt = int(inds[0]), int(inds[1])
        pair = (ref, tgt)
        if pair in self.jitter_worker.unviewed:
            self.jitter_worker.unviewed.discard(pair)
            self.colors_changed.emit(pair)
            return True
        return False

    def get_result(self, key):
        """Cached jitter tuple for *key*'s pair at its resolution, or None."""
        res_key = 'hi' if key.resolution == 'highres' else 'lo'
        return self.jitter_worker._cache.get((key.nd(), key.ref, key.tgt, res_key, self.seg()))

    def _update_status(self):
        running = self.jitter_worker.is_running()
        pending = self.jitter_worker._runner._pending
        if running and pending:
            task = pending[0]
            nav = self.nav
            seg_name = _ALL_SEGS
            if task.seg_arg is not None:
                try:
                    seg_name = str(nav.segment_name(int(task.seg_arg)))
                except Exception:
                    seg_name = f"seg{task.seg_arg}"
            n_extra = len(pending) - 1
            suffix = f" +{n_extra} queued" if n_extra > 0 else ''
            self.status_changed.emit(
                f"Jitter [{task.ref},{task.tgt}] {seg_name}…{suffix}")
        else:
            self.status_changed.emit('')


class JitterQueueDialog(QDialog):
    """Shows pending jitter tasks; allows deleting from queue."""

    def __init__(self, jitter_mgr, parent=None):
        super().__init__(parent)
        self._ctrl = jitter_mgr
        self.setWindowTitle("Jitter Queue")
        self.resize(420, 280)
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        layout.addWidget(self._list)

        row = QHBoxLayout()
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_selected)
        ref_btn = QPushButton("Refresh")
        ref_btn.clicked.connect(self._refresh)
        cls_btn = QPushButton("Close")
        cls_btn.clicked.connect(self.accept)
        row.addWidget(del_btn); row.addWidget(ref_btn)
        row.addStretch(); row.addWidget(cls_btn)
        layout.addLayout(row)
        self._refresh()

    def _refresh(self):
        self._list.clear()
        w = self._ctrl.jitter_worker
        pending = list(w._runner._pending)
        if w.is_running() and pending:
            self._list.addItem(f"[running] {pending[0]}")
            for t in pending[1:]:
                self._list.addItem(str(t))
        else:
            for t in pending:
                self._list.addItem(str(t))
        if self._list.count() == 0:
            self._list.addItem("(queue empty)")

    def _delete_selected(self):
        row = self._list.currentRow()
        if row < 0:
            return
        w = self._ctrl.jitter_worker
        pending = w._runner._pending
        # row 0 = running task if running — skip it
        offset = 1 if (w.is_running() and pending) else 0
        del_idx = row - offset
        if 0 <= del_idx < len(pending) - (1 if w.is_running() else 0):
            lst = list(pending)
            if w.is_running():
                lst = [lst[0]] + lst[1:del_idx + 1] + lst[del_idx + 2:]
            else:
                lst = lst[:del_idx] + lst[del_idx + 1:]
            pending.clear()
            pending.extend(lst)
        self._refresh()


class JitterBatchDialog(QDialog):
    """Run jitter over sessions x conn-types x segments, optionally limited to pair groups."""

    NJITTERS = (100, 200, 500, 1000, 2000)

    def __init__(self, nav, parent=None):
        super().__init__(parent)
        self.nav = nav
        self.setWindowTitle("Run Jitter Batch")
        self.resize(520, 300)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Results key on (session, conn-type, segment); running another "
                             "group later merges into the same store."))

        self._sess = self._picker(lay, "Sessions", self._sessions(), "sessions",
                                  [nav.current_session_str])
        self._ct   = self._picker(lay, "Conn types", self._conn_types(), "conn types",
                                  [nav.key.type_label()])
        self._seg  = self._picker(lay, "Segments", nav.cd.available_segments(nav.key),
                                  "segments", [_ALL_SEGS])
        self._grp  = self._picker(lay, "Groups", nav.sd.groups.pickable_groups(), "groups", [],
                                  select_all_when_empty=False, display=group_display)
        lay.addWidget(QLabel("No group selected = every screened pair."))

        row = QHBoxLayout()
        row.addWidget(QLabel("njitter:"))
        self._nj = QComboBox()
        self._nj.addItems([str(n) for n in self.NJITTERS])
        self._nj.setCurrentText(str(DEFAULT_NJITTER))
        row.addWidget(self._nj)
        self._hi = QCheckBox("High resolution")
        row.addWidget(self._hi)
        row.addStretch()
        lay.addLayout(row)

        btns = QHBoxLayout()
        run = QPushButton("Run"); run.clicked.connect(self._run_btn)
        close = QPushButton("Close"); close.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(run); btns.addWidget(close)
        lay.addLayout(btns)

    def _picker(self, lay, title: str, items: list, plural: str, selected: list,
                select_all_when_empty: bool = True, display=None) -> ListPickerButton:
        """A batch has no row above it, so nothing to follow: followable stays off."""
        p = ListPickerButton(title, [str(i) for i in items], plural=plural,
                             select_all_when_empty=select_all_when_empty, display=display)
        if selected:
            p.set_selected([str(s) for s in selected])
        lay.addWidget(p)
        return p

    def _sessions(self) -> list:
        return sorted({str(k.session) for k in self.nav.cd.ptr})

    def _conn_types(self) -> list:
        return sorted({k.type_label() for k in self.nav.cd.ptr})

    def _ptr_keys(self) -> list:
        sessions, cts = set(self._sess.selected), set(self._ct.selected)
        return [k for k in sorted(self.nav.cd.ptr, key=str)
                if str(k.session) in sessions and k.type_label() in cts]

    def _jobs(self) -> list:
        """(ptr_key, segment, group) triples this run covers; group None = all screened."""
        groups = self._grp.selected or [None]
        return [(k, seg, g) for k in self._ptr_keys()
                for seg in self._seg.selected for g in groups]

    def _run_btn(self):
        nav, jobs = self.nav, self._jobs()
        if not jobs:
            QMessageBox.information(self, "Jitter", "Nothing selected.")
            return
        njitter = int(self._nj.currentText())
        res_key = 'hi' if self._hi.isChecked() else 'lo'
        dlg = QProgressDialog(f"Jittering {len(jobs)} block(s) x {njitter}…",
                              "Stop", 0, len(jobs), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.show()
        total, failed = 0, []
        for i, (key, seg, grp) in enumerate(jobs):
            if dlg.wasCanceled():
                break
            dlg.setLabelText(f"{key.session} {key.type_label()} {seg}"
                             + (f" [{grp}]" if grp else ""))
            dlg.setValue(i)
            QApplication.processEvents()
            try:
                total += run_and_store(nav.cd, nav.sd, key, seg, group=grp,
                                       njitter=njitter, res_key=res_key,
                                       progress=lambda s: print(f"[jitter] {s}", flush=True))
            except Exception as exc:
                failed.append(f"{key.session} {key.type_label()} {seg}: {exc}")
        dlg.close()
        lines = [f"Stored {total} pair result(s) over {len(jobs)} block(s).",
                 f"Saved to {nav.cd.jitter_path()}"]
        if failed:
            lines += ["", f"{len(failed)} block(s) failed:"] + [f"  {f}" for f in failed]
        ResultsDialog.show_report("Jitter Batch", "\n".join(lines))
