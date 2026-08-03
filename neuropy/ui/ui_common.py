"""Toolkit-agnostic UI utilities — no tkinter or Qt imports.

Importable from both tkinter (ccg_ui.py) and Qt panels without creating
a cross-stack dependency.

Contents
--------
Sentinels        — _SEPARATOR_ROW, _SPECIAL_PREFIX, is_separator_row, is_special_group
UITheme          — frozen dark/light color palette (color strings only)
LRUCache         — generic bounded LRU cache
SelectionCommand — dataclass recording one undoable pair/group action
BiIndex          — bidirectional multimap: A ↔ set(B), both O(1)
CollapseState    — tracks which named sections are collapsed
group_header_label — formats a collapsible group header string
json_numpy_default — JSON encoder hook for numpy scalar types
purge_dir        — delete old/large files in a directory
fit_axes_to_data — set matplotlib axis limits with padding (matplotlib ax, not tk)
intersect_intervals — re-export from epoch_filter (two-pointer O(n+m))
_admitted_pairs / _admit_pair — sel_data pair admission helpers
BackgroundTaskRunner — single background process/thread + polling queue
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import multiprocessing as _mp
import os
import threading
from typing import Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')



_SEPARATOR_ROW   = "__separator_row__"

is_separator_row = lambda e: isinstance(e, tuple) and bool(e) and e[0] == _SEPARATOR_ROW

from neuropy.analyses.utils import _SPECIAL_PREFIX, is_special_group  # noqa: F401



@dataclasses.dataclass(frozen=True)
class UITheme:
    """Frozen dark/light color palette. Access via ui.theme.*"""
    dark: bool
    fg: str        # main text
    fg_muted: str  # secondary text
    fg_header: str # headers
    bg: str        # background
    dim: str       # inactive arrows / labels
    active: str    # active arrows / labels
    plot_bg: str   # pyqtgraph / matplotlib plot face
    neuron_grays: tuple  # per-channel neuron fill RGB triples, light→dark

    @classmethod
    def from_dark(cls, dark: bool) -> 'UITheme':
        if dark:
            return cls(
                True,
                '#dddddd', '#aaaaaa', '#ffffff', '#1e1e1e', '#777777', '#dddddd',
                '#1e1e1e',
                ((255, 255, 255), (230, 230, 230), (204, 204, 204),
                 (178, 178, 178), (153, 153, 153)),
            )
        return cls(
            False,
            '#000000', '#666666', '#333333', 'white', '#cccccc', '#333333',
            'white',
            ((0, 0, 0), (51, 51, 51), (102, 102, 102),
             (153, 153, 153), (187, 187, 187)),
        )


def qt_dark_mode() -> bool:
    """True when the active Qt palette uses a dark window background."""
    try:
        from pyqtgraph.Qt.QtWidgets import QApplication
        from pyqtgraph.Qt.QtGui import QPalette
        app = QApplication.instance()
        if app is None:
            return False
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    except Exception:
        return False



from neuropy.utils.data_storage_util import LRUCache  # re-export for existing importers



@dataclasses.dataclass
class SelectionCommand:
    """Records what changed for one undoable action."""
    pair_changes: dict   # {pair_tuple: (old_state, new_state)}
    group_changes: list  # [(group_name, session, pair, 'add'|'remove')]



# BiIndex moved to analyses/utils.py (pure data, no Qt). Re-exported here so
# existing `from neuropy.ui.ui_common import BiIndex` call sites keep working.
from neuropy.analyses.utils import BiIndex  # noqa: F401 (re-export)



class CollapseState:
    """Tracks which named sections are collapsed. UI-package independent."""

    def __init__(self, initial: set | None = None):
        self._collapsed: set = set(initial or [])

    def is_collapsed(self, name: str) -> bool:
        return name in self._collapsed

    def toggle(self, name: str) -> None:
        self._collapsed.discard(name) if name in self._collapsed else self._collapsed.add(name)

    def collapse_all(self, names) -> None:
        self._collapsed = set(names)

    def expand_all(self) -> None:
        self._collapsed.clear()

    def as_set(self) -> set:
        return set(self._collapsed)



def group_header_label(name: str, count: int, collapsed: bool) -> str:
    return f"── {name} ({count}) ──" + (" >>" if collapsed else "")



from neuropy.utils.data_storage_util import atomic_write_json  # noqa: F401


def group_names_for_pair(data, ui, inds) -> list:
    """Sorted group names containing this pair in the current session."""
    k = ui.key_for_pair(inds)
    return data.group_names_for_pair(k.session, (k.ref, k.tgt), ui.groups)


def pair_label(inds, *, bookmarked=False, group_names=None, pair_tags=None,
               any_mode=False) -> str:
    """Format a pair row label for Available / Selected lists."""
    ref, tgt = int(inds[-2]), int(inds[-1])
    label = f"[{ref}, {tgt}]"
    if any_mode and len(inds) >= 3:
        sess = getattr(inds[0], 'session', inds[0])
        label = f"{sess} {label}"
    if bookmarked:
        label = "★ " + label
    pt = pair_tags or {}
    names = set(group_names or [])   # live from Groups; pt['groups'] is a stale save-time snapshot
    tags = [f"[{g}]" for g in sorted(names)]
    for k, v in sorted(pt.items()):
        if k in ('groups', 'admitted', 'notes') or not v:
            continue
        tags.append(f"[{k}]")
    if tags:
        label += " " + " ".join(tags)
    return label



def json_numpy_default(obj):
    """JSON encoder for numpy scalar types — pass as ``default=`` to json.dump."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')



def purge_dir(directory: str, suffix: str = '.png',
              days: int = 3, max_gb: float = 1.5) -> None:
    """Delete files matching *suffix* older than *days*; evict oldest until total < *max_gb* GB."""
    if not os.path.isdir(directory):
        return
    files: list[tuple[float, int, str]] = []
    for fn in os.listdir(directory):
        if not fn.endswith(suffix):
            continue
        path = os.path.join(directory, fn)
        try:
            st = os.stat(path)
            files.append((st.st_mtime, st.st_size, path))
        except OSError:
            pass
    cutoff = (datetime.datetime.now() -
              datetime.timedelta(days=max(0, int(days)))).timestamp()
    for mtime, _, path in files:
        if mtime < cutoff:
            try:
                os.remove(path)
            except OSError:
                pass
    files = [(m, s, p) for m, s, p in files if os.path.exists(p)]
    total = sum(s for _, s, _ in files)
    limit = int(max_gb * 1024 ** 3)
    for _, size, path in sorted(files):
        if total <= limit:
            break
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass



def fit_axes_to_data(ax, x_all, y_all, x_pad_frac=0.08, y_pad_frac=0.06, min_pad=20):
    if not x_all or not y_all:
        return None
    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)
    pad_x = max((x_max - x_min) * x_pad_frac, min_pad)
    pad_y = max((y_max - y_min) * y_pad_frac, min_pad)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    return x_min, x_max, y_min, y_max



from neuropy.core.intervals import IntervalOp as _IntervalOp
intersect_intervals = _IntervalOp.intersect



def _admitted_pairs(app_state) -> set[tuple]:
    return set(app_state._listed)

def _admit_pair(app_state, bucket, pair: tuple) -> None:
    pair = tuple(pair)
    app_state._listed.add(pair)
    if pair not in bucket.selected and pair not in bucket.deleted:
        bucket.unselected.add(pair)



class BackgroundTaskRunner:
    """Background process/thread queue with pluggable polling.

    Handles one running task at a time. Supports:
    - subprocess.Popen   (fire-and-forget; max_queue=1, use_result_queue=False)
    - mp.Process         (returns data via mp.Queue; use_result_queue=True)
    - threading.Thread   (result written to shared state by caller)

    Polling is caller-managed: call poll() periodically (e.g. via QTimer or
    root.after) and handle the returned (task, result) pair.
    """

    def __init__(self, *, max_queue: int = 1, use_result_queue: bool = False):
        self._max_queue        = max_queue
        self._use_result_queue = use_result_queue
        self._pending: collections.deque = collections.deque()
        self._proc             = None
        self._result_queue     = None
        self._poll_id          = None  # framework-specific handle; set by caller
        self._root             = None  # set only when start_polling() used
        self._qt_timer         = None  # set by start_polling_qt()


    def is_running(self) -> bool:
        if self._proc is None:
            return False
        if hasattr(self._proc, 'is_alive'):
            return self._proc.is_alive()
        return self._proc.poll() is None  # Popen

    def pending_count(self) -> int:
        return len(self._pending)


    def enqueue(self, task) -> bool:
        if len(self._pending) >= self._max_queue:
            return False
        self._pending.append(task)
        return True

    def start_next(self, launch_fn) -> bool:
        """Launch next pending task via launch_fn(task, queue|None) → proc|None."""
        if self.is_running() or not self._pending:
            return False
        task = self._pending[0]
        q    = _mp.Queue() if self._use_result_queue else None
        proc = launch_fn(task, q)
        if proc is None:
            self._pending.popleft()
            return self.start_next(launch_fn)
        self._result_queue = q
        self._proc         = proc
        if isinstance(proc, _mp.Process):
            proc.start()
        return True


    def poll(self) -> tuple:
        """Return (completed_task, result) when done; (None, None) if still running."""
        if self.is_running():
            return None, None
        completed = self._pending.popleft() if self._pending else None
        result    = None
        if self._use_result_queue and self._result_queue is not None:
            try:
                if not self._result_queue.empty():
                    result = self._result_queue.get_nowait()
            except Exception:
                pass
        if self._proc is not None:
            if hasattr(self._proc, 'join'):
                self._proc.join(timeout=1)
            self._proc = None
        self._result_queue = None
        return completed, result

    def start_polling_qt(self, interval_ms: int, on_done) -> None:
        """Qt path: use QTimer instead of root.after()."""
        from pyqtgraph.Qt.QtCore import QTimer
        if self._qt_timer is not None:
            self._qt_timer.stop()
        self._qt_timer = QTimer()
        self._qt_timer.setInterval(interval_ms)
        self._qt_timer.timeout.connect(lambda: self._qt_poll(on_done))
        self._qt_timer.start()

    def _qt_poll(self, on_done) -> None:
        if self._proc is None:
            if self._qt_timer: self._qt_timer.stop()
            return
        if not self.is_running():
            if self._qt_timer: self._qt_timer.stop()
            task, result = self.poll()
            on_done(task, result)

    def start_polling(self, root, *, interval_ms: int, on_done) -> None:
        """Tkinter convenience: poll via root.after(). Qt callers use QTimer directly."""
        self._root   = root
        self._poll_id = root.after(interval_ms, self._tk_poll, interval_ms, on_done)

    def _tk_poll(self, interval_ms: int, on_done):
        self._poll_id = None
        if self._proc is None:
            return
        if self.is_running():
            self._poll_id = self._root.after(interval_ms, self._tk_poll, interval_ms, on_done)
        else:
            task, result = self.poll()
            on_done(task, result)


    def terminate(self) -> None:
        if self._poll_id is not None and self._root is not None:
            try:
                self._root.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        if self._proc is not None and not isinstance(self._proc, threading.Thread):
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc         = None
        self._result_queue = None
