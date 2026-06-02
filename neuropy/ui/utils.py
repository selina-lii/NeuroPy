"""Shared UI utilities for NeuroPy.

GenericUI            — base class: undo/redo stack, Tk window setup, heartbeat.
BackgroundTaskRunner — single background process/thread + Tk polling queue.
LRUCache             — generic bounded LRU cache (max_size=1 = single-slot invalidation).
UITheme              — frozen dark/light color theme; set on GenericUI as self.theme.
WrapFrame            — tk.Frame that auto-wraps children horizontally like word-wrap.
ArrowScroller        — ◀/▶ arrow scroll nav for chip rows / plot columns.
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import multiprocessing as _mp
import os
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')

# Sentinel constants
_SEPARATOR_ROW = "__separator_row__"
_SPECIAL_PREFIX = "__special_"

is_separator_row = lambda e: isinstance(e, tuple) and bool(e) and e[0] == _SEPARATOR_ROW
is_special_group = lambda n: str(n).startswith(_SPECIAL_PREFIX)

def _admitted_pairs(sel_data) -> set[tuple]:
    return {p for p, t in sel_data._pair_tags.items() if t.get('admitted')}

def _admit_pair(sel_data, pair: tuple) -> None:
    sel_data._pair_tags.setdefault(pair, {})['admitted'] = True


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

    @classmethod
    def from_dark(cls, dark: bool) -> 'UITheme':
        if dark:
            return cls(True,  '#dddddd', '#aaaaaa', '#ffffff', '#1e1e1e', '#777777', '#dddddd')
        return cls(False, '#000000', '#666666', '#333333', 'white',   '#cccccc', '#333333')


class WrapFrame(tk.Frame):
    """tk.Frame that auto-wraps child widgets horizontally (like word-wrap).

    Usage: create, pack/grid the frame, then call .add(widget) for each child.
    """

    def __init__(self, parent, pad_x: int = 2, pad_y: int = 1, **kw):
        super().__init__(parent, **kw)
        self._pad_x = pad_x
        self._pad_y = pad_y
        self._items: list[tk.Widget] = []
        self.bind('<Configure>', lambda e: self.after_idle(self._rewrap))

    def add(self, widget: tk.Widget) -> tk.Widget:
        self._items.append(widget)
        return widget

    def _rewrap(self):
        self.update_idletasks()
        avail_w = self.winfo_width()
        if avail_w <= 1:
            self.after(100, self._rewrap)
            return
        x, y, row_h = self._pad_x, self._pad_y, 0
        for w in self._items:
            if not w.winfo_exists():
                continue
            w.update_idletasks()
            ww, wh = w.winfo_reqwidth(), w.winfo_reqheight()
            if x + ww > avail_w - self._pad_x and x > self._pad_x:
                y += row_h + self._pad_y
                x, row_h = self._pad_x, 0
            w.place(x=x, y=y)
            x += ww + self._pad_x
            row_h = max(row_h, wh)
        self.configure(height=max(y + row_h + self._pad_y * 2, 22))


class ArrowScroller:
    """Manages horizontal scroll offset + ◀/▶ arrow labels for a chip/plot row.

    Create once; call install() on each rebuild (frame is cleared each time).
    """

    def __init__(self):
        self.offset: int = 0

    def install(self, frame: tk.Widget, n_total: int,
                on_change: Callable[[int], None],
                font_size: int = 9, dark: bool = False) -> tuple:
        """Add ◀ (LEFT) and ▶ (RIGHT) labels to *frame*.

        Arrows dim at boundaries. Calls on_change(new_offset) on click.
        Returns (btn_l, btn_r).
        """
        self.offset = max(0, min(self.offset, max(0, n_total - 1)))
        dim    = '#777777' if dark else '#cccccc'
        active = '#dddddd' if dark else '#333333'
        fs = ('Arial', font_size)

        btn_r = tk.Label(frame, text='▶', font=fs, cursor='hand2',
                         padx=3, fg=active if self.offset < n_total - 1 else dim)
        btn_r.pack(side=tk.RIGHT, padx=(0, 2))
        btn_r.bind('<Button-1>', lambda e: self._click(1, n_total, on_change))

        btn_l = tk.Label(frame, text='◀', font=fs, cursor='hand2',
                         padx=3, fg=active if self.offset > 0 else dim)
        btn_l.pack(side=tk.LEFT, padx=(2, 2))
        btn_l.bind('<Button-1>', lambda e: self._click(-1, n_total, on_change))

        return btn_l, btn_r

    def _click(self, direction: int, n_total: int, on_change: Callable[[int], None]):
        new = max(0, min(self.offset + direction, max(0, n_total - 1)))
        if new != self.offset:
            self.offset = new
            on_change(new)


class GenericUI:

    def __init__(self):
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._UNDO_LIMIT = 30

    def _setup_window(self):
        self._owns_mainloop = False
        try:
            existing = tk._default_root  # noqa: access internal
        except AttributeError:
            existing = None
        if existing is not None and existing.winfo_exists():
            self.root = tk.Toplevel(existing)
        else:
            self.root = tk.Tk()
            self._owns_mainloop = True
        self.root.title("CCG Manual Review")
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._set_window_icon()
        try:
            _style = ttk.Style(self.root)
            _raw_bg = _style.lookup('TFrame', 'background') or self.root.cget('bg')
            _rgb = self.root.winfo_rgb(_raw_bg)
            lum = (0.299 * _rgb[0] + 0.587 * _rgb[1] + 0.114 * _rgb[2]) / 65535
            self._dark = lum < 0.4
            print(f"[UITheme] bg={_raw_bg!r} rgb={_rgb} lum={lum:.3f} → {'dark' if self._dark else 'light'}")
        except Exception:
            self._dark = False
        if self._dark:
            print("[UITheme] using dark mode")
        self.theme = UITheme.from_dark(self._dark)
        self._heartbeat_id = None
        self._closing = False
        self._start_heartbeat()

    def _start_heartbeat(self):
        """Periodic no-op to keep the Tk event loop alive in Jupyter."""
        def _beat():
            if self._closing:
                return
            try:
                if self.root.winfo_exists():
                    self._heartbeat_id = self.root.after(2000, _beat)
            except Exception:
                pass
        self._heartbeat_id = self.root.after(2000, _beat)
        self.root.bind('<Destroy>', self._on_root_destroy, '+')

    def _on_root_destroy(self, event):
        if getattr(event, 'widget', None) is not self.root:
            return
        self._closing = True
        if self._heartbeat_id is not None:
            try:
                self.root.after_cancel(self._heartbeat_id)
            except Exception:
                pass
            self._heartbeat_id = None


class BackgroundTaskRunner:
    """Background process/thread queue with Tk polling.

    Handles one running task at a time. Supports:
    - subprocess.Popen   (fire-and-forget; max_queue=1, use_result_queue=False)
    - mp.Process         (returns data via mp.Queue; use_result_queue=True)
    - threading.Thread   (result written to shared state by caller; use_result_queue=False)

    Parameters
    ----------
    max_queue : int
        Maximum number of pending tasks (including the running one).
    use_result_queue : bool
        If True, mp.Queue passed to launch_fn; poll() → (task, result_dict).
        If False, poll() → (task, None).
    """

    def __init__(self, *, max_queue: int = 1, use_result_queue: bool = False):
        self._max_queue = max_queue
        self._use_result_queue = use_result_queue
        self._pending: collections.deque = collections.deque()
        self._proc = None            # Popen, mp.Process, or threading.Thread
        self._result_queue = None    # mp.Queue | None
        self._poll_id = None         # root.after handle
        self._root = None

    # ── Process state ──────────────────────────────────────────────────────

    def is_running(self) -> bool:
        if self._proc is None:
            return False
        if hasattr(self._proc, 'is_alive'):
            return self._proc.is_alive()
        return self._proc.poll() is None  # Popen

    def pending_count(self) -> int:
        # _pending[0] stays in the deque until poll() pops it
        return len(self._pending)

    # ── Enqueue / launch ───────────────────────────────────────────────────

    def enqueue(self, task) -> bool:
        if len(self._pending) >= self._max_queue:
            return False
        self._pending.append(task)
        return True

    def start_next(self, launch_fn) -> bool:
        """Launch the next pending task via launch_fn(task, queue|None) → proc|None.

        Return an unstarted mp.Process (start_next calls .start()), an already-running
        Popen/Thread, or None to skip the task.  Returns True if a task was launched.
        """
        if self.is_running() or not self._pending:
            return False
        task = self._pending[0]
        q = _mp.Queue() if self._use_result_queue else None
        proc = launch_fn(task, q)
        if proc is None:
            # Caller signalled skip — drop task and try the next one
            self._pending.popleft()
            return self.start_next(launch_fn)
        self._result_queue = q
        self._proc = proc
        # mp.Process must be started; Popen / Thread are already running
        if isinstance(proc, _mp.Process):
            proc.start()
        return True

    # ── Polling ────────────────────────────────────────────────────────────

    def start_polling(self, root: tk.Tk, *, interval_ms: int, on_done) -> None:
        self._root = root
        self._poll_id = root.after(interval_ms, self._poll, interval_ms, on_done)

    def poll(self) -> tuple:
        if self.is_running():
            return None, None
        completed = self._pending.popleft() if self._pending else None
        result = None
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

    def _poll(self, interval_ms: int, on_done):
        self._poll_id = None
        if self._proc is None:
            return
        if self.is_running():
            self._poll_id = self._root.after(interval_ms, self._poll, interval_ms, on_done)
        else:
            task, result = self.poll()
            on_done(task, result)

    # ── Terminate ──────────────────────────────────────────────────────────

    def terminate(self) -> None:
        """Cancel the polling loop and terminate the running process.

        For threading.Thread targets, terminate is a no-op on the thread
        itself; only the poll handle is cancelled so results are discarded.
        """
        if self._poll_id is not None:
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
        self._proc = None
        self._result_queue = None


class LRUCache(Generic[K, V]):
    """Generic bounded LRU cache backed by OrderedDict.

    Use max_size=1 as a single-slot invalidation cache (always keeps the
    most recently written entry).  Use max_size=N for bounded LRU eviction.

    Examples
    --------
    >>> c = LRUCache(max_size=3)
    >>> c.put('a', 1); c.put('b', 2); c.put('c', 3)
    >>> c.get('a')   # promotes 'a'
    1
    >>> c.put('d', 4)   # evicts 'b' (LRU)
    >>> 'b' in c
    False
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max_size = max_size

    # ── Read ───────────────────────────────────────────────────────────────

    def get(self, key: K) -> 'V | None':
        """Return cached value for *key*, or None. Promotes key to MRU on hit."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def __contains__(self, key) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self):
        return iter(self._cache)  # MRU-last order

    # ── Write ──────────────────────────────────────────────────────────────

    def put(self, key: K, value: V) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def pop(self, key: K, default=None):
        return self._cache.pop(key, default)

    def clear(self) -> None:
        self._cache.clear()


def intersect_intervals(a, b):
    """Two-pointer O(n+m) intersection of two sorted interval lists."""
    res, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            res.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return res


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
    for _, size, path in sorted(files):          # oldest-first
        if total <= limit:
            break
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass


def purge_png_dir(directory: str, days: int = 3, max_gb: float = 1.5) -> None:
    purge_dir(directory, suffix='.png', days=days, max_gb=max_gb)


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


def get_png_filename(
    ref: int,
    tgt: int,
    seg_name: str,
    norm_key: str,
    alpha_key: str,
    res_key: str,
    scale_key: str = '',
    j_key: str = '',
    ext_key: str = '',
    dk_key: str = '',
    sess_prefix: str = '',
) -> str:
    return (
        f"{sess_prefix}pair_{ref}_{tgt}_{seg_name}_{norm_key}"
        f"{alpha_key}{res_key}{scale_key}{j_key}{ext_key}{dk_key}.png"
    )
