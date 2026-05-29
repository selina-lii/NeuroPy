"""
Jitter UI glue for the CCG Review UI.

JitterWorker     — low-level process-pool / cache management (no Tk).
JitterController — high-level orchestration; holds a back-ref to CCGReviewUI.

CCGReviewUI creates one JitterController and delegates all jitter operations to it.
"""
from __future__ import annotations

import multiprocessing as _mp
from typing import TYPE_CHECKING

import tkinter as tk
from tkinter import messagebox

from neuropy.analyses._jitter_worker import jitter_worker
from neuropy.analyses.jitter import JitterTask
from neuropy.ui.utils import BackgroundTaskRunner, LRUCache

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

# maximum queued jitter tasks (running + pending)
_MAX_JITTER_QUEUE = 50

_ALL_SEGS = "All"

# ---------------------------------------------------------------------------
# JitterWorker
# ---------------------------------------------------------------------------

class JitterWorker:
    """Manages on-demand jitter computation queue, process lifecycle, and in-memory cache.

    Owns no Tk objects.  JitterController provides the Tk event-loop integration
    (root.after polling) and handles persistence to cd._jitter_results.

    Process/queue/deque lifecycle is delegated to BackgroundTaskRunner.

    Cache key: (ref: int, tgt: int, res_key: str, seg_arg: int | None)
    Cache value: (j_avg, j_pval, j_pval_bins, j_lo, j_hi)
    """

    CACHE_MAX = 500

    UNVIEWED_BG = '#FFEE58'
    UNVIEWED_FG = '#333333'
    VIEWED_BG   = '#FFF9C4'
    VIEWED_FG   = '#333333'

    def __init__(self):
        self._runner = BackgroundTaskRunner(
            max_queue=_MAX_JITTER_QUEUE, use_result_queue=True)
        self._cache: LRUCache = LRUCache(self.CACHE_MAX)
        self.unviewed: set = set()

    def enqueue(self, tag: str, ref: int, tgt: int, njitter: int,
                res_key: str, bin_size_eff: float,
                seg_arg=None, t0=None, t1=None, nd_key=None):
        self._runner.enqueue(
            JitterTask(tag, ref, tgt, njitter, res_key, bin_size_eff, seg_arg, t0, t1,
                       nd_key))

    def is_running(self) -> bool:
        return self._runner.is_running()

    def start_next(self, key, neurons, ccg_data_lo, ccg_data_hi, edge_times) -> bool:
        def _launch(task, q):
            ccg_data = ccg_data_hi if task.res_key == 'hi' else ccg_data_lo
            if ccg_data is None:
                return None  # signals skip to BackgroundTaskRunner
            return _mp.Process(
                target=jitter_worker,
                args=(q, key, neurons, ccg_data, edge_times,
                      task.ref, task.tgt, task.njitter, task.bin_size_eff),
                kwargs={'segment': task.seg_arg, 't0': task.t0, 't1': task.t1},
                daemon=True,
            )
        return self._runner.start_next(_launch)

    def cache_clear(self, ref: int, tgt: int, seg_key):
        for res_key in ('lo', 'hi'):
            self._cache.pop((ref, tgt, res_key, seg_key))


# ---------------------------------------------------------------------------
# JitterController
# ---------------------------------------------------------------------------

class JitterController:
    """Orchestrates jitter computation lifecycle for CCGReviewUI.

    CCGReviewUI creates one instance and delegates all jitter operations here.
    """

    def __init__(self, ui: 'CCGReviewUI'):
        self._ui = ui
        self.jitter_worker = JitterWorker()
        self._poll_id = None  # root.after() id — Tk, not shared

    # ------------------------------------------------------------------
    # Task scheduling
    # ------------------------------------------------------------------

    def on_run_jitter(self):
        ui = self._ui
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        if ui.neurons is None:
            messagebox.showerror("Jitter", "No neuron data attached.")
            return
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        panel = ui.center_container.jitter_panel
        njitter = int(panel._njitter.get())
        run_lo  = bool(panel._jitter_run_lo.get())
        run_hi  = bool(panel._jitter_run_hi.get())
        if not run_lo and not run_hi:
            messagebox.showwarning("Jitter", "Select at least one resolution (lo and/or hi).")
            return
        total = ((1 if self.jitter_worker.is_running() else 0)
                 + self.jitter_worker._runner.pending_count())
        if total >= _MAX_JITTER_QUEUE:
            messagebox.showwarning(
                "Jitter", f"Queue full ({total}/{_MAX_JITTER_QUEUE}).\n"
                          "Wait for running jitters to complete.")
            return
        seg_arg = self.seg()
        if seg_arg is not None:
            et = ui.ccg_ptr.edge_times
            jitter_t0 = float(et.iloc[seg_arg]['start'])
            jitter_t1 = float(et.iloc[seg_arg]['stop'])
        else:
            jitter_t0, jitter_t1 = None, None
        # In Any mode the current pair may belong to a different session than
        # ui.key; look up the pair's session key from the handle list.
        nd_key = ui.key.nd()
        if getattr(ui, '_session_any_mode', False):
            hl = getattr(ui, '_any_pair_handle_list', None) or []
            if ui.current_pair_idx < len(hl):
                nd_key = hl[ui.current_pair_idx][0].nd()
        lo_ccg = ui.cd._ccg.get(nd_key) if hasattr(ui.cd, '_ccg') else ui.ccg_data
        hi_ccg = ui.cd._ccg_highres.get(nd_key) if hasattr(ui.cd, '_ccg_highres') else None
        enqueued_res_keys = []
        for res_key, ccg_data, should_run in [('lo', lo_ccg, run_lo), ('hi', hi_ccg, run_hi)]:
            if not should_run:
                continue
            if ccg_data is None:
                if res_key == 'hi':
                    messagebox.showwarning(
                        "Jitter", "High-res CCG not loaded; cannot run hi jitter.")
                continue
            n = ccg_data.ccg.shape[-1]
            bin_size_eff = ccg_data.conf.duration / (n - 1) if n > 1 else ccg_data.conf.bin_size
            self.jitter_worker.enqueue('jitter', ref, tgt, njitter, res_key, bin_size_eff,
                                       seg_arg, jitter_t0, jitter_t1, nd_key)
            enqueued_res_keys.append(res_key)
        ui._dbg_log("H3", "jitter.py:on_run_jitter:enqueue", "Enqueued jitter task(s)", {
            "pair": [int(ref), int(tgt)],
            "res_keys": enqueued_res_keys,
            "seg_arg": seg_arg,
            "njitter": int(njitter),
            "current_segment": int(ui.current_segment),
        })
        self.update_btn_text()
        self.start_next()

    def is_task_running(self) -> bool:
        return self.jitter_worker.is_running()

    def start_next(self):
        ui = self._ui
        if not self.jitter_worker._runner.pending_count():
            self.update_btn_text()
            return
        task = self.jitter_worker._runner._pending[0]
        nd_key = task.nd_key if task.nd_key is not None else ui.key.nd()
        ccg_data_lo = (ui.cd._ccg.get(nd_key)
                       if hasattr(ui.cd, '_ccg') else ui.ccg_data)
        ccg_data_hi = (ui.cd._ccg_highres.get(nd_key)
                       if hasattr(ui.cd, '_ccg_highres') else None)
        started = self.jitter_worker.start_next(
            ui.key, ui.neurons, ccg_data_lo, ccg_data_hi,
            ui.ccg_ptr.edge_times)
        self.update_btn_text()
        if started and self._poll_id is None:
            self._poll_id = ui.root.after(300, self.poll)

    def _set_jitter_btn(self, text=''):
        self._ui.center_container.jitter_panel._jitter_btn_text.set(text or 'Run Jitter')

    def update_btn_text(self):
        ui = self._ui
        running = self.jitter_worker.is_running()
        pending = self.jitter_worker._runner._pending
        queued = len(pending)
        if running and pending:
            task = pending[0]
            ref, tgt = task.ref, task.tgt
            seg_arg = task.seg_arg
            if seg_arg is None:
                seg_name = _ALL_SEGS
            else:
                try:
                    seg_name = str(ui.ccg_ptr.segment_names[int(seg_arg)])
                except Exception:
                    seg_name = f"seg{seg_arg}"
            label = f"Jitter [{ref},{tgt}] {seg_name}…"
            suffix = f" +{queued - 1} queued" if queued > 1 else ''
            self._set_jitter_btn(f"{label}{suffix}")
        else:
            self._set_jitter_btn()

    def poll(self):
        ui = self._ui
        if self.jitter_worker.is_running():
            self._poll_id = ui.root.after(300, self.poll)
            return
        self._poll_id = None
        completed, result = self.jitter_worker._runner.poll()
        if result is not None and not result.get('error') and result.get('j_avg') is not None:
            res_key = completed.res_key if completed is not None else 'lo'
            seg_arg = completed.seg_arg if completed is not None else None
            cache_key = (result['ref'], result['tgt'], res_key, seg_arg)
            jitter_val = (
                result.get('j_avg'),
                result.get('j_pval'),
                result.get('j_pval_bins'),
                result.get('j_lo'),
                result.get('j_hi'),
            )
            self.jitter_worker._cache.put(cache_key, jitter_val)
            nd_key = ui.key.nd()
            if hasattr(ui.cd, '_jitter_results'):
                if nd_key not in ui.cd._jitter_results:
                    ui.cd._jitter_results[nd_key] = {}
                ui.cd._jitter_results[nd_key][cache_key] = jitter_val
            completed_pair = (result['ref'], result['tgt'])
            self.jitter_worker.unviewed.add(completed_pair)
            self.apply_list_colors(pair=completed_pair)
            if self.mark_viewed():
                ui._update_jitter_sig_buttons()
                ui.update_plot()
            ui._update_conn_str_metric_availability()
            ui.root.bell()
        elif result is not None and result.get('error'):
            messagebox.showerror("Jitter", f"Jitter failed:\n{result['error']}")
        self.start_next()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def on_save(self):
        if not hasattr(self._ui.cd, 'save_jitter'):
            messagebox.showerror(
                "Save Jitter",
                "CCGDataset does not support jitter persistence.")
            return
        try:
            self._ui.cd.save_jitter()
            total = sum(len(v) for v in self._ui.cd._jitter_results.values())
            messagebox.showinfo("Save Jitter", f"Saved {total} pair(s).")
        except Exception as exc:
            messagebox.showerror("Save Jitter", f"Save failed:\n{exc}")

    def load_from_cd(self):
        ui = self._ui
        if not hasattr(ui.cd, '_jitter_results'):
            return
        nd_key = ui.key.nd()
        pairs = ui.cd._jitter_results.get(nd_key, {})
        for cache_key, val in pairs.items():
            if len(cache_key) == 3:
                cache_key = cache_key + (None,)
            self.jitter_worker._cache.put(cache_key, val)
        if pairs:
            ui._update_jitter_sig_buttons()
            self.apply_list_colors()

    # ------------------------------------------------------------------
    # Segment / cache helpers
    # ------------------------------------------------------------------

    def seg(self, seg=None) -> int | None:
        """Return segment key for jitter cache: None for All/custom, int for real segments."""
        if seg is None:
            seg = self._ui.current_segment
        if seg == self._ui.n_segments or self._ui._is_custom_segment(seg):
            return None
        return int(seg)

    def on_clear(self):
        ui = self._ui
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        segk = self.seg()
        for rk in ('lo', 'hi'):
            self.jitter_worker._cache.pop((ref, tgt, rk, segk), None)
        if hasattr(ui.cd, '_jitter_results'):
            nd_key = ui.key.nd()
            if nd_key in ui.cd._jitter_results:
                for rk in ('lo', 'hi'):
                    ui.cd._jitter_results[nd_key].pop((ref, tgt, rk, segk), None)
        self.jitter_worker.unviewed.discard((ref, tgt))
        ui._clear_all_png_cache()
        ui._update_jitter_sig_buttons()
        self.apply_list_colors()
        ui.update_plot()

    # ------------------------------------------------------------------
    # List coloring
    # ------------------------------------------------------------------

    def pair_coords(self, inds) -> tuple[int, int] | None:
        """Return (ref, tgt) for jitter coloring from a selection triple or pair."""
        if not isinstance(inds, tuple) or len(inds) < 2:
            return None
        try:
            if len(inds) >= 3 and isinstance(inds[0], str):
                return int(inds[1]), int(inds[2])
            return int(inds[0]), int(inds[1])
        except (TypeError, ValueError):
            return None

    def apply_list_colors(self, pair=None):
        """Color pair list items based on jitter cache state."""
        ui = self._ui
        jitter_cache = self.jitter_worker._cache
        jitter_unviewed = self.jitter_worker.unviewed

        def _apply_row(listbox, idx, ref, tgt):
            p = (int(ref), int(tgt))
            if pair is not None and p != pair:
                return False
            has_any_res = any(k[0] == p[0] and k[1] == p[1] for k in jitter_cache)
            try:
                if p in jitter_unviewed:
                    listbox.itemconfig(idx,
                                       background=JitterWorker.UNVIEWED_BG,
                                       foreground=JitterWorker.UNVIEWED_FG)
                elif has_any_res:
                    listbox.itemconfig(idx,
                                       background=JitterWorker.VIEWED_BG,
                                       foreground=JitterWorker.VIEWED_FG)
                else:
                    listbox.itemconfig(idx, background='', foreground='')
            except tk.TclError:
                return False
            return True

        if not hasattr(ui, 'unselected_list'):
            return
        if getattr(ui, '_session_any_mode', False):
            try:
                n_items = int(ui.selected_list.size())
            except Exception:
                n_items = 0
            _lp = getattr(getattr(ui, 'left_container', None), 'left_panel', None)
            sel_map = getattr(_lp, '_sel_list_pairs', None) or []
            for idx, entry in enumerate(sel_map):
                if idx >= n_items or entry is None:
                    continue
                rt = self.pair_coords(entry)
                if rt is None:
                    continue
                if _apply_row(ui.selected_list, idx, *rt) and pair is not None:
                    ui._reapply_bookmark_list_styles()
                    return
            ui._reapply_bookmark_list_styles()
            return
        for listbox, inds_set in [(ui.unselected_list, ui.unselected_inds),
                                  (ui.selected_list, ui.selected_inds)]:
            sorted_items = sorted(inds_set)
            try:
                n_items = int(listbox.size())
            except Exception:
                n_items = 0
            for idx, inds in enumerate(sorted_items):
                if idx >= n_items:
                    break
                rt = self.pair_coords(inds)
                if rt is None:
                    continue
                if _apply_row(listbox, idx, *rt) and pair is not None:
                    ui._reapply_bookmark_list_styles()
                    return
        ui._reapply_bookmark_list_styles()

    def mark_viewed(self) -> bool:
        """Mark current pair's jitter as viewed; auto-enable overlay if available.

        Returns True if a pair was newly marked as viewed.
        Does NOT call update_plot() — the caller is responsible for that.
        """
        ui = self._ui
        if ui.current_pair_idx >= len(ui.all_inds):
            return False
        inds = ui.all_inds[ui.current_pair_idx]
        pair = (int(inds[0]), int(inds[1]))
        if pair in self.jitter_worker.unviewed:
            self.jitter_worker.unviewed.discard(pair)
            self.apply_list_colors(pair=pair)
            seg_key = self.seg()
            if self.jitter_worker._cache.get(
                    (pair[0], pair[1], 'lo', seg_key)) is not None:
                ui.center_container.baseline_panel._conn_str_method.set('jitter')
                ui.center_container.baseline_panel._sig_jitter_pc.set(True)
                ui.center_container.correlogram_panel._line_jitter.set(False)
            return True
        return False

class JitterQueueDialog:
    """Dialog showing all queued and running jitter/custom-CCG tasks."""

    @classmethod
    def show(cls, ui: "CCGReviewUI") -> None:
        cls(ui).win.wait_window()

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self.win = tk.Toplevel(ui.root)
        self.win.title("Jitter Queue")
        self.win.geometry("460x340")
        self._build()

    def _build(self):
        ui = self._ui
        win = self.win
        from tkinter import ttk
        frame = ttk.Frame(win, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Queued tasks",
                  font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        lb = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                        selectmode=tk.EXTENDED, font=('TkFixedFont', 10))
        scrollbar.config(command=lb.yview)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _refresh():
            lb.delete(0, tk.END)
            for i, task in enumerate(ui._jitter_pending):
                seg_s = f" seg{task.seg_arg}" if task.seg_arg is not None else ""
                status = "▶ RUNNING" if i == 0 and ui._is_task_running() else "  queued"
                lb.insert(tk.END, f"{status}  jitter [{task.ref},{task.tgt}] n={task.njitter} {task.res_key}{seg_s}")
            for i, task in enumerate(ui._custom_ccg_pending):
                name = (task.get('name') if isinstance(task, dict) else task[3])
                status = "▶ RUNNING" if i == 0 and ui._custom_ccg_is_running() else "  queued"
                lb.insert(tk.END, f"{status}  custom CCG '{name}'")
            if lb.size() == 0:
                lb.insert(tk.END, "  (empty)")

        def _delete_selected():
            sel = lb.curselection()
            if not sel:
                return
            n_jitter = len(ui._jitter_pending)
            running_jitter = ui._is_task_running()
            running_ccg = ui._custom_ccg_is_running()
            jitter_to_remove = []
            ccg_to_remove = []
            for s in sel:
                if s < n_jitter:
                    if s == 0 and running_jitter:
                        continue
                    jitter_to_remove.append(s)
                else:
                    ccg_idx = s - n_jitter
                    if ccg_idx == 0 and running_ccg:
                        continue
                    ccg_to_remove.append(ccg_idx)
            if jitter_to_remove:
                pending = list(ui._jitter_pending)
                for idx in sorted(jitter_to_remove, reverse=True):
                    pending.pop(idx)
                ui._jitter_pending.clear()
                ui._jitter_pending.extend(pending)
                ui._update_jitter_btn_text()
            if ccg_to_remove:
                pending = list(ui._custom_ccg_pending)
                for idx in sorted(ccg_to_remove, reverse=True):
                    removed = pending.pop(idx)
                    ui._on_split_batch_task_done(removed)
                ui._custom_ccg_pending.clear()
                ui._custom_ccg_pending.extend(pending)
            _refresh()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_frame, text="Delete selected",
                   command=_delete_selected).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Refresh",
                   command=_refresh).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Close",
                   command=win.destroy).pack(side=tk.RIGHT)
        _refresh()
