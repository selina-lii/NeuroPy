"""Custom CCG computation, segment management, and suggestion logic.

Manages the async CCG computation queue, per-session segment cache,
npz disk I/O, and spec/suggestion persistence for CCGReviewUI.
"""
from __future__ import annotations

import collections
import glob as _glob
import json
import os
import re
import shutil
import threading
import time as _time
import traceback
from pathlib import Path as _Path
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import numpy as np

from neuropy.analyses import custom_ccg as _custom_ccg_mod
from neuropy.analyses.utils import split_time_range as _split_time_range
from neuropy.ui.utils import BackgroundTaskRunner

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

_MAX_QUEUE = 50


class CustomCCGManager:
    """Custom CCG computation, segment management, and suggestion logic for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self._runner = BackgroundTaskRunner(max_queue=_MAX_QUEUE, use_result_queue=False)
        self._thread_result: list = []
        # Owned attrs (shims on CCGReviewUI delegate here)
        self._extend_cache: dict = {}
        self._by_session: dict = {}          # session_str -> list of custom segment dicts
        self._active_sess: str = ''
        self._stacked_segments: set = set()
        self._inventory_sig: tuple = ()
        self.cache_dir = str(_Path(__file__).resolve().parents[2] / "data" / "custom_ccg")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.suggestions_path = os.path.join(self.cache_dir, "suggested_custom_ccgs.json")
        # Backward-compat aliases on ui (shims delegate to self)
        ui._custom_ccg_pending = self._runner._pending
        self._bind_custom_segments_to_session(str(ui.key.session))

    def _restore_loaded_custom_ccgs_from_state(self):
        """Reload custom CCG .npz files listed in ui_state.json (additively)."""
        ui = self._ui
        paths = ui._ui_state_cache.get('loaded_custom_ccgs', []) or []
        if not paths:
            return
        session = str(ui.key.session)
        changed_active = False
        for p in paths:
            if not isinstance(p, str) or not p or not os.path.exists(p):
                continue
            try:
                base = os.path.basename(p)
                file_sess = base.split("__", 1)[0] if "__" in base else session
                cs = _custom_ccg_mod.load_custom_segment_from_npz(p)
                lst = self._by_session.setdefault(file_sess, [])
                self._upsert_custom_segment_by_name(lst, cs)
                if lst is self._active_list:
                    changed_active = True
            except Exception as ex:
                print(f"[CCGReviewUI] restore custom CCG failed: {p}: {ex}")
        if changed_active:
            try:
                ui._build_sig_chips()
                ui._update_segment_label()
                ui.update_plot()
            except Exception:
                pass

    def _custom_ccg_start_next(self):
        """Start the next queued custom CCG task if none is running."""
        ui = self._ui
        if self._runner.is_running() or not self._runner._pending:
            return
        task = self._runner._pending[0]
        t0 = float(task.get('t0', 0.0))
        t1 = float(task.get('t1', 0.0))
        name = str(task.get('name', 'custom'))
        intervals = task.get('intervals')
        active_duration = task.get('active_duration')
        filter_state = task.get('filter_state', {})
        key_for_task = task.get('key', ui.key)
        metadata = task.get('metadata', {})
        self._thread_result.clear()
        _t_start = _time.monotonic()
        nd_key = key_for_task.nd()
        ccg_data_obj = ui.cd._ccg.get(nd_key) if hasattr(ui.cd, '_ccg') else ui.ccg_data
        neurons_obj = (ui.cd.nd.data[nd_key]
                       if getattr(ui.cd, 'nd', None) is not None else None)
        if ccg_data_obj is None or neurons_obj is None:
            try:
                ui.cd.get_ccg()
            except Exception as ex:
                messagebox.showerror("Custom CCG", f"Session load failed for {key_for_task.session}:\n{ex}")
                try:
                    failed = ui._custom_ccg_pending.popleft()
                    ui._on_chunk_done(failed)
                except Exception:
                    pass
                self._custom_ccg_start_next()
                return
            ccg_data_obj = ui.cd._ccg.get(nd_key) if hasattr(ui.cd, '_ccg') else ui.ccg_data
            neurons_obj = (ui.cd.nd.data[nd_key]
                           if getattr(ui.cd, 'nd', None) is not None else None)
            if ccg_data_obj is None or neurons_obj is None:
                print(f"[CustomCCG] missing session data after load: {key_for_task.session}")
                try:
                    failed = ui._custom_ccg_pending.popleft()
                    ui._on_chunk_done(failed)
                except Exception:
                    pass
                self._custom_ccg_start_next()
                return

        def _ccg_worker(_t0=t0, _t1=t1, _name=name, _intervals=intervals,
                        _ad=active_duration, _fs=filter_state, _key=key_for_task,
                        _meta=metadata, _ccg_data=ccg_data_obj, _neurons=neurons_obj):
            try:
                neurons_override = (
                    ui.time_slider._apply_brain_state_intervals(_intervals, _t0, _t1, neurons_obj=_neurons)
                    if _intervals is not None else None)
                result = self._compute_custom_segment(
                    _t0, _t1, _name,
                    neurons_override=neurons_override, active_duration=_ad,
                    key_override=_key, neurons_obj=_neurons, ccg_data_obj=_ccg_data,
                    metadata=_meta)
                if result is not None:
                    result['filter_state'] = _fs
                    result['compute_sec'] = _time.monotonic() - _t_start
                    result['_task_session'] = str(_key.session)
                self._thread_result.append(
                    result if result is not None else {'error': 'compute returned None'})
            except Exception as ex:
                self._thread_result.append({'error': str(ex)})

        t = threading.Thread(target=_ccg_worker, daemon=True)
        t.start()
        self._runner._proc = t
        if self._runner._poll_id is None:
            self._runner.start_polling(
                ui.root, interval_ms=300,
                on_done=lambda _task, _result: self._on_custom_ccg_done())

    def _on_custom_ccg_done(self):
        """Called by BackgroundTaskRunner when the custom CCG thread exits."""
        ui = self._ui
        completed_task = self._runner._pending.popleft() if self._runner._pending else None
        result = self._thread_result[0] if self._thread_result else None
        self._thread_result.clear()

        if result is not None and not result.get('error'):
            if isinstance(completed_task, dict) and completed_task.get('auto_save'):
                key_for_save = completed_task.get('key', ui.key)
                _sess_save = str(key_for_save.session)
                self._purge_timestamped_custom_ccg_npz(_sess_save, str(result['name']))
                fname = self._ccg_cache_filename_for_key(result['name'], key_for_save)
                path = os.path.join(self.cache_dir, fname)
                _custom_ccg_mod.save_custom_segment_to_npz(result, path)
                result['src_path'] = path
                self._emit_inventory_event()
            should_load = (not isinstance(completed_task, dict)
                            or bool(completed_task.get('load_into_ui', True)))
            _tk_done = (completed_task.get('key', ui.key)
                        if isinstance(completed_task, dict) else ui.key)
            _lsess = str(result.get('_task_session', getattr(_tk_done, 'session', '')))
            _lst = self._by_session.setdefault(_lsess, [])
            idx, _did_append = self._upsert_custom_segment_by_name(_lst, result)
            if should_load and self._active_list is _lst:
                ui._build_sig_chips()
                ui.current_segment = ui.n_segments + 1 + idx
                ui._clamp_current_segment_for_session()
                ui._update_segment_label()
                ui.update_plot()
            if hasattr(ui, '_ts_status_var'):
                ui.time_slider._status_var.set(f"Done: {result.get('name', '')}")
            ui.root.bell()
        elif result is not None and result.get('error'):
            messagebox.showerror("Custom CCG", f"Computation failed:\n{result['error']}")

        if completed_task is not None:
            ui._on_chunk_done(completed_task)

        self._custom_ccg_start_next()

    def _custom_ccg_has_unsaved(self) -> bool:
        """True if any in-memory custom segment has no on-disk .npz (or file missing)."""
        for lst in self._by_session.values():
            for cs in lst:
                if not isinstance(cs, dict):
                    continue
                p = cs.get('src_path')
                if not p or not os.path.isfile(str(p)):
                    return True
        return False

    def _is_custom_segment(self, seg=None):
        if seg is None:
            seg = self._ui.current_segment
        return seg > self._ui.n_segments

    def _custom_seg_index(self, seg=None):
        """Return the index into _custom_segments for the given segment id."""
        if seg is None:
            seg = self._ui.current_segment
        return seg - self._ui.n_segments - 1

    def _remove_custom_segment(self, ci):
        """Remove a custom segment by its index in _custom_segments."""
        ui = self._ui
        if 0 <= ci < len(self._active_list):
            self._active_list.pop(ci)
            if ui.current_segment > ui.n_segments:
                ui.current_segment = min(ui.current_segment,
                                           ui._n_total_segments() - 1)
            ui._build_sig_chips()
            ui._update_segment_label()
            ui.update_plot()



    def _generate_suggested_custom_ccgs(self):
        ui = self._ui
        specs = self._load_custom_ccg_suggestions()
        if not specs:
            messagebox.showinfo(
                "Suggested custom CCGs",
                "No suggested custom CCG entries found. Use 'Refresh suggested custom CCGs' first.")
            return
        win = tk.Toplevel(ui.root)
        win.title("Suggested custom CCGs")
        win.geometry("620x360")
        win.grab_set()
        ttk.Label(win, text="Generate custom CCGs from availability list:").pack(
            anchor='w', padx=8, pady=(8, 4))
        lb = tk.Listbox(win, selectmode=tk.MULTIPLE, height=12)
        lb.pack(fill=tk.BOTH, expand=True, padx=8)

        def _fmt_t(v):
            if isinstance(v, str) and v.lower() in ('start', 'end'):
                return v
            try:
                return ui.time_slider._sec_to_hms(float(v))
            except Exception:
                return str(v)

        for i, spec in enumerate(specs):
            name = str(spec.get('name', 'custom'))
            t0 = _fmt_t(spec.get('t0', 0.0))
            t1 = _fmt_t(spec.get('t1', 0.0))
            scope = str(spec.get('scope', 'By session'))
            n_have = len(spec.get('sessions', []) or [])
            n_total = max(1, len(ui._real_nd_keys_ordered()))
            label = f"[{name} | {t0}-{t1}] for {'ALL' if scope == 'All' else scope} ({n_have}/{n_total})"
            lb.insert(tk.END, label)
            lb.select_set(i)

        def _run(selected_idxs):
            queued = 0
            for idx in selected_idxs:
                spec = specs[int(idx)]
                queued += self._queue_custom_ccg_for_spec(
                    spec, for_all=(str(spec.get('scope', '')).lower() == 'all'),
                    auto_save=True)
            if queued:
                ui.time_slider._status_var.set(f"Queued {queued} suggested custom CCG task(s)")
                self._custom_ccg_start_next()
            else:
                ui.time_slider._status_var.set("All suggested custom CCGs already exist")
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btns, text="Generate selected",
                   command=lambda: _run(lb.curselection())).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Generate all",
                   command=lambda: _run(range(len(specs)))).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)

    def _custom_npz_spec(self, path: str) -> dict | None:
        try:
            npz = np.load(path, allow_pickle=False)
            return {
                'name': str(npz['name_']),
                't0': float(npz['t0_']),
                't1': float(npz['t1_']),
                'filter_state': (json.loads(str(npz['filter_state_']))
                                 if 'filter_state_' in npz else {}),
            }
        except Exception:
            return None

    def _custom_ccg_name_session_coverage(self) -> tuple[dict[str, set[str]], int]:
        """Logical custom CCG name -> sessions that have an npz with that name (full cache scan)."""
        pattern = os.path.join(self.cache_dir, "*.npz")
        by_name: dict[str, set[str]] = {}
        for p in sorted(_glob.glob(pattern)):
            base = os.path.basename(p)
            sess = base.split("__", 1)[0] if "__" in base else ""
            if not sess:
                continue
            spec = self._custom_npz_spec(p)
            if not spec:
                continue
            nm = str(spec.get('name', '')).strip()
            if not nm:
                continue
            by_name.setdefault(nm, set()).add(sess)
        n_tot = len(self._ui._real_nd_keys_ordered())
        return by_name, n_tot

    def _custom_segment_disk_session(self, cs: dict) -> str:
        """Session string for a loaded/saved custom segment (metadata or npz filename)."""
        md = cs.get('metadata') or {}
        if md.get('session') is not None:
            return str(md['session'])
        sp = cs.get('src_path')
        if sp:
            bn = os.path.basename(sp)
            if "__" in bn:
                return bn.split("__", 1)[0]
        return str(self._ui.key.session)

    @property
    def _active_list(self) -> list:
        """The in-memory custom segment list for the currently active session."""
        return self._by_session.setdefault(self._active_sess, [])

    def _bind_custom_segments_to_session(self, sess: str):
        self._active_sess = str(sess)
        self._by_session.setdefault(self._active_sess, [])

    def _key_for_custom_segment_save(self, cs: dict):
        """``Key`` for npz filenames: segment's session + same connection-type label as UI."""
        ui = self._ui
        want_sess = self._custom_segment_disk_session(cs)
        cur_lbl = ui._type_label(ui.key)
        fallback = None
        for k in ui.cd.data.keys():
            if str(k.session) == want_sess:
                if ui._type_label(k) == cur_lbl:
                    return k
                fallback = fallback or k
        return fallback or ui.key

    def _enqueue_custom_ccg_task(self, *, key, t0, t1, name, intervals,
                                 active_duration, filter_state, metadata,
                                 auto_save: bool, load_into_ui: bool,
                                 batch_id: int | None = None) -> bool:
        task = {
            'kind': 'custom_ccg', 'key': key,
            't0': float(t0), 't1': float(t1), 'name': str(name),
            'intervals': intervals, 'active_duration': active_duration,
            'filter_state': filter_state or {}, 'metadata': metadata or {},
            'auto_save': bool(auto_save), 'load_into_ui': bool(load_into_ui),
            'batch_id': batch_id,
        }
        if not self._runner.enqueue(task):
            n = len(self._runner._pending)
            messagebox.showwarning(
                "Task queue full",
                f"Custom CCG queue full ({n}/{_MAX_QUEUE}). "
                "Wait for running tasks to complete.")
            return False
        return True

    def _queue_custom_ccg_for_spec(self, spec: dict, *, for_all: bool, auto_save: bool,
                                    target_sessions: list | None = None) -> int:
        """Enqueue custom CCG tasks for the given spec.

        target_sessions: explicit list of session strings to target (from picker dialog).
        for_all: if True (and target_sessions is None), targets all sessions.
        """
        ui = self._ui
        queued = 0
        if for_all and target_sessions is None:
            targets = ui._iter_type_keys_for_all_sessions()
        else:
            sess_set = ({str(s) for s in target_sessions}
                        if target_sessions is not None
                        else {str(s) for s in (spec.get('sessions') or []) if s != 'All'})
            targets = ([tk_ for nk in ui._real_nd_keys_ordered()
                        if str(nk.session) in sess_set
                        for tk_ in (ui._type_key_for_nd(nk),) if tk_ is not None]
                       if sess_set else [ui.key])
        n_splits = max(1, int(spec.get('n_splits') or 1))
        overlap_sec = max(0.0, float(spec.get('overlap_sec') or 0.0))
        scope_label = 'All' if (for_all and target_sessions is None) else str(spec.get('scope', ''))
        _any = getattr(ui, '_session_any_mode', False)
        for tk_ in targets:
            t_sess_start, t_sess_end = ui._session_wall_clock_extent_for_key(tk_)
            t0_r = ui._resolve_ts_time(spec.get('t0', 0.0), t_sess_start, t_sess_end)
            t1_r = ui._resolve_ts_time(spec.get('t1', t_sess_end), t_sess_start, t_sess_end)
            lone = ui._single_exclusive_segment_filter_label(spec.get('filter_state', {}))
            if lone is not None:
                span = ui._union_span_for_segment_label(tk_, lone)
                if span is not None:
                    t0_r, t1_r = span[0], span[1]
                    t0_r = ui._resolve_ts_time(t0_r, t_sess_start, t_sess_end)
                    t1_r = ui._resolve_ts_time(t1_r, t_sess_start, t_sess_end)
            chunks = _split_time_range(
                t0_r, t1_r, n_splits, overlap_sec, str(spec['name']))
            split_bid = None
            if len(chunks) > 1 and (_any or str(tk_.session) == str(ui.key.session)):
                split_bid = ui.time_slider._batch_next_id
                ui.time_slider._batch_next_id += 1
            split_names: list[str] = []
            for chunk_t0, chunk_t1, chunk_name in chunks:
                lo = min(t_sess_start, t_sess_end)
                hi = max(t_sess_start, t_sess_end)
                cs = min(max(float(chunk_t0), lo), hi)
                ce = min(max(float(chunk_t1), lo), hi)
                if ce <= cs:
                    continue
                chunk_t0, chunk_t1 = cs, ce
                chunk_spec = dict(spec, name=chunk_name, t0=chunk_t0, t1=chunk_t1)
                iv = ui._intervals_for_spec_on_key(chunk_spec, tk_)
                if iv is None or iv is False:
                    continue
                intervals, active_duration = iv
                if (isinstance(intervals, list) and len(intervals) == 0
                        and (active_duration is None or float(active_duration) <= 0.0)):
                    print(f"[CustomCCG] skip chunk (no overlap with filter): "
                          f"{chunk_name} session={tk_.session}")
                    continue
                metadata = {
                    'name': chunk_name,
                    'theme': (spec.get('filter_state') or {}).get('theme', 'segments'),
                    'labels': (spec.get('filter_state') or {}).get('labels', {}),
                    'scope': scope_label,
                    'session': str(tk_.session),
                    'timing': {'t0': chunk_t0, 't1': chunk_t1},
                }
                ok = self._enqueue_custom_ccg_task(
                    key=tk_,
                    t0=chunk_t0,
                    t1=chunk_t1,
                    name=chunk_name,
                    intervals=intervals,
                    active_duration=active_duration,
                    filter_state=spec.get('filter_state') or {},
                    metadata=metadata,
                    auto_save=auto_save,
                    load_into_ui=(_any or str(tk_.session) == str(ui.key.session)),
                    batch_id=split_bid,
                )
                if ok:
                    queued += 1
                    if split_bid is not None:
                        split_names.append(chunk_name)
            if split_bid is not None and split_names:
                ui.time_slider._batch_counts[split_bid] = len(split_names)
                ui.time_slider._batch_names[split_bid] = split_names
        return queued

    def _compute_custom_segment(self, t0: float, t1: float, name: str,
                                neurons_override=None, active_duration=None,
                                key_override=None, neurons_obj=None,
                                ccg_data_obj=None, metadata=None):
        ui = self._ui
        key_eff = key_override or ui.key
        neurons_eff = neurons_obj if neurons_obj is not None else ui.neurons
        cd_eff = ccg_data_obj if ccg_data_obj is not None else ui.ccg_data
        if neurons_eff is None:
            messagebox.showerror("Custom CCG", "No neuron data available.")
            return None
        try:
            neurons_slice = (neurons_override if neurons_override is not None
                             else neurons_eff.time_slice(t0, t1))
            has_highres = bool(getattr(ui.cd, '_ccg_highres', None))
            lo, hi = _custom_ccg_mod.compute_custom_ccg(
                t0, t1, name, neurons_slice, cd_eff.conf,
                has_highres=has_highres,
                active_duration=active_duration,
                excitability=getattr(key_eff, 'excitability', 'E'),
                metadata=metadata,
            )
            return _custom_ccg_mod._lo_hi_to_dict(lo, hi)
        except Exception as ex:
            print(f"[CustomSegment] ERROR: {ex}")
            traceback.print_exc()
            messagebox.showerror("Custom CCG", f"Error computing CCG:\n{ex}")
            return None

    def _ccg_cache_filename_for_key(self, seg_name: str, key=None) -> str:
        """Stable cache filename per (session, segment name); recomputes overwrite the same file."""
        key = key or self._ui.key
        session = str(key.session)
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', str(seg_name).replace(' ', '_'))
        return f"{session}__{safe}.npz"

    def _purge_timestamped_custom_ccg_npz(self, session: str, seg_name: str):
        """Remove legacy ``session__name__timestamp.npz`` files for this logical segment name."""
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', str(seg_name).replace(' ', '_'))
        patt = os.path.join(self.cache_dir, f"{session}__{safe}__*.npz")
        for p in _glob.glob(patt):
            try:
                os.remove(p)
            except OSError:
                pass

    def _upsert_custom_segment_by_name(self, lst: list, result: dict) -> tuple[int, bool]:
        """Replace an existing in-memory custom segment with the same name, else append.

        Returns (index_in_lst, did_append).
        """
        nm = str(result.get('name', ''))
        for i, existing in enumerate(lst):
            if str(existing.get('name', '')) == nm:
                lst[i] = result
                return i, False
        lst.append(result)
        return len(lst) - 1, True

    def _build_custom_spec(self, *, for_all: bool, for_session: str | None = None):
        ui = self._ui

        def _parse_time(raw: str, sentinel: str):
            raw = raw.strip().lower()
            if raw == sentinel:
                return sentinel
            try:
                return ui.time_slider._hms_to_sec(raw)
            except (ValueError, IndexError):
                messagebox.showerror("Time window",
                                     f"Invalid {sentinel} time. Use HH:MM:SS or '{sentinel}'.")
                return None

        t0 = _parse_time(ui.time_slider._start_var.get(), 'start')
        if t0 is None:
            return None
        t1 = _parse_time(ui.time_slider._end_var.get(), 'end')
        if t1 is None:
            return None
        if not isinstance(t0, str) and not isinstance(t1, str) and float(t1) <= float(t0):
            messagebox.showerror("Time window", "End time must be after start time.")
            return None
        try:
            n_splits = max(1, int(ui.time_slider._splits_var.get()))
        except (ValueError, TypeError):
            n_splits = 1
        try:
            overlap_sec = max(0.0, float(ui.time_slider._overlap_sec_var.get()))
        except (ValueError, TypeError):
            overlap_sec = 0.0
        name = ui.time_slider._name_var.get().strip()
        if not name:
            t0_str = t0 if isinstance(t0, str) else ui.time_slider._sec_to_hms(t0)
            t1_str = t1 if isinstance(t1, str) else ui.time_slider._sec_to_hms(t1)
            name = f"{t0_str}–{t1_str}"
        return {
            'name': name,
            't0': t0,
            't1': t1,
            'filter_state': ui._current_filter_state(),
            'scope': 'All' if for_all else str(for_session or ui._current_session_str()),
            'created_from_session': str(ui._current_session_str()),
            'sessions': ['All'] if for_all else [str(for_session or ui._current_session_str())],
            'n_splits': n_splits,
            'overlap_sec': overlap_sec,
        }

    def _normalize_custom_spec(self, spec: dict) -> dict:
        return _custom_ccg_mod.normalize_custom_spec(
            spec, default_session=self._ui._current_session_str()
        )

    def _update_json_list(self, fn):
        """Load suggestion list, call fn(list) to mutate it, then save."""
        specs = self._load_custom_ccg_suggestions()
        fn(specs)
        self._save_custom_ccg_suggestions(specs)

    def _load_custom_ccg_suggestions(self) -> list[dict]:
        path = self.suggestions_path
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding='utf-8') as f:
                raw = json.load(f)
            out = [self._normalize_custom_spec(x) for x in (raw.get('items', []) or [])
                   if isinstance(x, dict)]
            return [x for x in out if not self._ui._is_legacy_name(x.get('name', ''))]
        except Exception as ex:
            print(f"[CustomCCG] suggestion list load failed: {ex}")
            return []

    def _save_custom_ccg_suggestions(self, specs: list[dict]):
        payload = {'version': 1, 'items': [self._normalize_custom_spec(s) for s in specs]}
        self._ui._atomic_write_json(self.suggestions_path, payload)

    def _record_custom_ccg_suggestion(self, spec: dict):
        norm = self._normalize_custom_spec(spec)
        if self._ui._is_legacy_name(norm.get('name', '')):
            return
        key = (_custom_ccg_mod.custom_spec_key(norm), norm.get('scope', ''))
        def _add_if_new(specs):
            existing = {(_custom_ccg_mod.custom_spec_key(s), s.get('scope', '')) for s in specs}
            if key not in existing:
                specs.append(norm)
        self._update_json_list(_add_if_new)

    def _available_custom_ccg_specs(self) -> dict[tuple, dict]:
        pattern = os.path.join(self.cache_dir, "*.npz")
        by_key: dict[tuple, dict] = {}
        for p in sorted(_glob.glob(pattern)):
            try:
                npz = np.load(p, allow_pickle=False)
                session = str(os.path.basename(p).split("__", 1)[0]) if "__" in p else ""
                if self._ui._is_legacy_name(str(npz['name_'])):
                    continue
                spec = _custom_ccg_mod.RawCCGSpec.from_npz(npz, session).to_dict()
            except Exception:
                continue
            k = _custom_ccg_mod.custom_spec_key(spec)
            if k not in by_key:
                by_key[k] = self._normalize_custom_spec(spec)
            else:
                cur = set(by_key[k].get('sessions', []))
                cur.add(session)
                by_key[k]['sessions'] = sorted(cur)
        all_sessions = sorted(str(nk.session) for nk in self._ui._real_nd_keys_ordered())
        for entry in by_key.values():
            entry['scope'] = _custom_ccg_mod.RawCCGSpec.infer_scope(
                entry.get('sessions', []), all_sessions)
        return by_key

    def _emit_inventory_event(self):
        """Refresh suggestion list if available specs changed since last call."""
        avail = self._available_custom_ccg_specs()
        sig = tuple(sorted(
            (k, tuple(s.get('sessions', [])), str(s.get('scope', '')))
            for k, s in avail.items()
        ))
        if sig != getattr(self._ui, '_custom_ccg_inventory_sig', tuple()):
            self._inventory_sig = sig
            self._refresh_custom_ccg_suggestions(silent=True)

    def _refresh_custom_ccg_suggestions(self, silent: bool = False):
        """Rebuild suggestion list from saved custom CCG npz metadata."""
        specs = sorted(
            self._available_custom_ccg_specs().values(),
            key=lambda x: (x['name'], x['t0'], x['scope'])
        )
        specs = [s for s in specs
                 if not self._ui._is_legacy_name(s.get('name', ''))]
        self._save_custom_ccg_suggestions(specs)
        if not silent:
            messagebox.showinfo("Custom CCG suggestions",
                                f"Updated suggestion list with {len(specs)} item(s).")

    def _on_chunk_done(self, task):
        """Decrement split-batch counter (compute finished, load failed, or queue removed)."""
        if not isinstance(task, dict):
            return
        bid = task.get('batch_id')
        if bid is None:
            return
        ts = self._ui.time_slider
        if bid not in ts._batch_counts:
            return
        ts._batch_counts[bid] -= 1
        if ts._batch_counts[bid] > 0:
            return
        del ts._batch_counts[bid]
        names = list(ts._batch_names.pop(bid, []))
        self._ui.root.after(100, lambda n=names: self._prompt_save_chunks(n))

    def _prompt_save_chunks(self, names: list[str]):
        """After all tasks in a time-slider split batch finish, offer to save unsaved chunks."""
        name_set = set(names)
        if not name_set:
            return
        unsaved: list = []
        for lst in self._by_session.values():
            for cs in lst or []:
                if (isinstance(cs, dict) and cs.get('name') in name_set
                        and not cs.get('src_path')):
                    unsaved.append(cs)
        if not unsaved:
            return
        n = len(unsaved)
        if messagebox.askyesno(
                "Save split windows",
                f"{n} split window(s) finished computing but are not saved to disk yet.\n\n"
                "Save them as .npz files now? (You can reload them later from the cache.)"):
            self._save_custom_segment_objects(unsaved, show_saved_message=True)

    def _save_custom_segment_objects(self, segments: list, *, show_saved_message: bool = True) -> list[str]:
        """Write custom segment dicts to npz (correct session prefix per segment)."""
        ui = self._ui
        saved: list[str] = []
        saved_by_name: dict[str, list[str]] = {}
        for cs in segments:
            if not isinstance(cs, dict):
                continue
            try:
                save_key = self._key_for_custom_segment_save(cs)
                _sess_w = str(save_key.session)
                self._purge_timestamped_custom_ccg_npz(_sess_w, str(cs['name']))
                fname = self._ccg_cache_filename_for_key(cs['name'], key=save_key)
                path = os.path.join(self.cache_dir, fname)
                _custom_ccg_mod.save_custom_segment_to_npz(cs, path)
                cs['src_path'] = path
                name = str(cs['name'])
                saved.append(name)
                saved_by_name.setdefault(name, []).append(_sess_w)
            except Exception as _exc:
                print(f"[CCGReviewUI] failed to save segment "
                      f"'{cs.get('name', '?')}': {_exc}")
        if saved:
            if show_saved_message:
                lines = []
                for name, sessions in saved_by_name.items():
                    if len(sessions) == 1:
                        lines.append(f"{name}  ({sessions[0]})")
                    else:
                        lines.append(f"{name} > list ({len(sessions)} sessions)")
                messagebox.showinfo(
                    "Saved",
                    "Saved custom CCG segment(s):\n" + "\n".join(lines))
            if hasattr(ui, '_ts_status_var'):
                ui.time_slider._status_var.set(f"Saved: {', '.join(dict.fromkeys(saved))}")
            self._emit_inventory_event()
            ui._save_ui_state()
            ts = ui.time_slider
            fn, win = ts._load_custom_ccg_refresh, ts._load_custom_ccg_win
            if fn is not None and win is not None:
                try:
                    if win.winfo_exists():
                        fn()
                except tk.TclError:
                    ts._load_custom_ccg_win = ts._load_custom_ccg_refresh = None
        return saved

    def _archive_stale_custom_ccgs(self):
        """Move saved custom CCG files that pre-date the total_time_hours field to _trash/.
        Returns (n_archived, trash_dir) so the caller can notify the user."""
        pattern = os.path.join(self.cache_dir, f"{self._ui.key.session}__*.npz")
        trash_dir = os.path.join(self.cache_dir, '_trash')
        os.makedirs(trash_dir, exist_ok=True)
        archived = []
        for p in _glob.glob(pattern):
            try:
                m = np.load(p, allow_pickle=False)
                if 'total_time_hours_' not in m:
                    dest = os.path.join(trash_dir, os.path.basename(p))
                    shutil.move(p, dest)
                    archived.append(os.path.basename(p))
            except Exception:
                pass
        return len(archived), trash_dir
