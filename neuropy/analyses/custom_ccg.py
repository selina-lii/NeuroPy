"""Custom CCG computation, segment management, persistence, and UI coordination."""
from __future__ import annotations

import os
import json
import threading
import traceback
from pathlib import Path as _Path
from tkinter import messagebox
from typing import TYPE_CHECKING
from neuropy.ui.utils import BackgroundTaskRunner
from neuropy.analyses.ms_connectivity import CCGDataset, CCGSourceConfig
from neuropy.analyses.utils import Cacheable, split_time_range as _split_time_range

_ALL_SEGS = "All"  # must match ccg_ui._ALL_SEGS

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

_MAX_QUEUE = 50


class CustomCCGWorker:
    """Wraps BackgroundTaskRunner + thread result buffer for CCG computation."""

    def __init__(self, mgr: 'CustomCCGManager'):
        self._mgr = mgr
        self._ui = mgr._ui
        self._runner = BackgroundTaskRunner(max_queue=_MAX_QUEUE, use_result_queue=False)
        self._thread_result: list = []

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

    def _custom_ccg_start_next(self):
        """Start the next queued custom CCG task if none is running."""
        ui = self._ui

        def _launch(task, _q):
            t0 = float(task.get('t0', 0.0))
            t1 = float(task.get('t1', 0.0))
            name = str(task.get('name', 'custom'))
            intervals = task.get('intervals')
            active_duration = task.get('active_duration')
            filter_state = task.get('filter_state', {})
            key_for_task = task.get('key', ui.key)
            nd_key = key_for_task.nd()
            ccg_data_obj = ui.cd.ccg.get(nd_key) if hasattr(ui.cd, 'ccg') else ui.ccg_data
            neurons_obj = (ui.cd.nd.data[nd_key]
                           if getattr(ui.cd, 'nd', None) is not None else None)
            if ccg_data_obj is None or neurons_obj is None:
                try:
                    ui.cd.get_ccg()
                except Exception as ex:
                    messagebox.showerror("Custom CCG", f"Session load failed for {key_for_task.session}:\n{ex}")
                    self._on_chunk_done(task)
                    return None
                ccg_data_obj = ui.cd.ccg.get(nd_key) if hasattr(ui.cd, 'ccg') else ui.ccg_data
                neurons_obj = (ui.cd.nd.data[nd_key]
                               if getattr(ui.cd, 'nd', None) is not None else None)
                if ccg_data_obj is None or neurons_obj is None:
                    print(f"[CustomCCG] missing session data after load: {key_for_task.session}")
                    self._on_chunk_done(task)
                    return None
            self._thread_result.clear()

            def _ccg_worker():
                try:
                    neurons_override = (
                        ui.time_slider._apply_brain_state_intervals(intervals, t0, t1, neurons_obj=neurons_obj)
                        if intervals is not None else None)
                    result = self._compute_custom_segment(
                        t0, t1, name,
                        neurons_override=neurons_override, active_duration=active_duration,
                        key_override=key_for_task, neurons_obj=neurons_obj, ccg_data_obj=ccg_data_obj)
                    if result is not None:
                        result.source.filter_state = filter_state
                        result._task_session = str(key_for_task.session)
                    self._thread_result.append(
                        result if result is not None else {'error': 'compute returned None'})
                except Exception as ex:
                    self._thread_result.append({'error': str(ex)})

            t = threading.Thread(target=_ccg_worker, daemon=True)
            t.start()
            return t

        started = self._runner.start_next(_launch)
        if started and self._runner._poll_id is None:
            self._runner.start_polling(
                ui.root, interval_ms=300,
                on_done=self._on_custom_ccg_done)

    def _on_custom_ccg_done(self, completed_task, _result):
        """Called by BackgroundTaskRunner when the custom CCG thread exits."""
        ui = self._ui
        mgr = self._mgr
        result = self._thread_result[0] if self._thread_result else None
        self._thread_result.clear()

        is_error = isinstance(result, dict) and result.get('error')
        if result is not None and not is_error:
            if isinstance(completed_task, dict) and completed_task.get('auto_save'):
                result.save('lowres')
                if any(getattr(k, 'resolution', None) == 'highres' for k in result.ccg):
                    result.save('highres')
                mgr.state._emit_inventory_event()
            should_load = (not isinstance(completed_task, dict)
                            or bool(completed_task.get('load_into_ui', True)))
            _tk_done = (completed_task.get('key', ui.key)
                        if isinstance(completed_task, dict) else ui.key)
            _lsess = str(getattr(result, '_task_session', getattr(_tk_done, 'session', '')))
            _lst = mgr._by_session.setdefault(_lsess, [])
            nm = result.source.name
            idx = next((i for i, cd in enumerate(_lst) if cd.source.name == nm), -1)
            if idx >= 0:
                _lst[idx] = result
            else:
                _lst.append(result)
                idx = len(_lst) - 1
            if should_load and mgr._active_list is _lst:
                ui._build_sig_chips()
                ui.current_segment = ui._seg_name(ui.n_segments + 1 + idx)
                ui._clamp_current_segment_for_session()
                ui._update_segment_label()
                ui._plot_mgr.update_plot()
            if hasattr(ui, '_ts_status_var'):
                ui.time_slider._status_var.set(f"Done: {result.source.name}")
            ui.root.bell()
        elif is_error:
            messagebox.showerror("Custom CCG", f"Computation failed:\n{result['error']}")

        if completed_task is not None:
            self._on_chunk_done(completed_task)

        self._custom_ccg_start_next()

    def _compute_custom_segment(self, t0: float, t1: float, name: str,
                                 neurons_override=None, active_duration=None,
                                 key_override=None, neurons_obj=None,
                                 ccg_data_obj=None):
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
            cd = CCGDataset(conf=cd_eff.conf, source=CCGSourceConfig(
                name=name, t0=t0, t1=t1, active_duration=active_duration,
            ))
            cd.compute_custom(neurons_slice, has_highres=has_highres,
                              excitability=getattr(key_eff, 'excitability', 'E'))
            return cd
        except Exception as ex:
            print(f"[CustomCCG] ERROR: {ex}")
            traceback.print_exc()
            messagebox.showerror("Custom CCG", f"Error computing CCG:\n{ex}")
            return None

    def _on_chunk_done(self, task):
        """Decrement split-batch counter; prompt to save when batch completes."""
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
        """After all tasks in a split batch finish, offer to save unsaved chunks."""
        name_set = set(names)
        if not name_set:
            return
        unsaved: list = []
        for lst in self._mgr._by_session.values():
            for cd in lst or []:
                lo_key = next((k for k in cd.ccg if getattr(k, 'resolution', None) == 'lowres'), None)
                if cd.source.name in name_set and (lo_key is None or not cd.ccg[lo_key].is_saved):
                    unsaved.append(cd)
        if not unsaved:
            return
        n = len(unsaved)
        if messagebox.askyesno(
                "Save split windows",
                f"{n} split window(s) finished computing but are not saved to disk yet.\n\n"
                "Save them as .npz files now? (You can reload them later from the cache.)"):
            saved = []
            for cd in unsaved:
                try:
                    cd.save('lowres')
                    if any(getattr(k, 'resolution', None) == 'highres' for k in cd.ccg):
                        cd.save('highres')
                    saved.append(cd.source.name)
                except Exception as exc:
                    print(f"[CustomCCG] save failed '{cd.source.name}': {exc}")
            if saved:
                messagebox.showinfo("Saved", "Saved:\n" + "\n".join(saved))


class CustomCCGState:
    """UI state: active session, stacked segments, inventory change-detection."""

    def __init__(self, mgr: 'CustomCCGManager'):
        self._mgr = mgr
        self._ui = mgr._ui
        self.active_sess: str = ''
        self.inventory_sig: tuple = ()
        self._stacked_segments: set = set()

    def _emit_inventory_event(self):
        """Refresh suggestion list if available specs changed since last call."""
        avail = CCGDataset.available_specs(self._mgr.cache_dir)
        sig = tuple(sorted(
            (k, tuple(s.get('sessions', [])), str(s.get('scope', '')))
            for k, s in avail.items()
        ))
        if sig != self.inventory_sig:
            self.inventory_sig = sig
            self._refresh_custom_ccg_suggestions(silent=True)

    def _refresh_custom_ccg_suggestions(self, silent: bool = False):
        """Rebuild suggestion list from saved custom CCG npz metadata."""
        mgr = self._mgr
        specs = sorted(
            CCGDataset.available_specs(mgr.cache_dir).values(),
            key=lambda x: (x['name'], x['t0'], x['scope'])
        )
        specs = [s for s in specs
                 if not self._ui._is_legacy_name(s.get('name', ''))]
        mgr._save_suggestions(specs)
        if not silent:
            messagebox.showinfo("Custom CCG suggestions",
                                f"Updated suggestion list with {len(specs)} item(s).")

    def _record_custom_ccg_suggestion(self, spec: dict):
        norm = self._mgr._normalize_custom_spec(spec)
        if self._ui._is_legacy_name(norm.get('name', '')):
            return
        key = (CCGSourceConfig.spec_key(norm), norm.get('scope', ''))
        def _add_if_new(specs):
            existing = {(CCGSourceConfig.spec_key(s), s.get('scope', '')) for s in specs}
            if key not in existing:
                specs.append(norm)
        self._mgr._update_json_list(_add_if_new)


class CustomCCGManager(Cacheable):
    """Coordinator: owns worker/state and _by_session; houses cross-cutting UI methods."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self.cache_dir = str(_Path(__file__).resolve().parents[2] / "data" / "custom_ccg")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.suggestions_path = os.path.join(self.cache_dir, "suggested_custom_ccgs.json")
        self._by_session: dict = {}
        self.worker = CustomCCGWorker(self)
        self.state = CustomCCGState(self)
        ui._custom_ccg_pending = self.worker._runner._pending
        self.state.active_sess = str(ui.key.session)
        self._by_session.setdefault(self.state.active_sess, [])

    @property
    def _active_list(self) -> list:
        return self._by_session.setdefault(self.state.active_sess, [])

    def _is_custom_segment(self, seg=None):
        if seg is None:
            seg = self._ui.current_segment
        if isinstance(seg, str):
            if seg == _ALL_SEGS:
                return False
            names = self._ui.ccg_ptr.segment_names if self._ui.ccg_ptr is not None else []
            if seg in names:
                return False
            cs_list = getattr(self._ui, '_custom_segments', []) or []
            return any(cs.source.name == seg for cs in cs_list)
        return seg > self._ui.n_segments

    def _custom_seg_index(self, seg=None):
        if seg is None:
            seg = self._ui.current_segment
        if isinstance(seg, str):
            cs_list = getattr(self._ui, '_custom_segments', []) or []
            for ci, cs in enumerate(cs_list):
                if cs.source.name == seg:
                    return ci
            return -1
        return seg - self._ui.n_segments - 1

    def _remove_custom_segment(self, ci):
        ui = self._ui
        if 0 <= ci < len(self._active_list):
            self._active_list.pop(ci)
            if ui._custom_mgr._is_custom_segment():
                new_ci = ui._custom_mgr._custom_seg_index()
                if new_ci < 0 or new_ci >= len(self._active_list):
                    ui.current_segment = _ALL_SEGS
            ui._build_sig_chips()
            ui._update_segment_label()
            ui._plot_mgr.update_plot()

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
        sess = str(for_session or ui._setup_mgr._current_session_str())
        return CCGSourceConfig(
            name=name,
            t0=t0,
            t1=t1,
            filter_state=ui._current_filter_state(),
            scope='All' if for_all else sess,
            created_from_session=str(ui._setup_mgr._current_session_str()),
            sessions=['All'] if for_all else [sess],
            n_splits=n_splits,
            overlap_sec=overlap_sec,
        )

    def _normalize_custom_spec(self, spec: dict) -> dict:
        return CCGSourceConfig.normalize(
            spec, default_session=self._ui._setup_mgr._current_session_str()
        )

    def _load_suggestions(self) -> list:
        if not os.path.isfile(self.suggestions_path):
            return []
        try:
            with open(self.suggestions_path, encoding='utf-8') as f:
                raw = json.load(f)
            out = [self._normalize_custom_spec(x)
                   for x in (raw.get('items', []) or []) if isinstance(x, dict)]
            return [x for x in out if not self._ui._is_legacy_name(x.get('name', ''))]
        except Exception as ex:
            print(f"[CustomCCG] suggestion list load failed: {ex}")
            return []

    def _save_suggestions(self, specs: list) -> None:
        payload = {'version': 1, 'items': [self._normalize_custom_spec(s) for s in specs]}
        self._ui._atomic_write_json(self.suggestions_path, payload)

    def _update_json_list(self, fn):
        """Load suggestion list, call fn(list) to mutate it, then save."""
        specs = self._load_suggestions()
        fn(specs)
        self._save_suggestions(specs)

    def _generate_suggested_custom_ccgs(self):
        from neuropy.ui.dialogs import SuggestedCCGDialog
        ui = self._ui
        specs = self._load_suggestions()

        def _on_run(selected_specs):
            queued = sum(
                self._queue_custom_ccg_for_spec(
                    s, for_all=(str(s.get('scope', '')).lower() == 'all'),
                    auto_save=True)
                for s in selected_specs)
            if queued:
                ui.time_slider._status_var.set(f"Queued {queued} suggested custom CCG task(s)")
                self.worker._custom_ccg_start_next()
            else:
                ui.time_slider._status_var.set("All suggested custom CCGs already exist")

        SuggestedCCGDialog.show(ui, specs, _on_run)

    def _queue_custom_ccg_for_spec(self, spec: dict, *, for_all: bool, auto_save: bool,
                                    target_sessions: list | None = None) -> int:
        ui = self._ui
        queued = 0
        if for_all and target_sessions is None:
            targets = ui._sess_mgr._iter_type_keys_for_all_sessions()
        else:
            sess_set = ({str(s) for s in target_sessions}
                        if target_sessions is not None
                        else {str(s) for s in (spec.get('sessions') or []) if s != 'All'})
            targets = ([tk_ for nk in ui._sess_mgr._real_nd_keys_ordered()
                        if str(nk.session) in sess_set
                        for tk_ in (ui._sess_mgr._type_key_for_nd(nk),) if tk_ is not None]
                       if sess_set else [ui.key])
        n_splits = max(1, int(spec.get('n_splits') or 1))
        overlap_sec = max(0.0, float(spec.get('overlap_sec') or 0.0))
        _any = getattr(ui, '_session_any_mode', False)
        for tk_ in targets:
            t_sess_start, t_sess_end = ui._sess_mgr._session_wall_clock_extent_for_key(tk_)
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
                ok = self.worker._enqueue_custom_ccg_task(
                    key=tk_,
                    t0=chunk_t0,
                    t1=chunk_t1,
                    name=chunk_name,
                    intervals=intervals,
                    active_duration=active_duration,
                    filter_state=spec.get('filter_state') or {},
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

