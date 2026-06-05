"""
TimeSliderPanel — extracted from CCGReviewUI.

Full-width time-window selector with epoch display, zoom, and
custom-segment CCG creation.
"""
from __future__ import annotations

import collections
import glob as _glob
import json
import os
import re
import threading
import traceback
import tkinter as tk
from copy import deepcopy
from pathlib import Path as _Path
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

import numpy as np

from neuropy.core.epoch import Epoch as _Epoch
from neuropy.core.neurons import Neurons
from neuropy.ui.utils import intersect_intervals, UITheme

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

class TimeSliderPanel:
    """Time-slider panel: epoch display, zoom, custom-segment CCG creation."""

    _TS_COLORS = ['#BBDEFB', '#C8E6C9', '#FFF9C4', '#FFE0B2', '#E1BEE7',
                  '#F8BBD0', '#D7CCC8', '#B2EBF2', '#DCEDC8', '#F0F4C3']
    _TS_NONE_COLOR = '#E0E0E0'
    _TS_NONE = 'NONE'


    # ── Init / setup ────────────────────────────────────────────────────────

    def __init__(self, parent: tk.Widget, ui: "CCGReviewUI"):
        self.ui = ui
        self._slider_t_start: float = None
        self._slider_t_end: float = None
        self._slider_dragging: str = None
        self._epoch_bounds: list = []
        self._total_sec: float = 0.0
        self._active_label: str = None
        self._segment_name: str = ""
        self._load_custom_ccg_win = None
        self._load_custom_ccg_refresh = None
        self._themes: dict = {}
        self._current_theme: str = 'segments'
        self._all_theme_bounds: dict = {}
        self._theme_flag_vars: dict = {}
        self._filter_btn = None
        self._per_theme_label_state: dict = {}
        self._zoom_start: float = None
        self._zoom_end: float = None
        self._zoom_dragging: str = None
        self._batch_next_id: int = 1
        self._batch_counts: dict = {}
        self._batch_names: dict = {}
        self._setup(parent)

    def _setup(self, parent):
        """Full-width time-window selector — hidden by default."""
        self.time_slider_frame = ttk.LabelFrame(
            parent, text="Time Slider - Behavioral Epochs")
        # Not packed — shown when 'Time Slider' panel is enabled

        # Theme selector row
        theme_row = ttk.Frame(self.time_slider_frame)
        theme_row.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(theme_row, text="Theme:").pack(side=tk.LEFT)
        self._theme_var = tk.StringVar(value='segments')
        self._theme_combo = ttk.Combobox(
            theme_row, textvariable=self._theme_var,
            values=['segments'], width=16, state='readonly')
        self._theme_combo.pack(side=tk.LEFT, padx=4)
        self._theme_combo.bind('<<ComboboxSelected>>', self._on_theme_change)
        self._theme_info_var = tk.StringVar(value="")
        ttk.Label(theme_row, textvariable=self._theme_info_var,
                  font=('Courier', 8), foreground='#666').pack(
            side=tk.LEFT, padx=6)
        # Single per-theme "include in filter" toggle — variable swapped by _on_theme_change
        self._filter_btn = ttk.Checkbutton(
            theme_row, text="Include in filter",
            variable=tk.BooleanVar())  # placeholder; replaced when themes are discovered
        self._filter_btn.pack(side=tk.LEFT, padx=(4, 2))

        # ── Overlap label selector (inline, hidden until multi-label theme) ──
        self._overlap_row = ttk.Frame(theme_row)
        # Not packed initially — shown inline when theme has multiple labels
        ttk.Label(self._overlap_row, text="Show:").pack(side=tk.LEFT, padx=(0, 2))
        self._label_var = tk.StringVar(value='All')
        self._label_combo = ttk.Combobox(
            self._overlap_row, textvariable=self._label_var,
            values=['All'], width=12, state='readonly')
        self._label_combo.pack(side=tk.LEFT)
        self._label_combo.bind('<<ComboboxSelected>>', self._on_label_change)
        ttk.Button(self._overlap_row, text="All",
                   command=self._on_label_reset).pack(side=tk.LEFT, padx=2)

        # ── Tool bar (right side of theme row) ──
        toolbar = ttk.Frame(theme_row)
        toolbar.pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Button(toolbar, text="💾", width=2,
                   command=self._save_custom_ccg).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="📂", width=2,
                   command=self._load_custom_ccg).pack(side=tk.LEFT, padx=1)
        ttk.Label(toolbar, text="|", foreground='#BBB').pack(side=tk.LEFT, padx=2)
        self._snap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Snap",
                        variable=self._snap_var).pack(side=tk.LEFT, padx=2)
        self._tool_var = tk.StringVar(value='none')
        self._selection_btn = ttk.Checkbutton(
            toolbar, text="Zoom-in", variable=self._tool_var,
            onvalue='selection', offvalue='none',
            command=self._on_tool_change)
        self._selection_btn.pack(side=tk.LEFT, padx=2)
        self._lock_var = tk.BooleanVar(value=False)
        self._lock_btn = ttk.Checkbutton(
            toolbar, text="\U0001F512 Lock", variable=self._lock_var,
            command=self._on_tool_change)
        self._lock_btn.pack(side=tk.LEFT, padx=2)

        # ── Legend row (below theme row, populated by _update_legend) ──
        self._legend_frame = ttk.Frame(self.time_slider_frame)
        # Packed dynamically by _update_legend

        # ── Main canvas ──
        self._main_canvas_frame = ttk.Frame(self.time_slider_frame)
        top = self._main_canvas_frame
        top.pack(fill=tk.X, padx=4, pady=(2, 0))

        self.ts_canvas = tk.Canvas(top, height=56, bg='#F5F5F5', cursor='crosshair')
        self.ts_canvas.pack(fill=tk.X, expand=True)
        self.ts_canvas.bind('<Configure>',      self._redraw)
        self.ts_canvas.bind('<Button-1>',        self._mouse_press)
        self.ts_canvas.bind('<B1-Motion>',       self._mouse_drag)
        self.ts_canvas.bind('<ButtonRelease-1>', self._mouse_release)

        # ── CCG time range bar (below main canvas, always visible) ──
        self._ccg_ctrl = ttk.Frame(self.time_slider_frame)
        ccg_ctrl = self._ccg_ctrl
        ccg_ctrl.pack(fill=tk.X, padx=4, pady=(2, 0))
        self._start_var = tk.StringVar(value="00:00:00")
        self._end_var   = tk.StringVar(value="00:00:00")
        self._build_time_range_bar(ccg_ctrl, "CCG time range",
                                   self._start_var, self._end_var,
                                   self._on_time_slider_set)
        ttk.Label(ccg_ctrl, text="Name:").pack(side=tk.LEFT, padx=(8, 0))
        self._name_var = tk.StringVar(value="")
        ttk.Entry(ccg_ctrl, textvariable=self._name_var,
                  width=14).pack(side=tk.LEFT, padx=2)
        ttk.Label(ccg_ctrl, text="Splits:").pack(side=tk.LEFT, padx=(6, 0))
        self._splits_var = tk.IntVar(value=1)
        ttk.Spinbox(ccg_ctrl, from_=1, to=99, textvariable=self._splits_var,
                    width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(ccg_ctrl, text="Overlap(s):").pack(side=tk.LEFT, padx=(4, 0))
        self._overlap_sec_var = tk.StringVar(value="0")
        ttk.Entry(ccg_ctrl, textvariable=self._overlap_sec_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(ccg_ctrl, text="Clear",
                   command=self._on_time_slider_clear).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(ccg_ctrl, text="Apply to Multiple Sessions",
                   command=self._on_time_slider_apply_multiple_sessions).pack(side=tk.LEFT, padx=(2, 2))
        self._status_var = tk.StringVar(value="")
        ttk.Label(ccg_ctrl, textvariable=self._status_var,
                  font=('Courier', 8), foreground='#555').pack(side=tk.LEFT, padx=8)

        # ── Zoom detail canvas (hidden until zoom is active) ──
        self._zoom_frame = ttk.Frame(self.time_slider_frame)
        self._radiate_canvas = tk.Canvas(
            self._zoom_frame, height=16, bg='#FEFEFE', highlightthickness=0)
        self._radiate_canvas.pack(fill=tk.X, expand=True, side=tk.TOP)
        self._zoom_canvas = tk.Canvas(
            self._zoom_frame, height=56, bg='#FAFAFA', cursor='crosshair')
        self._zoom_canvas.pack(fill=tk.X, expand=True)
        self._zoom_canvas.bind('<Configure>', self._zoom_redraw)

        # ── Zoom time range bar (below zoom canvas, shown with zoom) ──
        self._zoom_ctrl = ttk.Frame(self._zoom_frame)
        self._zoom_ctrl.pack(fill=tk.X, padx=4, pady=(2, 0))
        self._zoom_start_var = tk.StringVar(value="00:00:00")
        self._zoom_end_var   = tk.StringVar(value="00:00:00")
        self._build_time_range_bar(self._zoom_ctrl, "Zoom range",
                                   self._zoom_start_var, self._zoom_end_var,
                                   self._on_zoom_range_set)

    def _bind_time_entry(self, entry, var):
        """Bind Return/FocusOut on *entry* to parse and normalise a HH:MM:SS time value."""
        def _resolve(_=None):
            raw = var.get().strip().lower()
            if raw in ('start', 'end'):
                var.set(raw)  # keep sentinel as-is
                return
            try:
                sec = self._hms_to_sec(raw)
                var.set(self._sec_to_hms(sec))
            except (ValueError, IndexError):
                pass
        entry.bind('<Return>', _resolve)
        entry.bind('<FocusOut>', _resolve)

    def _build_time_range_bar(self, parent, title, start_var, end_var, set_cmd):
        """Pack [title label] [Start entry] [End entry] [Set button] into *parent*."""
        ttk.Label(parent, text=title,
                  font=('Arial', 8, 'bold'), foreground='#444').pack(side=tk.LEFT, padx=(0, 6))
        for lbl, var in [("Start:", start_var), ("End:", end_var)]:
            ttk.Label(parent, text=lbl).pack(
                side=tk.LEFT, padx=(6, 0) if lbl == "End:" else 0)
            e = ttk.Entry(parent, textvariable=var, width=10)
            e.pack(side=tk.LEFT, padx=2)
            self._bind_time_entry(e, var)
        ttk.Button(parent, text="Set", command=set_cmd).pack(side=tk.LEFT, padx=4)

    def toggle(self):
        show = self.ui._panel_vars['Time Slider'].get()
        print(f"[CCGReviewUI] _toggle_time_slider show={show}")
        if show:
            try:
                self.time_slider_frame.pack(
                    in_=self.ui._main_frame, side=tk.TOP,
                    fill=tk.X, before=self.ui._paned, pady=(0, 4))
                self._discover_themes()
                self._init_times()
                print(f"[CCGReviewUI]   epoch_bounds={len(self._epoch_bounds)} "
                      f"total_sec={self._total_sec:.1f}")
                self._redraw()
            except Exception as ex:
                print(f"[CCGReviewUI]   ERROR in _toggle_time_slider: {ex}")
                traceback.print_exc()
        else:
            self.time_slider_frame.pack_forget()

    # ── Key / session change callbacks ──────────────────────────────────────

    def on_key_changed(self):
        self._reinit_times_for_current_key()

    def on_session_mode_changed(self):
        self._refresh_union_if_all_sessions_mode()

    def _reinit_times_for_current_key(self):
        """Refresh time-slider bounds from the current CCG pointer / theme (same session)."""
        if getattr(self, '_theme_combo', None) is None:
            return
        self._init_times()
        self._redraw()

    def _refresh_epochs_for_current_key(self):
        """Re-discover behavioral Epoch objects for the current session and refresh bounds."""
        combo = getattr(self, '_theme_combo', None)
        if combo is None:
            return
        self._discover_themes()
        vals = tuple(combo.cget('values') or ())
        if vals and self._theme_var.get() not in vals:
            self._theme_var.set('segments')
        self._on_theme_change()

    def _refresh_union_if_all_sessions_mode(self):
        """Recompute All-session label union without resetting the time-slider theme/handles."""
        if not getattr(self, '_session_any_mode', False):
            return
        if getattr(self, '_theme_combo', None) is None:
            return
        self._rebuild_theme_label_union_for_all_sessions()
        self._label_colors = None
        self._update_overlap_ui()
        self._redraw()

    def _rebuild_theme_label_union_for_all_sessions(self):
        """When Session=All, map each theme to sorted unique labels seen on any session."""
        self._theme_label_union_all_sessions = {}
        if not getattr(self, '_session_any_mode', False):
            return
        session_objs = self._all_process_data_sessions()
        if not session_objs:
            return
        for attr in self._themes.keys():
            acc = set()
            for s in session_objs:
                obj = getattr(s, attr, None)
                if obj is None or _Epoch is None or not isinstance(obj, _Epoch):
                    continue
                if obj.n_epochs <= 0:
                    continue
                for lbl in obj.labels:
                    lbl_str = str(lbl).strip()
                    if lbl_str:
                        acc.add(lbl_str)
            if acc:
                self._theme_label_union_all_sessions[attr] = sorted(acc)
        # segments: union of segment labels across every loaded CCG pointer
        seg = set()
        data = getattr(self.ui.cd, 'ptr', None)
        if data:
            for ptr in data.values():
                if ptr is None:
                    continue
                try:
                    et = ptr.edge_times
                except Exception:
                    continue
                cols = getattr(et, 'columns', None)
                if cols is None or 'label' not in cols:
                    continue
                for v in et['label'].values:
                    s = str(v).strip()
                    if s:
                        seg.add(s)
        if seg:
            self._theme_label_union_all_sessions['segments'] = sorted(seg)

    # ── Theme / label data ──────────────────────────────────────────────────

    def _discover_themes(self):
        """Discover available Epoch objects from the session for theme switching."""
        self._themes = {}
        self._theme_label_union_all_sessions = {}
        # Known Epoch attribute names on session objects (ProcessData)
        _EPOCH_ATTRS = [
            'paradigm', 'brainstates', 'theta', 'theta_epochs',
            'ripple', 'sw', 'spindle', 'pbe', 'off_epochs',
            'micro_arousals', 'artifact', 'handling',
            'maze1_run', 'maze2_run', 'maze_run', 'remaze_run',
        ]
        sessions = getattr(getattr(self.ui.cd, 'nd', None), '_sessions', None)
        if not sessions:
            return
        if not isinstance(sessions, (list, tuple)):
            sessions = [sessions]
        # Use the first session that matches our current key
        session_name = getattr(self.ui.key, 'session', None)
        session = None
        for s in sessions:
            nd = getattr(self.ui.cd, 'nd', None)
            if nd is not None:
                sname = nd._short_session_name(s)
                if sname == session_name:
                    session = s
                    break
        if session is None:
            print(f"[TimeSlider] no matching session object for {session_name}")
            return
        for attr in _EPOCH_ATTRS:
            obj = getattr(session, attr, None)
            if obj is not None and _Epoch is not None and isinstance(obj, _Epoch):
                if obj.n_epochs > 0:
                    self._themes[attr] = obj
        # Update combobox values
        theme_names = ['segments'] + sorted(self._themes.keys())
        self._theme_combo['values'] = theme_names
        n_themes = len(self._themes)
        self._theme_info_var.set(
            f"{n_themes} theme{'s' if n_themes != 1 else ''} available")
        self._rebuild_theme_label_union_for_all_sessions()
        # Pre-compute interval bounds for all themes
        self._all_theme_bounds = {}
        for theme_name, epoch_obj in self._themes.items():
            labs = [str(x).strip() for x in epoch_obj.labels]
            self._all_theme_bounds[theme_name] = [
                (float(s), float(e), lb)
                for s, e, lb in zip(epoch_obj.starts, epoch_obj.stops, labs)
            ]
        # Add BoolVars for any newly discovered themes (preserve existing state)
        for tname in self._all_theme_bounds:
            if tname not in self._theme_flag_vars:
                self._theme_flag_vars[tname] = tk.BooleanVar(value=False)
        # Wire the single filter button to the current theme's BoolVar
        cur = getattr(self, '_current_theme', None)
        if cur and cur in self._theme_flag_vars and self._filter_btn is not None:
            self._filter_btn.configure(variable=self._theme_flag_vars[cur])

    def _active_labels_for_theme(self, theme_name: str) -> set:
        """Active label set for a theme: live legend for current display theme, saved state for others."""
        if theme_name == getattr(self, '_current_theme', None):
            return {lb for lb, v in getattr(self, '_legend_toggles', {}).items() if v.get()}
        saved = self._per_theme_label_state.get(theme_name, {})
        all_labels = {lb for _, _, lb in self._all_theme_bounds.get(theme_name, [])}
        return {lb for lb in all_labels if saved.get(lb, True)}

    def _all_process_data_sessions(self):
        """ProcessData objects for every loaded session (for cross-session label union)."""
        nd = getattr(self.ui.cd, 'nd', None)
        if nd is None:
            return []
        raw = getattr(nd, '_sessions', None)
        if raw is None:
            raw = []
        elif not isinstance(raw, (list, tuple)):
            raw = [raw]
        objs = [s for s in raw if s is not None]
        if objs:
            return objs
        out = []
        seen = set()
        for nk in self.ui._sess_mgr._real_nd_keys_ordered():
            s = self.ui._sess_mgr._session_obj_for_nd_key(nk)
            if s is None:
                continue
            sid = id(s)
            if sid in seen:
                continue
            seen.add(sid)
            out.append(s)
        return out

    def _collect_theme_ui_labels(self) -> list[str]:
        """Non-blank labels for overlap + legend; Session=All includes union (also stripped)."""
        theme = getattr(self, '_current_theme', 'segments')
        acc = {str(lb).strip() for _, _, lb in self._epoch_bounds if str(lb).strip()}
        if getattr(self, '_session_any_mode', False):
            extra = (getattr(self, '_theme_label_union_all_sessions', None)
                     or {}).get(theme, ())
            acc |= {str(x).strip() for x in (extra or ()) if str(x).strip()}
        out = sorted(acc)
        if not out and theme != 'segments' and theme in getattr(self, '_themes', {}):
            return [theme]
        return out

    def _update_overlap_ui(self):
        """Update the label-filter dropdown for the current theme."""
        theme = getattr(self, '_current_theme', 'segments')
        all_labels = self._collect_theme_ui_labels()
        if len(all_labels) > 1:
            self._label_combo['values'] = ['All'] + all_labels + ['NONE']
            self._label_var.set('All')
        else:
            # Single-label theme (e.g. ripple): show theme name + NONE
            display_name = theme if theme != 'segments' else all_labels[0] if all_labels else 'segments'
            self._label_combo['values'] = [display_name, 'NONE']
            self._label_var.set(display_name)
        self._overlap_row.pack(side=tk.LEFT, padx=(8, 0))
        self._active_label = None
        # Reset label color cache so it rebuilds for new theme
        self._label_colors = None
        self._update_legend()

    def _init_times(self):
        """Populate epoch bounds from the selected theme or edge_times."""
        theme = getattr(self, '_current_theme', 'segments')

        if theme != 'segments' and theme in self._themes:
            # Use Epoch object directly
            epoch = self._themes[theme]
            labs = [str(x).strip() for x in epoch.labels]
            self._epoch_bounds = [
                (float(s), float(e), lb)
                for s, e, lb in zip(epoch.starts, epoch.stops, labs)]
            unique_nonblank = {lb for lb in labs if lb}
            # No usable labels (e.g. ripple): collapse to theme name for UI/chips
            if len(unique_nonblank) <= 1:
                self._epoch_bounds = [
                    (s, e, theme) for s, e, _ in self._epoch_bounds]
            self._total_sec = (float(epoch.stops.max())
                                  if len(epoch.stops) else 1.0)
            self._update_overlap_ui()
            return

        # Default: use CCG segment edge_times
        et = self.ui.ccg_ptr.edge_times
        cols = et.columns.tolist()

        # Find start/stop columns by common name conventions
        def _find_col(candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None

        start_col = _find_col(['start', 't_start', 'start_time', 'start_s'])
        stop_col  = _find_col(['stop',  't_end',   'end_time',   'stop_s', 'end'])

        self._epoch_bounds = []
        if start_col and stop_col:
            for _, row in et.iterrows():
                t0 = float(row[start_col])
                t1 = float(row[stop_col])
                self._epoch_bounds.append((t0, t1, str(row['label'])))
            self._total_sec = (
                max((b[1] for b in self._epoch_bounds), default=1.0)
                if self._epoch_bounds else 1.0)
        else:
            # Fall back: reconstruct from cumulative effective_time_hours
            t = 0.0
            for _, row in et.iterrows():
                dur = float(row['effective_time_hours']) * 3600.0
                self._epoch_bounds.append((t, t + dur, str(row['label'])))
                t += dur
            self._total_sec = t if t > 0 else 1.0
        self._update_overlap_ui()

    # ── Event handlers ──────────────────────────────────────────────────────

    def _on_theme_change(self, _event=None):
        """Handle theme combobox selection — repopulate epoch bounds."""
        # Save current legend toggle state before switching away
        old_theme = getattr(self, '_current_theme', None)
        if old_theme:
            toggles = getattr(self, '_legend_toggles', {})
            if toggles:
                self._per_theme_label_state[old_theme] = {
                    lb: v.get() for lb, v in toggles.items()}
        theme = self._theme_var.get()
        self._current_theme = theme
        # Swap filter button to the new theme's BoolVar (creates one if first visit)
        if self._filter_btn is not None:
            if theme not in self._theme_flag_vars:
                self._theme_flag_vars[theme] = tk.BooleanVar(value=False)
            self._filter_btn.configure(variable=self._theme_flag_vars[theme])
        # Reset handles
        self._slider_t_start = None
        self._slider_t_end = None
        if hasattr(self, '_start_var'):
            self._start_var.set("00:00:00")
        if hasattr(self, '_end_var'):
            self._end_var.set("00:00:00")
        # Reset zoom
        self._zoom_start = None
        self._zoom_end = None
        self._zoom_start_var.set("00:00:00")
        self._zoom_end_var.set("00:00:00")
        self._zoom_frame.pack_forget()
        # Reset label color cache for new theme
        self._label_colors = None
        self._init_times()
        self._redraw()

    def _on_label_change(self, _event=None):
        """Handle overlap label combobox selection."""
        val = self._label_var.get()
        if val == 'All':
            self._active_label = None
        elif val == self._TS_NONE:
            self._active_label = self._TS_NONE
        else:
            # Could be a real label or the theme display name (single-label themes)
            all_labels = self._collect_theme_ui_labels()
            if val in all_labels:
                self._active_label = val
            else:
                # Theme display name for single-label theme → show all intervals
                self._active_label = None
        self._redraw()

    def _on_label_reset(self):
        """Revert to showing all labels."""
        self._active_label = None
        vals = list(self._label_combo['values'])
        self._label_var.set(vals[0] if vals else 'All')
        self._redraw()

    def _on_tool_change(self):
        """Handle toolbar selection / lock toggle."""
        locked = self._lock_var.get()
        selection = self._tool_var.get() == 'selection'
        if locked:
            self.ts_canvas.config(cursor='arrow')
        elif selection:
            self.ts_canvas.config(cursor='plus')
        else:
            self.ts_canvas.config(cursor='crosshair')
        # Hide zoom panel when selection is off
        if not selection:
            self._zoom_frame.pack_forget()
            self._zoom_start = None
            self._zoom_end = None
        self._redraw()

    def _on_time_slider_set(self):
        spec = self.ui._custom_mgr._build_custom_spec(for_all=False)
        if spec is None:
            return
        filter_state = spec.get('filter_state', {})
        # Resolve sentinels for current session (use min/max times, not first/last table row)
        t_sess_start, t_sess_end = self.ui._sess_mgr._session_wall_clock_extent_for_key(self.ui.key)
        t0 = self.ui._resolve_time(spec['t0'], t_sess_start, t_sess_end)
        t1 = self.ui._resolve_time(spec['t1'], t_sess_start, t_sess_end)
        lone = self.ui._single_exclusive_segment_filter_label(filter_state)
        if lone is not None:
            span = self.ui._union_span_for_segment_label(self.ui.key, lone)
            if span is not None:
                t0, t1 = span[0], span[1]
                t0 = self.ui._resolve_time(t0, t_sess_start, t_sess_end)
                t1 = self.ui._resolve_time(t1, t_sess_start, t_sess_end)
        self._slider_t_start = t0
        self._slider_t_end = t1
        n_splits = max(1, int(spec.get('n_splits') or 1))
        overlap_sec = max(0.0, float(spec.get('overlap_sec') or 0.0))
        chunks = self.ui._split_time_range(t0, t1, n_splits, overlap_sec, spec['name'])
        split_bid = None
        if n_splits > 1:
            split_bid = self._batch_next_id
            self._batch_next_id += 1
        split_names: list[str] = []
        queued = 0
        flagged = [tn for tn, v in getattr(self, '_theme_flag_vars', {}).items() if v.get()]

        for chunk_t0, chunk_t1, chunk_name in chunks:
            if flagged:
                per_theme = []
                ok = True
                for tn in flagged:
                    active = self._active_labels_for_theme(tn)
                    ivs = sorted(
                        (max(s, chunk_t0), min(e, chunk_t1))
                        for s, e, lb in self._all_theme_bounds.get(tn, [])
                        if lb in active and min(e, chunk_t1) > max(s, chunk_t0)
                    )
                    if not ivs:
                        ok = False
                        break
                    merged = [list(ivs[0])]
                    for s, e in ivs[1:]:
                        if s <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], e)
                        else:
                            merged.append([s, e])
                    per_theme.append([(s, e) for s, e in merged])
                if not ok:
                    continue
                result_ivs = per_theme[0]
                for other in per_theme[1:]:
                    result_ivs = intersect_intervals(result_ivs, other)
                if not result_ivs:
                    continue
                intervals = result_ivs
                active_duration = sum(e - s for s, e in result_ivs)
            else:
                bs_result = self._brain_state_intervals(chunk_t0, chunk_t1)
                if bs_result is False:
                    continue
                intervals, active_duration = bs_result
            metadata = {
                'name': chunk_name,
                'theme': filter_state.get('theme', 'segments'),
                'labels': filter_state.get('labels', {}),
                'scope': spec.get('scope', self.ui._setup_mgr._current_session_str()),
                'session': str(self.ui.key.session),
                'timing': {'t0': chunk_t0, 't1': chunk_t1},
            }
            ok = self.ui._enqueue_custom_ccg_task(
                key=self.ui.key,
                t0=chunk_t0,
                t1=chunk_t1,
                name=chunk_name,
                intervals=intervals,
                active_duration=active_duration,
                filter_state=filter_state,
                metadata=metadata,
                auto_save=True,
                load_into_ui=True,
                batch_id=split_bid,
            )
            if ok:
                queued += 1
                if split_bid is not None:
                    split_names.append(chunk_name)
        if split_bid is not None and split_names:
            self._batch_counts[split_bid] = len(split_names)
            self._batch_names[split_bid] = split_names
        if queued:
            self.ui._custom_mgr.state._record_custom_ccg_suggestion(spec)
            label = spec['name'] if n_splits == 1 else f"{spec['name']} ({queued} chunks)"
            self._status_var.set(f"Queued: {label}")
            self.ui._custom_mgr._custom_ccg_start_next()

    def _on_time_slider_apply_multiple_sessions(self):
        spec = self.ui._custom_mgr._build_custom_spec(for_all=False)
        if spec is None:
            return
        all_nd_keys = self.ui._sess_mgr._real_nd_keys_ordered()
        if not all_nd_keys:
            messagebox.showinfo("Sessions", "No sessions available.")
            return
        session_labels = [str(nk.session) for nk in all_nd_keys]
        current_sess = str(self.ui.key.session) if not getattr(self, '_session_any_mode', False) else None
        selected = self.ui._pick_sessions_dialog(
            title="Apply custom CCG to sessions",
            sessions=session_labels,
            current_session=current_sess,
        )
        if not selected:
            return
        is_all = len(selected) == len(session_labels)
        spec['sessions'] = selected
        spec['scope'] = 'All' if is_all else (
            selected[0] if len(selected) == 1 else ', '.join(selected[:2]) + ('…' if len(selected) > 2 else ''))
        self.ui._custom_mgr.state._record_custom_ccg_suggestion(spec)
        queued = self.ui._queue_custom_ccg_for_spec(
            spec, for_all=is_all, auto_save=True,
            target_sessions=None if is_all else selected,
        )
        if queued:
            sess_label = "all sessions" if is_all else f"{len(selected)} session(s)"
            self._status_var.set(f"Queued {queued} task(s) for {sess_label}")
            self.ui._custom_mgr._custom_ccg_start_next()
        else:
            self._status_var.set("No missing custom CCGs for selected sessions")

    def _on_time_slider_clear(self):
        self.ui._custom_segments.clear()
        self._status_var.set("")
        # Reset time selection
        self._slider_t_start = None
        self._slider_t_end = None
        self._start_var.set("00:00:00")
        self._end_var.set("00:00:00")
        # Reset zoom
        self._zoom_start = None
        self._zoom_end = None
        self._zoom_start_var.set("00:00:00")
        self._zoom_end_var.set("00:00:00")
        self._zoom_frame.pack_forget()
        # Reset to first real segment
        self.ui.current_segment = self.ui._seg_name(0)
        self.ui._build_sig_chips()
        self.ui._update_segment_label()
        self.ui._plot_mgr.update_plot()

    def _on_zoom_range_set(self):
        """Set zoom range from the zoom time entry boxes."""
        try:
            t0 = self._hms_to_sec(self._zoom_start_var.get())
            t1 = self._hms_to_sec(self._zoom_end_var.get())
        except (ValueError, IndexError):
            return
        if t1 <= t0:
            return
        self._zoom_start = max(0.0, min(t0, self._total_sec))
        self._zoom_end = max(0.0, min(t1, self._total_sec))
        self._zoom_start_var.set(self._sec_to_hms(self._zoom_start))
        self._zoom_end_var.set(self._sec_to_hms(self._zoom_end))
        self._redraw()

    # ── Coordinate helpers ──────────────────────────────────────────────────

    def _t_to_x(self, t: float) -> int:
        w = max(self.ts_canvas.winfo_width(), 20)
        return int((t / max(self._total_sec, 1)) * (w - 20) + 10)

    def _x_to_t(self, x: int) -> float:
        w = max(self.ts_canvas.winfo_width(), 20)
        return max(0.0, min(self._total_sec,
                            (x - 10) / max(w - 20, 1) * self._total_sec))

    def _hms_to_sec(self, hms: str) -> float:
        s = hms.strip().lower()
        if s == 'start':
            return 0.0
        if s == 'end':
            return float(getattr(self, '_total_sec', 0))
        parts = s.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    @staticmethod

    def _sec_to_hms(sec) -> str:
        if sec is None:
            return "end"
        sec = int(sec)
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    # ── Draw / render ───────────────────────────────────────────────────────

    def _label_color_map(self):
        """Return {label: color} mapping — consistent color per unique label."""
        all_labels = self._collect_theme_ui_labels()
        expected = set(all_labels) | {'NONE'}
        if getattr(self, '_label_colors', None) is None or expected != set(self._label_colors.keys()):
            self._label_colors = {
                lbl: self._TS_COLORS[i % len(self._TS_COLORS)]
                for i, lbl in enumerate(all_labels)
            }
            self._label_colors['NONE'] = self._TS_NONE_COLOR
        return self._label_colors

    def _visible_bounds(self):
        """Return epoch bounds filtered by active label (or all if None).
        NONE returns gaps between all epoch intervals."""
        if self._active_label is None:
            return self._epoch_bounds
        if self._active_label == self._TS_NONE:
            return self._compute_gaps()
        return [(t0, t1, lbl) for t0, t1, lbl in self._epoch_bounds
                if lbl == self._active_label]

    def _compute_gaps(self):
        """Compute time gaps between all epoch intervals."""
        if not self._epoch_bounds:
            return [(0, self._total_sec, 'NONE')]
        # Merge overlapping intervals first
        sorted_bounds = sorted(self._epoch_bounds, key=lambda x: x[0])
        merged = []
        cur_start, cur_end = sorted_bounds[0][0], sorted_bounds[0][1]
        for t0, t1, _ in sorted_bounds[1:]:
            if t0 <= cur_end:
                cur_end = max(cur_end, t1)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = t0, t1
        merged.append((cur_start, cur_end))
        # Build gaps
        gaps = []
        if merged[0][0] > 0:
            gaps.append((0, merged[0][0], 'NONE'))
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i + 1][0]
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end, 'NONE'))
        if merged[-1][1] < self._total_sec:
            gaps.append((merged[-1][1], self._total_sec, 'NONE'))
        return gaps

    def _draw_epochs(self, canvas, t_to_x, bounds, h):
        """Draw epoch rectangles on a canvas using given t_to_x mapping."""
        cmap = self._label_color_map()
        y_bot = h - 16  # leave room for time axis
        for t0, t1, lbl in bounds:
            x0, x1 = t_to_x(t0), t_to_x(t1)
            color = cmap.get(lbl, '#E0E0E0')
            canvas.create_rectangle(x0, 6, x1, y_bot,
                                    fill=color, outline='#90A4AE')
            if x1 - x0 > 22:
                canvas.create_text((x0 + x1) // 2, (6 + y_bot) // 2,
                                   text=lbl, font=('Arial', 7), fill='#333')

    def _draw_time_axis(self, canvas, t_to_x, t_min, t_max, h):
        """Draw time axis tick marks and labels at the bottom of a canvas."""
        w = canvas.winfo_width()
        if w < 40:
            return
        span = t_max - t_min
        if span <= 0:
            return
        # Choose tick interval: aim for ~6-10 ticks
        raw = span / 8.0
        # Round to nice intervals (in seconds)
        nice_intervals = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
                          900, 1800, 3600, 7200, 14400, 28800]
        tick_step = nice_intervals[-1]
        for ni in nice_intervals:
            if ni >= raw:
                tick_step = ni
                break
        # Draw ticks
        y_tick = h - 2
        y_label = h - 1
        t = (int(t_min / tick_step) + 1) * tick_step if t_min % tick_step != 0 else t_min
        # Always draw first and last
        endpoints = {t_min, t_max}
        drawn_x = []
        for tp in sorted(endpoints):
            x = t_to_x(tp)
            if 5 <= x <= w - 5:
                canvas.create_line(x, y_tick - 4, x, y_tick, fill='#888')
                canvas.create_text(x, y_label, text=self._sec_to_hms(tp),
                                   font=('Courier', 6), fill='#666', anchor='s')
                drawn_x.append(x)
        while t <= t_max:
            x = t_to_x(t)
            # Skip if too close to an already-drawn label
            if 5 <= x <= w - 5 and all(abs(x - dx) > 38 for dx in drawn_x):
                canvas.create_line(x, y_tick - 4, x, y_tick, fill='#888')
                canvas.create_text(x, y_label, text=self._sec_to_hms(t),
                                   font=('Courier', 6), fill='#666', anchor='s')
                drawn_x.append(x)
            t += tick_step

    def _draw_handles(self, canvas, t_to_x, h, t_start, t_end,
                         color_start='#1565C0', color_end='#B71C1C'):
        """Draw cursor handles and selection range on a canvas."""
        for t, color in [(t_start, color_start), (t_end, color_end)]:
            if t is not None:
                x = t_to_x(t)
                canvas.create_line(x, 2, x, h - 2, fill=color, width=2)
                canvas.create_polygon(x - 5, 2, x + 5, 2, x, 10,
                                      fill=color, outline='')
        if t_start is not None and t_end is not None:
            x0, x1 = t_to_x(t_start), t_to_x(t_end)
            canvas.create_rectangle(x0, 8, x1, h - 8,
                                    fill='', outline=color_start,
                                    width=2, dash=(4, 2))

    def _draw_radiate_lines(self):
        """Draw radiating lines connecting zoom region on main canvas to zoom canvas."""
        rc = self._radiate_canvas
        rc.delete('all')
        if self._zoom_start is None or self._zoom_end is None:
            return
        w = rc.winfo_width()
        rh = rc.winfo_height()
        if w < 20:
            return

        # Top points: zoom region edges on main canvas
        zx0_main = self._t_to_x(self._zoom_start)
        zx1_main = self._t_to_x(self._zoom_end)

        # Bottom points: full width of zoom canvas
        zx0_zoom = 10
        zx1_zoom = max(self._zoom_canvas.winfo_width(), 20) - 10

        # Draw radiating lines
        rc.create_line(zx0_main, 0, zx0_zoom, rh,
                       fill='#E65100', width=1, dash=(3, 3))
        rc.create_line(zx1_main, 0, zx1_zoom, rh,
                       fill='#E65100', width=1, dash=(3, 3))

    def _update_legend(self):
        """Populate legend row with toggle-able swatches for each unique label."""
        frame = self._legend_frame
        for w in frame.winfo_children():
            w.destroy()
        cmap = self._label_color_map()
        self._legend_toggles = {}
        _fs = max(7, self.ui._settings_mgr.min_font_size())
        _saved = self._per_theme_label_state.get(self._current_theme, {})
        for lbl, color in cmap.items():
            initial = _saved.get(lbl, True)
            var = tk.BooleanVar(value=initial)
            self._legend_toggles[lbl] = var
            # Combined swatch+label as a single clickable button
            btn = tk.Frame(frame, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=(4, 6), pady=1)
            swatch = tk.Frame(btn, width=12, height=10,
                              bg=color if initial else '#D0D0D0',
                              highlightbackground='#90A4AE', highlightthickness=1)
            swatch.pack(side=tk.LEFT, padx=(0, 2))
            swatch.pack_propagate(False)
            lbl_w = tk.Label(btn, text=lbl, font=('Arial', _fs),
                             fg='#333' if initial else '#AAA',
                             relief=tk.RAISED if initial else tk.SUNKEN,
                             bd=1, padx=2)
            lbl_w.pack(side=tk.LEFT)

            def _toggle(v=var, s=swatch, l=lbl_w, c=color):
                v.set(not v.get())
                if v.get():
                    s.config(bg=c)
                    l.config(fg='#333', relief=tk.RAISED)
                else:
                    s.config(bg='#D0D0D0')
                    l.config(fg='#AAA', relief=tk.SUNKEN)
            for w in (swatch, lbl_w, btn):
                w.bind('<Button-1>', lambda e, t=_toggle: t())
        frame.pack(fill=tk.X, padx=4, pady=(1, 0),
                   before=self._main_canvas_frame)

    def _redraw(self, event=None):
        c = self.ts_canvas
        c.delete('all')
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20:
            return

        if getattr(self, '_session_any_mode', False):
            c.create_text(
                w // 2, h // 2,
                text="All sessions view — no single behavioral timeline to display.\n"
                     "Use 'Set' + 'Apply to Multiple Sessions' to compute custom CCGs for sessions.",
                anchor='center', font=('Arial', 9), fill='#555', justify='center')
            return

        if not self._epoch_bounds:
            return

        self._draw_epochs(c, self._t_to_x, self._visible_bounds(), h)
        self._draw_time_axis(c, self._t_to_x, 0, self._total_sec, h)

        # Draw select-mode handles (custom window cursors)
        self._draw_handles(c, self._t_to_x, h,
                              self._slider_t_start, self._slider_t_end)

        # Draw selection/zoom handles (orange) when selection tool active
        if self._tool_var.get() == 'selection':
            self._draw_handles(c, self._t_to_x, h,
                                  self._zoom_start, self._zoom_end,
                                  color_start='#E65100', color_end='#BF360C')
            # Shade zoom region
            if self._zoom_start is not None and self._zoom_end is not None:
                zx0 = self._t_to_x(self._zoom_start)
                zx1 = self._t_to_x(self._zoom_end)
                c.create_rectangle(zx0, 4, zx1, h - 4,
                                   fill='#FFF3E0', outline='#E65100',
                                   width=1, stipple='gray25')
            self._zoom_redraw()
            self._draw_radiate_lines()

    def _zoom_t_to_x(self, t: float) -> int:
        """Map time to x within the zoom canvas, using zoom region bounds."""
        w = max(self._zoom_canvas.winfo_width(), 20)
        z0 = self._zoom_start if self._zoom_start is not None else 0
        z1 = self._zoom_end if self._zoom_end is not None else self._total_sec
        span = max(z1 - z0, 1e-6)
        return int(((t - z0) / span) * (w - 20) + 10)

    def _zoom_redraw(self, event=None):
        """Redraw the zoomed-in detail canvas."""
        zc = self._zoom_canvas
        zc.delete('all')
        if self._zoom_start is None or self._zoom_end is None:
            return
        w = zc.winfo_width()
        h = zc.winfo_height()
        if w < 20:
            return

        z0, z1 = self._zoom_start, self._zoom_end

        # Filter epoch bounds that overlap the zoom region (respects active label)
        zoomed_bounds = [
            (max(t0, z0), min(t1, z1), lbl)
            for t0, t1, lbl in self._visible_bounds()
            if t1 > z0 and t0 < z1
        ]
        self._draw_epochs(zc, self._zoom_t_to_x, zoomed_bounds, h)
        self._draw_time_axis(zc, self._zoom_t_to_x, z0, z1, h)

        # Draw select-mode handles within zoom view
        self._draw_handles(zc, self._zoom_t_to_x, h,
                              self._slider_t_start, self._slider_t_end)

    # ── Mouse interaction ───────────────────────────────────────────────────

    def _snap_t(self, t: float, canvas_x: int) -> float:
        """Snap t to the nearest epoch boundary if within 25px."""
        if self._snap_var.get() and self._epoch_bounds:
            for t0, t1, _ in self._epoch_bounds:
                for bt in (t0, t1):
                    if abs(self._t_to_x(bt) - canvas_x) <= 25:
                        return bt
        return t

    def _update_handle(self, canvas_x: int, snap: bool = False):
        t = self._x_to_t(canvas_x)
        if snap:
            t = self._snap_t(t, canvas_x)
        if self._slider_dragging == 'start':
            cap = self._slider_t_end if self._slider_t_end is not None else self._total_sec
            self._slider_t_start = min(t, cap)
            self._start_var.set(self._sec_to_hms(self._slider_t_start))
        elif self._slider_dragging == 'end':
            floor = self._slider_t_start if self._slider_t_start is not None else 0.0
            self._slider_t_end = max(t, floor)
            self._end_var.set(self._sec_to_hms(self._slider_t_end))
        self._redraw()

    def _mouse_press(self, event):
        if self._lock_var.get():
            return  # all cursor interaction disabled
        selection = self._tool_var.get() == 'selection'
        if selection:
            self._zoom_mouse_press(event)
            return
        # Default: custom CCG window tool
        if self._slider_t_start is None:
            self._slider_dragging = 'start'
        elif self._slider_t_end is None:
            self._slider_dragging = 'end'
        else:
            xs = self._t_to_x(self._slider_t_start)
            xe = self._t_to_x(self._slider_t_end)
            self._slider_dragging = ('start'
                                     if abs(event.x - xs) <= abs(event.x - xe)
                                     else 'end')
        self._update_handle(event.x)

    def _mouse_drag(self, event):
        if self._lock_var.get():
            return
        if self._tool_var.get() == 'selection':
            self._zoom_mouse_drag(event)
            return
        self._update_handle(event.x)

    def _mouse_release(self, event):
        if self._lock_var.get():
            return
        if self._tool_var.get() == 'selection':
            self._zoom_mouse_release(event)
            return
        self._update_handle(event.x, snap=True)
        self._slider_dragging = None

    def _zoom_mouse_press(self, event):
        if self._zoom_start is None:
            self._zoom_dragging = 'start'
        elif self._zoom_end is None:
            self._zoom_dragging = 'end'
        else:
            xs = self._t_to_x(self._zoom_start)
            xe = self._t_to_x(self._zoom_end)
            self._zoom_dragging = ('start'
                                      if abs(event.x - xs) <= abs(event.x - xe)
                                      else 'end')
        self._zoom_update(event.x)

    def _zoom_mouse_drag(self, event):
        self._zoom_update(event.x)

    def _zoom_mouse_release(self, event):
        self._zoom_update(event.x, snap=True)
        self._zoom_dragging = None
        # Show zoom canvas once both ends are placed
        if self._zoom_start is not None and self._zoom_end is not None:
            self._zoom_frame.pack(fill=tk.X, padx=4, pady=(0, 0),
                                     after=self._ccg_ctrl)
            self._zoom_redraw()
            self._draw_radiate_lines()

    def _zoom_update(self, canvas_x: int, snap: bool = False):
        t = self._x_to_t(canvas_x)
        if snap:
            t = self._snap_t(t, canvas_x)
        if self._zoom_dragging == 'start':
            cap = self._zoom_end if self._zoom_end is not None else self._total_sec
            self._zoom_start = min(t, cap)
            self._zoom_start_var.set(self._sec_to_hms(self._zoom_start))
        elif self._zoom_dragging == 'end':
            floor = self._zoom_start if self._zoom_start is not None else 0.0
            self._zoom_end = max(t, floor)
            self._zoom_end_var.set(self._sec_to_hms(self._zoom_end))
        self._redraw()

    # ── Brain state / interval filtering ────────────────────────────────────

    def _brain_state_intervals(self, t0, t1):
        """Validate brain-state toggles and return the active intervals (main-thread, fast).

        Returns:
            (None, t1-t0)        — no restriction (all toggles on or no toggles)
            (intervals, active_sec) — filtered intervals list with total active seconds
            False                 — abort (no active labels or no intervals in range)
        """
        toggles = getattr(self, '_legend_toggles', {})
        if not toggles or all(v.get() for v in toggles.values()):
            return (None, t1 - t0)
        active_labels = {lbl for lbl, v in toggles.items() if v.get()}
        if not active_labels:
            messagebox.showwarning("Brain-state filter",
                                   "All epoch labels are toggled off. "
                                   "Enable at least one label to compute CCG.")
            return False
        if self.ui.neurons is None:
            return (None, t1 - t0)
        available_labels = {lbl for _, _, lbl in self._epoch_bounds}
        none_active = 'NONE' in active_labels
        real_active = active_labels - {'NONE'}
        real_labels = real_active & available_labels
        intervals = []
        for s, e, lbl in self._epoch_bounds:
            if lbl in real_labels:
                s_clipped, e_clipped = max(s, t0), min(e, t1)
                if e_clipped > s_clipped:
                    intervals.append((s_clipped, e_clipped))
        if none_active:
            epoch_times = sorted(
                (max(s, t0), min(e, t1))
                for s, e, _ in self._epoch_bounds
                if min(e, t1) > max(s, t0))
            cursor = t0
            for es, ee in epoch_times:
                if es > cursor:
                    intervals.append((cursor, es))
                cursor = max(cursor, ee)
            if cursor < t1:
                intervals.append((cursor, t1))
        if not intervals:
            messagebox.showwarning("Brain-state filter",
                                   "No active epoch intervals in the selected time range.")
            return False
        active_sec = sum(e - s for s, e in intervals)
        return (intervals, active_sec)

    def _apply_brain_state_intervals(self, intervals, t0, t1, neurons_obj=None):
        """Filter self.ui.neurons to the given intervals and return a new Neurons object.
        Called in the background worker thread — deepcopy stays off the main thread.
        """
        source_neurons = neurons_obj if neurons_obj is not None else self.ui.neurons
        neurons = deepcopy(source_neurons)
        filtered_trains = []
        for st in neurons.spiketrains:
            mask = np.zeros(len(st), dtype=bool)
            for s, e in intervals:
                mask |= (st >= s) & (st <= e)
            filtered_trains.append(st[mask])
        return Neurons(
            spiketrains=filtered_trains,
            t_stop=t1, t_start=t0,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids,
            neuron_type=neurons.neuron_type,
            waveforms=neurons.waveforms,
            waveforms_amplitude=neurons.waveforms_amplitude,
            peak_channels=getattr(neurons, 'peak_channels', None),
            shank_ids=getattr(neurons, 'shank_ids', None),
            metadata=neurons.metadata,
        )

    # ── Custom CCG save / load ──────────────────────────────────────────────

    def _save_custom_ccg(self):
        """Manage auto-saved custom segments: list by name, delete any the user unchecks."""
        buckets = getattr(self.ui, '_custom_segments_by_session', None) or {}

        # List all .npz files in the cache dir by filename only — no array loading
        pattern = os.path.join(self.ui._ccg_cache_dir, "*.npz")
        all_paths = sorted(_glob.glob(pattern))
        handles: list[tuple[str, str]] = [
            (os.path.basename(p)[:-4], p) for p in all_paths
        ]

        if not handles:
            messagebox.showinfo("Custom CCG segments",
                                "No auto-saved custom segments found.")
            return

        # Dialog: all segments pre-checked; uncheck to delete
        win = tk.Toplevel(self.ui.root)
        win.title("Custom CCG segments")
        win.geometry("400x300")
        win.grab_set()
        ttk.Label(win, text="Saved segments — uncheck to delete from disk:").pack(
            anchor='w', padx=8, pady=(8, 2))
        lb = tk.Listbox(win, selectmode=tk.MULTIPLE, height=10)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        for handle, _ in handles:
            lb.insert(tk.END, handle)
        lb.select_set(0, tk.END)

        confirmed: list[bool] = []

        def _ok():
            confirmed.append(True)
            win.destroy()

        btn_f = ttk.Frame(win)
        btn_f.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_f, text="Keep selected", command=_ok).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btn_f, text="Cancel",
                   command=win.destroy).pack(side=tk.RIGHT)
        win.wait_window(win)
        if not confirmed:
            return

        keep_idx = set(lb.curselection())
        to_delete = [(handle, p) for i, (handle, p) in enumerate(handles)
                     if i not in keep_idx]
        if not to_delete:
            return

        delete_paths = {p for _, p in to_delete}
        delete_handles = {h for h, _ in to_delete}
        for p in delete_paths:
            try:
                os.remove(p)
            except OSError:
                pass

        # Remove deleted entries from in-memory segment lists
        from neuropy.analyses.custom_ccg import CustomSegment
        for lst in buckets.values():
            lst[:] = [
                cs for cs in lst
                if isinstance(cs, CustomSegment) and (
                    cs.src_path not in delete_paths
                    and re.sub(r'[^A-Za-z0-9_\-]', '_',
                               str(cs.source.name).replace(' ', '_'))
                    not in delete_handles
                )
            ]

        try:
            self.ui._build_sig_chips()
            self.ui._update_segment_label()
        except Exception:
            pass
        self.ui._custom_mgr._emit_inventory_event()
        self.ui._settings_mgr.save_ui_state()
        ts = self
        fn, win = ts._load_custom_ccg_refresh, ts._load_custom_ccg_win
        if fn is not None and win is not None:
            try:
                if win.winfo_exists():
                    fn()
            except Exception:
                ts._load_custom_ccg_win = ts._load_custom_ccg_refresh = None
        self._status_var.set(f"Deleted {len(to_delete)} segment(s)")

    def _load_custom_ccg(self):
        """Scan cache dir for saved custom segments and load selected ones additively."""
        # Archive stale files (those missing total_time_hours) before showing dialog
        n_archived, trash_dir = self.ui._custom_mgr.store._archive_stale_custom_ccgs()
        if n_archived:
            messagebox.showinfo(
                "Stale custom CCGs archived",
                f"{n_archived} old custom CCG file(s) were moved to:\n  {trash_dir}\n\n"
                "These files lack the 'total_time' field required for correct Time Span "
                "normalisation and cannot be loaded. They are preserved in the trash folder.")
        pattern = os.path.join(self.ui._ccg_cache_dir, "*.npz")
        paths = sorted(_glob.glob(pattern))
        if not paths:
            messagebox.showinfo(
                "Load custom CCG",
                "No saved custom CCGs found.")
            return

        win = tk.Toplevel(self.ui.root)
        win.title("Load custom CCG")
        win.geometry("560x380")
        win.grab_set()
        ttk.Label(win, text="Select segments to load (click ▶ to expand details):").pack(
            anchor='w', padx=8, pady=(8, 2))

        tree_frame = ttk.Frame(win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tv = ttk.Treeview(tree_frame, selectmode='extended',
                          yscrollcommand=vsb.set, show='tree')
        vsb.config(command=tv.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tv.column('#0', width=520, stretch=True)

        # tag: detail rows shown in muted color; adapt to dark mode
        _t = self.ui.theme
        _fg_main, _fg_muted, _fg_header, _bg = _t.fg, _t.fg_muted, _t.fg_header, _t.bg
        win.configure(bg=_bg)
        if _t.dark:
            import tkinter.ttk as _ttk
            _s = _ttk.Style(win)
            _s.configure('Dark.Treeview',
                          background=_bg, foreground=_fg_main,
                          fieldbackground=_bg, rowheight=22)
            _s.map('Dark.Treeview', background=[('selected', '#4a6fa5')])
            tv.configure(style='Dark.Treeview')
        tv.tag_configure('detail', foreground=_fg_muted)
        tv.tag_configure('file',   foreground=_fg_main)
        tv.tag_configure('group',  foreground=_fg_header, font=('TkDefaultFont', 9, 'bold'))

        name_cov, _n_total_sessions = self.ui._custom_mgr.store._custom_ccg_name_session_coverage()
        _n_sess_denom = max(1, _n_total_sessions)

        def _dur_str(dur):
            if dur != dur:  return ""          # nan
            if dur >= 60:   return f"  ⏱ {dur/60:.1f}min"
            return f"  ⏱ {dur:.0f}s"

        def _parse_meta(p, *, child_session_line: bool = False):
            """Return (summary_line, [bullet_strings]) from npz metadata."""
            base = os.path.basename(p)
            file_sess = base.split("__", 1)[0] if "__" in base else ""
            parts = base.replace('.npz', '').rsplit('__', 1)
            date_str = parts[1] if len(parts) > 1 else ''
            safe_name = parts[0].replace('_', ' ') if parts else base
            bullets = []
            t0, t1 = None, None
            try:
                m = np.load(p, allow_pickle=False)
                if 'name_' in m:
                    safe_name = str(m['name_'])
                t0 = float(m['t0_']) if 't0_' in m else None
                t1 = float(m['t1_']) if 't1_' in m else None
                if t0 is not None and t1 is not None:
                    bullets.append(
                        f"Time range: {self._sec_to_hms(t0)} – {self._sec_to_hms(t1)}")
                act = float(m['active_duration_']) if 'active_duration_' in m else float('nan')
                if act == act:
                    bullets.append(f"Active duration: {act:.1f}s")
                if 'filter_state_' in m:
                    fs = json.loads(str(m['filter_state_']))
                    theme = fs.get('theme', '')
                    labels = fs.get('labels', {})
                    if theme:
                        bullets.append(f"Theme: {theme}")
                    if labels:
                        on  = [l for l, v in labels.items() if v]
                        off = [l for l, v in labels.items() if not v]
                        if on:
                            bullets.append(f"Active labels: {', '.join(on)}")
                        if off:
                            bullets.append(f"Inactive labels: {', '.join(off)}")
                if 'metadata_' in m:
                    md = json.loads(str(m['metadata_']))
                    src = md.get('session')
                    if src:
                        bullets.append(f"Derived from session: {src}")
                has_fr = 'firing_rates' in m
                has_hi = 'ccg_hi' in m
                flags = []
                if has_fr: flags.append("firing rates")
                if has_hi: flags.append("hi-res CCG")
                if flags:
                    bullets.append(f"Contains: {', '.join(flags)}")
            except Exception:
                pass
            if child_session_line:
                tr = ""
                if t0 is not None and t1 is not None:
                    tr = (f"{self._sec_to_hms(t0)}–{self._sec_to_hms(t1)}  ·  ")
                summary = f"{file_sess or '?'}  ·  {tr}[{date_str}]"
            else:
                summary = f"{safe_name}  [{date_str}]"
            return summary, bullets

        # file_meta maps iid → path
        file_meta = {}

        def _populate(path_list):
            for top in tv.get_children():
                tv.delete(top)
            file_meta.clear()
            _groups: dict[str, list[str]] = collections.defaultdict(list)
            for p in path_list:
                spec = self.ui._custom_mgr.store._custom_npz_spec(p) or {}
                nm = str(spec.get('name', '')).strip()
                if not nm:
                    nm = os.path.basename(p).replace('.npz', '')
                _groups[nm].append(p)
            for gname in sorted(_groups.keys(), key=lambda s: s.lower()):
                plist = sorted(
                    _groups[gname],
                    key=lambda pp: (os.path.basename(pp).split('__', 1)[0], pp),
                )
                n_have = len(name_cov.get(gname, set()))
                parent_text = (
                    f"▶ {gname}  ({n_have}/{_n_sess_denom} sessions)")
                pid = tv.insert('', tk.END, text=parent_text,
                                tags=('group',), open=True)
                for p in plist:
                    summary, bullets = _parse_meta(p, child_session_line=True)
                    iid = tv.insert(pid, tk.END, text=f"☐  {summary}",
                                    tags=('file',), open=False)
                    file_meta[iid] = p
                    for b in bullets:
                        tv.insert(iid, tk.END, text=f"    • {b}", tags=('detail',))

        _populate(paths)

        # Clicking a file row toggles its checkbox marker (selection via Treeview selection)
        # Prevent detail rows from being selected
        def _on_click(event):
            iid = tv.identify_row(event.y)
            if iid:
                tags = tv.item(iid, 'tags')
                if 'detail' in tags or 'group' in tags:
                    tv.selection_remove(iid)

        tv.bind('<ButtonRelease-1>', _on_click)

        def _refresh_list():
            new_paths = sorted(_glob.glob(pattern))
            _populate(new_paths)
            self.ui._custom_mgr._emit_inventory_event()
            if not file_meta:
                win.destroy()

        def _selected_file_iids():
            return [iid for iid in tv.selection()
                    if 'file' in tv.item(iid, 'tags')]

        def _delete():
            sel = _selected_file_iids()
            if not sel:
                return
            names = [tv.item(iid, 'text').lstrip('☐ ').split('  [')[0] for iid in sel]
            if not messagebox.askyesno(
                    "Delete files",
                    f"Permanently delete {len(sel)} file(s)?\n"
                    + "\n".join(f"  • {n}" for n in names)):
                return
            for iid in sel:
                try:
                    os.remove(file_meta[iid])
                except Exception as ex:
                    print(f"[LoadCustomCCG] delete failed: {ex}")
            _refresh_list()
            self.ui._custom_mgr._emit_inventory_event()

        def _ok():
            sel = _selected_file_iids()
            if not sel:
                win.destroy()
                return
            added = []
            touched_view = False
            last_idx = 0
            from neuropy.analyses.custom_ccg import CustomSegment
            for iid in sel:
                p = file_meta[iid]
                try:
                    cs = CustomSegment.load(p)
                    bn = os.path.basename(p)
                    file_sess = bn.split("__", 1)[0] if "__" in bn else str(self.ui.key.session)
                    lst = self.ui._custom_segments_by_session.setdefault(file_sess, [])
                    last_idx, _ = self.ui._custom_mgr.store._upsert_custom_segment_by_name(lst, cs)
                    if lst is self.ui._custom_segments:
                        touched_view = True
                    added.append(cs.source.name)
                except Exception as ex:
                    print(f"[LoadCustomCCG] failed to load {p}: {ex}")
            win.destroy()
            if added and touched_view:
                self.ui._build_sig_chips()
                self.ui.current_segment = self.ui._seg_name(self.ui.n_segments + 1 + last_idx)
                self.ui._clamp_current_segment_for_session()
                self.ui._update_segment_label()
                self.ui._plot_mgr.update_plot()
                if hasattr(self, '_status_var'):
                    from collections import Counter as _Ctr
                    _cnts = _Ctr(added)
                    _parts = [f"{n} ({c})" if c > 1 else n for n, c in _cnts.items()]
                    self._status_var.set(f"Loaded: {', '.join(_parts)}")
                self.ui._settings_mgr.save_ui_state()
            elif added:
                if hasattr(self, '_status_var'):
                    self._status_var.set(
                        f"Loaded {len(added)} segment(s) for other session(s); "
                        "switch pairs to that session to view chips.")
                self.ui._settings_mgr.save_ui_state()

        def _clear_load_dialog_ref(_e=None):
            if getattr(self, '_load_custom_ccg_win', None) is win:
                self._load_custom_ccg_win = None
                self._load_custom_ccg_refresh = None

        self._load_custom_ccg_win = win
        self._load_custom_ccg_refresh = _refresh_list
        win.bind(
            '<Destroy>',
            lambda e: _clear_load_dialog_ref() if e.widget is win else None,
        )

        btn_f = ttk.Frame(win)
        btn_f.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_f, text="Refresh list", command=_refresh_list).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Load selected", command=_ok).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btn_f, text="Delete selected",
                   command=_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Cancel",
                   command=win.destroy).pack(side=tk.RIGHT)
