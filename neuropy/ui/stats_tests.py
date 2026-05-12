"""Stats test panel for CCGReviewUI.

Provides a Toplevel dialog for running statistical comparisons
(t-tests) on connection strengths, firing rates, CCG baselines, etc.
"""

import contextlib
import datetime
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_ALL_SEGS = "All segments"
_ADMITTED_GROUP = "__admitted__"

_DATA_TYPES = [
    "Conn Strength",
    "Firing Rate",
    "Baseline",
    "Peak Width",    # grayed — not yet implemented
    "Peak Center",   # grayed — not yet implemented
]
_DATA_DISABLED = {"Peak Width", "Peak Center"}

_FIRING_RATE_TYPES = ["pyr", "int", "all"]


def _fmt_pair(p) -> str:
    """Format a pair identifier for outlier display."""
    if isinstance(p, (list, tuple)):
        if len(p) == 3:
            return f"{p[1]}-{p[2]} ({p[0]})"
        if len(p) == 2:
            return f"{p[0]}-{p[1]}"
    return str(p)


class StatsTestPanel:
    """Toplevel window for running statistical comparisons of CCG connection strengths.

    Each row defines a "group" of (session, conn_type, segment, group label, data type)
    whose values are collected.  A t-test is then run between the first two groups,
    with results shown in-window and optionally exported to a text file.
    """

    def __init__(self, ui):
        self.ui = ui
        self._row_frames: list = []
        self._result_data: dict | None = None
        self._pending_afters: list = []

        self.root = tk.Toplevel(ui.root)
        self.root.title("Stats Tests")
        self.root.geometry("960x580")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._export_btn: ttk.Button | None = None
        self._result_text: tk.Text | None = None

        self._setup()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self):
        if getattr(self.ui, '_stats_panel', None) is self:
            self.ui._stats_panel = None
        for aid in self._pending_afters:
            try:
                self.root.after_cancel(aid)
            except Exception:
                pass
        self._pending_afters.clear()
        self.root.destroy()

    def on_parent_display_option_changed(self):
        """Called by the main UI when display-affecting toggles change (e.g. non-negative)."""
        try:
            if not self.root.winfo_exists():
                return
            self.refresh_session_dropdowns()
            if self._result_data and self._result_data.get('dtype') == "Conn Strength":
                # Re-run so the text + plot reflect the new clamp mode.
                self._pending_afters.append(self.root.after_idle(self._run))
        except Exception:
            pass

    def refresh_session_dropdowns(self):
        """Prune invalid sessions on each row; refresh segment lists."""
        concrete = self._concrete_sessions()
        segments = self._available_segments()
        for r in list(self._row_frames):
            try:
                if not r['frame'].winfo_exists():
                    continue
            except Exception:
                continue
            sl = [s for s in (r.get('sess_list') or []) if s in concrete]
            if not sl and concrete:
                sl = [concrete[0]]
            r['sess_list'] = sl
            r['sess'].set(self._format_sess_list_summary(sl))
            seg_cb = r.get('seg_combo')
            if seg_cb is not None:
                cur_s = r['seg'].get()
                seg_cb.config(values=segments)
                if cur_s not in segments:
                    r['seg'].set(segments[0] if segments else '')

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup(self):
        root = self.root

        # ── Top controls ────────────────────────────────────────────────
        top = ttk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=(6, 2))

        ttk.Label(top, text="Groups", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Label(top, text="Test type:").pack(side=tk.LEFT, padx=(8, 2))
        self._test_type_var = tk.StringVar(value="Pairwise t-test")
        self._test_type_cb = ttk.Combobox(top, textvariable=self._test_type_var,
                     values=["Independent t-test", "Pairwise t-test", "One-way ANOVA + Tukey"],
                     state='readonly', width=24)
        self._test_type_cb.pack(side=tk.LEFT)

        ttk.Label(top, text="Sides:").pack(side=tk.LEFT, padx=(10, 2))
        self._sides_var = tk.StringVar(value="Two-sided")
        sides_cb = ttk.Combobox(
            top, textvariable=self._sides_var,
            values=["Two-sided", "One-sided"],
            state='readonly', width=10)
        sides_cb.pack(side=tk.LEFT)

        # Direction toggle for one-sided tests (alternative is relative to A)
        self._dir_var = tk.StringVar(value="A > B")
        self._dir_btn = ttk.Button(top, textvariable=self._dir_var,
                                   width=7, command=self._toggle_direction)
        self._dir_btn.pack(side=tk.LEFT, padx=(6, 0))

        def _sync_dir_enabled(*_):
            if self._sides_var.get() == "One-sided":
                self._dir_btn.state(['!disabled'])
            else:
                self._dir_btn.state(['disabled'])
        self._sides_var.trace_add('write', _sync_dir_enabled)
        _sync_dir_enabled()

        # Parametric / nonparametric + log-transform toggles (t-tests panel)
        self._nonparam_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="nonparametric",
                        variable=self._nonparam_var).pack(side=tk.LEFT, padx=(12, 2))
        self._log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="log-transform",
                        variable=self._log_var).pack(side=tk.LEFT, padx=(4, 0))

        # ── Column headers ──────────────────────────────────────────────
        hdr = ttk.Frame(root)
        hdr.pack(fill=tk.X, padx=8, pady=(2, 0))
        for col, w in [("Name", 5), ("Session", 22), ("ConnType", 14), ("Segment", 14),
                       ("Group", 14), ("Data", 15), ("", 3)]:
            ttk.Label(hdr, text=col, font=('Arial', 8, 'bold'),
                      width=w, anchor='w').pack(side=tk.LEFT, padx=1)

        # ── Rows ────────────────────────────────────────────────────────
        self._rows_frame = ttk.Frame(root)
        self._rows_frame.pack(fill=tk.X, padx=8)

        # ── Add-group button (above Results) ────────────────────────────
        add_row_frame = ttk.Frame(root)
        add_row_frame.pack(fill=tk.X, padx=8, pady=(4, 0))
        ttk.Button(add_row_frame, text="+ Add group",
                   command=self._add_row).pack(side=tk.LEFT)

        # ── Results (split pane: text | plot) ────────────────────────────
        res_frame = ttk.LabelFrame(root, text="Results")
        res_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        res_pane = ttk.PanedWindow(res_frame, orient=tk.VERTICAL)
        res_pane.pack(fill=tk.BOTH, expand=True)

        # Top: text + scrollbar
        text_frame = ttk.Frame(res_pane)
        _txt_fs = max(9, self.ui._min_font_size())
        self._result_text = tk.Text(
            text_frame, height=10, wrap=tk.WORD,
            font=('Courier', _txt_fs), state=tk.DISABLED,
            relief=tk.FLAT, bg='#FAFAFA')
        sb = ttk.Scrollbar(text_frame, command=self._result_text.yview)
        self._result_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._result_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Bottom: plot
        plot_frame = ttk.Frame(res_pane)
        self._plot_fig = Figure(figsize=(7.5, 2.6), dpi=100)
        self._plot_canvas = FigureCanvasTkAgg(self._plot_fig, master=plot_frame)
        self._plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._plot_canvas.draw()

        res_pane.add(text_frame, weight=2)
        res_pane.add(plot_frame, weight=1)
        self._results_pane = res_pane

        # ── Action buttons ───────────────────────────────────────────────
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Run", command=self._run).pack(side=tk.LEFT, padx=4)
        self._export_btn = ttk.Button(btn_frame, text="Export…",
                                      command=self._export, state=tk.DISABLED)
        self._export_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Load…", command=self._load_result).pack(side=tk.LEFT, padx=4)

        self._add_row()
        self._add_row()
        self.refresh_session_dropdowns()

    def _toggle_direction(self):
        self._dir_var.set("A < B" if self._dir_var.get().strip() == "A > B" else "A > B")

    def _sync_test_type_to_group_count(self):
        n = sum(1 for r in self._row_frames if r['frame'].winfo_exists())
        if n > 2:
            self._test_type_var.set("One-way ANOVA + Tukey")
        else:
            if self._test_type_var.get() == "One-way ANOVA + Tukey":
                self._test_type_var.set("Pairwise t-test")

    # ------------------------------------------------------------------
    # Dropdown data sources
    # ------------------------------------------------------------------

    def _concrete_sessions(self) -> list[str]:
        """Session strings present in ``cd.data`` (stable order), no synthetic ``All``."""
        seen: list[str] = []
        for k in getattr(self.ui.cd, 'data', {}).keys():
            s = getattr(k, 'session', None)
            if s is not None and str(s) not in seen:
                seen.append(str(s))
        return seen or [str(self.ui._current_session_str())]

    def _available_sessions(self) -> list[str]:
        """Backward-compat: ``All`` + concrete sessions (unused by session picker rows)."""
        concrete = self._concrete_sessions()
        return ['All'] + concrete if 'All' not in concrete else concrete

    def _format_sess_list_summary(self, sess_list: list[str]) -> str:
        if not sess_list:
            return '—'
        all_c = self._concrete_sessions()
        if set(sess_list) == set(all_c):
            return f"All ({len(sess_list)})"
        if len(sess_list) == 1:
            return sess_list[0]
        if len(sess_list) <= 3:
            return ', '.join(sess_list)
        return f"{len(sess_list)} sessions"

    def _pooling_mode(self, sess_list: list[str]) -> tuple[str, str | list[str] | None]:
        """``('all', None)`` | ``('one', session)`` | ``('multi', [sessions...])``."""
        concrete = self._concrete_sessions()
        sl = [s for s in (sess_list or []) if s in concrete]
        if not sl:
            sl = [concrete[0]] if concrete else []
        if set(sl) == set(concrete):
            return 'all', None
        if len(sl) == 1:
            return 'one', sl[0]
        return 'multi', sl

    def _pick_stats_sessions_dialog(
            self,
            *,
            title: str = 'Sessions for stats group',
            initial: list[str] | None = None,
            current_session: str | None = None,
    ) -> list[str] | None:
        """Multi-select session list (same pattern as custom CCG apply-to-sessions)."""
        sessions = self._concrete_sessions()
        if not sessions:
            messagebox.showinfo('Sessions', 'No sessions available.', parent=self.root)
            return None
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry('400x340')
        win.resizable(True, True)
        win.transient(self.root)
        win.grab_set()

        ttk.Label(
            win,
            text='Select sessions  (Shift / Ctrl+click for multi-select):',
        ).pack(anchor='w', padx=8, pady=(8, 2))

        list_frame = ttk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        lb = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, exportselection=False,
            height=14, activestyle='dotbox',
        )
        vsb = ttk.Scrollbar(list_frame, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=vsb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        for i, sess in enumerate(sessions):
            lb.insert(tk.END, sess)
            if current_session and sess == current_session:
                lb.itemconfigure(i, foreground='#1E5FBB')

        init = [s for s in (initial or sessions) if s in sessions]
        if not init:
            init = list(sessions)
        lb.selection_clear(0, tk.END)
        for i, sess in enumerate(sessions):
            if sess in init:
                lb.select_set(i)

        result: list[str] | None = [None]

        def _ok():
            sels = lb.curselection()
            if not sels:
                messagebox.showwarning(
                    'Sessions', 'Select at least one session.', parent=win)
                return
            result[0] = [sessions[i] for i in sels]
            win.destroy()

        def _cancel():
            result[0] = None
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text='Select All',
                   command=lambda: lb.select_set(0, tk.END)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text='Select None',
                   command=lambda: lb.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text='Cancel', command=_cancel).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text='Apply', command=_ok).pack(side=tk.RIGHT, padx=4)

        win.bind('<Return>', lambda e: _ok())
        win.bind('<Escape>', lambda e: _cancel())
        win.wait_window()
        return result[0]

    def _available_conn_types(self, data_type: str = "Conn Strength") -> list[str]:
        if data_type == "Firing Rate":
            return list(_FIRING_RATE_TYPES)
        ui = self.ui
        options: set[str] = set()
        ct = getattr(ui.key, 'conn_type', None)
        if ct:
            options.add(f"{ct[0]}-{ct[1]}")
        conf = getattr(getattr(ui, 'ccg_data', None), 'conf', None)
        if conf:
            for r, t in getattr(conf, 'conn_types_flat', []):
                options.add(f"{r}-{t}")
        return sorted(options) or ['pyr-pyr']

    def _available_segments(self) -> list[str]:
        """Builtin segment names plus unique custom CCG names across all sessions."""
        ui = self.ui
        seen: set[str] = set()
        ordered: list[str] = []

        def add(nm: str):
            if nm and nm not in seen:
                seen.add(nm)
                ordered.append(nm)

        for nk in ui._real_nd_keys_ordered():
            tk = ui._type_key_for_nd(nk)
            if tk is None:
                continue
            ptr = ui.cd.data.get(tk)
            if ptr is None or getattr(ptr, 'edge_times', None) is None:
                continue
            for nm in list(ptr.edge_times['label'].values):
                add(str(nm))
        buckets = getattr(ui, '_custom_segments_by_session', None) or {}
        for lst in buckets.values():
            for cs in lst:
                if isinstance(cs, dict) and cs.get('name'):
                    add(str(cs['name']))
        for cs in getattr(ui, '_custom_segments', []):
            if isinstance(cs, dict) and cs.get('name'):
                add(str(cs['name']))
        add(_ALL_SEGS)
        return ordered

    @contextlib.contextmanager
    def _stats_session_context(self, session_str: str):
        """Bind main UI CCG context to *session_str* for stats reads; restore on exit."""
        ui = self.ui
        saved = (
            ui.key,
            ui.ccg_pointer,
            ui.ccg_data,
            ui.neurons,
            ui.n_segments,
            tuple(ui.segment_names),
        )
        nd_key = ui._nd_key_for_session_str(session_str)
        tk = ui._type_key_for_nd(nd_key) if nd_key is not None else None
        bound = tk is not None
        if bound:
            ui._bind_context_to_type_key(tk)
            ui._bind_custom_segments_to_session(str(session_str))
        try:
            yield bound
        finally:
            if bound:
                k, ptr, cd, neu, ns, sn = saved
                ui.key = k
                ui.ccg_pointer = ptr
                ui.ccg_data = cd
                ui.neurons = neu
                ui.n_segments = ns
                ui.segment_names = list(sn)
                if getattr(ui, '_session_any_mode', False):
                    try:
                        ui._sync_any_plot_context(int(ui.current_pair_idx))
                    except Exception:
                        ui._bind_custom_segments_to_session(str(k.session))
                else:
                    ui._bind_custom_segments_to_session(str(k.session))

    def _sessions_with_segment(self, seg_name: str) -> list[str]:
        """Sessions that have builtin *seg_name* or a loaded custom CCG with that name."""
        ui = self.ui
        if seg_name == _ALL_SEGS:
            return [str(nk.session) for nk in ui._real_nd_keys_ordered()]
        out: list[str] = []
        buckets = getattr(ui, '_custom_segments_by_session', None) or {}
        for nk in ui._real_nd_keys_ordered():
            sess = str(nk.session)
            tk = ui._type_key_for_nd(nk)
            if tk is None:
                continue
            ptr = ui.cd.data.get(tk)
            if ptr is not None and getattr(ptr, 'edge_times', None) is not None:
                snames = [str(x) for x in ptr.edge_times['label'].values]
                if seg_name in snames:
                    out.append(sess)
                    continue
            for cs in buckets.get(sess, []):
                if isinstance(cs, dict) and cs.get('name') == seg_name:
                    out.append(sess)
                    break
        return sorted(set(out))

    def _available_groups(self) -> list[str]:
        non_internal = [g for g in self.ui._sel_data._groups if not g.startswith('__')]
        return ['(all pairs)'] + sorted(non_internal)

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _current_seg_name(self) -> str:
        ui = self.ui
        seg = ui.current_segment
        n = ui.n_segments
        if seg == n:
            return _ALL_SEGS
        if seg > n:
            ci = seg - n - 1
            cs_list = getattr(ui, '_custom_segments', [])
            if 0 <= ci < len(cs_list):
                return cs_list[ci]['name']
            return _ALL_SEGS
        names = ui.segment_names
        return names[seg] if seg < len(names) else _ALL_SEGS

    def _add_row(self):
        ui = self.ui
        concrete = self._concrete_sessions()
        conn_types = self._available_conn_types()
        segments = self._available_segments()
        groups = self._available_groups()

        frame = ttk.Frame(self._rows_frame)
        frame.pack(fill=tk.X, pady=1)

        cur_sess = str(ui._current_session_str())
        ct = getattr(ui.key, 'conn_type', None)
        cur_ct_str = f"{ct[0]}-{ct[1]}" if ct else (conn_types[0] if conn_types else '')
        cur_seg = self._current_seg_name()

        def _first(lst, val):
            return val if val in lst else (lst[0] if lst else '')

        # Name column: A, B, C, ...
        idx = len(self._row_frames)
        name_var = tk.StringVar(value=chr(65 + idx) if idx < 26 else f"G{idx+1}")

        sess_list = [cur_sess] if cur_sess in concrete else ([concrete[0]] if concrete else [])
        sess_var = tk.StringVar(value=self._format_sess_list_summary(sess_list))
        ct_var = tk.StringVar(value=_first(conn_types, cur_ct_str))
        seg_var = tk.StringVar(value=_first(segments, cur_seg))
        grp_var = tk.StringVar(value=groups[1] if len(groups) > 1 else (groups[0] if groups else ''))
        data_var = tk.StringVar(value="Conn Strength")

        ct_combo = ttk.Combobox(frame, textvariable=ct_var, values=conn_types,
                                state='readonly', width=14)
        data_combo = ttk.Combobox(frame, textvariable=data_var, values=_DATA_TYPES,
                                  state='readonly', width=15)

        def _on_data_change(*_):
            # Guard against ttk callback firing during teardown
            if not frame.winfo_exists():
                return
            dt = data_var.get()
            new_cts = self._available_conn_types(dt)
            ct_combo.config(values=new_cts)
            if ct_var.get() not in new_cts:
                ct_var.set(new_cts[0] if new_cts else '')

        # Prefer ComboboxSelected over variable traces to avoid ttk 'popdown'
        # errors when widgets are destroyed while the dropdown is open.
        data_combo.bind('<<ComboboxSelected>>', _on_data_change)

        ttk.Entry(frame, textvariable=name_var, width=5).pack(side=tk.LEFT, padx=2)
        ct_combo.pack(side=tk.LEFT, padx=2)
        seg_combo = ttk.Combobox(frame, textvariable=seg_var, values=segments,
                                 state='readonly', width=14)
        seg_combo.pack(side=tk.LEFT, padx=2)
        ttk.Combobox(frame, textvariable=grp_var, values=groups,
                     state='readonly', width=14).pack(side=tk.LEFT, padx=2)
        data_combo.pack(side=tk.LEFT, padx=2)

        row = dict(frame=frame, name=name_var, sess=sess_var, sess_list=sess_list,
                   ct=ct_var, seg=seg_var, grp=grp_var, data=data_var,
                   seg_combo=seg_combo)

        ttk.Button(frame, text='Sessions…', width=10,
                   command=lambda r=row: self._on_row_pick_sessions(r)).pack(
                       side=tk.LEFT, padx=(2, 2), before=ct_combo)
        ttk.Label(frame, textvariable=sess_var, width=22, anchor='w').pack(
            side=tk.LEFT, padx=(0, 2), before=ct_combo)

        def _del(r=row):
            # Defer destroy to let ttk combobox popdown close cleanly
            def _do():
                try:
                    if r in self._row_frames:
                        self._row_frames.remove(r)
                    if r['frame'].winfo_exists():
                        r['frame'].destroy()
                    self._sync_test_type_to_group_count()
                except Exception:
                    pass
            self._pending_afters.append(self.root.after_idle(_do))

        ttk.Button(frame, text="×", width=2, command=_del).pack(side=tk.LEFT, padx=2)
        self._row_frames.append(row)

        # Initial sync (ensures ConnType list matches Data type)
        _on_data_change()
        self._sync_test_type_to_group_count()

    def _on_row_pick_sessions(self, row: dict):
        """Open multi-select session dialog (same UX as custom CCG apply-to-sessions)."""
        try:
            if not row['frame'].winfo_exists():
                return
        except Exception:
            return
        cur = row['sess_list'][0] if len(row['sess_list']) == 1 else None
        sel = self._pick_stats_sessions_dialog(
            initial=list(row['sess_list']),
            current_session=cur,
        )
        if sel:
            row['sess_list'] = sel
            row['sess'].set(self._format_sess_list_summary(sel))

    def _seg_name_to_idx(self, name: str) -> int | None:
        """Resolve segment index for the **currently bound** UI session."""
        ui = self.ui
        if name == _ALL_SEGS:
            return ui.n_segments
        if name in ui.segment_names:
            return ui.segment_names.index(name)
        for ci, cs in enumerate(getattr(ui, '_custom_segments', [])):
            if isinstance(cs, dict) and cs.get('name') == name:
                return ui.n_segments + 1 + ci
        return None

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _get_pairs_for_group(self, group_name: str, session_str: str, ct_str: str | None = None):
        ui = self.ui
        if session_str == 'All':
            return set()
        if group_name == '(all pairs)':
            ptr = ui.ccg_pointer
            if ptr is None or getattr(ptr, 'inds2', None) is None:
                base_pairs: set = set()
            else:
                base = ptr.inds2
                base = base[base[:, 0] != base[:, 1]]
                base_pairs = set(map(tuple, base))
            admitted = ui._group_pairs(_ADMITTED_GROUP, session=session_str)
            pairs = base_pairs | {tuple(p) for p in admitted if len(p) >= 2 and p[0] != p[1]}
        else:
            pairs = ui._group_pairs(group_name, session=session_str)

        if ct_str and ct_str not in _FIRING_RATE_TYPES:
            # Filter by conn type for consistent counts across panels
            try:
                parts = ct_str.split('-', 1)
                ct_lbl = f"{parts[0]}→{parts[1]}" if len(parts) == 2 else ct_str
                pairs = set(ui._filter_pairs_to_conn_types(session_str, pairs, {ct_lbl}))
            except Exception:
                pass
        return pairs

    def _get_cs_values(self, session_str, ct_str, seg_name, group_name,
                       highres: bool = False):
        """Return (pairs, cs_vals) for Conn Strength data.

        When *session_str* is ``'All'``, pools sessions that have *seg_name* (builtin or
        custom), using each session's own CCG / custom segment; pair keys are
        ``(session, ref, tgt)`` for pairwise alignment across groups.
        """
        if session_str == 'All':
            return self._get_cs_values_all_sessions(ct_str, seg_name, group_name, highres)
        with self._stats_session_context(session_str) as ok:
            if not ok:
                return [], []
            return self._get_cs_values_bound(ct_str, seg_name, group_name, highres)

    def _get_cs_values_all_sessions(self, ct_str, seg_name, group_name, highres: bool):
        ui = self.ui
        method = ui.center_container.baseline_panel._conn_str_method_var.get()
        cs_vals: list[float] = []
        valid_pairs: list[tuple] = []
        for sess in self._sessions_with_segment(seg_name):
            with self._stats_session_context(sess) as ok:
                if not ok:
                    continue
                seg_idx = self._seg_name_to_idx(seg_name)
                if seg_idx is None:
                    continue
                conf = getattr(getattr(ui, 'ccg_data', None), 'conf', None)
                eff_min_lag = getattr(conf, 'min_lag', None) if conf else None
                eff_max_lag = getattr(conf, 'max_lag', None) if conf else None
                pairs = sorted(self._get_pairs_for_group(group_name, sess, ct_str))
                for ref, tgt in pairs:
                    try:
                        ui._compute_pair_conn_strength(int(ref), int(tgt), seg_idx,
                                                       highres=highres)
                        key = ui._cs_cache_key(int(ref), int(tgt), seg_idx, method,
                                               highres, eff_min_lag, eff_max_lag)
                        entry = ui._conn_strength_cache.get(key)
                        if entry is not None and entry[0] is not None:
                            cs_vals.append(float(entry[0]))
                            valid_pairs.append((sess, int(ref), int(tgt)))
                    except Exception as exc:
                        print(f"[StatsPanel] CS error ({sess},{ref},{tgt}): {exc}")
        return valid_pairs, cs_vals

    def _get_cs_values_session_subset(
            self, sessions: list[str], ct_str, seg_name, group_name, highres: bool):
        """Pool Conn Strength like ``All``, but only over *sessions* (order preserved)."""
        have = set(self._sessions_with_segment(seg_name))
        ui = self.ui
        method = ui.center_container.baseline_panel._conn_str_method_var.get()
        cs_vals: list[float] = []
        valid_pairs: list[tuple] = []
        for sess in sessions:
            if sess not in have:
                continue
            with self._stats_session_context(sess) as ok:
                if not ok:
                    continue
                seg_idx = self._seg_name_to_idx(seg_name)
                if seg_idx is None:
                    continue
                conf = getattr(getattr(ui, 'ccg_data', None), 'conf', None)
                eff_min_lag = getattr(conf, 'min_lag', None) if conf else None
                eff_max_lag = getattr(conf, 'max_lag', None) if conf else None
                pairs = sorted(self._get_pairs_for_group(group_name, sess, ct_str))
                for ref, tgt in pairs:
                    try:
                        ui._compute_pair_conn_strength(int(ref), int(tgt), seg_idx,
                                                       highres=highres)
                        key = ui._cs_cache_key(int(ref), int(tgt), seg_idx, method,
                                               highres, eff_min_lag, eff_max_lag)
                        entry = ui._conn_strength_cache.get(key)
                        if entry is not None and entry[0] is not None:
                            cs_vals.append(float(entry[0]))
                            valid_pairs.append((sess, int(ref), int(tgt)))
                    except Exception as exc:
                        print(f"[StatsPanel] CS error ({sess},{ref},{tgt}): {exc}")
        return valid_pairs, cs_vals

    def _get_cs_values_bound(self, ct_str, seg_name, group_name, highres: bool):
        """Conn strength for the session already bound on *ui*."""
        ui = self.ui
        sess = str(ui.key.session)
        method = ui.center_container.baseline_panel._conn_str_method_var.get()
        seg_idx = self._seg_name_to_idx(seg_name)
        if seg_idx is None:
            return [], []

        pairs = sorted(self._get_pairs_for_group(group_name, sess, ct_str))
        if not pairs:
            return [], []

        conf = getattr(getattr(ui, 'ccg_data', None), 'conf', None)
        eff_min_lag = getattr(conf, 'min_lag', None) if conf else None
        eff_max_lag = getattr(conf, 'max_lag', None) if conf else None

        cs_vals: list[float] = []
        valid_pairs: list[tuple] = []
        for ref, tgt in pairs:
            try:
                ui._compute_pair_conn_strength(int(ref), int(tgt), seg_idx, highres=highres)
                key = ui._cs_cache_key(int(ref), int(tgt), seg_idx, method,
                                       highres, eff_min_lag, eff_max_lag)
                entry = ui._conn_strength_cache.get(key)
                if entry is not None and entry[0] is not None:
                    cs_vals.append(float(entry[0]))
                    valid_pairs.append((int(ref), int(tgt)))
            except Exception as exc:
                print(f"[StatsPanel] CS error ({ref},{tgt}): {exc}")

        return valid_pairs, cs_vals

    def _get_firing_rate_values(self, session_str, neuron_type_str, seg_name, group_name):
        """Return (ids, fr_vals) for Firing Rate data.

        neuron_type_str is 'pyr', 'int', or 'all'.
        For 'pyr'/'int', returns rates for neurons matching that type.
        For 'all', returns all neurons.
        """
        if session_str == 'All':
            return self._get_firing_rate_values_all(neuron_type_str, seg_name, group_name)
        with self._stats_session_context(session_str) as ok:
            if not ok:
                return [], []
            return self._get_firing_rate_values_bound(neuron_type_str)

    def _get_firing_rate_values_all(self, neuron_type_str, seg_name, group_name):
        ui = self.ui
        ids: list[tuple] = []
        vals: list[float] = []
        for sess in [str(nk.session) for nk in ui._real_nd_keys_ordered()]:
            with self._stats_session_context(sess) as ok:
                if not ok:
                    continue
                id2, v2 = self._get_firing_rate_values_bound(neuron_type_str)
                for i, v in zip(id2, v2):
                    ids.append((sess, i))
                    vals.append(v)
        return ids, vals

    def _get_firing_rate_session_subset(self, sessions: list[str], neuron_type_str: str):
        ids: list[tuple] = []
        vals: list[float] = []
        for sess in sessions:
            with self._stats_session_context(sess) as ok:
                if not ok:
                    continue
                id2, v2 = self._get_firing_rate_values_bound(neuron_type_str)
                for i, v in zip(id2, v2):
                    ids.append((sess, i))
                    vals.append(v)
        return ids, vals

    def _get_firing_rate_values_bound(self, neuron_type_str: str):
        ui = self.ui
        neurons = getattr(ui, 'neurons', None)
        if neurons is None:
            return [], []

        ntype = neuron_type_str.lower()
        ids: list[int] = []
        vals: list[float] = []
        for i in range(len(neurons.firing_rate)):
            ct = getattr(neurons, 'cell_type', None)
            if ct is not None:
                cell_t = str(ct[i]).lower()
                if ntype == 'pyr' and 'pyr' not in cell_t:
                    continue
                if ntype == 'int' and ('int' not in cell_t and 'in' not in cell_t):
                    continue
            try:
                ids.append(i)
                vals.append(float(neurons.firing_rate[i]))
            except Exception:
                pass
        return ids, vals

    def _get_baseline_values(self, session_str, ct_str, seg_name, group_name):
        """Return (pairs, baseline_vals) — avg CCG outside ±5 ms."""
        if session_str == 'All':
            return self._get_baseline_values_all_sessions(ct_str, seg_name, group_name)
        with self._stats_session_context(session_str) as ok:
            if not ok:
                return [], []
            return self._get_baseline_values_bound(ct_str, seg_name, group_name)

    def _get_baseline_values_all_sessions(self, ct_str, seg_name, group_name):
        from neuropy.analyses.ms_connectivity import apply_norms_to_ccg  # noqa: PLC0415
        ui = self.ui
        bl_vals: list[float] = []
        valid_pairs: list[tuple] = []
        for sess in self._sessions_with_segment(seg_name):
            with self._stats_session_context(sess) as ok:
                if not ok:
                    continue
                p2, v2 = self._get_baseline_values_bound(ct_str, seg_name, group_name)
                for (ref, tgt), v in zip(p2, v2):
                    valid_pairs.append((sess, ref, tgt))
                    bl_vals.append(v)
        return valid_pairs, bl_vals

    def _get_baseline_session_subset(self, sessions: list[str], ct_str, seg_name, group_name):
        have = set(self._sessions_with_segment(seg_name))
        bl_vals: list[float] = []
        valid_pairs: list[tuple] = []
        for sess in sessions:
            if sess not in have:
                continue
            with self._stats_session_context(sess) as ok:
                if not ok:
                    continue
                p2, v2 = self._get_baseline_values_bound(ct_str, seg_name, group_name)
                for (ref, tgt), v in zip(p2, v2):
                    valid_pairs.append((sess, ref, tgt))
                    bl_vals.append(v)
        return valid_pairs, bl_vals

    def _get_baseline_values_bound(self, ct_str, seg_name, group_name):
        from neuropy.analyses.ms_connectivity import apply_norms_to_ccg  # noqa: PLC0415
        ui = self.ui
        seg_idx = self._seg_name_to_idx(seg_name)
        if seg_idx is None:
            return [], []

        sess = str(ui.key.session)
        pairs = sorted(self._get_pairs_for_group(group_name, sess, ct_str))
        if not pairs:
            return [], []

        cd = getattr(ui, 'ccg_data', None)
        if cd is None:
            return [], []
        conf = cd.conf
        bin_size = getattr(conf, 'bin_size', 0.001)
        window   = getattr(conf, 'duration', 0.02)
        n_bins   = int(round(window / bin_size))
        lags_ms  = (np.arange(n_bins) - n_bins // 2) * bin_size * 1000  # in ms
        outside_mask = np.abs(lags_ms) > 5.0

        norms_no_bl = ui.active_norms - {__import__(
            'neuropy.analyses.ms_connectivity', fromlist=['NormalizeBy']).NormalizeBy.BASELINE}

        bl_vals: list[float] = []
        valid_pairs: list[tuple] = []
        for ref, tgt in pairs:
            try:
                # Get raw CCG slice
                is_custom = ui._is_custom_segment(seg_idx)
                is_all    = (seg_idx == ui.n_segments)
                if is_all:
                    ccg_raw = np.sum(cd.ccg[:, ref, tgt, :], axis=0).astype(float)
                    null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0).astype(float)
                                if cd.ccg_null is not None else None)
                elif is_custom:
                    ci = ui._custom_seg_index(seg_idx)
                    cs = ui._custom_segments[ci]
                    ccg_raw  = cs['ccg'][0, ref, tgt, :].astype(float)
                    null_raw = (cs['ccg_null'][0, ref, tgt, :].astype(float)
                                if cs.get('ccg_null') is not None else None)
                else:
                    ccg_raw  = cd.ccg[seg_idx, ref, tgt, :].astype(float)
                    null_raw = (cd.ccg_null[seg_idx, ref, tgt, :].astype(float)
                                if cd.ccg_null is not None else None)

                ccg_norm, _ = apply_norms_to_ccg(
                    ccg_raw, null_raw, ref, tgt, seg_idx,
                    norms_no_bl, ui.neurons, ui.cd.nd, ui.key.nd(),
                    ui.n_segments, is_custom)

                if outside_mask.any():
                    bl_val = float(np.mean(ccg_norm[outside_mask[:len(ccg_norm)]]))
                    bl_vals.append(bl_val)
                    valid_pairs.append((int(ref), int(tgt)))
            except Exception as exc:
                print(f"[StatsPanel] Baseline error ({ref},{tgt}): {exc}")

        return valid_pairs, bl_vals

    def _collect_group_data(self, row: dict, highres: bool = False) -> dict:
        """Collect values for one row dict."""
        name = row.get('name').get() if row.get('name') is not None else ''
        concrete = self._concrete_sessions()
        sl = [s for s in (row.get('sess_list') or []) if s in concrete]
        if not sl and concrete:
            sl = [concrete[0]]
        row['sess_list'] = sl
        mode, arg = self._pooling_mode(sl)
        disp = self._format_sess_list_summary(sl)

        ct = row['ct'].get()
        seg = row['seg'].get()
        grp = row['grp'].get()
        dtype = row['data'].get()
        row['sess'].set(disp)

        if dtype == "Conn Strength":
            if mode == 'all':
                pairs, vals = self._get_cs_values('All', ct, seg, grp, highres=highres)
            elif mode == 'one':
                pairs, vals = self._get_cs_values(arg, ct, seg, grp, highres=highres)
            else:
                pairs, vals = self._get_cs_values_session_subset(
                    arg, ct, seg, grp, highres)
            try:
                if self.ui.center_container.cs_panel._conn_str_nonneg_var.get():
                    vals = [max(float(v), 0.0) for v in (vals or [])]
            except Exception:
                pass
        elif dtype == "Firing Rate":
            if mode == 'all':
                pairs, vals = self._get_firing_rate_values('All', ct, seg, grp)
            elif mode == 'one':
                pairs, vals = self._get_firing_rate_values(arg, ct, seg, grp)
            else:
                pairs, vals = self._get_firing_rate_session_subset(arg, ct)
        elif dtype == "Baseline":
            if mode == 'all':
                pairs, vals = self._get_baseline_values('All', ct, seg, grp)
            elif mode == 'one':
                pairs, vals = self._get_baseline_values(arg, ct, seg, grp)
            else:
                pairs, vals = self._get_baseline_session_subset(arg, ct, seg, grp)
        else:
            pairs, vals = [], []

        return dict(name=name, session=disp, conn_type=ct, segment=seg, group=grp,
                    data_type=dtype, pairs=pairs, vals=vals, highres=highres)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _maybe_log_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply log transform for skewed data (stable for zeros/negatives)."""
        if not getattr(self, '_log_var', None) or not self._log_var.get():
            return x
        x = np.asarray(x, dtype=float)
        # Signed log1p so CS/baseline values that can be negative still work.
        return np.sign(x) * np.log1p(np.abs(x))

    def _run_anova(self, groups: list[dict]) -> dict:
        from scipy import stats  # noqa: PLC0415
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd  # noqa: PLC0415
            _has_statsmodels = True
        except ImportError:
            _has_statsmodels = False

        all_vals: list = []
        all_labels: list = []
        for g in groups:
            vals = self._maybe_log_transform(
                np.asarray(g.get('vals', []) or [], dtype=float))
            if len(vals) < 2:
                return {'error': f"Group '{g.get('name', '?')}': need ≥2 values."}
            all_vals.append(vals)
            all_labels.extend([g.get('name', '?')] * len(vals))

        try:
            f_stat, p_val = stats.f_oneway(*all_vals)
            result: dict = dict(test='One-way ANOVA', f_stat=float(f_stat),
                                p_val=float(p_val),
                                n_used=sum(len(v) for v in all_vals))
            if _has_statsmodels:
                combined = np.concatenate(all_vals)
                tukey = pairwise_tukeyhsd(combined, all_labels)
                tukey_rows = []
                for row in tukey.summary().data[1:]:
                    a_nm, b_nm, meandiff, p_adj, lo, hi, reject = row
                    tukey_rows.append(dict(
                        a=str(a_nm), b=str(b_nm),
                        meandiff=float(meandiff),
                        p_adj=float(p_adj),
                        reject=bool(reject)))
                result['tukey'] = tukey_rows
            else:
                result['tukey_missing'] = True
        except Exception as exc:
            result = {'error': f"Error: {exc}"}
        return result

    def _run_test(self, a_vals, b_vals, a_pairs, b_pairs,
                  test_type, alternative: str, nonparametric: bool) -> dict:
        from scipy import stats  # noqa: PLC0415
        result: dict = {}

        a_vals = self._maybe_log_transform(np.asarray(a_vals, dtype=float))
        b_vals = self._maybe_log_transform(np.asarray(b_vals, dtype=float))

        if len(a_vals) < 2 or len(b_vals) < 2:
            result['error'] = "Insufficient data (need ≥2 values per group)."
            return result

        try:
            if test_type == "Pairwise t-test":
                set_b   = {tuple(p): i for i, p in enumerate(b_pairs)}
                common  = [p for p in a_pairs if tuple(p) in set_b]
                if len(common) < 2:
                    result['error'] = f"Pairwise test: only {len(common)} pair(s) in common."
                    return result
                idx_a = {tuple(p): i for i, p in enumerate(a_pairs)}
                vals_a = np.array([a_vals[idx_a[tuple(p)]] for p in common], dtype=float)
                vals_b = np.array([b_vals[set_b[tuple(p)]]  for p in common], dtype=float)

                if nonparametric:
                    # Paired nonparametric alternative: Wilcoxon signed-rank test
                    stat, p_val = stats.wilcoxon(vals_a, vals_b,
                                                 alternative=alternative, zero_method='wilcox')
                    result.update(test='Wilcoxon signed-rank',
                                  stat=stat, p_val=p_val,
                                  n_used=len(common), n_common=len(common))
                else:
                    t_stat, p_val = stats.ttest_rel(vals_a, vals_b, alternative=alternative)
                    result.update(test='Paired t-test',
                                  t_stat=t_stat, p_val=p_val,
                                  n_used=len(common), n_common=len(common))
            else:
                if nonparametric:
                    # Mann–Whitney U for independent samples
                    stat, p_val = stats.mannwhitneyu(a_vals, b_vals, alternative=alternative)
                    result.update(test='Mann–Whitney U',
                                  stat=stat, p_val=p_val,
                                  n_used=len(a_vals) + len(b_vals))
                else:
                    t_stat, p_val = stats.ttest_ind(a_vals, b_vals, alternative=alternative)
                    result.update(test='Independent t-test',
                                  t_stat=t_stat, p_val=p_val,
                                  n_used=len(a_vals) + len(b_vals))
        except Exception as exc:
            result['error'] = f"Error: {exc}"
        return result

    def _run(self):
        active = [r for r in self._row_frames if r['frame'].winfo_exists()]
        if len(active) < 2:
            self._show_result("Need at least 2 groups to compare.")
            return

        dtype = active[0]['data'].get()
        if dtype in _DATA_DISABLED:
            self._show_result(f"Data type '{dtype}' is not yet implemented.")
            return

        test_type = self._test_type_var.get()
        run_hilo = (dtype == "Conn Strength")

        # Alternative hypothesis (relative to A)
        if self._sides_var.get() == "Two-sided":
            alternative = "two-sided"
        else:
            alternative = "greater" if self._dir_var.get().strip() == "A > B" else "less"
        nonparametric = bool(getattr(self, '_nonparam_var', None) and self._nonparam_var.get())
        log_transform = bool(getattr(self, '_log_var', None) and self._log_var.get())

        # Collect lo-res data (or single pass for non-CS)
        groups_lo = [self._collect_group_data(r, highres=False) for r in active]
        groups_hi = result_hi = None

        if len(active) > 2:
            res_lo = self._run_anova(groups_lo)
            if run_hilo:
                groups_hi = [self._collect_group_data(r, highres=True) for r in active]
                result_hi = self._run_anova(groups_hi)
        else:
            a_lo, b_lo = groups_lo[0], groups_lo[1]
            res_lo = self._run_test(a_lo['vals'], b_lo['vals'],
                                    a_lo['pairs'], b_lo['pairs'],
                                    test_type, alternative, nonparametric)
            if run_hilo:
                groups_hi = [self._collect_group_data(r, highres=True) for r in active]
                a_hi, b_hi = groups_hi[0], groups_hi[1]
                result_hi = self._run_test(a_hi['vals'], b_hi['vals'],
                                           a_hi['pairs'], b_hi['pairs'],
                                           test_type, alternative, nonparametric)

        self._result_data = dict(
            groups=groups_lo, test_type=test_type, dtype=dtype,
            res_lo=res_lo, groups_hi=groups_hi, res_hi=result_hi,
            alternative=alternative, nonparametric=nonparametric,
            log_transform=log_transform)

        lines = self._build_result_lines(groups_lo, res_lo, test_type,
                                          groups_hi=groups_hi, res_hi=result_hi)
        self._show_result('\n'.join(lines))
        # Plot uses the same transformation as the statistical test
        def _tx(groups):
            out = []
            for g in groups:
                gg = dict(g)
                gg['vals'] = list(self._maybe_log_transform(np.asarray(g.get('vals', []) or [], dtype=float)))
                out.append(gg)
            return out
        self._update_result_plot(_tx(groups_lo), _tx(groups_hi) if groups_hi is not None else None)
        if self._export_btn:
            self._export_btn.config(state=tk.NORMAL)

    def _update_result_plot(self, groups_lo, groups_hi=None):
        """Draw bar-with-whiskers + scatter for all groups; annotate +3 SD outliers."""
        if not hasattr(self, '_plot_fig') or self._plot_fig is None:
            return
        fig = self._plot_fig
        fig.clf()

        _BAR_COLORS = ['#8FB3FF', '#FFB3B3', '#B3FFB3', '#FFD9B3', '#E0B3FF',
                       '#B3F0FF', '#FFB3E6']

        def _mean_sem(x):
            if x.size == 0:
                return np.nan, 0.0
            m = float(np.mean(x))
            return m, float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0

        def _fmt_pair_short(p) -> str:
            if isinstance(p, (list, tuple)):
                return f"{p[1]}-{p[2]}" if len(p) == 3 else f"{p[0]}-{p[1]}"
            return str(p)

        def _one_axis(ax, groups, title: str, paired: bool = False):
            if not groups or len(groups) < 1:
                ax.axis('off')
                return
            n_g = len(groups)
            names = [g.get('name', chr(65 + i)) or chr(65 + i) for i, g in enumerate(groups)]
            arrays = [np.array(g.get('vals', []) or [], dtype=float) for g in groups]
            pairs_lists = [g.get('pairs', []) for g in groups]

            xs = np.arange(n_g, dtype=float)
            ax.set_title(title, fontsize=9, pad=2)
            ax.set_xticks(xs, names)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=8)

            means = np.array([_mean_sem(arr)[0] for arr in arrays])
            errs  = np.array([_mean_sem(arr)[1] for arr in arrays])
            ax.bar(xs, means, yerr=errs, capsize=4,
                   color=_BAR_COLORS[:n_g], edgecolor='#333', linewidth=0.8)

            # Pre-compute jitter so lines and dots share the same x positions
            rng = np.random.default_rng(0)
            all_xpos = []
            for i, arr in enumerate(arrays):
                jitter = rng.normal(0, 0.05, size=arr.size) if arr.size > 0 else np.array([])
                all_xpos.append(np.full(arr.size, xs[i]) + jitter)

            # Paired connecting lines (drawn first so dots appear on top)
            if paired and n_g == 2:
                def _pk(p):
                    return tuple(p) if isinstance(p, (list, tuple)) else p
                map_B = {_pk(p): (all_xpos[1][j], float(arrays[1][j]))
                         for j, p in enumerate(pairs_lists[1])}
                for j, p in enumerate(pairs_lists[0]):
                    if _pk(p) in map_B:
                        xB, yB = map_B[_pk(p)]
                        ax.plot([all_xpos[0][j], xB],
                                [float(arrays[0][j]), yB],
                                color='#888888', alpha=0.1, linewidth=0.7, zorder=1)

            for i, (arr, pairs) in enumerate(zip(arrays, pairs_lists)):
                if arr.size == 0:
                    continue
                mean_v = float(np.mean(arr))
                std_v  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                xpos   = all_xpos[i]
                ax.scatter(xpos, arr, s=14, color='#222', alpha=0.6, linewidths=0, zorder=2)
                if std_v > 0:
                    for j, (v, xj) in enumerate(zip(arr, xpos)):
                        if abs(v - mean_v) > 3 * std_v:
                            lbl = _fmt_pair_short(pairs[j] if j < len(pairs) else j)
                            ax.annotate(lbl, (xj, float(v)), fontsize=6,
                                        ha='center', va='bottom', color='#CC0000',
                                        xytext=(0, 3), textcoords='offset points')
            ax.grid(axis='y', alpha=0.25, linewidth=0.7)

        dtype = (groups_lo[0].get('data_type') if groups_lo else '') or ''
        is_paired = (self._result_data or {}).get('test_type') == "Pairwise t-test"
        if groups_hi is not None:
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
            _one_axis(ax1, groups_lo, f"{dtype} (Lo-res)", paired=is_paired)
            _one_axis(ax2, groups_hi, f"{dtype} (Hi-res)", paired=is_paired)
            fig.tight_layout(pad=0.8, w_pad=1.2)
        else:
            ax = fig.add_subplot(1, 1, 1)
            _one_axis(ax, groups_lo, f"{dtype}", paired=is_paired)
            fig.tight_layout(pad=0.8)

        try:
            self._plot_canvas.draw()
        except Exception:
            pass

    def _build_result_lines(self, groups, res, test_type,
                             groups_hi=None, res_hi=None) -> list[str]:
        is_anova = (len(groups) > 2 or test_type == "One-way ANOVA + Tukey")
        a = groups[0].get('name', 'A') if groups else 'A'
        b = groups[1].get('name', 'B') if len(groups) > 1 else 'B'
        alt = (self._result_data or {}).get('alternative', 'two-sided')
        if is_anova:
            h0 = "H0: all group means equal"
            h1 = "H1: at least one group mean differs"
        else:
            h0 = f"H0: μ{a} = μ{b}"
            h1 = (f"H1: μ{a} ≠ μ{b}" if alt == 'two-sided' else
                  f"H1: μ{a} > μ{b}" if alt == 'greater' else f"H1: μ{a} < μ{b}")

        lines = [
            "Stats Test Results",
            "==================",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test:  {test_type}",
            f"Data:  {groups[0]['data_type']}",
            f"{h0}",
            f"{h1}",
            f"Mode:  {'Nonparametric' if (self._result_data or {}).get('nonparametric') else 'Parametric'}",
            f"Xform: {'Signed log1p' if (self._result_data or {}).get('log_transform') else 'None'}",
            "",
        ]
        for i, g in enumerate(groups):
            lines.append(
                f"Group {g.get('name', chr(65+i))}: {g['session']} | {g['conn_type']} "
                f"| {g['segment']} | {g['group']}  (n={len(g['vals'])})")
        lines.append("")

        def _stat_str(r):
            if 'f_stat' in r:
                return f"F={r['f_stat']:.4f}  p={r['p_val']:.4f}  n={r.get('n_used', '?')}"
            if 't_stat' in r:
                return f"t={r['t_stat']:.4f}  p={r['p_val']:.4f}  n={r.get('n_used', '?')}"
            if 'stat' in r:
                return f"stat={r['stat']:.4f}  p={r['p_val']:.4f}  n={r.get('n_used', '?')}"
            return "stat=?"

        def _tukey_lines(r, label=""):
            tl = []
            if label:
                tl.append(f"Tukey HSD ({label}):")
            else:
                tl.append("Tukey HSD:")
            if r.get('tukey_missing'):
                tl.append("  (statsmodels not available)")
                return tl
            for tk in r.get('tukey', []):
                sig = " *" if tk['reject'] else ""
                tl.append(f"  {tk['a']} vs {tk['b']}: "
                           f"Δ={tk['meandiff']:+.4f}  p_adj={tk['p_adj']:.4f}{sig}")
            return tl

        def _res_section(r, label=""):
            if 'error' in r:
                return [r['error']]
            rl = [_stat_str(r)]
            if 'tukey' in r or r.get('tukey_missing'):
                rl += _tukey_lines(r, label)
            return rl

        if groups_hi is not None:
            lines.append("Lo-res:")
            lines += ['  ' + ln for ln in _res_section(res, "Lo-res")]
            lines.append("Hi-res:")
            lines += ['  ' + ln for ln in _res_section(res_hi or {}, "Hi-res")]
        else:
            if 'error' in res:
                lines.append(res['error'])
            else:
                if 'f_stat' in res:
                    lines.append(f"F = {res['f_stat']:.4f}")
                elif 't_stat' in res:
                    lines.append(f"t = {res['t_stat']:.4f}")
                elif 'stat' in res:
                    lines.append(f"stat = {res['stat']:.4f}")
                else:
                    lines.append("stat = ?")
                lines.append(f"p = {res['p_val']:.4f}")
                if 'tukey' in res or res.get('tukey_missing'):
                    lines.append("")
                    lines += _tukey_lines(res)

        # Outlier listing (>3 SD from group mean)
        lines.append("")
        lines.append("Outliers (>3 SD from group mean):")
        any_outlier = False
        for i, g in enumerate(groups):
            vals_tx = self._maybe_log_transform(
                np.asarray(g.get('vals', []) or [], dtype=float))
            if vals_tx.size < 4:
                continue
            m = float(np.mean(vals_tx))
            s = float(np.std(vals_tx, ddof=1))
            if s == 0:
                continue
            pairs = g.get('pairs', [])
            out_items = [_fmt_pair(pairs[j] if j < len(pairs) else j)
                         for j, v in enumerate(vals_tx) if abs(v - m) > 3 * s]
            if out_items:
                any_outlier = True
                lines.append(f"  {g.get('name', chr(65+i))}: {', '.join(out_items)}")
        if not any_outlier:
            lines.append("  (none)")

        return lines

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _build_methods_string(self) -> str:
        ui = self.ui
        d  = self._result_data
        if d is None:
            return ""

        test_type = d['test_type']
        groups    = d['groups']
        dtype     = d.get('dtype', 'Conn Strength')

        ct_raw  = groups[0]['conn_type']
        parts   = ct_raw.split('-')
        ct_disp = (f"{parts[0].upper()}-{parts[1].upper()} connections"
                   if len(parts) == 2 else ct_raw)

        sessions  = list(dict.fromkeys(g['session'] for g in groups))
        sess_str  = ', '.join(f"Session: {s}" for s in sessions)

        g_names = list(dict.fromkeys(
            g['group'] for g in groups if g['group'] != '(all pairs)'))
        if not g_names:
            labels_str = "all pairs"
        elif len(g_names) == 1:
            labels_str = f"labeled as {g_names[0]}"
        elif len(g_names) == 2:
            labels_str = f"labeled as {g_names[0]} and {g_names[1]}"
        else:
            labels_str = "labeled as " + ", ".join(g_names[:-1]) + f", and {g_names[-1]}"

        segments = list(dict.fromkeys(g['segment'] for g in groups))
        if len(segments) == 1:
            seg_str = f"segment {segments[0]}"
        elif len(segments) == 2:
            seg_str = f"differences between segments {segments[0]} and {segments[1]}"
        else:
            seg_str = "segments: " + ", ".join(segments)

        if dtype == "Conn Strength":
            method  = ui.center_container.baseline_panel._conn_str_method_var.get()
            anorms  = getattr(ui, 'active_norms', set())
            method_map = {
                'conv':   'convolution (EranConv)',
                'tailed': 'ACG-deconvolution tail mean',
                'global': 'global maximum outside test window',
            }
            method_desc = method_map.get(method, method)
            norm_parts = []
            for n in anorms:
                nm = getattr(n, 'name', str(n))
                d_map = {'REF_FRATE': 'reference firing rate',
                         'TIME_SECOND': 'total time in seconds',
                         'TOTAL_AREA': 'total CCG area',
                         'BASELINE': 'baseline-subtracted'}
                norm_parts.append(d_map.get(nm, nm))
            if norm_parts:
                cs_def = (f"CCG strength is defined as CCG minus the {method_desc} baseline, "
                          f"divided by {' and '.join(norm_parts)}.")
            else:
                cs_def = f"CCG strength is defined as CCG minus the {method_desc} baseline."
            data_desc = cs_def
        elif dtype == "Firing Rate":
            data_desc = "Firing rate is the mean spike rate in spikes/s over the full session."
        elif dtype == "Baseline":
            data_desc = ("Baseline is the mean CCG bin value for bins with |lag| > 5 ms, "
                         "reflecting the spontaneous co-firing rate.")
        else:
            data_desc = ""

        if test_type == "One-way ANOVA + Tukey":
            return (f"One-way ANOVA with Tukey HSD post-hoc on {ct_disp} in {sess_str}, "
                    f"{labels_str}, on the {seg_str}. {data_desc}")
        return (f"{test_type} on {ct_disp} in {sess_str}, "
                f"{labels_str}, on the {seg_str}. {data_desc}")

    # ------------------------------------------------------------------
    # JSON serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe(obj):
        """Recursively convert numpy scalars / tuples to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: StatsTestPanel._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [StatsTestPanel._json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _build_snapshot(self) -> dict:
        """Build a JSON-serialisable snapshot of current UI state + result data."""
        import json as _json  # noqa: PLC0415
        d = self._result_data or {}
        ui = self.ui

        # Row configs
        rows_snap = []
        for r in self._row_frames:
            try:
                if not r['frame'].winfo_exists():
                    continue
            except Exception:
                continue
            rows_snap.append({
                'name':      r['name'].get(),
                'sess_list': list(r.get('sess_list') or []),
                'conn_type': r['ct'].get(),
                'segment':   r['seg'].get(),
                'group':     r['grp'].get(),
                'data_type': r['data'].get(),
            })

        # Collect custom segment names referenced in rows
        builtin = set()
        for nk in ui._real_nd_keys_ordered():
            tk_ = ui._type_key_for_nd(nk)
            ptr = ui.cd.data.get(tk_) if tk_ is not None else None
            if ptr is not None and getattr(ptr, 'edge_times', None) is not None:
                for nm in ptr.edge_times['label'].values:
                    builtin.add(str(nm))
        builtin.add(_ALL_SEGS)
        custom_segs_used = sorted({
            r['segment'] for r in rows_snap
            if r['segment'] and r['segment'] not in builtin
        })

        # CCGConfig snapshot
        conf = getattr(getattr(ui, 'ccg_data', None), 'conf', None)
        ccg_conf = {}
        if conf:
            for attr in ['name', 'bin_size', 'duration', 'min_lag', 'max_lag',
                         'alpha', 'n_jitter', 'n_jitter_shuffles']:
                v = getattr(conf, attr, None)
                if v is not None:
                    ccg_conf[attr] = v

        text_out = ''
        if self._result_text:
            try:
                text_out = self._result_text.get('1.0', tk.END).strip()
            except Exception:
                pass

        return {
            'version': 1,
            'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ui_state': {
                'test_type':     self._test_type_var.get(),
                'sides':         self._sides_var.get(),
                'direction':     self._dir_var.get(),
                'nonparametric': bool(self._nonparam_var.get()),
                'log_transform': bool(self._log_var.get()),
            },
            'rows': rows_snap,
            'custom_segments': custom_segs_used,
            'result': self._json_safe(d) if d else None,
            'methods': self._build_methods_string(),
            'ccg_conf': ccg_conf,
            'text_output': text_out,
        }

    def _export(self):
        if not self._result_data:
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension='.json',
            filetypes=[('Stats JSON', '*.json'), ('All files', '*.*')],
            title="Export Stats Test Results",
        )
        if not path:
            return

        import json as _json  # noqa: PLC0415
        snap = self._build_snapshot()

        # Write JSON
        try:
            with open(path, 'w', encoding='utf-8') as f:
                _json.dump(snap, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc), parent=self.root)
            return

        # Write .txt sidecar
        txt_path = path if path.endswith('.txt') else (
            path[:-5] + '.txt' if path.endswith('.json') else path + '.txt')
        try:
            txt = snap.get('text_output') or ''
            methods = snap.get('methods') or ''
            conf_lines = [f"  {k}: {v}" for k, v in (snap.get('ccg_conf') or {}).items()]
            full_txt = '\n'.join([
                '=' * 60, 'Stats Test Results', '=' * 60,
                txt, '',
                'Methods:', '-' * 40, methods, '',
                'CCGConfig:', '-' * 40,
            ] + (conf_lines or ['  (not available)']))
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_txt + '\n')
        except Exception as exc:
            print(f"[StatsPanel] txt sidecar write error: {exc}")

        print(f"[StatsPanel] Exported to {path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _ensure_custom_segments_loaded(self, names: list[str]) -> list[str]:
        """Try to load each custom segment *name* from the CCG disk cache if not in memory.

        Returns list of names that could NOT be found/loaded.
        """
        if not names:
            return []
        ui = self.ui

        # Build set of already-loaded names
        loaded: set[str] = set()
        buckets = getattr(ui, '_custom_segments_by_session', None) or {}
        for lst in buckets.values():
            for cs in lst:
                if isinstance(cs, dict) and cs.get('name'):
                    loaded.add(cs['name'])
        for cs in getattr(ui, '_custom_segments', []):
            if isinstance(cs, dict) and cs.get('name'):
                loaded.add(cs['name'])

        missing = [n for n in names if n not in loaded]
        if not missing:
            return []

        # Scan disk cache
        import glob as _glob  # noqa: PLC0415
        import json as _json  # noqa: PLC0415
        cache_dir = getattr(ui, '_ccg_cache_dir', None)
        if not cache_dir:
            return missing

        pattern = os.path.join(cache_dir, "*.npz")
        still_missing: list[str] = []
        for want in missing:
            found = False
            for p in sorted(_glob.glob(pattern)):
                try:
                    npz = np.load(p, allow_pickle=False)
                    nm = str(npz['name_'])
                    if nm != want:
                        continue
                    base = os.path.basename(p)
                    file_sess = base.split("__", 1)[0] if "__" in base else str(ui.key.session)
                    cs = dict(
                        name=nm,
                        t0=float(npz['t0_']),
                        t1=float(npz['t1_']),
                        ccg=npz['ccg'],
                        ccg_null=npz['ccg_null'],
                        pval=npz['pval'],
                        pval_corrected=npz['pval_corrected'],
                        compute_sec=(float(npz['compute_sec_'])
                                     if 'compute_sec_' in npz else float('nan')),
                        active_duration=(float(npz['active_duration_'])
                                         if 'active_duration_' in npz else float('nan')),
                        total_time_hours=(float(npz['total_time_hours_'])
                                          if 'total_time_hours_' in npz else None),
                        filter_state=(_json.loads(str(npz['filter_state_']))
                                      if 'filter_state_' in npz else {}),
                        metadata=(_json.loads(str(npz['metadata_']))
                                  if 'metadata_' in npz else {}),
                        src_path=p,
                    )
                    for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi',
                              'firing_rates'):
                        if k in npz:
                            cs[k] = npz[k]
                    lst = ui._custom_segments_by_session.setdefault(file_sess, [])
                    ui._upsert_custom_segment_by_name(lst, cs)
                    if lst is ui._custom_segments:
                        try:
                            ui._build_sig_chips()
                            ui._update_segment_label()
                        except Exception:
                            pass
                    found = True
                    break
                except Exception as exc:
                    print(f"[StatsPanel] load custom seg from {p}: {exc}")
                    continue
            if not found:
                still_missing.append(want)

        return still_missing

    def _restore_rows_from_snap(self, rows_snap: list[dict]):
        """Clear current rows and rebuild from snapshot."""
        # Destroy existing rows
        for r in list(self._row_frames):
            try:
                r['frame'].destroy()
            except Exception:
                pass
        self._row_frames.clear()

        # Add rows
        for snap_row in rows_snap:
            self._add_row()
            r = self._row_frames[-1]
            try:
                r['name'].set(snap_row.get('name', ''))
                sl = snap_row.get('sess_list') or []
                concrete = self._concrete_sessions()
                sl = [s for s in sl if s in concrete]
                if not sl and concrete:
                    sl = [concrete[0]]
                r['sess_list'] = sl
                r['sess'].set(self._format_sess_list_summary(sl))
                ct = snap_row.get('conn_type', '')
                ct_combo = None
                for child in r['frame'].winfo_children():
                    if isinstance(child, ttk.Combobox) and child.cget('width') == 14:
                        ct_combo = child
                        break
                if ct in (r.get('ct_combo_values') or [ct]):
                    r['ct'].set(ct)
                else:
                    r['ct'].set(ct)
                segs = self._available_segments()
                seg = snap_row.get('segment', '')
                r['seg'].set(seg if seg in segs else (segs[0] if segs else ''))
                grps = self._available_groups()
                grp = snap_row.get('group', '')
                r['grp'].set(grp if grp in grps else (grps[0] if grps else ''))
                r['data'].set(snap_row.get('data_type', 'Conn Strength'))
            except Exception as exc:
                print(f"[StatsPanel] restore row: {exc}")

    def _load_result(self):
        """Load a previously exported stats result JSON and restore UI + display."""
        import json as _json  # noqa: PLC0415
        path = filedialog.askopenfilename(
            parent=self.root,
            defaultextension='.json',
            filetypes=[('Stats JSON', '*.json'), ('All files', '*.*')],
            title="Load Stats Test Results",
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                snap = _json.load(f)
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not read file:\n{exc}", parent=self.root)
            return

        if snap.get('version', 0) != 1:
            messagebox.showerror("Load error",
                                 "Unsupported file version.", parent=self.root)
            return

        warnings: list[str] = []

        # Restore UI settings
        ui_st = snap.get('ui_state') or {}
        try:
            tt = ui_st.get('test_type', 'Pairwise t-test')
            if tt not in ['Independent t-test', 'Pairwise t-test', 'One-way ANOVA + Tukey']:
                tt = 'Pairwise t-test'
            self._test_type_var.set(tt)
            sides = ui_st.get('sides', 'Two-sided')
            self._sides_var.set(sides if sides in ['Two-sided', 'One-sided'] else 'Two-sided')
            direction = ui_st.get('direction', 'A > B')
            self._dir_var.set(direction if direction in ['A > B', 'A < B'] else 'A > B')
            self._nonparam_var.set(bool(ui_st.get('nonparametric', False)))
            self._log_var.set(bool(ui_st.get('log_transform', False)))
        except Exception as exc:
            warnings.append(f"UI settings partially restored: {exc}")

        # Load custom segments from disk if needed
        custom_needed = snap.get('custom_segments') or []
        if custom_needed:
            still_missing = self._ensure_custom_segments_loaded(custom_needed)
            if still_missing:
                warnings.append(
                    "Custom segment(s) not found in cache — CCG data may be unavailable:\n  "
                    + ', '.join(still_missing))

        # Restore rows
        rows_snap = snap.get('rows') or []
        if rows_snap:
            try:
                self._restore_rows_from_snap(rows_snap)
            except Exception as exc:
                warnings.append(f"Row config partially restored: {exc}")

        # Display stored results
        result = snap.get('result')
        if result:
            self._result_data = result
            text_out = snap.get('text_output') or ''
            note = '[Loaded from file — click Run to refresh with live data]\n\n'
            self._show_result(note + text_out)
            try:
                d = result
                groups_lo = d.get('groups') or []
                groups_hi = d.get('groups_hi')
                def _tx(groups):
                    out = []
                    for g in groups:
                        gg = dict(g)
                        gg['vals'] = list(self._maybe_log_transform(
                            np.asarray(g.get('vals', []) or [], dtype=float)))
                        out.append(gg)
                    return out
                self._update_result_plot(
                    _tx(groups_lo),
                    _tx(groups_hi) if groups_hi else None)
            except Exception as exc:
                print(f"[StatsPanel] load plot restore: {exc}")
            if self._export_btn:
                self._export_btn.config(state=tk.NORMAL)
        else:
            self._show_result('[Loaded from file — no result data. Click Run.]')

        if warnings:
            messagebox.showwarning(
                "Load Stats Results",
                "Loaded with warnings:\n\n" + '\n'.join(warnings),
                parent=self.root)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_result(self, text: str):
        if self._result_text:
            self._result_text.config(state=tk.NORMAL)
            self._result_text.delete('1.0', tk.END)
            self._result_text.insert('1.0', text)
            self._result_text.config(state=tk.DISABLED)
