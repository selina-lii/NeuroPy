"""Stats test panel for CCGReviewUI.

Provides a Toplevel dialog for running statistical comparisons
(t-tests) on connection strengths, firing rates, CCG baselines, etc.
"""

import datetime
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_ALL_SEGS = "All segments"

_DATA_TYPES = [
    "Conn Strength",
    "Firing Rate",
    "Baseline",
    "Peak Width",    # grayed — not yet implemented
    "Peak Center",   # grayed — not yet implemented
]
_DATA_DISABLED = {"Peak Width", "Peak Center"}

_FIRING_RATE_TYPES = ["pyr", "int", "all"]


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
        self.root.destroy()

    def on_parent_display_option_changed(self):
        """Called by the main UI when display-affecting toggles change (e.g. non-negative)."""
        try:
            if not self.root.winfo_exists():
                return
            if self._result_data and self._result_data.get('dtype') == "Conn Strength":
                # Re-run so the text + plot reflect the new clamp mode.
                self.root.after_idle(self._run)
        except Exception:
            pass

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
        ttk.Combobox(top, textvariable=self._test_type_var,
                     values=["Independent t-test", "Pairwise t-test"],
                     state='readonly', width=20).pack(side=tk.LEFT)

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
        self._result_text = tk.Text(
            text_frame, height=10, wrap=tk.WORD,
            font=('Courier', 9), state=tk.DISABLED,
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

        self._add_row()
        self._add_row()

    def _toggle_direction(self):
        self._dir_var.set("A < B" if self._dir_var.get().strip() == "A > B" else "A > B")

    # ------------------------------------------------------------------
    # Dropdown data sources
    # ------------------------------------------------------------------

    def _available_sessions(self) -> list[str]:
        seen: list[str] = []
        for k in getattr(self.ui.cd, 'data', {}).keys():
            s = getattr(k, 'session', None)
            if s and s not in seen:
                seen.append(s)
        return seen or [self.ui._current_session_str()]

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
        ui = self.ui
        segs = list(ui.segment_names)
        for cs in getattr(ui, '_custom_segments', []):
            segs.append(cs['name'])
        segs.append(_ALL_SEGS)
        return segs

    def _available_groups(self) -> list[str]:
        non_internal = [g for g in self.ui._groups if not g.startswith('__')]
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
        sessions   = self._available_sessions()
        conn_types = self._available_conn_types()
        segments   = self._available_segments()
        groups     = self._available_groups()

        frame = ttk.Frame(self._rows_frame)
        frame.pack(fill=tk.X, pady=1)

        cur_sess = ui._current_session_str()
        ct = getattr(ui.key, 'conn_type', None)
        cur_ct_str = f"{ct[0]}-{ct[1]}" if ct else (conn_types[0] if conn_types else '')
        cur_seg = self._current_seg_name()

        def _first(lst, val):
            return val if val in lst else (lst[0] if lst else '')

        # Name column: A, B, C, ...
        idx = len(self._row_frames)
        name_var = tk.StringVar(value=chr(65 + idx) if idx < 26 else f"G{idx+1}")

        sess_var = tk.StringVar(value=_first(sessions,   cur_sess))
        ct_var   = tk.StringVar(value=_first(conn_types, cur_ct_str))
        seg_var  = tk.StringVar(value=_first(segments,   cur_seg))
        grp_var  = tk.StringVar(value=groups[1] if len(groups) > 1 else (groups[0] if groups else ''))
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
        ttk.Combobox(frame, textvariable=sess_var, values=sessions,
                     state='readonly', width=22).pack(side=tk.LEFT, padx=2)
        ct_combo.pack(side=tk.LEFT, padx=2)
        ttk.Combobox(frame, textvariable=seg_var,  values=segments,
                     state='readonly', width=14).pack(side=tk.LEFT, padx=2)
        ttk.Combobox(frame, textvariable=grp_var,  values=groups,
                     state='readonly', width=14).pack(side=tk.LEFT, padx=2)
        data_combo.pack(side=tk.LEFT, padx=2)

        row = dict(frame=frame, name=name_var, sess=sess_var, ct=ct_var,
                   seg=seg_var, grp=grp_var, data=data_var)

        def _del(r=row):
            # Defer destroy to let ttk combobox popdown close cleanly
            def _do():
                try:
                    if r in self._row_frames:
                        self._row_frames.remove(r)
                    if r['frame'].winfo_exists():
                        r['frame'].destroy()
                except Exception:
                    pass
            self.root.after_idle(_do)

        ttk.Button(frame, text="×", width=2, command=_del).pack(side=tk.LEFT, padx=2)
        self._row_frames.append(row)

        # Initial sync (ensures ConnType list matches Data type)
        _on_data_change()

    # ------------------------------------------------------------------
    # Segment index helpers
    # ------------------------------------------------------------------

    def _seg_name_to_idx(self, name: str) -> int | None:
        ui = self.ui
        if name == _ALL_SEGS:
            return ui.n_segments
        if name in ui.segment_names:
            return ui.segment_names.index(name)
        for ci, cs in enumerate(getattr(ui, '_custom_segments', [])):
            if cs['name'] == name:
                return ui.n_segments + 1 + ci
        return None

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _get_pairs_for_group(self, group_name: str, session_str: str, ct_str: str | None = None):
        ui = self.ui
        if group_name == '(all pairs)':
            pairs = set(map(tuple, ui.all_inds))
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
        """Return (pairs, cs_vals) for Conn Strength data."""
        ui = self.ui
        method = ui._conn_str_method_var.get()
        seg_idx = self._seg_name_to_idx(seg_name)
        if seg_idx is None:
            return [], []

        pairs = sorted(self._get_pairs_for_group(group_name, session_str, ct_str))
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
        from neuropy.analyses.ms_connectivity import apply_norms_to_ccg  # noqa: PLC0415
        ui = self.ui
        seg_idx = self._seg_name_to_idx(seg_name)
        if seg_idx is None:
            return [], []

        pairs = sorted(self._get_pairs_for_group(group_name, session_str, ct_str))
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
        name  = row.get('name').get() if row.get('name') is not None else ''
        sess  = row['sess'].get()
        ct    = row['ct'].get()
        seg   = row['seg'].get()
        grp   = row['grp'].get()
        dtype = row['data'].get()

        if dtype == "Conn Strength":
            pairs, vals = self._get_cs_values(sess, ct, seg, grp, highres=highres)
            # Mirror main UI display option: clamp CS to non-negative if enabled.
            try:
                if getattr(self.ui, '_conn_str_nonneg_var', None) is not None and self.ui._conn_str_nonneg_var.get():
                    vals = [max(float(v), 0.0) for v in (vals or [])]
            except Exception:
                pass
        elif dtype == "Firing Rate":
            pairs, vals = self._get_firing_rate_values(sess, ct, seg, grp)
        elif dtype == "Baseline":
            pairs, vals = self._get_baseline_values(sess, ct, seg, grp)
        else:
            pairs, vals = [], []

        return dict(name=name, session=sess, conn_type=ct, segment=seg, group=grp,
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
        a_lo, b_lo = groups_lo[0], groups_lo[1]
        res_lo = self._run_test(a_lo['vals'], b_lo['vals'],
                                a_lo['pairs'], b_lo['pairs'],
                                test_type, alternative, nonparametric)

        groups_hi = result_hi = None
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
        """Draw bar-with-whiskers + scatter for the first two groups."""
        if not hasattr(self, '_plot_fig') or self._plot_fig is None:
            return
        fig = self._plot_fig
        fig.clf()

        def _one_axis(ax, groups, title: str):
            if not groups or len(groups) < 2:
                ax.axis('off')
                return
            gA, gB = groups[0], groups[1]
            a = np.array(gA.get('vals', []) or [], dtype=float)
            b = np.array(gB.get('vals', []) or [], dtype=float)

            nameA = gA.get('name', 'A') or 'A'
            nameB = gB.get('name', 'B') or 'B'

            ax.set_title(title, fontsize=9, pad=2)
            ax.set_xticks([0, 1], [nameA, nameB])
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=8)

            # Mean ± SEM whiskers (fallback to std when n<2)
            def _mean_sem(x):
                if x.size == 0:
                    return np.nan, np.nan
                m = float(np.mean(x))
                if x.size < 2:
                    return m, float(np.std(x)) if x.size == 1 else np.nan
                return m, float(np.std(x, ddof=1) / np.sqrt(x.size))

            mA, eA = _mean_sem(a)
            mB, eB = _mean_sem(b)

            xs = np.array([0, 1], dtype=float)
            means = np.array([mA, mB], dtype=float)
            errs = np.array([eA, eB], dtype=float)

            ax.bar(xs, means, yerr=errs, capsize=4,
                   color=['#8FB3FF', '#FFB3B3'], edgecolor='#333', linewidth=0.8)

            # Scatter with small x-jitter
            rng = np.random.default_rng(0)
            for i, arr in enumerate([a, b]):
                if arr.size == 0:
                    continue
                jitter = rng.normal(0, 0.05, size=arr.size)
                ax.scatter(np.full(arr.size, xs[i]) + jitter, arr,
                           s=14, color='#222', alpha=0.6, linewidths=0)

            ax.grid(axis='y', alpha=0.25, linewidth=0.7)

        dtype = (groups_lo[0].get('data_type') if groups_lo else '') or ''
        if groups_hi is not None:
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
            _one_axis(ax1, groups_lo, f"{dtype} (Lo-res)")
            _one_axis(ax2, groups_hi, f"{dtype} (Hi-res)")
            fig.tight_layout(pad=0.8, w_pad=1.2)
        else:
            ax = fig.add_subplot(1, 1, 1)
            _one_axis(ax, groups_lo, f"{dtype}")
            fig.tight_layout(pad=0.8)

        try:
            self._plot_canvas.draw()
        except Exception:
            pass

    def _build_result_lines(self, groups, res, test_type,
                             groups_hi=None, res_hi=None) -> list[str]:
        # Hypotheses (use first two groups as A and B)
        a = groups[0].get('name', 'A') if groups else 'A'
        b = groups[1].get('name', 'B') if len(groups) > 1 else 'B'
        alt = (self._result_data or {}).get('alternative', 'two-sided')
        if alt == 'two-sided':
            h1 = f"H1: μ{a} ≠ μ{b}"
        elif alt == 'greater':
            h1 = f"H1: μ{a} > μ{b}"
        else:
            h1 = f"H1: μ{a} < μ{b}"
        h0 = f"H0: μ{a} = μ{b}"

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

        if groups_hi is not None:
            # Side-by-side lo vs hi
            lines.append(f"{'Lo-res':^30}  {'Hi-res':^30}")
            lines.append("-" * 62)
            if 'error' in res:
                lo_str = res['error']
            else:
                if 't_stat' in res:
                    stat_str = f"t={res['t_stat']:.4f}"
                elif 'stat' in res:
                    stat_str = f"stat={res['stat']:.4f}"
                else:
                    stat_str = "stat=?"
                lo_str = (f"{stat_str}  p={res['p_val']:.4f}"
                          f"  n={res.get('n_used', '?')}")
            if result_hi := res_hi:
                if 'error' in result_hi:
                    hi_str = result_hi['error']
                else:
                    if 't_stat' in result_hi:
                        stat_str = f"t={result_hi['t_stat']:.4f}"
                    elif 'stat' in result_hi:
                        stat_str = f"stat={result_hi['stat']:.4f}"
                    else:
                        stat_str = "stat=?"
                    hi_str = (f"{stat_str}  p={result_hi['p_val']:.4f}"
                              f"  n={result_hi.get('n_used', '?')}")
            else:
                hi_str = "—"
            lines.append(f"{lo_str:<30}  {hi_str:<30}")
        else:
            if 'error' in res:
                lines.append(res['error'])
            else:
                if 't_stat' in res:
                    lines.append(f"t = {res['t_stat']:.4f}")
                elif 'stat' in res:
                    lines.append(f"stat = {res['stat']:.4f}")
                else:
                    lines.append("stat = ?")
                lines.append(f"p = {res['p_val']:.4f}")
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
            method  = ui._conn_str_method_var.get()
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

        return (f"{test_type} on {ct_disp} in {sess_str}, "
                f"{labels_str}, on the {seg_str}. {data_desc}")

    def _export(self):
        if not self._result_data:
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            title="Export Stats Test Results",
        )
        if not path:
            return

        d  = self._result_data
        ui = self.ui

        def _group_block(i, g):
            bl = [f"  Group {g.get('name', chr(65+i))}: "
                  f"{g['session']} | {g['conn_type']} | {g['segment']} | {g['group']}"]
            bl.append(f"    n = {len(g['vals'])}")
            if g['vals']:
                bl.append(f"    mean ± std = {np.mean(g['vals']):.6f} ± {np.std(g['vals']):.6f}")
                bl.append("    values: " + ", ".join(f"{v:.6f}" for v in g['vals']))
            return bl

        def _res_block(res, label=""):
            bl = [f"  {label}".strip()]
            if 'error' in res:
                bl.append(f"  {res['error']}")
            else:
                if 't_stat' in res:
                    bl += [f"  t = {res['t_stat']:.4f}"]
                elif 'stat' in res:
                    bl += [f"  stat = {res['stat']:.4f}"]
                bl += [f"  p = {res['p_val']:.4f}",
                       f"  n used = {res.get('n_used', '?')}",
                       f"  test = {res.get('test', '')}"]
            return bl

        lines = ["=" * 60, "Stats Test Results", "=" * 60,
                 f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Test:  {d['test_type']}",
                 f"Data:  {d.get('dtype', '?')}",
                 f"Alt:   {d.get('alternative', 'two-sided')}",
                 f"H0:    μA = μB",
                 f"H1:    " + ("μA ≠ μB" if d.get('alternative') == 'two-sided'
                              else "μA > μB" if d.get('alternative') == 'greater'
                              else "μA < μB"),
                 f"Mode:  {'Nonparametric' if d.get('nonparametric') else 'Parametric'}",
                 f"Xform: {'Signed log1p' if d.get('log_transform') else 'None'}",
                 "", "Groups:"]
        for i, g in enumerate(d['groups']):
            lines += _group_block(i, g)

        lines += ["", "Statistical Results:", "-" * 40]
        lines += _res_block(d['res_lo'], label="Lo-res" if d.get('res_hi') else "")
        if d.get('res_hi'):
            lines += _res_block(d['res_hi'], label="Hi-res")
            if d.get('groups_hi'):
                lines += ["", "Groups (hi-res):"]
                for i, g in enumerate(d['groups_hi']):
                    lines += _group_block(i, g)

        lines += ["", "Methods:", "-" * 40, self._build_methods_string()]
        lines += ["", "CCGConfig:", "-" * 40]
        conf = getattr(getattr(ui, 'ccg_data', None), 'conf', None)
        if conf:
            for attr in ['name', 'bin_size', 'duration', 'min_lag', 'max_lag',
                         'alpha', 'n_jitter', 'n_jitter_shuffles']:
                v = getattr(conf, attr, None)
                if v is not None:
                    lines.append(f"  {attr}: {v}")
        else:
            lines.append("  (not available)")

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"[StatsPanel] Exported to {path}")
        except Exception as exc:
            print(f"[StatsPanel] Export error: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_result(self, text: str):
        if self._result_text:
            self._result_text.config(state=tk.NORMAL)
            self._result_text.delete('1.0', tk.END)
            self._result_text.insert('1.0', text)
            self._result_text.config(state=tk.DISABLED)
