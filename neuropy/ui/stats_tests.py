"""Qt stats-test panel for CCGReviewUI.

Floating persistent window for running statistical comparisons of connection
strengths, firing rates, and baselines across groups and sessions.

Architecture
------------
StatsTestPanelQt(QWidget)
  - setWindowFlags(Qt.Window)  → free-floating, non-modal
  - Dynamic group rows: each row = dict of Qt widget refs
  - Results: QSplitter → QPlainTextEdit (text) + FigureCanvas (matplotlib)
  - Data collection reads nav (key, sel_data) and cd (ptr, ccg, nd) directly.
    No global-state mutation between sessions (contrast with tkinter version's
    _stats_session_context which temporarily swapped ui.key/ccg_data).

CS values: computed on-demand from CCGData arrays.
Custom segments: loaded on-demand from nav.custom_seg_index (name → path).
"""
from __future__ import annotations

import datetime
import json
import pathlib
from typing import TYPE_CHECKING
from scipy import stats as _sp

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QPlainTextEdit,
    QScrollArea, QFrame, QMessageBox, QSizePolicy, QColorDialog,
)
from pyqtgraph.Qt.QtGui import QColor
from neuropy.ui.utils import ListPickerButton
from neuropy.ui.ui_common import qt_dark_mode
from neuropy.ui.dialogs import VersionSaveDialog, VersionLoadDialog
from neuropy.analyses.neurons_dataset import Key
from neuropy.ui.pair_selection_panel import SelectionData
from neuropy.analyses.utils import _compact_json_str
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState
    from neuropy.analyses.ms_connectivity import CCGDataset

_BAR_COLORS = ['#8FB3FF', '#FFB3B3', '#B3FFB3', '#FFD9B3', '#E0B3FF',
               '#B3F0FF', '#FFB3E6']
_DATA_TYPES   = ["Conn Strength", "CS norm (% change)", "CS norm (geometric)",
                 "Ref Firing Rate", "Tgt Firing Rate", "Baseline",
                 "Peak Width", "Peak Center"]
_DATA_DISABLED = {"Peak Width", "Peak Center"}
_CS_NORM_TYPES = {"CS norm (% change)", "CS norm (geometric)"}
_ALL_SEGS = "All"

_TOP_CFG = (
    ('test_type', '_test_type', 'setCurrentText'),
    ('sides', '_sides', 'setCurrentText'),
    ('direction', '_dir_btn', 'setText'),
    ('nonparametric', '_nonparam', 'setChecked'),
    ('log_transform', '_log', 'setChecked'),
    ('violin', '_violin_check', 'setChecked'),
    ('outliers', '_outliers_check', 'setChecked'),
    ('sig_brackets', '_sig_check', 'setChecked'),
)


class StatsTestPanelQt(QWidget):
    """Persistent floating stats-test panel.

    Constructor: StatsTestPanelQt(nav, cd)
    Call .show() to display; panel re-uses the same window.
    """

    def __init__(self, nav: 'AppState', cd: 'CCGDataset', parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self._nav = nav
        self._rows: list[dict] = []
        self._result_data: dict | None = None


        self.setWindowTitle("Stats Tests")
        self.resize(1100, 580)
        self._build()
        self._connect_nav()
        self._add_row()
        self._add_row()


    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        top = QHBoxLayout()
        top.addWidget(QLabel("Groups", styleSheet="font-weight:bold;"))
        top.addWidget(QLabel("Test type:"))
        self._test_type = QComboBox()
        for t in ["Independent t-test", "Pairwise t-test",
                  "One-way ANOVA + Tukey", "Repeated-measures ANOVA"]:
            self._test_type.addItem(t)
        self._test_type.setCurrentText("Pairwise t-test")
        self._test_type.setFixedWidth(200)
        top.addWidget(self._test_type)

        top.addWidget(QLabel("Sides:"))
        self._sides = QComboBox()
        self._sides.addItems(["Two-sided", "One-sided"])
        self._sides.setFixedWidth(90)
        top.addWidget(self._sides)

        self._dir_btn = QPushButton("A > B")
        self._dir_btn.setFixedWidth(60)
        self._dir_btn.clicked.connect(self._toggle_direction)
        top.addWidget(self._dir_btn)

        self._nonparam = QCheckBox("nonparametric")
        self._log      = QCheckBox("log-transform")
        top.addWidget(self._nonparam)
        top.addWidget(self._log)
        top.addStretch()
        root.addLayout(top)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 2, 0, 2)
        _hdr_fixed = {"": 22, "Color": 42, "Name": 52}
        for col in ("", "Color", "Name", "Session", "ConnType", "Segment", "Group", "Data", ""):
            lbl = QLabel(col, styleSheet="font-weight:bold; padding: 4px 4px;")
            if col in _hdr_fixed:
                lbl.setFixedWidth(_hdr_fixed[col])
            else:
                lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            hdr.addWidget(lbl)
        root.addLayout(hdr)

        self._rows_area = QVBoxLayout()
        self._rows_area.setSpacing(2)
        rows_widget = QWidget()
        rows_widget.setLayout(self._rows_area)
        scroll = QScrollArea()
        scroll.setWidget(rows_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        add_btn = QPushButton("+ Add group")
        add_btn.clicked.connect(self._add_row)
        add_btn.setFixedWidth(100)
        root.addWidget(add_btn)

        res_frame = QFrame()
        res_frame.setFrameShape(QFrame.Shape.StyledPanel)
        res_root = QVBoxLayout(res_frame)
        res_root.setContentsMargins(4, 4, 4, 4)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)

        self._result_text = QPlainTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFont(QtWidgets.QApplication.font())
        self._result_text.setMinimumHeight(80)
        self._splitter.addWidget(self._result_text)

        plot_container = QWidget()
        pc_layout = QVBoxLayout(plot_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_row = QHBoxLayout()
        self._violin_check   = QCheckBox("Violin")
        self._outliers_check = QCheckBox("Show outliers")
        self._outliers_check.setChecked(True)
        self._sig_check      = QCheckBox("Sig. brackets")
        ctrl_row.addWidget(self._violin_check)
        ctrl_row.addWidget(self._outliers_check)
        ctrl_row.addWidget(self._sig_check)
        ctrl_row.addWidget(QLabel("W:H"))
        self._wh_input = QLineEdit("3:1")
        self._wh_input.setFixedWidth(45)
        self._wh_input.editingFinished.connect(self._replot)
        ctrl_row.addWidget(self._wh_input)
        ctrl_row.addStretch()
        pc_layout.addLayout(ctrl_row)
        for chk in (self._violin_check, self._outliers_check, self._sig_check):
            chk.toggled.connect(self._replot)
        self._plot_fig = Figure(dpi=100)
        self._plot_canvas = FigureCanvasQTAgg(self._plot_fig)
        self._plot_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _canvas_resize = self._plot_canvas.resizeEvent
        self._plot_canvas.resizeEvent = lambda ev: (
            self._fit_plot(redraw_only=True), _canvas_resize(ev))
        pc_layout.addWidget(self._plot_canvas, stretch=1)
        self._splitter.addWidget(plot_container)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.splitterMoved.connect(lambda *_: self._fit_plot(redraw_only=True))

        res_root.addWidget(self._splitter)
        root.addWidget(res_frame, stretch=1)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._run)
        btn_row.addWidget(run_btn)
        self._export_btn = QPushButton("Save…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        btn_row.addWidget(self._export_btn)
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._load_result)
        btn_row.addWidget(load_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _connect_nav(self):
        self._nav.key_changed.connect(lambda _: self._refresh_combos())
        self._nav.custom_segs_changed.connect(self._refresh_combos)
        self._nav.groups.changed.connect(self._refresh_combos)
        self._nav.selection_changed.connect(self._on_selection_reset)

    def _on_selection_reset(self):
        self._refresh_combos()

    def _set_picker(self, picker, opts: list, cfg: dict | None, key: str,
                    default: list | None = None):
        if cfg and cfg.get(key):
            picker._selected = [x for x in cfg[key] if x in opts]
        elif default:
            picker._selected = [x for x in default if x in opts]
        picker._update_label()

    def _add_row(self, cfg: dict | None = None):
        sessions = self._concrete_sessions()
        ct_opts  = self._available_conn_types()
        seg_opts = self._available_segments()
        grp_opts = self._available_groups()
        idx      = len(self._rows)
        color    = (cfg or {}).get('color', _BAR_COLORS[idx % len(_BAR_COLORS)])

        row_widget = QWidget()
        rw = QHBoxLayout(row_widget)
        rw.setContentsMargins(0, 0, 0, 0)
        rw.setSpacing(4)

        color_btn = QPushButton()
        color_btn.setFixedSize(20, 20)
        color_btn.setStyleSheet(f"background:{color}; border:1px solid #888; border-radius:2px;")
        color_btn.clicked.connect(lambda: self._pick_color(color_btn, row))
        rw.addWidget(color_btn)

        name_edit = QLineEdit((cfg or {}).get('name', chr(65 + idx) if idx < 26 else f"G{idx+1}"))
        name_edit.setFixedWidth(42)
        rw.addWidget(name_edit)

        sess_picker = ListPickerButton("Session", sessions, plural="sessions")
        key_sess = str(self._nav.key.session)
        self._set_picker(sess_picker, sessions, cfg, 'sessions',
                         [key_sess] if key_sess in sessions else None)
        rw.addWidget(sess_picker)

        ct_picker = ListPickerButton("ConnType", ct_opts, plural="types")
        cur_ct = self._nav.key.conn_type
        ct_default = [f"{cur_ct[0]}-{cur_ct[1]}"] if cur_ct and f"{cur_ct[0]}-{cur_ct[1]}" in ct_opts else None
        self._set_picker(ct_picker, ct_opts, cfg, 'conn_types', ct_default)
        rw.addWidget(ct_picker)

        seg_picker = ListPickerButton("Segment", seg_opts, plural="segments")
        self._set_picker(seg_picker, seg_opts, cfg, 'segments')
        rw.addWidget(seg_picker)

        grp_picker = ListPickerButton("Group", grp_opts, plural="groups")
        grp_default = [grp_opts[1]] if len(grp_opts) > 1 else None
        self._set_picker(grp_picker, grp_opts, cfg, 'groups', grp_default)
        rw.addWidget(grp_picker)

        dtype_combo = QComboBox()
        for dt in _DATA_TYPES:
            dtype_combo.addItem(dt)
            if dt in _DATA_DISABLED:
                dtype_combo.model().item(dtype_combo.count() - 1).setEnabled(False)
        if cfg and cfg.get('data_type') in _DATA_TYPES:
            dtype_combo.setCurrentText(cfg['data_type'])
        rw.addWidget(dtype_combo)

        del_btn = QPushButton("×")
        del_btn.setFixedWidth(22)
        row = {'widget': row_widget, 'color': color, 'color_btn': color_btn,
               'name': name_edit, 'sess': sess_picker, 'ct': ct_picker,
               'seg': seg_picker, 'grp': grp_picker, 'dtype': dtype_combo}
        del_btn.clicked.connect(lambda: self._del_row(row))
        rw.addWidget(del_btn)

        self._rows.append(row)
        self._rows_area.addWidget(row_widget)

    def _pick_color(self, btn: QPushButton, row: dict):
        c = QColorDialog.getColor(QColor(row['color']), self)
        if c.isValid():
            row['color'] = c.name()
            btn.setStyleSheet(f"background:{c.name()}; border:1px solid #888; border-radius:2px;")

    def _del_row(self, row: dict):
        if row in self._rows:
            self._rows.remove(row)
        w = row.get('widget')
        if w:
            w.deleteLater()

    def _refresh_combos(self):
        sessions = self._concrete_sessions()
        seg_opts = self._available_segments()
        ct_opts  = self._available_conn_types()
        grp_opts = self._available_groups()
        for r in self._rows:
            r['sess'].set_items(sessions, keep_selection=True)
            r['seg'].set_items(seg_opts, keep_selection=True)
            r['ct'].set_items(ct_opts, keep_selection=True)
            r['grp'].set_items(grp_opts, keep_selection=True)


    def _concrete_sessions(self) -> list[str]:
        return list(dict.fromkeys(str(k.session) for k in self._nav.cd.ptr.keys()))

    def _available_conn_types(self) -> list[str]:
        seen = set()
        for k in self._nav.cd.ptr.keys():
            if k.conn_type:
                seen.add(f"{k.conn_type[0]}-{k.conn_type[1]}")
        return sorted(seen)

    def _available_segments(self) -> list[str]:
        segs = {_ALL_SEGS}
        for k in self._nav.cd.ptr.keys():
            segs.update(self._nav.cd.segment_names_for(k))
        # Custom segment names from registry
        segs.update(self._nav.custom_seg_index.keys())
        return sorted(segs)

    def _available_groups(self) -> list[str]:
        groups = [g for g in self._nav.groups.defined_groups if not g.startswith('__')]
        return ['(all pairs)'] + sorted(groups)

    def _toggle_direction(self):
        cur = self._dir_btn.text().strip()
        self._dir_btn.setText("A < B" if cur == "A > B" else "A > B")


    def _key_for(self, sess_str: str, ct_str: str):
        """Find matching Key in cd.ptr for (session, conn_type) without mutating state."""
        for k in self._nav.cd.ptr.keys():
            if str(k.session) != sess_str:
                continue
            if not ct_str or (k.conn_type and f"{k.conn_type[0]}-{k.conn_type[1]}" == ct_str):
                return k
        return None

    def _seg_idx_for(self, key, seg_name: str) -> int | None:
        """Segment index for a given key. Returns n_segments for All."""
        if self._nav.cd.ptr.get(key) is None:
            return None
        if seg_name in (_ALL_SEGS, 'All segments', 'All', ''):
            return self._nav.cd.n_segments_for(key)
        names = self._nav.cd.segment_names_for(key)
        for i, lb in enumerate(names):
            if str(lb) == seg_name:
                return i
        return None

    def _get_pairs_for_group(self, group_name: str, sess_str: str, key) -> set:
        ptr = self._nav.cd.ptr.get(key)
        valid = {tuple(p) for p in ptr.inds2} if ptr is not None else set()
        if group_name == '(all pairs)':
            return valid
        return self._nav.groups.pairs_in_group(group_name, sess_str) & valid


    def _compute_cs_value(self, data, seg_idx: int, ref: int, tgt: int) -> float | None:
        """Compute STG or JBSI for one pair from CCGData arrays."""
        metric = self._nav.cs_metric
        if metric == 'JBSI':
            if data.pval_corrected is None:
                return None
            return float(1.0 - data.pval_corrected[seg_idx, ref, tgt])
        # STG: peak(ccg - null) in test window / mean(null)
        ccg     = data.ccg[seg_idx, ref, tgt, :]
        null_1d = data.ccg_null[seg_idx, ref, tgt, :]
        conf    = data.conf
        min_lag = conf.min_lag
        max_lag = conf.max_lag
        dur     = conf.duration
        n  = len(ccg)
        bs = dur / (n - 1) if n > 1 else 0.001
        lo = max(0, int(round(min_lag / bs)) + n // 2)
        hi = min(n, int(round(max_lag / bs)) + n // 2)
        bl = float(np.mean(null_1d)) if null_1d.size else 0.0
        if bl <= 0 or lo >= hi:
            return None
        return float(np.max(ccg[lo:hi] - null_1d[lo:hi])) / bl


    def _get_cs_values_for_sess(self, sess_str: str, ct_str: str,
                                 seg_name: str, group_name: str,
                                 highres: bool) -> tuple[list, list]:
        key = self._key_for(sess_str, ct_str)
        if key is None:
            return [], []
        nd_key = key.nd()
        if highres:
            data = self._nav.cd.ccg_for(nd_key, 'highres') or self._nav.cd.ccg_for(nd_key, 'lowres')
        else:
            data = self._nav.cd.ccg_for(nd_key, 'lowres')
        if data is None:
            return [], []
        seg_idx = self._seg_idx_for(key, seg_name)
        if seg_idx is None:
            return [], []
        pairs = sorted(self._get_pairs_for_group(group_name, sess_str, key))
        vals, valid = [], []
        for ref, tgt in pairs:
            v = self._compute_cs_value(data, seg_idx, int(ref), int(tgt))
            if v is not None:
                vals.append(v)
                valid.append(Key.pair(sess_str, ref, tgt))
        return valid, vals

    def _get_fr_for_sess(self, sess_str: str, ct_str: str,
                          group_name: str, role: int) -> tuple[list, list]:
        key = self._key_for(sess_str, ct_str)
        if key is None:
            return [], []
        nd_key = key.nd()
        neurons = self._nav.cd.nd.data.get(nd_key)
        if neurons is None:
            return [], []
        fr = neurons.firing_rate
        pairs = sorted(self._get_pairs_for_group(group_name, sess_str, key))
        seen, ids, vals = set(), [], []
        for p in pairs:
            idx = int(p[role])
            if idx not in seen:
                seen.add(idx)
                ids.append(Key.pair(sess_str, idx, idx))
                vals.append(float(fr[idx]))
        return ids, vals

    def _get_baseline_for_sess(self, sess_str: str, ct_str: str,
                                seg_name: str, group_name: str) -> tuple[list, list]:
        key = self._key_for(sess_str, ct_str)
        if key is None:
            return [], []
        data = self._nav.cd.ccg_for(key.nd(), 'lowres')
        if data is None:
            return [], []
        null = data.ccg_null
        if null is None:
            return [], []
        seg_idx = self._seg_idx_for(key, seg_name)
        if seg_idx is None:
            return [], []
        pairs = sorted(self._get_pairs_for_group(group_name, sess_str, key))
        vals, valid = [], []
        for ref, tgt in pairs:
            bl = float(np.mean(null[seg_idx, int(ref), int(tgt), :]))
            vals.append(bl)
            valid.append(Key.pair(sess_str, ref, tgt))
        return valid, vals

    def _collect_group_data(self, row: dict, highres: bool = False) -> dict:
        sessions  = row['sess'].selected or self._concrete_sessions()
        ct_strs   = row['ct'].selected
        seg_names = row['seg'].selected or [_ALL_SEGS]
        grp_names = row['grp'].selected or ['(all pairs)']
        dtype     = row['dtype'].currentText()
        name      = row['name'].text()
        color     = row.get('color', _BAR_COLORS[self._rows.index(row) % len(_BAR_COLORS)])

        ct_str = ct_strs[0] if ct_strs else ''

        seen: set = set()
        pairs: list = []
        vals:  list = []

        for grp_name in grp_names:
            for seg_name in seg_names:
                for sess in sessions:
                    if dtype in {"Conn Strength"} | _CS_NORM_TYPES:
                        gp, gv = self._get_cs_values_for_sess(
                            sess, ct_str, seg_name, grp_name, highres)
                    elif dtype in ("Ref Firing Rate", "Tgt Firing Rate"):
                        role = 0 if dtype == "Ref Firing Rate" else 1
                        gp, gv = self._get_fr_for_sess(sess, ct_str, grp_name, role)
                    elif dtype == "Baseline":
                        gp, gv = self._get_baseline_for_sess(
                            sess, ct_str, seg_name, grp_name)
                    else:
                        gp, gv = [], []

                    for p, v in zip(gp, gv):
                        pk = SelectionData.as_pair_key(p)
                        if pk not in seen:
                            seen.add(pk)
                            pairs.append(pk)
                            vals.append(v)

        sess_str = sessions[0] if len(sessions) == 1 else ','.join(sessions)
        return dict(name=name, session=sess_str, conn_type=ct_str,
                    segment=seg_names[0], group=grp_names[0], data_type=dtype,
                    pairs=pairs, vals=vals, highres=highres, color=color)


    def _apply_log_transform_to_groups(self, groups):
        return [dict(g, vals=list(self._maybe_log_transform(
            np.asarray(g.get('vals', []) or [], dtype=float)))) for g in groups]

    def _maybe_log_transform(self, x: np.ndarray) -> np.ndarray:
        if not (self._log.isChecked()):
            return x
        shifted = x - np.nanmin(x) + 1e-9 if np.any(x <= 0) else x
        return np.log(shifted)

    def _run_anova(self, groups: list[dict]) -> dict:
        nonparam = self._nonparam.isChecked()
        arrays = [np.asarray(g.get('vals', []) or [], dtype=float) for g in groups]
        arrays = [self._maybe_log_transform(a) for a in arrays]
        if nonparam:
            stat, p = _sp.kruskal(*[a for a in arrays if a.size])
            return dict(test='Kruskal-Wallis', stat=float(stat), p_val=float(p))
        stat, p = _sp.f_oneway(*[a for a in arrays if a.size])
        result = dict(test='One-way ANOVA', f_stat=float(stat), p_val=float(p),
                      n_groups=len(groups))
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd  # noqa: PLC0415
            combined = np.concatenate(arrays)
            labels = []
            for g, a in zip(groups, arrays):
                labels.extend([g.get('name', '')] * len(a))
            tukey = pairwise_tukeyhsd(combined, labels)
            rows = []
            for row in tukey.summary().data[1:]:
                a_nm, b_nm, meandiff, p_adj, lo, hi, reject = row
                rows.append(dict(a=str(a_nm), b=str(b_nm), meandiff=float(meandiff),
                                 p_adj=float(p_adj), reject=bool(reject)))
            result['tukey'] = rows
        except ImportError:
            result['tukey_missing'] = True
        return result

    def _run_rm_anova(self, groups: list[dict]) -> dict:
        pair_maps = []
        for g in groups:
            pairs = [SelectionData.as_pair_key(p) for p in (g.get('pairs') or [])]
            vals  = self._maybe_log_transform(
                np.asarray(g.get('vals', []) or [], dtype=float))
            pair_maps.append({p: v for p, v in zip(pairs, vals)})
        common = sorted(set.intersection(*(set(pm) for pm in pair_maps)))
        if len(common) < 2:
            return {'error': f"Need ≥2 pairs in all groups (found {len(common)})."}
        group_names = [g.get('name', f'G{i+1}') for i, g in enumerate(groups)]
        arrays = [np.array([pm[p] for p in common], dtype=float) for pm in pair_maps]
        nonparam = self._nonparam.isChecked()
        n_comp = max(1, len(groups) * (len(groups) - 1) // 2)
        if nonparam:
            stat, p = _sp.friedmanchisquare(*arrays)
            posthoc = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    w, wp = _sp.wilcoxon(arrays[i], arrays[j], zero_method='wilcox')
                    posthoc.append(dict(a=group_names[i], b=group_names[j],
                                        stat=float(w), p_raw=float(wp),
                                        p_adj=min(float(wp)*n_comp, 1.0),
                                        reject=float(wp)*n_comp < 0.05))
            return dict(test='Friedman test', stat=float(stat), p_val=float(p),
                        n_subjects=len(common), n_conditions=len(groups),
                        posthoc=posthoc, posthoc_method='Wilcoxon (Bonferroni)',
                        common_pairs=common)
        try:
            import pingouin as pg  # noqa: PLC0415
            import pandas as pd    # noqa: PLC0415
            rows = [{'subject': str(p), 'condition': gn, 'val': pm[p]}
                    for gn, pm in zip(group_names, pair_maps) for p in common]
            df  = pd.DataFrame(rows)
            aov = pg.rm_anova(data=df, dv='val', within='condition',
                              subject='subject', detailed=True)
            cr  = aov[aov['Source'] == 'condition'].iloc[0]
            df_num   = float(cr.get('DF1', cr.get('ddof1', float('nan'))))
            df_denom = float(cr.get('DF2', cr.get('ddof2', float('nan'))))
            return dict(test='Repeated-measures ANOVA',
                        f_stat=float(cr['F']), p_val=float(cr['p-unc']),
                        df=f"{df_num:.0f},{df_denom:.0f}",
                        n_subjects=len(common), n_conditions=len(groups),
                        common_pairs=common)
        except ImportError:
            pass
        try:
            from statsmodels.formula.api import mixedlm  # noqa: PLC0415
            import pandas as pd  # noqa: PLC0415
            rows = [{'subject': str(p), 'condition': float(gi), 'val': pm[p]}
                    for gi, (gn, pm) in enumerate(zip(group_names, pair_maps))
                    for p in common]
            df  = pd.DataFrame(rows)
            res = mixedlm("val ~ condition", df, groups=df["subject"]).fit(reml=True)
            return dict(test='Mixed LM (statsmodels)',
                        p_val=float(res.pvalues.get('condition', float('nan'))),
                        n_subjects=len(common), n_conditions=len(groups),
                        common_pairs=common)
        except ImportError:
            pass
        stat, p = _sp.friedmanchisquare(*arrays)
        return dict(test='Friedman (fallback)', stat=float(stat), p_val=float(p),
                    n_subjects=len(common), n_conditions=len(groups),
                    common_pairs=common)

    def _run_test(self, a_vals, b_vals, a_pairs, b_pairs,
                  test_type: str, alternative: str, nonparametric: bool) -> dict:
        a = self._maybe_log_transform(np.asarray(a_vals or [], dtype=float))
        b = self._maybe_log_transform(np.asarray(b_vals or [], dtype=float))
        if a.size < 2 or b.size < 2:
            return {'error': f"Need ≥2 values per group (got {a.size}, {b.size})."}
        paired = (test_type == "Pairwise t-test")
        if paired:
            a_map = SelectionData.pairs_vals_map(a_pairs, a)
            b_map = SelectionData.pairs_vals_map(b_pairs, b)
            common = sorted(set(a_map) & set(b_map))
            if len(common) < 2:
                return {'error': f"Only {len(common)} matched pairs — need ≥2."}
            a = np.array([a_map[p] for p in common])
            b = np.array([b_map[p] for p in common])
        if nonparametric:
            if paired:
                stat, p = _sp.wilcoxon(a, b, zero_method='wilcox',
                                       alternative=alternative)
                test_name = 'Wilcoxon signed-rank'
            else:
                stat, p = _sp.mannwhitneyu(a, b, alternative=alternative)
                test_name = 'Mann-Whitney U'
        else:
            if paired:
                stat, p = _sp.ttest_rel(a, b, alternative=alternative)
                test_name = 'Paired t-test'
            else:
                stat, p = _sp.ttest_ind(a, b, equal_var=False, alternative=alternative)
                test_name = "Welch's t-test"
        return dict(test=test_name, stat=float(stat), p_val=float(p),
                    n_a=int(a.size), n_b=int(b.size),
                    mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                    sem_a=float(np.std(a,ddof=1)/np.sqrt(a.size)),
                    sem_b=float(np.std(b,ddof=1)/np.sqrt(b.size)),
                    paired=paired, alternative=alternative)

    def _run_cs_norm(self, g_a: dict, g_b: dict, dtype: str) -> dict:
        # Match by (ref, tgt) only — sessions differ intentionally across groups
        a_map = {p.get('ref', 'tgt'): v for p, v in zip(g_a.get('pairs', []), g_a.get('vals', []))}
        b_map = {p.get('ref', 'tgt'): v for p, v in zip(g_b.get('pairs', []), g_b.get('vals', []))}
        common = sorted(set(a_map) & set(b_map))
        if len(common) < 2:
            return {'error': f"Only {len(common)} matched pairs."}
        a = np.array([a_map[p] for p in common], dtype=float)
        b = np.array([b_map[p] for p in common], dtype=float)
        if '% change' in dtype:
            with np.errstate(divide='ignore', invalid='ignore'):
                norm = np.where(a != 0, (b - a) / np.abs(a), np.nan)
        else:  # geometric: (b-a) / sqrt(|a|*|b|)
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = np.sqrt(np.abs(a) * np.abs(b))
                norm = np.where(denom > 0, (b - a) / denom, np.nan)
        norm = norm[np.isfinite(norm)]
        if norm.size < 2:
            return {'error': f"Only {norm.size} finite normalized values."}
        stat, p = _sp.ttest_1samp(norm, 0.0, alternative='two-sided')
        return dict(test='One-sample t-test', stat=float(stat), p_val=float(p),
                    n=int(norm.size), mean=float(np.mean(norm)),
                    sem=float(np.std(norm, ddof=1) / np.sqrt(norm.size)) if norm.size > 1 else 0.0,
                    norm_vals=norm.tolist(), norm_pairs=common)


    def _dispatch_test(self, test_type: str, groups: list[dict],
                       dtype: str, alternative: str, nonparam: bool) -> dict:
        if test_type == "Repeated-measures ANOVA":
            return self._run_rm_anova(groups)
        if len(groups) > 2:
            return self._run_anova(groups)
        if dtype in _CS_NORM_TYPES:
            return self._run_cs_norm(groups[0], groups[1], dtype)
        a, b = groups[0], groups[1]
        return self._run_test(a['vals'], b['vals'], a['pairs'], b['pairs'],
                              test_type, alternative, nonparam)

    def _run(self):
        active = [r for r in self._rows if r.get('widget') and r['widget'].isVisible()]
        if len(active) < 2:
            self._show_result("Need at least 2 groups to compare.")
            return
        dtype = active[0]['dtype'].currentText()
        if dtype in _DATA_DISABLED:
            self._show_result(f"Data type '{dtype}' is not yet implemented.")
            return
        if dtype in _CS_NORM_TYPES and len(active) != 2:
            self._show_result(f"'{dtype}' requires exactly 2 groups.")
            return

        test_type = self._test_type.currentText()
        run_hilo  = dtype in {"Conn Strength"} | _CS_NORM_TYPES
        sides = self._sides.currentText()
        alternative = ("two-sided" if sides == "Two-sided" else
                       ("greater" if self._dir_btn.text().strip() == "A > B" else "less"))
        nonparam = self._nonparam.isChecked()

        all_sessions = list(dict.fromkeys(
            s for r in active
            for s in (r['sess'].selected or self._concrete_sessions())
        ))
        self._nav.ensure_groups_loaded_for(all_sessions)

        if test_type == "Pairwise t-test":
            sessions_per_row = [r['sess'].selected or self._concrete_sessions() for r in active]
            if len(set(map(tuple, sessions_per_row))) > 1:
                self._show_result("Pairwise t-test requires same sessions in all groups.")
                return

        groups_lo = [self._collect_group_data(r, highres=False) for r in active]
        groups_hi = [self._collect_group_data(r, highres=True) for r in active] if run_hilo else None
        res_lo = self._dispatch_test(test_type, groups_lo, dtype, alternative, nonparam)
        res_hi = self._dispatch_test(test_type, groups_hi, dtype, alternative, nonparam) if groups_hi else None

        self._result_data = dict(
            groups=groups_lo, test_type=test_type, dtype=dtype,
            res_lo=res_lo, groups_hi=groups_hi, res_hi=res_hi,
            alternative=alternative, nonparametric=nonparam)

        lines = self._build_result_lines(groups_lo, res_lo, test_type,
                                          groups_hi=groups_hi, res_hi=res_hi)
        self._show_result('\n'.join(lines))

        if test_type == "Repeated-measures ANOVA":
            def _trim(groups, result):
                common = result.get('common_pairs') if result and 'error' not in result else None
                if not common:
                    return groups
                trimmed = []
                for g in groups:
                    pmap = SelectionData.pairs_vals_map(g.get('pairs'), g.get('vals'))
                    trimmed.append(dict(g, vals=[pmap[p] for p in common if p in pmap],
                                        pairs=list(common)))
                return trimmed
            self._result_data['is_paired_override'] = True
            self._result_data['plot_groups_lo'] = _trim(groups_lo, res_lo)
            self._result_data['plot_groups_hi'] = _trim(groups_hi, res_hi) if groups_hi else None
        elif dtype in _CS_NORM_TYPES:
            a_name = groups_lo[0].get('name', 'A')
            b_name = groups_lo[1].get('name', 'B') if len(groups_lo) > 1 else 'B'
            lbl = (f"({a_name}−{b_name})/{a_name}" if '% change' in dtype
                   else f"({a_name}−{b_name})/({a_name}+{b_name})")
            def _norm_grp(result, lb, dt):
                if not result or 'error' in result or 'norm_vals' not in result:
                    return None
                return [dict(name=lb, vals=list(result['norm_vals']),
                             pairs=list(result.get('norm_pairs', [])), data_type=dt)]
            self._result_data['is_one_sample'] = True
            self._result_data['plot_groups_lo'] = _norm_grp(res_lo, lbl, dtype)
            self._result_data['plot_groups_hi'] = _norm_grp(res_hi, lbl, dtype) if res_hi else None
        else:
            self._result_data['plot_groups_lo'] = self._apply_log_transform_to_groups(groups_lo)
            self._result_data['plot_groups_hi'] = (self._apply_log_transform_to_groups(groups_hi)
                                                   if groups_hi else None)

        self._update_result_plot()
        self._export_btn.setEnabled(True)

    @staticmethod
    def _get_pairwise_pvals(result, groups) -> list:
        if not result or 'error' in result:
            return []
        for key in ('tukey', 'posthoc'):
            rows = result.get(key)
            if rows:
                return [{'a': r.get('a',''), 'b': r.get('b',''),
                          'p_adj': r.get('p_adj', 1.0)} for r in rows]
        return []


    def _replot(self):
        if self._result_data:
            self._update_result_plot()

    def _fit_plot(self, redraw_only: bool = False):
        w = max(self._plot_canvas.width(), 50)
        h = max(self._plot_canvas.height(), 50)
        dpi = self._plot_fig.get_dpi()
        try:
            wr, hr = (float(x) for x in self._wh_input.text().strip().split(':'))
            ratio = max(wr, 0.1) / max(hr, 0.1)
            fig_w, fig_h = w / dpi, h / dpi
            if fig_w / fig_h > ratio:
                fig_w = fig_h * ratio
            else:
                fig_h = fig_w / ratio
        except Exception:
            fig_w, fig_h = w / dpi, h / dpi
        self._plot_fig.set_size_inches(fig_w, fig_h, forward=False)
        if redraw_only and self._plot_fig.axes:
            self._plot_fig.tight_layout(pad=0.8)
            self._plot_canvas.draw_idle()

    def _update_result_plot(self):
        rd = self._result_data or {}
        groups_lo = rd.get('plot_groups_lo') or []
        groups_hi = rd.get('plot_groups_hi')
        fig = self._plot_fig
        fig.clf()
        self._fit_plot()

        dark = qt_dark_mode()
        if dark:
            fig.patch.set_facecolor('#2b2b2b')
        else:
            fig.patch.set_facecolor('white')

        use_violin    = self._violin_check.isChecked()
        show_outliers = self._outliers_check.isChecked()
        show_sig      = self._sig_check.isChecked()
        sig_lo = self._get_pairwise_pvals(rd.get('res_lo'), groups_lo or []) if show_sig else []
        sig_hi = self._get_pairwise_pvals(rd.get('res_hi'), groups_hi or []) if show_sig else []

        def _star(p):
            return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else None))

        def _mean_sem(x):
            if x.size == 0: return np.nan, 0.0
            return float(np.mean(x)), (float(np.std(x,ddof=1)/np.sqrt(x.size)) if x.size>1 else 0.0)

        fg, bg  = ('#ffffff', '#1e1e1e') if dark else ('#000000', 'white')
        sp_col  = '#555555' if dark else '#cccccc'

        def _draw(ax, groups, title, paired=False, sig_pairs=[]):
            if not groups:
                ax.axis('off'); return
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=8)
            ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for sp in ax.spines.values(): sp.set_color(sp_col)
            n_g    = len(groups)
            names  = [g.get('name', chr(65+i)) or chr(65+i) for i, g in enumerate(groups)]
            arrays = [np.array(g.get('vals',[]) or [], dtype=float) for g in groups]
            colors = [g.get('color') or _BAR_COLORS[i % len(_BAR_COLORS)] for i, g in enumerate(groups)]
            pairs_lists = [g.get('pairs',[]) for g in groups]
            xs = np.arange(n_g, dtype=float)
            ax.set_title(title, fontsize=9, pad=2)
            ax.set_xticks(xs); ax.set_xticklabels(names)
            ax.tick_params(axis='x', labelsize=9); ax.tick_params(axis='y', labelsize=8)
            rng = np.random.default_rng(0)
            all_xpos = [np.full(a.size, xs[i]) + rng.normal(0,.05,a.size) for i, a in enumerate(arrays)]
            if use_violin:
                nonempty = [(xs[i], arr, colors[i]) for i, arr in enumerate(arrays) if arr.size >= 2]
                if nonempty:
                    vp = ax.violinplot([a for _, a, _ in nonempty],
                                       positions=[x for x, _, _ in nonempty],
                                       showmedians=True, showextrema=True, widths=0.6)
                    for i, pc in enumerate(vp.get('bodies', [])):
                        pc.set_facecolor(nonempty[i][2]); pc.set_alpha(0.85)
                        pc.set_edgecolor('#333' if not dark else '#aaa'); pc.set_linewidth(0.8)
                    med = vp.get('cmedians')
                    if med:
                        med.set_linewidth(2); med.set_color(fg)
            else:
                means = [_mean_sem(a)[0] for a in arrays]
                errs  = [_mean_sem(a)[1] for a in arrays]
                ax.bar(xs, means, yerr=errs, capsize=4, color=colors, edgecolor='#333', lw=0.8)
            if paired and n_g >= 2:
                gmaps = [{SelectionData.as_pair_key(p): (all_xpos[gi][j], float(arrays[gi][j]))
                          for j, p in enumerate(pairs_lists[gi])
                          if j < len(arrays[gi])} for gi in range(n_g)]
                ck = set(gmaps[0])
                for gm in gmaps[1:]:
                    ck &= set(gm)
                for pk in ck:
                    ax.plot([gmaps[gi][pk][0] for gi in range(n_g)],
                            [gmaps[gi][pk][1] for gi in range(n_g)],
                            color='#888', alpha=0.1, lw=0.7, zorder=1)
            for i, (arr, xpos) in enumerate(zip(arrays, all_xpos)):
                if arr.size == 0: continue
                mean_v = float(np.mean(arr)); std_v = float(np.std(arr,ddof=1)) if arr.size>1 else 0.0
                if show_outliers:
                    ax.scatter(xpos, arr, s=14, color='#222', alpha=0.2, lw=0, zorder=2)
                else:
                    mask = np.abs(arr-mean_v) <= 3*std_v if std_v>0 else np.ones(arr.size,bool)
                    ax.scatter(xpos[mask], arr[mask], s=14, color='#222', alpha=0.2, lw=0, zorder=2)
            ax.grid(axis='y', alpha=0.25, lw=0.7)
            if rd.get('is_one_sample'):
                ax.axhline(0, color='#555', lw=0.8, ls='--', zorder=0)
            if sig_pairs:
                nm2x = {n: x for n, x in zip(names, xs)}
                valid = [(r['a'],r['b'],r['p_adj']) for r in sig_pairs
                         if r['a'] in nm2x and r['b'] in nm2x]
                if valid:
                    ylo, ytop = ax.get_ylim(); span = max(abs(ytop-ylo), 1e-6)
                    step = span * 0.13; tick = step * 0.3
                    valid.sort(key=lambda t: abs(nm2x[t[0]]-nm2x[t[1]]))
                    ybase = ytop + step * 0.2
                    sig_valid = [(a, b, p) for a, b, p in valid if _star(p) is not None]
                    for lv, (a_nm, b_nm, p_adj) in enumerate(sig_valid):
                        x0, x1 = nm2x[a_nm], nm2x[b_nm]; y = ybase + step*lv
                        ax.plot([x0,x0,x1,x1],[y-tick,y,y,y-tick], color='#333', lw=0.9, clip_on=False)
                        ax.text((x0+x1)/2, y+tick*0.2, _star(p_adj),
                                ha='center', va='bottom', fontsize=8, color='#333', clip_on=False)
                    if sig_valid:
                        ax.set_ylim(top=ybase+step*(len(sig_valid)+0.8))

        dtype = (groups_lo[0].get('data_type') if groups_lo else rd.get('dtype', '')) or ''
        is_paired = (rd.get('test_type') == "Pairwise t-test" or rd.get('is_paired_override', False))
        if groups_hi is not None:
            ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
            _draw(ax1, groups_lo, f"{dtype} (Lo-res)", paired=is_paired, sig_pairs=sig_lo)
            _draw(ax2, groups_hi, f"{dtype} (Hi-res)", paired=is_paired, sig_pairs=sig_hi)
            fig.tight_layout(pad=0.8, w_pad=1.2)
        elif groups_lo:
            _draw(fig.add_subplot(1, 1, 1), groups_lo, dtype, paired=is_paired, sig_pairs=sig_lo)
            fig.tight_layout(pad=0.8)
        else:
            self._plot_canvas.draw()
            return
        self._plot_canvas.draw()

    def _build_result_lines(self, groups, res, test_type,
                             groups_hi=None, res_hi=None) -> list[str]:
        dtype = (self._result_data or {}).get('dtype', '')
        alt = (self._result_data or {}).get('alternative', 'two-sided')

        def _fmt_res(r, groups_) -> list[str]:
            if not r: return ["(no result)"]
            if 'error' in r: return [f"Error: {r['error']}"]
            lines = [f"  Test: {r.get('test','?')}",
                     f"  N: {[len(g.get('vals') or []) for g in groups_]}",
                     f"  Means: {[round(float(np.mean(g['vals'])),4) if g.get('vals') else 'n/a' for g in groups_]}"]
            if 'f_stat' in r:
                lines.append(f"  F = {r['f_stat']:.4f}")
            elif 'stat' in r:
                lines.append(f"  stat = {r['stat']:.4f}")
            if 'p_val' in r:
                p = r['p_val']
                lines.append(f"  p = {p:.4g}" + (" ***" if p<0.001 else (" **" if p<0.01 else (" *" if p<0.05 else ""))))
            if r.get('tukey') or r.get('posthoc'):
                rows = r.get('tukey') or r.get('posthoc') or []
                for row in rows:
                    lines.append(f"    {row.get('a','?')} vs {row.get('b','?')}: p_adj={row.get('p_adj',1.0):.4g}")
            return lines

        out = [f"── Stats: {test_type} ({'two-sided' if alt=='two-sided' else alt}) ──",
               f"   dtype={dtype}  lo-res:"]
        out += _fmt_res(res, groups)
        if groups_hi and res_hi:
            out.append("   hi-res:")
            out += _fmt_res(res_hi, groups_hi)
        return out


    def _show_result(self, text: str):
        self._result_text.setPlainText(text)

    def _row_cfg(self, row: dict) -> dict:
        return {
            'color': row.get('color'),
            'name': row['name'].text(),
            'sessions': row['sess'].selected,
            'conn_types': row['ct'].selected,
            'segments': row['seg'].selected,
            'groups': row['grp'].selected,
            'data_type': row['dtype'].currentText(),
        }

    def _config_dict(self) -> dict:
        cfg = {key: getattr(getattr(self, attr), meth)()  # widget attribute access by name
               for key, attr, meth in (
                   ('test_type', '_test_type', 'currentText'),
                   ('sides', '_sides', 'currentText'),
                   ('direction', '_dir_btn', 'text'),
                   ('nonparametric', '_nonparam', 'isChecked'),
                   ('log_transform', '_log', 'isChecked'),
                   ('violin', '_violin_check', 'isChecked'),
                   ('outliers', '_outliers_check', 'isChecked'),
                   ('sig_brackets', '_sig_check', 'isChecked'),
               )}
        cfg['direction'] = cfg['direction'].strip()
        cfg['wh_ratio'] = self._wh_input.text().strip()
        cfg['splitter_sizes'] = self._splitter.sizes()
        cfg['rows'] = [self._row_cfg(r) for r in self._rows
                       if r.get('widget') and r['widget'].isVisible()]
        return cfg

    def _apply_config(self, cfg: dict | None):
        if not cfg:
            return
        for r in list(self._rows):
            self._del_row(r)
        for rc in cfg.get('rows') or []:
            self._add_row(rc)
        while len(self._rows) < 2:
            self._add_row()
        for key, attr, meth in _TOP_CFG:
            if key in cfg:
                getattr(getattr(self, attr), meth)(cfg[key])
        if cfg.get('wh_ratio'):
            self._wh_input.setText(cfg['wh_ratio'])
        sizes = cfg.get('splitter_sizes')
        if sizes and len(sizes) == 2:
            self._splitter.setSizes(sizes)

    def _stats_save_dir(self) -> str:
        root = self._nav.root
        save_dir = (pathlib.Path(root.paths.data_root) / root.paths.project_dir / 'stats_results'
                    if hasattr(root, 'paths') else pathlib.Path('data') / 'stats_results')
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir)

    def _stats_versions(self) -> list:
        out = []
        for p in sorted(pathlib.Path(self._stats_save_dir()).glob('*.json')):
            try:
                meta = json.loads(p.read_text())
                saved_at = meta.get('saved_at', str(p.stat().st_mtime))
            except Exception:
                saved_at = ''
            out.append((p.stem, str(p), saved_at, True, False))
        return out

    def _export(self):
        if not self._result_data:
            return
        default = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        def _do_save(name):
            path = str(pathlib.Path(self._stats_save_dir()) / f"{name}.json")
            data = dict(self._result_data, config=self._config_dict(),
                        saved_at=datetime.datetime.now().isoformat())
            atomic_write_json(path, text=_compact_json_str(data))
        VersionSaveDialog.show(self, "Save Stats Result", default, on_save=_do_save)

    def _load_result(self):
        versions = self._stats_versions()
        def _do_load(path):
            try:
                data = json.loads(pathlib.Path(path).read_text())
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", str(exc))
                return
            self._result_data = data
            self._apply_config(data.get('config'))
            lines = self._build_result_lines(
                data.get('groups', []), data.get('res_lo'),
                data.get('test_type', ''),
                data.get('groups_hi'), data.get('res_hi'))
            self._show_result('\n'.join(lines))
            self._update_result_plot()
            self._export_btn.setEnabled(True)
        VersionLoadDialog.show(self, "Load Stats Result", versions, on_load=_do_load,
                               empty_msg="No saved stats results found.")
