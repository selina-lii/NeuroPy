"""Floating stats-test panel: compare connection strengths, firing rates, and
baselines across groups/sessions.

Layering (same file, logical split):
- Metric registry + StatsPanelBackend + data classes touch **no Qt** — compute layer.
- StatsRow / StatsPlotWidget / StatsTestPanel are the Qt frontend, calling the backend.
Widgets are the source of truth; configs are snapshots taken at run/save.
"""
from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal
from scipy import stats as _sp

import numpy as np
import pandas as pd
import matplotlib

from neuropy.analyses import ccg_transforms
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pingouin as pg

from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt.QtCore import Qt, Signal
from pyqtgraph.Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QPlainTextEdit,
    QScrollArea, QFrame, QMessageBox, QSizePolicy, QColorDialog,
)
from pyqtgraph.Qt.QtGui import QColor
from neuropy.ui.utils import ListPickerButton, make_combo, make_button
from neuropy.ui.ui_common import qt_dark_mode
from neuropy.ui.dialogs import VersionSaveDialog, VersionLoadDialog
from neuropy.ui.app_state import _ALL_SEGS, ALL_PAIRS
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.ccg_transforms import ConnStrengthConfig
from neuropy.ui.pair_selection_panel import SelectionData
from neuropy.analyses.utils import _compact_json_str, JsonSavable
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState

_BAR_COLORS = ['#8FB3FF', '#FFB3B3', '#B3FFB3', '#FFD9B3', '#E0B3FF',
               '#B3F0FF', '#FFB3E6']


# ─────────────────────────── config↔widget binding ───────────────────────────

def read_cfg(host, cls, bind):
    return cls(**{f.name: bind[f.name][0](host) for f in fields(cls) if f.name in bind})


def write_cfg(host, cfg, bind):
    for f in fields(type(cfg)):
        if f.name in bind:
            bind[f.name][1](host, getattr(cfg, f.name))


# ─────────────────────────── data classes (Qt-free) ───────────────────────────
@dataclass
class RowConfig(JsonSavable):
    """Input/data config for one group row: which pairs to pull and as what metric."""
    name: str = ''
    color: str = ''
    sessions: list = field(default_factory=list)
    conn_types: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    resolution: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    data_type: str = ''


@dataclass
class StatsTestConfig(JsonSavable):
    """Stats-test config: how to test (independent of which data)."""
    test_type: str = 'Pairwise t-test'
    sides: str = 'Two-sided'
    direction: str = 'A > B'
    nonparametric: bool = False
    log_transform: bool = False
    remove_outliers: bool = False   # drop >3 SD pairs before testing

    @property
    def alternative(self) -> str:
        if self.sides == 'Two-sided':
            return 'two-sided'
        return 'greater' if self.direction.strip() == 'A > B' else 'less'


@dataclass
class _ViewConfig(JsonSavable):
    """Pure display toggles for the plot widget (not part of the test)."""
    violin: bool = False
    outliers: bool = True
    sig_brackets: bool = False
    wh_ratio: str = '3:1'


def _restore_group_pairs(groups):
    """Rebuild list[Key] in each group dict's 'pairs' after a JSON load."""
    def _key(p):
        if isinstance(p, dict) and '__keystr__' in p:
            k = Key(); k.__setstate__(p); return k
        if isinstance(p, str):   # legacy flat form
            return Key.from_str(p)
        return p
    for g in groups or []:
        if 'pairs' in g:
            g['pairs'] = [_key(p) for p in g['pairs']]


@dataclass
class StatsResult(JsonSavable):
    resolution: str = 'lowres'
    groups: list = field(default_factory=list)
    res: dict = field(default_factory=dict)
    plot_groups: list | None = None
    is_paired: bool = False
    is_one_sample: bool = False
    outliers: dict = field(default_factory=dict)      # group index → outlier value indices
    orig_groups: list | None = None                   # pre-removal groups the indices refer to

    @property
    def flagged_groups(self) -> list:
        """Groups the outlier indices index into (untrimmed when removal is on)."""
        return self.orig_groups if self.orig_groups is not None else self.groups

    def __setstate__(self, state: dict) -> None:
        JsonSavable.__setstate__(self, state)
        _restore_group_pairs(self.groups)
        _restore_group_pairs(self.plot_groups)


# ─────────────────────────── metrics (Qt-free) ───────────────────────────

NormMode = Literal["pct", "geom"]
Source = Literal[
    "conn_strength",
    "ref_firing_rate",
    "tgt_firing_rate",
    "baseline",
]

@dataclass(frozen=True, slots=True)
class Metric:
    source: Source
    enabled: bool = True
    highres: bool = False
    norm: NormMode | None = None


METRICS: dict[str, Metric] = {
    "Conn Strength":          Metric("conn_strength", highres=True),
    "CS norm (% change)":     Metric("conn_strength", highres=True, norm="pct"),
    "CS norm (geometric)":    Metric("conn_strength", highres=True, norm="geom"),
    "Ref Firing Rate":        Metric("ref_firing_rate"),
    "Tgt Firing Rate":        Metric("tgt_firing_rate"),
    "Baseline":               Metric("baseline"),
    "Peak Width":             Metric("conn_strength", enabled=False),
    "Peak Center":            Metric("conn_strength", enabled=False),
}


def cs_norm(a, b, mode: NormMode) -> np.ndarray:
    """Per-pair normalized change; ``a`` = group A, ``b`` = group B (same ref/tgt)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "pct":
            x = np.where(a != 0, (b - a) / np.abs(a), np.nan)
        else:
            d = np.sqrt(np.abs(a) * np.abs(b))
            x = np.where(d > 0, (b - a) / d, np.nan)
    return x[np.isfinite(x)]


# ─────────────────────────── backend (Qt-free) ───────────────────────────

class StatsPanelBackend:
    """Compute backend for the stats panel: pulls per-group values from cd, runs tests,
    and prepares plot data. Holds ``nav`` (state, not widgets); touches no Qt."""

    def __init__(self, nav: 'AppState'):
        self.nav = nav
        self.test_config: StatsTestConfig | None = None

    @staticmethod
    def maybe_log_transform(x, log: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not log:
            return x
        shifted = x - np.nanmin(x) + 1e-9 if np.any(x <= 0) else x
        return np.log(shifted)

    @staticmethod
    def outlier_indices(group: dict, log: bool) -> list[int]:
        """Indices of pairs >3 SD from the group mean, on the test scale (log if enabled)."""
        vals = StatsPanelBackend.maybe_log_transform(
            np.asarray(group.get('vals', []) or [], dtype=float), log)
        if vals.size < 4:
            return []
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        if s == 0:
            return []
        return [j for j in range(vals.size) if abs(vals[j] - m) > 3 * s]

    @staticmethod
    def is_matched(test_type: str, dtype: str) -> bool:
        """True when the test compares the same pair across groups (paired/RM/CS-norm)."""
        return (test_type in ("Pairwise t-test", "Repeated-measures ANOVA")
                or bool(METRICS[dtype].norm))

    @staticmethod
    def union_outlier_indices(groups: list, outliers: dict, by_ref_tgt: bool = False) -> dict:
        """Re-map per-group outliers so any pair flagged in one group is dropped from all.

        ``by_ref_tgt`` matches on (ref, tgt) alone — the identity CS-norm uses, where
        groups intentionally come from different sessions."""
        def _id(p):
            k = SelectionData.as_pair_key(p)
            return (k.ref, k.tgt) if by_ref_tgt else k
        bad = {_id(g['pairs'][j]) for i, g in enumerate(groups)
               for j in outliers.get(i, []) if j < len(g.get('pairs', []))}
        return {i: [j for j, p in enumerate(g.get('pairs', []) or []) if _id(p) in bad]
                for i, g in enumerate(groups)}

    @staticmethod
    def drop_indices(group: dict, idx: list[int]) -> dict:
        """Group with the given value indices removed (vals and pairs stay aligned)."""
        if not idx:
            return group
        drop = set(idx)
        keep = [j for j in range(len(group.get('vals', []) or [])) if j not in drop]
        pairs = group.get('pairs') or []
        return dict(group, vals=[group['vals'][j] for j in keep],
                    pairs=[pairs[j] for j in keep if j < len(pairs)])

    @staticmethod
    def apply_log_transform(groups: list, log: bool) -> list:
        return [dict(g, vals=list(StatsPanelBackend.maybe_log_transform(g.get('vals', []) or [], log)))
                for g in groups]

    @staticmethod
    def outlier_lines(label: str, sr: 'StatsResult', removed: bool) -> list[str]:
        """Outlier pairs per group, from the indices stored at run time."""
        verb = "removed" if removed else "flagged"
        lines = ["", f"Outliers (>3 SD from group mean, {verb}){label}:"]
        any_out = False
        for i, g in enumerate(sr.flagged_groups):
            pairs = g.get('pairs') or []
            items = [f"{(p := pairs[j]).ref}-{p.tgt} ({p.session})"
                     for j in sr.outliers.get(i, []) if j < len(pairs)]
            if items:
                any_out = True
                lines.append(f"  {g.get('name') or chr(65+i)}: {', '.join(items)}")
        if not any_out:
            lines.append("  (none)")
        return lines

    # -- top-level -----------------------------------------------------------
    def sessions_for(self, rows: list[RowConfig]) -> list[str]:
        """Deduped session-id list participating across all rows (row sessions ∪ all-concrete)."""
        return list(dict.fromkeys(
            s for r in rows for s in (r.sessions or self.nav.available_sessions())))

    def run(self, rows: list[RowConfig]) -> list[StatsResult]:
        dtype = rows[0].data_type
        all_sessions = self.sessions_for(rows)
        self.nav.ensure_groups_loaded_for(all_sessions)

        resolutions = list(dict.fromkeys(
            res for r in rows for res in (r.resolution or self.nav.available_resolutions())))
        results = []
        for resolution in resolutions:
            rows_at_res = [r for r in rows
                           if resolution in (r.resolution or self.nav.available_resolutions())]
            groups = [self.collect_group(r, resolution) for r in rows_at_res]
            cfg = self.test_config
            tt, log, nonparam = cfg.test_type, cfg.log_transform, cfg.nonparametric
            outliers = {i: self.outlier_indices(g, log) for i, g in enumerate(groups)}
            orig_groups = groups if cfg.remove_outliers else None
            if cfg.remove_outliers:
                # Matched tests compare the same pair across groups: drop the whole pair everywhere,
                # else an outlier in A would silently pull its partner out of B via the key intersection.
                if self.is_matched(tt, dtype):
                    outliers = self.union_outlier_indices(
                        groups, outliers, by_ref_tgt=bool(METRICS[dtype].norm))
                groups = [self.drop_indices(g, outliers[i]) for i, g in enumerate(groups)]
            if tt == "Repeated-measures ANOVA":
                res = self.run_rm_anova(groups, nonparam, log)
            elif len(groups) < 2:
                res = {'error': 'Need at least 2 groups.'}
            elif len(groups) > 2:
                res = self.run_anova(groups, nonparam, log)
            elif (norm := METRICS[dtype].norm):
                res = self.run_cs_norm(groups[0], groups[1], norm)
            else:
                res = self.run_test(groups[0]['vals'], groups[1]['vals'],
                                    groups[0]['pairs'], groups[1]['pairs'],
                                    tt, cfg.alternative, nonparam, log)
            sr = StatsResult(resolution=resolution, groups=groups, res=res,
                             outliers=outliers, orig_groups=orig_groups)
            self.finalize_plot_groups(sr, dtype)
            results.append(sr)
        return results

    # -- collection ----------------------------------------------------------
    def collect_group(self, cfg: RowConfig, resolution: str) -> dict:
        sessions  = cfg.sessions or self.nav.available_sessions()
        conn_type = cfg.conn_types[0] if cfg.conn_types else ''
        seg_names = cfg.segments or [_ALL_SEGS]
        grp_names = cfg.groups or [ALL_PAIRS]
        dtype     = cfg.data_type
        m = METRICS[dtype]
        print(m.source)
        merged: dict[Key, float] = {}
        used_sessions: set[str] = set()
        for grp_name in grp_names:
            for seg_name in seg_names:
                ei, ct = self.nav.cd.conf.parse_conn_type_label(conn_type)
                for sess in sessions:
                    ptr_key = Key(session=sess, excitability=ei, conn_type=ct)
                    if ptr_key not in self.nav.cd.ptr:
                        continue
                    if not m.enabled:
                        vals_map = {}
                    elif m.source == "conn_strength":
                        try:
                            vals_map = self.get_cs_values_for_sess(ptr_key, resolution, seg_name, grp_name)
                        except FileNotFoundError:
                            continue
                    elif m.source in ("ref_firing_rate", "tgt_firing_rate"):
                        try:
                            vals_map = self.get_fr_for_sess(ptr_key, grp_name,
                                                            0 if m.source == "ref_firing_rate" else 1)
                        except FileNotFoundError:
                            continue
                    elif m.source == "baseline":
                        try:
                            vals_map = self.get_baseline_for_sess(ptr_key, resolution, seg_name, grp_name)
                        except FileNotFoundError:
                            continue
                    if vals_map:
                        used_sessions.add(sess)
                    for k, v in vals_map.items():
                        merged.setdefault(k, v)

        sess_str = sessions[0] if len(sessions) == 1 else ','.join(sessions)
        return dict(name=cfg.name, session=sess_str, conn_type=conn_type,
                    segment=seg_names[0], group=grp_names[0], data_type=dtype,
                    pairs=list(merged), vals=list(merged.values()), color=cfg.color,
                    sessions_used=sorted(used_sessions))

    def _pair_value_dict(self, ptr_key, group_name, value_fn) -> dict[Key, float]:
        """{Key.pair(session,ref,tgt): value_fn(ref,tgt)} over a group's valid pairs."""
        out: dict[Key, float] = {}
        for ref, tgt in sorted(self.nav.pairs_for_group(group_name, ptr_key)):
            out[Key.pair(ptr_key.session, int(ref), int(tgt))] = float(value_fn(int(ref), int(tgt)))
        return out

    def get_cs_values_for_sess(self, ptr_key, resolution, seg_name, group_name) -> dict[Key, float]:
        nav = self.nav
        data_key = ptr_key.change(resolution=resolution, segment=seg_name)   # keeps excitability for CS sign
        data = nav.cd.ccg_for(data_key)
        if data is None or data.ccg is None:
            return {}
        cfg = ConnStrengthConfig(nav.baseline_method, nav.cs_metric,
                                nav.cd.conf.min_lag_bin, nav.cd.conf.max_lag_bin)
        grid = nav.cd.get_conn_strength_for(data_key, nav.active_norms, cfg)
        return self._pair_value_dict(ptr_key, group_name,
                                     lambda r, t: grid[r, t])

    def get_fr_for_sess(self, ptr_key, group_name, role) -> dict[Key, float]:
        neurons = self.nav.cd.nd.neurons_for(ptr_key)
        out: dict[Key, float] = {}
        for ref, tgt in sorted(self.nav.pairs_for_group(group_name, ptr_key)):
            idx = int((ref, tgt)[role])
            out.setdefault(Key.pair(ptr_key.session, idx, idx), float(neurons.firing_rate[idx]))
        return out

    def get_baseline_for_sess(self, ptr_key, resolution, seg_name, group_name) -> dict[Key, float]:
        nav = self.nav
        data_key = ptr_key.change(resolution=resolution, segment=seg_name)
        seg = nav.cd.segment_index(data_key, seg_name)   # resolve first: may lazy-load a segment
        data = nav.cd.ccg_for(data_key)
        if data is None or data.ccg_null is None:
            return {}
        # Normalized null (same active_norms as the CCG shown in the UI; BASELINE excluded there too).
        _, null = nav.cd.apply_ccg_transform_for(data_key, nav.active_norms)
        return self._pair_value_dict(ptr_key, group_name,
                                     lambda r, t: np.mean(null[seg, r, t, :]))

    # -- statistical tests ---------------------------------------------------
    def run_anova(self, groups: list, nonparam: bool, log: bool) -> dict:
        arrays = [self.maybe_log_transform(g.get('vals', []) or [], log) for g in groups]
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

    def run_rm_anova(self, groups: list, nonparam: bool, log: bool) -> dict:
        """Repeated-measures test: pairs common to every group are the within-subject units; conditions = groups."""
        pair_maps = []
        for g in groups:
            pairs = [SelectionData.as_pair_key(p) for p in (g.get('pairs') or [])]
            vals  = self.maybe_log_transform(g.get('vals', []) or [], log)
            pair_maps.append({p: v for p, v in zip(pairs, vals)})
        common = sorted(set.intersection(*(set(pm) for pm in pair_maps)),
                        key=lambda k: k.pair_sort_key())
        if len(common) < 2:
            return {'error': f"Need ≥2 pairs in all groups (found {len(common)})."}
        group_names = [g.get('name', f'G{i+1}') for i, g in enumerate(groups)]
        arrays = [np.array([pm[p] for p in common], dtype=float) for pm in pair_maps]
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

    def run_test(self, a_vals, b_vals, a_pairs, b_pairs,
                 test_type, alternative, nonparametric, log) -> dict:
        a = self.maybe_log_transform(a_vals, log)
        b = self.maybe_log_transform(b_vals, log)
        if a.size < 2 or b.size < 2:
            return {'error': f"Need ≥2 values per group (got {a.size}, {b.size})."}
        paired = (test_type == "Pairwise t-test")
        if paired:
            a_map = SelectionData.pairs_vals_map(a_pairs, a)
            b_map = SelectionData.pairs_vals_map(b_pairs, b)
            common = sorted(set(a_map) & set(b_map), key=lambda k: k.pair_sort_key())
            if len(common) < 2:
                return {'error': f"Only {len(common)} matched pairs — need ≥2."}
            a = np.array([a_map[p] for p in common])
            b = np.array([b_map[p] for p in common])
        if nonparametric:
            if paired:
                stat, p = _sp.wilcoxon(a, b, zero_method='wilcox', alternative=alternative)
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
                    sem_a=float(np.std(a, ddof=1)/np.sqrt(a.size)),
                    sem_b=float(np.std(b, ddof=1)/np.sqrt(b.size)),
                    paired=paired, alternative=alternative)

    def run_cs_norm(self, g_a: dict, g_b: dict, norm: NormMode) -> dict:
        # Match by (ref, tgt) only — sessions differ intentionally across groups
        def _rt(p):
            k = SelectionData.as_pair_key(p)
            return (k.ref, k.tgt)
        a_map = {_rt(p): v for p, v in zip(g_a.get('pairs', []), g_a.get('vals', []))}
        b_map = {_rt(p): v for p, v in zip(g_b.get('pairs', []), g_b.get('vals', []))}
        common = sorted(set(a_map) & set(b_map))
        if len(common) < 2:
            return {'error': f"Only {len(common)} matched pairs."}
        a = np.array([a_map[p] for p in common], dtype=float)
        b = np.array([b_map[p] for p in common], dtype=float)
        norm_arr = cs_norm(a, b, norm)
        if norm_arr.size < 2:
            return {'error': f"Only {norm_arr.size} finite normalized values."}
        stat, p = _sp.ttest_1samp(norm_arr, 0.0, alternative='two-sided')
        return dict(test='One-sample t-test', stat=float(stat), p_val=float(p),
                    n=int(norm_arr.size), mean=float(np.mean(norm_arr)),
                    sem=float(np.std(norm_arr, ddof=1) / np.sqrt(norm_arr.size)) if norm_arr.size > 1 else 0.0,
                    norm_vals=norm_arr.tolist(), norm_pairs=common)

    # -- plot-data prep + text ----------------------------------------------
    def finalize_plot_groups(self, sr: StatsResult, dtype: str):
        """Derive sr.plot_groups (and paired/one-sample flags) for one resolution."""
        cfg = self.test_config
        groups = sr.groups
        if cfg.test_type == "Repeated-measures ANOVA":
            common = (sr.res.get('common_pairs')
                      if sr.res and 'error' not in sr.res else None)
            sr.is_paired = True
            if not common:
                sr.plot_groups = groups
            else:
                sr.plot_groups = [
                    dict(g, vals=[pmap[p] for p in common if p in pmap], pairs=list(common))
                    for g in groups
                    for pmap in [SelectionData.pairs_vals_map(g.get('pairs'), g.get('vals'))]]
        elif (metric := METRICS[dtype]).norm:
            a_name = groups[0].get('name', 'A')
            b_name = groups[1].get('name', 'B') if len(groups) > 1 else 'B'
            if metric.norm == "pct":
                lbl = f"({a_name}−{b_name})/{a_name}"
            else:
                lbl = f"({a_name}−{b_name})/√(|{a_name}||{b_name}|)"
            sr.is_one_sample = True
            if not sr.res or 'error' in sr.res or 'norm_vals' not in sr.res:
                sr.plot_groups = None
            else:
                sr.plot_groups = [dict(name=lbl, vals=list(sr.res['norm_vals']),
                                       pairs=list(sr.res.get('norm_pairs', [])),
                                       data_type=dtype)]
        else:
            sr.plot_groups = self.apply_log_transform(groups, cfg.log_transform)

    def build_result_lines(self, results: list[StatsResult]) -> list[str]:
        cfg = self.test_config
        test_type = cfg.test_type if cfg else ''
        alt = cfg.alternative if cfg else 'two-sided'
        log = cfg.log_transform if cfg else False
        dtype = (results[0].groups[0].get('data_type')
                 if results and results[0].groups else '') or ''
        _res_lbl = {'lowres': 'lo-res', 'highres': 'hi-res'}
        multi = len(results) > 1

        def _fmt_res(r, groups_) -> list[str]:
            if not r: return ["(no result)"]
            if 'error' in r: return [f"Error: {r['error']}"]
            if r.get('paired') and 'n_a' in r:   # matched N differs from raw group N pre-intersection
                n_line = f"  N (matched pairs): {[r['n_a'], r['n_b']]}"
                m_line = f"  Means (matched): {[round(r['mean_a'],4), round(r['mean_b'],4)]}"
            elif 'n_subjects' in r:
                n_line = f"  N (matched pairs): {r['n_subjects']}"
                m_line = f"  Means: {[round(float(np.mean(g['vals'])),4) if g.get('vals') else 'n/a' for g in groups_]}"
            else:
                n_line = f"  N: {[len(g.get('vals') or []) for g in groups_]}"
                m_line = f"  Means: {[round(float(np.mean(g['vals'])),4) if g.get('vals') else 'n/a' for g in groups_]}"
            lines = [f"  Test: {r.get('test','?')}", n_line, m_line]
            if 'f_stat' in r:
                lines.append(f"  F = {r['f_stat']:.4f}")
            elif 'stat' in r:
                lines.append(f"  stat = {r['stat']:.4f}")
            if 'p_val' in r:
                p = r['p_val']
                lines.append(f"  p = {p:.4g}" + (" ***" if p<0.001 else (" **" if p<0.01 else (" *" if p<0.05 else ""))))
            if r.get('tukey') or r.get('posthoc'):
                for row in (r.get('tukey') or r.get('posthoc') or []):
                    lines.append(f"    {row.get('a','?')} vs {row.get('b','?')}: p_adj={row.get('p_adj',1.0):.4g}")
            return lines

        out = [f"── Stats: {test_type} ({'two-sided' if alt=='two-sided' else alt}) ──",
               f"   dtype={dtype}"]
        for sr in results:
            out.append(f"   {_res_lbl.get(sr.resolution, sr.resolution)}:")
            out += _fmt_res(sr.res, sr.groups)
        for sr in results:
            label = f" [{_res_lbl.get(sr.resolution, sr.resolution).capitalize()}]" if multi else ""
            out += self.outlier_lines(label, sr, self.test_config.remove_outliers)
        out.append("── Raw values ──")
        for sr in results:
            for g in sr.groups:
                vals = g.get('vals') or []
                tag = f"   {_res_lbl.get(sr.resolution, sr.resolution)} {g.get('name','?')}"
                out.append(f"{tag} sessions: {g.get('sessions_used') or []}")
                out.append(f"{tag} pairs: {[str(p) for p in (g.get('pairs') or [])]}")
                out.append(f"{tag} (n={len(vals)}): {[round(float(v), 6) for v in vals]}")
        return out


# ─────────────────────────── plot widget (frontend) ───────────────────────────

class StatsPlotWidget(QWidget):
    """Owns the matplotlib figure, the view toggles, and all plot rendering."""

    rerun_requested = Signal()   # a toggle that changes the test, not just the view

    _BIND = {
        'violin':       (lambda w: w._violin_check.isChecked(),   lambda w, v: w._violin_check.setChecked(v)),
        'outliers':     (lambda w: w._outliers_check.isChecked(), lambda w, v: w._outliers_check.setChecked(v)),
        'sig_brackets': (lambda w: w._sig_check.isChecked(),      lambda w, v: w._sig_check.setChecked(v)),
        'wh_ratio':     (lambda w: w._wh_input.text().strip(),    lambda w, v: w._wh_input.setText(v)),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[StatsResult] = []
        self.test_config: StatsTestConfig | None = None

        pc = QVBoxLayout(self)
        pc.setContentsMargins(0, 0, 0, 0)
        ctrl = QHBoxLayout()
        self._violin_check   = QCheckBox("Violin")
        self._outliers_check = QCheckBox("Show outliers")
        self._sig_check      = QCheckBox("Sig. brackets")
        self._outliers_check.setChecked(True)
        for chk in (self._violin_check, self._outliers_check, self._sig_check):
            chk.toggled.connect(self._replot)
            ctrl.addWidget(chk)
        self._rm_outliers_check = QCheckBox("Remove outliers")
        self._rm_outliers_check.setToolTip("Exclude >3 SD pairs and re-run the test")
        self._rm_outliers_check.toggled.connect(self.rerun_requested)
        ctrl.addWidget(self._rm_outliers_check)
        ctrl.addWidget(QLabel("W:H"))
        self._wh_input = QLineEdit("3:1")
        self._wh_input.setFixedWidth(45)
        self._wh_input.editingFinished.connect(self._replot)
        ctrl.addWidget(self._wh_input)
        ctrl.addStretch()
        pc.addLayout(ctrl)

        self._fig = Figure(dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _canvas_resize = self._canvas.resizeEvent
        self._canvas.resizeEvent = lambda ev: (self.refresh_plot(), _canvas_resize(ev))
        pc.addWidget(self._canvas, stretch=1)

    # -- config --------------------------------------------------------------
    def get_view_config(self) -> _ViewConfig:
        return read_cfg(self, _ViewConfig, self._BIND)

    def apply_view(self, vc: _ViewConfig):
        write_cfg(self, vc, self._BIND)

    # -- render --------------------------------------------------------------
    def render(self, results: list[StatsResult]):
        self._results = results
        self._update_plot()

    def _replot(self):
        if self._results:
            self._update_plot()

    def apply_theme(self):
        """Repaint the matplotlib canvas for the current light/dark palette."""
        self._fig.patch.set_facecolor('#2b2b2b' if qt_dark_mode() else 'white')
        if self._results:
            self._update_plot()
        else:
            self._canvas.draw_idle()

    def refresh_plot(self):
        self._fit_plot(redraw_only=True)

    def _target_ratio(self) -> float:
        """Parse W:H box → axes aspect ratio (default 3:1)."""
        try:
            wr, hr = (float(x) for x in self._wh_input.text().strip().split(':'))
            return max(wr, 0.1) / max(hr, 0.1)
        except Exception:
            return 3.0

    def _fit_plot(self, redraw_only: bool = False):
        """Figure fills the whole canvas; W:H shapes the centered axes rect via margins."""
        w = max(self._canvas.width(), 50)
        h = max(self._canvas.height(), 50)
        dpi = self._fig.get_dpi()
        self._fig.set_size_inches(w / dpi, h / dpi, forward=False)
        ratio, fig_ar = self._target_ratio(), w / h
        if fig_ar > ratio:
            fx, fy = ratio / fig_ar, 1.0
        else:
            fx, fy = 1.0, fig_ar / ratio
        pad_x, pad_y = (1 - fx) / 2, (1 - fy) / 2
        self._fig.subplots_adjust(left=0.10 + pad_x * 0.9, right=1 - (0.03 + pad_x * 0.9),
                                  bottom=0.12 + pad_y * 0.9, top=1 - (0.06 + pad_y * 0.9),
                                  wspace=0.25)
        if redraw_only and self._fig.axes:
            self._canvas.draw_idle()

    @staticmethod
    def _get_pairwise_pvals(result, groups) -> list:
        if not result or 'error' in result:
            return []
        for key in ('tukey', 'posthoc'):
            rows = result.get(key)
            if rows:
                return [{'a': r.get('a', ''), 'b': r.get('b', ''),
                         'p_adj': r.get('p_adj', 1.0)} for r in rows]
        return []

    def _update_plot(self):
        results = self._results
        fig = self._fig
        fig.clf()
        self._fit_plot()

        dark = qt_dark_mode()
        fig.patch.set_facecolor('#2b2b2b' if dark else 'white')
        if not results:
            self._canvas.draw()
            return

        log = self.test_config.log_transform if self.test_config else False
        use_violin    = self._violin_check.isChecked()
        show_outliers = self._outliers_check.isChecked()
        show_sig      = self._sig_check.isChecked()

        def _star(p):
            return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else None))

        def _mean_sem(x):
            if x.size == 0: return np.nan, 0.0
            return float(np.mean(x)), (float(np.std(x, ddof=1)/np.sqrt(x.size)) if x.size > 1 else 0.0)

        fg, bg  = ('#ffffff', '#1e1e1e') if dark else ('#000000', 'white')
        sp_col  = '#555555' if dark else '#cccccc'

        def _draw(ax, groups, title, paired=False, sig_pairs=[], one_sample=False):
            if not groups:
                ax.axis('off'); return
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=8)
            ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for sp in ax.spines.values(): sp.set_color(sp_col)
            n_g    = len(groups)
            names  = [g.get('name', chr(65+i)) or chr(65+i) for i, g in enumerate(groups)]
            arrays = [np.array(g.get('vals', []) or [], dtype=float) for g in groups]
            colors = [g.get('color') or _BAR_COLORS[i % len(_BAR_COLORS)] for i, g in enumerate(groups)]
            pairs_lists = [g.get('pairs', []) for g in groups]
            xs = np.arange(n_g, dtype=float)
            ax.set_title(title, fontsize=9, pad=2)
            ax.set_xticks(xs); ax.set_xticklabels(names)
            ax.tick_params(axis='x', labelsize=9); ax.tick_params(axis='y', labelsize=8)
            rng = np.random.default_rng(0)
            all_xpos = [np.full(a.size, xs[i]) + rng.normal(0, .05, a.size) for i, a in enumerate(arrays)]
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
                out_idx = set(StatsPanelBackend.outlier_indices(groups[i], log))
                if show_outliers:
                    ax.scatter(xpos, arr, s=14, color='#222', alpha=0.2, lw=0, zorder=2)
                    for j in out_idx:
                        if j < arr.size and j < len(pairs_lists[i]):
                            p = pairs_lists[i][j]
                            ax.annotate(f"{p.ref}-{p.tgt}", (xpos[j], float(arr[j])), fontsize=6,
                                        ha='center', va='bottom', color='#CC0000',
                                        xytext=(0, 3), textcoords='offset points')
                else:
                    keep = [k for k in range(arr.size) if k not in out_idx]
                    ax.scatter(xpos[keep], arr[keep], s=14, color='#222', alpha=0.2, lw=0, zorder=2)
            ax.grid(axis='y', alpha=0.25, lw=0.7)
            if one_sample:
                ax.axhline(0, color='#555', lw=0.8, ls='--', zorder=0)
            if sig_pairs:
                nm2x = {n: x for n, x in zip(names, xs)}
                valid = [(r['a'], r['b'], r['p_adj']) for r in sig_pairs
                         if r['a'] in nm2x and r['b'] in nm2x]
                if valid:
                    ylo, ytop = ax.get_ylim(); span = max(abs(ytop-ylo), 1e-6)
                    step = span * 0.13; tick = step * 0.3
                    valid.sort(key=lambda t: abs(nm2x[t[0]]-nm2x[t[1]]))
                    ybase = ytop + step * 0.2
                    sig_valid = [(a, b, p) for a, b, p in valid if _star(p) is not None]
                    for lv, (a_nm, b_nm, p_adj) in enumerate(sig_valid):
                        x0, x1 = nm2x[a_nm], nm2x[b_nm]; y = ybase + step*lv
                        ax.plot([x0, x0, x1, x1], [y-tick, y, y, y-tick], color='#333', lw=0.9, clip_on=False)
                        ax.text((x0+x1)/2, y+tick*0.2, _star(p_adj),
                                ha='center', va='bottom', fontsize=8, color='#333', clip_on=False)
                    if sig_valid:
                        ax.set_ylim(top=ybase+step*(len(sig_valid)+0.8))

        test_type = self.test_config.test_type if self.test_config else ''
        is_paired = (test_type == "Pairwise t-test"
                     or any(sr.is_paired for sr in results))
        n = len(results)
        _res_lbl = {'lowres': 'Lo-res', 'highres': 'Hi-res'}
        first_ax = None
        for i, sr in enumerate(results):
            groups = sr.plot_groups or []
            dtype = (groups[0].get('data_type') if groups else '') or ''
            title = dtype + (f" ({_res_lbl.get(sr.resolution, sr.resolution)})" if n > 1 else '')
            sig = self._get_pairwise_pvals(sr.res, groups) if show_sig else []
            ax = fig.add_subplot(1, n, i + 1, sharey=first_ax)
            if first_ax is None:
                first_ax = ax
            _draw(ax, groups, title, paired=is_paired, sig_pairs=sig,
                  one_sample=sr.is_one_sample)
        self._canvas.draw()


# ─────────────────────────── row widget (frontend) ───────────────────────────

class StatsRow(QWidget):
    """One group row: color, name, four multi-select pickers, data-type combo, delete.
    Owns its widgets and maps them to/from a RowConfig."""

    deleted = Signal(object)

    # each tuple: (picker key, button label, RowConfig field, AppState method for options)
    _PICKERS = (('sess', "Session", "sessions", 'available_sessions'),
                ('ct',   "ConnType", "types",   'available_conn_types'),
                ('seg',  "Segment",  "segments", 'all_available_segments'),
                ('res',  "Resolution", "resolutions", 'available_resolutions'),
                ('grp',  "Group",    "groups",  'available_groups'))

    # RowConfig field -> (getter reading the widget, setter writing the widget)
    _BIND = {
        'name':       (lambda r: r._name.text(),                lambda r, v: r._name.setText(v)),
        'color':      (lambda r: r._color,                      lambda r, v: r._set_color(v)),
        'sessions':   (lambda r: r._pickers['sess'].selected,   lambda r, v: r._pickers['sess'].set_selected(v)),
        'conn_types': (lambda r: r._pickers['ct'].selected,     lambda r, v: r._pickers['ct'].set_selected(v)),
        'segments':   (lambda r: r._pickers['seg'].selected,    lambda r, v: r._pickers['seg'].set_selected(v)),
        'resolution': (lambda r: r._pickers['res'].selected,    lambda r, v: r._pickers['res'].set_selected(v)),
        'groups':     (lambda r: r._pickers['grp'].selected,    lambda r, v: r._pickers['grp'].set_selected(v)),
        'data_type':  (lambda r: r._dtype.currentText(),        lambda r, v: r._dtype.setCurrentText(v)),
    }

    def __init__(self, nav: 'AppState', cfg: RowConfig | None = None, idx: int = 0, parent=None):
        super().__init__(parent)
        self.nav = nav
        rw = QHBoxLayout(self)
        rw.setContentsMargins(0, 0, 0, 0)
        rw.setSpacing(4)

        self._color = cfg.color if cfg and cfg.color else _BAR_COLORS[idx % len(_BAR_COLORS)]
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._style_color()
        self._color_btn.clicked.connect(self._pick_color)
        rw.addWidget(self._color_btn)

        self._name = QLineEdit(cfg.name if cfg and cfg.name
                               else (chr(65 + idx) if idx < 26 else f"G{idx+1}"))
        self._name.setFixedWidth(42)
        rw.addWidget(self._name)

        self._pickers = {}
        for rkey, label, plural, prov_attr in self._PICKERS:
            prov = getattr(nav, prov_attr)
            p = ListPickerButton(label, prov(), plural=plural, refresh_provider=prov)
            rw.addWidget(p)
            self._pickers[rkey] = p

        self._dtype = QComboBox()
        for name, metric in METRICS.items():
            self._dtype.addItem(name)
            if not metric.enabled:
                self._dtype.model().item(self._dtype.count() - 1).setEnabled(False)
        rw.addWidget(self._dtype)

        del_btn = QPushButton("x")
        del_btn.setFixedWidth(22)
        del_btn.clicked.connect(lambda: self.deleted.emit(self))
        rw.addWidget(del_btn)

        if cfg is not None:
            self.apply(cfg)
        else:
            self._seed_defaults()

    # -- config --------------------------------------------------------------
    @property
    def config(self) -> RowConfig:
        return read_cfg(self, RowConfig, self._BIND)

    def apply(self, cfg: RowConfig):
        write_cfg(self, cfg, self._BIND)

    def refresh(self):
        for rkey, _, _, prov_attr in self._PICKERS:
            self._pickers[rkey].set_items(getattr(self.nav, prov_attr)(), keep_selection=True)

    def _seed_defaults(self):
        """New row: default to current session + conn-type + first real group."""
        key_sess = str(self.nav.key.session)
        cur_ct   = self.nav.key.conn_type
        ct_lbl   = f"{cur_ct[0]}-{cur_ct[1]}" if cur_ct else None
        grp_opts = self.nav.available_groups()
        if key_sess in self._pickers['sess']._items:
            self._pickers['sess'].set_selected([key_sess])
        if ct_lbl and ct_lbl in self._pickers['ct']._items:
            self._pickers['ct'].set_selected([ct_lbl])
        if len(grp_opts) > 1:
            self._pickers['grp'].set_selected([grp_opts[1]])

    # -- color ---------------------------------------------------------------
    def _style_color(self):
        self._color_btn.setStyleSheet(
            f"background:{self._color}; border:1px solid #888; border-radius:2px;")

    def _set_color(self, v):
        if v:
            self._color = v
            self._style_color()

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._set_color(c.name())


# ─────────────────────────── panel (frontend orchestration) ───────────────────────────

class StatsTestPanel(QWidget):
    """Persistent floating stats-test panel: top controls, group rows, result plot."""

    TEST_CONFIG_WIDGETS = {
        'test_type':     (lambda p: p._test_type.currentText(), lambda p, v: p._test_type.setCurrentText(v)),
        'sides':         (lambda p: p._sides.currentText(),     lambda p, v: p._sides.setCurrentText(v)),
        'direction':     (lambda p: p._dir_btn.text().strip(),  lambda p, v: p._dir_btn.setText(v)),
        'nonparametric': (lambda p: p._nonparam.isChecked(),    lambda p, v: p._nonparam.setChecked(v)),
        'log_transform': (lambda p: p._log.isChecked(),         lambda p, v: p._log.setChecked(v)),
        'remove_outliers': (lambda p: p._plot._rm_outliers_check.isChecked(),
                            lambda p, v: p._plot._rm_outliers_check.setChecked(v)),
    }

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.nav = nav
        self.backend = StatsPanelBackend(nav)
        self._rows: list[StatsRow] = []
        self.test_config: StatsTestConfig | None = None
        self.results: list[StatsResult] = []

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
        self._test_type = make_combo(
            ["Independent t-test", "Pairwise t-test",
             "One-way ANOVA + Tukey", "Repeated-measures ANOVA"],
            200, current="Pairwise t-test")
        top.addWidget(self._test_type)
        top.addWidget(QLabel("Sides:"))
        self._sides = make_combo(["Two-sided", "One-sided"], 90)
        top.addWidget(self._sides)
        self._dir_btn = make_button("A > B", self._toggle_1sided_direction, 60)
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
        add_btn.clicked.connect(lambda: self._add_row())
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
        self._plot = StatsPlotWidget()
        self._plot.rerun_requested.connect(lambda: self.results and self._run())
        self._splitter.addWidget(self._plot)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.splitterMoved.connect(lambda *_: self._plot.refresh_plot())
        res_root.addWidget(self._splitter)
        root.addWidget(res_frame, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addWidget(make_button("Run", self._run))
        self._export_btn = make_button("Save…", self._export)
        self._export_btn.setEnabled(False)
        btn_row.addWidget(self._export_btn)
        btn_row.addWidget(make_button("Load…", self._load_result))
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _connect_nav(self):
        self.nav.key_changed.connect(lambda _: self._refresh_rows())
        self.nav.custom_segs_changed.connect(self._refresh_rows)
        self.nav.groups.changed.connect(self._refresh_rows)
        self.nav.selection_changed.connect(self._refresh_rows)

    # -- rows ----------------------------------------------------------------
    def _add_row(self, cfg: RowConfig | None = None):
        row = StatsRow(self.nav, cfg, idx=len(self._rows))
        row.deleted.connect(self._del_row)
        self._rows.append(row)
        self._rows_area.addWidget(row)

    def _del_row(self, row: StatsRow):
        if row in self._rows:
            self._rows.remove(row)
        row.deleteLater()

    def _refresh_rows(self):
        for r in self._rows:
            r.refresh()

    def _toggle_1sided_direction(self):
        cur = self._dir_btn.text().strip()
        self._dir_btn.setText("A < B" if cur == "A > B" else "A > B")

    # -- config --------------------------------------------------------------
    def get_test_config(self) -> StatsTestConfig:
        return read_cfg(self, StatsTestConfig, self.TEST_CONFIG_WIDGETS)

    def apply_test(self, cfg: StatsTestConfig):
        write_cfg(self, cfg, self.TEST_CONFIG_WIDGETS)

    # -- run -----------------------------------------------------------------
    def _validate(self) -> str | None:
        if len(self._rows) < 2:
            return "Need at least 2 groups to compare."
        dtype = self._rows[0].config.data_type
        metric = METRICS[dtype]
        if not metric.enabled:
            return f"Data type '{dtype}' is not yet implemented."
        if metric.norm and len(self._rows) != 2:
            return f"'{dtype}' requires exactly 2 groups."
        if self._test_type.currentText() == "Pairwise t-test":
            sess_sets = [tuple(r.config.sessions or self.nav.available_sessions())
                         for r in self._rows]
            if len(set(sess_sets)) > 1:
                return "Pairwise t-test requires same sessions in all groups."
        return None

    def _run(self):
        if (err := self._validate()):
            self._show_result(err)
            return
        cfg = self.get_test_config()
        self.test_config = cfg
        self.backend.test_config = cfg
        self._plot.test_config = cfg
        rows_configs = [r.config for r in self._rows]
        # Gap-fill: for every resolution the test will run, ensure each participating session has
        # it (generate/load the missing ones) so no session is silently dropped from a resolution.
        all_sessions = self.backend.sessions_for(rows_configs)
        nd_keys = [k for k in self.nav.real_nd_keys() if str(k.session) in all_sessions]
        for res in self.nav.available_resolutions():
            self.nav.root._ensure_sessions_loaded(nd_keys, res)
        self.results = self.backend.run(rows_configs)
        self._show_result('\n'.join(self.backend.build_result_lines(self.results)))
        self._plot.render(self.results)
        self._export_btn.setEnabled(True)

    def apply_theme(self):
        self._plot.apply_theme()

    def _show_result(self, text: str):
        self._result_text.setPlainText(text)

    # -- save / load ---------------------------------------------------------
    def _save_bundle(self) -> dict:
        return dict(
            test=self.get_test_config().serialize(),
            view=self._plot.get_view_config().serialize(),
            splitter_sizes=list(self._splitter.sizes()),
            rows=[r.config.serialize() for r in self._rows])

    def _apply_bundle(self, d: dict):
        for r in list(self._rows):
            self._del_row(r)
        for rd in d.get('rows') or []:
            rc = RowConfig(); rc.__setstate__(rd)
            self._add_row(rc)
        while len(self._rows) < 2:
            self._add_row()
        if d.get('test'):
            tc = StatsTestConfig(); tc.__setstate__(d['test'])
            self.apply_test(tc)
        if d.get('view'):
            vc = _ViewConfig(); vc.__setstate__(d['view'])
            self._plot.apply_view(vc)
        sizes = d.get('splitter_sizes')
        if sizes and len(sizes) == 2:
            self._splitter.setSizes(sizes)

    def _stats_save_dir(self) -> str:
        save_dir = pathlib.Path(self.nav.cd.stats_results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir)

    def _saved_results(self) -> list:
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
        if not self.results:
            return
        default = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        def _do_save(name):
            path = str(pathlib.Path(self._stats_save_dir()) / f"{name}.json")
            data = dict(self._save_bundle(),
                        results=[r.serialize() for r in self.results],
                        saved_at=datetime.datetime.now().isoformat())
            atomic_write_json(path, text=_compact_json_str(data))
        VersionSaveDialog.show(self, "Save Stats Result", default, on_save=_do_save)

    def _load_result(self):
        saved = self._saved_results()
        def _do_load(path):
            try:
                data = json.loads(pathlib.Path(path).read_text())
                self._apply_bundle(data)
                self.test_config = self.get_test_config()
                self.backend.test_config = self.test_config
                self._plot.test_config = self.test_config
                self.results = []
                for rd in data.get('results') or []:
                    sr = StatsResult(); sr.__setstate__(rd)
                    self.results.append(sr)
                self._show_result('\n'.join(self.backend.build_result_lines(self.results)))
                self._plot.render(self.results)
                self._export_btn.setEnabled(True)
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", str(exc))
        VersionLoadDialog.show(self, "Load Stats Result", saved, on_load=_do_load,
                               empty_msg="No saved stats results found.")
