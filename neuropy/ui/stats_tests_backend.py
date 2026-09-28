"""Stats-test compute layer: metrics, configs and StatsTestBackend.

Takes data, never widgets; the frontend calls in and reads results back.
"""
from __future__ import annotations

import ast
import datetime
import json
import pathlib
import re
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as _sp
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from neuropy.ui.app_state import _ALL_SEGS, ALL_PAIRS, DisplayConfig
from neuropy.ui.pair_selection_panel import SelectionData
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.ccg_transforms import ConnStrengthConfig, NormalizeBy
from neuropy.analyses.utils import _compact_json_str, JsonSavable
from neuropy.utils.data_storage_util import atomic_write_json

_BAR_COLORS = ['#8FB3FF', '#FFB3B3', '#B3FFB3', '#FFD9B3', '#E0B3FF',
               '#B3F0FF', '#FFB3E6']


# ─────────────────────────── data classes ───────────────────────────
# picker key -> (RowConfig field, button label, plural noun, backend options method).
# The only list of pickers: widgets, binding and follow resolution all read it.
PICKERS = {
    'sess': ('sessions',   "Session",    "sessions",    'available_sessions'),
    'ct':   ('conn_types', "ConnType",   "types",       'available_conn_types'),
    'seg':  ('segments',   "Segment",    "segments",    'available_segments'),
    'res':  ('resolution', "Resolution", "resolutions", 'available_resolutions'),
    'grp':  ('groups',     "Group",      "groups",      'available_groups'),
    'data': ('data_type',  "Data",       "metrics",     'available_data_types'),
}
PICKER_FIELD = {k: v[0] for k, v in PICKERS.items()}
SINGLE_PICKERS = {'data'}   # one metric per row: the value is a str, not a list


@dataclass
class RowConfig(JsonSavable):
    """Input/data config for one group row: which pairs to pull and as what metric."""
    id: int = 0
    name: str = ''
    color: str = ''
    sessions: list = field(default_factory=list)
    conn_types: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    resolution: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    data_type: str = ''
    follow: int | None = None                      # row id this one mirrors
    follow_pickers: list = field(default_factory=list)   # which pickers follow; empty = all
    follow_exclude: bool = False                   # invert: these are the ones that do NOT follow


TEST_TYPES = ("Independent t-test", "Pairwise t-test",
              "One-way ANOVA + Tukey", "Repeated-measures ANOVA")
# test -> how many groups it accepts; None = any number above the minimum
TEST_N_GROUPS = {"Independent t-test": (2, 2), "Pairwise t-test": (2, 2),
                 "One-way ANOVA + Tukey": (3, None), "Repeated-measures ANOVA": (3, None)}


@dataclass
class StatsTestConfig(JsonSavable):
    """Stats-test config: how to test (independent of which data). None test_type = infer."""
    test_type: str | None = 'Pairwise t-test'
    sides: str = 'Two-sided'
    direction: str = 'A > B'
    nonparametric: bool = False
    log_transform: bool = False
    remove_outliers: bool = False   # drop >3 SD pairs before testing
    post_hoc: bool = True

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


def _restore_key(p):
    if isinstance(p, dict) and '__keystr__' in p:
        k = Key(); k.__setstate__(p); return k
    if isinstance(p, str):   # legacy flat form
        return Key.from_str(p)
    return p


def _restore_group_pairs(groups):
    """Rebuild list[Key] in each group dict's 'pairs' after a JSON load."""
    for g in groups or []:
        if 'pairs' in g:
            g['pairs'] = [_restore_key(p) for p in g['pairs']]
        g['seg_configs'] = [_seg_config(c) for c in (g.get('seg_configs') or [])]


_SEG_CONFIG_RE = re.compile(
    r"^(?P<session>\S+) (?P<segment>.+?): "
    r"(?:t0=(?P<t0>\S+) t1=(?P<t1>\S+) dur=(?P<dur>\S+) filter=(?P<filter>.*?)"
    r"|whole session) \((?P<n_pairs>\d+) pairs\)$")


def _seg_config(c):
    """Seg configs were once saved preformatted; parse those back to the dict form."""
    if isinstance(c, dict):
        return c
    m = _SEG_CONFIG_RE.match(c)
    out = dict(session=m['session'], segment=m['segment'], n_pairs=int(m['n_pairs']))
    if m['t0'] is not None:
        out.update(t0=float(m['t0']), t1=float(m['t1']), dur=float(m['dur']),
                   filter=ast.literal_eval(m['filter']))
    return out


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
    display: DisplayConfig | None = None              # settings this result was computed under

    @property
    def flagged_groups(self) -> list:
        """Groups the outlier indices index into (untrimmed when removal is on)."""
        return self.orig_groups if self.orig_groups is not None else self.groups

    def __setstate__(self, state: dict) -> None:
        JsonSavable.__setstate__(self, state)
        _restore_group_pairs(self.groups)
        _restore_group_pairs(self.plot_groups)
        _restore_group_pairs(self.orig_groups)
        if 'common_pairs' in self.res:
            self.res['common_pairs'] = [_restore_key(p) for p in self.res['common_pairs']]
        # int keys survive as pair lists; an empty dict stays a dict
        self.outliers = {int(k): v for k, v in (self.outliers.items()
                         if isinstance(self.outliers, dict) else self.outliers)}
        self.display = DisplayConfig()
        self.display.__setstate__(state['display'])


# ─────────────────────────── metrics ───────────────────────────

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
    off_mode: str = 'flag'   # 'binarize' reports the on/off label rather than the CS value


METRICS: dict[str, Metric] = {
    "Conn Strength":          Metric("conn_strength", highres=True),
    "CS binarized (on/off)":  Metric("conn_strength", highres=True, off_mode='binarize'),
    "CS norm (% change)":     Metric("conn_strength", highres=True, norm="pct"),
    "CS norm (geometric)":    Metric("conn_strength", highres=True, norm="geom"),
    "Ref Firing Rate":        Metric("ref_firing_rate"),
    "Tgt Firing Rate":        Metric("tgt_firing_rate"),
    "Baseline":               Metric("baseline"),
    "Peak Width":             Metric("conn_strength", enabled=False),
    "Peak Center":            Metric("conn_strength", enabled=False),
}
DISABLED_METRICS = tuple(n for n, m in METRICS.items() if not m.enabled)


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


# ─────────────────────────── backend ───────────────────────────

class StatsTestBackend:
    """One stats test: owns its rows, configs and results. Takes data, never widgets."""

    def __init__(self, cd, sd, *, display: 'DisplayConfig' = None):
        self.cd = cd
        self.sd = sd
        self.display = display if display is not None else DisplayConfig()
        self.rows: list[RowConfig] = []
        self._next_id = 1
        self.test_config = StatsTestConfig()
        self.view_config = _ViewConfig()
        self.results: list[StatsResult] = []
        self.text: str = ''
        self.missing_segments: dict[str, set] = {}   # segment → sessions it was never computed for

    @property
    def norms(self) -> set:
        """The NormalizeBy set the display config names."""
        return {NormalizeBy[n.rsplit('.', 1)[-1]] for n in self.display.norms}

    # -- options (what a picker may offer) -----------------------------------
    def available_sessions(self) -> list[str]:
        seen, out = set(), []
        for k in self.cd.nd.session_keys:
            s = str(k.session)
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    def available_resolutions(self) -> list[str]:
        return self.cd.available_resolutions()

    def available_conn_types(self) -> list[str]:
        return self.cd.conf.conn_type_labels

    def available_groups(self) -> list[str]:
        return [ALL_PAIRS] + self.sd.groups.pickable_groups()

    def available_segments(self) -> list[str]:
        return sorted(self.cd.available_segments())

    @staticmethod
    def available_data_types() -> list[str]:
        return list(METRICS)

    def pairs_for_group(self, group_name: str, ptr_key) -> set:
        """Valid (significant) pairs of a group for a ptr key; ALL_PAIRS = every valid pair."""
        valid = self.cd.ptr[ptr_key].pair_set
        if group_name == ALL_PAIRS:
            return valid
        return self.sd.groups.pairs_in_group(group_name, ptr_key.session) & valid

    # -- rows ----------------------------------------------------------------
    def add_row(self, name: str = '', *, color: str = '', sessions=(), conn_types=(),
                segments=(), resolution=(), groups=(), data_type: str = '',
                follow: int | None = None, pickers=(), exclude: bool = False) -> int:
        """Append a row; returns its id. *pickers* lists what `follow` applies to
        (or, with *exclude*, what it does not). *groups* is the pair groups it draws from."""
        cfg = RowConfig(id=self._next_id, name=name, color=color,
                        sessions=list(sessions), conn_types=list(conn_types),
                        segments=list(segments), resolution=list(resolution),
                        groups=list(groups), data_type=data_type,
                        follow=follow, follow_pickers=list(pickers), follow_exclude=exclude)
        self._next_id += 1
        self.rows.append(cfg)
        return cfg.id

    def remove_row(self, row_id: int) -> None:
        self.rows = [r for r in self.rows if r.id != row_id]
        for r in self.rows:      # a row following the removed one becomes independent
            if r.follow == row_id:
                r.follow = None

    def update_row(self, row_id: int, **kw) -> None:
        row = self.row(row_id)
        for k, v in kw.items():
            setattr(row, k, v)

    def row(self, row_id: int) -> RowConfig:
        return next(r for r in self.rows if r.id == row_id)

    def followed_pickers(self, row: RowConfig) -> list[str]:
        """Picker keys this row mirrors from its leader."""
        if row.follow is None:
            return []
        keys = list(PICKER_FIELD)
        chosen = set(row.follow_pickers) if row.follow_pickers else set(keys)
        return [k for k in keys if (k in chosen) != row.follow_exclude]

    def resolve_follows(self) -> None:
        """Copy mirrored picker values from each leader; a no-op for live Qt pickers."""
        by_id = {r.id: r for r in self.rows}
        for row in self.rows:
            leader = by_id.get(row.follow)
            if leader is None or leader is row:
                continue
            for key in self.followed_pickers(row):
                setattr(row, PICKER_FIELD[key], list(getattr(leader, PICKER_FIELD[key])))

    @staticmethod
    def maybe_log_transform(x, log: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not log:
            return x
        shifted = x - np.nanmin(x) + 1e-9 if np.any(x <= 0) else x
        return np.log(shifted)

    @staticmethod
    def outlier_indices(row: dict, log: bool) -> list[int]:
        """Indices of pairs >3 SD from the row mean, on the test scale (log if enabled)."""
        vals = StatsTestBackend.maybe_log_transform(
            np.asarray(row.get('vals', []), dtype=float), log)
        if vals.size < 4:
            return []
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        if s == 0:
            return []
        return [j for j in range(vals.size) if abs(vals[j] - m) > 3 * s]

    @staticmethod
    def infer_test_type(n_groups: int, dtype: str) -> str:
        """The test the data implies, for a None test_type: 2 groups compare, 3+ go to ANOVA."""
        if METRICS[dtype].norm:
            return "Pairwise t-test"        # a norm metric is per-pair by construction
        return "One-way ANOVA + Tukey" if n_groups > 2 else "Pairwise t-test"

    @staticmethod
    def check_test_type(test_type: str, n_groups: int) -> str | None:
        """Why *test_type* cannot run on this many groups, or None."""
        if test_type not in TEST_N_GROUPS:
            return (f"Unknown test type {test_type!r}; expected one of "
                    + ", ".join(TEST_TYPES) + ".")
        lo, hi = TEST_N_GROUPS[test_type]
        if n_groups < lo or (hi is not None and n_groups > hi):
            want = f"exactly {lo}" if lo == hi else f"at least {lo}"
            return f"'{test_type}' needs {want} groups; {n_groups} selected."
        return None

    @staticmethod
    def is_matched(test_type: str, dtype: str) -> bool:
        """True when the test compares the same pair across groups (paired/RM/CS-norm)."""
        return (test_type in ("Pairwise t-test", "Repeated-measures ANOVA")
                or bool(METRICS[dtype].norm))

    @staticmethod
    def union_outlier_indices(rows: list, outliers: dict, by_ref_tgt: bool = False) -> dict:
        """Re-map per-row outliers so any pair flagged in one row is dropped from all.

        ``by_ref_tgt`` matches on (ref, tgt) alone — the identity CS-norm uses, where
        rows intentionally come from different sessions."""
        def _id(p):
            k = SelectionData.as_pair_key(p)
            return (k.ref, k.tgt) if by_ref_tgt else k
        bad = {_id(g.get('pairs', [])[j]) for i, g in enumerate(rows)
               for j in outliers.get(i, []) if j < len(g.get('pairs', []))}
        return {i: [j for j, p in enumerate(g.get('pairs', [])) if _id(p) in bad]
                for i, g in enumerate(rows)}

    @staticmethod
    def drop_indices(row: dict, idx: list[int]) -> dict:
        """Row with the given value indices removed (vals and pairs stay aligned)."""
        if not idx:
            return row
        drop = set(idx)
        keep = [j for j in range(len(row.get('vals', []))) if j not in drop]
        pairs = row.get('pairs') or []
        return dict(row, vals=[row['vals'][j] for j in keep],
                    pairs=[pairs[j] for j in keep if j < len(pairs)])

    @staticmethod
    def apply_log_transform(rows: list, log: bool) -> list:
        return [dict(g, vals=list(StatsTestBackend.maybe_log_transform(g.get('vals', []), log)))
                for g in rows]

    # -- top-level -----------------------------------------------------------
    def sessions_for(self) -> list[str]:
        """Deduped session-id list participating across all rows (row sessions ∪ all-concrete)."""
        return list(dict.fromkeys(
            s for r in self.rows
            for s in (r.sessions or self.available_sessions())))

    def ensure_loaded(self) -> None:
        """Force every participating session to hold CCGs at every resolution the run needs."""
        sessions = self.sessions_for()
        nd_keys = [k for k in self.cd.nd.session_keys if str(k.session) in sessions]
        pending = [(k, res) for res in self.available_resolutions() for k in nd_keys
                   if self.cd._ccg.get(k.change(resolution=res).cd()) is None]
        if pending:
            print(f"[stats] {len(pending)} session/resolution pair(s) missing — "
                  f"generating CCGs first, this will take longer.", flush=True)
        for k, res in pending:
            self.cd.get_ccg(k.change(resolution=res))

    def run(self) -> list[StatsResult]:
        rows = self.rows
        self.resolve_follows()
        dtype = rows[0].data_type
        self.missing_segments = {}
        all_sessions = self.sessions_for()
        self.sd.ensure_groups_loaded_for(all_sessions)
        self.ensure_loaded()

        resolutions = list(dict.fromkeys(
            res for r in rows for res in (r.resolution or self.available_resolutions())))
        results = []
        for resolution in resolutions:
            rows_at_res = [r for r in rows
                           if resolution in (r.resolution or self.available_resolutions())]
            collected = [self.collect_row(r, resolution) for r in rows_at_res]
            cfg = self.test_config
            log, nonparam = cfg.log_transform, cfg.nonparametric
            tt = cfg.test_type or self.infer_test_type(len(collected), dtype)
            outliers = {i: self.outlier_indices(g, log) for i, g in enumerate(collected)}
            orig = collected if cfg.remove_outliers else None
            if cfg.remove_outliers:
                # Matched tests compare the same pair across rows: drop the whole pair everywhere,
                # else an outlier in A would silently pull its partner out of B via the key intersection.
                if self.is_matched(tt, dtype):
                    outliers = self.union_outlier_indices(
                        collected, outliers, by_ref_tgt=bool(METRICS[dtype].norm))
                collected = [self.drop_indices(g, outliers[i]) for i, g in enumerate(collected)]
            if (bad := self.check_test_type(tt, len(collected))):
                res = {'error': bad}
            elif tt == "Repeated-measures ANOVA":
                res = self.run_rm_anova(collected, nonparam, log, cfg.post_hoc)
            elif tt == "One-way ANOVA + Tukey":
                res = self.run_anova(collected, nonparam, log, cfg.post_hoc)
            elif (norm := METRICS[dtype].norm):
                res = self.run_cs_norm(collected[0], collected[1], norm)
            else:
                res = self.run_test(collected[0]['vals'], collected[1]['vals'],
                                    collected[0]['pairs'], collected[1]['pairs'],
                                    tt, cfg.alternative, nonparam, log)
            sr = StatsResult(resolution=resolution, groups=collected, res=res,
                             outliers=outliers, orig_groups=orig,
                             display=self.display)
            self.finalize_plot_groups(sr, dtype)
            results.append(sr)
        self.results = results
        self.text = self.result_text(results)
        return results

    # -- collection ----------------------------------------------------------
    def collect_row(self, cfg: RowConfig, resolution: str) -> dict:
        sessions  = cfg.sessions or self.available_sessions()
        conn_types = cfg.conn_types or ['']
        seg_names = cfg.segments or [_ALL_SEGS]
        grp_names = cfg.groups or [ALL_PAIRS]
        dtype     = cfg.data_type
        m = METRICS[dtype]
        merged: dict[Key, float] = {}
        used_sessions: set[str] = set()
        seg_configs: list[dict] = []
        for grp_name, seg_name, conn_type in product(grp_names, seg_names, conn_types):
            ei, ct = self.cd.conf.parse_conn_type_label(conn_type)
            for sess in sessions:
                ptr_key = Key(session=sess, excitability=ei, conn_type=ct)
                if ptr_key not in self.cd.ptr:
                    continue
                if not m.enabled:
                    vals_map = {}
                elif m.source == "conn_strength":
                    try:
                        vals_map = self.get_cs_values_for_sess(ptr_key, resolution, seg_name,
                                                               grp_name, m.off_mode)
                    except FileNotFoundError:
                        self.missing_segments.setdefault(seg_name, set()).add(sess)
                        continue
                elif m.source in ("ref_firing_rate", "tgt_firing_rate"):
                    try:
                        vals_map = self.get_fr_for_sess(
                            ptr_key, resolution, seg_name, grp_name,
                            0 if m.source == "ref_firing_rate" else 1)
                    except FileNotFoundError:
                        self.missing_segments.setdefault(seg_name, set()).add(sess)
                        continue
                elif m.source == "baseline":
                    try:
                        vals_map = self.get_baseline_for_sess(ptr_key, resolution, seg_name, grp_name)
                    except FileNotFoundError:
                        self.missing_segments.setdefault(seg_name, set()).add(sess)
                        continue
                data = self.cd._ccg.get(
                    ptr_key.change(resolution=resolution, segment=seg_name).cd())
                src = data.sources.get(seg_name) if data is not None else None
                seg_configs.append(dict(
                    session=sess, segment=seg_name, n_pairs=len(vals_map),
                    **(dict(t0=src.t0, t1=src.t1, dur=src.active_duration,
                            filter=src.filter_state) if src is not None else {})))
                if vals_map:
                    used_sessions.add(sess)
                for k, v in vals_map.items():
                    merged.setdefault(k, v)

        sess_str = sessions[0] if len(sessions) == 1 else ','.join(sessions)
        return dict(name=cfg.name, session=sess_str, conn_type=','.join(conn_types),
                    segment=seg_names[0], pair_groups=','.join(grp_names), data_type=dtype,
                    pairs=list(merged), vals=list(merged.values()), color=cfg.color,
                    sessions_used=sorted(used_sessions), seg_configs=seg_configs)

    def _pair_value_dict(self, ptr_key, group_name, value_fn) -> dict[Key, float]:
        """{Key.pair(session,ref,tgt): value_fn(ref,tgt)} over a group's valid pairs."""
        out: dict[Key, float] = {}
        for ref, tgt in sorted(self.pairs_for_group(group_name, ptr_key)):
            out[Key.pair(ptr_key.session, int(ref), int(tgt))] = float(value_fn(int(ref), int(tgt)))
        return out

    def _seg_data(self, ptr_key, resolution, seg_name, array: str):
        """(data_key, data) for one segment; data is None when *array* is not there to read."""
        data_key = ptr_key.change(resolution=resolution, segment=seg_name)   # keeps excitability for CS sign
        data = self.cd.ccg_for(data_key)
        return data_key, (data if data is not None and getattr(data, array) is not None else None)

    def get_cs_values_for_sess(self, ptr_key, resolution, seg_name, group_name,
                               off_mode: str = 'flag') -> dict[Key, float]:
        data_key, data = self._seg_data(ptr_key, resolution, seg_name, 'ccg')
        if data is None:
            return {}
        c = self.cd.conf.at(resolution)   # the bins must match the array being measured
        if off_mode == 'flag' and self.display.cs_offzero:
            off_mode = 'zero'
        cfg = ConnStrengthConfig(self.display.baseline_method, self.display.cs_metric,
                                 c.min_lag_bin, c.max_lag_bin, off_mode)
        grid = self.cd.get_conn_strength_for(
            data_key, self.norms, cfg,
            overrides=self.sd.seg_overrides(str(ptr_key.session), seg_name))
        return self._pair_value_dict(ptr_key, group_name,
                                     lambda r, t: grid[r, t])

    def get_fr_for_sess(self, ptr_key, resolution, seg_name, group_name, role) -> dict[Key, float]:
        """A neuron counts once, so its key is the self-pair rather than every pair it joins."""
        _, data = self._seg_data(ptr_key, resolution, seg_name, 'ccg')
        src = data.sources.get(seg_name) if data is not None else None
        rates = src.firing_rates if src is not None else None
        if rates is None:   # the whole-session segment declares none of its own
            rates = self.cd.nd.neurons_for(ptr_key).firing_rate
        out: dict[Key, float] = {}
        for ref, tgt in sorted(self.pairs_for_group(group_name, ptr_key)):
            idx = int((ref, tgt)[role])
            out.setdefault(Key.pair(ptr_key.session, idx, idx), float(rates[idx]))
        return out

    def get_baseline_for_sess(self, ptr_key, resolution, seg_name, group_name) -> dict[Key, float]:
        data_key, data = self._seg_data(ptr_key, resolution, seg_name, 'ccg_null')
        if data is None:
            return {}
        seg = self.cd.segment_index(data_key, seg_name)
        # Normalized null (same active_norms as the CCG shown in the UI; BASELINE excluded there too).
        _, null = self.cd.apply_ccg_transform_for(data_key, self.norms)
        return self._pair_value_dict(ptr_key, group_name,
                                     lambda r, t: np.mean(null[seg, r, t, :]))

    # -- statistical tests ---------------------------------------------------
    def run_anova(self, test_rows: list, nonparam: bool, log: bool, post_hoc: bool = True) -> dict:
        arrays = [self.maybe_log_transform(g.get('vals', []), log) for g in test_rows]
        if any(a.size < 2 for a in arrays):
            return {'error': "Need ≥2 values per group (got "
                             + ", ".join(str(a.size) for a in arrays) + ")."}
        if nonparam:
            stat, p = _sp.kruskal(*[a for a in arrays if a.size])
            return dict(test='Kruskal-Wallis', stat=float(stat), p_val=float(p))
        stat, p = _sp.f_oneway(*[a for a in arrays if a.size])
        out = dict(test='One-way ANOVA', f_stat=float(stat), p_val=float(p),
                   n_groups=len(test_rows))
        if post_hoc:
            labels = [g.get('name', '') for g, a in zip(test_rows, arrays) for _ in range(len(a))]
            tukey = pairwise_tukeyhsd(np.concatenate(arrays), labels)
            out['tukey'] = [dict(a=str(r[0]), b=str(r[1]), meandiff=float(r[2]),
                                 p_adj=float(r[3]), reject=bool(r[6]))
                            for r in tukey.summary().data[1:]]
        return out

    def run_rm_anova(self, test_rows: list, nonparam: bool, log: bool, post_hoc: bool = True) -> dict:
        """Repeated-measures test: pairs common to every row are the within-subject units; conditions = rows."""
        pair_maps = []
        for g in test_rows:
            pairs = [SelectionData.as_pair_key(p) for p in (g.get('pairs') or [])]
            vals  = self.maybe_log_transform(g.get('vals', []), log)
            pair_maps.append({p: v for p, v in zip(pairs, vals)})
        common = sorted(set.intersection(*(set(pm) for pm in pair_maps)),
                        key=lambda k: k.pair_sort_key())
        if len(common) < 2:
            return {'error': f"Need ≥2 pairs in all rows (found {len(common)})."}
        row_names = [g.get('name', f'G{i+1}') for i, g in enumerate(test_rows)]
        arrays = [np.array([pm[p] for p in common], dtype=float) for pm in pair_maps]
        n_comp = max(1, len(test_rows) * (len(test_rows) - 1) // 2)
        if nonparam:
            stat, p = _sp.friedmanchisquare(*arrays)
            posthoc = []
            for i in range(len(test_rows) if post_hoc else 0):
                for j in range(i + 1, len(test_rows)):
                    w, wp = _sp.wilcoxon(arrays[i], arrays[j], zero_method='wilcox')
                    posthoc.append(dict(a=row_names[i], b=row_names[j],
                                        stat=float(w), p_raw=float(wp),
                                        p_adj=min(float(wp)*n_comp, 1.0),
                                        reject=float(wp)*n_comp < 0.05))
            return dict(test='Friedman test', stat=float(stat), p_val=float(p),
                        n_subjects=len(common), n_conditions=len(test_rows),
                        posthoc=posthoc, posthoc_method='Wilcoxon (Bonferroni)',
                        common_pairs=common)
        records = [{'subject': str(p), 'condition': gn, 'val': pm[p]}
                   for gn, pm in zip(row_names, pair_maps) for p in common]
        aov = pg.rm_anova(data=pd.DataFrame(records), dv='val', within='condition',
                          subject='subject', detailed=True)
        cr  = aov[aov['Source'] == 'condition'].iloc[0]
        df_num   = float(cr.get('DF1', cr.get('ddof1', float('nan'))))
        df_denom = float(cr.get('DF2', cr.get('ddof2', float('nan'))))
        return dict(test='Repeated-measures ANOVA',
                    f_stat=float(cr['F']), p_val=float(cr['p-unc']),
                    df=f"{df_num:.0f},{df_denom:.0f}",
                    n_subjects=len(common), n_conditions=len(test_rows),
                    common_pairs=common)

    def run_test(self, a_vals, b_vals, a_pairs, b_pairs,
                 test_type, alternative='two-sided', nonparametric=False, log=False) -> dict:
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
                    sem=float(np.std(norm_arr, ddof=1) / np.sqrt(norm_arr.size)),
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

    # -- result text (one function per section) --------------------------------
    _RES_LBL = {'lowres': 'lo-res', 'highres': 'hi-res'}

    def _res(self, sr) -> str:
        return self._RES_LBL.get(sr.resolution, sr.resolution)

    def section_missing(self) -> str:
        """Sessions a selected segment was never computed for; empty when nothing is missing."""
        if not self.missing_segments:
            return ''
        body = '\n'.join(f"   {seg}: {', '.join(sorted(s))}"
                         for seg, s in sorted(self.missing_segments.items()))
        return ("-- Missing segments --\n"
                "   these sessions have no CCG for the segment and were left out:\n" + body)

    def section_stats(self, results: list[StatsResult] = None) -> str:
        results = self.results if results is None else results
        cfg = self.test_config
        alt = cfg.alternative if cfg else 'two-sided'
        dtype = (results[0].groups[0].get('data_type', '')
                 if results and results[0].groups else '')
        # the test that ran, not the one picked: nonparametric turns a t-test into Wilcoxon
        ran = next((sr.res['test'] for sr in results if sr.res.get('test')),
                   cfg.test_type if cfg else '')
        out = [f"-- Stats: {ran} "
               f"({'two-sided' if alt == 'two-sided' else alt}) --",
               f"   dtype={dtype}"]
        for sr in results:
            out.append(f"   {self._res(sr)}:")
            out += self._fmt_res(sr.res, sr.groups)
        return '\n'.join(out)

    @staticmethod
    def _fmt_res(r, groups_) -> list[str]:
        if not r:
            return ["(no result)"]
        if 'error' in r:
            return [f"Error: {r['error']}"]
        means = [round(float(np.mean(g['vals'])), 4) if g.get('vals') else 'n/a' for g in groups_]
        if r.get('paired') and 'n_a' in r:   # matched N differs from raw group N pre-intersection
            n_line = f"  N (matched pairs): {[r['n_a'], r['n_b']]}"
            m_line = f"  Means (matched): {[round(r['mean_a'], 4), round(r['mean_b'], 4)]}"
        elif 'n_subjects' in r:
            n_line = f"  N (matched pairs): {r['n_subjects']}"
            m_line = f"  Means: {means}"
        else:
            n_line = f"  N: {[len(g.get('vals') or []) for g in groups_]}"
            m_line = f"  Means: {means}"
        lines = [f"  Test: {r.get('test', '?')}", n_line, m_line]
        if 'f_stat' in r:
            lines.append(f"  F = {r['f_stat']:.4f}")
        elif 'stat' in r:
            lines.append(f"  stat = {r['stat']:.4f}")
        if 'p_val' in r:
            p = r['p_val']
            lines.append(f"  p = {p:.4g}" + (f" {_star(p)}" if _star(p) else ""))
        for row in (r.get('tukey') or r.get('posthoc') or []):
            lines.append(f"    {row.get('a','?')} vs {row.get('b','?')}: "
                         f"p_adj={row.get('p_adj',1.0):.4g}")
        return lines

    def section_segments(self, results: list[StatsResult] = None) -> str:
        """Which window each segment contributed, sorted by segment then session."""
        results = self.results if results is None else results
        out = ["-- Segments --"]
        for sr in results:
            for g in sr.groups:
                for c in sorted(g.get('seg_configs') or [],
                                key=lambda d: (d['segment'], d['session'])):
                    extent = (f"t0={c['t0']} t1={c['t1']} dur={c['dur']} filter={c['filter']}"
                              if 't0' in c else "whole session")
                    out.append(f"   {self._res(sr)} {g.get('name','?')} | "
                               f"{c['session']} {c['segment']}: {extent} "
                               f"({c['n_pairs']} pairs)")
        return '\n'.join(out)

    def section_outliers(self, results: list[StatsResult] = None) -> str:
        results = self.results if results is None else results
        multi = len(results) > 1
        verb = "removed" if self.test_config.remove_outliers else "flagged"
        out = []
        for sr in results:
            label = f" [{self._res(sr).capitalize()}]" if multi else ""
            out += ["", f"Outliers (>3 SD from group mean, {verb}){label}:"]
            found = [f"  {g.get('name') or chr(65+i)}: {', '.join(items)}"
                     for i, g in enumerate(sr.flagged_groups)
                     if (items := [f"{(p := (g.get('pairs') or [])[j]).ref}-{p.tgt} ({p.session})"
                                   for j in sr.outliers.get(i, [])
                                   if j < len(g.get('pairs') or [])])]
            out += found or ["  (none)"]
        return '\n'.join(out)

    def section_raw(self, results: list[StatsResult] = None) -> str:
        results = self.results if results is None else results
        out = ["-- Raw values --"]
        for sr in results:
            for g in sr.groups:
                vals = g.get('vals') or []
                tag = f"   {self._res(sr)} {g.get('name','?')}"
                out.append(f"{tag} sessions: {g.get('sessions_used') or []}")
                out.append(f"{tag} pairs: {[str(p) for p in (g.get('pairs') or [])]}")
                out.append(f"{tag} (n={len(vals)}): {[round(float(v), 6) for v in vals]}")
        return '\n'.join(out)

    def result_text(self, results: list[StatsResult] = None) -> str:
        """Every section in display order; missing segments lead so they are never buried."""
        results = self.results if results is None else results
        return '\n'.join(s for s in (self.section_missing(),
                                     self.section_stats(results),
                                     self.section_segments(results),
                                     self.section_outliers(results),
                                     self.section_raw(results)) if s)

    # -- validation and persistence -------------------------------------------
    def validate(self) -> str | None:
        """Why this test cannot run, or None."""
        rows = self.rows
        if len(rows) < 2:
            return "Need at least 2 groups to compare."
        dtype = rows[0].data_type
        metric = METRICS[dtype]
        if not metric.enabled:
            return f"Data type '{dtype}' is not yet implemented."
        if metric.norm and len(rows) != 2:
            return f"'{dtype}' requires exactly 2 groups."
        if self.test_config.test_type is not None:
            if (bad := self.check_test_type(self.test_config.test_type, len(rows))):
                return bad
        if self.test_config.test_type == "Pairwise t-test":
            sess_sets = [tuple(r.sessions or self.available_sessions()) for r in rows]
            if len(set(sess_sets)) > 1:
                return "Pairwise t-test requires same sessions in all groups."
        return None

    def to_dict(self) -> dict:
        return dict(rows=[r.serialize() for r in self.rows],
                    test=self.test_config.serialize(),
                    view=self.view_config.serialize())

    def from_dict(self, d: dict) -> None:
        self.rows = []
        for rd in d.get('rows') or []:
            rc = RowConfig(); rc.__setstate__(rd)
            if not rc.id:
                rc.id = self._next_id
            self._next_id = max(self._next_id, rc.id) + 1
            self.rows.append(rc)
        for key, cls, attr in (('test', StatsTestConfig, 'test_config'),
                               ('view', _ViewConfig, 'view_config')):
            if d.get(key):
                cfg = cls(); cfg.__setstate__(d[key]); setattr(self, attr, cfg)

    # -- save / load ---------------------------------------------------------
    @property
    def save_dir(self) -> pathlib.Path:
        d = pathlib.Path(self.cd.stats_results_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def saved_results(self) -> list:
        """Name, path and mtime per saved result; the files are too big to parse for a date."""
        return [(p.stem, str(p),
                 datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(), True, False)
                for p in sorted(self.save_dir.glob('*.json'))]

    def save(self, name: str, **extra) -> str:
        """Write the config, the results and the caller's widget state under *name*."""
        path = str(self.save_dir / f"{name}.json")
        data = dict(self.to_dict(), **extra,
                    results=[r.serialize() for r in self.results],
                    saved_at=datetime.datetime.now().isoformat())
        atomic_write_json(path, text=_compact_json_str(data))
        return path

    def load(self, path: str) -> dict:
        """Restore config and results from *path*; returns the raw dict for widget state."""
        d = json.loads(pathlib.Path(path).read_text())
        self.from_dict(d)
        self.results = []
        for rd in d.get('results') or []:
            sr = StatsResult(); sr.__setstate__(rd)
            self.results.append(sr)
        return d


def _star(p) -> str | None:
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else None))
