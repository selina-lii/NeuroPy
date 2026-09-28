"""Per-neuron tags: assigned ones are stored on Neurons.tags, gradient ones (NeuronTagSpec) are a rule recomputed from cached cut edges."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from neuropy.analyses.utils import JsonSavable
from neuropy.core.intervals import IntervalOp
from neuropy.core.neurons import NO_TAG, is_labelled

SCOPE_ALL = 'all'
SCOPE_SESSION = 'session'

NORMALIZERS = ('', 'zscore', 'log10', 'rank')

FIRING_RATE = 'firing_rate'   # not a metadata column: derived from the spike trains

STABLE_TAG = 'stable'         # read from the recording at load, never saved
TAG_VALUE_PREFIX = 'tag:'     # source 'tag:<name>' reads Neurons.tag_values[<name>]


CUT_QUANTILE = 'quantile'     # cuts follow percentiles of the data
CUT_ABSOLUTE = 'absolute'     # cuts are the values themselves, edited on the slider


ACTIVITY_PREFIX = 'activity:'   # 'activity:<fn>:(seg,…),(seg,…)'; a segment is a custom CCG or an epoch label


def activity_source(fn: str, groups: list) -> str:
    return f"{ACTIVITY_PREFIX}{fn}:" + ','.join(f"({','.join(g)})" for g in groups)


def parse_activity(source: str) -> tuple:
    fn, groups = source[len(ACTIVITY_PREFIX):].split(':', 1)
    return fn, [[s.strip() for s in g.split(',') if s.strip()] for g in re.findall(r'\(([^)]*)\)', groups)]


def tag_value_source(name: str) -> str:
    """Source string naming an existing tag's stored values."""
    return f"{TAG_VALUE_PREFIX}{name}"


def labels_from_cutoffs(values, cutoffs: list, labels: list) -> np.ndarray:
    """Label per value: the rightmost cutoff at or below it wins; non-finite reads NO_TAG."""
    if len(labels) != len(cutoffs) + 1:
        raise ValueError(f"{len(cutoffs)} cutoffs need {len(cutoffs) + 1} labels, "
                         f"got {len(labels)}")
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), NO_TAG, dtype=object)
    known = np.isfinite(v)   # else digitize files an infinite ratio in the top band
    binned = np.digitize(v[known], list(cutoffs), right=False)
    out[known] = [labels[b] for b in binned]
    return out


def normalize(values: np.ndarray, method: str) -> np.ndarray:
    """Transform values before binning; unknown/absent method passes them through."""
    v = np.asarray(values, dtype=float)
    if method == 'log10':
        return np.log10(np.clip(v, 1e-12, None))
    if method == 'zscore':
        sd = v.std()
        return (v - v.mean()) / sd if sd else np.zeros_like(v)
    if method == 'rank':
        order = v.argsort().argsort().astype(float)
        return order / max(1, len(v) - 1)
    return v


def label_kind(entry) -> str:
    """How one session's label list should be read: 'index', 'mask' or 'categorical'."""
    arr = np.asarray(list(entry), dtype=object)
    if arr.size == 0:
        return 'index'
    if all(isinstance(v, (bool, np.bool_)) for v in arr):
        return 'mask'
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
           for v in arr):
        return 'index'
    return 'categorical'


def resolve_labels(entry, n_neurons: int) -> np.ndarray:
    """One session's labels as a length-n_neurons object array, NO_TAG where unlabelled."""
    kind = label_kind(entry)
    if kind != 'index' and len(entry) != n_neurons:
        raise ValueError(f"{kind} labels must cover every neuron: "
                         f"got {len(entry)}, need {n_neurons}")

    if kind == 'categorical':
        return np.array(list(entry), dtype=object)
    if kind == 'mask':
        hits = np.flatnonzero(np.asarray(entry, dtype=bool))
    else:
        hits = np.asarray(list(entry), dtype=int)
        if hits.size and (hits.min() < 0 or hits.max() >= n_neurons):
            raise ValueError(f"neuron index out of range for {n_neurons} neurons: "
                             f"{hits.min()}..{hits.max()}")
    out = np.full(n_neurons, NO_TAG, dtype=object)
    out[hits] = True
    return out


def resolve_values(entry, labels: np.ndarray) -> np.ndarray:
    """Values aligned to *labels*, None where unlabelled; object dtype so a value may be an array."""
    out = np.full(len(labels), None, dtype=object)
    if entry is None:
        return out
    supplied = list(entry)
    tagged = np.flatnonzero(np.array([is_labelled(lab) for lab in labels]))
    if len(supplied) == len(labels):
        positions = range(len(labels))
    elif len(supplied) == len(tagged):
        positions = tagged
    else:
        raise ValueError(f"values must cover either every neuron ({len(labels)}) or "
                         f"every tagged neuron ({len(tagged)}); got {len(supplied)}")
    for i, value in zip(positions, supplied):
        out[i] = value
    return out


def check_labels(nd, labels: dict, values: dict = None) -> list:
    """Problems with a labelling before it is written; empty when it is clean."""
    problems = []
    by_key = {k.nd(): v for k, v in labels.items()}
    vals_by_key = {k.nd(): v for k, v in (values or {}).items()}
    live = set(nd.session_keys)

    for missing in sorted(str(k.session) for k in live - set(by_key)):
        problems.append(f"session not in the labelling: {missing}")
    for unknown in sorted(str(k.session) for k in set(by_key) - live):
        problems.append(f"labelling names an unknown session: {unknown}")
    for orphan in sorted(str(k.session) for k in set(vals_by_key) - set(by_key)):
        problems.append(f"values given for an unlabelled session: {orphan}")

    for key in nd.session_keys:
        if key not in by_key:
            continue
        session = str(key.session)
        try:
            resolved = resolve_labels(by_key[key], nd.neurons_for(key).n_neurons)
        except ValueError as exc:
            problems.append(f"{session}: {exc}")
            continue
        if key in vals_by_key:
            try:
                resolve_values(vals_by_key[key], resolved)
            except ValueError as exc:
                problems.append(f"{session}: values: {exc}")
    return problems


def confirm(problems: list) -> bool:
    """Print each problem and ask whether to write anyway."""
    print(f"{len(problems)} problem(s) with this labelling:")
    for line in problems:
        print(f"  - {line}")
    return input("proceed anyway? [y/N] ").strip().lower().startswith('y')


class NeuronTagSpec(JsonSavable):
    """One gradient tag group: how to read a value, cut it, and colour the bins."""

    def __init__(self, prefix: str = '', source: str = FIRING_RATE, segment: str = '',
                 normalization: str = 'zscore', scope: str = SCOPE_ALL,
                 quantiles: list = None, labels: list = None,
                 base_rgb: tuple = (33, 89, 180), enabled: bool = True,
                 enabled_labels: dict = None, cached_edges: list = None,
                 cut_mode: str = CUT_QUANTILE, cutoffs: list = None, cell_type: str = '',
                 label_colors: dict = None, log_axis: bool = False):
        JsonSavable.__init__(self)
        self.prefix = prefix
        self.source = source
        self.segment = segment          # '' = whole session; else a custom segment's rates
        self.cell_type = cell_type
        self.normalization = normalization
        self.scope = scope
        self.quantiles = list(quantiles or [0.5])
        self.cut_mode = cut_mode
        self.cutoffs = list(cutoffs) if cutoffs else []
        self.labels = list(labels or ['lo', 'hi'])
        self.base_rgb = tuple(base_rgb)
        self.enabled = enabled
        self.enabled_labels = dict(enabled_labels or {})
        self.cached_edges = list(cached_edges) if cached_edges else []
        self.label_colors = dict(label_colors or {})   # user-picked; unpicked labels follow the ramp
        self.log_axis = log_axis

    @property
    def tag_names(self) -> list:
        return [f"{self.prefix}_{label}" for label in self.labels]

    def label_enabled(self, label: str) -> bool:
        return self.enabled and self.enabled_labels.get(label, True)

    def colors(self) -> dict:
        """Label -> rgb, darkest first, lifting toward white across the bins."""
        n = len(self.labels)
        return {label: tuple(self.label_colors.get(label, _ramp(self.base_rgb, i, n)))
                for i, label in enumerate(self.labels)}

    @property
    def is_absolute(self) -> bool:
        """True when cuts are edited values rather than percentiles."""
        return self.cut_mode == CUT_ABSOLUTE

    def definition_key(self) -> tuple:
        """What the cached edges depend on — a change here invalidates them."""
        return (self.source, self.segment, self.cell_type, self.normalization, self.scope,
                self.cut_mode, tuple(self.quantiles), tuple(self.cutoffs))


def _ramp(rgb, step: int, n: int) -> tuple:
    """Later bins lift toward white, keeping the hue readable at every step."""
    f = step / max(1, n - 1) * 0.55
    return tuple(int(c + (255 - c) * f) for c in rgb)


class NeuronTagSet(JsonSavable):
    """The project's gradient tag specs, plus the value lookup that feeds them."""

    _custom_types = {'specs': NeuronTagSpec}

    def __init__(self, save_dir: str = ''):
        JsonSavable.__init__(self, ignored_attrs=['cd'])
        self.specs: dict[str, NeuronTagSpec] = {}
        self.cd = None
        self._save_dir = save_dir
        self._assign_cache: dict = {}
        self._values_cache: dict = {}
        self._missing: dict = {}   # segment -> sessions lacking it, since the last take_missing()

    def bind(self, cd) -> None:
        self.cd = cd

    def save_path(self, **_) -> str | None:
        return os.path.join(self._save_dir, 'neuron_tags') if self._save_dir else None

    def save_if_bound(self) -> bool:
        """Persist the specs when this set has somewhere to write; else do nothing."""
        if not self._save_dir:
            return False
        self.save()
        return True

    def __setstate__(self, state: dict) -> None:
        JsonSavable.__setstate__(self, state)
        for prefix, spec in self.specs.items():
            spec.prefix = prefix

    # ── values ──────────────────────────────────────────────────────────

    def tag_values_numeric(self, name: str, key) -> np.ndarray | None:
        """One tag's stored values as floats, or None when they are not scalar numbers."""
        neurons = self.cd.nd.neurons_for(key.nd())
        stored = neurons.tag_values.get(name)
        if stored is None:
            return None
        out = np.full(len(stored), np.nan)
        for i, value in enumerate(stored):
            if isinstance(value, (int, float, np.integer, np.floating)) \
                    and not isinstance(value, bool):
                out[i] = float(value)
        return None if np.all(np.isnan(out)) else out

    def values_for(self, spec: NeuronTagSpec, key) -> np.ndarray | None:
        """The raw per-neuron values *spec* reads, NaN outside its cell type; None when this session lacks them."""
        ck = (spec.source, spec.segment, key.nd())
        if ck not in self._values_cache:
            self._values_cache[ck] = self._source_values(spec, key)
        values = self._values_cache[ck]
        if values is None or not spec.cell_type:
            return values
        types = np.asarray(self.cd.nd.neurons_for(key.nd()).neuron_type)
        return np.where(types == spec.cell_type, values, np.nan)

    def _source_values(self, spec: NeuronTagSpec, key) -> np.ndarray | None:
        if spec.source.startswith(ACTIVITY_PREFIX):
            return self.activity_level(spec.source, key)
        neurons = self.cd.nd.neurons_for(key.nd())
        if spec.source.startswith(TAG_VALUE_PREFIX):
            return self.tag_values_numeric(spec.source[len(TAG_VALUE_PREFIX):], key)
        if spec.source == FIRING_RATE:
            if spec.segment:
                src = self.cd.source_config(key.change(segment=spec.segment), spec.segment)
                rates = src.firing_rates if src is not None else None
                return np.asarray(rates, dtype=float) if rates else None
            return np.asarray(neurons.firing_rate, dtype=float)
        values = (neurons.metadata or {}).get(spec.source)
        if values is None:
            return None
        arr = np.asarray(values)
        return arr.astype(float) if arr.dtype.kind in 'iuf' else None

    def activity_level(self, source: str, key) -> np.ndarray | None:
        """Per-neuron rate in one segment group (frate_during), or group A's rate over group B's (frate_ratio)."""
        fn, groups = parse_activity(source)
        found = {i: self._group_intervals(key, g) for i, g in enumerate(groups) if g != ['none']}
        if any(iv is None for iv in found.values()):
            return None
        others = IntervalOp.merge([iv for ivs in found.values() for iv in ivs])
        rates = [self._rate(key, found[i] if i in found else
                            IntervalOp.complement(others, *self.cd.nd.session_bounds(key.nd())))
                 for i in range(len(groups))]
        if fn == 'frate_during':
            return rates[0]
        return np.divide(rates[0], rates[1], out=np.full(len(rates[0]), np.nan), where=rates[1] > 0)

    def _group_intervals(self, key, names: list) -> list | None:
        out = []
        for name in names:
            iv = self.cd.segment_intervals(key, name)
            if iv is None:
                self._missing.setdefault(name, set()).add(str(key.session))
                return None
            out += iv
        return IntervalOp.merge(out)

    def _rate(self, key, intervals: list) -> np.ndarray:
        trains = self.cd.nd.neurons_for(key.nd()).spiketrains
        dur = sum(b - a for a, b in intervals)
        if dur <= 0:
            return np.full(len(trains), np.nan)
        starts, stops = np.array(intervals, dtype=float).T
        return np.array([np.sum(np.searchsorted(st, stops) - np.searchsorted(st, starts))
                         for st in trains], dtype=float) / dur

    def take_missing(self) -> dict:
        """Segments some session lacked while computing values, then forget them."""
        missing, self._missing = self._missing, {}
        return missing

    def pooled_values(self, spec: NeuronTagSpec, key=None) -> np.ndarray:
        """Values for *spec*: one session when *key* is given, else every session pooled.

        Percentile cuts normalize per session before pooling, so a z-score says
        'high for its own session' alike everywhere; absolute cuts must stay raw
        or the numbers on the slider would not be the numbers in the data.
        """
        keys = [key] if key is not None else self.cd.nd.session_keys
        chunks = []
        for nd_key in keys:
            values = self.values_for(spec, nd_key)
            if values is not None and len(values):
                chunks.append(values if spec.is_absolute
                              else normalize(values, spec.normalization))
        if not chunks:
            return np.array([])
        pooled = np.concatenate(chunks)
        return pooled[np.isfinite(pooled)]   # a ratio source divides by zero

    def edges_for(self, spec: NeuronTagSpec, key) -> list:
        """Cut points for *spec*; project-wide percentile ones are cached on the spec."""
        if spec.is_absolute:
            return list(spec.cutoffs)
        if spec.scope == SCOPE_SESSION:
            values = self.values_for(spec, key)
            if values is None or not len(values):
                return []
            return list(np.quantile(normalize(values, spec.normalization), spec.quantiles))
        if not spec.cached_edges:
            pooled = self.pooled_values(spec)
            spec.cached_edges = (list(np.quantile(pooled, spec.quantiles))
                                 if len(pooled) else [])
        return spec.cached_edges

    def band_counts(self, spec: NeuronTagSpec, key=None) -> list:
        """Neurons falling in each label band — what the slider's histogram reports."""
        values = self.pooled_values(spec, key)
        if not len(values):
            return [0] * len(spec.labels)
        assigned = labels_from_cutoffs(values, self.edges_for(spec, key), spec.labels)
        return [int(np.sum(assigned == label)) for label in spec.labels]

    def clear_values(self) -> None:
        self._values_cache.clear()
        self._assign_cache.clear()

    def invalidate(self, spec: NeuronTagSpec) -> None:
        """Drop cached edges and assignments — call after any change to the spec."""
        spec.cached_edges = []
        self._assign_cache.clear()

    # ── assignment ──────────────────────────────────────────────────────

    def assign(self, spec: NeuronTagSpec, key) -> dict:
        """neuron id -> label, or {} when this session cannot supply the source."""
        labels = self.assign_array(spec, key)
        if labels is None:
            return {}
        ids = self.cd.nd.neurons_for(key.nd()).neuron_ids   # ids are not positions
        return {int(nid): label for nid, label in zip(ids, labels) if is_labelled(label)}

    def assign_array(self, spec: NeuronTagSpec, key) -> np.ndarray | None:
        """One session's labels as a per-neuron array, or None when the source is missing."""
        values = self.values_for(spec, key)
        edges = self.edges_for(spec, key)
        if values is None or not len(values) or not edges:
            return None
        cut = values if spec.is_absolute else normalize(values, spec.normalization)
        return labels_from_cutoffs(cut, edges, spec.labels)

    def apply(self, spec: NeuronTagSpec, key=None) -> int:
        """Write *spec*'s labels into Neurons.tags; one session when *key* is given.

        Returns the sessions written.
        """
        keys = [key] if key is not None else self.cd.nd.session_keys
        labels, values = {}, {}
        for nd_key in keys:
            assigned = self.assign_array(spec, nd_key)
            if assigned is not None:
                labels[nd_key] = list(assigned)
                values[nd_key] = self.values_for(spec, nd_key).tolist()
        if labels:
            own = tag_value_source(spec.prefix)
            self._values_cache = {k: v for k, v in self._values_cache.items() if k[0] != own}
            self.cd.nd.tag_neurons(spec.prefix, labels, values, strict=False,
                                   replace=key is None)
        return [str(k.session) for k in labels]

    def preview(self, spec: NeuronTagSpec, key=None) -> pd.DataFrame:
        """What apply would write, as tag_table rows, without writing."""
        rows = []
        for nd_key in ([key] if key is not None else self.cd.nd.session_keys):
            labels = self.assign_array(spec, nd_key)
            if labels is not None:
                rows += [(str(nd_key.session), int(nid), lab, val) for nid, lab, val in
                         zip(self.cd.nd.neurons_for(nd_key).neuron_ids, labels,
                             self.values_for(spec, nd_key)) if is_labelled(lab)]
        return pd.DataFrame(rows, columns=['session', 'neuron_id', 'label', 'value'])

    def neuron_labels(self, key, neuron: int) -> list:
        """``[(spec, label)]`` of every enabled tag covering one neuron, memoized per spec and session."""
        out = []
        for spec in self.specs.values():
            if not spec.enabled:
                continue
            ck = (spec.prefix, spec.definition_key(), key.nd())
            if ck not in self._assign_cache:
                self._assign_cache[ck] = self.assign(spec, key)
            label = self._assign_cache[ck].get(neuron)
            if label is not None and spec.label_enabled(label):
                out.append((spec, label))
        return out

    def neuron_rgb(self, key, neuron: int) -> list:
        """Gradient dots for one neuron: one per enabled tag that covers it."""
        return [spec.colors()[label] for spec, label in self.neuron_labels(key, neuron)]

    def neuron_words(self, key, neuron: int) -> list:
        """``[(word, rgb)]`` naming each enabled tag covering one neuron, e.g. ``NREM_active``."""
        return [(f"{spec.prefix}_{label}", spec.colors()[label])
                for spec, label in self.neuron_labels(key, neuron)]

    def counts(self, spec: NeuronTagSpec, key) -> dict:
        """How many neurons fall in each label — what the manage page reports."""
        assigned = self.assign(spec, key)
        return {label: sum(1 for v in assigned.values() if v == label)
                for label in spec.labels}

    # ── editing ─────────────────────────────────────────────────────────

    def add_spec(self, spec: NeuronTagSpec) -> None:
        self.specs[spec.prefix] = spec

    def remove_spec(self, prefix: str) -> None:
        self.specs.pop(prefix, None)

    def disable_unavailable(self, key) -> None:
        """Turn off tags whose source this session does not carry."""
        for spec in self.specs.values():
            if self.values_for(spec, key) is None:
                spec.enabled = False
