"""Per-neuron gradient tags: a numeric column cut into named bins.

Unlike pair groups, membership is never stored — a tag is a rule (source column,
normalization, scope, cut points) and the assignment is recomputed from it. Only
the cut edges are cached, because a project-wide scope has to read every session.
"""
from __future__ import annotations

import os

import numpy as np

from neuropy.analyses.utils import JsonSavable

SCOPE_ALL = 'all'
SCOPE_SESSION = 'session'

NORMALIZERS = ('', 'zscore', 'log10', 'rank')

FIRING_RATE = 'firing_rate'   # not a metadata column: derived from the spike trains


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


class NeuronTagSpec(JsonSavable):
    """One gradient tag group: how to read a value, cut it, and colour the bins."""

    def __init__(self, prefix: str = '', source: str = FIRING_RATE, segment: str = '',
                 normalization: str = 'zscore', scope: str = SCOPE_ALL,
                 quantiles: list = None, labels: list = None,
                 base_rgb: tuple = (33, 89, 180), enabled: bool = True,
                 enabled_labels: dict = None, cached_edges: list = None):
        JsonSavable.__init__(self)
        self.prefix = prefix
        self.source = source
        self.segment = segment          # '' = whole session; else a custom segment's rates
        self.normalization = normalization
        self.scope = scope
        self.quantiles = list(quantiles or [0.5])
        self.labels = list(labels or ['lo', 'hi'])
        self.base_rgb = tuple(base_rgb)
        self.enabled = enabled
        self.enabled_labels = dict(enabled_labels or {})
        self.cached_edges = list(cached_edges) if cached_edges else []

    @property
    def tag_names(self) -> list:
        return [f"{self.prefix}_{label}" for label in self.labels]

    def label_enabled(self, label: str) -> bool:
        return self.enabled and self.enabled_labels.get(label, True)

    def colors(self) -> dict:
        """Label -> rgb, darkest first, lifting toward white across the bins."""
        n = len(self.labels)
        return {label: _ramp(self.base_rgb, i, n) for i, label in enumerate(self.labels)}

    def definition_key(self) -> tuple:
        """What the cached edges depend on — a change here invalidates them."""
        return (self.source, self.segment, self.normalization, self.scope,
                tuple(self.quantiles))


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

    def bind(self, cd) -> None:
        self.cd = cd

    def save_path(self, **_) -> str | None:
        return os.path.join(self._save_dir, 'neuron_tags') if self._save_dir else None

    def __setstate__(self, state: dict) -> None:
        JsonSavable.__setstate__(self, state)
        for prefix, spec in self.specs.items():
            spec.prefix = prefix

    # ── values ──────────────────────────────────────────────────────────

    def available_sources(self, key) -> list:
        """Numeric per-neuron columns a tag can read for *key*'s session."""
        neurons = self.cd.nd.neurons_for(key.nd())
        found = [FIRING_RATE]
        for name, values in (neurons.metadata or {}).items():
            arr = np.asarray(values)
            if arr.ndim == 1 and arr.dtype.kind in 'iuf' and len(arr) == neurons.n_neurons:
                found.append(name)
        return found

    def values_for(self, spec: NeuronTagSpec, key) -> np.ndarray | None:
        """The raw per-neuron values *spec* reads, or None when this session lacks them."""
        neurons = self.cd.nd.neurons_for(key.nd())
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

    def _pooled_values(self, spec: NeuronTagSpec) -> np.ndarray:
        """Every session's values for *spec*, normalized per session then pooled.

        Normalizing before pooling is what makes a project-wide cut comparable:
        a z-score says 'high for its own session' in every session alike.
        """
        chunks = []
        for nd_key in self.cd.nd.session_keys():
            values = self.values_for(spec, nd_key)
            if values is not None and len(values):
                chunks.append(normalize(values, spec.normalization))
        return np.concatenate(chunks) if chunks else np.array([])

    def edges_for(self, spec: NeuronTagSpec, key) -> list:
        """Cut points for *spec*; project-wide ones are cached on the spec."""
        if spec.scope == SCOPE_SESSION:
            values = self.values_for(spec, key)
            if values is None or not len(values):
                return []
            return list(np.quantile(normalize(values, spec.normalization), spec.quantiles))
        if not spec.cached_edges:
            pooled = self._pooled_values(spec)
            spec.cached_edges = (list(np.quantile(pooled, spec.quantiles))
                                 if len(pooled) else [])
        return spec.cached_edges

    def invalidate(self, spec: NeuronTagSpec) -> None:
        """Drop cached edges — call after any change to the spec's definition."""
        spec.cached_edges = []

    # ── assignment ──────────────────────────────────────────────────────

    def assign(self, spec: NeuronTagSpec, key) -> dict:
        """neuron index -> label, or {} when this session cannot supply the source."""
        values = self.values_for(spec, key)
        edges = self.edges_for(spec, key)
        if values is None or not len(values) or not edges:
            return {}
        binned = np.digitize(normalize(values, spec.normalization), edges)
        return {i: spec.labels[min(int(b), len(spec.labels) - 1)]
                for i, b in enumerate(binned)}

    def neuron_rgb(self, key, neuron: int) -> list:
        """Gradient dots for one neuron: one per enabled tag that covers it."""
        out = []
        for spec in self.specs.values():
            if not spec.enabled:
                continue
            label = self.assign(spec, key).get(neuron)
            if label is not None and spec.label_enabled(label):
                out.append(spec.colors()[label])
        return out

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
