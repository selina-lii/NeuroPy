"""Translates CCG selection pair indices across neuron datasets via shared neuron IDs.

When a selection JSON was saved with nd1 (neuron indices 0..N-1) and you want
to load it under nd2 (a different or subset dataset), the pair indices must be
translated. ``neuron_ids`` stored in the JSON file is the stable bridge.

Typical use (notebook / script)::

    from neuropy.ui.selection_migration import SelectionMigration

    m = SelectionMigration.from_file("data/selections/RatJ_Day1__latest.json",
                                     nd2.neurons.neuron_ids)
    if m and not m.is_identity:
        remapped = m.apply(data)
"""
from __future__ import annotations

import copy
import json


class SelectionMigration:
    """Translates CCG selection pair indices across neuron datasets via shared neuron IDs.

    Usage::

        m = SelectionMigration(data['neuron_ids'], neurons.neuron_ids)
        if not m.is_identity:
            data = m.apply(data)
    """

    def __init__(self, saved_nids: list, cur_nids: list,
                 nid_map: dict | None = None):
        """
        nid_map: optional {saved_nid: cur_nid} for cross-animal/cross-session matching.
        Applied to saved_nids before building the index map.
        """
        if nid_map:
            saved_nids = [nid_map.get(int(n), int(n)) for n in saved_nids]
        self.saved_nids = list(saved_nids)
        self.cur_nids = list(cur_nids)
        cur_id_to_idx = {int(nid): i for i, nid in enumerate(cur_nids)}
        self._remap: dict[int, int] = {
            old: new
            for old, nid in enumerate(self.saved_nids)
            if (new := cur_id_to_idx.get(int(nid))) is not None
        }

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_valid(self) -> bool:
        return bool(self._remap)

    @property
    def is_identity(self) -> bool:
        return all(self._remap.get(i, -1) == i for i in range(len(self.saved_nids)))

    @property
    def n_matched(self) -> int:
        return len(self._remap)

    @property
    def n_dropped(self) -> int:
        return len(self.saved_nids) - self.n_matched

    def summary(self) -> str:
        return (f"Remap: {len(self.saved_nids)} saved, {len(self.cur_nids)} current, "
                f"{self.n_matched} matched, {self.n_dropped} dropped")

    # ── Application ─────────────────────────────────────────────────────────

    def apply(self, data: dict) -> dict:
        """Return deep copy of *data* with all pair indices translated.

        Pairs where either neuron is absent are silently dropped.
        """
        d = copy.deepcopy(data)
        remap = self._remap

        def _remap_pairs(pairs):
            return [[remap[r], remap[c]] for r, c in pairs if r in remap and c in remap]

        for tk in d.get('selections', {}):
            d['selections'][tk] = _remap_pairs(d['selections'][tk])
        for tk in d.get('deleted_by_type', {}):
            d['deleted_by_type'][tk] = _remap_pairs(d['deleted_by_type'][tk])
        new_pt = {}
        for key, val in d.get('pair_tags', {}).items():
            try:
                r, t = map(int, key.split(','))
            except (ValueError, AttributeError):
                continue
            nr, nt = remap.get(r), remap.get(t)
            if nr is not None and nt is not None:
                new_pt[f"{nr},{nt}"] = val
        d['pair_tags'] = new_pt
        return d

    # ── Factory / io ─────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str, cur_nids: list,
                  nid_map: dict | None = None) -> 'SelectionMigration | None':
        """Construct from a saved selection JSON file.

        Returns None if the file has no neuron_ids field.
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        saved_nids = data.get('neuron_ids')
        if saved_nids is None:
            return None
        return cls(saved_nids, cur_nids, nid_map=nid_map)
