"""Build a labeled CCG training set from a project's saved selections.

Selections are per-session JSON: ``selections[str(Key)]['tags'][f'{ref},{tgt}']['groups']``.
Each pair carries zero or more group labels, so the learning problem is multi-label.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Labels that name a saved view or a note rather than a CCG shape.
# '?' means "unsure pattern": it co-occurs with every other label and is only
# a third of the time applied alone, so it marks the labeler's confidence, not a
# shape. The model reproduces it by abstaining, never by learning it as a class.
_NON_SHAPE_PREFIXES = ('__special_', '__admitted__')
_NON_SHAPE = {'deleted', 'Interesting', 'bad', 'emerging', 'pruning', '?'}


@dataclass
class PairSample:
    """One labeled pair: where it came from, plus its raw CCG traces."""
    session: str
    rat: str
    conn_type: str
    ref: int
    tgt: int
    ccg: np.ndarray        # [n_bin] raw counts
    null: np.ndarray       # [n_bin] jitter/convolution baseline
    labels: list[str] = field(default_factory=list)


@dataclass
class LabeledSet:
    """Samples plus the label vocabulary they were encoded against."""
    samples: list[PairSample]
    label_names: list[str]

    def __post_init__(self):
        # Stacked once and reused: these are read on every fold, and restacking
        # per access made a second full copy of the traces each time.
        self._X_ccg = np.stack([s.ccg for s in self.samples]).astype(float)
        self._X_null = np.stack([s.null for s in self.samples]).astype(float)
        # Each sample now views its row of the stack rather than owning a copy.
        for i, s in enumerate(self.samples):
            s.ccg, s.null = self._X_ccg[i], self._X_null[i]

    @property
    def X_ccg(self) -> np.ndarray:
        return self._X_ccg

    @property
    def X_null(self) -> np.ndarray:
        return self._X_null

    @property
    def Y(self) -> np.ndarray:
        """Multi-label indicator ``[n_sample, n_label]``."""
        idx = {n: i for i, n in enumerate(self.label_names)}
        y = np.zeros((len(self.samples), len(self.label_names)), dtype=int)
        for i, s in enumerate(self.samples):
            for lab in s.labels:
                if lab in idx:
                    y[i, idx[lab]] = 1
        return y

    @property
    def rats(self) -> np.ndarray:
        """Grouping variable for leave-one-rat-out CV."""
        return np.array([s.rat for s in self.samples])

    def counts(self) -> Counter:
        return Counter(lab for s in self.samples for lab in s.labels)


def is_shape_label(name: str) -> bool:
    """True for labels describing CCG shape (the only ones worth learning)."""
    return not name.startswith(_NON_SHAPE_PREFIXES) and name not in _NON_SHAPE


def rat_of(session: str) -> str:
    return session.split('_')[0]


def read_selection_labels(selections_dir: str) -> dict[str, dict[str, list[str]]]:
    """``{str(Key): {'ref,tgt': [label, ...]}}`` merged over every session file."""
    out: dict[str, dict[str, list[str]]] = {}
    for path in sorted(glob.glob(os.path.join(selections_dir, '*.json'))):
        stem = os.path.basename(path)[:-len('.json')]
        if stem in ('groups', 'selection_dataset'):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        for key_str, sel in doc.get('selections', {}).items():
            for pair, meta in sel.get('tags', {}).items():
                labels = [g for g in meta.get('groups', []) if is_shape_label(g)]
                if labels:
                    out.setdefault(key_str, {})[pair] = labels
    return out


def build_labeled_set(cd, selections_dir: str, min_count: int = 60,
                      min_rats: int = 4) -> LabeledSet:
    """Join saved labels to their CCG traces; keep labels with enough support.

    A label needs *min_count* examples across at least *min_rats* animals, since
    a label seen in one rat cannot be shown to generalize across animals.
    """
    labels_by_key = read_selection_labels(selections_dir)
    ptr_by_str = {str(k): k for k in cd.ptr}

    samples: list[PairSample] = []
    for key_str, pairs in labels_by_key.items():
        key = ptr_by_str.get(key_str)
        if key is None:
            continue
        data = cd.ccg_for(key)
        ccg, null = data.ccg[0], data.ccg_null[0]
        session = str(key.session)
        conn = '-'.join(key.conn_type) if key.conn_type else '?'
        for pair, labs in pairs.items():
            ref, tgt = (int(v) for v in pair.split(','))
            if ref >= ccg.shape[0] or tgt >= ccg.shape[1]:
                continue
            samples.append(PairSample(
                session=session, rat=rat_of(session), conn_type=conn,
                ref=ref, tgt=tgt,
                ccg=np.asarray(ccg[ref, tgt], dtype=float),
                null=np.asarray(null[ref, tgt], dtype=float),
                labels=labs))

    counts = Counter(lab for s in samples for lab in s.labels)
    rats = {}
    for s in samples:
        for lab in s.labels:
            rats.setdefault(lab, set()).add(s.rat)
    names = sorted(lab for lab, n in counts.items()
                   if n >= min_count and len(rats[lab]) >= min_rats)
    return LabeledSet(samples=samples, label_names=names)
