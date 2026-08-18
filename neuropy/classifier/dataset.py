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
    """One labeled pair: where it came from, plus its raw CCG traces.

    Both resolutions span the same ±duration/2 window — lowres at 1 ms bins,
    highres at 1/30 ms — so they carry different information over identical
    time, not different extents.
    """
    session: str
    rat: str
    conn_type: str
    ref: int
    tgt: int
    ccg: np.ndarray             # [n_bin] raw counts, lowres
    null: np.ndarray            # [n_bin] convolution baseline, lowres
    labels: list[str] = field(default_factory=list)
    ccg_hi: np.ndarray = None   # [n_bin_hi] highres, None when not on disk
    null_hi: np.ndarray = None


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
        self._X_ccg_hi = self._stack_hi('ccg_hi')
        self._X_null_hi = self._stack_hi('null_hi')

    def _stack_hi(self, attr: str) -> np.ndarray | None:
        """Stack a highres field, or None when any sample lacks it."""
        vals = [getattr(s, attr) for s in self.samples]
        if any(v is None for v in vals):
            return None
        return np.stack(vals).astype(float)

    @property
    def has_highres(self) -> bool:
        return self._X_ccg_hi is not None

    @property
    def X_ccg(self) -> np.ndarray:
        return self._X_ccg

    @property
    def X_null(self) -> np.ndarray:
        return self._X_null

    @property
    def X_ccg_hi(self) -> np.ndarray | None:
        return self._X_ccg_hi

    @property
    def X_null_hi(self) -> np.ndarray | None:
        return self._X_null_hi

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

    def provenance(self) -> dict:
        """What this set was built from — recorded with any model trained on it."""
        per_session: dict[str, int] = {}
        for smp in self.samples:
            per_session[smp.session] = per_session.get(smp.session, 0) + 1
        return {'n_samples': len(self.samples),
                'n_negatives': sum(1 for s in self.samples if not s.labels),
                'sessions': sorted(per_session),
                'pairs_per_session': per_session,
                'rats': sorted(set(self.rats.tolist())),
                'labels': list(self.label_names),
                'label_counts': {k: int(v) for k, v in self.counts().items()},
                'n_bins_lowres': int(self._X_ccg.shape[1]),
                'n_bins_highres': (None if self._X_ccg_hi is None
                                   else int(self._X_ccg_hi.shape[1]))}

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


def read_selection_labels(selections_dir: str) -> dict[str, dict]:
    """``{str(Key): {'labels': {'ref,tgt': [...]}, 'complete': bool}}`` over every session file.

    ``complete`` marks a slice the user reviewed exhaustively, so its untagged
    pairs are true negatives rather than pairs nobody looked at yet.
    """
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(selections_dir, '*.json'))):
        stem = os.path.basename(path)[:-len('.json')]
        if stem in ('groups', 'selection_dataset'):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        for key_str, sel in doc.get('selections', {}).items():
            entry = out.setdefault(key_str, {'labels': {}, 'complete': False})
            entry['complete'] |= bool(sel.get('complete', False))
            for pair, meta in sel.get('tags', {}).items():
                labels = [g for g in meta.get('groups', []) if is_shape_label(g)]
                if labels:
                    entry['labels'][pair] = labels
    return out


def loaded_ccg(cd, key, resolution: str):
    """CCGData for *key* at *resolution*, read from disk and never computed.

    ``ccg_for`` recomputes on a cache miss, which needs neurons a scoring-only
    dataset does not carry. ``load`` caches under the full key while ``ccg_for``
    reads ``key.cd()``, so both spellings are checked here.
    """
    k = key.change(resolution=resolution)
    for probe in (k.cd(), k):
        data = cd._ccg.get(probe)
        if data is not None and data.ccg is not None:
            return data
    cd.load(k)
    for probe in (k.cd(), k):
        data = cd._ccg.get(probe)
        if data is not None and data.ccg is not None:
            return data
    return None


def _highres_arrays(cd, key, compute: bool = False):
    """``(ccg, null)`` at high resolution, or ``(None, None)`` when absent.

    With *compute*, a session missing highres is computed and saved. A model
    trained on both resolutions emits a fixed feature width, so it can never
    score a lowres-only session — the data has to exist, not be worked around.
    """
    data = loaded_ccg(cd, key, 'highres')
    if data is None and compute:
        cd.get_ccg(key.change(resolution='highres'))
        data = loaded_ccg(cd, key, 'highres')
    if data is None or data.ccg is None:
        return None, None
    return data.ccg[0], data.ccg_null[0]


def build_labeled_set(cd, selections_dir: str = None, min_count: int = 60,
                      min_rats: int = 4, highres: bool = True,
                      compute_highres: bool = False) -> LabeledSet:
    """Join saved labels to their CCG traces; keep labels with enough support.

    A label needs *min_count* examples across at least *min_rats* animals, since
    a label seen in one rat cannot be shown to generalize across animals.
    With *highres*, each sample also carries its fine-binned trace over the same
    time window; a session missing it drops highres for the whole set, so the
    models never learn "missing" as a feature.
    """
    labels_by_key = read_selection_labels(selections_dir or cd.selections_dir)
    ptr_by_str = {str(k): k for k in cd.ptr}

    samples: list[PairSample] = []
    for key_str, entry in labels_by_key.items():
        key = ptr_by_str.get(key_str)
        if key is None:
            continue
        pairs = dict(entry['labels'])
        if entry['complete']:   # reviewed slice → untagged pairs are negatives
            for ref, tgt in cd.ptr[key.ptr()].pair_set:
                pairs.setdefault(f'{ref},{tgt}', [])
        data = cd.ccg_for(key)
        ccg, null = data.ccg[0], data.ccg_null[0]
        ccg_hi, null_hi = (_highres_arrays(cd, key, compute_highres)
                           if highres else (None, None))
        session = str(key.session)
        conn = '-'.join(key.conn_type) if key.conn_type else '?'
        for pair, labs in pairs.items():
            ref, tgt = (int(v) for v in pair.split(','))
            if ref >= ccg.shape[0] or tgt >= ccg.shape[1]:
                continue
            has_hi = ccg_hi is not None and ref < ccg_hi.shape[0] and tgt < ccg_hi.shape[1]
            samples.append(PairSample(
                session=session, rat=rat_of(session), conn_type=conn,
                ref=ref, tgt=tgt,
                ccg=np.asarray(ccg[ref, tgt], dtype=float),
                null=np.asarray(null[ref, tgt], dtype=float),
                labels=labs,
                ccg_hi=np.asarray(ccg_hi[ref, tgt], dtype=float) if has_hi else None,
                null_hi=np.asarray(null_hi[ref, tgt], dtype=float) if has_hi else None))

    return LabeledSet(samples=samples,
                      label_names=supported_labels(samples, min_count, min_rats))


def supported_labels(samples: list, min_count: int, min_rats: int) -> list[str]:
    """Labels with enough examples across enough animals to be learnable.

    A label seen in one animal cannot be shown to generalize, so both bars apply.
    """
    counts = Counter(lab for s in samples for lab in s.labels)
    rats: dict[str, set] = {}
    for s in samples:
        for lab in s.labels:
            rats.setdefault(lab, set()).add(s.rat)
    return sorted(lab for lab, n in counts.items()
                  if n >= min_count and len(rats[lab]) >= min_rats)


def build_multi(datasets: list, min_count: int = 20, min_rats: int = 4,
                highres: bool = True, compute_highres: bool = False,
                only_labels: list[str] = None) -> LabeledSet:
    """One labeled set pooled over several projects.

    Label support is judged on the pooled counts, so a label too rare in any one
    project can still qualify once its examples are combined. Bin widths must
    agree across projects — a trace of a different length is not the same feature.

    *only_labels* narrows training to one family of shapes (say the quality
    grades, or the fast patterns). The rest stay in the samples as unlabeled
    negatives, which is the point: telling best from good is an easier problem
    than telling all thirteen apart at once.
    """
    samples, sources = [], []
    for cd in datasets:
        part = build_labeled_set(cd, min_count=1, min_rats=1, highres=highres,
                                 compute_highres=compute_highres)
        samples.extend(part.samples)
        sources.append(cd.conf.name)
    if not samples:
        raise ValueError('no labeled pairs found in the selected projects')
    widths = {(s.ccg.shape[-1],
               None if s.ccg_hi is None else s.ccg_hi.shape[-1]) for s in samples}
    if len(widths) > 1:
        raise ValueError(f'projects disagree on CCG bin widths {sorted(widths)}; '
                         'they must be computed with the same window and bin size')
    names = supported_labels(samples, min_count, min_rats)
    if only_labels:
        names = [n for n in names if n in set(only_labels)]
        if not names:
            raise ValueError(f'none of {sorted(only_labels)} has enough examples to train on')
    ls = LabeledSet(samples=samples, label_names=names)
    ls.sources = sources
    return ls
