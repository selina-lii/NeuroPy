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

from neuropy.analyses.utils import is_shape_label


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
    # ACGs: the matrix diagonal the pointer drops as self-pairs
    acg_ref: np.ndarray = None
    acg_tgt: np.ndarray = None


@dataclass
class PairArrays:
    """Every array view of a batch of pairs as one argument; ``None`` means unavailable, not empty."""
    ccg: np.ndarray
    null: np.ndarray
    ccg_hi: np.ndarray = None
    null_hi: np.ndarray = None
    acg_ref: np.ndarray = None
    acg_tgt: np.ndarray = None

    @property
    def has_highres(self) -> bool:
        return self.ccg_hi is not None

    @property
    def has_acg(self) -> bool:
        return self.acg_ref is not None

    def subset(self, mask) -> 'PairArrays':
        """The same views restricted to a CV fold."""
        def cut(a):
            return None if a is None else a[mask]
        return PairArrays(self.ccg[mask], self.null[mask], cut(self.ccg_hi),
                          cut(self.null_hi), cut(self.acg_ref), cut(self.acg_tgt))


@dataclass
class LabeledSet:
    """Samples plus the label vocabulary they were encoded against."""
    samples: list[PairSample]
    label_names: list[str]

    def __post_init__(self):
        if not self.samples:
            raise ValueError('no pairs in the selected scope — widen the session '
                             'or connection-type filter')
        # Stacked once and reused: these are read on every fold, and restacking
        # per access made a second full copy of the traces each time.
        self._X_ccg = np.stack([s.ccg for s in self.samples]).astype(float)
        self._X_null = np.stack([s.null for s in self.samples]).astype(float)
        # Each sample now views its row of the stack rather than owning a copy.
        for i, s in enumerate(self.samples):
            s.ccg, s.null = self._X_ccg[i], self._X_null[i]
        self._X_ccg_hi = self._stack_hi('ccg_hi')
        self._X_null_hi = self._stack_hi('null_hi')
        self._X_acg_ref = self._stack_hi('acg_ref')
        self._X_acg_tgt = self._stack_hi('acg_tgt')

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
    def X_acg_ref(self) -> np.ndarray | None:
        return self._X_acg_ref

    @property
    def X_acg_tgt(self) -> np.ndarray | None:
        return self._X_acg_tgt

    @property
    def has_acg(self) -> bool:
        return self._X_acg_ref is not None

    def arrays(self, model_uses_highres: bool = True) -> PairArrays:
        """Every view of this set as one object, for a model's fit/predict."""
        hi = self._X_ccg_hi if model_uses_highres else None
        return PairArrays(self._X_ccg, self._X_null, hi,
                          self._X_null_hi if hi is not None else None,
                          self._X_acg_ref, self._X_acg_tgt)

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
                      highres: bool = True,
                      compute_highres: bool = False,
                      conn_types: list[str] = None, sessions: list[str] = None,
                      all_pairs: bool = False) -> LabeledSet:
    """Join saved labels to their CCG traces; keep labels with enough support.

    A label needs *min_count* examples. Which animal a pair came from is carried
    as provenance but never gates anything: the labels describe the CCG alone.
    With *highres*, each sample also carries its fine-binned trace over the same
    time window; a session missing it drops highres for the whole set, so the
    models never learn "missing" as a feature.

    *all_pairs* widens from tagged pairs to every pointer pair — for review tools, not training.
    """
    labels_by_key = read_selection_labels(selections_dir or cd.selections_dir)
    ptr_by_str = {str(k): k for k in cd.ptr}
    want_type, want_sess = set(conn_types or []), set(sessions or [])
    samples: list[PairSample] = []
    for key_str, key in ptr_by_str.items():
        entry = labels_by_key.get(key_str)
        if entry is None:
            if not all_pairs:
                continue
            entry = {'labels': {}, 'complete': False}
        if want_type and key.type_label() not in want_type:
            continue
        # narrow before loading CCGs: a filtered session should cost no load
        if want_sess and str(key.session) not in want_sess:
            continue
        pairs = dict(entry['labels'])
        if all_pairs or entry['complete']:   # reviewed slice → untagged are negatives
            for ref, tgt in cd.ptr[key.ptr()].pair_set:
                pairs.setdefault(f'{ref},{tgt}', [])
        data = cd.ccg_for(key)
        ccg, null = data.ccg[0], data.ccg_null[0]
        ccg_hi, null_hi = (_highres_arrays(cd, key, compute_highres)
                           if highres else (None, None))
        session = str(key.session)
        conn = key.type_label()   # same spelling scope_keys uses
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
                null_hi=np.asarray(null_hi[ref, tgt], dtype=float) if has_hi else None,
                acg_ref=np.asarray(ccg[ref, ref], dtype=float),
                acg_tgt=np.asarray(ccg[tgt, tgt], dtype=float)))

    return LabeledSet(samples=samples,
                      label_names=supported_labels(samples, min_count))


def supported_labels(samples: list, min_count: int) -> list[str]:
    """Labels with enough examples to be learnable."""
    counts = Counter(lab for s in samples for lab in s.labels)
    return sorted(lab for lab, n in counts.items() if n >= min_count)


def build_multi(datasets: list, min_count: int = 20,
                highres: bool = True, compute_highres: bool = False,
                only_labels: list[str] = None,
                conn_types: list[str] = None, sessions: list[str] = None,
                all_pairs: bool = False) -> LabeledSet:
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
        part = build_labeled_set(cd, min_count=1, highres=highres,
                                 compute_highres=compute_highres,
                                 conn_types=conn_types, sessions=sessions,
                                 all_pairs=all_pairs)
        samples.extend(part.samples)
        sources.append(cd.conf.name)
    widths = {(s.ccg.shape[-1],
               None if s.ccg_hi is None else s.ccg_hi.shape[-1]) for s in samples}
    if len(widths) > 1:
        raise ValueError(f'projects disagree on CCG bin widths {sorted(widths)}; '
                         'they must be computed with the same window and bin size')
    names = supported_labels(samples, min_count)
    if only_labels:
        names = [n for n in names if n in set(only_labels)]
        if not names:
            raise ValueError(f'none of {sorted(only_labels)} has enough examples to train on')
    ls = LabeledSet(samples=samples, label_names=names)
    ls.sources = sources
    return ls
