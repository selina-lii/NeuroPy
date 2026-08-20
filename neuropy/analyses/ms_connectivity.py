import datetime
import glob
import json
import os
import copy
import shutil
import dataclasses
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import hickle as hkl
from scipy.signal import windows
from scipy.stats import poisson
from scipy import ndimage
from statsmodels.stats.multitest import multipletests

try:
    import cupy as cp
except ImportError:
    cp = None

import neuropy.analyses.correlations as correlations
from neuropy.analyses.utils import (_san_np, _hasvalue, Config, AnalysisDataset, JsonSavable,
                                    NpzSavable, HklSavable, Cacheable, SetOp, SessionMemoryCache)
from neuropy.analyses.neurons_dataset import Key, NeuronsDataset, NeuronsDatasetConfig
from neuropy.analyses.pair_selection_data import (SelectionDataset,
                                                  adopt_project_groups)
from neuropy.core.nwb_session import NWBDataset
from neuropy.io.fieldmap import FieldMap
from neuropy.io.nwbio import UNITS_SCHEMA
from neuropy.analyses.ccg_transforms import (
    CCGNorm, ConnectionStrength, NormalizeBy, ConnStrengthConfig)
from neuropy.utils.data_storage_util import atomic_write_json
from collections import defaultdict

_REPO_ROOT = _Path(__file__).resolve().parents[2]
DATA_ROOT = str(_REPO_ROOT / "data")

_CCG_RESOLUTION = {
    'lowres':  1e-3,    # 1 ms   — default, fast
    'highres': 1/3*1e-4,  # 0.1 ms — finer temporal resolution (must exceed 1/sample_rate)
}


def _san(v) -> str:
    """Sanitize for dir names ('.' → '-')."""
    return str(v).replace('.', '-')


class CCGConfig(Config):
    """CCG compute config (`key` → recompute; `derived` → EranConv only)."""

    _groups = {
        'key': ['name', 'resolution', 'bin_size', 'duration', 'conv_window',
                'conn_types', 'use_acceleration', 'symmetrize_ccg'],
        'derived': ['alpha', 'alpha2', 'min_lag', 'max_lag',
                    'min_spkcount', 'spkcnt_scope', 'multiple_correction'],
    }

    # "init param": ("kind", choices)  — what a builder UI exposes; defaults come from __init__
    _options = {
        'duration':            ('metric', ['ms', 's']),
        'bin_size':            ('metric', ['ms', 's']),   # blank -> resolution preset
        'resolution':          ('choice', ['lowres', 'highres']),
        'conv_window':         ('metric', ['ms', 's']),
        'alpha':               ('float', None),
        'alpha2':              ('float', None),
        'min_lag':             ('metric', ['ms', 's']),
        'max_lag':             ('metric', ['ms', 's']),
        'min_spkcount':        ('float', None),
        'spkcount_scope':      ('metric', ['ms', 's']),
        'multiple_correction': ('choice', ['bonferroni', 'fdr_bh']),
        'use_acceleration':    ('choice', [None, True, False]),   # None -> auto-detect CuPy
        'symmetrize_ccg':      ('bool', None),
    }

    def __init__(
        self,
        name="default",
        conn_types: list = None,
        duration: float = 20e-3,
        bin_size: float = None,
        resolution: str = 'lowres',
        conv_window: float = 5e-3,
        alpha: float = 0.05,
        alpha2: float = 0.1,
        min_lag: float = 1e-3,
        max_lag: float = 3e-3,
        min_spkcount=2.5,
        spkcount_scope=12e-3,
        multiple_correction: str = 'bonferroni',  # 'bonferroni' or 'fdr_bh'
        use_acceleration=None,  # None → auto-detect CuPy; True/False to override
        symmetrize_ccg=True,
    ):
        if conn_types is None:
            conn_types = [('E', ('pyr', 'pyr')),
                          ('E', ('pyr', 'inter')),
                          ('I', ('inter', 'inter')),
                          ('I', ('inter', 'pyr'))]
        super().__init__()
        self.name = name
        self.resolution = resolution
        self._root = DATA_ROOT

        # bin_size: explicit value takes priority; otherwise use resolution preset
        if bin_size is None:
            bin_size = _CCG_RESOLUTION.get(resolution, 1e-3)
        self.bin_size = bin_size

        self.conn_types = conn_types
        self.duration = duration
        self.conv_window = conv_window
        self.alpha = alpha
        self.alpha2 = alpha2
        self.multiple_correction = multiple_correction
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_spkcount = min_spkcount
        self.spkcnt_scope = spkcount_scope
        self.use_acceleration = use_acceleration
        self.symmetrize_ccg = symmetrize_ccg

    @property
    def conn_types_E(self):
        return [ct for ei, ct in self.conn_types if ei == 'E']

    @property
    def conn_types_I(self):
        return [ct for ei, ct in self.conn_types if ei == 'I']

    @property
    def conn_types_flat(self):
        return [ct for _, ct in self.conn_types]

    @property
    def conn_types_labeled(self):
        """Same order as ``conn_types_flat``, with E/I labels."""
        return list(self.conn_types)

    @staticmethod
    def conn_type_label(ei, ct) -> str:
        """Labeled conn-type string, e.g. "E PYR→PYR" (matches Key.type_label())."""
        parts = [p for p in (ei, Key.format_conn_type(ct) if ct else '') if p]
        return ' '.join(parts)

    @property
    def conn_type_labels(self) -> list[str]:
        """All conn types as labeled strings, in config order."""
        return [self.conn_type_label(ei, ct) for ei, ct in self.conn_types]

    def parse_conn_type_label(self, label: str):
        """(excitability, conn_type) for a label, or (None, None) if unknown."""
        for ei, ct in self.conn_types:
            if self.conn_type_label(ei, ct) == label:
                return ei, ct
        return None, None

    def at(self, resolution: str) -> 'CCGConfig':
        """This conf at another resolution — bin_size and every derived lag bin follow."""
        return self if resolution == self.resolution else self.copy(
            bin_size=_CCG_RESOLUTION[resolution], resolution=resolution)

    @property
    def center_bin(self) -> int:
        return int(self.duration / self.bin_size // 2)

    @property
    def nbins(self) -> int:
        return int(self.duration / self.bin_size) + 1

    @property
    def spkcnt_bins(self) -> int:
        return int(self.spkcnt_scope / self.bin_size)

    @property
    def min_lag_bin(self) -> int:
        return self.center_bin + int(self.min_lag / self.bin_size)

    @property
    def max_lag_bin(self) -> int:
        return self.center_bin + int(self.max_lag / self.bin_size) + 1

    @property
    def min_spkcnt_bin(self) -> int:
        return self.center_bin - self.spkcnt_bins // 2

    @property
    def max_spkcnt_bin(self) -> int:
        return self.center_bin + self.spkcnt_bins // 2 + 1

    def save_path(self, suffix='config') -> str:
        """Path under ``data/project_{name}/ccg/config/``."""
        base = os.path.join(self._root, f"project_{self.name}", "ccg", "config",
                            f"{self.name}_{self.resolution}")
        return f"{base}_{suffix}" if suffix else base

    def __setstate__(self, state: dict) -> None:
        """JSON has no tuples, so conn types come back as lists — they must stay hashable."""
        super().__setstate__(state)
        self.conn_types = [(ei, tuple(ct)) for ei, ct in self.conn_types]

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.save_path()}\n"
        return s


class ProjectConfig(JsonSavable):
    """Where a project's sessions came from — header at ``project_<name>/project_plan.json``."""

    def __init__(self, name: str = 'default', source: str = None, format: str = None,
                 dataset: str = None, fields: dict = None, nd_conf: dict = None,
                 sampling_rate: float = None, resolution: list = None,
                 built_at: str = None, n_sessions: int = None, root: str = DATA_ROOT):
        super().__init__()
        self.name = name
        self.source = source
        self.format = format
        self.dataset = dataset      # neuropy.io.datasets module supplying session_name + FIELDS
        self.fields = fields
        self.nd_conf = nd_conf      # NeuronsDatasetConfig kwargs this project was built with
        self.sampling_rate = sampling_rate   # spike clock; derived per file when None
        self.resolution = resolution
        self.built_at = built_at    # set once the build succeeded; absent means it never finished
        self.n_sessions = n_sessions
        self._root = root

    def save_path(self, **_) -> str:
        return os.path.join(self._root, f"project_{self.name}", "project_plan")

    @property
    def built(self) -> bool:
        """Whether a build ever ran to completion — a bare header is a failed attempt."""
        return self.built_at is not None

    def mark_built(self, n_sessions: int) -> None:
        """Stamp the header now that the project loaded; call only after a build succeeds."""
        self.built_at = datetime.datetime.now().isoformat(timespec='seconds')
        self.n_sessions = n_sessions
        self.save()

    @property
    def conventions(self):
        """The dataset module — its session_name, FIELDS, and label lists."""
        return import_module(f'neuropy.io.datasets.{self.dataset}') if self.dataset else None

    @property
    def naming(self):
        """How this dataset names a session; the file stem when it declares no rule."""
        return self.conventions.session_name if self.dataset else None

    @property
    def scannable(self) -> bool:
        """True when the header alone can find its sessions: a source to scan, or a dataset naming its own."""
        return bool(self.source) or hasattr(self.conventions, 'sessions')

    def sessions(self) -> list:
        """This project's sessions, read and mapped as the header describes."""
        if not self.source:
            return self.conventions.sessions()
        return NWBDataset(self.source, naming=self.naming).sessions(
            FieldMap(UNITS_SCHEMA, self.fields), sampling_rate=self.sampling_rate)


def build_project(header: ProjectConfig, ccg_conf: CCGConfig, compute: bool = False):
    """Create a project: record how to read it, open it, and stamp it once the run finished."""
    header.save()
    ccg_conf.save()
    neurons, cd, sd = open_project(header.name)
    if compute:
        cd.get_ccg()
    header.mark_built(len(neurons.session_keys()))
    return neurons, cd, sd


def open_project(name: str, sessions: list = None):
    """Everything a project is, from its header; pre-header projects supply their own sessions."""
    header = ProjectConfig(name=name)
    header.load()
    conf = CCGConfig(name=name)
    conf.load()
    nd_conf = NeuronsDatasetConfig(**(header.nd_conf or {}))
    # A scannable project finds and names its own sessions, so it ignores any the
    # caller offered: those belong to whichever project the caller had open, and
    # its naming rule reads paths, not the session objects handed in here.
    if header.scannable:                       # scannable source -> stage 1 names them
        if sessions is not None:
            print(f"[open_project] {name!r} scans its own source; ignoring the "
                  f"{len(sessions)} session(s) passed in", flush=True)
        neurons = NeuronsDataset(header.sessions(), nd_conf,
                                 naming=None if header.source else header.naming)
    elif sessions is None:
        raise ValueError(
            f"project {name!r} is {header.format}: it has no source to scan, "
            "so open_project(name, sessions) must be given its sessions")
    else:                                      # caller-supplied (ProcessData) -> name them here
        neurons = NeuronsDataset(sessions, nd_conf, naming=header.naming)
    cd = CCGDataset(conf, neurons)
    cd.missing_sessions()
    cd.load()
    adopt_project_groups(cd)   # one-time: pre-sharing groups.json moves up to data_root
    sd = SelectionDataset(cd)
    if os.path.isfile(sd.save_path() + '.json'):   # a project starts with nothing selected
        sd.load()
    return neurons, cd, sd


@dataclass(eq=False)
class CCGBatchRequest:
    """UI batch ticket expanded into per-session ``CCGSourceConfig`` children."""
    name: str
    t0: float | str
    t1: float | str
    scope: str = ''                                      # '' | 'all' | a session
    sessions: list = field(default_factory=list)
    n_splits: int = 1
    overlap_raw: float = 0.0
    overlap_unit: str = 'sec'                            # 'sec' | 'min' | 'hr' | '%'
    filter_state: list = field(default_factory=list)
    split_mode: str = 'raw_span'                          # 'raw_span' | 'equal_effective'

    _JSON_KEYS = frozenset([
        'name', 't0', 't1', 'scope', 'sessions', 'n_splits',
        'overlap_raw', 'overlap_unit', 'filter_state', 'split_mode',
    ])

    @staticmethod
    def resolve_overlap_sec(t0, t1, overlap_raw, overlap_unit) -> float:
        """Convert overlap value + unit to seconds (`%` needs numeric t0/t1)."""
        dur = (t1 - t0) if not (isinstance(t0, str) or isinstance(t1, str)) else 0.0
        factor = {'sec': 1.0, 'min': 60.0, 'hr': 3600.0, '%': dur / 100.0}[overlap_unit]
        return max(0.0, overlap_raw * factor)

    def _key(self) -> tuple:
        fs = tuple((str(th['name']), tuple(str(x) for x in th['labels']))
                   for th in self.filter_state)
        return (str(self.name), str(self.t0), str(self.t1), str(self.scope),
                str(self.split_mode), fs)
    def __eq__(self, other) -> bool:
        return isinstance(other, CCGBatchRequest) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def serialize(self) -> dict:
        return {k: getattr(self, k) for k in self._JSON_KEYS}

    @classmethod
    def deserialize(cls, d: dict) -> 'CCGBatchRequest':
        return cls(**{k: d[k] for k in cls._JSON_KEYS if k in d})


class CCGSourceConfig(Config):
    """One custom segment's timing/filter metadata (owns its ``Key``)."""

    _groups = {
        'key': ['name', 't0', 't1', 'filter_state'],
    }
    _JSON_KEYS = frozenset([
        't0', 't1', 'overlap_sec', 'filter_state', 'active_duration',
    ])

    def __init__(self, key, t0=None, t1=None, overlap_sec=0.0,
                 filter_state=None, active_duration=None, firing_rates=None):
        super().__init__()
        self.key = key
        self.t0 = self._coerce_time(t0)
        self.t1 = self._coerce_time(t1)
        self.overlap_sec = overlap_sec
        self.firing_rates = firing_rates
        self.filter_state = filter_state
        # effective seconds; fall back to span only when both bounds are numeric
        if active_duration is None and not (isinstance(self.t0, str) or self.t0 is None
                                            or isinstance(self.t1, str) or self.t1 is None):
            active_duration = self.t1 - self.t0
        self.active_duration = active_duration

    @staticmethod
    def _coerce_time(v):
        """Pass through None/start/end; else float."""
        if v is None or (isinstance(v, str) and v.lower() in ('start', 'end')):
            return v
        return float(v)

    @property
    def name(self) -> str:
        return self.key.segment

    @property
    def total_time_hours(self) -> float | None:
        """From ``active_duration`` (not stored separately)."""
        return None if self.active_duration is None else self.active_duration / 3600.0

    def _stem(self) -> str:
        return f"{_san(self.key.segment)}.{_san(self.key.session)}"

    def data_dir(self, resolution) -> str:
        return os.path.join(self._root, 'custom_ccg', 'ccgdata', f"{self._stem()}.{_san(resolution)}")

    def save_path(self) -> str:
        """Res-independent meta json path (``.json`` appended on save)."""
        return os.path.join(self._root, 'custom_ccg', 'config', self._stem())

    @staticmethod
    def stem_to_key(stem: str) -> 'Key':
        segment, session = stem.split('.')
        return Key(segment=segment, session=session)

    @property
    def t0_sec(self) -> float:
        """Numeric t0; raises if unresolved sentinel."""
        if isinstance(self.t0, str):
            raise ValueError(f"t0='{self.t0}' is unresolved — call resolve(session) first")
        return float(self.t0)

    @property
    def t1_sec(self) -> float:
        """Numeric t1; raises if unresolved sentinel."""
        if isinstance(self.t1, str):
            raise ValueError(f"t1='{self.t1}' is unresolved — call resolve(session) first")
        return float(self.t1)


class CCGPointer(HklSavable):
    """Indices of significant pairs into ``CCGData``."""
    IGNORED_KEYS = ('conf', '_root', '_ignored_attrs')

    def __init__(
        self,
        key,
        inds,
        conf: CCGConfig = None,
        root=DATA_ROOT,
    ):
        super().__init__(ignored_attrs=self.IGNORED_KEYS)
        self.key = key
        raw = _san_np(inds)
        self._inds = np.empty((0, 3), dtype=int) if (raw is None or raw.size == 0) else raw
        if conf is None:
            raise ValueError("CCGPointer requires a CCGConfig")
        self.conf = conf
        self._root = root

    @property
    def inds(self):
        if self._inds.ndim == 2 and self._inds.shape[1] == 2:
            self._inds = np.column_stack(
                [np.zeros(len(self._inds), dtype=int), self._inds])
        return self._inds

    @property
    def pairs(self):
        return SetOp.unique(self.inds[:, 1:])  # unique (ref, tgt)

    @property
    def pair_set(self) -> set:
        """Unique ``(ref, tgt)`` pairs."""
        return {(int(r), int(t)) for r, t in self.pairs}

    @property
    def n_pairs(self):
        return self.pairs.shape[0]

    @staticmethod
    def _dir_base(root, conf) -> str:
        """Pointer file stem under project root."""
        return os.path.join(root, "ccg", "pointers", f"{conf.name}_{conf.resolution}")

    def save_path(self) -> str:
        return f"{self._dir_base(self._root, self.conf)}_ccgpointers_{self.key.session}"

    @classmethod
    def load(cls, ptr: dict, conf: 'CCGConfig', root: str = DATA_ROOT) -> str:
        base = cls._dir_base(root, conf)
        files = glob.glob(base + '_ccgpointers_*.hkl')
        if not files:
            return 'missing'
        ptr.clear()
        for f in files:
            try:
                bundle = hkl.load(f)
                for p in (bundle.values() if isinstance(bundle, dict) else [bundle]):
                    if isinstance(p, cls):
                        p.conf = conf
                        p._root = root
                        ptr[p.key.ptr()] = p
            except Exception as exc:
                print(f"[CCGPointer] skip {f}: {exc}")
        if not ptr:
            return 'missing'
        print(f"[CCGPointer] loaded {len(ptr)} pointers ← {base}_*.hkl")
        return 'loaded'

    def __str__(self):
        sess = getattr(self.key, 'session', '?')
        ct   = getattr(self.key, 'conn_type', '?')
        return (f"CCGPointer(session={sess}, conn_type={ct}, "
                f"n_pairs={self.n_pairs}, inds={self._inds.shape})")


class _CCGData(NpzSavable):
    def __init__(self, key, ccg=None, ccg_null=None, pval=None, qval=None, root=DATA_ROOT):
        super().__init__()
        self.key = key
        self._root = root  # caller-supplied npy target dir
        self.ccg = ccg
        self.ccg_null = ccg_null
        self.pval = pval
        self.qval = qval

    def save(self):
        super().save(path=self._root)  # writes npy into caller-supplied dir (meta saved by CCGData)


class CCGData(NpzSavable):
    """Session CCG arrays ``[seg, ref, tgt, bin]`` and p-values."""
    IGNORED_KEYS = ('conf', 'key', '_root', '_ignored_attrs', 'sources')

    def __init__(self, key, conf, ccg=None, ccg_null=None, pval=None, qval=None, root=None):
        super().__init__(ignored_attrs=self.IGNORED_KEYS)
        self.key = key
        self.conf = conf
        self._root = _Path(root) if root is not None else DATA_ROOT / f'project_{self.conf.name}'
        self.ccg = ccg
        self.ccg_null = ccg_null
        self.pval = pval
        self.qval = qval
        self.sources = {}
        self.pval_corrected = None
        self.qval_corrected = None
        self.set_arrays()

    def copy(self) -> 'CCGData':
        """Deep copy of arrays (safe for in-place transforms)."""
        c = CCGData(key=self.key, conf=self.conf,
                    ccg=None if self.ccg is None else self.ccg.copy(),
                    ccg_null=None if self.ccg_null is None else self.ccg_null.copy(),
                    pval=None if self.pval is None else self.pval.copy(),
                    qval=None if self.qval is None else self.qval.copy(),
                    root=str(self._root))
        c.sources = dict(self.sources)
        return c

    def set_arrays(self):
        """Ensure 4D arrays and refresh multiple-test correction."""
        for name in ('ccg', 'ccg_null', 'pval', 'qval'):
            setattr(self, name, CCGData._ensure_4d(getattr(self, name)))
        self.refresh_corrected()

    def load(self, path=None):
        super().load(path=path)
        self.set_arrays()
        return self

    def refresh_corrected(self):
        if self.pval is not None: _, self.pval_corrected = _multiple_correction(
            self.pval, self.conf.alpha, method=self.conf.multiple_correction)
            
        if self.qval is not None: _, self.qval_corrected = _multiple_correction(
            self.qval, self.conf.alpha, method=self.conf.multiple_correction)

    @staticmethod
    def _ensure_4d(a):
        return a[np.newaxis, ...] if a is not None and a.ndim == 3 else a

    @property
    def n_segment(self):
        return self.ccg.shape[0]
    
    @property
    def n_ref(self):
        return self.ccg.shape[1]

    @property
    def n_tgt(self):
        return self.ccg.shape[2]

    def _ref_tgt_grid(self):
        n_ref, n_tgt = self.n_ref, self.n_tgt
        return np.meshgrid(np.arange(n_ref), np.arange(n_tgt), indexing="ij")

    @property
    def segment_names(self) -> list:
        """In-memory segment labels (dim0 stack order); index 0 is whole-session ``all``."""
        return ['all'] + list(self.sources.keys())

    def has_segment(self, label: str) -> bool:
        """True if *label* is on dim0 of this array (index 0 'all' is always present)."""
        if not label or label == 'all':
            return self.ccg is not None
        return label in self.segment_names and self.segment_index(label) < self.n_segment

    def segment_index(self, label: str) -> int:
        """dim0 index of *label*; unknown/'all'/None → 0 (whole session)."""
        names = self.segment_names
        return names.index(label) if label in names else 0

    def segment_name(self, idx: int) -> str:
        """Label at dim0 *idx* (inverse of segment_index)."""
        return self.segment_names[idx]
    
    def attach_segment(self, src: 'CCGSourceConfig', seg: 'CCGData', *, save: bool = True):
        self.drop_segment(src.name)
        for attr in ('ccg', 'ccg_null', 'pval', 'qval'):
            a, b = getattr(self, attr), CCGData._ensure_4d(getattr(seg, attr))
            if a is not None and b is not None:
                setattr(self, attr, np.concatenate([a, b[:1]], axis=0))
        self.sources[src.name] = src
        self.refresh_corrected()
        if save:
            self.save_segment(src.name)

    def save(self):
        _CCGData(key=self.key,
                ccg=self.ccg[0],
                ccg_null=self.ccg_null[0] if self.ccg_null is not None else None,
                pval=self.pval[0] if self.pval is not None else None,
                qval=self.qval[0] if self.qval is not None else None,
                root =self.save_path()
        ).save()

    def save_segment(self, seg_name: str):
        seg_idx = self.segment_index(seg_name)
        src = self._src(seg_name)
        _CCGData(key=self.key,
                ccg=self.ccg[seg_idx],
                ccg_null=self.ccg_null[seg_idx] if self.ccg_null is not None else None,
                pval=self.pval[seg_idx] if self.pval is not None else None,
                qval=self.qval[seg_idx] if self.qval is not None else None,
                root=src.data_dir(self.key.resolution)
        ).save()
        src.save()

    def drop_segment(self, seg_name: str):
        """Drop segment from memory (index 0 is permanent)."""
        if seg_name not in self.segment_names:  # not attached in this resolution → nothing to drop
            return
        i = self.segment_index(seg_name)
        if i != 0:
            for _ in ('ccg', 'ccg_null', 'pval', 'qval'):
                a = getattr(self, _)
                if a is not None:
                    setattr(self, _, np.delete(a, i, axis=0))
            self.sources.pop(seg_name, None)

    def delete_segment(self, seg_name: str):
        """Remove segment from disk and memory."""
        src = CCGSourceConfig(key=self.key.change(segment=seg_name))
        src._root = str(self._root)
        if seg_name in self.sources:
            self.drop_segment(seg_name)
        for res in _CCG_RESOLUTION:
            shutil.rmtree(src.data_dir(res), ignore_errors=True)
        _Path(src.save_path() + '.json').unlink(missing_ok=True)

    def load_segment(self, seg_name: str):
        print("[load_segment]", str(self.key))
        src = CCGSourceConfig(key=self.key.change(segment=seg_name))
        src._root = str(self._root)
        src.load()
        d = src.data_dir(self.key.resolution)
        seg = _CCGData(key=self.key, root=str(d))
        seg.load(path=d)
        self.attach_segment(src, seg, save=False)

    def __str__(self):
        shape = self.ccg.shape if self.ccg is not None else None
        return (f"CCGData(session={getattr(self.key,'session','?')}, "
                f"resolution={getattr(self.key,'resolution','?')}, shape={shape})")

    def to_save_name(self):
        return f"{_san(self.key.session)}.{_san(self.key.resolution)}"

    @staticmethod
    def from_save_name(name):
        session, resolution = name.split('.')
        return Key(session=session, resolution=resolution)

    def save_path(self) -> str:
        return self._root / 'ccg' / 'ccgdata' / self.to_save_name()

    def _src(self, seg_name) -> 'CCGSourceConfig':
        """Segment ``CCGSourceConfig`` with project root set."""
        src = self.sources[seg_name]
        src._root = str(self._root)
        return src

    def pair(self, seg_idx: int, ref: int, tgt: int) -> tuple:
        """Per-bin traces for one pair on one segment."""
        def sl(a):
            return None if a is None else a[seg_idx, ref, tgt, :]
        return (sl(self.ccg), sl(self.ccg_null), sl(self.pval),
                sl(self.pval_corrected), sl(self.qval))

    def mean_ccg(self, ref: int, tgt: int, seg_idx: int) -> float:
        ccg = self.ccg
        return float(np.mean(ccg[seg_idx, ref, tgt, :]))

    def min_pval(self, ref: int, tgt: int, seg_idx: int) -> float:
        if self.pval is None:
            return 1.0
        lb, ub = int(self.conf.min_lag_bin), int(self.conf.max_lag_bin)
        sl = self.pval[seg_idx, ref, tgt, lb:ub]
        if sl.size == 0:
            return 1.0
        m = float(np.nanmin(sl))
        return m if np.isfinite(m) else 1.0


class CCGDataset(AnalysisDataset, Cacheable):
    """Experiment CCG store, pointers, and compute/load API."""

    _ccg: dict[CCGData]
    ptr: dict[CCGPointer] 
    src_conf: CCGSourceConfig
    conf: CCGConfig
    nd: NeuronsDataset

    def __init__(self, conf: CCGConfig, nd=None, src_conf=None, save_path=DATA_ROOT):
        if conf is None:
            raise ValueError("CCGDataset requires a CCGConfig — conf must not be None")
        super().__init__(conf)
        self.conf_meta = {
            'resolution':['lowres', 'highres']
            }
        self.save_path = os.path.join(save_path, f"project_{self.conf.name}")
        self.conf._root = save_path
        self.nd = nd
        self._ccg = {}
        self.ptr = {}
        self.src_conf = src_conf
        if self.src_conf is not None:
            self.src_conf._root = save_path
        self.cache = SessionMemoryCache(self._ccg)
        self._applied_norm_methods: list[str] | None = None
        self.conn_strength_config: ConnStrengthConfig | None = None
        self.conn_strength: dict = {}
        if src_conf is not None:
            return
        CCGPointer.load(self.ptr, self.conf, root=self.save_path)

    def ccg_for(self, key: Key):
        """Lazy-load ``CCGData`` for ``key`` (segments on dim0)."""
        sk = key.cd()
        data = self._ccg.get(sk)
        if data is None:
            self.get_ccg(sk)
            data = self._ccg.get(sk)
        if not data.has_segment(key.segment):
            self._load_segment(key)
        return data

    def pair_slices(self, key: Key):
        """Per-pair CCG slices for ``key.segment`` (label resolved against this resolution's array)."""
        data = self.ccg_for(key)
        if data is None or data.ccg is None:
            return None
        seg = self.segment_index(key, key.segment)
        if seg >= data.ccg.shape[0]:   # segment not computed at this resolution
            return None
        return data.pair(seg, key.ref, key.tgt)

    def live_sessions(self) -> set:
        """Sessions with in-memory ``CCGData``."""
        return {str(k.session) for k in self._ccg}

    def missing_sessions(self) -> list:
        """``nd`` sessions with no CCG pointers on disk; warns if any."""
        on_disk = {str(k.session) for k in self.ptr}
        missing = [name for s in self.nd._sessions
                   if (name := s.session_name) not in on_disk]
        if missing:
            print(f"[CCGDataset] {len(missing)} of {len(self.nd._sessions)} sessions "
                  f"have no CCGs computed yet: {missing}")
        return missing

    @property
    def ccg_dir(self):
        return os.path.join(self.save_path, "ccg", "ccgdata")

    def check_file_integrity(self, file_path):
        pass

    def _list_dir(self, dir, decode):
        out = []
        for d in glob.glob(os.path.join(dir, '*')):
            if os.path.isdir(d):
                out.append(decode(os.path.basename(d)))
        return out

    @property
    def data_root(self):
        """Parent of all ``project_*`` dirs (dataset-family root)."""
        return os.path.dirname(self.save_path)

    @property
    def selections_dir(self):
        return os.path.join(self.save_path, "selections")

    @property
    def stats_results_dir(self):
        return os.path.join(self.save_path, "stats_results")

    @property
    def custom_dir(self):
        return os.path.join(self.save_path, "custom_ccg")

    @property
    def custom_config_dir(self):
        return os.path.join(self.custom_dir, "config")

    @property
    def custom_data_dir(self):
        return os.path.join(self.custom_dir, "ccgdata")

    def saved_sessions(self):
        return self._list_dir(self.ccg_dir, CCGData.from_save_name)

    def saved_customs(self):
        """Custom segment keys from on-disk meta json."""
        stems = [_Path(f).stem for f in glob.glob(os.path.join(self.custom_config_dir, '*.json'))]
        return [CCGSourceConfig.stem_to_key(s) for s in stems]
        
    def by_session(self, session) -> dict:
        """In-memory ``CCGData`` for one session, keyed by ``Key``."""
        return {k: v for k, v in self._ccg.items() if str(k.session) == str(session)}

    def by_resolution(self, resolution) -> dict:
        """In-memory ``CCGData`` at one resolution."""
        return {k: v for k, v in self._ccg.items() if k.resolution == resolution}

    def by_segment(self, segment) -> dict:
        """In-memory ``CCGData`` with matching segment label."""
        return {k: v for k, v in self._ccg.items() if k.segment == segment}

    def available_resolutions(self) -> list:
        """Resolutions available in memory or on disk."""
        return self.conf_meta['resolution']
    
    @property
    def available_sessions(self) -> set:
        """All sessions in memory or on disk."""
        return self.live_sessions() | self.saved_sessions()

    def check_all_sessions_have_all_resolutions(self, sessions=None) -> dict:
        """``{session: [missing resolutions]}``."""
        if sessions is None:
            sessions = self.available_sessions
        gaps = {}
        for s in sessions:
            missing = []
            for res in self.available_resolutions():
                key = Key(session=s, resolution=res)
                probe = CCGData(key=key, conf=self.conf.at(res), ccg=None, ccg_null=None,
                                pval=None, qval=None, root=self.save_path)
                if key not in self._ccg and not probe.is_saved:
                    missing.append(res)
            if missing:
                gaps[s] = missing
        return gaps

    @property
    def applied_norm_methods(self) -> list[str] | None:
        return self._applied_norm_methods

    def apply_ccg_transform_for(self, key, active_norms):
        data = self.ccg_for(key).copy()
        if not active_norms:
            return data.ccg, data.ccg_null
        neurons = self.nd.neurons_for(key)
        refs, tgts = data._ref_tgt_grid()
        data.ccg = data.ccg.astype(float)          # norms yield fractions; int counts would truncate
        data.ccg_null = data.ccg_null.astype(float)
        for seg, label in enumerate(data.segment_names):
            data.ccg[seg], data.ccg_null[seg] = CCGNorm.apply(
                data.ccg[seg], data.ccg_null[seg], refs, tgts, active_norms, neurons,
                custom_time_hours=self.time_hours_for(key, label))
        return data.ccg, data.ccg_null

    # PATCH: 'all' has no CCGSourceConfig, so it falls back to session extent — seg 0 should carry its own
    def time_hours_for(self, key, label) -> float | None:
        """Recording hours for a segment — TIME norm divisor."""
        src = self.source_config(key, label)
        if src is not None and src.total_time_hours is not None:
            return float(src.total_time_hours)
        neurons = self.nd.neurons_for(key)
        return (float(neurons.t_stop) - float(neurons.t_start)) / 3600.0

    def apply_ccg_transform(self, active_norms):
        out = copy.deepcopy(self)
        for key, data in out._ccg.items():
            data.ccg, data.ccg_null = out.apply_ccg_transform_for(key, active_norms)
        return out

    def get_conn_strength_for(self, key, active_norms, cfg: ConnStrengthConfig) -> np.ndarray:
        """Connection-strength grid ``[seg, ref, tgt]`` after norms — batch call of the per-pair chain."""
        data = self.ccg_for(key)
        refs, tgts = data._ref_tgt_grid()
        seg = self.segment_index(key, key.segment)
        return ConnectionStrength.compute(
            data.ccg[seg], data.ccg_null[seg], refs, tgts, data.conf,
            metric=cfg.cs_metric, method=cfg.baseline_method, active_norms=active_norms,
            neurons=self.nd.neurons_for(key),
            custom_time_hours=self.time_hours_for(key, key.segment),
            excitability=key.excitability)

    def available_segments(self, key=None):
        """Computed segment labels on disk (ccgdata); ``key=None`` → project-wide, else that session."""
        keys = self._list_dir(self.custom_data_dir,
                              lambda b: CCGSourceConfig.stem_to_key(b.rsplit('.', 1)[0]))
        names = ['all']
        for sk in keys:
            nm = str(sk.segment)
            if (key is None or str(sk.session) == str(key.session)) and nm not in names:
                names.append(nm)
        return names

    def segment_names(self, key) -> list:
        """In-memory segment labels (dim0 stack order) for *key*'s session."""
        return self.ccg_for(key).segment_names

    def n_segments(self, key) -> int:
        return self.ccg_for(key).n_segment

    def segment_index(self, key, label) -> int:
        data = self.ccg_for(key)
        if label and label != 'all' and label not in data.segment_names \
                and label in self.available_segments(key):
            self._load_segment(key.change(segment=label))
        return data.segment_index(label)

    def segment_name(self, key, idx) -> str:
        return self.ccg_for(key).segment_name(idx)

    def source_config(self, key, label) -> 'CCGSourceConfig | None':
        return self.ccg_for(key).sources.get(label)

    def check_segment_integrity(self, key, label) -> dict:
        """TODO: per-resolution npy completeness for a segment."""
        raise NotImplementedError

    def ccg_config_relpath(self, key) -> str:
        """TODO: project-relative ``CCGConfig`` path for segment meta."""
        raise NotImplementedError

    def save_batch_request(self, request: 'CCGBatchRequest', kind: str = 'suggest') -> None:
        """TODO: persist batch request (`suggest` capped list or `history` log)."""
        raise NotImplementedError

    def list_batch_requests(self, kind: str = 'suggest') -> list:
        """TODO: load batch requests (newest first)."""
        raise NotImplementedError

    def delete_batch_request(self, request: 'CCGBatchRequest') -> None:
        """TODO: remove one saved batch request."""
        raise NotImplementedError

    def _request_sessions(self, spec: 'CCGBatchRequest') -> list:
        """Sessions targeted by ``spec``: the explicit picker list, else ``scope``."""
        # every session nd knows (not just those already on disk — missing ones lazy-compute)
        all_sess = sorted(str(k.session) for k in self.nd.session_keys())
        want = {str(s) for s in (spec.sessions or []) if str(s).lower() != 'all'}
        if want:
            return [s for s in all_sess if s in want]
        return all_sess if str(spec.scope).lower() == 'all' else [str(spec.scope)]

    def parse_ccg_batch_request(self, spec: 'CCGBatchRequest') -> tuple:
        """Expand batch request → ``(work, skipped)`` segment configs."""
        from neuropy.core.intervals import IntervalOp
        work, skipped = [], []
        for sess in self._request_sessions(spec):
            key = Key(session=str(sess))
            # resolve window against this session's bounds
            t0, t1 = self.nd.resolve_time(key, spec.t0), self.nd.resolve_time(key, spec.t1)
            t_start, t_end = self.nd.session_bounds(key)
            if not (t_start <= t0 < t1 <= t_end):
                skipped.append((sess, f"window {t0:.0f}–{t1:.0f}s outside session {t_end:.0f}s"))
                continue
            # per-session overlap ('%' needs window length)
            overlap = CCGBatchRequest.resolve_overlap_sec(t0, t1, spec.overlap_raw, spec.overlap_unit)
            before = len(work)
            n, name, fs = max(1, spec.n_splits), str(spec.name), list(spec.filter_state)
            active, _ = self.nd.resolve_intervals(key, t0, t1, fs) if fs else (None, None)
            if spec.split_mode == 'equal_effective':  # cut the active-time axis, not the real-time axis
                chunks = IntervalOp.partition_effective(active, n, name) if active else []
            else:
                if active:   # raw_span with a filter: split the filtered span, not the full window
                    t0, t1 = active[0][0], active[-1][1]
                chunks = IntervalOp.partition(t0, t1, n, overlap, name)
            for c0, c1, cname in chunks:  # keep chunks that have real active intervals
                src = CCGSourceConfig(key=key.change(segment=cname), t0=c0, t1=c1,
                                      overlap_sec=overlap, filter_state=fs)
                intervals, active_dur = self.nd.resolve_intervals(src.key, src.t0, src.t1, src.filter_state)
                if intervals or active_dur:
                    src.active_duration = active_dur
                    work.append(src)
            if len(work) == before:
                names = '∩'.join(t['name'] for t in fs)
                skipped.append((sess, f"no '{names}' overlap"))
        return work, skipped

    def compute_segment(self, key, src: 'CCGSourceConfig', neurons_slice) -> 'CCGData':
        """Compute one segment from pre-sliced neurons (no ``cd`` mutation)."""
        src.firing_rates = neurons_slice.firing_rate
        return self._compute_ccg_data(key, neurons_slice, np.arange(neurons_slice.n_neurons),
                                      self.conf.at(key.resolution))

    def attach_segment(self, key, src: 'CCGSourceConfig', seg: 'CCGData'):
        """Append computed segment to session dim0 (main thread)."""
        data = self.ccg_for(key.cd())   # base array; key.segment is what we are attaching
        data.attach_segment(src, seg)

    def _drop_segment(self, key) -> None:
        self.ccg_for(key.cd()).drop_segment(key.segment)   # base array; drop_segment no-ops if absent

    def drop_segment(self, keys, resolutions=None) -> None:
        for key in keys:
            for res in resolutions or self.available_resolutions():
                self._drop_segment(key.change(resolution=res))

    def _load_segment(self, key) -> None:
        """Load ``key.segment`` at ``key.resolution``; compute it when that resolution has no data."""
        data = self.ccg_for(key.cd())   # cd() drops segment → no recursion into this method
        try:
            data.load_segment(key.segment)
        except FileNotFoundError:
            src = CCGSourceConfig(key=key)
            src._root = self.save_path
            src.load()
            neurons_slice, _ = self.nd.sliced_neurons_for(src)
            data.attach_segment(src, self.compute_segment(key, src, neurons_slice))

    def load_segment(self, keys, resolutions=None) -> None:
        for key in keys:
            for res in resolutions or self.available_resolutions():
                self._load_segment(key.change(resolution=res))

    def _delete_segment(self, key) -> None:
        self.ccg_for(key).delete_segment(key.segment)

    def delete_segment(self, keys, resolutions=None) -> None:
        """Remove segment on disk across resolutions (main thread)."""
        for key in keys:
            for res in resolutions or self.available_resolutions():
                self._delete_segment(key.change(resolution=res))

    def _try_load_cached(self, conv, key=None) -> bool:
        """Load cached CCG + pointers; True if complete."""
        if self.load(key) != 'loaded':
            return False
        if CCGPointer.load(self.ptr, self.conf, root=self.save_path) == 'loaded':
            for k in self._ccg:
                self.cache.touch(k)
            return True
        print("[CCGDataset] re-running significance detection.")
        for k, cd in self._ccg.items():
            self._rerun_significance(k, cd, conv)
        self._save_pointers()
        self.conf.save()
        for k in self._ccg:
            self.cache.touch(k)
        return True

    def _missing_keys(self, keys: list, resolution: str) -> list:
        """Try disk load per key; return keys still missing."""
        missing = []
        for k in keys:
            rk = k.change(resolution=resolution)
            if rk in self._ccg:
                self.cache.touch(rk)
            elif self.load(k.change(resolution=resolution)) == 'loaded':
                self.cache.touch(rk)
            else:
                missing.append(k)
        return missing

    def _compute_ccg_data(self, key, neurons, neuron_inds, conf,
                          start_end_times=None) -> 'CCGData':
        """Compute CCG + baseline; result not stored on ``cd``."""
        print(f"[CCGDataset] {key.session} {key.resolution} "
              f"bin={conf.bin_size*1e3:.2f}ms …", flush=True)
        ccg = correlations.spike_correlations(
            neurons=neurons, neuron_inds=neuron_inds,
            bin_size=conf.bin_size, window_size=conf.duration,
            use_acceleration=conf.use_acceleration, symmetrize=conf.symmetrize_ccg,
            start_end_times=start_end_times,
        )
        if ccg.ndim == 3:  # 'all' mode (no segments) → add n_seg axis; CCGData is 4D [seg,ref,tgt,bin]
            ccg = ccg[None]
        pvals, pred, qvals = EranConv._conv(
            ccg, W=conf.conv_window / conf.bin_size, wintype="gauss", hollow_frac=None)
        print(f"[CCGDataset]   → shape {ccg.shape}")
        return CCGData(key=key, conf=conf, ccg=ccg, ccg_null=pred,
                       pval=pvals, qval=qvals, root=self.save_path)

    def get_ccg(self, key=None):
        """Load or compute whole-session CCG (seg0 ``full``); auto-save."""
        resolution = key.resolution if key is not None else 'lowres'
        highres = resolution == 'highres'
        conf = self.conf.at(resolution)
        conv = EranConv(self.conf)
        et = None
        keys = [key] if key is not None else self.nd.session_keys()

        if not highres and self._try_load_cached(conv, key=key):
            return
        missing = self._missing_keys(keys, resolution)
        if not missing:
            return

        computed = []
        for k in missing:
            if highres:
                nd = self.nd.neurons_for(k)
                hi_key = k.change(resolution='highres')
                _set = et[['start', 'stop']].values.T if et is not None else None
                self._ccg[hi_key] = self._compute_ccg_data(
                    hi_key, nd, np.arange(nd.n_neurons), conf, start_end_times=_set)
                self._rerun_significance(hi_key, self._ccg[hi_key], conv=None, skip_ptr=True)
                self.cache.touch(hi_key)
                computed.append(hi_key)
            else:
                self._compute_and_store(key=k, conv=conv, edge_times=et)
                computed.append(k)

        self.save(keys=computed)
        if not highres:
            for sess in {str(k.session) for k in missing}:
                self._save_pointers(session=sess)
        for k in self._ccg:
            self.cache.touch(k)

    def find(self, query: str, *, type_label: str = '', strict: bool = True):
        """First ``Key`` matching ``query`` session; ``type_label`` pins conn type; ``strict=False`` returns None instead of raising."""
        if query is None:                             # only saved-restore passes None
            return None
        live = {str(k.session) for k in self.nd.session_keys()}   # nd owns session identity
        if not (matches := [                          # '' matches every session → first key
                k for k in self.ptr.keys()
                if k.session and str(k.session) in live
                and query.replace('_', '').replace(' ', '').lower()
                in k.session.replace('_', '').replace(' ', '').lower()]):
            if strict:
                raise KeyError(f"No CCG session matching {query!r}")
            return None
        if type_label:
            hit = next((k for k in matches if k.type_label() == type_label), None)
            if hit is not None:
                return hit
        for ei, ct in self.conf.conn_types:
            hit = next((k for k in matches if k.excitability == ei and tuple(k.conn_type or ()) == tuple(ct)), None)
            if hit is not None:
                return hit
        return matches[0]

    def _summary(self, key) -> str:
        """Log line: significant pair counts for a session."""
        neurons = self.nd.neurons_for(key)
        nd_attrs = {f: getattr(key, f) for f in key.__dataclass_fields__
                    if getattr(key, f) is not None}

        def belongs(k):
            return all(getattr(k, attr, None) == val for attr, val in nd_attrs.items())

        def n_pairs(ptr):
            return ptr.n_pairs if ptr is not None else 0

        s = f"======={key.session}=======\n"
        for ntype in getattr(getattr(self.nd, 'conf', None), 'neuron_types', []):
            s += f"{ntype}={neurons.get_neuron_type(ntype).n_neurons} "
        s += "\n"

        N_E = sum(n_pairs(ptr) for k, ptr in self.ptr.items()
                  if belongs(k) and k.excitability == 'E')
        N_I = sum(n_pairs(ptr) for k, ptr in self.ptr.items()
                  if belongs(k) and k.excitability == 'I')
        printstr = f"{'session':10}: E/I {N_E:03d}/{N_I:03d} | "
        for EI, (ref, tgt) in self.conf.conn_types_labeled:
            N = next((n_pairs(ptr) for k, ptr in self.ptr.items()
                      if belongs(k) and k.excitability == EI
                      and tuple(getattr(k, 'conn_type', ()) or ()) == (ref, tgt)), 0)
            printstr += f"{ref}-{tgt}/{EI} {N:02d} | "
        printstr += '\n'
        return s + printstr

    def _rerun_significance(self, nd_key, ccg_data, conv, skip_ptr=False):
        """Re-run EranConv on loaded data; ``skip_ptr`` for highres."""
        conf = ccg_data.conf
        ccg = ccg_data.ccg
        n_bins = ccg.shape[-1]
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
        W = max(1, int(round(conf.conv_window / bin_size_eff)))

        if skip_ptr:
            pvals, pred, qvals = EranConv._conv(ccg, W=W)
            sig, _ = _multiple_correction(pvals, alpha=conf.alpha,
                                                   method=conf.multiple_correction)
            ccg_data.ccg_null = pred
            ccg_data.pval = pvals
            ccg_data.qval = qvals
            ccg_data.set_arrays()
            n_sig = int(sig.any(axis=-1).sum()) if sig is not None else 0
            print(f"[EranConv] {nd_key} shape={ccg.shape} W={W} → {n_sig} sig pair-segs")
            return

        neurons = self.nd.neurons_for(nd_key)
        pvals, pred, qvals, (inds_E, inds_I), printstr = conv.eranconv(
            neurons_key=nd_key, ccg=ccg, edge_times=None,
            neuron_type=neurons.neuron_type, conf=self.conf)
        ccg_data.ccg_null = pred
        ccg_data.pval = pvals
        ccg_data.qval = qvals
        ccg_data.set_arrays()
        self._attr_append(nd_key, self._build_ptrs(nd_key, inds_E, inds_I), 'ptr')
        print(printstr)

    def _build_ptrs(self, key, inds_E: dict, inds_I: dict) -> dict:
        """Build ``CCGPointer`` dict from eranconv index maps."""
        ptrs = {}
        for EI, inds_map in (('E', inds_E), ('I', inds_I)):
            for conn_type, inds in inds_map.items():
                ccg_key = key.add(conn_type=conn_type, excitability=EI)
                ptrs[ccg_key.ptr()] = CCGPointer(key=ccg_key, conf=self.conf,
                                                 inds=inds, root=self.save_path)
        return ptrs

    def _compute_and_store(self, key, conv, edge_times):
        """Compute CCG, significance, and store in ``self._ccg`` / ``self.ptr``."""
        print("EranConv significant pairs")

        neurons = self.nd.neurons_for(key)
        conf = self.conf.at(key.resolution)

        ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(neurons.n_neurons),  # all
            bin_size=conf.bin_size,
            window_size=conf.duration,
            use_acceleration=conf.use_acceleration,
            symmetrize=conf.symmetrize_ccg,
            start_end_times=edge_times,
        )
        if ccg.ndim == 3:  # 'all' mode (no segments) → add n_seg axis; CCGData is 4D [seg,ref,tgt,bin]
            ccg = ccg[None]

        pvals, pred, qvals, (inds_E, inds_I), printstr = conv.eranconv(
            neurons_key=key, ccg=ccg, edge_times=edge_times,
            neuron_type=neurons.neuron_type, conf=conf)

        ccg_data = CCGData(
            key=key, conf=conf,
            ccg=ccg, ccg_null=pred,
            pval=pvals, qval=qvals,
            root=self.save_path,
        )

        self._ccg[key] = ccg_data
        self._attr_append(key, self._build_ptrs(key, inds_E, inds_I), 'ptr')

        print(self._summary(key))

    def save_path(self, suffix='') -> str:
        """Project data root for save/load paths."""
        return self.conf.save_path(suffix=suffix)

    def _save_pointers(self, session=None) -> None:
        if not self.ptr:
            print("[CCGDataset] no pointers to save.")
            return
        by_sess: dict = defaultdict(dict)
        for p in self.ptr.values():
            if session is None or str(p.key.session) == str(session):
                by_sess[p.key.session][str(p.key)] = p
        n = 0
        for sess, bundle in by_sess.items():
            path = next(iter(bundle.values())).save_path() + '.hkl'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            hkl.dump(bundle, path)
            n += len(bundle)
        print(f"[CCGDataset] saved {n} pointers in {len(by_sess)} session file(s)")

    def save(self, keys=None) -> None:
        """Save in-memory ``CCGData`` (all or ``keys``)."""
        for k in (list(self._ccg) if keys is None else list(keys)):
            self._ccg[k].save()
        self.conf.save()

    def load(self, key: Key = None) -> str:
        """Load from disk; returns ``loaded`` | ``missing`` | ``stale``."""
        resolution = key.resolution if key is not None else 'lowres'
        keys = [key] if key is not None else self.nd.session_keys()
        if not keys:
            return 'missing'
        loaded = 0
        for sk in keys:
            load_key = sk.change(resolution=resolution)
            cd = CCGData(key=load_key, conf=self.conf.at(resolution),
                         ccg=None, ccg_null=None, pval=None, qval=None,
                         root=self.save_path)
            if not os.path.isdir(cd.save_path()):     # unit dir of memmapped .npy
                continue
            try:
                cd.load()
                self._ccg[load_key] = cd
                self.cache.touch(load_key)
                loaded += 1
            except Exception as exc:
                print(f"[CCGDataset] load failed {sk}: {exc}")
        if loaded:
            print(f"[CCGDataset] {resolution} loaded ({loaded}/{len(keys)} session(s))")
        return 'loaded' if loaded == len(keys) else ('stale' if loaded else 'missing')


def _multiple_correction(pvals: np.ndarray, alpha: float, method: str = 'bonferroni') -> tuple:
    """Multiple-test correction over lag bins; returns ``(sig, p_correct)``."""
    if method == 'bonferroni':
        corrected = np.minimum(pvals * pvals.shape[-1], 1.0)
        return corrected <= alpha, corrected
    significance = np.zeros_like(pvals, dtype=bool)
    p_correct = np.ones_like(pvals, dtype=float)
    for idx in np.ndindex(pvals.shape[:-1]):
        s, pc, _, _ = multipletests(pvals[idx], alpha=alpha, method=method)
        significance[idx] = s
        p_correct[idx] = pc
    return significance, p_correct


class EranConv:
    """Convolution baseline significance (Stark & Abeles 2009)."""

    def __init__(self, conf):
        self._pvals = []
        self._qvals = []
        self.significant = []  # final filtering results
        self.conf = conf

    @staticmethod
    def _conv(ccg, W=5, wintype="gauss", hollow_frac=None):
        """Hollow convolution baseline; returns ``pvals``, ``pred``, ``qvals``."""
        if len(ccg.shape) == 1:
            ccg = ccg[np.newaxis, ...]

        assert wintype in ["gauss", "rect", "triang"]
        assert W <= ccg.shape[-1]

        # Auto-assign appropriate hollow fraction if not specified
        # generate window
        # get center indices of window
        if wintype == "gauss":
            hollow_frac = hollow_frac or 0.6
            sigma = W / 2
            W = int(6 * sigma + (2 if W % 2 else 1))
            center = int(3 * sigma + (0.5 if W % 2 else 0))
            window = windows.gaussian(W, std=sigma) / (2 * np.pi * sigma)
        elif wintype == "rect":
            hollow_frac = hollow_frac or 0.42
            if W % 2 == 0:
                W += 1
            center = W // 2
            window = windows.boxcar(W)
        elif wintype == "triang":
            hollow_frac = hollow_frac or 0.63
            W = 2 * W + (-1 if W % 2 else 1)
            center = W // 2
            window = windows.triang(W)

        # hollow and normalize window
        window[center] *= (1 - hollow_frac)
        window /= np.sum(window)
        # padding
        ccg_pad = np.concatenate(
            [ccg[..., :W][..., ::-1], ccg, ccg[..., -W:][..., ::-1]], axis=-1)

        # convolve window with ccg
        pred = ndimage.convolve1d(ccg_pad, window, axis=-1)
        pred = pred[..., W:-W]

        # mid-p Poisson test: P( val<=pred ) + half of P ( val==pred )
        pvals = 1 - poisson.cdf(ccg - 1, pred) - poisson.pmf(ccg, pred) * 0.5
        qvals = 1 - pvals
        return pvals, pred, qvals

    def spkcount_mask(self, ccg):
        min_bin = self.conf.min_spkcnt_bin
        max_bin = self.conf.max_spkcnt_bin
        threshold = self.conf.min_spkcount
        # Mean spike count in lag window (center bin may be zero).
        pair_inds = np.argwhere(
            ccg[..., min_bin:max_bin].mean(axis=-1) >= threshold)
        # NOTE right now it's the same criteria for excitation/inhibition
        return pair_inds

    def significance_mask(self, p, excitability):
        """Pair indices with significant peaks (E: lag window; I: trough + neighbor)."""
        conf = self.conf
        method = conf.multiple_correction if conf.multiple_correction is not None else 'bonferroni'

        if excitability == 'E':
            sig, self._pvals = _multiple_correction(p, conf.alpha, method=method)
            # At least one corrected-significant bin in the excitatory test window.
            has_valid_peak = sig[..., conf.min_lag_bin:conf.max_lag_bin].any(axis=-1)
            pair_inds = np.argwhere(has_valid_peak)
        elif excitability == 'I':
            sig1, self._qvals = _multiple_correction(p, conf.alpha, method=method)
            sig2, _ = _multiple_correction(p, conf.alpha2, method=method)
            # Bin must be significant at alpha AND have a neighbour at alpha2.
            neighbor = sig1 & (np.roll(sig2, 1, -1) | np.roll(sig2, -1, -1))
            pair_inds = np.argwhere(neighbor.any(-1))
        else:
            raise ValueError(f"Unknown excitability: {excitability!r}")
        # remove autocorrelations
        return pair_inds[pair_inds[:, -2] != pair_inds[:, -1]]

    def _cell_type_mask(self, pair_inds, neuron_type, conn_types):
        if pair_inds.ndim == 1:
            pair_inds = pair_inds.reshape(0, 2)
        sig_pairs = {}
        if not _hasvalue(pair_inds):
            for ct in conn_types:
                sig_pairs[ct] = None
            return sig_pairs

        for ct in conn_types:
            inds = np.where(
                np.isin(pair_inds[:, -2], np.where(neuron_type == ct[0])) &
                np.isin(pair_inds[:, -1], np.where(neuron_type == ct[1])))[0]
            sig_pairs[ct] = pair_inds[inds] if inds.shape[0] else None
        return sig_pairs

    def eranconv(
        self,
        neurons_key: Key,
        ccg,
        edge_times: pd.DataFrame,
        neuron_type,
        conf: CCGConfig,
    ):
        """Full EranConv pass for one neurons key."""
        print("running eranconv (1st pass)")
        key = neurons_key
        self.conf = conf
        if edge_times is None:
            edge_times = pd.DataFrame([{
                'start': 0.0, 'stop': 0.0,
                'label': 'session',
                'total_time_hours': 0.0,
                'effective_time_hours': 0.0,
            }])
        self.n_segments = edge_times.shape[0]

        pvals, pred, qvals = EranConv._conv(ccg,
                                            W=self.conf.conv_window / self.conf.bin_size, # number of bins in conv window
                                            wintype="gauss",
                                            hollow_frac=None)

        def build_inds(p, EI, conn_types):
            rough_inds = SetOp.intersect(self.significance_mask(p, EI),
                                         self.spkcount_mask(ccg))
            inds = self._cell_type_mask(rough_inds, neuron_type, conn_types)
            return rough_inds, inds

        # [n_seg, n_pair, 2]
        rough_inds_E, inds_E = build_inds(pvals, 'E', conf.conn_types_E)
        rough_inds_I, inds_I = build_inds(qvals, 'I', conf.conn_types_I)

        self.significant = np.zeros(ccg.shape[:3], dtype=bool)
        for inds in [inds_E, inds_I]:
            if inds is None:
                continue
            for k, v in inds.items():
                if v is None:
                    continue
                self.significant[tuple(v.T)] = True

        printstr = ''
        for i, (_, edge_time) in enumerate(edge_times.iterrows()):
            N_totalE = int((rough_inds_E[:, 0] == i).sum())
            N_totalI = int((rough_inds_I[:, 0] == i).sum())
            printstr += f"{edge_time['label']:10}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "
            for EI, (ref, tgt) in self.conf.conn_types_labeled:
                inds = inds_E.get((ref, tgt)) if EI == 'E' else inds_I.get((ref, tgt))
                n = int((inds[:, 0] == i).sum()) if _hasvalue(inds) else 0
                printstr += f"{ref}-{tgt}/{EI} {n:02d} | "
            printstr += '\n'

        print("eranconv done")
        return pvals, pred, qvals, (inds_E, inds_I), printstr

