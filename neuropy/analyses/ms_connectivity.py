import glob
import json
import os
import dataclasses
from dataclasses import dataclass, field
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
from neuropy.analyses.neurons_dataset import Key, NeuronsDataset
from collections import defaultdict

_REPO_ROOT = _Path(__file__).resolve().parents[2]
DATA_ROOT = str(_REPO_ROOT / "data")

_CCG_RESOLUTION = {
    'lowres':  1e-3,    # 1 ms   — default, fast
    'highres': 1/3*1e-4,  # 0.1 ms — finer temporal resolution (must exceed 1/sample_rate)
}


class CCGConfig(Config):
    """CCG config. 'key' fields require full recompute; 'derived' fields only re-run EranConv."""

    _groups = {
        'key': ['name', 'resolution', 'bin_size', 'duration', 'conv_window',
                'conn_types', 'use_acceleration', 'symmetrize_ccg'],
        'derived': ['alpha', 'alpha2', 'min_lag', 'max_lag',
                    'min_spkcount', 'spkcnt_scope', 'multiple_correction'],
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
        self._root = DATA_ROOT  # set by CCGDataset to enforce path consistency

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
        """[(EI, conn_type), ...] — same order as conn_types_flat."""
        return list(self.conn_types)

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
        """data/project_{name}/ccg/config/{name}_{resolution}_{suffix}"""
        base = os.path.join(self._root, f"project_{self.name}", "ccg", "config",
                            f"{self.name}_{self.resolution}")
        return f"{base}_{suffix}" if suffix else base

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.save_path()}\n"
        return s


@dataclass(eq=False)
class CCGSourceConfig(Config):
    """Data-source parameters for a custom CCG segment."""
    name: str
    t0: float | str
    t1: float | str
    scope: str = ''
    created_from_session: str = ''
    sessions: list = field(default_factory=list)
    n_splits: int = 1
    overlap_sec: float = 0.0
    filter_state: dict = field(default_factory=dict)
    active_duration: float = None
    total_time_hours: float = None
    windows: list = field(default_factory=list)          # list[Epoch]; each Epoch = one CCG segment
    firing_rates: object = field(default=None, repr=False)  # ndarray [n_segs, n_neurons]
    tags: dict = field(default_factory=dict)             # e.g. {'kind': 'custom'}
    src_path: str = None

    _groups = {
        'key': ['name', 't0', 't1', 'filter_state'],
    }
    _JSON_KEYS = frozenset([
        'name', 't0', 't1', 'scope', 'created_from_session', 'sessions',
        'n_splits', 'overlap_sec', 'filter_state', 'active_duration',
        'total_time_hours', 'tags', 'src_path',
    ])

    def __post_init__(self):
        super().__init__()
        self._root = DATA_ROOT  # set by CCGDataset
        # Coerce t0/t1: sentinel strings ('start'/'end') stay str, else float
        for attr in ('t0', 't1'):
            v = getattr(self, attr)
            if not (isinstance(v, str) and v.lower() in ('start', 'end')):
                setattr(self, attr, float(v))
        # Normalize sessions: sorted list of non-None strings
        self.sessions = sorted(str(s) for s in (self.sessions or []) if s is not None)
        # Normalize n_splits / overlap_sec types
        self.n_splits = int(self.n_splits or 1)
        self.overlap_sec = float(self.overlap_sec or 0.0)
        # Normalize filter_state
        fs = self.filter_state or {}
        labels = fs.get('labels', {}) or {}
        self.filter_state = {
            'theme': str(fs.get('theme', 'segments')),
            'labels': {str(k): bool(v) for k, v in labels.items()},
        }
        if self.active_duration is None and isinstance(self.t0, (int, float)) and isinstance(self.t1, (int, float)):
            self.active_duration = self.t1 - self.t0
        if self.total_time_hours is None and self.active_duration is not None:
            self.total_time_hours = self.active_duration / 3600.0

    def __eq__(self, other) -> bool:
        if not isinstance(other, CCGSourceConfig):
            return NotImplemented
        return self.matches(other, 'key')

    def windows_to_edge_times(self) -> pd.DataFrame | None:
        """Pack self.windows (list[Epoch]) into an edge_times DataFrame."""
        if not self.windows:
            return None
        frames = [w._epochs for w in self.windows if hasattr(w, '_epochs')]
        if not frames:
            return None
        et = pd.concat(frames, ignore_index=True)
        if 'total_time_hours' not in et.columns:
            dur = (et['stop'] - et['start']) / 3600.0
            et['total_time_hours'] = dur
            et['effective_time_hours'] = dur
        return et

    def _key(self) -> tuple:
        fs = self.filter_state or {}
        labels = fs.get('labels', {}) or {}
        t0_key = str(self.t0) if isinstance(self.t0, str) else float(self.t0)
        t1_key = str(self.t1) if isinstance(self.t1, str) else float(self.t1)
        return (str(self.name), t0_key, t1_key,
                str(fs.get('theme', 'segments')),
                tuple(sorted((str(k), bool(v)) for k, v in labels.items())))

    def __hash__(self) -> int:
        return hash(self._key())

    def save_path(self, suffix='source') -> str:
        base = os.path.join(self._root, f"project_{self.name}", "ccg", "config",
                            f"{self.name}_{suffix}")
        return base


    @property
    def t0_sec(self) -> float:
        """Resolved numeric t0; raises if still a sentinel string."""
        if isinstance(self.t0, str):
            raise ValueError(f"t0='{self.t0}' is unresolved — call resolve(session) first")
        return float(self.t0)

    @property
    def t1_sec(self) -> float:
        """Resolved numeric t1; raises if still a sentinel string."""
        if isinstance(self.t1, str):
            raise ValueError(f"t1='{self.t1}' is unresolved — call resolve(session) first")
        return float(self.t1)

    @classmethod
    def deserialize(cls, d: dict, *, default_session: str = '') -> 'CCGSourceConfig':
        return cls(
            name=str(d.get('name', '')),
            t0=d.get('t0', 0.0),
            t1=d.get('t1', 0.0),
            filter_state=d.get('filter_state', {}),
            scope=str(d.get('scope', default_session)),
            created_from_session=str(d.get('created_from_session', default_session)),
            sessions=d.get('sessions', []),
            n_splits=d.get('n_splits', 1),
            overlap_sec=d.get('overlap_sec', 0.0),
            active_duration=d.get('active_duration'),
            total_time_hours=d.get('total_time_hours'),
        )


class CCGPointer(HklSavable):
    """Positional pointer to CCGData locations for significant pairs."""
    IGNORED_KEYS = ('conf', '_root', '_ignored_attrs')

    def __init__(
        self,
        key,
        inds,
        conf: CCGConfig = None,
        root: str = DATA_ROOT,
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
        return self._inds  # always (n, 3): [seg, ref, tgt]

    @property
    def inds2(self):
        return SetOp.unique(self._inds[:, 1:])  # unique (ref, tgt)

    @property
    def n_pairs(self):
        return self.inds2.shape[0]

    def get_segment(self, i: int) -> 'CCGPointer':
        return CCGPointer(
            key=self.key.add(segment=i),
            inds=self._inds[self._inds[:, 0] == i],
            conf=self.conf,
            root=self._root,
        )

    def split(self, n_segments: int) -> list['CCGPointer']:
        return [self.get_segment(i) for i in range(n_segments)]

    def save_path(self) -> str:
        base = os.path.join(self._root, f"project_{self.conf.name}", "ccg", "pointers",
                            f"{self.conf.name}_{self.conf.resolution}")
        return f"{base}_ccgpointers_{self.key.session}"

    @classmethod
    def load(cls, ptr: dict, conf: 'CCGConfig', root: str = DATA_ROOT) -> str:
        base = os.path.join(root, f"project_{conf.name}", "ccg", "pointers",
                            f"{conf.name}_{conf.resolution}")
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
                        ptr[p.key] = p
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


class CCGData(NpzSavable):
    """CCG arrays [n_seg, n_pair, n_bins] and p-values for one session."""
    IGNORED_KEYS = ('conf', 'key', '_root', '_ignored_attrs')

    def __init__(self, key, conf, ccg, ccg_null, pval, qval, root: str = DATA_ROOT):
        super().__init__(ignored_attrs=self.IGNORED_KEYS)
        self.key = key
        self.conf = conf
        self._root = root
        self.ccg = ccg
        self.ccg_null = ccg_null
        self.pval = pval
        self.qval = qval
        self.significant = None
        # Ensure all arrays are 4D [n_segs, n_neurons, n_neurons, n_bins]
        for attr in ('ccg', 'ccg_null', 'pval', 'qval'):
            a = getattr(self, attr, None)
            if a is not None and hasattr(a, 'ndim') and a.ndim == 3:
                setattr(self, attr, a[np.newaxis, ...])

    @property
    def n_segment(self):
        return self.ccg.shape[0]

    @property
    def pval_corrected(self):
        if self.pval is None or self.conf is None:
            return None
        _, pc = _multiple_correction(
            self.pval, self.conf.alpha, method=self.conf.multiple_correction)
        return pc

    def __str__(self):
        shape = self.ccg.shape if self.ccg is not None else None
        return (f"CCGData(session={getattr(self.key,'session','?')}, "
                f"resolution={getattr(self.key,'resolution','?')}, shape={shape})")

    def save_path(self, suffix='ccgdata') -> str:
        res = self.key.resolution or 'lowres'
        base = os.path.join(self._root, f"project_{self.conf.name}", "ccg", "ccgdata",
                            f"{self.conf.name}_{res}")
        return f"{base}_{suffix}_{self.key.session}__{res}"

    def mean_ccg(self, ref: int, tgt: int, seg_idx: int) -> float:
        ccg = self.ccg
        if seg_idx >= ccg.shape[0]:
            return float(np.mean(ccg[:, ref, tgt, :].sum(axis=0)))
        return float(np.mean(ccg[seg_idx, ref, tgt, :]))

    def min_pval(self, ref: int, tgt: int, seg_idx: int) -> float:
        if self.pval is None:
            return 1.0
        pval = self.pval
        lb, ub = int(self.conf.min_lag_bin), int(self.conf.max_lag_bin)
        if seg_idx >= pval.shape[0]:
            arr = pval[:, ref, tgt, :].reshape(-1)
        else:
            arr = pval[seg_idx, ref, tgt, :]
        sl = arr[lb:ub]
        if sl.size == 0:
            return 1.0
        m = float(np.nanmin(sl))
        return m if np.isfinite(m) else 1.0


class CCGDataset(AnalysisDataset, Cacheable):
    """CCGs and significance for an experiment. ptr: significant pairs; nd: source neurons."""

    ccg: dict[CCGData]
    ptr: dict[CCGPointer] 
    src_conf: CCGSourceConfig
    conf: CCGConfig
    nd: NeuronsDataset

    def __init__(self, conf: CCGConfig, nd=None, src_conf=None, save_path=DATA_ROOT):
        if conf is None:
            raise ValueError("CCGDataset requires a CCGConfig — conf must not be None")
        super().__init__(conf)
        self._save_path = save_path  # project folder; anchor for all subcomponent paths
        self.conf._root = save_path
        self.nd = nd
        self.ccg = {}   # Key(session, resolution) → CCGData
        self.ptr = {}
        self._seg_fr_cache = {}   # (nd_key, starts, stops) → per-segment firing-rate array
        self.src_conf = src_conf
        if self.src_conf is not None:
            self.src_conf._root = save_path
        self.cache = SessionMemoryCache(self.ccg)
        if src_conf is not None:
            return
        ptr_status = CCGPointer.load(self.ptr, self.conf, root=save_path)
        if ptr_status in ('missing', 'stale'):
            self.get_ccg()
            self._get_ccg_highres()
        else:
            self.load()   # ptr loaded → load CCGData arrays from disk
            if self.load('highres') != 'loaded':
                self._get_ccg_highres()

    def ccg_for(self, nd_key, resolution='lowres'):
        """Return CCGData for session *nd_key* at *resolution* ('lowres'|'highres')."""
        if nd_key is None:
            return None
        k = (nd_key if getattr(nd_key, 'resolution', None) == resolution
             else nd_key.change(resolution=resolution))
        return self.ccg.get(k)

    def edge_times_for(self, key) -> pd.DataFrame | None:
        """Segment timing table from src_conf windows."""
        if self.src_conf is None:
            return None
        return self.src_conf.windows_to_edge_times()

    def n_segments_for(self, key) -> int:
        et = self.edge_times_for(key)
        return int(et.shape[0]) if et is not None else 0

    def segment_names_for(self, key) -> list:
        et = self.edge_times_for(key)
        if et is None or 'label' not in et.columns:
            return []
        return list(et['label'].values)

    def segment_firing_rates(self, nd_key) -> 'np.ndarray | None':
        """Per-segment firing-rate table [n_seg, n_neuron] for one session (cached).
        rate[s, n] = neuron n's spikes within segment s / segment duration. Returns
        None when segment timing or neurons are unavailable (callers then fall back
        to the whole-session rate). Cache is keyed by the segment bounds, so a theme
        change that alters the segments recomputes automatically."""
        et = self.edge_times_for(nd_key)
        neurons = self.nd.data.get(nd_key) if self.nd is not None else None
        if et is None or neurons is None:
            return None
        starts = tuple(float(s) for s in et['start'].values)
        stops  = tuple(float(s) for s in et['stop'].values)
        sig = (nd_key, starts, stops)
        cached = self._seg_fr_cache.get(sig)
        if cached is not None:
            return cached
        fr = np.zeros((len(starts), neurons.n_neurons), dtype=float)
        for i, (t0, t1) in enumerate(zip(starts, stops)):
            fr[i] = neurons.time_slice(t0, t1).firing_rate
        self._seg_fr_cache[sig] = fr
        return fr

    def _try_load_cached(self, conv, nd_key=None) -> bool:
        """Load CCGData + pointers from disk. Returns True if fully loaded."""
        if self.load(nd_key=nd_key) != 'loaded':
            return False
        if CCGPointer.load(self.ptr, self.conf, root=self._save_path) == 'loaded':
            for k in self.ccg:
                self.cache.touch(k)
            return True
        print("[CCGDataset] re-running significance detection.")
        for k, cd in self.ccg.items():
            self._rerun_significance(k, cd, conv)
        self._save_pointers()
        self.conf.save()
        for k in self.ccg:
            self.cache.touch(k)
        return True

    def _missing_keys(self, keys: list, resolution: str) -> list:
        """Load from disk for each key; return keys not found in cache."""
        missing = []
        for k in keys:
            rk = k.change(resolution=resolution)
            if rk in self.ccg:
                self.cache.touch(rk)
            elif self.load(resolution, nd_key=k) == 'loaded':
                self.cache.touch(rk)
            else:
                missing.append(k)
        return missing

    def _compute_ccg_data(self, key, neurons, neuron_inds, conf,
                          start_end_times=None) -> 'CCGData':
        """Compute CCG + convolution baseline. Returns CCGData (not yet stored)."""
        print(f"[CCGDataset] {key.session} {key.resolution} "
              f"bin={conf.bin_size*1e3:.2f}ms …", flush=True)
        ccg = correlations.spike_correlations(
            neurons=neurons, neuron_inds=neuron_inds,
            bin_size=conf.bin_size, window_size=conf.duration,
            use_acceleration=conf.use_acceleration, symmetrize=conf.symmetrize_ccg,
            start_end_times=start_end_times,
        )
        pvals, pred, qvals = EranConv._conv(
            ccg, W=conf.conv_window / conf.bin_size, wintype="gauss", hollow_frac=None)
        print(f"[CCGDataset]   → shape {ccg.shape}")
        return CCGData(key=key, conf=conf, ccg=ccg, ccg_null=pred,
                       pval=pvals, qval=qvals, root=self._save_path)

    def get_ccg(self, nd_key=None, use_segments=True, resolution='lowres'):
        """Load or compute CCGs for one or all sessions."""
        if resolution == 'highres':
            self._get_ccg_highres(nd_key)
            return

        conv = EranConv(self.conf)
        et = self.src_conf.windows_to_edge_times() if self.src_conf else None
        keys = [nd_key] if nd_key is not None else list(self.nd.data.keys())

        if self._try_load_cached(conv, nd_key=nd_key):
            return

        missing = self._missing_keys(keys, 'lowres')
        if not missing:
            return
        for key in missing:
            self._compute_and_store(key=key, conv=conv, edge_times=et, use_segments=use_segments)
        self.save()
        self._save_pointers()
        self.conf.save()
        for k in self.ccg:
            self.cache.touch(k)

    def _get_ccg_highres(self, nd_key=None) -> None:
        conf = self.conf.copy(bin_size=_CCG_RESOLUTION['highres'], resolution='highres',
                              conv_window=1e-4)
        keys = [nd_key] if nd_key else list(self.nd.data.keys())
        missing = self._missing_keys(keys, 'highres')
        if not missing:
            return
        et = self.src_conf.windows_to_edge_times() if self.src_conf else None
        _set = et[['start', 'stop']].values.T if et is not None else None
        for k in missing:
            hi_key = k.change(resolution='highres')
            self.ccg[hi_key] = self._compute_ccg_data(
                hi_key, self.nd.data[k], np.arange(self.nd.data[k].n_neurons),
                conf, start_end_times=_set)
            self._rerun_significance(hi_key, self.ccg[hi_key], conv=None, skip_ptr=True)
            self.cache.touch(hi_key)
        self.save('highres')
        print(f"[get_ccg] highres done — {len(missing)} session(s), "
              f"{conf.bin_size*1e3:.2f} ms bins.")

    def get_ccg_custom(self, neurons_slice, *, has_highres: bool = False,
                       excitability: str = 'E') -> None:
        """Run CCG for self.src_conf time window; populate self.ccg with lo (and hi) CCGData."""
        src = self.src_conf
        n_neurons = neurons_slice.n_neurons
        neuron_inds = np.arange(n_neurons)
        conf = self.conf

        active = src.active_duration or (src.t1_sec - src.t0_sec)
        src.firing_rates = np.array(
            [len(st) for st in neurons_slice.spiketrains], dtype=float
        ) / max(active, 1e-9)
        for resolution, bin_size in _CCG_RESOLUTION.items():
            if resolution == 'highres' and not has_highres:
                continue
            k = Key(session=src.name, resolution=resolution)
            cd = self._compute_ccg_data(k, neurons_slice, neuron_inds,
                                        conf.copy(bin_size=bin_size, resolution=resolution))
            if excitability != 'E':
                cd.pval, cd.qval = cd.qval, None
            self.ccg[k] = cd

    def find(self, query: str):
        """Return the key of the first CCGPointer whose session matches *query*.
        Matching is case-insensitive and ignores underscores/spaces.
        Pass the returned key directly to
        launch_ccg_review(cd, cd.find("RatUDay2"))
        """
        q = query.replace('_', '').replace(' ', '').lower()
        # Collect all full keys from cd.ptr whose session matches
        matches = [
            k for k in self.ptr.keys()
            if k.session and q in k.session.replace('_', '').replace(' ', '').lower()
        ]
        if not matches:
            raise KeyError(f"No CCG session matching {query!r}")
        for ei, ct in self.conf.conn_types:
            hit = next((k for k in matches if k.excitability == ei and tuple(k.conn_type or ()) == tuple(ct)), None)
            if hit is not None:
                return hit
        return matches[0]

    def _summary(self, key) -> str:
        """Per-session pair-count table logged after compute or cache hit."""
        neurons = self.nd.data[key.nd()]
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
        """Run EranConv significance detection on already-loaded CCGData.

        skip_ptr=True: skip pointer building (used for highres where bin_size is
        inferred from array shape and ptr tracking is not needed).
        """
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
            n_sig = int(sig.any(axis=-1).sum()) if sig is not None else 0
            print(f"[EranConv] {nd_key} shape={ccg.shape} W={W} → {n_sig} sig pair-segs")
            return

        neurons = self.nd.data[nd_key]
        edge_times = self.edge_times_for(nd_key)
        pvals, pred, qvals, (inds_E, inds_I), printstr = conv.eranconv(
            neurons_key=nd_key, ccg=ccg, edge_times=edge_times,
            neuron_type=neurons.neuron_type, conf=self.conf)
        ccg_data.ccg_null = pred
        ccg_data.pval = pvals
        ccg_data.qval = qvals
        ccg_data.significant = conv.significant
        self._attr_append(nd_key, self._build_ptrs(nd_key, inds_E, inds_I), 'ptr')
        print(printstr)

    def _build_ptrs(self, key, inds_E: dict, inds_I: dict) -> dict:
        """Build CCGPointer dict from raw inds returned by eranconv."""
        ptrs = {}
        for EI, inds_map in (('E', inds_E), ('I', inds_I)):
            for conn_type, inds in inds_map.items():
                ccg_key = key.add(conn_type=conn_type, excitability=EI)
                ptrs[ccg_key] = CCGPointer(key=ccg_key, conf=self.conf,
                                           inds=inds, root=self._save_path)
        return ptrs

    def _compute_and_store(self, key, conv, edge_times, use_segments=True):
        """
        Run CCG and generate a convolution-based baseline for all neurons in my NeuronsDataset.
        Run significance tests.
        Store results in objects:
            self.ccg
            self.ptr

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        print("EranConv significant pairs")

        neurons = self.nd.data[key.nd()]

        _set = (edge_times[['start', 'stop']].values.T
                if (use_segments and edge_times is not None) else None)
        ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(neurons.n_neurons),  # all
            bin_size=self.conf.bin_size,
            window_size=self.conf.duration,
            use_acceleration=self.conf.use_acceleration,
            symmetrize=self.conf.symmetrize_ccg,
            start_end_times=_set,
        )

        pvals, pred, qvals, (inds_E, inds_I), printstr = conv.eranconv(
            neurons_key=key, ccg=ccg, edge_times=edge_times,
            neuron_type=neurons.neuron_type, conf=self.conf)

        ccg_data = CCGData(
            key=key, conf=self.conf,
            ccg=ccg, ccg_null=pred,
            pval=pvals, qval=qvals,
            root=self._save_path,
        )
        ccg_data.significant = conv.significant

        self.ccg[key] = ccg_data
        self._attr_append(key, self._build_ptrs(key, inds_E, inds_I), 'ptr')

        print(self._summary(key))

    def save_path(self, suffix='') -> str:
        """Project folder; subcomponent paths anchored here."""
        return self.conf.save_path(suffix=suffix)

    def _save_pointers(self) -> None:
        if not self.ptr:
            print("[CCGDataset] no pointers to save.")
            return
        by_sess: dict = defaultdict(dict)
        for p in self.ptr.values():
            by_sess[p.key.session][str(p.key)] = p
        base = os.path.join(self._save_path, f"project_{self.conf.name}", "ccg", "pointers",
                            f"{self.conf.name}_{self.conf.resolution}")
        for sess, bundle in by_sess.items():
            path = f"{base}_{sess}.hkl"
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            hkl.dump(bundle, path)
        print(f"[CCGDataset] saved {len(self.ptr)} pointers in {len(by_sess)} session file(s)")

    def save(self, resolution: str = 'lowres') -> None:
        target = {k: v for k, v in self.ccg.items() if k.resolution == resolution}
        if not target:
            print(f"[CCGDataset] {resolution}: nothing to save.")
            return
        for cd in target.values():
            cd.save()
        if resolution == 'lowres':
            self.conf.save()
        print(f"[CCGDataset] {resolution} saved ({len(target)} session(s))")

    def load(self, resolution: str = 'lowres', nd_key=None) -> str:
        """Load CCGData from disk. Returns loaded|missing|stale."""
        keys = [nd_key] if nd_key else (list(self.nd.data.keys()) if self.nd else [])
        if not keys:
            return 'missing'
        loaded = 0
        for sk in keys:
            load_key = sk.change(resolution=resolution)
            cd = CCGData(key=load_key, conf=self.conf,
                         ccg=None, ccg_null=None, pval=None, qval=None,
                         root=self._save_path)
            path = cd.save_path() + '.npz'
            if not os.path.isfile(path):
                continue
            try:
                cd.load()
                self.ccg[load_key] = cd
                self.cache.touch(load_key)
                loaded += 1
            except Exception as exc:
                print(f"[CCGDataset] load failed {sk}: {exc}")
        if loaded:
            print(f"[CCGDataset] {resolution} loaded ({loaded}/{len(keys)} session(s))")
        return 'loaded' if loaded == len(keys) else ('stale' if loaded else 'missing')


def _multiple_correction(pvals: np.ndarray, alpha: float, method: str = 'bonferroni') -> tuple:
    """Correct p-values over bins per (seg,ref,tgt) triple. Returns (sig_bool, p_correct)."""
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
    """Runs EranConv convolution-based significance detection."""

    def __init__(self, conf):
        self._pvals = []
        self._qvals = []
        self.significant = []  # final filtering results
        self.conf = conf

    @staticmethod
    def _conv(ccg, W=5, wintype="gauss", hollow_frac=None):
        """
        Estimate chance-level correlations using convolution method from Stark and Abeles (2009, J. Neuro Methods).
        Referencing MATLAB script EranConv.m written by the authors

        Parameters
        ----------
        ccg: np.array. 
            1D or 2D. (CCGs in columns)
            If 2D, elements in the first dimension are individual ccgs and second dimension are bins.
        W: 
            defines the width (unit: ms) of the convolution window, should be same as size of jitter window if were to use one
            `gauss`: W is standard deviation (sigma). Total window length will be 
            `rect`: Half size of window = W, total length is always odd
            `triang`: Window length is W rounded up to the nearest odd number

        wintype: ["gauss", "rect", "triang"]
            Type of convolution window.
            `gauss`: Gaussian kernel
            `rect`: rectangular kernel
            `triang`: triangular kernel

        hollow_frac: weight of the current bin
        
        Returns
        -------
        pvals: p-values (bin-wise)
        pred: predictor (expected values) 
        qvals: p-values (bin-wise) for inhibition
        """
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
        # Use mean across the spkcount window so that a hollow center bin
        # (zero spike count at lag=0) doesn't discard the whole pair.
        # Previously used .all(axis=-1) which required EVERY bin >= threshold.
        pair_inds = np.argwhere(
            ccg[..., min_bin:max_bin].mean(axis=-1) >= threshold)
        # NOTE right now it's the same criteria for excitation/inhibition
        return pair_inds

    def significance_mask(self, p, excitability):
        """Return pair indices with significant CCG peaks.

        Excitatory (E): a bin in [min_lag, max_lag) must survive MC correction.
        Inhibitory (I): a surviving bin must have a surviving neighbour at the
        looser ``alpha2`` threshold (ensures trough, not just noise).
        """
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
        return pair_inds

    def _cell_type_mask(self, pair_inds, neuron_type, conn_types):
        if pair_inds.ndim == 1:
            pair_inds = pair_inds.reshape(0, 2)
        sig_pairs = {}
        if not _hasvalue(pair_inds):
            for ct in conn_types:
                sig_pairs[ct] = None
            return sig_pairs

        # Condition 3: Ref/Target are specific cell types
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
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
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

        # Record a global map of significant pairs
        self.significant = np.zeros(ccg.shape[:3], dtype=bool)
        for inds in [inds_E, inds_I]:
            if inds is None:
                continue
            for k, v in inds.items():
                if v is None:
                    continue
                self.significant[tuple(v.T)] = True

        # Force CCG to be 4D
        if ccg.ndim == 3:
            ccg = ccg[None]
            pred = pred[None]
            for attr in ("_pvals", "_qvals"):
                setattr(self, attr, getattr(self, attr)[None])

        # Build printstr using raw inds (no CCGPointer created here)
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
        # Return raw inds dicts; CCGDataset builds CCGPointers with its root
        return pvals, pred, qvals, (inds_E, inds_I), printstr

