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
from neuropy.analyses.utils import (_san_np, _hasvalue, Config, AnalysisDataset,
                                    Savable, Cacheable, SetOp, SessionMemoryCache)
from neuropy.analyses.neurons_dataset import Key, NeuronsDataset

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
        conn_types: list = [('E', ('pyr', 'pyr')), 
                            ('E', ('pyr', 'inter')),
                            ('I', ('inter', 'inter')), 
                            ('I', ('inter', 'pyr'))],
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
        super().__init__()
        self.name = name
        self.resolution = resolution

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
        self.center_bin = int(self.duration / self.bin_size // 2)
        self.nbins = int(self.duration / self.bin_size) + 1  # NOTE

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_spkcount = min_spkcount
        self.spkcnt_scope = spkcount_scope
        self.spkcnt_bins = int(self.spkcnt_scope / self.bin_size)

        self.min_lag_bin = self.center_bin + int(
            self.min_lag / self.bin_size)  # leftmost bin for p value test
        self.max_lag_bin = self.center_bin + int(
            self.max_lag / self.bin_size) + 1  # rightmost bin for p value test
        self.min_spkcnt_bin = self.center_bin - self.spkcnt_bins // 2  # leftmost bin requiring minimum spike count
        self.max_spkcnt_bin = self.center_bin + self.spkcnt_bins // 2 + 1  # rightmost bin requiring minimum spike count

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

    def save_path(self, root=DATA_ROOT, suffix='config') -> str:
        """data/ccg/default/default_lowres_config.hkl"""
        base = os.path.expanduser(os.path.join(root, "ccg", self.name, f"{self.name}_{self.resolution}"))
        return f"{base}_{suffix}" if suffix else base

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.save_path}\n"
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
    windows: list = field(default_factory=list)          # list[IntervalSet]
    firing_rates: object = field(default=None, repr=False)  # ndarray [n_segs, n_neurons]
    tags: dict = field(default_factory=dict)             # e.g. {'kind': 'custom'}
    src_path: str = None

    _groups = {
        'key': ['name', 't0', 't1', 'filter_state'],
    }

    def __post_init__(self):
        super().__init__()
        if self.active_duration is None and isinstance(self.t0, (int, float)) and isinstance(self.t1, (int, float)):
            self.active_duration = self.t1 - self.t0
        if self.total_time_hours is None and self.active_duration is not None:
            self.total_time_hours = self.active_duration / 3600.0

    def __eq__(self, other) -> bool:
        if not isinstance(other, CCGSourceConfig):
            return NotImplemented
        return self.matches(other, 'key')

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

    @classmethod
    def spec_key(cls, spec: dict) -> tuple:
        return cls.from_dict(spec)._key()

    @classmethod
    def normalize(cls, spec: dict, *, default_session: str = '') -> dict:
        return cls.from_dict(spec, default_session=default_session).to_dict()

    @classmethod
    def from_dict(cls, d: dict, *, default_session: str = '') -> 'CCGSourceConfig':
        fs = d.get('filter_state', {}) or {}
        labels = fs.get('labels', {}) or {}
        sessions = sorted(str(s) for s in (d.get('sessions', []) or []) if s is not None)
        t0_raw, t1_raw = d.get('t0', 0.0), d.get('t1', 0.0)
        t0 = str(t0_raw) if isinstance(t0_raw, str) and t0_raw.lower() in ('start', 'end') else float(t0_raw)
        t1 = str(t1_raw) if isinstance(t1_raw, str) and t1_raw.lower() in ('start', 'end') else float(t1_raw)
        return cls(
            name=str(d.get('name', '')),
            t0=t0, t1=t1,
            filter_state={'theme': str(fs.get('theme', 'segments')),
                          'labels': {str(k): bool(v) for k, v in labels.items()}},
            scope=str(d.get('scope', default_session)),
            created_from_session=str(d.get('created_from_session', default_session)),
            sessions=sessions,
            n_splits=int(d.get('n_splits') or 1),
            overlap_sec=float(d.get('overlap_sec') or 0.0),
            active_duration=d.get('active_duration'),
            total_time_hours=d.get('total_time_hours'),
        )

    @classmethod
    def from_npz(cls, npz, session: str) -> 'CCGSourceConfig':
        fs = json.loads(str(npz['filter_state_'])) if 'filter_state_' in npz else {}
        return cls(
            name=str(npz['name_']), t0=float(npz['t0_']), t1=float(npz['t1_']),
            scope=session, created_from_session=session, sessions=[session], filter_state=fs,
        )

    @staticmethod
    def infer_scope(sessions: list[str], all_sessions: list[str]) -> str:
        if all_sessions and sorted(sessions) == sorted(all_sessions):
            return 'All'
        return sessions[0] if len(sessions) == 1 else 'By session'

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class CCGPointer(Savable):
    """Positional pointer to CCGData locations for significant pairs."""
    def __init__(
        self,
        key,
        inds,
        selected_inds=None,
        conf: CCGConfig = None,
        significant=None,
        edge_times=None,
    ):
        super().__init__()
        self.key = key
        self._inds = _san_np(inds)
        self.edge_times = edge_times
        self.conf = conf
        self.significant = significant
        self.selected_inds = selected_inds

    @property
    def segment_names(self) -> list:
        return list(self.edge_times['label'].values)

    @property
    def stored_by_segment(self):
        if not _hasvalue(self._inds):
            return False
        return self._inds.shape[-1] == 3  #otherwise 2

    @property
    def inds2(self):
        if self.stored_by_segment:
            return SetOp.unique(self._inds[:, -2:])
        else:
            return self.inds

    @property
    def unselected_inds(self) -> set:
        all_pairs = set(map(tuple, self.inds2))
        sel = getattr(self, 'selected_inds', None)
        if sel is None:
            return all_pairs
        return all_pairs - set(map(tuple, sel))

    @property
    def n_pairs(self):
        if not _hasvalue(self._inds):
            return 0
        if self.stored_by_segment:
            return self.inds2.shape[0]
        else:
            return self.inds.shape[0]

    @property
    def inds(self):
        if self._inds is None:
            return np.empty((0, 2), dtype=int)
        return self._inds
    
    @property
    def ref_ind(self):
        return self.inds[-2]

    @property
    def n_segments(self):
        return self.edge_times.shape[0]

    def get_segment(self, i: int) -> 'CCGPointer':
        if self._inds is None:
            inds = None
        elif self.stored_by_segment is False:
            assert i < self.n_segments
            inds = np.hstack([np.zeros(self.n_pairs), self.inds])
        else:
            inds = self.inds[np.where(self.inds[:, 0] == i)[0]][:, 1:]
        return CCGPointer(
            key=self.key.add(segment=i),
            inds=inds,
            edge_times=self.edge_times.iloc[i],
        )

    def split(self) -> list['CCGPointer']:
        return [self.get_segment(i) for i in range(self.edge_times.shape[0])]

    def save_path(self, root=DATA_ROOT) -> str:
        """data/ccg/default/default_lowres_ccgpointers_RatU_Day1.hkl"""
        return self.conf.save_path(root=root, suffix=f'ccgpointers_{self.key.session}')

    def save(self) -> None:
        super().save()

    @staticmethod
    def save_all(ptr: dict) -> None:
        for ptr_obj in ptr.values():
            ptr_obj.save()
        print(f"[CCGPointer] saved {len(ptr)} pointers")

    @classmethod
    def load(cls, ptr: dict, conf: 'CCGConfig') -> str:
        import glob
        base = conf.save_path(suffix='ccgpointers')
        files = glob.glob(base + '_*.hkl')
        if not files:
            return 'missing'
        ptr.clear()
        for f in files:
            try:
                obj = hkl.load(f)
                if not isinstance(obj, CCGPointer):
                    print(f"[CCGPointer] unexpected type in {f}: {type(obj).__name__}")
                    ptr.clear()
                    return 'stale'
                ptr[obj.key] = obj
            except Exception as exc:
                print(f"[CCGPointer] load failed {f}: {exc}")
                ptr.clear()
                return 'stale'
        print(f"[CCGPointer] loaded {len(ptr)} pointers ← {base}_*.hkl")
        return 'loaded'

    def __str__(self):
        s = 'CCG Pointer\n'
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray) or isinstance(val, list):
                s += f"{key}\tshape={np.array(val).shape}"
                sval = "\n".join(str(val[0:2]).splitlines()[:3])
                s += f"\tval={sval}...\n"
            elif isinstance(val, dict):
                k, v = next(iter(val.items()))
                s += f"{key} dict keys={k}...\n"
                item_str = str(v)
                for line in item_str.splitlines()[:3]:
                    s += f"\t\t{line}\n"
            elif key != 'conf':
                sval = "\n".join(str(val).splitlines()[:3])
                s += f"{key}: {sval}\n"
        return s


class CCGData(Savable):
    """CCG arrays [n_seg, n_pair, n_bins] and p-values for one session."""
    IGNORED_KEYS = ('conf', 'key', '_ignored_attrs')
    _save_format = 'npz'

    def __init__(self, key, conf, ccg, ccg_null, pval, qval):
        super().__init__(ignored_attrs=self.IGNORED_KEYS)
        self.key = key
        self.conf = conf
        self.ccg = ccg
        self.ccg_null = ccg_null
        self.pval = pval
        self.qval = qval
        self.significant = None

    @property
    def n_segment(self):
        return self.ccg.shape[0]

    @property
    def pval_corrected(self):
        if self.pval is None or self.conf is None:
            return None
        _, pc = EranConv.multiple_correction(
            self.pval, self.conf.alpha, method=self.conf.multiple_correction)
        return pc

    def save_path(self, suffix='ccgdata') -> str:
        """data/ccg/default/default_lowres_ccgdata_RatU_Day1.npz"""
        res = self.key.resolution or 'lowres'
        return self.conf.save_path(suffix=f'{suffix}_{self.key.session}__{res}')

    @classmethod
    def load(cls, blob, conf: 'CCGConfig') -> 'CCGData':
        if isinstance(blob, cls):
            if conf is not None:
                blob.conf = conf
            return blob
        if isinstance(blob, dict):
            return cls(key=blob['key'], conf=conf, ccg=blob['ccg'],
                       ccg_null=blob.get('ccg_null'), pval=blob['pval'], qval=blob.get('qval'))
        raise TypeError(f"Cannot rehydrate CCGData from {type(blob)!r}")


class CCGDataset(AnalysisDataset, Cacheable):
    """CCGs and significance for an experiment. ptr: significant pairs; nd: source neurons."""

    ccg: dict[CCGData]
    ptr: dict[CCGPointer]
    conf: CCGConfig
    nd: NeuronsDataset

    def __init__(self, conf=None, nd=None, source=None):
        super().__init__(conf)
        self.nd = nd
        self.ccg = {}   # Key(session, resolution) → CCGData
        self.ptr = {}
        self.source = source    # CCGSourceConfig | None; has .tags, .src_path
        self.cache = SessionMemoryCache(self.ccg)
        ptr_status = CCGPointer.load(self.ptr, self.conf)
        if ptr_status in ('missing', 'stale'):
            self.get_ccg()

    def get_ccg(self, nd_key=None, baseline_method="eran_conv", use_segments=True,
                resolution='lowres'):
        """Load or compute CCG arrays, optionally restricted to one session.

        Parameters
        ----------
        nd_key : optional
            When given, load/compute only that session's CCG.
        resolution : 'lowres' | 'highres'
            'highres' computes finer-bin arrays and runs EranConv without rebuilding pointers.

        Cache strategy (split files):
           Exact same data exists?
           Y -> ask overwrite
           N -> COMPUTE_FIELDS changed?
                Y -> recompute CCG, save, ask overwrite for pointers, recompute pointers, save
                N -> SIGNIF_FIELDS changed?
                        Y -> re-run pointers, save
                        N -> load pointers, touch cache
        """
        if self.nd is None:
            return
        if baseline_method == "jitter":
            raise NotImplementedError(
                "CCG jitter is in neuropy.analyses.jitter. Use Jitter/JitterDataset.")
        if baseline_method != "eran_conv":
            raise ValueError(f"Unknown baseline_method: {baseline_method!r}")
        if resolution == 'highres':
            self._get_ccg_highres(nd_key)
            return
        conv = EranConv(self.conf)
        if nd_key is not None:
            self._get_ccg_session(nd_key, conv, use_segments)
        else:
            self._get_ccg_all(conv, use_segments)

    def _get_ccg_session(self, nd_key, conv, use_segments):
        if nd_key in self.ccg:
            self.cache.touch(nd_key)
            return
        if self.load(nd_key=nd_key) == 'loaded':
            self.cache.touch(nd_key)
            return
        if self.nd.edge_times.get(nd_key) is None:
            print(f"[CCGDataset] get_ccg: nd_key {nd_key} not in edge_times")
            return
        self.__ccg_eranconv(key=nd_key, conv=conv,
                            edge_times=self.nd.edge_times[nd_key],
                            use_segments=use_segments)
        self.save()
        CCGPointer.save_all(self.ptr, self.conf)
        self.conf.save()
        self.cache.touch(nd_key)

    def _get_ccg_all(self, conv, use_segments):
        ccgdata_status = self.load()
        if ccgdata_status == 'loaded':
            ptr_status = CCGPointer.load(self.ptr, self.conf)
            if ptr_status == 'loaded':
                print("[CCGDataset] Loaded CCGData + CCGPointers from cache.")
                for k in self.ccg:
                    self.cache.touch(k)
                return
            if ptr_status == 'stale':
                if not self._ask_overwrite(
                        self.conf.save_path(suffix='ccgpointers') + '.hkl', 'CCGPointers'):
                    print("[CCGDataset] Aborted.")
                    return
            print("[CCGDataset] CCGData cached; re-running significance detection.")
            for k, ccg_data in self.ccg.items():
                self.__run_eranconv_on_ccgdata(k, ccg_data, conv)
            CCGPointer.save_all(self.ptr, self.conf)
            self.conf.save()
            for k in self.ccg:
                self.cache.touch(k)
            return

        if ccgdata_status == 'stale':
            if not self._ask_overwrite(
                    self.conf.save_path(suffix='ccgdata') + '.hkl', 'CCGData'):
                print("[CCGDataset] Aborted.")
                return

        if self.conf.check_cache() and self.load_data():
            print("[CCGDataset] Loaded from legacy monolithic cache.")
            return

        missing_keys = [k for k in self.nd.edge_times.keys()
                        if k.change(resolution='lowres') not in self.ccg]
        for key in self.nd.edge_times.keys():
            if key not in missing_keys:
                print(self._summary(key))
        if not missing_keys:
            print("[CCGDataset] All sessions in cache.")
            return
        for key in missing_keys:
            self.__ccg_eranconv(key=key, conv=conv,
                                edge_times=self.nd.edge_times[key],
                                use_segments=use_segments)
        self.save()
        CCGPointer.save_all(self.ptr, self.conf)
        self.conf.save()
        for k in self.ccg:
            self.cache.touch(k)

    def _get_ccg_highres(self, nd_key=None) -> None:
        conf = self.conf.copy(bin_size=_CCG_RESOLUTION['highres'], resolution='highres')
        keys = [nd_key] if nd_key else list(self.nd.data.keys())
        missing = [k for k in keys
                   if k.change(resolution='highres') not in self.ccg
                   and self.load('highres', nd_key=k) != 'loaded']
        for k in set(keys) - set(missing):
            self.cache.touch(k.change(resolution='highres'))
        if not missing:
            return
        for k in missing:
            neurons = self.nd.data[k]
            print(f"[get_ccg] highres {k} — bin_size={conf.bin_size*1e3:.2f} ms …", flush=True)
            ccg = correlations.spike_correlations(
                neurons=neurons, neuron_inds=np.arange(neurons.n_neurons),
                bin_size=conf.bin_size, window_size=conf.duration,
                use_acceleration=conf.use_acceleration, symmetrize=conf.symmetrize_ccg,
                edge_times=self.nd.edge_times[k],
            )
            hi_key = k.change(resolution='highres')
            self.ccg[hi_key] = CCGData(key=hi_key, conf=conf, ccg=ccg,
                                       ccg_null=None, pval=None, qval=None)
            print(f"[get_ccg]   → shape {ccg.shape}")
            self.cache.touch(hi_key)
        hi_items = [(k.change(resolution='highres'), self.ccg[k.change(resolution='highres')])
                    for k in missing]
        for hi_key, ccg_hi in hi_items:
            self.__run_eranconv_on_ccgdata(hi_key, ccg_hi, conv=None, skip_ptr=True)
        self.save('highres')
        print(f"[get_ccg] highres complete — {len(missing)} session(s), "
              f"{conf.bin_size*1e3:.2f} ms bins.")

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
        # Prefer E excitability + pyr-pyr conn_type; fall back to first match
        preferred = [
            k for k in matches
            if getattr(k, 'excitability', None) == 'E'
            and getattr(k, 'conn_type', None) == ('pyr', 'pyr')
        ]
        return (preferred or matches)[0]

    def _summary(self, key) -> str:
        """Per-session segment / pair-count table (for logs after compute or cache hit)."""
        neurons = self.nd.data[key.nd()]
        edge_times = self.nd.edge_times[key]
        et = edge_times.effective_time_hours.values

        s = f"======={key.session}=======\n"
        s += f"Segment(s) are {[f'{_:.2f}' for _ in et]} hours long "
        if getattr(self.nd.conf, 'sleep', None) is not None:
            s += f"\nand contain {[f'{_:.2f}' for _ in et]} hours of actual sleep "
        for _ in self.nd.conf.neuron_types:
            s += f"{_}={neurons.get_neuron_type(_).n_neurons} "
        s += "\n"

        # Non-None fields of key used to filter stored CCGPointers to this session.
        nd_attrs = {
            f: getattr(key, f)
            for f in key.__dataclass_fields__
            if getattr(key, f) is not None
        }

        def belongs(k):
            return all(getattr(k, attr, None) == val for attr, val in nd_attrs.items())

        def count_seg(pointer, seg_i):
            if pointer is None or pointer.inds is None or len(pointer.inds) == 0:
                return 0
            if pointer.stored_by_segment:
                return int((pointer.inds[:, 0] == seg_i).sum())
            return pointer.n_pairs  # not segment-split; applies to all segments

        printstr = ''
        for i, (_, edge_time) in enumerate(edge_times.iterrows()):
            N_totalE, N_totalI = 0, 0
            for k, ptr in self.ptr.items():
                if not belongs(k):
                    continue
                n = count_seg(ptr, i)
                if k.excitability == 'E':
                    N_totalE += n
                elif k.excitability == 'I':
                    N_totalI += n

            printstr += f"{edge_time['label']:10}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "

            for EI, (ref, target) in self.conf.conn_types_labeled:
                ct = (ref, target)
                N = 0
                for k, ptr in self.ptr.items():
                    if not belongs(k):
                        continue
                    if (k.excitability == EI
                            and k.conn_type is not None
                            and tuple(k.conn_type) == ct):
                        N = count_seg(ptr, i)
                        break
                printstr += f"{ref}-{target}/{EI} {f'{N:02d}' if N > 0 else ' -'} | "
            printstr += '\n'

        return s + printstr

    def __run_eranconv_on_ccgdata(self, nd_key, ccg_data, conv, skip_ptr=False):
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
            sig, _ = EranConv.multiple_correction(pvals, alpha=conf.alpha,
                                                   method=conf.multiple_correction)
            ccg_data.ccg_null = pred
            ccg_data.pval = pvals
            ccg_data.qval = qvals
            n_sig = int(sig.any(axis=-1).sum()) if sig is not None else 0
            print(f"[EranConv] {nd_key} shape={ccg.shape} W={W} → {n_sig} sig pair-segs")
            return

        neurons = self.nd.data[nd_key]
        edge_times = self.nd.edge_times[nd_key]
        pvals, pred, qvals, ccg_ptrs, printstr = conv.eranconv(
            neurons_key=nd_key, ccg=ccg, edge_times=edge_times,
            neuron_type=neurons.neuron_type, conf=self.conf)
        ccg_data.ccg_null = pred
        ccg_data.pval = pvals
        ccg_data.qval = qvals
        ccg_data.significant = conv.significant
        self._attr_append(nd_key, ccg_ptrs, 'ptr')
        print(printstr)

    def __ccg_eranconv(self, key, conv, edge_times, use_segments=True):
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

        ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(neurons.n_neurons),  # all
            bin_size=self.conf.bin_size,
            window_size=self.conf.duration,
            use_acceleration=self.conf.use_acceleration,
            symmetrize=self.conf.symmetrize_ccg,
            edge_times=edge_times if use_segments else None,
        )

        pvals, pred, qvals, ccg_ptrs, printstr = conv.eranconv(
            neurons_key=key,
            ccg=ccg,
            edge_times=edge_times,
            neuron_type=neurons.neuron_type,
            conf=self.conf)

        ccg_data = CCGData(
            key=key,
            conf=self.conf,
            ccg=ccg,
            ccg_null=pred,
            pval=pvals,
            qval=qvals,
        )
        ccg_data.significant = conv.significant

        self.ccg[key] = ccg_data
        self._attr_append(key, ccg_ptrs, 'ptr')

        print(self._summary(key))

    def copy(self) -> "CCGDataset":
        """Copy only conf and nd (nd is a shallow reference)"""
        new = self.__class__(conf=self.conf)
        new.nd = self.nd
        return new

    def save_path(self, root=DATA_ROOT, suffix='') -> str:
        """data/ccg/default/default_lowres.hkl"""
        return self.conf.save_path(root=root, suffix=suffix)

    def save(self, resolution: str = 'lowres') -> None:
        target = {k: v for k, v in self.ccg.items() if k.resolution == resolution}
        if not target:
            print(f"[CCGDataset] {resolution}: nothing to save.")
            return
        for cd in target.values():
            cd.save()
        if self.source is not None:
            src_p = self.conf.save_path(suffix=f'source__{resolution}')
            self.source.save(path=src_p)
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
                         ccg=None, ccg_null=None, pval=None, qval=None)
            if not os.path.isfile(cd.save_path() + '.npz'):
                continue
            try:
                cd.load()
                self.ccg[load_key] = cd
                self.cache.touch(load_key)
                loaded += 1
            except Exception as exc:
                print(f"[CCGDataset] load failed {sk}: {exc}")
        if loaded and self.source is None:
            src_p = self.conf.save_path(suffix=f'source__{resolution}') + '.hkl'
            if os.path.isfile(src_p):
                try:
                    from neuropy.analyses.custom_ccg import CCGSourceConfig
                    shell = CCGSourceConfig(name='', t0=0.0, t1=0.0)
                    shell.load(path=src_p[:-4])  # strip .hkl; load() appends it
                    self.source = shell
                except Exception as exc:
                    print(f"[CCGDataset] source load failed: {exc}")
        if loaded:
            print(f"[CCGDataset] {resolution} loaded ({loaded}/{len(keys)} session(s))")
        return 'loaded' if loaded == len(keys) else ('stale' if loaded else 'missing')

    def compute_custom(self, neurons_slice, *, has_highres: bool = False,
                       excitability: str = 'E') -> None:
        """Run CCG for self.source time window; populate self.ccg with lo (and hi) CCGData."""
        src = self.source
        n_neurons = neurons_slice.n_neurons
        neuron_inds = np.arange(n_neurons)
        conf = self.conf

        def _run(bin_size: float, resolution: str) -> None:
            print(f"[CCGDataset] computing {resolution} CCG for {src.name} "
                  f"({src.t1-src.t0:.1f}s, {n_neurons} neurons, bin={bin_size*1e3:.2f}ms) ...")
            ccg = correlations.spike_correlations(
                neurons=neurons_slice, neuron_inds=neuron_inds,
                bin_size=bin_size, window_size=conf.duration,
                symmetrize=conf.symmetrize_ccg, use_acceleration=conf.use_acceleration,
            )[np.newaxis, ...]
            W = conf.conv_window / bin_size
            pvals, pred, qvals = EranConv._conv(ccg, W=W, wintype="gauss", hollow_frac=None)
            print(f"[CCGDataset] {resolution} done. shape={ccg.shape}")
            k = Key(session=src.name, resolution=resolution)
            self.ccg[k] = CCGData(
                key=k, conf=conf,
                ccg=ccg, ccg_null=pred,
                pval=pvals if excitability == 'E' else qvals, qval=None,
            )

        active = src.active_duration or (src.t1 - src.t0)
        src.firing_rates = np.array(
            [len(st) for st in neurons_slice.spiketrains], dtype=float
        ) / max(active, 1e-9)
        _run(_CCG_RESOLUTION['lowres'], 'lowres')
        if has_highres:
            _run(_CCG_RESOLUTION['highres'], 'highres')


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

    @staticmethod
    def multiple_correction(pvals: np.ndarray,
                            alpha: float,
                            method: str = 'bonferroni'
                            ) -> tuple:
        """Per-pair multiple-comparison correction over bins only.

        For each (seg, ref, tgt) triple independently, correct the ``n_bins``
        p-values for that pair.  Segments and pairs are fully decoupled — they
        never inflate each other's correction penalty.

        Parameters
        ----------
        pvals : ndarray, shape ``[n_seg, n_ref, n_tgt, n_bins]``
            Raw Poisson p-values from :meth:`_conv`.
        alpha : float
            Significance threshold (applied to corrected p-values).
        method : str
            ``'bonferroni'`` (default) — multiply each p-value by ``n_bins``,
            clip at 1.  Fast, conservative, and transparent.
            Any other string accepted by
            ``statsmodels.stats.multitest.multipletests`` also works
            (e.g. ``'fdr_bh'``).

        Returns
        -------
        significance : bool ndarray, same shape as *pvals*
        corrected_pvals : float ndarray, same shape as *pvals*
        """
        if method == 'bonferroni':
            n_bins = pvals.shape[-1]
            corrected = np.minimum(pvals * n_bins, 1.0)
            return corrected <= alpha, corrected

        # Fallback for FDR-BH and other statsmodels methods.
        significance = np.zeros_like(pvals, dtype=bool)
        corrected_pvals = np.ones_like(pvals, dtype=float)
        for idx in np.ndindex(pvals.shape[:-1]):
            row = pvals[idx]
            s, pc, _, _ = multipletests(row, alpha=alpha, method=method)
            significance[idx] = s
            corrected_pvals[idx] = pc
        return significance, corrected_pvals

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
            sig, self._pvals = EranConv.multiple_correction(p, conf.alpha, method=method)
            # At least one corrected-significant bin in the excitatory test window.
            has_valid_peak = sig[..., conf.min_lag_bin:conf.max_lag_bin].any(axis=-1)
            pair_inds = np.argwhere(has_valid_peak)
        elif excitability == 'I':
            sig1, self._qvals = EranConv.multiple_correction(p, conf.alpha, method=method)
            sig2, _ = EranConv.multiple_correction(p, conf.alpha2, method=method)
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
        self.n_segments = edge_times.shape[0]

        pvals, pred, qvals = EranConv._conv(ccg,
                                            W=self.conv_window / self.bin_size, # number of bins in conv window
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

        ccg_inds_by_type = {}

        # Force CCG to be 4D
        if ccg.ndim == 3:
            ccg = ccg[None]
            pred = pred[None]
            for attr in (
                    "_pvals",
                    "_qvals",
            ):
                setattr(self, attr, getattr(self, attr)[None])

        count = np.zeros((edge_times.shape[0], len(self.conf.conn_types_flat)),
                         dtype=int)
        j = 0
        for EI in ['E', 'I']:
            for conn_type in (self.conf.conn_types_E if EI == 'E' else self.conf.conn_types_I):
                inds = inds_E[conn_type] if EI == 'E' else inds_I[conn_type]
                ccg_key = key.add(conn_type=conn_type, excitability=EI)
                ccg_ptr = CCGPointer(key=ccg_key,
                                         conf=self.conf,
                                         inds=inds if _hasvalue(inds) else None,
                                         edge_times=edge_times)
                for i, ccg in enumerate(ccg_ptr.split()):
                    count[i, j] = ccg.n_pairs if ccg is not None else 0
                ccg_inds_by_type[ccg_key] = ccg_ptr
                j += 1

        printstr = ''
        for i, (segment_i, edge_time) in enumerate(edge_times.iterrows()):
            N_totalE = (rough_inds_E[:, 0] == i).sum()
            N_totalI = (rough_inds_I[:, 0] == i).sum()
            printstr += f"{edge_time['label']:10}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "

            for N, (EI, (ref, target)) in zip(count[i], self.conf.conn_types_labeled):
                printstr += f"{ref}-{target}/{EI} {f'{N:02d}' if N>0 else ' -'} | "
            printstr += '\n'

        print("eranconv done")

        return pvals, pred, qvals, ccg_inds_by_type, printstr


