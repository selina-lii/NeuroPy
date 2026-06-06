import numpy as np
from dataclasses import dataclass, field, replace
from typing import Union, Optional, Dict, Any, Tuple, TypeVar, Type
from collections import defaultdict
import hickle as hkl
import glob as _glob
import json
import os
import re
import shutil


def _san(var, wrap_none=False):
    """
    Sanitize array
    """
    if var is None:
        return [None] if wrap_none else None
    if not isinstance(var, list):
        return [var]
    return var


def _san_np(var, wrap_none=False):
    """
    Sanitize array
    """
    if var is None:
        return np.array(None) if wrap_none else None
    if not isinstance(var, np.ndarray):
        return np.ndarray(var)
    return var


def _hasvalue(x):
    """
    Determine if a size-able object is empty
    """
    return x is not None and x.size > 0


class Savable:

    _save_format: str = 'hkl'  # subclasses override to 'npz'

    def __init__(self, ignored_attrs: list = []):
        self._ignored_attrs = ignored_attrs

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in self._ignored_attrs
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def pack(self):
        """Pickle-safe state dict (respects ``_ignored_attrs``)."""
        return self.__getstate__()

    @staticmethod
    def _to_state(obj):
        return obj.pack() if isinstance(obj, Savable) else obj

    @classmethod
    def save_mapping(cls, mapping: dict, path: str) -> None:
        """Write a dict of Savable instances (or plain dict states) to one .hkl file."""
        hkl.dump({k: cls._to_state(v) for k, v in mapping.items()}, path)

    @staticmethod
    def _ask_overwrite(path: str, label: str) -> bool:
        print(f"\n[{label}] file already exists at:\n  {path}")
        try:
            answer = input("  Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            answer = ''
        return answer in ('y', 'yes')

    def save_path(self, **kwargs):
        return "./tmp"

    def save(self,
             path: str = None,
             ignored_attrs: list = [],
             split_into_chunks=False,
             chunk_size_MB: int = 20):
        if ignored_attrs:
            self._ignored_attrs = _san(ignored_attrs)
        fmt = getattr(self, '_save_format', 'hkl')
        if fmt == 'npz':
            p = (path or self.save_path()) + '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            state = {k: v for k, v in self.pack().items() if isinstance(v, np.ndarray)}
            np.savez_compressed(p, **state)
            return
        if fmt == 'json':
            p = (path or self.save_path()) + '.json'
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, 'w') as _f:
                json.dump(self.pack(), _f)
            return
        p = (path or self.save_path()) + '.hkl'
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        state = self.pack()
        if not split_into_chunks:
            hkl.dump(state, p)
        else:
            chunk_size = chunk_size_MB * 1024 * 1024
            folder = p if os.path.isdir(p) else p + "_files"
            os.makedirs(folder, exist_ok=True)
            file = os.path.join(folder, "temp.hkl")
            hkl.dump(state, file)
            with open(file, "rb") as f:
                i = 0
                while chunk := f.read(chunk_size):
                    with open(os.path.join(folder, f"part{i}"), "wb") as out:
                        out.write(chunk)
                    i += 1
            os.remove(file)

    def load(self, path: str = None):
        fmt = getattr(self, '_save_format', 'hkl')
        if fmt == 'npz':
            p = (path or self.save_path()) + '.npz'
            if not os.path.exists(p):
                print(f"File not found: {p}")
                return
            self.__setstate__(dict(np.load(p, allow_pickle=False)))
            return
        if fmt == 'json':
            p = (path or self.save_path()) + '.json'
            if not os.path.exists(p):
                print(f"File not found: {p}")
                return
            with open(p) as _f:
                self.__setstate__(json.load(_f))
            return
        p = (path or self.save_path()) + '.hkl'
        splitted = False
        file = p
        if not os.path.exists(p):
            splitted = True
            folder = p if os.path.isdir(p) else p + "_files"
            file = os.path.join(folder, "recombined.hkl")
            with open(file, "wb") as out:
                i = 0
                while True:
                    part_file = os.path.join(folder, f"part{i}")
                    if not os.path.exists(part_file):
                        break
                    with open(part_file, "rb") as part:
                        out.write(part.read())
                    i += 1
        try:
            loaded = hkl.load(file)
            if isinstance(loaded, dict) and not isinstance(loaded, Savable):
                self.__setstate__(loaded)
            else:
                for k, v in loaded.__dict__.items():
                    setattr(self, k, v)
        except Exception as e:
            print(f"Failed to load {self.__class__} object: {e}")
        finally:
            if splitted:
                os.remove(file)

    @property
    def is_saved(self) -> bool:
        fmt = getattr(self, '_save_format', 'hkl')
        ext = {'npz': '.npz', 'json': '.json'}.get(fmt, '.hkl')
        return os.path.isfile(self.save_path() + ext)

    def find_cached(self, suffix='') -> str:
        """Return 'ok', 'missing', or 'stale' for save_path(suffix=suffix).hkl."""
        p = self.save_path(suffix=suffix) + '.hkl'
        if not os.path.isfile(p):
            return 'missing'
        return 'ok' if self.check_cache() else 'stale'

    def check_cache(self, group='key') -> bool:
        """Return True if saved hkl matches current config for the given field group."""
        p = self.save_path() + '.hkl'
        if not os.path.isfile(p):
            return False
        try:
            saved = hkl.load(p)
            fields = self._groups.get(group, [])
            _sv = getattr(self.__class__, 'serialize_value', lambda v: v)
            for f in fields:
                if _sv(getattr(saved, f, None)) != _sv(getattr(self, f, None)):
                    return False
            return True
        except Exception:
            return False


class Cacheable:
    """Mixin for classes that own a cache directory of session__name.ext files."""

    cache_dir: str

    def cache_filename(self, session: str, name: str, ext: str = 'npz') -> str:
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', str(name).replace(' ', '_'))
        return os.path.join(self.cache_dir, f"{session}__{safe}.{ext}")

    def purge_versioned(self, session: str, name: str, ext: str = 'npz'):
        """Remove session__name__*.ext legacy timestamped files."""
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', str(name).replace(' ', '_'))
        for p in _glob.glob(os.path.join(self.cache_dir, f"{session}__{safe}__*.{ext}")):
            try:
                os.remove(p)
            except OSError:
                pass

    def archive_stale(self, pattern: str, is_stale) -> tuple[int, str]:
        """Move files matching pattern where is_stale(path) is True to _trash/."""
        trash = os.path.join(self.cache_dir, '_trash')
        os.makedirs(trash, exist_ok=True)
        n = 0
        for p in _glob.glob(pattern):
            try:
                if is_stale(p):
                    shutil.move(p, os.path.join(trash, os.path.basename(p)))
                    n += 1
            except Exception:
                pass
        return n, trash

    def load_json_list(self, path: str) -> list:
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f).get('items', [])
        except Exception:
            return []

    def save_json_list(self, path: str, items: list, write_fn=None):
        payload = {'version': 1, 'items': items}
        if write_fn:
            write_fn(path, payload)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)


class Config(Savable):

    _groups: dict = {}

    def matches(self, other: 'Config', group: str) -> bool:
        return all(getattr(self, f) == getattr(other, f)
                   for f in self._groups[group])

    def diff(self, other: 'Config', group: str) -> dict:
        return {f: (getattr(self, f), getattr(other, f))
                for f in self._groups[group]
                if getattr(self, f) != getattr(other, f)}

    def copy(self, **overrides):
        """Return a new instance of the same class with overridden init params."""
        import inspect
        params = {k: v for k, v in self.__dict__.items()
                  if k in inspect.signature(self.__class__.__init__).parameters}
        params.update(overrides)
        return self.__class__(**params)

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.filepath}\n"
        return s

    @staticmethod
    def _serialize_conf_value(v):
        """JSON-safe config value for cache metadata."""
        if hasattr(v, 'name'):
            return v.name
        if isinstance(v, (list, tuple)):
            return [Config._serialize_conf_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): Config._serialize_conf_value(val) for k, val in v.items()}
        try:
            import json as _json
            _json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)


Config.serialize_value = Config._serialize_conf_value


from enum import Enum


class ConfigOption(Enum):

    def __eq__(self, x):
        return getattr(self, "name", None) == getattr(x, "name", None)

    # __eq__ override suppresses __hash__ in Python — restore it explicitly
    __hash__ = Enum.__hash__


K = TypeVar("K", bound="GenericKey")


@dataclass(frozen=True)
class GenericKey:
    """
    Indexing object.
    """

    def __str__(self):
        pass

    def __eq__(self, other):
        try:
            return all(
                getattr(self, f) == getattr(other, f)
                for f in self.__dataclass_fields__)
        except:
            return False  # NOTE no type check, allows comparison with other 'key' classes

    def matches(self, **kwargs) -> bool:
        """Check if this key matches given criteria (for filtering)"""
        for k, v in kwargs.items():
            if isinstance(v, list):
                if getattr(self, k, None) not in v:
                    return False
            else:
                if v is not None and getattr(self, k, None) != v:
                    return False
        return True

    def _new(self: K, **kwargs) -> K:
        return replace(self, **kwargs)

    def get(self: K, *dimensions) -> K:
        return type(self)(**{d: getattr(self, d, None) for d in dimensions})

    def remove(self: K, *dimensions) -> K:
        return type(self)(**{
            f: getattr(self, f)
            for f in self.__dataclass_fields__
            if f not in dimensions
        })

    def add(self: K, **kwargs) -> K:
        for k in kwargs:
            assert getattr(self, k) is None
        return self._new(**kwargs)

    def change(self: K, **kwargs) -> K:
        return self._new(**kwargs)

    def nd(self) -> K:
        return self.get('session')

    @staticmethod
    def groupby(data, *dimensions) -> dict:
        """
        Group keys by specified dimensions.
        
        Example:
            dataset.groupby('session', 'epoch')
            # Returns: {('s1', 'pre'): {key: data, ...}, ('s1', 'post'): {...}}
        """
        groups = defaultdict(dict)
        for key, value in data.items():
            # group_key = tuple(getattr(key, dim, None) for dim in dimensions)
            groups[key.get(*dimensions)][key] = value
        return dict(groups)


class AnalysisDataset(Savable):
    """
    Container for an analysis dataset.
    """

    def __init__(self, conf=None):
        super().__init__()
        self.data: Dict[K, Any] = {}
        self._conf = conf

    def __len__(self):
        return len(self.data)

    def example(self, field=None, i=None) -> Any:
        """Get an example from data or another field"""
        var = self.__dict__[field] if field else self.data
        if i:
            try:
                return var[list(var.keys())[i]]
            except:
                return None
        return var.get(next(iter(var.keys())), None)

    def filter(self, attrname='data', **filters) -> Dict[K, Any]:
        """
        Filter data by any combination of key attributes.
        
        Example:
            dataset.filter(session='s1', epoch='pre')
            dataset.filter(analysis_type='correlogram', neuron_type='pyramidal')
        """
        return {
            k: v
            for k, v in getattr(self, attrname).items()
            if k.matches(**filters)
        }

    def filter_keys(self, attrname='data', **filters) -> list[K]:
        """Get all keys matching criteria"""
        return [
            k for k in getattr(self, attrname).keys() if k.matches(**filters)
        ]

    def groupby(self, *dimensions, source='data') -> Dict[K, Dict[K, Any]]:
        """
        Group data by specified dimensions.
        
        Example:
            dataset.groupby('session', 'epoch')
            # Returns: {('s1', 'pre'): {key: data, ...}, ('s1', 'post'): {...}}
        """
        items = getattr(self, source)
        groups = defaultdict(dict)
        for key, value in items.items():
            # group_key = tuple(getattr(key, dim, None) for dim in dimensions)
            groups[key.get(*dimensions)][key] = value
        return dict(groups)

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, conf):
        self._conf = conf
        print(
            f"{self.__class__.__name__}Config changed, which might create inconsistencies between existing data and config. Rerun if necessary."
        )

    def _attr_append(self,
                     base_key: K,
                     inputs: Dict[K, Any],
                     attrname: str = 'data'):
        getattr(self, attrname).update({
            type(base_key)(**{
                **base_key.__dict__,
                **k.__dict__
            }): v for k, v in inputs.items()
        })

    def save_data(self, ignored_attrs=None):
        """Save this dataset to disk at self.save_path()."""
        path = self.save_path()
        os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
        self.save(path=path, ignored_attrs=ignored_attrs or [])
        print(f"[{self.__class__.__name__}] saved → {path}.hkl")

    def load_data(self) -> bool:
        """Load dataset from disk. Returns True if successful, False if not found."""
        p = self.save_path() + '.hkl'
        if not os.path.isfile(os.path.expanduser(p)):
            return False
        try:
            self.load(path=self.save_path())
            print(f"[{self.__class__.__name__}] loaded ← {p}")
            return True
        except Exception as e:
            print(f"[{self.__class__.__name__}] load failed: {e}")
            return False

    def copy(self) -> "AnalysisDataset":
        """Copy only conf"""
        new = self.__class__(conf=self._conf)
        return new


class SetOp():
    @staticmethod
    def __set_op(x, y, f):
        """
        Perform set operation of two N-dim arrays by their row elements.
        x,y: np.ndarray of shape [...,k]
        ravel_dims: (d1,...,dk), each d is sufficiently large

        Ravels row values to v = v1*d1+...+vn*dn for comparison and then conver back
    
        """
        ax = tuple(np.arange(len(x.shape) - 1))
        ravel_dims = np.max(np.vstack([x.max(axis=ax),
                                       y.max(axis=ax)]), axis=0) + 1
        xr, yr = np.ravel_multi_index(x.T, ravel_dims), np.ravel_multi_index(
            y.T, ravel_dims)
        res = f(xr, yr)
        return np.array(np.unravel_index(res, ravel_dims)).T

    @staticmethod
    def intersect(x, y):
        """
        Intersect two N-dim arrays by their row elements
        """
        if x is None or y is None:
            return np.array([])
        if x.size == 0 or y.size == 0:
            ncols = x.shape[1] if x.ndim > 1 else (y.shape[1] if y.ndim > 1 else 0)
            return np.empty((0, ncols), dtype=int) if ncols else np.array([])
        return SetOp.__set_op(x, y, np.intersect1d)

    @staticmethod
    def setdiff(x, y):  #n2=None
        """
        X minus Y for two N-dim arrays by their row elements
        """
        # Set difference of coordinate lists
        if x is None or y is None:
            return x if x is not None else np.array([])
        if x.size == 0:
            return np.array([])
        if y.size == 0:
            return x
        return SetOp.__set_op(x, y, np.setdiff1d)

    @staticmethod
    def union(x, y):  #n2=None
        """
        Union two N-dim arrays by their row elements
        """
        # Set difference of coordinate lists
        if x is None:
            return y if y is not None else np.array([])
        elif y is None:
            return x if x is not None else np.array([])
        if x.size == 0:
            return y
        if y.size == 0:
            return x
        return SetOp.__set_op(x, y, np.union1d)

    @staticmethod
    def unique(x):
        """
        np.unique by row elements
        """
        return np.unique(x, axis=0)


def filter_neurons_to_intervals(neurons, intervals, t0: float, t1: float):
    """Return a deepcopy of *neurons* with spiketrains masked to *intervals*.

    Safe to call from background threads — deepcopy stays off the main thread.
    """
    from copy import deepcopy
    from neuropy.core.neurons import Neurons
    neurons = deepcopy(neurons)
    filtered_trains = []
    for st in neurons.spiketrains:
        mask = np.zeros(len(st), dtype=bool)
        for s, e in intervals:
            mask |= (st >= s) & (st <= e)
        filtered_trains.append(st[mask])
    return Neurons(
        spiketrains=filtered_trains,
        t_stop=t1, t_start=t0,
        sampling_rate=neurons.sampling_rate,
        neuron_ids=neurons.neuron_ids,
        neuron_type=neurons.neuron_type,
        waveforms=neurons.waveforms,
        waveforms_amplitude=neurons.waveforms_amplitude,
        peak_channels=getattr(neurons, 'peak_channels', None),
        shank_ids=getattr(neurons, 'shank_ids', None),
        metadata=neurons.metadata,
    )


def split_time_range(t0: float, t1: float, n_splits: int, overlap_sec: float,
                     base_name: str) -> list:
    """Partition [t0, t1] into n_splits overlapping chunks.

    Returns list of (chunk_t0, chunk_t1, chunk_name) tuples.
    """
    n_splits = max(1, int(n_splits))
    overlap_sec = max(0.0, float(overlap_sec))
    if n_splits == 1 and overlap_sec == 0.0:
        return [(t0, t1, base_name)]
    total = t1 - t0
    if total <= 0:
        return [(t0, t1, base_name)]
    chunk_len = (total + (n_splits - 1) * overlap_sec) / n_splits
    stride = chunk_len - overlap_sec
    if stride <= 0:
        stride = total / n_splits
        chunk_len = stride
    chunks = []
    for i in range(n_splits):
        cs = t0 + i * stride
        ce = min(cs + chunk_len, t1)
        chunks.append((cs, ce, base_name + str(i + 1)))
    return chunks


class SessionMemoryCache:
    """LRU eviction for the CCGDataset.ccg dict (keyed by Key with resolution)."""

    def __init__(self, ccg: dict):
        import collections as _col
        self._ccg = ccg
        self._order = _col.deque()

    def touch(self, key) -> None:
        try:
            self._order.remove(key)
        except (ValueError, AttributeError):
            pass
        self._order.append(key)

    def evict(self, key) -> None:
        self._ccg.pop(key, None)
        try:
            self._order.remove(key)
        except (ValueError, AttributeError):
            pass

    def estimated_mb(self) -> float:
        total = 0
        for cd_obj in self._ccg.values():
            for arr in (cd_obj.ccg, getattr(cd_obj, 'ccg_null', None)):
                if arr is not None:
                    total += arr.nbytes
        return total / 1024 ** 2

    def enforce_limit(self, limit_gb: float = 8.0) -> None:
        while self.estimated_mb() > limit_gb * 1024 and self._order:
            lru_key = self._order[0]
            if lru_key not in self._ccg:
                self._order.popleft()
                continue
            print(f"[CCGDataset] evicting {lru_key} "
                  f"(memory limit {limit_gb:.1f} GB)")
            self.evict(lru_key)


class IntervalOp:
    """Set algebra on interval lists [(t0, t1), ...]. All ops return sorted, non-overlapping lists."""

    @staticmethod
    def merge(intervals) -> list:
        """Merge overlapping/adjacent intervals."""
        iv = sorted((float(a), float(b)) for a, b in intervals if b > a)
        if not iv:
            return []
        out = [list(iv[0])]
        for t0, t1 in iv[1:]:
            if t0 <= out[-1][1]:
                out[-1][1] = max(out[-1][1], t1)
            else:
                out.append([t0, t1])
        return [tuple(x) for x in out]

    @staticmethod
    def clip(intervals, t_start=None, t_stop=None) -> list:
        """Clip interval list to [t_start, t_stop] bounds."""
        out = []
        for t0, t1 in intervals:
            if t_start is not None:
                t0 = max(t0, t_start)
            if t_stop is not None:
                t1 = min(t1, t_stop)
            if t0 < t1:
                out.append((t0, t1))
        return out

    @staticmethod
    def union(a: list, b: list) -> list:
        return IntervalOp.merge(list(a) + list(b))

    @staticmethod
    def intersect(a: list, b: list) -> list:
        out, i, j = [], 0, 0
        a, b = IntervalOp.merge(a), IntervalOp.merge(b)
        while i < len(a) and j < len(b):
            lo = max(a[i][0], b[j][0])
            hi = min(a[i][1], b[j][1])
            if lo < hi:
                out.append((lo, hi))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return out

    @staticmethod
    def difference(a: list, b: list) -> list:
        """Intervals in a not covered by b."""
        out = []
        a, b = IntervalOp.merge(a), IntervalOp.merge(b)
        j = 0
        for t0, t1 in a:
            cur = t0
            while j < len(b) and b[j][1] <= cur:
                j += 1
            k = j
            while k < len(b) and b[k][0] < t1:
                if cur < b[k][0]:
                    out.append((cur, b[k][0]))
                cur = max(cur, b[k][1])
                k += 1
            if cur < t1:
                out.append((cur, t1))
        return out

    @staticmethod
    def complement(a: list, t_start: float, t_end: float) -> list:
        return IntervalOp.difference([(t_start, t_end)], a)

    @staticmethod
    def duration(intervals) -> float:
        return sum(float(t1) - float(t0) for t0, t1 in intervals)

    @staticmethod
    def mask_spikes(spikes: np.ndarray, intervals) -> np.ndarray:
        """Keep only spikes falling within any of the intervals (binary search, fast)."""
        intervals = IntervalOp.merge(intervals)
        keep = []
        for t0, t1 in intervals:
            i0 = np.searchsorted(spikes, t0, 'left')
            i1 = np.searchsorted(spikes, t1, 'right')
            if i1 > i0:
                keep.append(spikes[i0:i1])
        return np.concatenate(keep) if keep else np.array([], dtype=spikes.dtype)


@dataclass(frozen=True)
class IntervalSet:
    """Named, immutable set of time intervals with set algebra."""
    label: str
    intervals: tuple                     # tuple of (t0, t1) float pairs
    tags: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, 'intervals',
                           tuple(IntervalOp.merge(self.intervals)))

    @property
    def active_duration(self) -> float:
        return IntervalOp.duration(self.intervals)

    def intersect(self, other: 'IntervalSet', label: str) -> 'IntervalSet':
        return IntervalSet(label, IntervalOp.intersect(self.intervals, other.intervals))

    def union(self, other: 'IntervalSet', label: str) -> 'IntervalSet':
        return IntervalSet(label, IntervalOp.union(self.intervals, other.intervals))

    def difference(self, other: 'IntervalSet', label: str) -> 'IntervalSet':
        return IntervalSet(label, IntervalOp.difference(self.intervals, other.intervals))

    def complement(self, t_start: float, t_end: float, label: str) -> 'IntervalSet':
        return IntervalSet(label, IntervalOp.complement(self.intervals, t_start, t_end))

    def clip(self, t_start: float, t_end: float) -> 'IntervalSet':
        return IntervalSet(self.label, IntervalOp.clip(self.intervals, t_start, t_end), self.tags)

    def mask_spikes(self, spikes: np.ndarray) -> np.ndarray:
        return IntervalOp.mask_spikes(spikes, self.intervals)

    @classmethod
    def from_arrays(cls, t_starts, t_stops, label: str) -> 'IntervalSet':
        return cls(label, list(zip(t_starts, t_stops)))

    def __hash__(self):
        return hash((self.label, self.intervals))

