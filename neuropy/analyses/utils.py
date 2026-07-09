import datetime
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
    """Base: state serialization helpers. Subclass HklSavable/NpzSavable/JsonSavable for I/O."""

    def __init__(self, ignored_attrs: list = []):
        self._ignored_attrs = ignored_attrs

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._ignored_attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def serialize(self) -> dict:
        return self.__getstate__()

    def save_path(self, **kwargs) -> str | None:
        return None

    @staticmethod
    def _to_state(obj):
        return obj.serialize() if isinstance(obj, Savable) else obj

    @classmethod
    def save_mapping(cls, mapping: dict, path: str) -> None:
        hkl.dump({k: cls._to_state(v) for k, v in mapping.items()}, path)

    @staticmethod
    def _ask_overwrite(path: str, label: str) -> bool:
        print(f"\n[{label}] file already exists at:\n  {path}")
        try:
            answer = input("  Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            print("y  (non-interactive)")
            answer = 'y'
        return answer in ('y', 'yes')


class HklSavable(Savable):
    """Savable backed by hickle (.hkl)."""

    def save(self, path: str = None, ignored_attrs: list = [],
             split_into_chunks=False, chunk_size_MB: int = 20):
        if ignored_attrs:
            self._ignored_attrs = _san(ignored_attrs)
        p = (path or self.save_path()) + '.hkl'
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        state = self.serialize()
        if not split_into_chunks:
            hkl.dump(state, p)
            return
        chunk_size = chunk_size_MB * 1024 * 1024
        folder = p if os.path.isdir(p) else p + "_files"
        os.makedirs(folder, exist_ok=True)
        tmp = os.path.join(folder, "temp.hkl")
        hkl.dump(state, tmp)
        with open(tmp, "rb") as f:
            i = 0
            while chunk := f.read(chunk_size):
                with open(os.path.join(folder, f"part{i}"), "wb") as out:
                    out.write(chunk)
                i += 1
        os.remove(tmp)

    def load(self, path: str = None):
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
                    part = os.path.join(folder, f"part{i}")
                    if not os.path.exists(part):
                        break
                    with open(part, "rb") as pf:
                        out.write(pf.read())
                    i += 1
        try:
            loaded = hkl.load(file)
            if isinstance(loaded, dict) and not isinstance(loaded, Savable):
                self.__setstate__(loaded)
            else:
                for k, v in loaded.__dict__.items():
                    setattr(self, k, v)
        except Exception as e:
            print(f"Failed to load {self.__class__}: {e}")
        finally:
            if splitted:
                os.remove(file)

    @property
    def is_saved(self) -> bool:
        return os.path.isfile(self.save_path() + '.hkl')

    def find_cached(self, suffix='') -> str:
        p = self.save_path(suffix=suffix) + '.hkl'
        if not os.path.isfile(p):
            return 'missing'
        return 'ok' if self.check_cache() else 'stale'

    def check_cache(self, group='key') -> bool:
        p = self.save_path() + '.hkl'
        if not os.path.isfile(p):
            return False
        try:
            saved = hkl.load(p)
            fields = self._groups.get(group, [])
            _sv = getattr(self.__class__, 'serialize_value', lambda v: v)
            return all(_sv(getattr(saved, f, None)) == _sv(getattr(self, f, None))
                       for f in fields)
        except Exception:
            return False


class NpzSavable(Savable):
    """Savable backed by numpy .npz."""

    def save(self, path: str = None, **_):
        p = (path or self.save_path()) + '.npz'
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        state = {k: v for k, v in self.serialize().items() if isinstance(v, np.ndarray)}
        np.savez_compressed(p, **state)

    def load(self, path: str = None):
        p = (path or self.save_path()) + '.npz'
        if not os.path.exists(p):
            print(f"File not found: {p}")
            return
        self.__setstate__(dict(np.load(p, allow_pickle=False)))

    @property
    def is_saved(self) -> bool:
        return os.path.isfile(self.save_path() + '.npz')


def _json_key(k):
    """Serialize a non-string dict key to a JSON-native form."""
    if isinstance(k, (str, int, float, bool)) or k is None:
        return k
    if isinstance(k, (list, tuple)):
        return [_to_json(x) for x in k]
    return str(k)


def _to_json(v):
    """Recursively convert a value to a JSON-serializable form.

    JsonSavable with save_path() → saves to own file, returns {"__ref__": path}.
    Other Savable                → inline serialize().
    set / frozenset              → sorted plain list.
    dict with 2-int-tuple keys   → {"r,t": value} str-key dict.
    dict with other non-str keys → [[key, value], ...] list-of-pairs.
    """
    if isinstance(v, JsonSavable):
        sp = v.save_path()
        if sp is not None:
            v.save()
            return {"__ref__": sp + '.json'}
        return v.serialize()
    if isinstance(v, Savable):
        return v.serialize()
    if isinstance(v, dict):
        if v and not all(isinstance(k, str) for k in v):
            if all(isinstance(k, (tuple, list)) and len(k) == 2
                   and all(isinstance(x, (int, np.integer)) for x in k)
                   for k in v):
                return {f"{k[0]},{k[1]}": _to_json(v2) for k, v2 in v.items()}
            return [[_json_key(k), _to_json(v2)] for k, v2 in v.items()]
        return {str(k): _to_json(v2) for k, v2 in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted(_to_json(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_to_json(x) for x in v]
    return v


def _from_json(v):
    """Reverse _to_json. Handles legacy __set__/__dict__ tags for backward compat."""
    if isinstance(v, dict):
        if '__set__' in v:
            return {tuple(x) if isinstance(x, list) else x for x in v['__set__']}
        if '__dict__' in v:
            return {(tuple(k) if isinstance(k, list) else k): _from_json(val)
                    for k, val in v['__dict__']}
        return {k: _from_json(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_from_json(x) for x in v]
    return v


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _compact_json_str(obj) -> str:
    """JSON with indent=2, but short arrays and flat objects kept on one line."""
    s = json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)
    # Compact [int, int] pair arrays
    s = re.sub(r'\[\s*(-?\d+),\s*(-?\d+)\s*\]', r'[\1, \2]', s)
    # Compact arrays of only string literals
    def _join_str_array(m):
        inner = re.sub(r'\s+', ' ', m.group(1)).strip().rstrip(',')
        return f'[{inner}]'
    s = re.sub(r'\[\s*((?:"[^"]*",?\s*)+)\s*\]', _join_str_array, s)
    # Compact flat objects (no nested {}) that fit within 100 chars
    def _compact_obj(m):
        flat = re.sub(r'\s+', ' ', m.group(0)).strip()
        return flat if len(flat) <= 100 else m.group(0)
    s = re.sub(r'\{[^{}]*\}', _compact_obj, s, flags=re.DOTALL)
    return s


class JsonSavable(Savable):
    """Savable backed by JSON (atomic write).

    serialize(): walks __dict__, skips _-prefixed and explicitly ignored attrs.
    __setstate__(): reverse; use _custom_types = {'field': Type} for dict-value
                    reconstruction when Type.__init__() is zero-arg (or all-defaulted).
    _ignored_attrs: list of non-_-prefixed attrs to exclude (pass to __init__ or set as
                    class attr).
    """

    _custom_types: dict = {}

    def __init__(self, ignored_attrs: list = []):
        self._ignored_attrs = list(ignored_attrs)

    def _public_state(self) -> dict:
        ignored = getattr(self, '_ignored_attrs', [])
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k not in ignored}

    def serialize(self) -> dict:
        return {k: _to_json(v) for k, v in self._public_state().items()}

    def __setstate__(self, state: dict) -> None:
        """Restore state from a JSON-loaded dict.

        _custom_types = {'field': Type or (key_type, value_type)}
            Reconstructs dict values as typed Savable objects.
            key_type must have from_str(s) classmethod; omit for str keys.
        _set_fields: frozenset of field names that are sets of tuples.
        _pair_key_dict_fields: frozenset of field names whose str keys are "r,t" → (r,t).
        {"__ref__": path} on a JsonSavable field calls load() on the existing instance.
        """
        ignored = getattr(self, '_ignored_attrs', [])
        custom: dict = {}
        set_fields: set = set()
        pair_key_dict_fields: set = set()
        for cls in type(self).__mro__:
            custom.update(getattr(cls, '_custom_types', {}))
            set_fields |= getattr(cls, '_set_fields', set())
            pair_key_dict_fields |= getattr(cls, '_pair_key_dict_fields', set())
        for k, v in state.items():
            if k.startswith('_') or k in ignored:
                continue
            if isinstance(v, dict) and '__ref__' in v:
                existing = getattr(self, k, None)
                if isinstance(existing, JsonSavable):
                    existing.load(v['__ref__'][:-len('.json')])
                continue
            v = _from_json(v)
            if k in set_fields and isinstance(v, list):
                setattr(self, k, {tuple(x) if isinstance(x, list) else x for x in v})
                continue
            if k in pair_key_dict_fields and isinstance(v, dict):
                setattr(self, k, {
                    tuple(int(i) for i in dk.split(',')): _from_json(dv)
                    for dk, dv in v.items()
                })
                continue
            vtype = custom.get(k)
            if vtype is not None and isinstance(v, dict):
                ktype, vtype_inner = vtype if isinstance(vtype, tuple) else (None, vtype)
                existing = getattr(self, k, None)
                result = defaultdict(vtype_inner) if isinstance(existing, defaultdict) else {}
                for dk, dv in v.items():
                    key = ktype.from_str(dk) if ktype is not None else dk
                    obj = vtype_inner.__new__(vtype_inner)
                    vtype_inner.__init__(obj)
                    obj.__setstate__(dv if isinstance(dv, dict) else {})
                    result[key] = obj
                setattr(self, k, result)
            else:
                setattr(self, k, v)

    def save(self, path: str = None, **_):
        from neuropy.utils.data_storage_util import atomic_write_json
        p = (path or self.save_path()) + '.json'
        atomic_write_json(p, text=_compact_json_str(self.serialize()))

    def load(self, path: str = None):
        p = (path or self.save_path()) + '.json'
        if not os.path.exists(p):
            print(f"File not found: {p}")
            return
        with open(p, encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict) and 'selections' in raw:
            sel = raw['selections']
            sample_tags = {}
            for k, v in (list(sel.items())[:2] if isinstance(sel, dict) else []):
                if isinstance(v, dict):
                    sample_tags[k] = {sk: sv for sk, sv in list(v.items())[:3]}
        self.__setstate__(raw)

    @property
    def is_saved(self) -> bool:
        return os.path.isfile(self.save_path() + '.json')


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


class Config(JsonSavable):

    _groups: dict = {}

    def check_cache(self, group: str = 'key') -> bool:
        """True if saved JSON matches current key fields."""
        p = self.save_path() + '.json'
        if not os.path.isfile(p):
            return False
        try:
            with open(p, encoding='utf-8') as f:
                saved = json.load(f)
            fields = self._groups.get(group, [])
            sv = self.__class__.serialize_value
            return all(sv(saved.get(f)) == sv(getattr(self, f, None)) for f in fields)
        except Exception:
            return False

    def find_cached(self) -> str:
        p = self.save_path() + '.json'
        if not os.path.isfile(p):
            return 'missing'
        return 'ok' if self.check_cache() else 'stale'

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
    def serialize_value(v):
        """JSON-safe config value for cache metadata."""
        if hasattr(v, 'name'):
            return v.name
        if isinstance(v, (list, tuple)):
            return [Config.serialize_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): Config.serialize_value(val) for k, val in v.items()}
        try:
            import json as _json
            _json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)


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
        # self.data: Dict[K, Any] = {}
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


class Autosave:
    """Mixin: periodic autosave + purge. Parallel to Savable, not type-specific.

    Config (override as class or instance attrs):
        autosave_subdir          : str = '.autosave'
        autosave_interval_minutes: int = 30
        autosave_retain_days     : int = 7
        autosave_suffix          : str = '.json'
    """
    autosave_subdir          : str = '.autosave'
    autosave_interval_minutes: int = 30
    autosave_retain_days     : int = 7
    autosave_suffix          : str = '.json'

    def autosave_base_dir(self) -> str:
        sp = self.save_path()  # type: ignore[attr-defined]
        return os.path.dirname(sp) if sp else '.'

    def autosave_label(self) -> str:           raise NotImplementedError
    def do_autosave(self, path: str) -> None:
        p = path[:-len(self.autosave_suffix)] if path.endswith(self.autosave_suffix) else path
        self.save(p)  # type: ignore[attr-defined]

    def autosave_active(self) -> bool:         return True

    def schedule_autosave(self) -> None:
        from pyqtgraph.Qt.QtCore import QTimer
        t = QTimer()
        t.setInterval(self.autosave_interval_minutes * 60 * 1000)
        t.timeout.connect(lambda: self.write_autosave('.periodic') if self.autosave_active() else None)
        t.start()

    @property
    def autosave_dir(self) -> str:
        return os.path.join(self.autosave_base_dir(), self.autosave_subdir)

    def write_autosave(self, suffix: str = '') -> str:
        hdir = self.autosave_dir
        os.makedirs(hdir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        fname = f"{self.autosave_label()}__{ts}{suffix}{self.autosave_suffix}"
        path = os.path.join(hdir, fname)
        self.do_autosave(path)
        return path

    def write_autosave_fixed(self, filename: str) -> None:
        hdir = self.autosave_dir
        os.makedirs(hdir, exist_ok=True)
        self.do_autosave(os.path.join(hdir, filename))

    def purge_autosaves(self) -> None:
        hdir = self.autosave_dir
        if not os.path.isdir(hdir):
            return
        cutoff = datetime.datetime.now() - datetime.timedelta(days=self.autosave_retain_days)
        removed = 0
        for fname in os.listdir(hdir):
            if not fname.endswith(self.autosave_suffix):
                continue
            fpath = os.path.join(hdir, fname)
            try:
                if datetime.datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                pass
        if removed:
            print(f"[Autosave] purged {removed} files older than {self.autosave_retain_days} days")


class UndoRedo:
    """Mixin: undo/redo command stack. Parallel to Savable, not type-specific.

    Config:
        undo_limit: int = 50
    """
    undo_limit: int = 50

    def apply_command(self, cmd, reverse: bool = False) -> None: raise NotImplementedError

    def __init_undo__(self):
        self._undo_stack: list = []
        self._redo_stack: list = []

    def push_undo(self, cmd) -> None:
        self._undo_stack.append(cmd)
        if len(self._undo_stack) > self.undo_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        self._redo_stack.append(cmd)
        self.apply_command(cmd, reverse=True)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        self._undo_stack.append(cmd)
        self.apply_command(cmd, reverse=False)


from neuropy.core.intervals import IntervalOp, IntervalSet  # noqa: F401
