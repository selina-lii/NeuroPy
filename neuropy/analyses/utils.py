import numpy as np
from dataclasses import dataclass, field, replace
from typing import Union, Optional, Dict, Any, Tuple, TypeVar, Type
from collections import defaultdict
import hickle as hkl
import os


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

    def save_path(self, **kwargs):
        return "./tmp"

    def save(self,
             path: str = None,
             ignored_attrs: list = [],
             split_into_chunks=False,
             chunk_size_MB: int = 20):
        self._ignored_attrs = _san(ignored_attrs)
        p = (path or self.save_path()) + '.hkl'

        if not split_into_chunks:
            hkl.dump(self, p)
        else:
            chunk_size = chunk_size_MB * 1024 * 1024
            folder = p
            folder = p if os.path.isdir(p) else p + "_files"
            os.makedirs(folder, exist_ok=True)
            file = os.path.join(folder, "temp.hkl")

            hkl.dump(self, file)

            with open(file, "rb") as f:
                i = 0
                while chunk := f.read(chunk_size):
                    with open(os.path.join(folder, f"part{i}"), "wb") as out:
                        out.write(chunk)
                    i += 1
            os.remove(file)

    def load(self, path: str = None):
        p = (path or self.save_path()) + '.hkl'
        splitted = False
        file = p  # default; overwritten below if loading from chunks
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

        # Load object from combined file
        try:
            obj = hkl.load(file)
            for k, v in obj.__dict__.items():
                setattr(self, k, v)
        except Exception as e:
            print(f"Failed to load {self.__class__} object: {e}")
        finally:
            if splitted:
                os.remove(file)


class Config(Savable):

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.filepath}\n"
        return s


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
        return SetOp.__set_op(x, y, np.intersect1d)

    @staticmethod
    def setdiff(x, y):  #n2=None
        """
        X minus Y for two N-dim arrays by their row elements
        """
        # Set difference of coordinate lists
        if x is None or y is None:
            return x if x is not None else np.array([])
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
        return SetOp.__set_op(x, y, np.union1d)

    @staticmethod
    def unique(x):
        """
        np.unique by row elements
        """
        return np.unique(x, axis=0)
