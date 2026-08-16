import collections
import json
import os
import tempfile
import numpy as np
import math
from collections.abc import Iterable
from itertools import chain
from typing import Generic, TypeVar


def _json_numpy_default(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def atomic_write_json(path: str, data: dict = None, *, text: str = None) -> None:
    """Atomically write JSON to path (temp file + os.replace)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            if text is not None:
                fh.write(text)
            else:
                json.dump(data, fh, indent=2, default=_json_numpy_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Generic bounded LRU cache backed by OrderedDict."""

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max_size = max_size

    def get(self, key: K) -> 'V | None':
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def __contains__(self, key) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self):
        return iter(self._cache)

    def put(self, key: K, value: V) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def pop(self, key: K, default=None):
        return self._cache.pop(key, default)

    def clear(self) -> None:
        self._cache.clear()

    def resize(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


def find_nearest(array, value):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return array[idx - 1]
    else:
        return array[idx]


def arg_find_nearest(array, value):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def get_interval(period, nwindows):

    interval = np.linspace(period[0], period[1], nwindows + 1)
    interval = [[interval[i], interval[i + 1]] for i in range(nwindows)]
    return interval


def flatten(list_in):
    """Flatten a ragged list of different sized lists into one continuous list.
    Unlike `flatten_all` this only flattens the top level."""
    return list(chain.from_iterable(list_in))

def flatten_all(xs):
    """Completely flattens an iterable of iterables into one long generator"""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_all(x)
        else:
            yield x


def nan_helper(y: np.ndarray):
    """Helper to handle indices and logical indices of NaNs.
    From https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nans(y: np.ndarray):
    """interpolate nans based on values on either side of them in an array! In case of 2d array
    will move along rows"""

    if y.ndim == 2:
        for idr, yrow in enumerate(y):
            y[idr] = interp_nans(yrow)
    else:

        nans, x = nan_helper(y)
        if not np.all(np.isnan(y)):
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y
