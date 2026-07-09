"""Core interval set algebra and named interval containers."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


class IntervalOp:
    """Set algebra on interval lists [(t0, t1), ...]. All ops return sorted, non-overlapping lists."""

    @staticmethod
    def merge(intervals) -> list:
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
        intervals = IntervalOp.merge(intervals)
        keep = []
        for t0, t1 in intervals:
            i0 = np.searchsorted(spikes, t0, 'left')
            i1 = np.searchsorted(spikes, t1, 'right')
            if i1 > i0:
                keep.append(spikes[i0:i1])
        return np.concatenate(keep) if keep else np.array([], dtype=spikes.dtype)

    @staticmethod
    def partition(t0: float, t1: float, n_splits: int, overlap_sec: float = 0.0,
                  base_name: str = '') -> list[tuple[float, float, str]]:
        """Partition [t0, t1] into n_splits overlapping chunks → [(t0, t1, name), ...]."""
        n_splits = max(1, int(n_splits))
        overlap_sec = max(0.0, float(overlap_sec))
        total = t1 - t0
        if n_splits == 1 or total <= 0:
            return [(t0, t1, base_name)]
        chunk_len = (total + (n_splits - 1) * overlap_sec) / n_splits
        stride = max(chunk_len - overlap_sec, total / n_splits)
        chunk_len = stride if stride == total / n_splits else chunk_len
        return [(t0 + i * stride,
                 min(t0 + i * stride + chunk_len, t1),
                 base_name + str(i + 1))
                for i in range(n_splits)]

    @staticmethod
    def split_n(t0: float, t1: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Equal n-way split → (starts, stops) arrays."""
        edges = np.linspace(t0, t1, n + 1)
        return edges[:-1], edges[1:]

    @staticmethod
    def sliding_windows(t0: float, t1: float, stride: float, seg_len: float,
                        keep_incomplete: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Sliding window → (starts, stops) arrays."""
        if seg_len >= t1 - t0:
            return np.array([t0]), np.array([t1])
        starts = np.arange(t0, t1 - seg_len + 1, stride)
        stops = starts + seg_len
        if keep_incomplete and stops[-1] < t1:
            starts = np.append(starts, starts[-1] + stride)
            stops = np.append(stops, min(t1, starts[-1] + seg_len))
        return starts, stops


@dataclass(frozen=True)
class IntervalSet:
    """Named, immutable set of time intervals with set algebra."""
    label: str
    intervals: tuple
    tags: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, 'intervals', tuple(IntervalOp.merge(self.intervals)))

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
