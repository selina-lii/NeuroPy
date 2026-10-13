"""Spike-pair events of one pair, attributed to CCG bins once so playback is a slice per frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairEvents:
    ref: np.ndarray
    tgt: np.ndarray
    times: np.ndarray
    bins: np.ndarray
    n_bins: int
    bin_size: float
    ref_idx: np.ndarray
    tgt_idx: np.ndarray

    @classmethod
    def build(cls, ref_t, tgt_t, bin_size: float, n_bins: int, sampling_rate: float) -> 'PairEvents':
        """Every spike pair inside the CCG window, binned on samples like spike_correlations and timed by its later spike."""
        ref = np.sort(np.asarray(ref_t, dtype=float))
        tgt = np.sort(np.asarray(tgt_t, dtype=float))
        rs, ts = (ref * sampling_rate).astype(np.int64), (tgt * sampling_rate).astype(np.int64)
        b, half = int(sampling_rate * bin_size), n_bins // 2
        reach = (half + 1) * b
        lo = np.searchsorted(ts, rs - reach, side='left')
        hi = np.searchsorted(ts, rs + reach, side='right')
        counts = hi - lo
        ri = np.repeat(np.arange(len(ref)), counts)
        ti = np.repeat(lo - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
        d = ts[ti] - rs[ri]
        k = (np.abs(d) + b // 2) // b
        keep = k <= half
        bins = (half + np.sign(d) * k)[keep]
        times = np.maximum(ref[ri], tgt[ti])[keep]
        order = np.argsort(times, kind='stable')
        return cls(ref, tgt, times[order], bins[order], n_bins, bin_size, ri[keep][order], ti[keep][order])

    @property
    def span(self) -> tuple[float, float]:
        both = np.concatenate([self.ref[:1], self.tgt[:1], self.ref[-1:], self.tgt[-1:]])
        return (float(both.min()), float(both.max())) if both.size else (0.0, 0.0)

    def hist(self, t0: float, t1: float) -> np.ndarray:
        """The CCG of the spike pairs completed in (t0, t1]."""
        a, b = np.searchsorted(self.times, [t0, t1], side='right')
        return np.bincount(self.bins[a:b], minlength=self.n_bins)
