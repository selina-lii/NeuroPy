"""Spike attribution for CCG bins.

Identifies the specific (ref_spike, tgt_spike) pairs that contribute to a
given lag bin in a cross-correlogram.
"""
from __future__ import annotations

import numpy as np

RASTER_WINDOW_SEC = 0.05   # ±50 ms half-width for spike-pair raster plots


def bin_ms_from_input(bin_val: float, unit: str, ccg_data) -> float:
    """Convert UI bin input to lag centre in milliseconds."""
    conf = ccg_data.conf
    n_bins = ccg_data.ccg.shape[-1]
    bin_size_eff = (conf.duration / (n_bins - 1)
                    if n_bins > 1 else conf.bin_size)
    if unit == '#':
        return float(bin_val) * bin_size_eff * 1000.0
    return float(bin_val)


def compute_spike_pairs(
    neurons,
    ref: int,
    tgt: int,
    bin_val: float,
    unit: str,
    ccg_data,
    t0: float | None = None,
    t1: float | None = None,
) -> list[tuple[float, float]]:
    """Find (ref_t, tgt_t) pairs contributing to a CCG bin for the current pair.

    ``t0``/``t1`` bound the segment window (seconds); ``None`` = whole session."""
    if neurons is None or ccg_data is None:
        return []
    if ref >= len(neurons.spiketrains) or tgt >= len(neurons.spiketrains):
        return []
    conf = ccg_data.conf
    n_bins = ccg_data.ccg.shape[-1]
    bin_size_eff = (conf.duration / (n_bins - 1)
                    if n_bins > 1 else conf.bin_size)
    bin_ms = bin_ms_from_input(bin_val, unit, ccg_data)
    return find_spike_pairs(
        np.asarray(neurons.spiketrains[ref]),
        np.asarray(neurons.spiketrains[tgt]),
        bin_ms, bin_size_eff, t0=t0, t1=t1,
    )


def find_spike_pairs(
    ref_spikes: np.ndarray,
    tgt_spikes: np.ndarray,
    bin_ms: float,
    bin_size_sec: float,
    t0: float | None = None,
    t1: float | None = None,
) -> list[tuple[float, float]]:
    """Find (ref_t, tgt_t) spike pairs whose lag falls in the CCG bin at bin_ms.

    Parameters
    ----------
    ref_spikes, tgt_spikes : spike time arrays (seconds)
    bin_ms : bin centre lag in milliseconds
    bin_size_sec : bin width in seconds
    t0, t1 : optional segment bounds (seconds); spikes outside are excluded
    """
    lag_sec = bin_ms / 1000.0
    bin_lo = lag_sec - bin_size_sec / 2.0
    bin_hi = lag_sec + bin_size_sec / 2.0

    if t0 is not None and t1 is not None:
        ref_spikes = ref_spikes[(ref_spikes >= t0) & (ref_spikes <= t1)]
        tgt_spikes = tgt_spikes[(tgt_spikes >= t0) & (tgt_spikes <= t1)]

    pairs: list[tuple[float, float]] = []
    tgt_sorted = np.sort(tgt_spikes)
    for rt in ref_spikes:
        idx_lo = np.searchsorted(tgt_sorted, rt + bin_lo, side='left')
        idx_hi = np.searchsorted(tgt_sorted, rt + bin_hi, side='right')
        for j in range(idx_lo, idx_hi):
            pairs.append((float(rt), float(tgt_sorted[j])))
    return pairs
