"""CCG feature extraction: residual traces, derivatives, and scalar shape descriptors.

Every model here works on the *residual* ``ccg - null`` rather than raw counts —
the null already absorbs firing rate and slow co-modulation, so the residual is
what the eye actually judges. Derivatives are carried alongside because peak
sharpness (a 1 ms msconn peak vs. a broad rhythm hump) is a slope property, not
an amplitude one.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def residual(ccg: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Baseline-subtracted CCG, normalized so shape (not rate) drives the model."""
    res = ccg - null
    scale = np.sqrt(np.maximum(null, 1.0))       # Poisson noise of the baseline
    return res / scale


def smooth(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter1d(x, sigma, axis=-1, mode='nearest')


def derivative(x: np.ndarray, order: int = 1) -> np.ndarray:
    """*order*-th finite difference along the lag axis, length preserved."""
    out = x
    for _ in range(order):
        out = np.gradient(out, axis=-1)
    return out


def lag_axis(n_bin: int, duration: float) -> np.ndarray:
    """Bin centers in ms for a symmetric ±duration/2 window."""
    half = duration * 1e3 / 2
    return np.linspace(-half, half, n_bin)


def trace_stack(ccg: np.ndarray, null: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Per-pair channel stack ``[n, 3, n_bin]`` = residual, its slope, its curvature."""
    res = smooth(residual(ccg, null), sigma)
    return np.stack([res, derivative(res, 1), derivative(res, 2)], axis=-2)


# --- scalar descriptors -----------------------------------------------------

def _flanks(n_bin: int) -> np.ndarray:
    """Mask of the outer thirds — the part of the window treated as baseline."""
    m = np.zeros(n_bin, dtype=bool)
    edge = max(1, n_bin // 3)
    m[:edge] = m[-edge:] = True
    return m


def shape_features(ccg: np.ndarray, null: np.ndarray, duration: float) -> np.ndarray:
    """Interpretable per-pair descriptors ``[n, 14]``.

    These name the things the user's own group notes describe: peak height and
    lag, peak width, symmetry about 0 ms, rifts (dips flanking a peak), and
    oscillation. They double as the diagnostic axes used to verify predictions.
    """
    ccg = np.atleast_2d(ccg)
    null = np.atleast_2d(null)
    res = smooth(residual(ccg, null), 1.0)
    n_bin = res.shape[1]
    lags = lag_axis(n_bin, duration)
    d1 = derivative(res, 1)
    zero = n_bin // 2

    flank = _flanks(n_bin)
    noise = res[:, flank].std(axis=1) + 1e-9

    peak_i = res.argmax(axis=1)
    peak = res.max(axis=1)
    trough = res.min(axis=1)
    rows = np.arange(len(res))

    # Width at half the peak, measured in bins either side of the maximum.
    half = peak[:, None] / 2
    above = res >= half
    width = above.sum(axis=1).astype(float)

    pos = res[:, zero + 1:].sum(axis=1)
    neg = res[:, :zero].sum(axis=1)
    total = np.abs(pos) + np.abs(neg) + 1e-9

    # Rhythm: strongest non-DC component of the residual's spectrum.
    spec = np.abs(np.fft.rfft(res - res.mean(axis=1, keepdims=True), axis=1))
    rhythm = spec[:, 1:].max(axis=1) / (spec[:, 1:].sum(axis=1) + 1e-9)

    return np.column_stack([
        peak / noise,                                  # peak SNR
        trough / noise,                                # deepest dip SNR
        lags[peak_i],                                  # peak lag (ms)
        width,                                         # peak width (bins)
        np.abs(d1).max(axis=1) / noise,                # max slope — sharpness
        (pos - neg) / total,                           # asymmetry about 0 ms
        res[:, zero] / noise,                          # exact 0 ms bin (leak)
        res[:, max(zero - 1, 0):zero + 2].max(axis=1) / noise,   # ±1 ms (msconn)
        rhythm,                                        # oscillation index
        np.sign(peak + trough),                        # excitatory vs inhibitory
        res.std(axis=1) / noise,                       # overall deviation
        ccg.sum(axis=1),                               # raw co-firing count
        null.mean(axis=1),                             # baseline level
        (res > 2 * noise[:, None]).sum(axis=1).astype(float),    # bins above 2σ
    ])


FEATURE_NAMES = [
    'peak_snr', 'trough_snr', 'peak_lag_ms', 'peak_width_bins', 'max_slope',
    'asymmetry', 'zero_bin', 'ms_peak', 'rhythm_index', 'sign',
    'deviation', 'total_count', 'baseline', 'bins_above_2sd',
]
