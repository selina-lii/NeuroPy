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
from sklearn.decomposition import PCA


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


# --- localized kernel bank --------------------------------------------------
#
# The alternative to a PCA basis. PCA components are whole-window templates
# chosen to explain dataset variance, so a 0.3 ms peak at lag +1 gets smeared
# across many components and mixed with whatever covaries with it. Visual
# inspection instead judges *local* shape — a peak's width, a dip's depth, where
# a rift sits relative to 0 ms — which is what `PeakRule(lag, width, polarity)`
# in the older template classifier encoded by hand.
#
# So the basis here is a fixed bank of Gaussian derivatives placed at a grid of
# lags and widths. Fixed means nothing is fitted, so it cannot overfit n≈3000;
# localized means a narrow feature lights up a few coefficients whose lag and
# width are known, keeping the representation readable.

# (width in ms, order): order 0 matches a bump, 1 an edge/slope, 2 a
# peak-with-flanking-dips — the rift/2side_dips shape.
KERNEL_WIDTHS_MS = (0.2, 0.5, 1.0, 2.0, 4.0)
KERNEL_ORDERS = (0, 1, 2)


def _gaussian_derivative(x: np.ndarray, sigma: float, order: int) -> np.ndarray:
    """Normalized *order*-th derivative of a Gaussian over lag offsets *x* (ms)."""
    g = np.exp(-0.5 * (x / sigma) ** 2)
    if order == 1:
        g = -(x / sigma ** 2) * g
    elif order == 2:
        g = ((x ** 2 - sigma ** 2) / sigma ** 4) * g
    norm = np.sqrt(np.sum(g ** 2))
    return g / norm if norm > 0 else g


def kernel_bank(n_bin: int, duration: float, lag_step_ms: float = 0.5,
                widths_ms=KERNEL_WIDTHS_MS, orders=KERNEL_ORDERS):
    """Matched-filter bank ``([n_kernel, n_bin], [(lag, width, order), ...])``.

    Kernels narrower than the bin size are dropped — they cannot be represented
    at that resolution, so including them would only inject noise.
    """
    lags = lag_axis(n_bin, duration)
    bin_ms = float(lags[1] - lags[0]) if n_bin > 1 else 1.0
    half = duration * 1e3 / 2
    centers = np.arange(-half, half + 1e-9, lag_step_ms)
    rows, meta = [], []
    for width in widths_ms:
        if width < bin_ms:
            continue
        for order in orders:
            for c in centers:
                rows.append(_gaussian_derivative(lags - c, width, order))
                meta.append((float(c), float(width), int(order)))
    return np.asarray(rows), meta


def kernel_response(ccg: np.ndarray, null: np.ndarray, duration: float,
                    bank: np.ndarray = None, **kw) -> np.ndarray:
    """Project each pair's residual onto the kernel bank → ``[n, n_kernel]``.

    Each column is "how strongly this pair shows a feature of this width at this
    lag" — a direct numeric read of the judgment the peak rules described.
    """
    ccg = np.atleast_2d(ccg)
    null = np.atleast_2d(null)
    res = residual(ccg, null)
    if bank is None:
        bank, _ = kernel_bank(res.shape[1], duration, **kw)
    return res @ bank.T


def sliding_windows(res: np.ndarray, width: int, stride: int) -> np.ndarray:
    """Overlapping patches ``[n * n_pos, width]`` cut from each trace.

    Learning a filter bank from patches rather than whole traces is what keeps
    the parameter count low enough for ~3000 samples: a whole-window basis needs
    as many weights as bins, while a patch basis reuses one small filter at every
    lag and so sees ``n_pos`` times more training examples per filter.
    """
    starts = range(0, res.shape[1] - width + 1, stride)
    return np.concatenate([res[:, s:s + width] for s in starts], axis=0)


def learned_bank(res: np.ndarray, n_filters: int, width: int, stride: int,
                 random_state: int = 0) -> np.ndarray:
    """Filters discovered from the data by PCA over sliding patches ``[k, width]``.

    Unlike a PCA basis over whole traces, these components are *localized*: each
    describes a shape of ``width`` bins that can occur at any lag, so the basis
    is translation-covariant the way a CNN's first layer is, without needing the
    ~80k training pairs a full CNN does.
    """
    patches = sliding_windows(res, width, stride)
    k = min(n_filters, patches.shape[0], width)
    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(patches - patches.mean(axis=1, keepdims=True))
    return pca.components_


def bank_response(res: np.ndarray, bank: np.ndarray, stride: int) -> np.ndarray:
    """Convolve *bank* over each trace and pool ``[n, k*3]``.

    Max, mean, and argmax-lag per filter: what the strongest match was, how
    consistently it appears, and *where* — the lag that the peak rules named
    explicitly and that plain max-pooling would discard.
    """
    n, n_bin = res.shape
    width = bank.shape[1]
    starts = np.arange(0, n_bin - width + 1, stride)
    # [n, n_pos, width] windows -> [n, n_pos, k] activations
    windows = np.stack([res[:, s:s + width] for s in starts], axis=1)
    act = windows @ bank.T
    lags = starts / max(len(starts) - 1, 1)
    return np.hstack([act.max(axis=1), act.mean(axis=1),
                      lags[act.argmax(axis=1)]])


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
