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


# --- peak structure: counts and spacings, which shape_features' one argmax cannot express ---

MAX_PEAKS = 3


def find_peaks(res: np.ndarray, noise: np.ndarray,
               min_prominence: float = 1.5) -> list[np.ndarray]:
    """Local maxima at least *min_prominence* noise above baseline, per row."""
    left = res[:, 1:-1] > res[:, :-2]
    right = res[:, 1:-1] >= res[:, 2:]
    tall = res[:, 1:-1] >= (min_prominence * noise)[:, None]
    hit = left & right & tall
    return [np.flatnonzero(row) + 1 for row in hit]


def _half_width(row: np.ndarray, idx: int) -> float:
    """Width in bins where the trace stays above half this peak's height."""
    half = row[idx] / 2.0
    lo = hi = idx
    while lo > 0 and row[lo - 1] >= half:
        lo -= 1
    while hi < len(row) - 1 and row[hi + 1] >= half:
        hi += 1
    return float(hi - lo) or 1.0


def peak_features(ccg: np.ndarray, null: np.ndarray, duration: float,
                  sigma: float = 0.4) -> np.ndarray:
    """Multi-peak structure ``[n, 17]`` — counts, spacings, widths, and a mirrored trough block."""
    ccg = np.atleast_2d(ccg)
    null = np.atleast_2d(null)
    # sigma is swept, not fixed: at 1.0 two peaks 2 bins apart merge
    res = smooth(residual(ccg, null), sigma)
    n_bin = res.shape[1]
    lags = lag_axis(n_bin, duration)
    noise = res[:, _flanks(n_bin)].std(axis=1) + 1e-9

    out = np.zeros((len(res), 5 + 3 * MAX_PEAKS + 3))
    troughs = find_peaks(-res, noise)
    for i, idx in enumerate(find_peaks(res, noise)):
        row = res[i]
        order = idx[np.argsort(-row[idx])][:MAX_PEAKS] if len(idx) else idx
        out[i, 0] = len(idx)
        for k, p in enumerate(order):
            out[i, 5 + 3 * k:8 + 3 * k] = (lags[p], row[p] / noise[i],
                                           _half_width(row, p))
        if len(idx) >= 2:
            spacing = np.diff(np.sort(lags[idx]))
            out[i, 1] = spacing.mean()
            # 0 = perfectly regular, which is what a rhythmic train looks like.
            out[i, 2] = spacing.std() / (spacing.mean() + 1e-9)
            two = np.sort(order[:2])
            if len(two) == 2:
                out[i, 3] = lags[two[1]] - lags[two[0]]
                # a sharp dip separates two peaks; a shallow one is one broad hump
                between = row[two[0]:two[1] + 1]
                out[i, 4] = (row[two].min() - between.min()) / noise[i]
        tr = troughs[i]
        if len(tr):
            deepest = tr[np.argmin(row[tr])]
            out[i, -3:] = (len(tr), lags[deepest], -row[deepest] / noise[i])
    return out


PEAK_FEATURE_NAMES = (
    ['n_peaks', 'peak_spacing_mean', 'peak_spacing_cv', 'top2_separation',
     'dip_between_peaks']
    + [f'peak{k}_{f}' for k in range(MAX_PEAKS) for f in ('lag', 'snr', 'width')]
    + ['n_troughs', 'trough_lag', 'trough_snr'])


def flank_dip_features(ccg: np.ndarray, null: np.ndarray, duration: float,
                       sigma: float = 0.4) -> np.ndarray:
    """Dips either side of the main peak ``[n, 7]``; sides kept separate so two-sided is distinguishable."""
    ccg = np.atleast_2d(ccg)
    null = np.atleast_2d(null)
    res = smooth(residual(ccg, null), sigma)
    n_bin = res.shape[1]
    lags = lag_axis(n_bin, duration)
    noise = res[:, _flanks(n_bin)].std(axis=1) + 1e-9

    out = np.zeros((len(res), 7))
    for i, row in enumerate(res):
        peak = int(np.argmax(row))
        depth, width, lag = [], [], []
        for side in (slice(0, peak), slice(peak + 1, n_bin)):
            seg = row[side]
            if not len(seg) or seg.min() >= 0:
                depth.append(0.0)
                width.append(0.0)
                lag.append(0.0)
                continue
            k = int(np.argmin(seg))
            depth.append(-seg[k] / noise[i])
            # broad for the soft scoop this label names, narrow for a notch
            width.append(float((seg <= seg[k] / 2).sum()))
            lag.append(abs(lags[np.arange(n_bin)[side][k]] - lags[peak]))
        both = min(depth)
        out[i] = (depth[0], depth[1], both,
                  min(depth) / (max(depth) + 1e-9),      # symmetry of the pair
                  width[0] + width[1],
                  (lag[0] + lag[1]) / 2,
                  float(both > 1.0))                     # present on both sides
    return out


FLANK_FEATURE_NAMES = ['dip_left_snr', 'dip_right_snr', 'dip_both_snr',
                       'dip_symmetry', 'dip_total_width', 'dip_mean_offset_ms',
                       'dip_two_sided']


WINDOW_MS = ((1.0, 3.0), (0.0, 0.5), (0.5, 1.0), (3.0, 5.0), (5.0, 10.0))


def window_features(ccg: np.ndarray, null: np.ndarray, duration: float,
                    windows=None, sigma: float = 0.4) -> np.ndarray:
    """Per-window excess, raw and normalized ``[n, 4*len(windows)+3]``; raw counts for count-based rules."""
    windows = WINDOW_MS if windows is None else windows
    ccg = np.atleast_2d(ccg)
    null = np.atleast_2d(null)
    res = smooth(residual(ccg, null), sigma)
    raw = smooth(ccg - null, sigma)
    lags = lag_axis(res.shape[1], duration)
    noise = res[:, _flanks(res.shape[1])].std(axis=1) + 1e-9

    cols = []
    for lo, hi in windows:
        # both lag signs: direction is the asymmetry the burst rule turns on
        for sel in ((lags >= lo) & (lags <= hi), (lags <= -lo) & (lags >= -hi)):
            if not sel.any():
                cols += [np.zeros(len(res)), np.zeros(len(res))]
                continue
            cols += [res[:, sel].max(axis=1) / noise, raw[:, sel].max(axis=1)]
    cols += [ccg.max(axis=1), ccg.mean(axis=1), null.mean(axis=1)]
    return np.column_stack(cols)


WINDOW_FEATURE_NAMES = (
    [f'{s}{lo}-{hi}ms_{k}' for lo, hi in WINDOW_MS
     for s in ('pos', 'neg') for k in ('snr', 'raw')]
    + ['ccg_peak_count', 'ccg_mean_count', 'null_mean_count'])


# --- autocorrelogram: sorting quality is a per-neuron property no CCG contains ---


def acg_features(acg: np.ndarray, duration: float) -> np.ndarray:
    """Unit-quality descriptors from one autocorrelogram ``[n, 7]``."""
    acg = np.atleast_2d(acg).astype(float)
    n_bin = acg.shape[1]
    zero = n_bin // 2
    lags = lag_axis(n_bin, duration)
    flank = acg[:, _flanks(n_bin)].mean(axis=1) + 1e-9

    refr = (lags >= -1.5) & (lags <= 1.5)
    early = (np.abs(lags) > 1.5) & (np.abs(lags) <= 5.0)
    return np.column_stack([
        acg[:, zero] / flank,                      # 0 ms fill — contamination
        acg[:, refr].mean(axis=1) / flank,         # refractory depth
        acg[:, early].mean(axis=1) / flank,        # burst shoulder
        acg.max(axis=1) / flank,                   # peak relative to baseline
        acg.mean(axis=1),                          # raw firing level
        acg.sum(axis=1),                           # total spikes in window
        (acg[:, refr].mean(axis=1) < 0.5 * flank).astype(float),   # clean unit
    ])


ACG_FEATURE_NAMES = ['acg_zero_fill', 'acg_refractory', 'acg_burst',
                     'acg_peak', 'acg_mean', 'acg_total', 'acg_clean']


def deconvolved(ccg: np.ndarray, acg_ref: np.ndarray,
                acg_tgt: np.ndarray) -> np.ndarray:
    """CCG with both neurons' autocorrelation divided out; regularized against near-zero spectrum bins."""
    ccg = np.atleast_2d(ccg).astype(float)
    F = np.fft.rfft(ccg, axis=1)
    out = F
    for acg in (acg_ref, acg_tgt):
        A = np.fft.rfft(np.atleast_2d(acg).astype(float), axis=1)
        mag = np.abs(A)
        out = out * np.conj(A) / (mag ** 2 + 0.01 * (mag ** 2).max(axis=1,
                                                                  keepdims=True))
    return np.fft.irfft(out, n=ccg.shape[1], axis=1)


# --- scalar descriptors -----------------------------------------------------

def _flanks(n_bin: int) -> np.ndarray:
    """Mask of the outer thirds — the part of the window treated as baseline."""
    m = np.zeros(n_bin, dtype=bool)
    edge = max(1, n_bin // 3)
    m[:edge] = m[-edge:] = True
    return m


def shape_features(ccg: np.ndarray, null: np.ndarray, duration: float,
                   sigma: float = 1.0) -> np.ndarray:
    """Interpretable per-pair descriptors ``[n, 14]``.

    These name the things the user's own group notes describe: peak height and
    lag, peak width, symmetry about 0 ms, rifts (dips flanking a peak), and
    oscillation. They double as the diagnostic axes used to verify predictions.
    """
    ccg = np.atleast_2d(ccg)
    null = np.atleast_2d(null)
    res = smooth(residual(ccg, null), sigma)
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
