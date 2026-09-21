from __future__ import annotations
from dataclasses import dataclass
from enum import Enum as _Enum, auto as _auto
import numpy as np
from scipy.signal import windows
from scipy.stats import poisson
from scipy import ndimage
from statsmodels.stats.multitest import multipletests
from neuropy.analyses.jitter import compute_jbsi, JitterConfig


def multiple_correction(pvals: np.ndarray, alpha: float, method: str = 'bonferroni') -> tuple:
    """Multiple-test correction over lag bins; returns ``(sig, p_correct)``."""
    if method == 'bonferroni':
        corrected = np.minimum(pvals * pvals.shape[-1], 1.0)
        return corrected <= alpha, corrected
    significance = np.zeros_like(pvals, dtype=bool)
    p_correct = np.ones_like(pvals, dtype=float)
    for idx in np.ndindex(pvals.shape[:-1]):
        s, pc, _, _ = multipletests(pvals[idx], alpha=alpha, method=method)
        significance[idx] = s
        p_correct[idx] = pc
    return significance, p_correct


def hollow_conv(ccg, W=5, wintype="gauss", hollow_frac=None):
    """Hollow convolution baseline (Stark & Abeles 2009); returns ``pvals``, ``pred``, ``qvals``."""
    if len(ccg.shape) == 1:
        ccg = ccg[np.newaxis, ...]

    assert wintype in ["gauss", "rect", "triang"]
    assert W <= ccg.shape[-1]

    if wintype == "gauss":
        hollow_frac = hollow_frac or 0.6
        sigma = W / 2
        W = int(6 * sigma + (2 if W % 2 else 1))
        center = int(3 * sigma + (0.5 if W % 2 else 0))
        window = windows.gaussian(W, std=sigma) / (2 * np.pi * sigma)
    elif wintype == "rect":
        hollow_frac = hollow_frac or 0.42
        if W % 2 == 0:
            W += 1
        center = W // 2
        window = windows.boxcar(W)
    elif wintype == "triang":
        hollow_frac = hollow_frac or 0.63
        W = 2 * W + (-1 if W % 2 else 1)
        center = W // 2
        window = windows.triang(W)

    window[center] *= (1 - hollow_frac)
    window /= np.sum(window)
    ccg_pad = np.concatenate(
        [ccg[..., :W][..., ::-1], ccg, ccg[..., -W:][..., ::-1]], axis=-1)

    pred = ndimage.convolve1d(ccg_pad, window, axis=-1)
    pred = pred[..., W:-W]

    # mid-p Poisson test: P( val<=pred ) + half of P ( val==pred )
    pvals = 1 - poisson.cdf(ccg - 1, pred) - poisson.pmf(ccg, pred) * 0.5
    qvals = 1 - pvals
    return pvals, pred, qvals


class NormalizeBy(_Enum):
    REF_FRATE  = _auto()
    TARGET_FRATE = _auto()
    TIME_SPAN  = _auto()
    TIME_SECOND = _auto()
    TOTAL_AREA = _auto()
    BASELINE   = _auto()


class CCGNorm:
    """CCG normalization transforms."""

    @staticmethod
    def deconv_autocorr(ccg, acg1, nspks1, acg2=None, nspks2=None):
        """Deconvolve ACGs from a CCG trace. From Eran Stark's cchdeconv.m.

        Last axis = bins; leading axes (e.g. pairs) are batched — each row uses
        its own ACGs, FFT runs along axis=-1. `nspks` may be scalar or per-row."""
        ccg = np.asarray(ccg, dtype=float)
        acg1 = np.asarray(acg1, dtype=float)
        m = ccg.shape[-1]
        assert m % 2 == 1, f"deconvolution needs an odd bin count, got {m}"
        if acg2 is not None:
            acg2 = np.asarray(acg2, dtype=float)
        hw = (m - 1) // 2
        hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])

        def _norm(acg, nspks):
            denom = np.maximum(np.asarray(nspks, dtype=float), 1.0)
            if denom.ndim:
                denom = denom[..., None]
            a = (acg - acg.mean(axis=-1, keepdims=True)) / denom
            a[..., hw] = 1 - a[..., hidx].sum(axis=-1)
            # zero lag sits at hw, but the FFT reads it at index 0
            return np.fft.ifftshift(a, axes=-1)

        den = np.fft.fft(_norm(acg1, nspks1), axis=-1)
        if acg2 is not None:
            den = den * np.fft.fft(_norm(acg2, nspks2), axis=-1)
        den = np.where(np.abs(den) < 1e-10, 1e-10, den)
        dcccg = np.real(np.fft.ifft(np.fft.fft(ccg, axis=-1) / den, axis=-1))
        return np.where(dcccg < 0, 0.0, dcccg)

    @staticmethod
    def deconv_for_pair(ccg, neurons, ref, tgt, *, acg_ref=None, acg_tgt=None):
        """deconv_autocorr for the ACGs given; *ccg* unchanged when neither is."""
        acgs = []
        if acg_ref is not None:
            acgs += [acg_ref, neurons.n_spikes[ref]]
        if acg_tgt is not None:
            acgs += [acg_tgt, neurons.n_spikes[tgt]]
        return CCGNorm.deconv_autocorr(ccg, *acgs) if acgs else ccg

    @staticmethod
    def apply(ccg_raw, ccg_null_raw, ref, tgt,
              active_norms, neurons=None, custom_time_hours=None,):
        """Return (ccg, ccg_null) with active normalizations applied (copies).

        Last axis = bins; leading axes are batched. `ref`/`tgt` may be scalars
        (single pair) or arrays (one per leading pair) — per-pair factors
        broadcast over the bin axis."""
        if not active_norms:
            return ccg_raw, ccg_null_raw
        ccg = ccg_raw.copy().astype(float)
        ccg_null = ccg_null_raw.copy().astype(float) if ccg_null_raw is not None else None

        def _div(factor):
            nonlocal ccg, ccg_null
            factor = np.maximum(np.asarray(factor, dtype=float), 1e-12)
            if factor.ndim > 0:
                factor = factor[..., None]  # broadcast over bins
            ccg = ccg / factor
            if ccg_null is not None:
                ccg_null = ccg_null / factor

        if NormalizeBy.REF_FRATE in active_norms and neurons is not None:
            _div(neurons.firing_rate[ref])
        if NormalizeBy.TARGET_FRATE in active_norms and neurons is not None:
            _div(neurons.firing_rate[tgt])
        if NormalizeBy.TIME_SPAN in active_norms or NormalizeBy.TIME_SECOND in active_norms:
            if custom_time_hours is not None:
                et = float(custom_time_hours)
                if NormalizeBy.TIME_SPAN in active_norms:
                    _div(et)
                if NormalizeBy.TIME_SECOND in active_norms:
                    _div(et * 3600.0)
        if NormalizeBy.TOTAL_AREA in active_norms:
            _div(np.sum(np.abs(ccg), axis=-1))
        if NormalizeBy.BASELINE in active_norms and ccg_null is not None:
            ccg -= ccg_null
            ccg_null = np.zeros_like(ccg, dtype=float)
        return ccg, ccg_null



class ConnectionStrength:

    @staticmethod
    def _bins(n_bins, conf, lo=None, hi=None):
        """(lo, hi) test-window bins; *lo*/*hi* override conf. The window must fit the array."""
        lo = int(conf.min_lag_bin) if lo is None else int(lo)
        hi = int(conf.max_lag_bin) if hi is None else int(hi)
        if not 0 <= lo < hi <= n_bins:
            raise ValueError(f"test window [{lo},{hi}) outside {n_bins}-bin CCG "
                             f"— conf resolution ({conf.resolution}) does not match the array")
        return lo, hi

    @staticmethod
    def baseline(ccg, ccg_null, conf, method, *, bin_size_eff=None):
        """Per-bin baseline, same shape as `ccg` (last axis = bins); leading axes batched.

        Dispatches on *method*; each baseline_* fits or reuses its own null."""
        if method == 'conv':
            return ConnectionStrength.baseline_conv(ccg, ccg_null, conf,
                                                    bin_size_eff=bin_size_eff)
        if method == 'jitter':
            return ConnectionStrength.baseline_jitter(ccg, ccg_null, conf)
        if method == 'tailed':
            src = getattr(conf, 'tail_source', 'bins')
            source = (None if src == 'bins'
                      else ConnectionStrength.baseline(ccg, ccg_null, conf, src,
                                                       bin_size_eff=bin_size_eff))
            return ConnectionStrength.baseline_tail(ccg, conf, bin_size_eff=bin_size_eff,
                                                    source=source)
        return ConnectionStrength.baseline_global(ccg, ccg_null, conf)

    @staticmethod
    def baseline_conv(ccg, ccg_null=None, conf=None, *, bin_size_eff=None):
        """Hollow-convolution prediction; fits the CCG when no null is supplied."""
        if ccg_null is not None:
            return np.asarray(ccg_null, dtype=float).copy()
        ccg = np.asarray(ccg, dtype=float)
        n_bins = ccg.shape[-1]
        bs = bin_size_eff or (conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size)
        W = max(1, int(min(conf.conv_window / bs, (n_bins - 1) / 3)))
        _, pred, _ = hollow_conv(ccg, W=W, wintype="gauss")
        return pred.reshape(ccg.shape)

    @staticmethod
    def baseline_jitter(ccg, j_avg=None, conf=None):
        """Jitter-surrogate baseline — the jitter mean, which the caller supplies.

        TODO: compute it here (neuropy.analyses.jitter) when absent, so this is
        generative rather than cache-fed; until then the jitter control is disabled."""
        if j_avg is None:
            print("[ConnectionStrength] no jitter result — run jitter first", flush=True)
            return None
        return np.asarray(j_avg, dtype=float).copy()

    @staticmethod
    def tail_mask(n_bins, conf, bin_size_eff=None) -> np.ndarray:
        """Bool mask over bins, the union of conf's tail intervals; None edges reach the window edge."""
        bs = bin_size_eff or (conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size)
        half = (n_bins // 2) * bs
        mask = np.zeros(n_bins, dtype=bool)
        for start, end in conf.tail_intervals:
            a = -half if start is None else max(-half, float(start))
            b = half if end is None else min(half, float(end))
            if b < a:      # the interval lies wholly outside the window
                continue
            lo, hi = lag_window_bins(a, b, bs, n_bins // 2)
            mask[max(0, lo):min(n_bins, hi)] = True
        return mask

    @staticmethod
    def baseline_tail(ccg, conf, *, bin_size_eff=None, source=None):
        """Flat baseline at the tail-interval mean, of the CCG itself or of *source*."""
        ccg = np.asarray(ccg, dtype=float)
        n_bins = ccg.shape[-1]
        mask = ConnectionStrength.tail_mask(n_bins, conf, bin_size_eff)
        if not mask.any():
            mask[:max(1, n_bins // 10)] = mask[-max(1, n_bins // 10):] = True
        vals = ccg if source is None else np.broadcast_to(
            np.asarray(source, dtype=float), ccg.shape)
        bv = np.mean(vals[..., mask], axis=-1, keepdims=True)
        return np.broadcast_to(bv, ccg.shape).copy()

    @staticmethod
    def baseline_global(ccg, ccg_null=None, conf=None):
        """Flat baseline at the peak of the null, or of the CCG outside the test window."""
        ccg = np.asarray(ccg, dtype=float)
        if ccg_null is not None:
            bv = np.max(np.asarray(ccg_null, dtype=float), axis=-1, keepdims=True)
        else:
            n_bins = ccg.shape[-1]
            lo, hi = ConnectionStrength._bins(n_bins, conf)
            mask = np.ones(n_bins, dtype=bool); mask[lo:hi] = False
            bv = np.max(ccg[..., mask] if mask.any() else ccg, axis=-1, keepdims=True)
        return np.broadcast_to(bv, ccg.shape).copy()

    @staticmethod
    def conn_strength_STG_batch(ccg, baseline, lo, hi, excitability='E'):
        """Area between CCG and baseline over bins [lo, hi); leading axes batched."""
        d = (np.asarray(ccg, dtype=float) - np.asarray(baseline, dtype=float))[..., lo:hi]
        return np.sum(np.clip(d, None, 0) if excitability == 'I' else np.clip(d, 0, None), axis=-1)

    @staticmethod
    def conn_strength_JBSI_batch(ccg, j_avg, fr_ref, fr_tgt, conf, lo, hi, jscale=None):
        """Summed JBSI (Agmon 2012) over bins [lo, hi); leading axes batched."""
        if jscale is None:
            jscale = float(JitterConfig(ccg=conf, njitter=1).jscale)
        jbsi = compute_jbsi(real_ccg=ccg, j_ccg_avg=j_avg, fr_ref=fr_ref, fr_tgt=fr_tgt,
                            bin_size=float(conf.bin_size), jscale=jscale)
        return np.sum(jbsi[..., lo:hi], axis=-1)

    @staticmethod
    def conn_strength_STG(ccg, baseline, lo, hi, excitability='E') -> float:
        """One pair's STG — the batch routine on a single trace."""
        return float(ConnectionStrength.conn_strength_STG_batch(
            ccg, baseline, lo, hi, excitability))

    @staticmethod
    def conn_strength_JBSI(ccg, j_avg, fr_ref, fr_tgt, conf, lo, hi, jscale=None) -> float:
        """One pair's JBSI — the batch routine on a single trace."""
        return float(ConnectionStrength.conn_strength_JBSI_batch(
            ccg, j_avg, fr_ref, fr_tgt, conf, lo, hi, jscale))

    @staticmethod
    def conn_strength(ccg_raw, null_raw, ref, tgt, conf, *,
                      metric='CS', method='conv', active_norms=(), neurons=None,
                      baseline=None, bin_size_eff=None, lo=None, hi=None,
                      custom_time_hours=None, fr_ref=None, fr_tgt=None,
                      j_avg=None, nonneg=False, excitability='E'):
        """Connection strength for a CCG of any shape (last axis = bins).

        *baseline* skips the internal fit; *bin_size_eff* sets the test window from
        this array's own bins rather than conf's stored resolution; *lo*/*hi* override
        conf's test window outright."""
        ccg_raw = np.asarray(ccg_raw)
        if ccg_raw.shape[-1] == 0:
            return None
        n_bins = ccg_raw.shape[-1]
        ccg, ccg_null = CCGNorm.apply(
            ccg_raw, null_raw, ref, tgt, active_norms,
            neurons, custom_time_hours=custom_time_hours)
        bl = (ConnectionStrength.baseline(ccg, ccg_null, conf, method,
                                          bin_size_eff=bin_size_eff)
              if baseline is None else np.asarray(baseline, dtype=float))
        if bl is None:      # jitter asked for but not computed
            return None

        if bin_size_eff is None:
            lo, hi = ConnectionStrength._bins(n_bins, conf, lo, hi)
        elif lo is None or hi is None:
            lo, hi = lag_window_bins(conf.min_lag, conf.max_lag, bin_size_eff,
                                     center_bin=n_bins // 2)
            if not 0 <= lo < hi <= n_bins:
                return None

        if metric == 'JBSI':
            if j_avg is not None:   # same norm as the CCG it is subtracted from
                j_avg, _ = CCGNorm.apply(j_avg, None, ref, tgt, active_norms,
                                         neurons, custom_time_hours=custom_time_hours)
            cs_val = ConnectionStrength.conn_strength_JBSI_batch(
                ccg, j_avg if j_avg is not None else bl, fr_ref, fr_tgt, conf, lo, hi)
        else:
            cs_val = ConnectionStrength.conn_strength_STG_batch(
                ccg, bl, lo, hi, excitability)
        if nonneg:
            cs_val = np.maximum(cs_val, 0.0)
        return float(cs_val) if np.ndim(cs_val) == 0 else cs_val


def _fill_waveform(wf_neuron, shank_id: int, ch_per_shank: int, discarded,
                   peak_channel: int = None, channels=None, start=None):
    """Expand a (possibly trimmed) per-neuron waveform to a full (ch_per_shank, T) array."""
    if wf_neuron.ndim == 1:
        if peak_channel is None:
            return np.tile(wf_neuron, (ch_per_shank, 1))
        clean = np.full((ch_per_shank, wf_neuron.shape[-1]), np.nan)
        clean[int(peak_channel) % ch_per_shank] = wf_neuron
        return clean
    sid  = int(shank_id)
    disc = np.asarray(discarded, dtype=int) if discarded is not None else np.empty(0, dtype=int)
    channel_ids = (np.asarray(channels, dtype=int) if channels is not None
                   else ch_per_shank * sid + np.arange(ch_per_shank))
    mask   = ~np.isin(channel_ids, disc)
    if start is None:
        start = int(ch_per_shank * sid - np.sum(disc < ch_per_shank * sid))
    length = int(np.sum(mask))
    clean  = np.full((len(channel_ids), wf_neuron.shape[-1]), np.nan)
    rows = wf_neuron[start:start + length]
    clean[np.flatnonzero(mask)[:len(rows)]] = rows
    return clean


def load_peak_waveform(ref: int, waveforms, peak_channels, shank_ids,
                       ch_per_shank: int, discarded):
    """Extract (t_ms, amp) for neuron *ref*'s peak-channel waveform. Returns (None, None) on failure."""
    if waveforms is None or peak_channels is None or shank_ids is None:
        return None, None
    try:
        peak_ch = int(peak_channels[ref])
        rs      = int(shank_ids[ref])
    except (IndexError, TypeError, ValueError):
        return None, None
    discarded_arr = None if discarded is None else np.asarray(discarded, dtype=int)
    if discarded_arr is not None and discarded_arr.size and np.isin(peak_ch, discarded_arr):
        return None, None
    local_idx = peak_ch - ch_per_shank * rs
    if not (0 <= local_idx < ch_per_shank):
        return None, None
    ref_full = _fill_waveform(waveforms[ref], rs, ch_per_shank, discarded_arr)
    tr = ref_full[local_idx]
    if not np.any(np.isfinite(tr)):
        return None, None
    n = int(tr.shape[0])
    return np.arange(n, dtype=float) - n // 2, np.asarray(tr, dtype=float)


def peak_waveform_on_lag_axis(neurons, ref: int, ch_per_shank: int,
                              discarded=None, *, sampling_rate: float = None):
    """Peak-channel waveform of *ref* as (lag_ms, amp) on its own sample rate."""
    t_samples, amp = load_peak_waveform(
        ref, neurons.waveforms, neurons.peak_channels, neurons.shank_ids,
        ch_per_shank, discarded)
    if t_samples is None:
        return None, None
    rate = float(sampling_rate or neurons.sampling_rate or 1.0)
    return (t_samples * (1000.0 / rate) if rate > 1 else t_samples), amp


def lag_window_bins(start: float, end: float, bin_size: float,
                    center_bin: int) -> tuple[int, int]:
    """Half-open bins [lo, hi) whose spans intersect lags *start*..*end* (seconds).

    Bins are centred on k*bin_size, so bin k spans (k-0.5)..(k+0.5)*bin_size."""
    if end < start:
        raise ValueError(f"window end {end} precedes start {start}")
    lo = center_bin + int(np.floor(start / bin_size + 0.5 + 1e-9))
    hi = center_bin + int(np.ceil(end / bin_size - 0.5 - 1e-9)) + 1
    return lo, max(hi, lo + 1)


@dataclass(frozen=True, slots=True)
class ConnStrengthConfig:
    baseline_method: str = "conv"
    cs_metric: str = "CS"
    min_lag_bin: int | None = None
    max_lag_bin: int | None = None
