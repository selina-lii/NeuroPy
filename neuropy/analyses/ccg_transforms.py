from __future__ import annotations
from dataclasses import dataclass
from enum import Enum as _Enum, auto as _auto
import numpy as np
from neuropy.analyses.jitter import compute_jbsi, JitterConfig


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
        if m % 2 == 0:
            m -= 1
            ccg = ccg[..., :m]; acg1 = acg1[..., :m]
            if acg2 is not None:
                acg2 = np.asarray(acg2, dtype=float)[..., :m]
        hw = (m - 1) // 2
        hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])

        def _norm(acg, nspks):
            denom = np.maximum(np.asarray(nspks, dtype=float), 1.0)
            if denom.ndim:
                denom = denom[..., None]
            a = (acg - acg.mean(axis=-1, keepdims=True)) / denom
            a[..., hw] = 1 - a[..., hidx].sum(axis=-1)
            return a

        den = np.fft.fft(_norm(acg1, nspks1), axis=-1)
        if acg2 is not None:
            den = den * np.fft.fft(_norm(acg2, nspks2), axis=-1)
        den = np.where(np.abs(den) < 1e-10, 1e-10, den)
        dcccg = np.real(np.fft.ifft(np.fft.fft(ccg, axis=-1) / den, axis=-1))
        dcccg = np.concatenate([dcccg[..., 1:], dcccg[..., :1]], axis=-1)
        return np.where(dcccg < 0, 0.0, dcccg)

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

    @staticmethod
    def deconvolve(ccg_raw, null_raw, acg_ref, nspks_ref, acg_tgt, nspks_tgt,
                   dref: bool, dtgt: bool):
        """Apply ACG deconvolution to a CCG + null pair. Returns (ccg_out, null_out)."""
        if not dref and not dtgt:
            return ccg_raw, null_raw
        a1, n1 = (acg_ref, nspks_ref) if dref else (acg_tgt, nspks_tgt)
        a2, n2 = (acg_tgt, nspks_tgt) if (dref and dtgt) else (None, None)

        def _deconv_1d(x):
            if x is None:
                return None
            return CCGNorm.deconv_autocorr(x.copy().astype(float), a1, n1, a2, n2)

        return _deconv_1d(ccg_raw), _deconv_1d(null_raw)


class ConnectionStrength:

    @staticmethod
    def _bins(n_bins, conf):
        """(lo, hi) test-window bins from conf; the window must fit the array."""
        lo, hi = int(conf.min_lag_bin), int(conf.max_lag_bin)
        if not 0 <= lo < hi <= n_bins:
            raise ValueError(f"test window [{lo},{hi}) outside {n_bins}-bin CCG "
                             f"— conf resolution ({conf.resolution}) does not match the array")
        return lo, hi

    @staticmethod
    def baseline(ccg, ccg_null, conf, method):
        """Per-bin baseline, same shape as `ccg` (last axis = bins). conv/jitter
        reuse the precomputed null. Leading axes (e.g. pairs) are batched."""
        n_bins = ccg.shape[-1]
        if method in ('conv', 'jitter'):
            return ccg_null.copy() if ccg_null is not None else np.zeros_like(ccg, dtype=float)
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
        center = n_bins // 2
        if method == 'tailed':
            hw = max(1, int(11e-3 / bin_size_eff))
            l_idx, r_idx = center - hw, center + hw + 1
            if l_idx > 0 and r_idx < n_bins:
                tail = np.concatenate([ccg[..., :l_idx], ccg[..., r_idx:]], axis=-1)
            else:
                edge = max(1, n_bins // 10)
                tail = np.concatenate([ccg[..., :edge], ccg[..., -edge:]], axis=-1)
            bv = np.mean(tail, axis=-1, keepdims=True)
        elif ccg_null is not None:  # global
            bv = np.max(ccg_null, axis=-1, keepdims=True)
        else:  # global, no null
            lo, hi = ConnectionStrength._bins(n_bins, conf)
            mask = np.ones(n_bins, dtype=bool); mask[lo:hi] = False
            bv = np.max(ccg[..., mask] if mask.any() else ccg, axis=-1, keepdims=True)
        return np.broadcast_to(bv, ccg.shape).copy()

    @staticmethod
    def conn_strength(ccg, baseline, conf, excitability='E'):
        """Area between CCG and baseline in the test window — the shaded CS overlay region."""
        lo, hi = ConnectionStrength._bins(ccg.shape[-1], conf)
        d = (ccg - baseline)[..., lo:hi]
        return np.sum(np.clip(d, None, 0) if excitability == 'I' else np.clip(d, 0, None), axis=-1)

    @staticmethod
    def jbsi_strength(ccg, fr_ref, fr_tgt, conf, *, j_avg, jscale=None):
        """Summed JBSI over the test window (last bin axis; leading axes batched).

        j_avg is the jitter mean (Agmon 2012); callers pass the selected baseline
        instead when no jitter result exists."""
        if jscale is None:
            jscale = float(JitterConfig(ccg=conf, njitter=1).jscale)
        jbsi = compute_jbsi(real_ccg=ccg, j_ccg_avg=j_avg, fr_ref=fr_ref, fr_tgt=fr_tgt,
                            bin_size=float(conf.bin_size), jscale=jscale)
        lo, hi = ConnectionStrength._bins(jbsi.shape[-1], conf)
        return np.sum(jbsi[..., lo:hi], axis=-1)

    @staticmethod
    def compute(ccg_raw, null_raw, ref, tgt, conf, *,
                metric, method, active_norms, neurons,
                custom_time_hours=None, fr_ref=None, fr_tgt=None,
                j_avg=None, nonneg=False, excitability='E'):
        """Full connection-strength pipeline for one pair from raw CCG.
        """
        if len(ccg_raw) == 0:
            return None
        method_eff = 'conv' if method == 'jitter' else method
        ccg, ccg_null = CCGNorm.apply(
            ccg_raw, null_raw, ref, tgt, active_norms,
            neurons, custom_time_hours=custom_time_hours)
        bl = ConnectionStrength.baseline(ccg, ccg_null, conf, method_eff)
        if metric == 'JBSI':
            if j_avg is not None:   # same norm as the CCG it is subtracted from
                j_avg, _ = CCGNorm.apply(j_avg, None, ref, tgt, active_norms,
                                         neurons, custom_time_hours=custom_time_hours)
            cs_val = ConnectionStrength.jbsi_strength(
                ccg, fr_ref, fr_tgt, conf,
                j_avg=j_avg if j_avg is not None else bl)
        else:
            cs_val = ConnectionStrength.conn_strength(ccg, bl, conf, excitability)
        if cs_val is not None and nonneg:
            cs_val = max(0.0, float(cs_val))
        return cs_val


@dataclass(frozen=True, slots=True)
class ConnStrengthConfig:
    baseline_method: str = "conv"
    cs_metric: str = "CS"
    min_lag_bin: int | None = None
    max_lag_bin: int | None = None
