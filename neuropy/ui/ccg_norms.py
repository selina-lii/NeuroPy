from enum import Enum as _Enum, auto as _auto
from dataclasses import dataclass
from typing import Optional
import numpy as np


class NormalizeBy(_Enum):
    REF_FRATE = _auto()
    TARGET_FRATE = _auto()
    TIME_SPAN = _auto()    # divide by effective recording time in hours
    TIME_SECOND = _auto()  # divide by effective recording time in seconds
    TOTAL_AREA = _auto()   # divide by sum of all bin counts (area under CCG)
    BASELINE = _auto()     # subtract conv baseline (ccg_null) from ccg


class NormBackend:
    """CCG normalization and connection-strength computation for display."""

    @dataclass
    class PanelData:
        """Normalized 1-D CCG data for plot_ccg_panel."""
        ccg: np.ndarray
        ccg_null: Optional[np.ndarray]
        baseline_1d: Optional[np.ndarray]
        cs_val: Optional[float]
        eff_min_lag: Optional[float]
        eff_max_lag: Optional[float]

    @staticmethod
    def deconv_autocorr(ccg, acg1, nspks1, acg2, nspks2):
        """Deconvolve ACGs from a CCG trace. From Eran Stark's cchdeconv.m."""
        m = len(ccg)
        if m % 2 == 0:
            m -= 1
            ccg = ccg[:m]; acg1 = acg1[:m]; acg2 = acg2[:m]
        hw = (m - 1) // 2
        hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])
        a1 = acg1.copy()
        a1 = (a1 - np.mean(a1)) / max(nspks1, 1)
        a1[hw] = 1 - np.sum(a1[hidx])
        a2 = acg2.copy()
        a2 = (a2 - np.mean(a2)) / max(nspks2, 1)
        a2[hw] = 1 - np.sum(a2[hidx])
        den = np.fft.fft(a1) * np.fft.fft(a2)
        den = np.where(np.abs(den) < 1e-10, 1e-10, den)
        dcccg = np.real(np.fft.ifft(np.fft.fft(ccg) / den))
        dcccg = np.concatenate([dcccg[1:], [dcccg[0]]])
        dcccg[dcccg < 0] = 0
        return dcccg

    @staticmethod
    def conn_strength(ccg, ccg_null, conf, method,
                      acg_ref=None, acg_tgt=None,
                      nspks_ref=1.0, nspks_tgt=1.0,
                      min_lag_override=None, max_lag_override=None):
        """Return (cs, baseline_1d) for a normalized 1-D CCG."""
        n_bins = len(ccg)
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
        center = n_bins // 2
        eff_min = min_lag_override if min_lag_override is not None else conf.min_lag
        eff_max = max_lag_override if max_lag_override is not None else conf.max_lag
        lo = max(0, center + int(eff_min / bin_size_eff))
        hi = min(n_bins, center + int(eff_max / bin_size_eff))

        if method == 'conv':
            if ccg_null is not None:
                return float(np.sum((ccg - ccg_null)[lo:hi])), ccg_null.copy()
            return float(np.sum(ccg[lo:hi])), np.zeros_like(ccg)

        if method == 'tailed':
            try:
                hw = max(1, int(11e-3 / bin_size_eff))
                l_idx, r_idx = center - hw, center + hw + 1
                tail = (np.concatenate([ccg[:l_idx], ccg[r_idx:]]) if l_idx > 0 and r_idx < n_bins
                        else np.concatenate([ccg[:max(1, n_bins // 10)], ccg[-max(1, n_bins // 10):]]))
                bv = float(np.mean(tail))
                return float(np.sum((ccg - bv)[lo:hi])), np.full(n_bins, bv)
            except Exception:
                return None, None

        if method == 'global':
            try:
                if ccg_null is not None:
                    bv = float(np.max(ccg_null))
                else:
                    mask = np.ones(n_bins, dtype=bool); mask[lo:hi] = False
                    bv = float(np.max(ccg[mask] if np.any(mask) else ccg))
                return float(np.sum((ccg - bv)[lo:hi])), np.full(n_bins, bv)
            except Exception:
                return None, None

        return None, None

    @staticmethod
    def apply(ccg_raw, ccg_null_raw, ref, tgt, seg,
              active_norms, neurons=None, nd=None, nd_key=None,
              n_segments=0, is_custom_seg=False, custom_time_hours=None):
        """Return (ccg, ccg_null) with active normalizations applied (copies)."""
        if not active_norms:
            return ccg_raw, ccg_null_raw
        ccg = ccg_raw.copy().astype(float)
        ccg_null = ccg_null_raw.copy().astype(float) if ccg_null_raw is not None else None

        def _div(factor):
            nonlocal ccg, ccg_null
            ccg /= max(factor, 1e-12)
            if ccg_null is not None:
                ccg_null /= max(factor, 1e-12)

        if NormalizeBy.REF_FRATE in active_norms and neurons is not None:
            _div(float(neurons.firing_rate[ref]))
        if NormalizeBy.TARGET_FRATE in active_norms and neurons is not None:
            _div(float(neurons.firing_rate[tgt]))
        if NormalizeBy.TIME_SPAN in active_norms or NormalizeBy.TIME_SECOND in active_norms:
            et = None
            if is_custom_seg and custom_time_hours is not None:
                et = float(custom_time_hours)
            elif nd is not None and not is_custom_seg:
                if seg == n_segments and n_segments > 0:
                    et = sum(float(nd.edge_times[nd_key].iloc[s]['effective_time_hours'])
                             for s in range(n_segments))
                else:
                    et = float(nd.edge_times[nd_key].iloc[seg]['effective_time_hours'])
            if et is not None:
                if NormalizeBy.TIME_SPAN in active_norms:
                    _div(et)
                if NormalizeBy.TIME_SECOND in active_norms:
                    _div(et * 3600.0)
        if NormalizeBy.TOTAL_AREA in active_norms:
            total = float(np.sum(np.abs(ccg)))
            if total > 1e-12:
                _div(total)
        if NormalizeBy.BASELINE in active_norms and ccg_null is not None:
            ccg -= ccg_null
            ccg_null = None
        return ccg, ccg_null

    @staticmethod
    def compute(ccg_raw, ccg_null_raw, conf, method, active_norms,
                ref, tgt, segment, n_segments, is_custom, custom_time_hours,
                eff_min_lag, eff_max_lag,
                neurons=None, nd=None, nd_key=None,
                acg_ref=None, acg_tgt=None,
                nspks_ref=1.0, nspks_tgt=1.0):
        """Normalize CCG + compute CS in one pass.

        Order matters: norms except BASELINE → CS + baseline_1d → BASELINE norm.
        """
        has_bl = NormalizeBy.BASELINE in active_norms
        ccg, ccg_null = NormBackend.apply(
            ccg_raw, ccg_null_raw, ref, tgt, segment,
            active_norms - {NormalizeBy.BASELINE},
            neurons, nd, nd_key, n_segments, is_custom, custom_time_hours)
        cs_val, baseline_1d = NormBackend.conn_strength(
            ccg, ccg_null, conf, method,
            acg_ref=acg_ref, acg_tgt=acg_tgt,
            nspks_ref=nspks_ref, nspks_tgt=nspks_tgt,
            min_lag_override=eff_min_lag, max_lag_override=eff_max_lag)
        if has_bl:
            bl = baseline_1d if baseline_1d is not None else ccg_null
            if bl is not None:
                ccg = ccg - bl
                ccg_null = None
        display_null = None if has_bl else (ccg_null if method == 'conv' else baseline_1d)
        return NormBackend.PanelData(
            ccg=ccg, ccg_null=display_null, baseline_1d=baseline_1d,
            cs_val=cs_val, eff_min_lag=eff_min_lag, eff_max_lag=eff_max_lag)

    @staticmethod
    def deconvolve(ccg_raw, null_raw, acg_ref, nspks_ref, acg_tgt, nspks_tgt,
                   dref: bool, dtgt: bool):
        """Apply ACG deconvolution to a CCG + null pair. Returns (ccg_out, null_out)."""
        def _deconv_1d(x):
            if x is None:
                return None
            return NormBackend.deconv_autocorr(
                x.copy().astype(float),
                acg_ref if dref else np.zeros_like(acg_ref, dtype=float),
                nspks_ref if dref else 1.0,
                acg_tgt if dtgt else np.zeros_like(acg_tgt, dtype=float),
                nspks_tgt if dtgt else 1.0,
            )
        try:
            return _deconv_1d(ccg_raw), _deconv_1d(null_raw)
        except Exception:
            return ccg_raw, null_raw

