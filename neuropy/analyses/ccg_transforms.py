from __future__ import annotations
from enum import Enum as _Enum, auto as _auto
from typing import TYPE_CHECKING
import numpy as np
from neuropy.analyses.jitter import compute_jbsi, JitterConfig
from neuropy.utils.data_storage_util import LRUCache
from dataclasses import dataclass

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState


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
        """
        def _norm(acg, nspks, m, hw, hidx):
            a = acg.copy()
            a = (a - np.mean(a)) / max(nspks, 1)
            a[hw] = 1 - np.sum(a[hidx])
            return a

        m = len(ccg)
        if m % 2 == 0:
            m -= 1
            ccg = ccg[:m]; acg1 = acg1[:m]
            if acg2 is not None:
                acg2 = acg2[:m]
        hw = (m - 1) // 2
        hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])
        a1 = _norm(acg1, nspks1, m, hw, hidx)
        den = np.fft.fft(a1)
        if acg2 is not None:
            den = den * np.fft.fft(_norm(acg2, nspks2, m, hw, hidx))
        den = np.where(np.abs(den) < 1e-10, 1e-10, den)
        dcccg = np.real(np.fft.ifft(np.fft.fft(ccg) / den))
        dcccg = np.concatenate([dcccg[1:], [dcccg[0]]])
        dcccg[dcccg < 0] = 0
        return dcccg

    @staticmethod
    def apply(ccg_raw, ccg_null_raw, ref, tgt, seg,
              active_norms, neurons=None, custom_time_hours=None,
              nd=None, nd_key=None, n_segments=0, is_custom_seg=False,
              fr_ref=None, fr_tgt=None):
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

        if NormalizeBy.REF_FRATE in active_norms:
            rate = fr_ref if fr_ref is not None else (
                float(neurons.firing_rate[ref]) if neurons is not None else None)
            if rate is not None:
                _div(rate)
        if NormalizeBy.TARGET_FRATE in active_norms:
            rate = fr_tgt if fr_tgt is not None else (
                float(neurons.firing_rate[tgt]) if neurons is not None else None)
            if rate is not None:
                _div(rate)
        if NormalizeBy.TIME_SPAN in active_norms or NormalizeBy.TIME_SECOND in active_norms:
            if custom_time_hours is not None:
                et = float(custom_time_hours)
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
                ref, tgt, segment, custom_time_hours,
                eff_min_lag, eff_max_lag, neurons=None):
        """Normalize CCG + compute CS in one pass.

        Returns (ccg, display_null, baseline_1d, cs_val, eff_min_lag, eff_max_lag).
        Order: norms except BASELINE → CS + baseline_1d → BASELINE norm.
        """
        has_bl = NormalizeBy.BASELINE in active_norms
        ccg, ccg_null = CCGNorm.apply(
            ccg_raw, ccg_null_raw, ref, tgt, segment,
            active_norms - {NormalizeBy.BASELINE},
            neurons, custom_time_hours=custom_time_hours)
        cs_val, baseline_1d = ConnectionStrength.conn_strength(
            ccg, ccg_null, conf, method,
            min_lag_override=eff_min_lag, max_lag_override=eff_max_lag)
        if has_bl:
            bl = baseline_1d if baseline_1d is not None else ccg_null
            if bl is not None:
                ccg = ccg - bl
                ccg_null = None
        display_null = None if has_bl else (ccg_null if method == 'conv' else baseline_1d)
        return ccg, display_null, baseline_1d, cs_val, eff_min_lag, eff_max_lag

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


@dataclass
class _CSContext:
    ref: int
    tgt: int
    seg: object
    highres: bool
    nd_key: object
    eff_min_lag: float | None
    eff_max_lag: float | None


class ConnectionStrength:
    """Connection strength computation. Injected deps: nav, pair_mgr, custom_mgr,
    jitter_ctrl, jitter_cache."""

    _CS_METHOD_DESCRIPTIONS = {
        'conv':   "Conv baseline: smoothed null (EranConv)",
        'tailed': "Tailed: ACG deconvolution, tail-bin baseline",
        'global': "Global: max bin outside test window as baseline",
        'jitter': "Jitter: surrogate spike baseline",
    }

    def __init__(self, nav: 'AppState', pair_mgr, custom_mgr, jitter_ctrl, jitter_cache: dict):
        self._nav = nav
        self._pair_mgr = pair_mgr
        self._custom_mgr = custom_mgr
        self._jitter_ctrl = jitter_ctrl
        self._jitter_cache = jitter_cache
        self._cache: LRUCache = LRUCache(256)

    def _ctx(self, ref: int, tgt: int, seg, highres: bool) -> _CSContext:
        cd = self._nav.ccg_data
        nd_key = self._nav.key.nd() if self._nav.key else None
        conf = cd.conf if cd is not None else None
        eff_min = getattr(conf, 'min_lag', None) if conf else None
        eff_max = getattr(conf, 'max_lag', None) if conf else None
        return _CSContext(ref=ref, tgt=tgt, seg=seg, highres=highres,
                          nd_key=nd_key, eff_min_lag=eff_min, eff_max_lag=eff_max)

    @staticmethod
    def conn_strength(ccg, ccg_null, conf, method,
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

    def _fmt_cs_value(self, v, *, nonneg: bool = False) -> str:
        if v is None:
            return "n/a"
        x = float(v)
        if nonneg:
            x = max(x, 0.0)
        return f"{x:.2f}" if abs(x) >= 1000 else f"{x:.3g}"

    def _cache_key(self, ctx: _CSContext, method: str) -> tuple:
        return (ctx.ref, ctx.tgt, ctx.seg, method, ctx.highres, ctx.eff_min_lag, ctx.eff_max_lag)

    def _get_cs_val(self, ctx: _CSContext, method: str):
        k = self._cache_key(ctx, method)
        if k not in self._cache:
            self._compute_pair_conn_strength(ctx)
        entry = self._cache.get(k)
        v, _ = entry if entry is not None else (None, None)
        return v

    def _get_stg_cs(self, ctx: _CSContext, method: str, nonneg: bool = False,
                    acg_deconv_active: bool = False,
                    display_pair_temp: dict | None = None) -> str:
        if acg_deconv_active and display_pair_temp is not None:
            resk = 'hi' if ctx.highres else 'lo'
            tmp = display_pair_temp.get((ctx.ref, ctx.tgt, int(ctx.seg), resk))
            if tmp is not None and tmp.get("method") == method:
                return self._fmt_cs_value(tmp.get("cs_val"), nonneg=nonneg)
        return self._fmt_cs_value(self._get_cs_val(ctx, method), nonneg=nonneg)

    def _firing_rates(self, ctx: _CSContext):
        nav = self._nav
        cd = nav.cd
        if (ctx.nd_key is not None and cd.nd is not None
                and ctx.seg != nav.n_segments
                and not self._custom_mgr._is_custom_segment(ctx.seg)):
            seg_fr = cd.segment_firing_rates(ctx.nd_key)
            if (seg_fr is not None and ctx.seg < seg_fr.shape[0]
                    and ctx.ref < seg_fr.shape[1] and ctx.tgt < seg_fr.shape[1]):
                return float(seg_fr[ctx.seg, ctx.ref]), float(seg_fr[ctx.seg, ctx.tgt])
        fr = getattr(nav.neurons, 'firing_rate', None)
        if fr is not None and ctx.ref < len(fr) and ctx.tgt < len(fr):
            return float(fr[ctx.ref]), float(fr[ctx.tgt])
        return None, None

    def _jitter_avg(self, ctx: _CSContext):
        _jseg = self._jitter_ctrl.seg(ctx.seg)
        entry = self._jitter_cache.get((ctx.ref, ctx.tgt, 'lo', _jseg))
        if entry is None and _jseg is not None:
            entry = self._jitter_cache.get((ctx.ref, ctx.tgt, 'lo', None))
        return entry[0] if entry is not None else None

    def _jscale(self, ctx: _CSContext) -> float:
        nav = self._nav
        lo_cd = (nav.cd.ccg.get(ctx.nd_key) if hasattr(nav.cd, 'ccg') else None) or nav.ccg_data
        return float(JitterConfig(ccg=lo_cd.conf, njitter=1).jscale)

    def _get_jbsi_cs(self, ctx: _CSContext, nonneg: bool = False) -> str:
        nav = self._nav
        cd = nav.cd
        cd_res = (cd.ccg_for(ctx.nd_key, 'highres') if ctx.highres else None) or \
                 (cd.ccg_for(ctx.nd_key, 'lowres') if hasattr(cd, 'ccg_for') else None) or \
                 nav.ccg_data
        if cd_res is None:
            return "n/a"
        d = self._pair_mgr._resolve_segment_data(ctx.ref, ctx.tgt, ctx.seg,
                                                 highres=ctx.highres, include_pval=False, _cd=cd_res)
        real_ccg = d.get('ccg_raw')
        if real_ccg is None:
            return "n/a"
        fr_ref, fr_tgt = self._firing_rates(ctx)
        if fr_ref is None:
            return "n/a"
        j_avg = self._jitter_avg(ctx) if not ctx.highres else None
        if j_avg is None:
            j_avg = np.zeros_like(real_ccg, dtype=float)
        jbsi = compute_jbsi(real_ccg=real_ccg, j_ccg_avg=j_avg,
                            fr_ref=fr_ref, fr_tgt=fr_tgt,
                            bin_size=float(cd_res.conf.bin_size),
                            jscale=self._jscale(ctx))
        n_bins = len(jbsi)
        bin_size_eff = cd_res.conf.duration / (n_bins - 1) if n_bins > 1 else cd_res.conf.bin_size
        center = n_bins // 2
        lo = max(0, center + int(ctx.eff_min_lag / bin_size_eff))
        hi = min(n_bins, center + int(ctx.eff_max_lag / bin_size_eff))
        return self._fmt_cs_value(float(np.sum(jbsi[lo:hi])), nonneg=nonneg)

    def _get_cs(self, ref, tgt, seg, hr, method, metric,
                nonneg: bool = False, acg_deconv_active: bool = False,
                display_pair_temp: dict | None = None) -> str:
        ctx = self._ctx(ref, tgt, seg, hr)
        if metric == 'STG':
            return self._get_stg_cs(ctx, method, nonneg=nonneg,
                                    acg_deconv_active=acg_deconv_active,
                                    display_pair_temp=display_pair_temp)
        return self._get_jbsi_cs(ctx, nonneg=nonneg)

    def _time_hours(self, ctx: _CSContext) -> float | None:
        cd = getattr(self._nav, '_cd', None)
        conf = getattr(getattr(self._nav, 'ccg_data', None), 'conf', None)
        if conf is None and cd is not None:
            conf = getattr(cd, 'conf', None)
        return getattr(conf, 'total_time_hours', None)

    def _tailed_ccg(self, ccg, ccg_null, ctx: _CSContext):
        d = self._pair_mgr._resolve_segment_data(ctx.ref, ctx.tgt, ctx.seg,
                                                 highres=ctx.highres, include_pval=False, include_acg=True)
        acg_ref = d['acg_ref'].copy().astype(float)
        acg_tgt = d['acg_tgt'].copy().astype(float)
        return CCGNorm.deconvolve(ccg, ccg_null, acg_ref, float(np.sum(acg_ref)),
                                  acg_tgt, float(np.sum(acg_tgt)), True, True)[0]

    def _compute_jbsi(self, ctx: _CSContext) -> float | None:
        nav = self._nav
        raw_cd = nav.cd
        cd = nav.ccg_data
        cd_res = ((raw_cd.ccg_for(ctx.nd_key, 'highres') if ctx.highres else None) or
                  (raw_cd.ccg.get(ctx.nd_key) if hasattr(raw_cd, 'ccg') else None) or cd)
        d = self._pair_mgr._resolve_segment_data(ctx.ref, ctx.tgt, ctx.seg,
                                                 highres=ctx.highres, include_pval=False, _cd=cd_res)
        real_ccg = d.get('ccg_raw')
        fr_ref, fr_tgt = self._firing_rates(ctx)
        if real_ccg is None or fr_ref is None:
            return None
        jbsi_arr = compute_jbsi(real_ccg=real_ccg,
                                j_ccg_avg=np.zeros_like(real_ccg, dtype=float),
                                fr_ref=fr_ref, fr_tgt=fr_tgt,
                                bin_size=float(cd_res.conf.bin_size),
                                jscale=self._jscale(ctx))
        n_bins = len(jbsi_arr)
        bin_size_eff = cd_res.conf.duration / (n_bins - 1) if n_bins > 1 else cd_res.conf.bin_size
        center = n_bins // 2
        lo = max(0, center + int(ctx.eff_min_lag / bin_size_eff)) if ctx.eff_min_lag is not None else 0
        hi = min(n_bins, center + int(ctx.eff_max_lag / bin_size_eff)) if ctx.eff_max_lag is not None else n_bins
        return float(np.sum(jbsi_arr[lo:hi]))

    def _compute_pair_conn_strength(self, ctx: _CSContext):
        nav = self._nav
        cd = nav.ccg_data
        _kw = dict(min_lag_override=ctx.eff_min_lag, max_lag_override=ctx.eff_max_lag)

        d = self._pair_mgr._resolve_segment_data(ctx.ref, ctx.tgt, ctx.seg,
                                                 highres=ctx.highres, include_pval=False, include_acg=False)
        ccg, ccg_null = CCGNorm.apply(
            d['ccg_raw'], d['ccg_null_raw'], ctx.ref, ctx.tgt, ctx.seg,
            nav.active_norms - {NormalizeBy.BASELINE},
            nav.neurons, custom_time_hours=self._time_hours(ctx))

        for method in ('conv', 'global', 'tailed'):
            ccg_in = self._tailed_ccg(ccg, ccg_null, ctx) if method == 'tailed' else ccg
            cs, bl = ConnectionStrength.conn_strength(ccg_in, ccg_null, cd.conf, method, **_kw)
            self._cache.put(self._cache_key(ctx, method), (cs, bl))

        self._cache.put(self._cache_key(ctx, 'JBSI'), (self._compute_jbsi(ctx), None))

        return self._cache.get(self._cache_key(ctx, nav.baseline_method)) or (None, None)
