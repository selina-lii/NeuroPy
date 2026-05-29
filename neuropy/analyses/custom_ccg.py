"""
Backend helpers for custom CCG computation and persistence.

These are pure-computation / pure-I/O functions with no tkinter dependency.
The UI layer (ccg_ui.py) wraps them with error dialogs and widget updates.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from neuropy.analyses.correlations import spike_correlations
from neuropy.analyses.ms_connectivity import (CCGConfig, CCGData, CustomCCGMeta,
                                               EranConv, _CCG_RESOLUTION)


def compute_custom_ccg(
    t0: float,
    t1: float,
    name: str,
    neurons_slice,
    conf,
    *,
    has_highres: bool = False,
    active_duration: float | None = None,
    excitability: str = 'E',
    metadata: dict | None = None,
    filter_state: dict | None = None,
) -> tuple[CCGData, CCGData | None]:
    """Compute the full CCG pipeline for an arbitrary time window.

    Runs spike_correlations → EranConv._conv → multiple_correction at low-res
    (1 ms) and, when *has_highres* is True, also at high-res (0.1 ms) so that
    the Ctrl+R resolution toggle works on custom segments.

    Parameters
    ----------
    t0, t1 : float
        Window boundaries in seconds.
    name : str
        Segment label stored in the result.
    neurons_slice : Neurons
        Already time-sliced Neurons object (caller does time_slice(t0, t1)).
    conf : CCGConfig
        Computation parameters (alpha, conv_window, symmetrize_ccg, …).
    has_highres : bool
        Whether to also compute high-res CCG.
    active_duration : float or None
        Effective recording duration in seconds; defaults to t1 - t0.
    excitability : str
        'E' (use pvals) or 'I' (use qvals) for significance selection.
    metadata : dict or None
        Arbitrary metadata stored verbatim on the result.
    filter_state : dict or None
        Time-slider filter state at the time the segment was created.

    Returns
    -------
    (lo, hi)
        ``lo`` is always a ``CCGData`` with ``custom_meta`` populated.
        ``hi`` is a ``CCGData`` (arrays only, no custom_meta) when
        *has_highres* is True, else ``None``.

    Raises
    ------
    Exception
        Any computation error propagates to the caller (no messagebox here).
    """
    if active_duration is None:
        active_duration = t1 - t0

    n_neurons = neurons_slice.n_neurons
    neuron_inds = np.arange(n_neurons)
    method = conf.multiple_correction if conf.multiple_correction is not None else 'bonferroni'

    def _run_pipeline(bin_size: float, label: str) -> CCGData:
        print(f"[CustomSegment] computing {label} CCG for {name} "
              f"({t1-t0:.1f}s, {n_neurons} neurons, "
              f"bin={bin_size*1e3:.2f}ms) ...")
        ccg = spike_correlations(
            neurons=neurons_slice,
            neuron_inds=neuron_inds,
            bin_size=bin_size,
            window_size=conf.duration,
            symmetrize=conf.symmetrize_ccg,
            use_acceleration=conf.use_acceleration,
        )
        ccg = ccg[np.newaxis, ...]
        W = conf.conv_window / bin_size
        pvals, pred, qvals = EranConv._conv(ccg, W=W, wintype="gauss", hollow_frac=None)
        p_raw = pvals if excitability == 'E' else qvals
        _, pval_corrected = EranConv.multiple_correction(p_raw, conf.alpha, method=method)
        print(f"[CustomSegment] {label} done. shape={ccg.shape}")
        conf_res = CCGConfig(
            name=name,
            bin_size=bin_size,
            duration=conf.duration,
            conv_window=conf.conv_window,
            alpha=conf.alpha,
            multiple_correction=method,
        )
        return CCGData(
            key=None, _conf=conf_res,
            ccg=ccg, ccg_null=pred,
            pval=p_raw, qval=None,
            pval_corrected=pval_corrected, qval_corrected=None,
            significant=None, norm_factors=None, conn_strength=None,
        )

    lo = _run_pipeline(_CCG_RESOLUTION['lowres'], 'lowres')

    firing_rates = np.array(
        [len(st) for st in neurons_slice.spiketrains],
        dtype=float) / max(active_duration, 1e-9)

    lo.custom_meta = CustomCCGMeta(
        name=name,
        t0=t0,
        t1=t1,
        active_duration=active_duration,
        total_time_hours=active_duration / 3600.0,
        firing_rates=firing_rates,
        filter_state=filter_state or {},
        metadata=metadata or {},
    )

    hi = _run_pipeline(_CCG_RESOLUTION['highres'], 'highres') if has_highres else None

    return lo, hi


# ---------------------------------------------------------------------------
# CustomSegmentNpz — schema-driven NPZ serialisation
# ---------------------------------------------------------------------------

class CustomSegmentNpz:
    """Schema-driven NPZ serialiser for ``CCGData`` custom segments.

    Field schema is defined once; ``dump`` and ``load`` both derive from it so
    the two cannot drift.  All metadata lives on ``lo.custom_meta``
    (a ``CustomCCGMeta``); ``hi`` carries only the CCG arrays.
    """

    # (CustomCCGMeta attribute, python type, default when absent in npz)
    _SCALAR = [
        ('name',             str,   ''),
        ('t0',               float, 0.0),
        ('t1',               float, 0.0),
        ('compute_sec',      float, float('nan')),
        ('active_duration',  float, float('nan')),
        ('total_time_hours', float, float('nan')),
    ]
    # JSON-encoded scalar fields
    _JSON = [('filter_state', {}), ('metadata', {})]
    # lo / hi array field names
    _LO  = ('ccg', 'ccg_null', 'pval', 'pval_corrected')
    _HI  = ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi')

    @classmethod
    def dump(cls, lo: CCGData, path: str, hi: CCGData | None = None) -> None:
        """Write *lo* (and optionally *hi*) to a compressed NumPy archive."""
        m = lo.custom_meta or CustomCCGMeta(
            name='', t0=0.0, t1=0.0, active_duration=0.0, total_time_hours=0.0)
        arrays = {}
        for attr, typ, _ in cls._SCALAR:
            arrays[attr + '_'] = np.array(typ(getattr(m, attr)))
        for attr, default in cls._JSON:
            arrays[attr + '_'] = np.array(json.dumps(getattr(m, attr) or default))
        for attr in cls._LO:
            arrays[attr] = getattr(lo, attr)
        if hi is not None:
            for lo_k, hi_k in zip(cls._LO, cls._HI):
                val = getattr(hi, lo_k, None)
                if val is not None:
                    arrays[hi_k] = val
        if m.firing_rates is not None:
            arrays['firing_rates'] = m.firing_rates
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> tuple[CCGData, CCGData | None]:
        """Load from a compressed NumPy archive.

        Returns ``(lo, hi)`` where ``lo.custom_meta`` is populated and
        ``hi`` is ``None`` if no high-res data was stored.

        Raises on failure — caller should catch and skip the file.
        """
        npz = np.load(path, allow_pickle=False)
        meta_kw: dict = {}
        for attr, typ, default in cls._SCALAR:
            k = attr + '_'
            meta_kw[attr] = typ(npz[k]) if k in npz else default
        for attr, default in cls._JSON:
            k = attr + '_'
            meta_kw[attr] = json.loads(str(npz[k])) if k in npz else default
        meta_kw['firing_rates'] = npz['firing_rates'] if 'firing_rates' in npz else None
        meta_kw['src_path'] = path
        meta = CustomCCGMeta(**meta_kw)
        lo = CCGData(
            key=None, _conf=None,
            ccg=npz['ccg'], ccg_null=npz['ccg_null'],
            pval=npz['pval'], qval=None,
            pval_corrected=npz['pval_corrected'], qval_corrected=None,
            significant=None, norm_factors=None, conn_strength=None,
            custom_meta=meta,
        )
        hi = None
        if 'ccg_hi' in npz:
            hi = CCGData(
                key=None, _conf=None,
                ccg=npz['ccg_hi'],
                ccg_null=npz.get('ccg_null_hi'),
                pval=npz.get('pval_hi'), qval=None,
                pval_corrected=npz.get('pval_corrected_hi'), qval_corrected=None,
                significant=None, norm_factors=None, conn_strength=None,
            )
        return lo, hi


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases (dict ↔ CCGData conversion)
#
# Callers that still use the old dict format (ccg_ui.py, time_slider.py,
# custom_ccg_manager.py) continue to work unchanged.  New code should use
# ``CustomSegmentNpz.dump / .load`` directly with ``CCGData`` objects.
# ---------------------------------------------------------------------------

def _dict_to_lo_hi(segment: dict) -> tuple[CCGData, 'CCGData | None']:
    """Build (lo, hi) CCGData pair from the legacy segment dict format."""
    meta = CustomCCGMeta(
        name=str(segment.get('name', '')),
        t0=float(segment.get('t0', 0.0)),
        t1=float(segment.get('t1', 0.0)),
        active_duration=float(segment.get('active_duration', float('nan'))),
        total_time_hours=float(segment.get('total_time_hours', float('nan'))),
        firing_rates=segment.get('firing_rates'),
        filter_state=segment.get('filter_state') or {},
        metadata=segment.get('metadata') or {},
        compute_sec=float(segment.get('compute_sec', float('nan'))),
        src_path=segment.get('src_path'),
    )
    lo = CCGData(
        key=None, _conf=None,
        ccg=segment['ccg'], ccg_null=segment['ccg_null'],
        pval=segment['pval'], qval=None,
        pval_corrected=segment['pval_corrected'], qval_corrected=None,
        significant=None, norm_factors=None, conn_strength=None,
        custom_meta=meta,
    )
    hi = None
    if 'ccg_hi' in segment:
        hi = CCGData(
            key=None, _conf=None,
            ccg=segment['ccg_hi'],
            ccg_null=segment.get('ccg_null_hi'),
            pval=segment.get('pval_hi'), qval=None,
            pval_corrected=segment.get('pval_corrected_hi'), qval_corrected=None,
            significant=None, norm_factors=None, conn_strength=None,
        )
    return lo, hi


def _lo_hi_to_dict(lo: CCGData, hi: 'CCGData | None') -> dict:
    """Convert (lo, hi) CCGData pair back to the legacy segment dict format."""
    m = lo.custom_meta or CustomCCGMeta(
        name='', t0=0.0, t1=0.0, active_duration=0.0, total_time_hours=0.0)
    result = {
        'name':             m.name,
        't0':               m.t0,
        't1':               m.t1,
        'ccg':              lo.ccg,
        'ccg_null':         lo.ccg_null,
        'pval':             lo.pval,
        'pval_corrected':   lo.pval_corrected,
        'firing_rates':     m.firing_rates,
        'active_duration':  m.active_duration,
        'total_time_hours': m.total_time_hours,
        'filter_state':     m.filter_state,
        'metadata':         m.metadata,
        'compute_sec':      m.compute_sec,
        'src_path':         m.src_path,
    }
    if hi is not None:
        result['ccg_hi']             = hi.ccg
        result['ccg_null_hi']        = hi.ccg_null
        result['pval_hi']            = hi.pval
        result['pval_corrected_hi']  = hi.pval_corrected
    return result


def save_custom_segment_to_npz(segment: dict, path: str) -> None:
    """Save a legacy segment dict to a compressed NumPy archive."""
    lo, hi = _dict_to_lo_hi(segment)
    CustomSegmentNpz.dump(lo, path, hi)


def load_custom_segment_from_npz(path: str) -> dict:
    """Load a compressed NumPy archive and return the legacy segment dict."""
    lo, hi = CustomSegmentNpz.load(path)
    return _lo_hi_to_dict(lo, hi)


# ---------------------------------------------------------------------------
# RawCCGSpec — typed spec container
# ---------------------------------------------------------------------------

@dataclass
class RawCCGSpec:
    """Typed container for a custom CCG spec. t0/t1 may be float or 'start'/'end' sentinel."""
    name:                str
    t0:                  float | str
    t1:                  float | str
    scope:               str         = ''
    created_from_session: str        = ''
    sessions:            list = field(default_factory=list)
    n_splits:            int         = 1
    overlap_sec:         float       = 0.0
    filter_state:        dict        = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, *, default_session: str = '') -> 'RawCCGSpec':
        """Build from a raw spec dict, normalising values via normalize_custom_spec."""
        nd = normalize_custom_spec(d, default_session=default_session)
        return cls(
            name=nd['name'], t0=nd['t0'], t1=nd['t1'],
            scope=nd['scope'], created_from_session=nd['created_from_session'],
            sessions=nd['sessions'], n_splits=nd['n_splits'],
            overlap_sec=nd['overlap_sec'], filter_state=nd.get('filter_state', {}),
        )

    @classmethod
    def from_npz(cls, npz, session: str) -> 'RawCCGSpec':
        """Build from a loaded NPZ archive (standard CCG npz field names)."""
        fs = json.loads(str(npz['filter_state_'])) if 'filter_state_' in npz else {}
        return cls(
            name=str(npz['name_']),
            t0=float(npz['t0_']),
            t1=float(npz['t1_']),
            scope=session,
            created_from_session=session,
            sessions=[session],
            filter_state=fs,
        )

    @staticmethod
    def infer_scope(sessions: list[str], all_sessions: list[str]) -> str:
        """Return 'All', a single session string, or 'By session' based on coverage."""
        if all_sessions and sorted(sessions) == sorted(all_sessions):
            return 'All'
        if len(sessions) == 1:
            return sessions[0]
        return 'By session'

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Spec helpers (unchanged)
# ---------------------------------------------------------------------------

def custom_spec_key(spec: dict) -> tuple:
    """Return a hashable key that uniquely identifies a custom CCG spec."""
    fs = spec.get('filter_state', {}) or {}
    labels = fs.get('labels', {}) or {}
    t0_raw = spec.get('t0', 0.0)
    t1_raw = spec.get('t1', 0.0)
    t0_key = str(t0_raw) if isinstance(t0_raw, str) else float(t0_raw)
    t1_key = str(t1_raw) if isinstance(t1_raw, str) else float(t1_raw)
    return (
        str(spec.get('name', '')),
        t0_key,
        t1_key,
        str(fs.get('theme', 'segments')),
        tuple(sorted((str(k), bool(v)) for k, v in labels.items())),
    )


def normalize_custom_spec(spec: dict, *, default_session: str = '') -> dict:
    """Normalise and fill defaults in a custom CCG spec dict."""
    fs = spec.get('filter_state', {}) or {}
    labels = fs.get('labels', {}) or {}
    sessions = spec.get('sessions', []) or []
    sessions = sorted(str(s) for s in sessions if s is not None)
    t0_raw = spec.get('t0', 0.0)
    t1_raw = spec.get('t1', 0.0)
    t0 = str(t0_raw) if isinstance(t0_raw, str) and t0_raw.lower() in ('start', 'end') else float(t0_raw)
    t1 = str(t1_raw) if isinstance(t1_raw, str) and t1_raw.lower() in ('start', 'end') else float(t1_raw)
    return {
        'name': str(spec.get('name', '')),
        't0': t0,
        't1': t1,
        'filter_state': {
            'theme': str(fs.get('theme', 'segments')),
            'labels': {str(k): bool(v) for k, v in labels.items()},
        },
        'scope': str(spec.get('scope', default_session)),
        'created_from_session': str(spec.get('created_from_session', default_session)),
        'sessions': sessions,
        'n_splits': int(spec.get('n_splits') or 1),
        'overlap_sec': float(spec.get('overlap_sec') or 0.0),
    }
