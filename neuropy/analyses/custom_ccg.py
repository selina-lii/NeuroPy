"""
Backend helpers for custom CCG computation and persistence.

These are pure-computation / pure-I/O functions with no tkinter dependency.
The UI layer (ccg_ui.py) wraps them with error dialogs and widget updates.
"""
from __future__ import annotations

import json

import numpy as np

from neuropy.analyses.correlations import spike_correlations
from neuropy.analyses.ms_connectivity import EranConv, _CCG_RESOLUTION


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
) -> dict:
    """Compute the full CCG pipeline for an arbitrary time window.

    Runs spike_correlations → EranConv._conv → multiple_correction at low-res
    (1 ms) and, when *has_highres* is True, also at high-res (0.1 ms) so that
    the Ctrl+R resolution toggle works on custom segments.

    Parameters
    ----------
    t0, t1 : float
        Window boundaries in seconds.
    name : str
        Segment label stored in the result dict.
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
        Arbitrary metadata stored verbatim in the result dict.

    Returns
    -------
    dict
        Keys: name, t0, t1, ccg, ccg_null, pval, pval_corrected, firing_rates,
        active_duration, total_time_hours, metadata.
        Plus ccg_hi / ccg_null_hi / pval_hi / pval_corrected_hi when has_highres.

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

    def _run_pipeline(bin_size: float, label: str):
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
        # Compute W from the actual bin_size used, not from conf
        # (conf.bin_size may be mutated to high-res)
        W = conf.conv_window / bin_size
        pvals, pred, qvals = EranConv._conv(ccg, W=W, wintype="gauss", hollow_frac=None)
        p_raw = pvals if excitability == 'E' else qvals
        _, pval_corrected = EranConv.multiple_correction(p_raw, conf.alpha, method=method)
        print(f"[CustomSegment] {label} done. shape={ccg.shape}")
        return ccg, pred, p_raw, pval_corrected

    lo_bs = _CCG_RESOLUTION['lowres']
    ccg_lo, pred_lo, pval_lo, pvalc_lo = _run_pipeline(lo_bs, 'lowres')

    firing_rates = np.array(
        [len(st) for st in neurons_slice.spiketrains],
        dtype=float) / max(active_duration, 1e-9)

    result = {
        'name':             name,
        't0':               t0,
        't1':               t1,
        'ccg':              ccg_lo,
        'ccg_null':         pred_lo,
        'pval':             pval_lo,
        'pval_corrected':   pvalc_lo,
        'firing_rates':     firing_rates,
        'active_duration':  active_duration,
        # active recording time in hours — needed for TIME_SPAN normalisation
        'total_time_hours': active_duration / 3600.0,
        'metadata':         metadata or {},
    }

    if has_highres:
        hi_bs = _CCG_RESOLUTION['highres']
        ccg_hi, pred_hi, pval_hi, pvalc_hi = _run_pipeline(hi_bs, 'highres')
        result['ccg_hi'] = ccg_hi
        result['ccg_null_hi'] = pred_hi
        result['pval_hi'] = pval_hi
        result['pval_corrected_hi'] = pvalc_hi

    return result


def save_custom_segment_to_npz(segment: dict, path: str) -> None:
    """Write a custom segment dict to a compressed NumPy archive at *path*."""
    arrays = dict(
        name_=np.array(segment['name']),
        t0_=np.array(segment['t0']),
        t1_=np.array(segment['t1']),
        ccg=segment['ccg'],
        ccg_null=segment['ccg_null'],
        pval=segment['pval'],
        pval_corrected=segment['pval_corrected'],
        compute_sec_=np.array(segment.get('compute_sec', float('nan'))),
        active_duration_=np.array(segment.get('active_duration', float('nan'))),
        total_time_hours_=np.array(segment.get('total_time_hours', float('nan'))),
        filter_state_=np.array(json.dumps(segment.get('filter_state', {}))),
        metadata_=np.array(json.dumps(segment.get('metadata', {}))),
        **({'firing_rates': segment['firing_rates']} if 'firing_rates' in segment else {}),
    )
    for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
        if k in segment:
            arrays[k] = segment[k]
    np.savez_compressed(path, **arrays)


def load_custom_segment_from_npz(path: str) -> dict:
    """Load a custom segment dict from a compressed NumPy archive.

    Raises on failure — caller should catch and skip the file.
    """
    npz = np.load(path, allow_pickle=False)
    cs = dict(
        name=str(npz['name_']),
        t0=float(npz['t0_']),
        t1=float(npz['t1_']),
        ccg=npz['ccg'],
        ccg_null=npz['ccg_null'],
        pval=npz['pval'],
        pval_corrected=npz['pval_corrected'],
        compute_sec=(float(npz['compute_sec_']) if 'compute_sec_' in npz else float('nan')),
        active_duration=(float(npz['active_duration_']) if 'active_duration_' in npz else float('nan')),
        total_time_hours=(float(npz['total_time_hours_']) if 'total_time_hours_' in npz else None),
        filter_state=(json.loads(str(npz['filter_state_'])) if 'filter_state_' in npz else {}),
        metadata=(json.loads(str(npz['metadata_'])) if 'metadata_' in npz else {}),
        src_path=path,
        **(({'firing_rates': npz['firing_rates']}) if 'firing_rates' in npz else {}),
    )
    for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
        if k in npz:
            cs[k] = npz[k]
    return cs


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
