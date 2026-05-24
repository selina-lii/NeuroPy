"""Jitter worker for multiprocessing — lightweight, no UI imports."""
import numpy as np


def jitter_worker(queue, key, neurons, ccg_data, edge_times,
                  ref, tgt, njitter, bin_size_eff,
                  segment=None, t0=None, t1=None):
    """Run jitter in a separate process (no GIL contention).

    Communicates result back via *queue*.  All arguments must be picklable.

    Parameters
    ----------
    segment : int or None
        If not None, run jitter for this specific segment only (spike trains
        filtered to [t0, t1] and real CCG taken from ccg_data.ccg[segment]).
        None means whole-session (all segments summed).
    t0, t1 : float or None
        Time boundaries for the segment window (used when segment is not None).
    """
    try:
        import copy
        import types
        from neuropy.analyses.jitter import Jitter, JitterConfig

        conf_eff = copy.copy(ccg_data.conf)
        conf_eff.bin_size = bin_size_eff
        jconf = JitterConfig(ccg=conf_eff, njitter=njitter)

        if segment is not None and t0 is not None and t1 is not None:
            # Segment-specific jitter: filter spike trains to [t0, t1]
            neurons_eff = neurons.time_slice(t_start=t0, t_stop=t1)
            ptr = types.SimpleNamespace(
                inds=np.array([[segment, ref, tgt]]),
                stored_by_segment=True,
                edge_times=edge_times,
                n_pairs=1,
            )
        else:
            neurons_eff = neurons
            ptr = types.SimpleNamespace(
                inds=np.array([[ref, tgt]]),
                stored_by_segment=False,
                edge_times=edge_times,
                n_pairs=1,
            )
        j = Jitter(key=key, neurons=neurons_eff, conf=jconf,
                    ccg_ptr=ptr, ccg_data=ccg_data)
        j.run()

        j_avg, j_lo, j_hi = j._j_ccg_cache.get(0, (None, None, None))
        j_pval = (float(j.pval[0])
                  if j.pval is not None and len(j.pval) else None)
        j_pval_bins = j.pval_bins[0] if j.pval_bins is not None else None
        queue.put({
            'ref': ref, 'tgt': tgt,
            'j_avg': j_avg, 'j_lo': j_lo, 'j_hi': j_hi,
            'j_pval': j_pval, 'j_pval_bins': j_pval_bins,
            'error': None,
        })
    except Exception as ex:
        queue.put({'error': str(ex)})
