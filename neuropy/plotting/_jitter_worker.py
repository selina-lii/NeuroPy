"""Jitter worker for multiprocessing — lightweight, no UI imports."""
import numpy as np


def jitter_worker(queue, key, neurons, ccg_data, edge_times,
                  ref, tgt, njitter, bin_size_eff):
    """Run jitter in a separate process (no GIL contention).

    Communicates result back via *queue*.  All arguments must be picklable.
    """
    try:
        import copy
        import types
        from neuropy.analyses.jitter import Jitter, JitterConfig

        conf_eff = copy.copy(ccg_data.conf)
        conf_eff.bin_size = bin_size_eff
        jconf = JitterConfig(ccg=conf_eff, njitter=njitter)

        ptr = types.SimpleNamespace(
            inds=np.array([[ref, tgt]]),
            stored_by_segment=False,
            edge_times=edge_times,
            n_pairs=1,
        )
        j = Jitter(key=key, neurons=neurons, conf=jconf,
                    ccg_pointer=ptr, ccg_data=ccg_data)
        j.run()

        j_avg, _, _ = j._j_ccg_cache.get(0, (None, None, None))
        j_pval = (float(j.pval[0])
                  if j.pval is not None and len(j.pval) else None)
        j_pval_bins = j.pval_bins[0] if j.pval_bins is not None else None
        queue.put({
            'ref': ref, 'tgt': tgt,
            'j_avg': j_avg, 'j_pval': j_pval, 'j_pval_bins': j_pval_bins,
            'error': None,
        })
    except Exception as ex:
        queue.put({'error': str(ex)})
