"""Batch jitter: session x conn-type x segment, optionally restricted to a pair group.

Results key on (session, conn-type, segment) whatever group produced them, so a later
run over a different group merges into the same store rather than replacing it.
"""

from __future__ import annotations

import time
import types

import numpy as np

from neuropy.analyses.jitter import Jitter, JitterConfig

DEFAULT_NJITTER = 500


def pairs_for(cd, sd, ptr_key, group: str | None) -> list:
    """Screened pairs for *ptr_key*, narrowed to *group* when one is named."""
    valid = cd.ptr[ptr_key].pair_set
    if not group:
        return sorted(valid)
    sess = str(ptr_key.session)
    sd.ensure_groups_loaded_for([sess])
    return sorted(sd.groups.pairs_in_group(group, sess) & valid)


def run_batch(cd, sd, ptr_key, segment: str, *, group: str | None = None,
              njitter: int = DEFAULT_NJITTER, res_key: str = 'lo',
              jitter_chunk: int = 5, n_threads: int = 0,
              progress=None) -> dict:
    """Jitter every pair of *ptr_key* in *segment*; returns {(ref,tgt): jitter tuple}."""
    pairs = pairs_for(cd, sd, ptr_key, group)
    if not pairs:
        return {}
    resolution = 'highres' if res_key == 'hi' else 'lowres'
    key = ptr_key.change(resolution=resolution, segment=segment)
    data = cd.ccg_for(key)
    seg_idx = cd.segment_index(key, segment)
    neurons = _neurons_for(cd, key, segment)

    ptr = types.SimpleNamespace(
        inds=np.array([[seg_idx, r, t] for r, t in pairs]),
        stored_by_segment=True,
        edge_times=getattr(cd.ptr[ptr_key], 'edge_times', None),
        n_pairs=len(pairs),
    )
    jconf = JitterConfig(ccg=data.conf, njitter=njitter,
                         jitter_chunk=jitter_chunk, n_threads=n_threads)
    t0 = time.perf_counter()
    j = Jitter(key=key, neurons=neurons, conf=jconf, ccg_ptr=ptr, ccg_data=data)
    j.run()
    if progress is not None:
        progress(f"{ptr_key.session} {ptr_key.type_label()} {segment}: "
                 f"{len(pairs)} pairs, njitter={njitter}, "
                 f"{time.perf_counter() - t0:.1f}s")

    out = {}
    cache_seg = seg_idx if seg_idx > 0 else None   # dim0[0] is the whole session; the UI keys it None
    for i, (r, t) in enumerate(pairs):
        avg, lo, hi = j._j_ccg_cache.get(i, (None, None, None))
        out[(int(r), int(t), res_key, cache_seg)] = (
            avg, float(j.pval[i]), j.pval_bins[i], lo, hi, j.percentiles.get(i))
    return out


def _neurons_for(cd, key, segment: str):
    """Neurons over the segment's active fragments, so the null sees the CCG's own time."""
    src = cd.source_config(key, segment) if segment and segment != 'all' else None
    sliced = cd.nd.sliced_neurons_for(src) if src is not None else None
    return sliced[0] if sliced is not None else cd.nd.neurons_for(key)


def run_and_store(cd, sd, ptr_key, segment: str, *, group: str | None = None,
                  njitter: int = DEFAULT_NJITTER, res_key: str = 'lo',
                  save: bool = True, progress=None) -> int:
    """Run one batch and merge it into ``cd._jitter_results``; returns pairs added."""
    got = run_batch(cd, sd, ptr_key, segment, group=group, njitter=njitter,
                    res_key=res_key, progress=progress)
    if not got:
        return 0
    store = cd._jitter_results.setdefault(ptr_key.nd(), {})
    store.update(got)          # merge: another group's pairs in this segment survive
    if save:
        cd.save_jitter()
    return len(got)
