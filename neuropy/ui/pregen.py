#!/usr/bin/env python
"""
Standalone CCG PNG pre-generation script.

Launched as a subprocess by CCGReviewUI._launch_pregen_subprocess().
Reads a JSON job file, loads CCG data from disk, and renders PNGs to the
cache directory.  Prints progress lines to stdout; the parent UI polls them.

Usage:
    python pregen.py <job_file.json>
"""
from __future__ import annotations
import sys
import os
import json
import traceback
import numpy as np


def _load_ccg_data(hkl_path: str, nd_key: str):
    """Load a CCGData object for nd_key from a hickle file."""
    try:
        import hickle as hkl
        data = hkl.load(hkl_path)
        # Keys may be Key objects, not plain strings — match by str() representation
        by_str = {str(k): v for k, v in data.items()}
        return by_str.get(nd_key)
    except Exception as exc:
        print(f"[pregen] Could not load {hkl_path}: {exc}", flush=True)
        return None


def main(job_path: str) -> None:
    with open(job_path, encoding='utf-8') as f:
        job = json.load(f)

    nd_key        = job['nd_key']
    tmp_dir       = job['tmp_dir']
    n_segments    = job['n_segments']
    segment_names = job['segment_names']
    pairs         = job['pairs']          # [[ref, tgt], ...]
    has_highres   = job.get('has_highres', False)
    cache_config  = job['cache_config']

    ccg_lo_path = job.get('ccg_lo_path')
    ccg_hi_path = job.get('ccg_hi_path')

    neurons_fr    = job.get('neurons_firing_rate')  # list[float] or None
    neurons_shank = job.get('neurons_shank_ids')    # list[int] or None
    edge_times    = job.get('edge_times')           # list[float] per seg or None

    # ── Load CCG data ──────────────────────────────────────────────────
    ccg_lo = None
    ccg_hi = None

    if ccg_lo_path and os.path.isfile(ccg_lo_path):
        ccg_lo = _load_ccg_data(ccg_lo_path, nd_key)
    else:
        print(f"[pregen] CCG lo-res file not found: {ccg_lo_path}", flush=True)

    if has_highres and ccg_hi_path and os.path.isfile(ccg_hi_path):
        ccg_hi = _load_ccg_data(ccg_hi_path, nd_key)

    if ccg_lo is None:
        print(f"[pregen] No lo-res CCG data for {nd_key!r}, aborting.", flush=True)
        return

    # ── Build renderer ────────────────────────────────────────────────
    from neuropy.ui.ccg_render import CCGRenderer, _NeuronsProxy

    neurons = None
    if neurons_fr is not None:
        neurons = _NeuronsProxy(
            firing_rate=np.array(neurons_fr, dtype=float),
            shank_ids=(np.array(neurons_shank, dtype=int)
                       if neurons_shank is not None else None),
        )

    renderer = CCGRenderer(
        ccg_lo=ccg_lo,
        ccg_hi=ccg_hi,
        n_segments=n_segments,
        segment_names=segment_names,
        tmp_dir=tmp_dir,
        cache_config=cache_config,
        neurons=neurons,
        edge_times=edge_times,
        nd_key=nd_key,
    )

    # ── Work list ─────────────────────────────────────────────────────
    segs     = list(range(n_segments)) + [n_segments]  # include virtual "All"
    res_list = ([False, True] if (has_highres and ccg_hi is not None)
                else [False])
    total    = len(pairs) * len(segs) * len(res_list)
    n_done   = 0
    n_skip   = 0

    print(f"[pregen] Starting: {total} items "
          f"({len(pairs)} pairs × {len(segs)} segs × {len(res_list)} res)",
          flush=True)

    for pair in pairs:
        ref, tgt = int(pair[0]), int(pair[1])
        for seg in segs:
            for hires in res_list:
                path = renderer.png_path(ref, tgt, seg, hires)
                if os.path.exists(path):
                    n_skip += 1
                    continue
                try:
                    renderer.render_png(ref, tgt, seg, hires)
                    n_done += 1
                    if n_done % 100 == 0 or n_done == 1:
                        print(f"[pregen] progress {n_done}/{total} done, "
                              f"{n_skip} skipped",
                              flush=True)
                except Exception as exc:
                    print(f"[pregen] ERROR ({ref},{tgt}) seg={seg} "
                          f"hires={hires}: {exc}",
                          flush=True)
                    traceback.print_exc(file=sys.stdout)

    print(f"[pregen] done {n_done} rendered, {n_skip} skipped", flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pregen.py <job_file.json>")
        sys.exit(1)
    main(sys.argv[1])
