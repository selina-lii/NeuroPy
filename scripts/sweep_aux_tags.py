"""Result 1: sweep the five auxiliary tests over the post_* segment families.

A pair is on for a (session, segment) only when all five rules pass.
Writes AuxResults: per-rule value/passed grids plus the combined on, one unit dir per segment.
"""

import sys
import time

import numpy as np

from neuropy.analyses.ccg_transforms import AuxTestConfig, ConnectionStrength
from neuropy.analyses.ms_connectivity import open_project
from neuropy.analyses.pair_selection_data import SegmentGroups

FAMILIES = ('post_tenth', 'post_tenth_sleep-only', 'post_half')
ALL_TESTS = tuple(ConnectionStrength.AUX_TESTS)


def target_segments(cd) -> list:
    """Every post_* segment on disk, deduped, in family order."""
    have = set(cd.available_segments())
    return [s for s in sorted(have)
            if any(s.startswith(f) for f in FAMILIES)]


def sweep(cd, sd, segments, tests=ALL_TESTS, verbose=True) -> dict:
    """Run every test over every (ptr key, segment); returns a per-family tally."""
    cfg = AuxTestConfig(tuple(tests))
    cd.aux_test_config = cfg
    tally = {'pairs': 0, 'on': 0, 'segments': 0, 'skipped': 0, 'by_test': {t: 0 for t in tests}}

    for ptr_key, ptr in sorted(cd.ptr.items(), key=str):
        pairs = sorted(ptr.pair_set)
        if not pairs:
            continue
        sess = str(ptr_key.session)
        ct = ptr_key.type_label()
        for seg in segments:
            if seg not in cd.available_segments(ptr_key):
                continue                      # that session never computed this segment
            key = ptr_key.change(segment=seg)
            try:
                results = cd.sweep_aux_tests(key, tests=cfg.specs(),
                                             overrides=sd.seg_overrides(sess, seg))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                tally['skipped'] += 1
                if verbose:
                    print(f"  skip {sess} {seg}: {type(exc).__name__} {exc}", flush=True)
                continue

            store = sd.aux_results(ptr_key, seg)
            store.set_results(*results)
            store.save()
            grids, on = results
            idx = tuple(np.array(pairs).T)
            tally['pairs'] += len(pairs)
            for t in tests:
                tally['by_test'][t] += int(grids[t][1][idx].sum())
            tally['on'] += int(on[idx].sum())
            tally['segments'] += 1
            if verbose:
                print(f"  {sess:16s} {ct:12s} {seg:32s} "
                      f"{int(on[idx].sum()):4d}/{len(pairs):4d} on", flush=True)
    return tally


def main():
    t0 = time.perf_counter()
    nd, cd, sd = open_project('test2')
    segments = target_segments(cd)
    print(f"[sweep] {len(segments)} segments: {segments}", flush=True)
    tally = sweep(cd, sd, segments)
    sd.save()
    cd.conf.save()   # the enabled-test choice is what 'on' means; it has to outlive this run
    print(f"\n[sweep] {tally['segments']} (key, segment) combos, "
          f"{tally['pairs']} pair-segments, {tally['on']} on "
          f"({100.0 * tally['on'] / max(tally['pairs'], 1):.1f}%), "
          f"{tally['skipped']} skipped, {time.perf_counter() - t0:.1f}s")
    for t, n in tally['by_test'].items():
        print(f"    {t:24s} passed {n:6d} / {tally['pairs']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
