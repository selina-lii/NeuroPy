"""Backfill firing_rates into segment configs written before the field existed.

Recomputes only the per-neuron rate over each segment's stored window; the CCG
arrays are untouched, so this is far cheaper than recomputing the segments.
"""
from __future__ import annotations

import argparse
import glob
import os

from neuropy.analyses.ms_connectivity import CCGSourceConfig, Key, open_project


def backfill(project: str, dry_run: bool = False) -> None:
    nd, cd, _sd = open_project(project)
    live = {str(k.session) for k in nd.session_keys}
    filled = skipped = 0

    for path in sorted(glob.glob(os.path.join(cd.custom_dir, '*.json'))):
        segment, session = os.path.basename(path)[:-len('.json')].split('.')
        if session not in live:
            skipped += 1
            continue
        src = CCGSourceConfig(key=Key(session=session, segment=segment))
        src._root = cd.save_path
        src.load()
        if src.firing_rates:
            continue
        sliced = nd.sliced_neurons_for(src)
        if sliced is None:
            skipped += 1
            continue
        neurons_slice, _ = sliced
        src.firing_rates = neurons_slice.firing_rate
        if not dry_run:
            src.save()
        filled += 1
        print(f"  {segment}.{session}: {len(src.firing_rates)} rates")

    print(f"[{project}] filled {filled}, skipped {skipped}"
          + (" (dry run — nothing written)" if dry_run else ""))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('project')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    backfill(a.project, a.dry_run)
