"""Stamp each NWB file's spike clock into scratch, so loading stops re-deriving it.

`NWBFile.sampling_rate` recovers the clock by pooling and sorting every spike in the
session — 26 s across dandi001695, on every single load. Files we write already state
the rate outright; this does the same for files that arrived from upstream without one.

    python notebooks/stamp_sampling_rate.py <dir>            # report only
    python notebooks/stamp_sampling_rate.py <dir> --apply    # write in place
"""
import argparse
import uuid
from pathlib import Path

import h5py
import numpy as np

from neuropy.io.nwbio import NWBFile, NWB_DEFAULT

NAME = 'spike_sampling_rate'
# Recorded as inferred, not declared: recovered from spike quantization, not stated
# by the recording system, so a later reader can tell the two apart.
NOTES = 'clock the spike times quantize onto (Hz); inferred from spike quantization'


def stamp(path: Path, rate: float) -> None:
    """Append the rate as a ScratchData dataset, matching what nwbwriter emits."""
    with h5py.File(path, 'r+') as f:
        scratch = f.require_group('scratch')
        if NAME in scratch:
            del scratch[NAME]
        ds = scratch.create_dataset(NAME, data=np.array([rate], dtype=float))
        ds.attrs['namespace'] = 'core'
        ds.attrs['neurodata_type'] = 'ScratchData'
        ds.attrs['notes'] = NOTES
        ds.attrs['object_id'] = str(uuid.uuid4())


def main(root: str, apply: bool):
    paths = sorted(Path(root).glob('**/*.nwb'))
    for path in paths:
        with NWBFile(path, fields=NWB_DEFAULT) as f:
            declared = f.declared_sampling_rate
            if declared:
                print(f'{path.name[:56]:58s} already states {declared:g} Hz')
                continue
            rate = f.sampling_rate
        if rate is None:
            print(f'{path.name[:56]:58s} no spikes; skipped')
            continue
        print(f'{path.name[:56]:58s} {rate:g} Hz{"  written" if apply else ""}')
        if apply:
            stamp(path, rate)
    if not apply:
        print(f'\n{len(paths)} files inspected; re-run with --apply to write')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='directory of .nwb files')
    ap.add_argument('--apply', action='store_true', help='write (default: report only)')
    a = ap.parse_args()
    main(a.root, a.apply)
