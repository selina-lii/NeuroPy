#!/usr/bin/env python
"""Convert the Hiro MATLAB export to one .nwb per session.

    python notebooks/convert_hiro.py                       # everything
    python notebooks/convert_hiro.py --no-lfp              # skip spectrograms
    python notebooks/convert_hiro.py --session RoyMaze1    # just one

Conversion is one-time: the export is organised by data type, so reading a single
session means opening a multi-hundred-MB file either way. Afterwards nothing in
the pipeline reads the .mat tree.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np

from neuropy.io.datasets.hiro.reader import sessions
from neuropy.io.nwbwriter import write_nwb

SOURCE = Path('/Users/selinl/Documents/ms_synchrony/hiro')
DEST = Path('/Users/selinl/Documents/ms_synchrony/hiro_nwb')


def convert(s, dest: Path, lfp: bool = True, behavior: bool = True) -> Path | None:
    neurons = s.neurons()
    if neurons is None:
        print(f'  {s.name}: no spikes, skipped')
        return None
    basics = s.basics()
    epochs = {**s.epochs(), **s.events()}
    ts = s.timeseries() if behavior else {}
    spec = s.spectrograms() if lfp else {}

    regions = {}
    for shank in basics.get('ca1_shanks', []):
        regions[int(shank) - 1] = 'CA1'   # MATLAB shank numbers are 1-based
    md = {'subject': s.subject, 'lab': 'Hiro', 'session_description':
          f'{s.condition} session {s.name}; t_origin={s.t_origin():.6f}s '
          f'subtracted from the campaign clock',
          'identifier': s.name, 'device': 'silicon probe',
          'channel_regions': regions,
          'experiment_description':
              'PRE/track/POST' if s.condition == 'wake' else 'rest/sleep'}
    return write_nwb(dest / f'{s.name}.nwb', s.name, neurons=neurons,
                     epochs=epochs, timeseries=ts, spectrograms=spec,
                     probegroup=_probegroup(basics), metadata=md)


def _probegroup(basics: dict):
    """Shank/channel layout as a table; the export carries no physical geometry."""
    chans = basics.get('shank_channels')
    if not chans:
        return None
    import pandas as pd

    rows = [{'shank_id': i, 'channel_id': int(c), 'x': np.nan, 'y': np.nan}
            for i, shank in enumerate(chans) for c in shank]

    class _PG:
        def to_dataframe(self):
            return pd.DataFrame(rows)
    return _PG()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=str(SOURCE))
    ap.add_argument('--dest', default=str(DEST))
    ap.add_argument('--session', help='convert only this session')
    ap.add_argument('--condition', choices=['sleep', 'wake'])
    ap.add_argument('--no-lfp', action='store_true', help='skip spectrograms')
    ap.add_argument('--no-behavior', action='store_true',
                    help='skip position and speed')
    args = ap.parse_args()

    dest = Path(args.dest)
    todo = sessions(args.source, args.condition)
    if args.session:
        todo = [s for s in todo if s.name == args.session]
        if not todo:
            raise SystemExit(f'no session named {args.session!r}')

    print(f'converting {len(todo)} session(s) -> {dest}')
    total = 0.0
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        print(f'[{i}/{len(todo)}] {s.name} ({s.condition})…', flush=True)
        path = convert(s, dest, lfp=not args.no_lfp,
                       behavior=not args.no_behavior)
        if path:
            mb = path.stat().st_size / 1e6
            total += mb
            print(f'      {mb:8.1f} MB  {time.time() - t0:5.1f}s')
    print(f'\ndone: {total:.0f} MB in {dest}')
