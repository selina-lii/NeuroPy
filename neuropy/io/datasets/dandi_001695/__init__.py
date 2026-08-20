"""Conventions of DANDI 001695: how its sessions are named and which columns map to fields."""
from __future__ import annotations

from pathlib import Path


FIELDS = {
    'spike_times':  'spike_times',
    'neuron_type':  {'col': 'cell_type',
                     'map': {'Pyramidal Cell': 'pyr',
                             'Narrow Interneuron': 'inter',
                             'Wide Interneuron': 'inter'}},
    'peak_channel': 'maxWaveformCh',
    'shank_id':     'x_position_probe',
    'position':     ['x_position_probe', 'y_position_probe'],
    'waveforms':    'waveforms',
}

def session_name(path: Path) -> str:
    """sub-M01_ses-20240308T100000_ecephys -> M01-0308 (subject, then month and day)."""
    sub, ses = Path(path).stem.split('_')[:2]
    return f"{sub[len('sub-'):]}-{ses[len('ses-') + 4:len('ses-') + 8]}"
