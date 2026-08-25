"""Conventions of DANDI 001695: how its sessions are named and which columns map to fields."""
from __future__ import annotations

from pathlib import Path


# NP2.0: 4 shanks 250um apart, 2 columns 32um apart; no shank field, so fold it out of x
SHANK_PITCH_UM = 250
_X0 = 27
SHANK_FROM_X = {_X0 + col + SHANK_PITCH_UM * shank: shank
                for shank in range(4) for col in (0, 32)}

FIELDS = {
    'spike_times':  'spike_times',
    'neuron_type':  {'col': 'cell_type',
                     'map': {'Pyramidal Cell': 'pyr',
                             'Narrow Interneuron': 'inter',
                             'Wide Interneuron': 'inter'}},
    'peak_channel': 'maxWaveformCh',
    'shank_id':     {'col': 'x_position_probe', 'map': SHANK_FROM_X},
    'position':     ['x_position_probe', 'y_position_probe'],
    'waveforms':    'waveforms',
    'cell_area':    'cell_area',
}

def session_name(path: Path) -> str:
    """sub-M01_ses-20240308T100000_ecephys -> M01-0308 (subject, then month and day)."""
    sub, ses = Path(path).stem.split('_')[:2]
    return f"{sub[len('sub-'):]}-{ses[len('ses-') + 4:len('ses-') + 8]}"
