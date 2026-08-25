"""Conventions of the Hiro-lab dataset: how its sessions are named and labelled."""
from __future__ import annotations

from pathlib import Path

from neuropy.io.datasets.hiro.reader import HiroSession, sessions

EPOCHS = ['pre', 'maze', 'post']

THEMES = ['paradigm', 'brainstates', 'ripple', 'spindle']

# Written NeuroPy-named, so a converted file needs no field map to read back.
FIELDS = {
    'spike_times':  'spike_times',
    'neuron_type':  'neuron_type',
    'shank_id':     'shank_id',   # the export carries no per-unit channel
}


def session_name(path: Path) -> str:
    """RoySleep1.nwb -> RoySleep1 (the export's own session key)."""
    return Path(path).stem
