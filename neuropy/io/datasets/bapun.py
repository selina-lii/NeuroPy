"""Conventions of the Bapun-lab dataset: how its sessions are named and labelled."""
from __future__ import annotations

from pathlib import Path


EPOCHS = ['pre', 'maze', 'post', 're-maze']

THEMES = ['paradigm', 'brainstates', 'ripple']


def session_name(session) -> str:
    return '_'.join(Path(session.filePrefix).name.split('_')[:2])
