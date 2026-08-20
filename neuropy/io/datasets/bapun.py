"""Conventions of the Bapun-lab dataset: how its sessions are named and labelled."""
from __future__ import annotations

from pathlib import Path


EPOCHS = ['pre', 'maze', 'post', 're-maze']

THEMES = ['paradigm', 'brainstates', 'ripple']


def session_name(path: Path) -> str:
    """/…/RatJ_Day1_2019-05-31_… -> RatJ_Day1 (subject and day)."""
    return '_'.join(Path(path).name.split('_')[:2])
