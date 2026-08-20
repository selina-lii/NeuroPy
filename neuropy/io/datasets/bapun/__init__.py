"""Conventions of the Bapun-lab dataset: how its sessions are named, found, and labelled."""
from __future__ import annotations

from pathlib import Path

from neuropy.io.datasets.bapun.reader import ProcessData
from neuropy.io.datasets.bapun.sessions import ARMS, BASEDIR, NSD, SD

EPOCHS = ['pre', 'maze', 'post', 're-maze']

THEMES = ['paradigm', 'brainstates', 'ripple']


def session_name(path: Path) -> str:
    """/…/RatJ_Day1_2019-05-31_… -> RatJ_Day1 (subject and day)."""
    return '_'.join(Path(path).name.split('_')[:2])


def sessions(arms: list[str] = None, basedir=None) -> list[ProcessData]:
    """Read the named study arms; all of them when none are named."""
    root = Path(basedir) if basedir else BASEDIR
    return [ProcessData(root / rel, tag)
            for tag in (arms or ARMS) for rel in ARMS[tag]]
