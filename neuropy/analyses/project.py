"""Project-level construction — one loader shared by the headless backend and the Qt GUI."""
from __future__ import annotations

from neuropy.analyses.neurons_dataset import NeuronsDataset, NeuronsDatasetConfig
from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig
from neuropy.analyses.pair_selection_data import SelectionDataset


def load_project(name: str, sessions: list,
                  duration: float = 20e-3, alpha: float = 0.05,
                  use_acceleration: bool = False):
    """Build (neurons, cd, sd) for `name` from caller-supplied sessions/params."""
    neurons = NeuronsDataset(sessions, NeuronsDatasetConfig(epochs=None))
    cd = CCGDataset(
        CCGConfig(use_acceleration=use_acceleration, duration=duration,
                  alpha=alpha, name=name),
        neurons)
    cd.missing_sessions()
    sd = SelectionDataset(cd)
    return neurons, cd, sd


# ---------------------------------------------------------------------------


def open_project(name: str, sessions: list):
    """Load (neurons, cd, sd) from `name`."""
    conf = CCGConfig(name=name)
    conf.load()
    neurons = NeuronsDataset(sessions, NeuronsDatasetConfig(epochs=None))
    cd = CCGDataset(conf, neurons)
    cd.missing_sessions()
    cd.load()
    sd = SelectionDataset(cd)
    sd.load()
    return neurons, cd, sd
