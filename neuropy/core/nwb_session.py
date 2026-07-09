"""NWBSession — duck-type shim over NWBFile that satisfies the ProcessData interface.

Usage::

    from neuropy.core.nwb_session import NWBSession
    from neuropy.analyses.neurons_dataset import NeuronsDataset, NeuronsDatasetConfig
    from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig
    from neuropy.ui.ccg_review import CCGReviewUI as _CCGReviewUI
def launch_ccg_review(cd, key=None): return _CCGReviewUI.launch(cd, key)

    sess = [NWBSession("recording.nwb")]
    nd = NeuronsDataset(sess, NeuronsDatasetConfig(epochs=['pre', 'maze', 'post']))
    cd = CCGDataset(CCGConfig(duration=20e-3, alpha=0.05, name='test'), nd)
    launch_ccg_review(cd)
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from neuropy.core.neurons import Neurons
from neuropy.core.epoch import Epoch
from neuropy.io.nwbio import NWBFile


class NWBSession:
    """Wraps a single .nwb file to look like a ProcessData session object.

    NeuronsDataset accesses: basepath, filePrefix, neurons_stable, paradigm,
    brainstates (optional), ripple (optional), recinfo (optional).
    All other ProcessData attributes are absent by design.
    """

    def __init__(
        self,
        nwb_path: str | Path,
        session_name: str | None = None,
        epoch_table: str = 'epochs',
        neuron_type_col: str = 'cell_type',
    ):
        """
        Parameters
        ----------
        nwb_path : path to the .nwb file
        session_name : name used as session identifier; defaults to file stem
        epoch_table : name of the NWB intervals table to use as paradigm epochs;
            'epochs' is the NWB standard table; override if your lab uses a
            different table name (e.g. 'trials', 'behavior')
        neuron_type_col : column in nwbfile.units holding cell type labels
            ('pyr', 'inter', etc.); falls back to all 'pyr' if absent
        """
        self._path = Path(nwb_path)
        self._session_name = session_name or self._path.stem
        self._epoch_table = epoch_table
        self._nwb = NWBFile(self._path, neuron_type_col=neuron_type_col)

    # ── ProcessData path attributes ────────────────────────────────────

    @property
    def basepath(self) -> Path:
        return self._path.parent

    @property
    def filePrefix(self) -> Path:
        return self._path.parent / self._session_name

    # ── recinfo (optional) ────────────────────────────────────────────
    # NeuronsDataset accesses only recinfo.skipped_channels; return a
    # minimal object so getattr(sess, 'recinfo', None) returns something
    # harmless rather than None (which would still work, but this is safer).

    @cached_property
    def recinfo(self):
        class _MinimalRecinfo:
            skipped_channels = np.array([], dtype=int)
        return _MinimalRecinfo()

    # ── Core data ─────────────────────────────────────────────────────

    @cached_property
    def neurons_stable(self) -> Neurons:
        nwb = self._nwb
        spiketrains = nwb.spiketrains
        if not spiketrains:
            raise ValueError(f"NWB file has no units: {self._path}")
        return Neurons(
            spiketrains=spiketrains,
            t_stop=nwb.t_stop,
            t_start=0.0,
            sampling_rate=1,
            neuron_ids=nwb.neuron_ids,
            neuron_type=nwb.neuron_type,
            waveforms=nwb.waveforms,
            peak_channels=nwb.peak_channels,
            shank_ids=nwb.shank_ids,
        )

    @cached_property
    def paradigm(self) -> Epoch:
        df = self._nwb.paradigm_df(self._epoch_table)
        if df is None or df.empty:
            # create a single 'session' epoch covering the full recording
            t_stop = self._nwb.t_stop
            df = pd.DataFrame({'start': [0.0], 'stop': [t_stop], 'label': ['session']})
        return Epoch(df)

    @cached_property
    def brainstates(self) -> Epoch | None:
        df = self._nwb.brainstates_df()
        return Epoch(df) if df is not None and not df.empty else None

    @cached_property
    def ripple(self) -> Epoch | None:
        df = self._nwb.ripple_df()
        return Epoch(df) if df is not None and not df.empty else None

    # ── Convenience ───────────────────────────────────────────────────

    def close(self):
        self._nwb.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"NWBSession('{self._session_name}', path='{self._path}')"
