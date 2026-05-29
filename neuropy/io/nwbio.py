"""NWB file reader — parses pynwb objects into NeuroPy primitives."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pynwb
except ImportError as _e:
    raise ImportError(
        "pynwb is required for NWB support.  Install with: pip install pynwb"
    ) from _e


class NWBFile:
    """Lazy reader wrapping a single .nwb file.

    Parses on first access; underlying HDF5 file stays open for the lifetime
    of this object.  Call .close() when done, or use as a context manager.
    """

    def __init__(self, path: str | Path, neuron_type_col: str = 'cell_type'):
        self._path = Path(path)
        self._neuron_type_col = neuron_type_col
        self._io = pynwb.NWBHDF5IO(str(self._path), mode='r', load_namespaces=True)
        self._nwb = self._io.read()

    def close(self):
        self._io.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Spike trains ────────────────────────────────────────────────────

    @cached_property
    def spiketrains(self) -> list[np.ndarray]:
        units = self._nwb.units
        if units is None:
            return []
        return [np.asarray(units['spike_times'][i], dtype=float)
                for i in range(len(units))]

    @cached_property
    def neuron_ids(self) -> np.ndarray:
        units = self._nwb.units
        if units is None:
            return np.array([], dtype=int)
        return np.asarray(units.id[:], dtype=int)

    @cached_property
    def neuron_type(self) -> np.ndarray:
        units = self._nwb.units
        if units is None:
            return np.array([], dtype=object)
        col = self._neuron_type_col
        if units is not None and col in units.colnames:
            return np.asarray(units[col][:], dtype=object)
        return np.array(['pyr'] * len(units), dtype=object)

    @cached_property
    def peak_channels(self) -> np.ndarray | None:
        units = self._nwb.units
        if units is None:
            return None
        for cname in ('peak_channel', 'max_channel', 'electrode_id'):
            if cname in units.colnames:
                return np.asarray(units[cname][:], dtype=int)
        # fall back to first electrode index per unit if linked
        try:
            return np.array([
                int(units['electrodes'][i].index[0])
                for i in range(len(units))
            ], dtype=int)
        except Exception:
            return None

    @cached_property
    def shank_ids(self) -> np.ndarray | None:
        units = self._nwb.units
        if units is None:
            return None
        for cname in ('shank_id', 'shank', 'group_id'):
            if cname in units.colnames:
                return np.asarray(units[cname][:], dtype=int)
        # derive from electrodes group label if available
        try:
            elec_table = self._nwb.electrodes
            if elec_table is not None and 'group_name' in elec_table.colnames:
                groups = elec_table['group_name'][:]
                unique_groups = {g: i for i, g in enumerate(dict.fromkeys(groups))}
                peak_ch = self.peak_channels
                if peak_ch is not None:
                    return np.array([unique_groups.get(groups[c], 0) for c in peak_ch],
                                    dtype=int)
        except Exception:
            pass
        return None

    @cached_property
    def waveforms(self) -> np.ndarray | None:
        units = self._nwb.units
        if units is None:
            return None
        for cname in ('waveform_mean', 'waveforms', 'mean_waveform'):
            if cname in units.colnames:
                try:
                    return np.array([units[cname][i] for i in range(len(units))])
                except Exception:
                    return None
        return None

    @cached_property
    def t_stop(self) -> float:
        # prefer explicit session end stored in file
        try:
            if hasattr(self._nwb, 'session_description'):
                pass  # not the stop time
            stop = self._nwb.trials  # sometimes carries timing
        except Exception:
            pass
        # fallback: max spike time + small buffer
        all_spikes = self.spiketrains
        if all_spikes:
            max_t = max((st.max() for st in all_spikes if len(st)), default=0.0)
            return float(max_t) + 1.0
        return 1.0

    # ── Epochs ──────────────────────────────────────────────────────────

    def _load_intervals(self, table_name: str) -> pd.DataFrame | None:
        """Load a named intervals table → DataFrame with start/stop/label columns."""
        table = None
        if table_name == 'epochs':
            table = self._nwb.epochs
        else:
            ivs = self._nwb.intervals
            if ivs is not None and table_name in ivs:
                table = ivs[table_name]
        if table is None:
            return None
        df = table.to_dataframe()
        # normalise column names
        rename = {}
        for src, dst in [('start_time', 'start'), ('stop_time', 'stop'),
                         ('tags', 'label'), ('label', 'label')]:
            if src in df.columns and dst not in df.columns:
                rename[src] = dst
        df = df.rename(columns=rename)
        if 'label' not in df.columns:
            df['label'] = table_name
        else:
            # tags column from NWB is often a list; take first element
            df['label'] = df['label'].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and x else str(x))
        # ensure start/stop columns exist
        if 'start' not in df.columns or 'stop' not in df.columns:
            return None
        return df[['start', 'stop', 'label']].reset_index(drop=True)

    def paradigm_df(self, table_name: str = 'epochs') -> pd.DataFrame | None:
        return self._load_intervals(table_name)

    def brainstates_df(self) -> pd.DataFrame | None:
        for name in ('brainstates', 'brain_states', 'sleep_states'):
            df = self._load_intervals(name)
            if df is not None:
                return df
        return None

    def ripple_df(self) -> pd.DataFrame | None:
        for name in ('ripple', 'ripples', 'SWR', 'sharp_wave_ripples'):
            df = self._load_intervals(name)
            if df is not None:
                return df
        return None
