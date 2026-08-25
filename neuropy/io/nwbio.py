"""NWB file reader — parses pynwb objects into NeuroPy primitives."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from neuropy.io.fieldmap import ANY, Field, FieldMap, FieldSchema, ONE, OPTIONAL

try:
    import pynwb
except ImportError as _e:
    raise ImportError(
        "pynwb is required for NWB support.  Install with: pip install pynwb"
    ) from _e


# What a unit table must supply for the CCG pipeline. Arity and value_map are the
# constraints the mapping widget enforces; a dataset satisfies them with a plain dict.
UNITS_SCHEMA = FieldSchema([
    Field('spike_times',   ONE,      note='per-unit spike train'),
    Field('neuron_type',   ONE,      value_map=True, values=['pyr', 'inter'],
          note="cell class; values must end up 'pyr' / 'inter'"),
    Field('neuron_id',     OPTIONAL, note='unit id; defaults to the units table row id'),
    Field('peak_channel',  OPTIONAL, note='channel of largest waveform'),
    Field('shank_id',      OPTIONAL, value_map=True,
          note='probe shank grouping; value_map folds probe coordinates into indices'),
    Field('position',      ANY,      note='unit position on the probe: x, then y'),
    Field('waveforms',     OPTIONAL, note='mean waveform per unit'),
    Field('cell_area',     OPTIONAL, value_map=True,
          note='brain region the unit was recorded in'),
])


# A dataset's field map: {"target": "input"}. The value is a column name, a list of
# them, or {'col': ..., 'map': {...}} when the column's values need renaming too.
# Per-dataset maps live in that dataset's module, e.g. io/dandi_001695.py.

NWB_DEFAULT = {
    'spike_times':  'spike_times',
    'neuron_type':  'cell_type',
    'peak_channel': 'peak_channel',
    'shank_id':     'shank_id',
    'waveforms':    'waveform_mean',
}


class NWBFile:
    """Lazy reader wrapping a single .nwb file.

    Parses on first access; underlying HDF5 file stays open for the lifetime
    of this object.  Call .close() when done, or use as a context manager.
    """

    def __init__(self, path: str | Path, fields: dict = None):
        self._path = Path(path)
        self.fields = FieldMap(UNITS_SCHEMA, fields or NWB_DEFAULT)
        self._io = pynwb.NWBHDF5IO(str(self._path), mode='r', load_namespaces=True)
        self._nwb = self._io.read()

    def close(self):
        self._io.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def input_fields(self) -> list:
        """Source fields a map can bind to — the units table's column names."""
        units = self._nwb.units
        return [] if units is None else list(units.colnames)

    _MAX_CATEGORIES = 64   # beyond this a column is a measurement, not a label set

    def is_categorical(self, column: str) -> bool:
        """Whether a column takes few enough distinct values to be worth value-mapping.

        Judged by how many values it takes, not by dtype: shank ids and quality
        scores are integers, and stability is a bool, yet all three are label sets.
        """
        units = self._nwb.units
        if units is None or column not in units.colnames:
            return False
        head = np.asarray(units[column][:1])
        if head.ndim != 1 or head.dtype.kind == 'f':
            return False   # per-unit waveforms/acgs are 2-D; floats are measurements
        return len(set(units[column][:])) <= self._MAX_CATEGORIES

    def column_values(self, column: str) -> list:
        """The distinct values a units column holds — what a value map must translate."""
        units = self._nwb.units
        if units is None or column not in units.colnames:
            return []
        return sorted({str(v) for v in units[column][:]})

    # ── Spike trains ────────────────────────────────────────────────────

    @cached_property
    def spiketrains(self) -> list[np.ndarray]:
        units = self._nwb.units
        if units is None:
            return []
        column = self.fields.get('spike_times').column
        return [np.asarray(units[column][i], dtype=float) for i in range(len(units))]

    @cached_property
    def neuron_ids(self) -> np.ndarray:
        units = self._nwb.units
        if units is None:
            return np.array([], dtype=int)
        binding = self.fields.get('neuron_id')
        column = units[binding.column][:] if binding is not None else units.id[:]
        return np.asarray(column, dtype=int)

    @cached_property
    def neuron_type(self) -> np.ndarray:
        units = self._nwb.units
        if units is None:
            return np.array([], dtype=object)
        binding = self.fields.get('neuron_type')
        return np.array(binding.apply(units[binding.column][:]), dtype=object)

    @cached_property
    def positions(self) -> np.ndarray | None:
        """(n_units, 2) probe coordinates, or None when the map binds no position columns."""
        units = self._nwb.units
        binding = self.fields.get('position')
        if units is None or binding is None:
            return None
        xy = np.column_stack([np.asarray(units[c][:], dtype=float)
                              for c in binding.columns])
        if xy.shape[1] == 2 and np.array_equal(xy[:, 0], xy[:, 1]):
            return None   # duplicated column, not two axes
        return xy

    def _optional(self, name: str, dtype=None) -> np.ndarray | None:
        """One optional field's column as an array, or None when the map leaves it unbound."""
        units = self._nwb.units
        binding = self.fields.get(name)
        if units is None or binding is None:
            return None
        values = units[binding.column][:]
        if binding.value_map:
            values = binding.apply(values)
        return np.asarray(values, dtype=dtype)

    @cached_property
    def peak_channels(self) -> np.ndarray | None:
        return self._optional('peak_channel', dtype=int)

    @cached_property
    def shank_ids(self) -> np.ndarray | None:
        return self._optional('shank_id', dtype=int)

    @cached_property
    def cell_area(self) -> np.ndarray | None:
        return self._optional('cell_area', dtype=object)

    @cached_property
    def extra_columns(self) -> dict:
        """The map's wildcard columns, value-mapped, keyed by the names it gave them."""
        units = self._nwb.units
        if units is None:
            return {}
        return {name: np.asarray(binding.apply(units[binding.column][:]), dtype=object)
                for name, binding in self.fields.extra.items()
                if binding.column in units.colnames}

    @cached_property
    def waveforms(self) -> np.ndarray | None:
        return self._optional('waveforms')

    def _series_end(self, series) -> float | None:
        """When a TimeSeries covers the recording, the time its last sample lands on."""
        if series.timestamps is not None:
            return float(series.timestamps[-1])
        if series.rate:
            return float((series.starting_time or 0.0) + len(series.data) / series.rate)
        return None

    @cached_property
    def recorded_series(self) -> list:
        """Every continuous series under processing — an acquired signal spans the session."""
        return [ts for mod in self._nwb.processing.values()
                for interface in mod.data_interfaces.values()
                for ts in (getattr(interface, 'electrical_series', None) or {}).values()]

    @cached_property
    def declared_sampling_rate(self) -> float | None:
        """The rate the file states outright, when its writer recorded one."""
        rate = (self._nwb.scratch or {}).get('spike_sampling_rate')
        if rate is None:
            return None
        return float(np.asarray(getattr(rate, 'data', rate)).ravel()[0])

    @cached_property
    def sampling_rate(self) -> float | None:
        """The clock spike times were quantized onto, recovered from their smallest gap.

        NWB stores spike times in seconds and records no spike clock, but the times are
        quantized, so gaps pooled across units are multiples of one tick. Needs the pool:
        within one unit the smallest gap is the refractory period, not the tick.
        """
        pooled = np.sort(np.concatenate([st for st in self.spiketrains if len(st)]))
        gaps = np.diff(pooled)
        gaps = gaps[gaps > 0]
        if not len(gaps):
            return None
        return float(np.round(1.0 / gaps.min()))

    @cached_property
    def t_stop(self) -> float:
        """Session end: how long the signal ran, or the last spike when nothing was recorded."""
        ends = [t for t in map(self._series_end, self.recorded_series) if t is not None]
        spikes = [st.max() for st in self.spiketrains if len(st)]
        return float(max(ends + spikes, default=0.0))

    # ── Epochs ──────────────────────────────────────────────────────────

    @property
    def interval_tables(self) -> list:
        """Every TimeIntervals table this file holds — one theme each."""
        names = ['epochs'] if self._nwb.epochs is not None else []
        return names + list(self._nwb.intervals or [])

    @property
    def interval_labels(self) -> dict:
        """Each intervals table mapped to the distinct labels its rows carry, in first-seen order."""
        found = {}
        for name in self.interval_tables:
            df = self.intervals_df(name)
            if df is not None and not df.empty:
                found[name] = list(dict.fromkeys(df['label']))
        return found

    def intervals_df(self, table_name: str) -> pd.DataFrame | None:
        """Load a named intervals table → DataFrame with start/stop/label columns."""
        table = (self._nwb.epochs if table_name == 'epochs'
                 else (self._nwb.intervals or {}).get(table_name))
        if table is None:
            return None
        df = table.to_dataframe().rename(columns={'start_time': 'start',
                                                  'stop_time': 'stop'})
        if 'start' not in df.columns or 'stop' not in df.columns:
            return None
        df['label'] = self._labels_of(df, table_name)
        return df[['start', 'stop', 'label']].reset_index(drop=True)

    @staticmethod
    def _labels_of(df: pd.DataFrame, table_name: str):
        """Row labels from the table's own label column, falling back to the table name."""
        columns = [c for c in df.columns if c not in ('start', 'stop')]
        if not columns:
            return table_name
        column = 'tags' if 'tags' in columns else ('label' if 'label' in columns
                                                   else columns[0])
        # a tags column holds a list per row; the first tag is the label
        return df[column].apply(
            lambda x: str(x[0]) if isinstance(x, (list, tuple, np.ndarray)) and len(x)
            else str(x))
