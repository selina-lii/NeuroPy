"""NWB writer — NeuroPy primitives out to a .nwb file.

The counterpart to ``nwbio.NWBFile``, which reads. Everything here takes objects
that are already in NeuroPy's vocabulary (``Neurons``, ``Epoch``, ``ProbeGroup``),
so a dataset's own parsing lives in its reader and never in this module. Columns
are named to match ``NWB_DEFAULT`` so a file written here reads back with no
field map.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.tz import tzlocal

import pynwb
from hdmf.common import VectorData
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import DecompositionSeries

# Written for every unit beyond spike_times; a reader can then map or ignore them.
_UNIT_COLUMNS = {
    'neuron_type':  'cell class, "pyr" or "inter"',
    'shank_id':     'probe shank the unit was clustered on',
    'peak_channel': 'channel carrying the largest waveform',
    'quality':      "cluster quality as scored by the source dataset",
}


def _session_start(metadata: dict) -> datetime:
    """A tz-aware start time — NWB refuses a naive one."""
    start = (metadata or {}).get('session_start_time')
    if start is None:
        return datetime.fromtimestamp(0, tzlocal())
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    return start if start.tzinfo else start.replace(tzinfo=tzlocal())


def _new_file(session: str, metadata: dict) -> pynwb.NWBFile:
    md = dict(metadata or {})
    return pynwb.NWBFile(
        session_description=md.get('session_description', session),
        identifier=md.get('identifier', session),
        session_start_time=_session_start(md),
        session_id=session,
        experimenter=md.get('experimenter'),
        lab=md.get('lab'),
        institution=md.get('institution'),
        experiment_description=md.get('experiment_description'),
        subject=pynwb.file.Subject(subject_id=md['subject'])
        if md.get('subject') else None)


def _add_electrodes(nwb: pynwb.NWBFile, probegroup, metadata: dict) -> list:
    """One electrode row per contact; returns the row indices in order.

    A shank becomes an electrode group, which is what NWB uses to express the
    same grouping ``shank_id`` carries on the unit table.
    """
    device = nwb.create_device(name=(metadata or {}).get('device', 'probe'))
    df = probegroup.to_dataframe() if probegroup is not None else pd.DataFrame()
    if df.empty:
        return []
    regions = (metadata or {}).get('channel_regions', {})
    groups = {}
    for shank in sorted(df['shank_id'].unique()):
        name = f'shank{int(shank)}'
        groups[shank] = nwb.create_electrode_group(
            name=name, description=f'shank {int(shank)}', device=device,
            location=str(regions.get(int(shank), 'unknown')))
    for col in ('x', 'y'):
        if col not in df:
            df[col] = np.nan
    for _, row in df.iterrows():
        nwb.add_electrode(group=groups[row['shank_id']],
                          location=str(regions.get(int(row['shank_id']), 'unknown')),
                          x=float(row['x']), y=float(row['y']), z=0.0,
                          imp=float('nan'), filtering='unknown')
    return list(range(len(df)))


def _add_units(nwb: pynwb.NWBFile, neurons) -> None:
    """Units table, one row per spiketrain, in the order ``neurons`` holds them."""
    n = len(neurons.spiketrains)
    cols = {name: getattr(neurons, attr, None) for name, attr in
            (('neuron_type', 'neuron_type'), ('shank_id', 'shank_ids'),
             ('peak_channel', 'peak_channels'), ('quality', 'clu_q'))}
    extra = getattr(neurons, 'metadata', None) or {}
    cols.update({k: v for k, v in extra.items()
                 if isinstance(v, (list, np.ndarray)) and len(v) == n})
    present = {k: np.asarray(v) for k, v in cols.items() if v is not None}
    for name in present:
        nwb.add_unit_column(name=name, description=_UNIT_COLUMNS.get(
            name, f'{name} (from the source dataset)'))

    ids = getattr(neurons, 'neuron_ids', None)
    waveforms = getattr(neurons, 'waveforms', None)
    for i, train in enumerate(neurons.spiketrains):
        row = {k: v[i] for k, v in present.items()}
        # NWB stores strings, not numpy scalars, and bools survive the round trip
        # only as ints on some backends.
        row = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        if waveforms is not None and len(waveforms) > i:
            row['waveform_mean'] = np.asarray(waveforms[i], dtype=float)
        nwb.add_unit(spike_times=np.asarray(train, dtype=float),
                     id=int(ids[i]) if ids is not None else int(i), **row)


def _add_epochs(nwb: pynwb.NWBFile, name: str, epoch) -> None:
    """One ``TimeIntervals`` table per theme, named as the reader named it.

    Built column-at-a-time: ``add_row`` rebuilds the table's index on every call,
    which a ripple table of tens of thousands of intervals cannot afford.
    """
    df = epoch.to_dataframe() if hasattr(epoch, 'to_dataframe') else pd.DataFrame(epoch)
    if df.empty:
        return
    # 'label' is not one of TimeIntervals' predefined columns, so it is declared
    # alongside whatever else the reader attached.
    extra = ['label'] + [c for c in df.columns if c not in ('start', 'stop', 'label')]
    columns = [VectorData(name='start_time', description='interval start',
                          data=df['start'].to_numpy(dtype=float)),
               VectorData(name='stop_time', description='interval stop',
                          data=df['stop'].to_numpy(dtype=float))]
    for c in extra:
        values = (df[c].astype(str).tolist() if c == 'label'
                  else df[c].to_numpy())
        columns.append(VectorData(name=c, description=c, data=values))
    nwb.add_time_intervals(pynwb.epoch.TimeIntervals(
        name=name, description=f'{name} intervals', columns=columns))


def _add_timeseries(nwb: pynwb.NWBFile, name: str, ts: dict) -> None:
    """A behavioural signal: ``{'t': …, 'data': …, 'unit': …}``.

    Irregularly sampled series keep their own timestamps rather than a rate, so
    a camera clock that drifts stays truthful.
    """
    data = np.asarray(ts['data'], dtype=float)
    t = np.asarray(ts['t'], dtype=float)
    nwb.add_acquisition(pynwb.TimeSeries(
        name=name, data=data, timestamps=t,
        unit=ts.get('unit', 'unknown'), description=ts.get('description', name)))


def _add_lfp(nwb: pynwb.NWBFile, lfp: dict, n_electrodes: int) -> None:
    """Raw LFP as an ``ElectricalSeries`` on an electrode region."""
    data = np.asarray(lfp['data'])
    region = nwb.create_electrode_table_region(
        region=list(lfp.get('channels', range(min(data.shape[1], n_electrodes)))),
        description='LFP channels')
    kw = ({'timestamps': np.asarray(lfp['t'], dtype=float)} if 't' in lfp
          else {'starting_time': float(lfp.get('starting_time', 0.0)),
                'rate': float(lfp['rate'])})
    nwb.add_acquisition(ElectricalSeries(
        name=lfp.get('name', 'LFP'), data=data, electrodes=region,
        conversion=float(lfp.get('conversion', 1.0)), **kw))


def _add_spectrogram(nwb: pynwb.NWBFile, name: str, spec: dict) -> None:
    """A precomputed spectrogram — ``DecompositionSeries``, not ``ElectricalSeries``.

    Some datasets ship only the time-frequency decomposition and no raw trace;
    NWB has a type for exactly that, so it is not forced into a signal series.
    """
    # (time, channel, frequency) is what DecompositionSeries expects.
    power = np.asarray(spec['power'], dtype=float)
    if power.ndim == 2:
        power = power.T[:, np.newaxis, :]
    ds = DecompositionSeries(
        name=name, data=power, metric=spec.get('metric', 'power'),
        timestamps=np.asarray(spec['t'], dtype=float),
        description=spec.get('description', f'{name} spectrogram'))
    # One call per band: add_band takes a single (2,) limit, not the whole grid.
    for hz in np.asarray(spec['frequencies'], dtype=float):
        ds.add_band(band_name=f'{hz:g}Hz', band_limits=np.array([hz, hz]))
    nwb.add_acquisition(ds)


def write_nwb(path, session: str, neurons=None, epochs: dict = None,
              probegroup=None, timeseries: dict = None, lfp: dict = None,
              spectrograms: dict = None, metadata: dict = None) -> Path:
    """Write one session to *path*; returns the path written.

    Only what is supplied is written, so a dataset lacking position or LFP simply
    passes fewer arguments rather than fabricating empties.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nwb = _new_file(session, metadata)
    n_elec = len(_add_electrodes(nwb, probegroup, metadata))
    if neurons is not None:
        _add_units(nwb, neurons)
        # Stated outright: recovering it from spike gaps misreads a session whose
        # spikes are sparse or irregular.
        rate = float(getattr(neurons, 'sampling_rate', 0) or 0)
        if rate:
            nwb.add_scratch(np.array([rate], dtype=float),
                            name='spike_sampling_rate',
                            description='clock the spike times quantize onto (Hz)')
    for name, ep in (epochs or {}).items():
        _add_epochs(nwb, name, ep)
    for name, ts in (timeseries or {}).items():
        _add_timeseries(nwb, name, ts)
    if lfp is not None:
        _add_lfp(nwb, lfp, n_elec)
    for name, spec in (spectrograms or {}).items():
        _add_spectrogram(nwb, name, spec)
    with pynwb.NWBHDF5IO(str(path), mode='w') as io:
        io.write(nwb)
    return path
