"""Reading the Hiro-lab MATLAB export into NeuroPy primitives.

The export is organised by *data type*, not by session: ``sleep-spikes.mat`` holds
every sleep session's units. Reading one session therefore means opening a large
file and pulling one group out of it, which is why conversion is a one-time pass
rather than something done on demand.

Two MATLAB vintages are in play — most files are v7.3 (HDF5) but the wake spikes
are v7, which ``scipy.io`` must read whole. Both are hidden behind ``_sessions``.

Timestamps are microseconds throughout and are converted to seconds here, so
nothing downstream has to remember the unit.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io

from neuropy.core.epoch import Epoch
from neuropy.core.neurons import Neurons

US = 1e6

# behavior.list third row; the codes are documented in DataDetails.pdf.
BRAIN_STATES = {1: 'nrem', 2: 'rem', 3: 'quiet', 4: 'active'}

# Wake sessions run PRE / track / POST; a sleep session is one continuous block.
PARADIGM = ['pre', 'maze', 'post']

CONDITIONS = {'sleep': 'sleep', 'wake': 'wake_new'}


def _is_hdf5(path: Path) -> bool:
    """v7.3 hides the HDF5 signature behind MATLAB's 128-byte text header."""
    return h5py.is_hdf5(path)


def _mat_str(h, ref) -> str:
    return ''.join(chr(c) for c in np.asarray(h[ref][()]).ravel())


class HiroFile:
    """One ``<cond>-<kind>.mat``, opened whichever way its vintage requires."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.hdf5 = _is_hdf5(self.path)
        self._h = self._v7 = None
        if self.hdf5:
            self._h = h5py.File(self.path, 'r')
            self._root = next(k for k in self._h if k != '#refs#')
        else:
            # v7 cannot be read lazily, so the whole file lands in memory at once.
            self._v7 = scipy.io.loadmat(self.path, struct_as_record=False,
                                        squeeze_me=True)
            self._root = next(k for k in self._v7 if not k.startswith('__'))

    def close(self):
        if self._h is not None:
            self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def sessions(self) -> list[str]:
        if self.hdf5:
            return list(self._h[self._root].keys())
        return list(getattr(self._v7[self._root], '_fieldnames', []))

    def session(self, name: str):
        if self.hdf5:
            return self._h[self._root][name]
        return getattr(self._v7[self._root], name)

    def deref(self, refs) -> list:
        """MATLAB object arrays hold references; resolve them to arrays."""
        return [self._h[r][()] for r in np.asarray(refs).ravel()]


class _Absent:
    """Stand-in for a file this export does not ship; holds no sessions."""

    sessions: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class HiroSession:
    """One session, read across the per-type files it appears in."""

    def __init__(self, root, condition: str, name: str):
        self.root = Path(root)
        self.condition = condition
        self.name = name

    def _path(self, kind: str) -> Path:
        return self.root / CONDITIONS[self.condition] / f'{self.condition}-{kind}.mat'

    def _open(self, kind: str) -> 'HiroFile | _Absent':
        """Missing kinds answer as empty rather than None: the export is partial
        (sleep has no position, one animal has no basics) and every caller would
        otherwise repeat the same guard."""
        p = self._path(kind)
        return HiroFile(p) if p.is_file() else _Absent()

    @property
    def subject(self) -> str:
        """RoySleep1 -> Roy; the condition and index follow the animal's name."""
        for marker in ('Sleep', 'Rest', 'Maze'):
            if marker in self.name:
                return self.name.split(marker)[0]
        return self.name

    def basics(self) -> dict:
        """Sampling rates, channel counts and shank layout; ``{}`` when absent."""
        with self._open('basics') as f:
            if self.name not in f.sessions:
                return {}
            s = f.session(self.name)
            out = {k: float(np.asarray(s[k][()]).ravel()[0])
                   for k in ('SampleRate', 'lfpSampleRate', 'nChannels',
                             'posSampleRate') if k in s}
            if 'SpkGrps' in s:
                out['shank_channels'] = [c.ravel().astype(int)
                                         for c in f.deref(s['SpkGrps']['Channels'][()])]
            if 'Ch' in s and 'CA1Shanks' in s['Ch']:
                out['ca1_shanks'] = np.asarray(
                    s['Ch']['CA1Shanks'][()]).ravel().astype(int)
            return out

    @lru_cache(maxsize=1)
    def t_origin(self) -> float:
        """Seconds to subtract so this session starts near zero.

        The export timestamps everything against one campaign-wide clock, so
        spikes and epochs are rebased together against the declared session start.
        """
        candidates = []
        with self._open('behavior') as f:
            if self.name in f.sessions:
                t = np.asarray(f.session(self.name)['time'][()], dtype=float)
                if t.size:
                    candidates.append(float(t.ravel().min()))
        with self._open('spikes') as f:
            if self.name in f.sessions:
                s = f.session(self.name)
                firsts = ([np.asarray(t).ravel()[0] for t in f.deref(s['time'][()])
                           if np.asarray(t).size] if f.hdf5 else
                          [np.asarray(u.time).ravel()[0] for u in np.atleast_1d(s)
                           if np.asarray(u.time).size])
                if firsts:
                    candidates.append(float(min(firsts)))
        # Earliest of the two: RamboSleep1's spikes precede its behavior window by
        # 6 h, and anchoring on behavior alone would push them negative.
        return min(candidates) / US if candidates else 0.0

    def sampling_rate(self) -> float:
        """This session's rate, or the export's most common one when it declares none."""
        own = self.basics().get('SampleRate')
        return float(own) if own else common_sampling_rate(self.root)

    def neurons(self) -> Neurons | None:
        """Units with cell class derived from quality, per DataDetails.pdf.

        Every unit is kept. ``quality`` and ``is_stable`` travel with them so the
        pyr/inter gate can be revisited without re-reading the MATLAB export.
        """
        with self._open('spikes') as f:
            if self.name not in f.sessions:
                return None
            s = f.session(self.name)
            if f.hdf5:
                trains = [np.asarray(t).ravel() / US for t in f.deref(s['time'][()])]
                quality = np.array([np.asarray(q).ravel()[0]
                                    for q in f.deref(s['quality'][()])])
                stable = np.array([bool(np.all(np.asarray(v)))
                                   for v in f.deref(s['isStable'][()])])
                ids = np.array([np.asarray(i).ravel() for i in f.deref(s['id'][()])])
            else:
                units = np.atleast_1d(s)
                trains = [np.asarray(u.time).ravel() / US for u in units]
                quality = np.array([np.asarray(u.quality).ravel()[0] for u in units])
                stable = np.array([bool(np.all(np.asarray(u.isStable)))
                                   for u in units])
                ids = np.array([np.asarray(u.id).ravel() for u in units])

        neuron_type = np.where(quality < 4, 'pyr',
                               np.where(quality == 8, 'inter', 'unclassified'))
        # Timestamps are absolute across the whole campaign (Ted reaches 18 days),
        # so each session is rebased to its own start.
        t0 = self.t_origin()
        trains = [t - t0 for t in trains]
        t_start = min((t[0] for t in trains if len(t)), default=0.0)
        t_stop = max((t[-1] for t in trains if len(t)), default=0.0)
        return Neurons(
            spiketrains=np.array(trains, dtype=object),
            t_start=t_start, t_stop=t_stop,
            sampling_rate=self.sampling_rate(),
            neuron_ids=np.arange(len(trains)),
            neuron_type=neuron_type,
            shank_ids=ids[:, 0].astype(int) if ids.size else None,
            clu_q=quality,
            metadata={'is_stable': stable,
                      'cluster_id': ids[:, 1].astype(int) if ids.size else None,
                      't_origin': t0})   # seconds removed; add back for campaign time

    def epochs(self) -> dict[str, Epoch]:
        """``paradigm`` (PRE/track/POST) and ``brainstates``, both in seconds."""
        out = {}
        t0 = self.t_origin()
        with self._open('behavior') as f:
            if self.name not in f.sessions:
                return out
            s = f.session(self.name)
            time = np.asarray(s['time'][()], dtype=float)
            # (2, n): row 0 starts, row 1 stops — one column per paradigm block.
            if time.size:
                starts, stops = time[0] / US - t0, time[1] / US - t0
                # KevinMaze1 carries a 4th block the documentation does not name,
                # so anything past PRE/track/POST is numbered rather than guessed.
                labels = ([f'block{i}' if i >= len(PARADIGM) else PARADIGM[i]
                           for i in range(len(starts))] if len(starts) > 1
                          else [self.condition])
                out['paradigm'] = Epoch(pd.DataFrame(
                    {'start': starts, 'stop': stops, 'label': labels}))
            lst = np.asarray(s['list'][()], dtype=float)
            if lst.size:
                lst = lst if lst.shape[0] == 3 else lst.T
                out['brainstates'] = Epoch(pd.DataFrame(
                    {'start': lst[0] / US - t0, 'stop': lst[1] / US - t0,
                     'label': [BRAIN_STATES.get(int(c), str(int(c)))
                               for c in lst[2]]}))
        return out

    def events(self) -> dict[str, Epoch]:
        """Detected ripples and spindles as intervals, with their peak times."""
        out = {}
        t0 = self.t_origin()
        for kind in ('ripple', 'spindle'):
            with self._open(kind) as f:
                if self.name not in f.sessions:
                    continue
                s = f.session(self.name)
                # Spindles are nested one level deeper, under the region detected on.
                if 'time' not in s:
                    region = next((k for k in s.keys() if 'time' in s[k]), None)
                    if region is None:
                        continue
                    s = s[region]
                t = np.asarray(s['time'][()], dtype=float)
                t = t if t.shape[0] == 2 else t.T
                peak = np.asarray(s['peakTime'][()], dtype=float).ravel()
                out[kind] = Epoch(pd.DataFrame(
                    {'start': t[0] / US - t0, 'stop': t[1] / US - t0,
                     'label': kind, 'peak_time': peak / US - t0}))
        return out

    def timeseries(self) -> dict[str, dict]:
        """Position and speed; sampled on different clocks, so kept separate."""
        out = {}
        with self._open('position') as f:
            if self.name in f.sessions:
                s = f.session(self.name)
                t = np.asarray(s['t'][()], dtype=float).ravel() / US
                xy = np.column_stack([np.asarray(s['x'][()], dtype=float).ravel(),
                                      np.asarray(s['y'][()], dtype=float).ravel()])
                out['position'] = {'t': t, 'data': xy, 'unit': 'cm',
                                   'description': 'tracked position (x, y)'}
        with self._open('speed') as f:
            if self.name in f.sessions:
                s = f.session(self.name)
                out['speed'] = {
                    't': np.asarray(s['t'][()], dtype=float).ravel() / US,
                    'data': np.asarray(s['v'][()], dtype=float).ravel(),
                    'unit': 'cm/s', 'description': 'running speed'}
        return out

    def spectrograms(self) -> dict[str, dict]:
        """Precomputed time-frequency power — the export ships no raw LFP trace."""
        out = {}
        for kind, name in (('pfcEeg', 'pfc_spectrogram'),
                           ('spectrum', 'hpc_spectrogram')):
            with self._open(kind) as f:
                if self.name not in f.sessions:
                    continue
                s = f.session(self.name)
                out[name] = {
                    'power': np.asarray(s['Pxx'][()], dtype=float),
                    'frequencies': np.asarray(s['freq'][()], dtype=float).ravel(),
                    't': np.asarray(s['time'][()], dtype=float).ravel() / US,
                    'description': f'{kind} power spectrogram'}
        return out


@lru_cache(maxsize=4)
def common_sampling_rate(root) -> float:
    """The rate most sessions in this export were recorded at."""
    rates = []
    for cond, sub in CONDITIONS.items():
        p = Path(root) / sub / f'{cond}-basics.mat'
        if not p.is_file():
            continue
        with HiroFile(p) as f:
            for name in f.sessions:
                s = f.session(name)
                if 'SampleRate' in s:
                    rates.append(float(np.asarray(s['SampleRate'][()]).ravel()[0]))
    return float(Counter(rates).most_common(1)[0][0]) if rates else 30000.0


def sessions(root, condition: str = None) -> list[HiroSession]:
    """Every session the export holds, or only one condition's when named.

    Sessions are collected from the union of the per-type files, because the
    export is not complete: one animal has spikes and behaviour but no basics.
    """
    root = Path(root)
    out = []
    for cond in ([condition] if condition else CONDITIONS):
        found: list[str] = []
        for kind in ('spikes', 'behavior', 'basics'):
            p = root / CONDITIONS[cond] / f'{cond}-{kind}.mat'
            if p.is_file():
                with HiroFile(p) as f:
                    found += [s for s in f.sessions if s not in found]
        out += [HiroSession(root, cond, name) for name in sorted(found)]
    return out
