"""NeuronsDataset: a collection of Neurons objects wrapped for CCG analysis.

Extracted from ms_connectivity.py to allow independent construction and reuse.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union, Optional

from neuropy.analyses.utils import _san, Config, AnalysisDataset, Savable, JsonSavable
from neuropy.core.neurons import Neurons
from neuropy.core.probe import ProbeGroup
from neuropy.core.epoch import Epoch


@dataclass(eq=False)
class Key(JsonSavable):
    """
    Indexing object for CCG analysis.

    key.session      # Top level
    key.segment      # Finest time division
    key.conn_type    # Connection type (ref -> target)
    key.resolution   # Resolution (lowres, highres)

    [Dependencies]
    conn_type -> excitability
    """

    session: Optional[str] = None
    segment: Optional[int] = None
    excitability: Optional[str] = None
    conn_type: Optional[tuple[str, str]] = None
    resolution: Optional[str] = 'lowres'
    ref: Optional[int] = None
    tgt: Optional[int] = None

    def __eq__(self, other):
        try:
            return all(
                getattr(self, f) == getattr(other, f)
                for f in self.__dataclass_fields__)
        except:
            return False  # NOTE no type check, allows comparison with other 'key' classes

    def __hash__(self):
        # Not frozen (mutable), but still used as dict/set keys — hash the field tuple,
        # consistent with __eq__. Do not mutate a Key while it lives in a set/dict.
        return hash(tuple(getattr(self, f) for f in self.__dataclass_fields__))

    def serialize(self) -> dict:
        # Compact JSON-boundary form; _to_json inlines this for any nested Key.
        return {'__keystr__': self.json_str()}

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, state: dict) -> None:
        k = Key.from_str(state['__keystr__'])
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(k, f))

    def matches(self, **kwargs) -> bool:
        """Check if this key matches given criteria (for filtering)"""
        for k, v in kwargs.items():
            if isinstance(v, list):
                if getattr(self, k, None) not in v:
                    return False
            else:
                if v is not None and getattr(self, k, None) != v:
                    return False
        return True

    def get(self, *dimensions) -> 'Key':
        return Key(**{dim: getattr(self, dim, None) for dim in dimensions})

    def remove(self, *dimensions) -> 'Key':
        return Key(
            **{
                f: getattr(self, f)
                for f in self.__dataclass_fields__
                if f not in dimensions
            })

    def add(self, **kwargs) -> 'Key':
        for k, v in kwargs.items():
            assert getattr(self, k) is None, f"{k} is not None"
        return replace(self, **kwargs)

    def change(self, **kwargs) -> 'Key':
        return replace(self, **kwargs)

    def is_excitatory(self) -> bool:
        return self.excitability == 'E'

    @staticmethod
    def format_conn_type(ct) -> str:
        if ct is None:
            return "unknown"
        _map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}
        if isinstance(ct, (tuple, list)) and len(ct) == 2:
            a = _map.get(str(ct[0]).lower(), str(ct[0]).upper())
            b = _map.get(str(ct[1]).lower(), str(ct[1]).upper())
            return f"{a}→{b}"
        return str(ct)

    def type_label(self) -> str:
        parts = []
        if self.excitability:
            parts.append(self.excitability)
        if self.conn_type:
            parts.append(Key.format_conn_type(self.conn_type))
        return ' '.join(parts) if parts else str(self)

    def nd(self) -> 'Key':
        return self.get('session')
    
    def cd(self) -> 'Key':
        return self.get('session', 'resolution')

    def ptr(self) -> 'Key':
        """Pointer/significance key: resolution-invariant (pointers don't depend on resolution)."""
        return self.get('session', 'conn_type', 'excitability')

    def json_str(self) -> str:
        """Canonical JSON key for this Key (stable serialization at JSON boundary)."""
        return str(self)

    def ct_label(self) -> str:
        """Conn-type label string, e.g. 'pyr-inter'."""
        ct = self.conn_type
        if ct and len(ct) == 2:
            return f"{ct[0]}-{ct[1]}"
        return str(ct) if ct else ''

    @classmethod
    def pair(cls, session: str, ref: int, tgt: int) -> 'Key':
        """Pair identity for cross-group matching (session + ref + tgt)."""
        return cls(session=str(session), ref=int(ref), tgt=int(tgt))

    def pair_sort_key(self) -> tuple:
        """Stable tuple for sort order (session, ref, tgt)."""
        return (str(self.session or ''), int(self.ref or 0), int(self.tgt or 0))

    def __lt__(self, other) -> bool:
        if not isinstance(other, Key):
            return NotImplemented
        return self.pair_sort_key() < other.pair_sort_key()

    def __str__(self):
        parts = [
            f"sess_{self.session}" if self.session else "",
            f"seg_{self.segment}" if self.segment is not None else "",
            f"ex_{self.excitability}" if self.excitability else "",
            f"type_{self.conn_type[0]}-{self.conn_type[1]}" if self.conn_type else "",
            f"res_{self.resolution}" if self.resolution != 'lowres' else "",
            f"ref_{self.ref}" if self.ref is not None else "",
            f"tgt_{self.tgt}" if self.tgt is not None else "",
        ]
        return ".".join(filter(None, parts)) or "root"

    @classmethod
    def from_str(cls, s: str) -> 'Key':
        """Inverse of __str__. Parses the canonical dot-separated form back to a Key."""
        if s == 'root':
            return cls()
        fields: dict = {}
        for part in s.split('.'):
            if part.startswith('sess_'):
                fields['session'] = part[5:]
            elif part.startswith('epoch_'):
                continue   # legacy custom-window keys: epoch dissolved into dim0 segments
            elif part.startswith('seg_'):
                fields['segment'] = part[4:]
            elif part.startswith('ex_'):
                fields['excitability'] = part[3:]
            elif part.startswith('type_'):
                a, b = part[5:].split('-', 1)
                fields['conn_type'] = (a, b)
            elif part.startswith('res_'):
                fields['resolution'] = part[4:]
            elif part.startswith('ref_'):
                fields['ref'] = int(part[4:])
            elif part.startswith('tgt_'):
                fields['tgt'] = int(part[4:])
        return cls(**fields)


class NeuronsDatasetConfig(Config):
    """NeuronsDataset build config.

    themes:
        None (Default)=All available
        []=No themes
    """

    # "init param": ("kind", choices)  — what a builder UI exposes; defaults come from __init__
    _options = {
        'neuron_types': ('multi', ['pyr', 'inter']),
        'themes':       ('multi', None),   # None choices -> filled by the source scan
        'ch_per_shank': ('int', None),
    }

    def __init__(
        self,
        name: str = "default",
        neuron_types: Union[list[str], str, None] = ['pyr', 'inter'],
        themes: Union[list[str], str, None] = None,
        ch_per_shank: Union[dict[Key, int], int, None] = 16,
    ):
        super().__init__()
        self.name = name
        self.neuron_types = _san(neuron_types)
        self.themes = _san(themes)
        self.ch_per_shank = ch_per_shank

    def __str__(self):
        return self.__class__.__name__ + ': ' + '\n'.join(
            [f"{key}={val}" for key, val in self.__dict__.items()])


class NeuronsDataset(AnalysisDataset):
    """
    A collection of neurons wrapped for analysis. Coupled with NeuronsDatasetConfig.

    neurons: dict[Key, Neurons]
    probe_info: dict[Key, ProbeGroup]  # ProbeGroup per session Key
    themes: dict[Key, Epoch]  # theme name # Epoch per session Key
    _sessions: list of ProcessData accessed internally by ND
    conf:
    """

    # class NeuronSegmentStats(Savable):
    #     # per-segment, per-neuron data
    #     firing_rates: np.ndarray

    #     def __str__(self):
    #         fr = getattr(self, 'firing_rates', None)
    #         if fr is None:
    #             return "NeuronSegmentStats(empty)"
    #         return f"NeuronSegmentStats shape={np.asarray(fr).shape}"

    neurons: dict[Neurons]
    probe_info: dict
    themes: dict
    _sessions: dict
    conf: NeuronsDatasetConfig

    def __init__(
        self,
        sessions,
        conf: NeuronsDatasetConfig,
        naming=None,
    ):
        super().__init__()
        self._neurons = {}
        self.probe_info = {}
        self.themes = {}
        self._slice_cache = {}   # TRANSIENT memo, never persisted
        self._sessions = _san(sessions)
        self.conf = conf
        # naming takes a path, the same argument NWBDataset gives it, so one
        # dataset's rule reads the same either way it is reached.
        naming = naming or (lambda path: Path(path).name)
        for session in self._sessions:
            session.session_name = naming(session.filePrefix)
        self._prep(self._sessions)

    def neurons_for(self, key: Key) -> Neurons:
        return self._neurons[key.nd()]

    def get_themes(self, key: Key) -> dict:
        """Sole provider of a session's non-empty theme Epochs (built once in _prep)."""
        return {name: ep for name, ep in (self.themes.get(key.nd()) or {}).items()
                if ep is not None and ep.n_epochs > 0}

    def get_themes_any(self) -> dict:
        """Union of theme Epochs across all sessions (all-session mode); concatenated per name."""
        frames: dict[str, list] = {}
        for k in self.session_keys():
            for name, ep in self.get_themes(k).items():
                frames.setdefault(name, []).append(ep.to_dataframe()[['start', 'stop', 'label']])
        return {name: Epoch(pd.concat(dfs, ignore_index=True))
                for name, dfs in frames.items() if dfs}

    def session_bounds(self, key) -> tuple:
        """Session recording (t_start, t_stop) in seconds."""
        n = self.neurons_for(key)
        return float(n.t_start), float(n.t_stop)

    def resolve_time(self, key, raw) -> float:
        """Sole start/end resolver: 'start'/'end' → session bounds; else numeric seconds."""
        t_start, t_stop = self.session_bounds(key)
        return {'start': t_start, 'end': t_stop}.get(raw, None) \
            if isinstance(raw, str) else float(raw)

    def _theme_intervals(self, key, theme, labels, t0, t1) -> tuple:
        """Active intervals for one theme's label whitelist within [t0,t1]. Returns (intervals, dur)."""
        from neuropy.analyses.epoch_filter import EpochFilter
        active = {str(x) for x in (labels or [])}  # whitelist: keep only these
        bounds = []
        ep = self.get_themes(key).get(theme) if theme != 'segments' else None
        raw_lbls = [str(lb).strip() for lb in (ep.labels if ep is not None else [])]
        if ep is not None:
            bounds = [(float(s), float(e), str(lb).strip())
                      for s, e, lb in zip(ep.starts, ep.stops, ep.labels)]
            if len({lb for _, _, lb in bounds if lb}) <= 1:  # single-label theme → theme name
                bounds = [(s, e, theme) for s, e, _ in bounds]
        print(f"[dbg:_theme_intervals] sess={key.session} theme={theme!r} "
              f"whitelist={sorted(active)} raw_lbls={sorted(set(raw_lbls))} "
              f"t0={t0:.1f} t1={t1:.1f} ep={'yes' if ep is not None else 'NO'}",
              flush=True)
        before = len(bounds)
        if active:
            bounds = [b for b in bounds if b[2] in active]
        if not bounds:
            print(f"[dbg:_theme_intervals] EMPTY label-filter "
                  f"before={before} whitelist={sorted(active)}", flush=True)
            return [], 0.0   # no label match → skip session
        result = EpochFilter(bounds).filter(active or {lb for _, _, lb in bounds}, t0, t1)
        print(f"[dbg:_theme_intervals] EpochFilter→ type={type(result).__name__} "
              f"val={result!r}", flush=True)
        if result is False or result[0] is None:
            print(f"[dbg:_theme_intervals] DROP None/False as empty", flush=True)
            return [], 0.0
        return result

    def resolve_intervals(self, key, t0, t1, filter_state) -> tuple:
        """Active intervals = intersection over each theme's label whitelist. Returns (intervals, active_dur)."""
        from neuropy.core.intervals import IntervalOp
        t0 = self.resolve_time(key, t0)
        t1 = self.resolve_time(key, t1)
        if t1 <= t0:
            return None, None
        acc = None
        for th in filter_state:
            iv, _ = self._theme_intervals(key, th.get('name', 'segments'), th.get('labels'), t0, t1)
            if not iv:
                return [], 0.0 
            acc = iv if acc is None else IntervalOp.intersect(acc, iv)
            if not acc:
                return [], 0.0
        return acc, IntervalOp.duration(acc)

    def sliced_neurons_for(self, src: 'CCGSourceConfig'):
        """Neurons windowed to a segment's active intervals; cached per (session, intervals)
        so lowres/highres tasks reuse one slice. Returns (neurons_slice, active_dur) or None."""
        intervals, active_dur = self.resolve_intervals(src.key, src.t0, src.t1, src.filter_state)
        if intervals is None or not intervals:
            return None
        ck = (src.key.nd(), tuple((float(a), float(b)) for a, b in intervals))
        sl = self._slice_cache.get(ck)
        if sl is None:
            sl = self.neurons_for(src.key).time_multislices(*zip(*intervals))
            self._slice_cache[ck] = sl
        return sl, active_dur

    def clear_slice_cache(self):
        """Drop the transient windowed-Neurons memo (call when a compute batch finishes)."""
        self._slice_cache.clear()

    def session_keys(self) -> list:
        return list(self._neurons.keys())

    def _prep(self, sessions):
        """
        Init ND
        """
        for session in sessions:
            session_name = session.session_name
            key = Key(session=session_name)
            self._neurons[key] = self._load_neurons(session)  # TODO: use neuron ids?
            self.probe_info[key] = self._load_probe_info(session)
            self.themes[key] = self._load_themes(session)

    def _load_neurons(self, session):
        """Load and filter neurons."""
        n = session.neurons #TODO neurons_stable?
        n.metadata['intervals'] = np.array([[n.t_start, n.t_stop]]) 
        if self.conf.neuron_types is not None:
            n = n.get_neuron_type(self.conf.neuron_types)
        return n

    # TODO
    def _load_probe_info(self, session):
        """Load ProbeGroup info.
        TODO Probe-related info is incomplete and needs to be gathered from recinfo and by user input"""
        pfiles = sorted(session.basepath.glob('*.probegroup.npy'))
        if not pfiles:
            return None
        pg = ProbeGroup.from_file(pfiles[0])
        
        recinfo = getattr(session, 'recinfo', None)
        skipped = getattr(recinfo, 'skipped_channels', None)
        if skipped is not None and len(skipped) > 0:
            mask = pg._data['channel_id'].isin(np.asarray(skipped))
            pg._data.loc[mask, 'connected'] = False

        cps = self.conf.ch_per_shank
        pg.metadata = pg.metadata or {}
        if isinstance(self.conf.ch_per_shank, dict):
            pg.metadata['ch_per_shank'] = int(cps.get(session.key))
        else:
            pg.metadata['ch_per_shank'] = int(cps)
        return pg
    
    def _load_themes(self, session):
        """Load designated or discover Epochs from session as themes."""
        themes = {}
        if self.conf.themes is not None:
            names = self.conf.themes
        else:
            names = [_ for _ in dir(session) if isinstance(getattr(session, _, None), Epoch)] # discover themes
        for name in names:
            themes[name] = None
            ep = getattr(session, name, None)
            if isinstance(ep, Epoch) and ep.n_epochs > 0:
                themes[name] = ep
        return themes

    @property
    def session_names(self):
        return [session.session_name for session in self._sessions]

    def __str__(self):
        lines = [f"NeuronsDataset name={self.conf.name} sessions={len(self._neurons)}"]
        theme_store = getattr(self, 'themes', None) or {}
        for k, n in self._neurons.items():
            has_probe = self.probe_info.get(k) is not None
            per_sess = theme_store.get(k, {}) if isinstance(theme_store, dict) else {}
            theme_names = list(per_sess)
            n_iv = sum(ep.n_epochs for ep in per_sess.values() if isinstance(ep, Epoch))
            lines.append(
                f"  {k}: neurons={n.n_neurons} probe={has_probe} "
                f"themes={theme_names} intervals={n_iv}")
        return "\n".join(lines)
