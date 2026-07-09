"""NeuronsDataset: a collection of Neurons objects wrapped for CCG analysis.

Extracted from ms_connectivity.py to allow independent construction and reuse.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from typing import Union, Optional

from neuropy.analyses.utils import _san, Config, AnalysisDataset, Savable
from neuropy.core.neurons import Neurons
from neuropy.core.probe import ProbeGroup
from neuropy.core.epoch import Epoch


@dataclass(frozen=True)
class Key:
    """
    Indexing object for CCG analysis.

    key.session      # Top level
    key.epoch        # Mid level
    key.segment      # Finest time division
    key.conn_type    # Connection type (ref -> target)
    key.resolution   # Resolution (lowres, highres)

    [Dependencies]
    conn_type -> excitability
    """

    session: Optional[str] = None
    epoch: Optional[str] = None
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
        if self.epoch:
            parts.append(f'[{self.epoch}]')
        return ' '.join(parts) if parts else str(self)

    def nd(self) -> 'Key':
        return self.get('session')

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

    def __str__(self):
        parts = [
            f"sess_{self.session}" if self.session else "",
            f"epoch_{self.epoch}" if self.epoch else "",
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
                fields['epoch'] = part[6:]
            elif part.startswith('seg_'):
                fields['segment'] = int(part[4:])
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
    epochs (neuron behav_slice via paradigm only):
        None (Default)=All available
        [list]=try loading all epochs in list and silent skip missing labels
    """
    def __init__(
        self,
        name: str = "default",
        neuron_types: Union[list[str], str, None] = ['pyr', 'inter'],
        epochs: Union[list[str], str, None] = ['pre', 'maze', 'post', 're-maze'],
        themes: Union[list[str], str, None] = ['paradigm', 'brainstates', 'ripple'],
        ch_per_shank: Union[dict[Key, int], int, None] = 16,
    ):
        super().__init__()
        self.name = name
        self.neuron_types = _san(neuron_types)
        self.themes = _san(themes)
        self.epochs = _san(epochs)
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
    ):
        super().__init__()
        self.neurons = {}
        self.probe_info = {}
        self.themes = {}
        self._sessions = _san(sessions)
        self.conf = conf
        self._prep(self._sessions)

    @property
    def data(self):
        return self.neurons

    def _prep(self, sessions):
        """
        Init ND
        """
        for session in sessions:
            session_name = self._short_session_name(session)
            key = Key(session=session_name)
            self.neurons[key] = self._load_neurons(session)  # TODO: use neuron ids?
            self.probe_info[key] = self._load_probe_info(session)
            self.themes[key] = self._load_themes(session)

    def _short_session_name(self, session):
        """Get a printable session name ANIMAL_DayX """
        sess_name = session.filePrefix.parts[-1].split('_')[:2]
        sess_name = '_'.join(sess_name)
        return sess_name

    def _load_neurons(self, session):
        """Load and filter neurons."""
        n = session.neurons #TODO neurons_stable?
        n.metadata['intervals'] = np.array([[n.t_start, n.t_stop]]) 
        if self.conf.neuron_types is not None:
            n = n.get_neuron_type(self.conf.neuron_types)
        if self.conf.epochs is not None:
            all_epochs = session.paradigm.get_unique_labels()
            epochs = np.intersect1d(self.conf.epochs, all_epochs)
            if len(epochs) < len(all_epochs):
                n = n.behav_slice(behav_times=session.paradigm, 
                              labels=epochs)
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
        return [self._short_session_name(session) for session in self._sessions]

    def __str__(self):
        lines = [f"NeuronsDataset name={self.conf.name} sessions={len(self.neurons)}"]
        theme_store = getattr(self, 'themes', None) or {}
        for k, n in self.neurons.items():
            has_probe = self.probe_info.get(k) is not None
            per_sess = theme_store.get(k, {}) if isinstance(theme_store, dict) else {}
            theme_names = list(per_sess)
            n_iv = sum(ep.n_epochs for ep in per_sess.values() if isinstance(ep, Epoch))
            lines.append(
                f"  {k}: neurons={n.n_neurons} probe={has_probe} "
                f"themes={theme_names} intervals={n_iv}")
        return "\n".join(lines)

    # def _get_edge_times(self, key: Key, neurons: Neurons, session, theme_name):
    #     # TODO
    #     """
    #     Get the start and end of each segment. The edge timing are processed by epoch
    #     A segment is the smallest time period in the
    #     dataset where analysis will be performed (e.g. data used to calculate one CCG). There
    #     can be many overlapping segments within a dataset depending on configuration.

    #     Define segment edges of each neurons group
    #     see neurons.py:
    #         _edges_time_split   time_split
    #         _edges_time_window  time_windows
    #         _edges_spikecount   spikecount_split

    #     passing in key because we need to generate finer keys for objects in edge_times
    #     """

    #     dfs = []
    #     ivs = neurons.metadata['intervals']
    #     for i, e in enumerate(_san(self.conf.epochs, wrap_none=True)):
    #         try:
    #             theme = self.themes[key][theme_name]
    #         except:
    #             try:
    #                 theme = getattr(session, theme_name)
    #             except:
    #                 print(f"{key.session} doesn't have theme {theme_name}")
    #                 return None
                 
    #         if not e in theme.labels: #TODO
    #             print(f"{key.session} doesn't have epoch {e}")
    #             continue

    #         k = key.add(epoch=e)
    #         t_start, t_stop = theme.timing_by_label(e) if e \
    #                             else (neurons.t_start, neurons.t_stop)

    #         if self.conf.seg_spikecount is not None:
    #             # TODO spikecount segmentation code is not maintained
    #             neus = neurons.time_slice(t_start, t_stop)
    #             for i in range(neus.n_neurons):
    #                 k = key.add(epoch=e, ref_ind=i)
    #                 starts, stops = neus._edges_spikecount(
    #                     i=i, n=self.conf.seg_spikecount, discard_tail=False)
    #         elif self.conf.seg_stride is not None and self.conf.seg_len is not None:
    #             starts, stops = neurons._edges_time_window(
    #                 stride=self.conf.seg_stride,
    #                 seg_len=self.conf.seg_len,
    #                 t_start=t_start,
    #                 t_stop=t_stop)
    #         elif self.conf.n_segments is not None and self.conf.n_segments[
    #                 i] > 1:
    #             starts, stops = neurons._edges_time_split(
    #                 n_segments=self.conf.n_segments,
    #                 t_start=t_start,
    #                 t_stop=t_stop)
    #         else:
    #             starts, stops = np.array([t_start]), np.array([t_stop])
    #         """
    #         Calculate total/actual time lengths of each segment
    #         """
    #         edges = pd.DataFrame({
    #             "start": starts,
    #             "stop": stops,
    #             "key": [k.add(segment=i) for i in range(len(starts))],
    #             "label": [e + str(i) for i in range(len(starts))],
    #             "total_time_hours": (stops - starts) / 3600,
    #         })

    #         eths = []
    #         for row in edges.itertuples(index=False):
    #             start, stop, tth = row.start, row.stop, row.total_time_hours

    #             # find intervals that overlap the edge
    #             overlap_mask = (ivs[:, 1] > start) & (ivs[:, 0] < stop)
    #             overlapping_ivs = ivs[overlap_mask]

    #             # clip intervals to edge boundaries
    #             clipped_start = np.clip(overlapping_ivs[:, 0], start, stop)
    #             clipped_stop = np.clip(overlapping_ivs[:, 1], start, stop)

    #             # compute effective time in hours
    #             effective_hours = np.sum(clipped_stop - clipped_start) / 3600
    #             eths.append(min(effective_hours, tth))
    #         edges['effective_time_hours'] = np.array(eths)
    #         dfs.append(edges)
    #     return pd.concat(dfs, axis=0)


    # def _time_filter(self, session):
    ...
    #     if self.conf.sleep is not None:
    #         neurons = neurons.behav_slice(behav_times=session.brainstates,
    #                                       labels=self.conf.sleep.labels OR labels=None,
    #                                       discard=self.conf.sleep.discard,
    #                                       min_dur=self.conf.sleep.min_dur)


    # def _get_firing_rates_by_segment(self, edge_times: pd.DataFrame,
    #                                  neurons: Neurons):
    #     """
    #         Calculate and store segment-specific firing rates
    #     """
    #     x = np.zeros((edge_times.shape[0], neurons.n_neurons))
    #     for i, (t_start,
    #             t_end) in enumerate(zip(edge_times['start'],
    #                                     edge_times['stop'])):
    #         x[i] = neurons.time_slice(t_start, t_end).firing_rate
    #     return x

    # def frate_stats(self, key, alpha=0.05):
    #     """
    #     Generate a stats description of firing rates
    #     """
    #     from scipy.stats import describe, ttest_ind

    #     edge_times = self.edge_times[key]
    #     frates = self.segment_firing_rates[key]
    #     neuron_types = self.data[key].neuron_type
    #     labels = edge_times['label'].values
    #     stats_name = "firing rate"
    #     neuron_type_conf = self.conf.neuron_types

    #     for neuron_type in neuron_type_conf:
    #         print(f"{stats_name} stats {neuron_type}")
    #         print(
    #             f"segment | num | mean | iqr | min | max | variance | skew | kurt"
    #         )
    #         for i, (vi) in enumerate(edge_times.itertuples()):
    #             fr = frates[i][neuron_types == neuron_type]
    #             mean = np.mean(fr)
    #             iqr = np.percentile(fr, 75) - np.percentile(fr, 25)
    #             desc = describe(fr)
    #             print(
    #                 f"{str(i)+':'+str(vi.label):7} | {desc.nobs} | {mean:.2f} | {iqr:.2f} | {desc.minmax[0]:.2f} | {desc.minmax[1]:.2f} | {desc.variance:.2f} | {desc.skewness:.2f} | {desc.kurtosis:.2f}"
    #             )
    #         print("\n")

    #         print(f"Difference in mean {stats_name} P VALUES")
    #         printstr = ""
    #         decimal_places = int(-np.floor(np.log10(alpha)))

    #         # print p-value matrix
    #         printstr = f'{neuron_type:{decimal_places+3}}|'
    #         for i in range(len(labels)):
    #             printstr += f"{str(i):{decimal_places+3}}|"
    #         printstr += "\n"
    #         for i, (vi) in enumerate(frates):
    #             printstr += f"{str(i):{decimal_places+3}}|"
    #             for j, (vj) in enumerate(frates):
    #                 fri = frates[i][neuron_types == neuron_type]
    #                 frj = frates[j][neuron_types == neuron_type]
    #                 if j >= i:
    #                     continue
    #                 if len(fri) < 5:
    #                     printstr += f"{'-':{decimal_places}}|"
    #                 else:
    #                     p = ttest_ind(fri, frj, equal_var=True).pvalue
    #                     printstr += f"{p:.{decimal_places}f}{'*' if p<=alpha else ' '}|"
    #                     # Standard t-test,  check if mean firing rate changes per cell type
    #             printstr += f"{labels[i]}"
    #             printstr += "\n"
    #         print(printstr)


# class EpochSlicingConfig(Config):
#     """
#     Config to slice a behavioral epoch
#     """
#     def __init__(
#         self,
#         labels: Union[list[str], str, None] = None,
#         min_dur=0,
#         discard=False,
#     ):
#         super().__init__()
#         self.labels = _san(labels)
#         self.min_dur = min_dur
#         self.discard = discard

#     def __str__(self):
#         return self.__class__.__name__ + ': ' + '\n'.join(
#             [f"{key}={val}" for key, val in self.__dict__.items()])

