"""NeuronsDataset: a collection of Neurons objects wrapped for CCG analysis.

Extracted from ms_connectivity.py to allow independent construction and reuse.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from typing import Union, Optional

from neuropy.io import NeuroscopeIO
from neuropy.analyses.utils import _san, Config, AnalysisDataset, Savable
from neuropy.core.neurons import Neurons


# ──────────────────────────────────────────────────────────────────────
# Key — hierarchical indexing object
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Key:
    """
    Indexing object for CCG analysis.

    key.session      # Top level
    key.epoch        # Mid level
    key.segment      # Finest time division
    key.conn_type    # Connection type (ref, target)
    key.ref_ind      # Source neuron index
    key.target_ind   # Target neuron index

    [Dependencies]
    conn_type -> excitability
    ref_ind
        Do not set a ref_ind unless the analysis is one reference neuron vs the rest of the population.
        All other neurons in a NeuronDataset are targets; remove any unecessary neurons from NeuronDataset.
    """

    session: Optional[str] = None
    epoch: Optional[str] = None
    ref_ind: Optional[int] = None
    target_ind: Optional[int] = None
    segment: Optional[int] = None
    excitability: Optional[str] = None
    conn_type: Optional[tuple[str, str]] = None

    def __str__(self):
        parts = [
            f"sess_{self.session}" if self.session else "",
            f"epoch_{self.epoch}" if self.epoch else "",
            f"ref_{self.ref_ind}" if self.ref_ind is not None else "",
            f"tgt_{self.target_ind}" if self.target_ind is not None else "",
            f"seg_{self.segment}" if self.segment is not None else "",
            f"ex_{self.excitability}" if self.excitability else "",
            f"type_{self.conn_type[0]}-{self.conn_type[1]}"
            if self.conn_type else ""
        ]
        return ".".join(filter(None, parts)) or "root"

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

    def nd(self) -> 'Key':
        return self.get('session')


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

class EpochSlicingConfig(Config):
    """
    Configure way to slice a behavioral epoch
    """

    def __init__(
        self,
        labels: Union[list[str], str, None] = None,
        min_dur=0,
        discard=False,
    ):
        super().__init__()
        self.labels = _san(labels)
        self.min_dur = min_dur
        self.discard = discard

    def __str__(self):
        return self.__class__ + ': ' + '\n'.join(
            [f"{key}={val}" for key, val in self.__dict__.items()])


class NeuronsDatasetConfig(Config):
    """
    Metadata of NeuronsDataset

    tight_time: bool
    if true, try to shrink start and end of epoch to where brainstates are happening

    n_segments: int
    Splits session time axis into equal-lengthed blocks if >1

    """

    def __init__(
        self,
        name: str = "default",
        neuron_types: Union[list[str], str, None] = ['pyr', 'inter'],
        epochs: Union[list[str], str,
                      None] = ['pre', 'maze', 'post', 're-maze'],
        sleep: Union[EpochSlicingConfig, None] = None,
        ripple: Union[EpochSlicingConfig, None] = None,
        n_segments: Union[list[int], int] = None,
        seg_stride: int = None,
        seg_len: int = None,
        seg_spikecount: int = None,
        zero_spike_times=False,
        recinfo: NeuroscopeIO = None,
        ch_per_shank: int = 16,
    ):
        super().__init__()
        self.name = name
        self.session_names = []
        self.neuron_types = _san(neuron_types)
        self.epochs = epochs
        self.sleep = sleep
        self.ripple = ripple
        self.n_segments = _san(n_segments)
        self.seg_stride = seg_stride
        self.seg_len = seg_len
        self.seg_spikecount = seg_spikecount
        self.zero_spike_times = zero_spike_times
        self.recinfo = recinfo
        self.ch_per_shank = ch_per_shank

        if epochs is not None and n_segments is not None:
            assert len(self.n_segments) == len(self.epochs)


# ──────────────────────────────────────────────────────────────────────
# NeuronsDataset
# ──────────────────────────────────────────────────────────────────────

class NeuronsDataset(AnalysisDataset):
    """
    A collection of neurons wrapped for analysis.
    Arguments of the analysis should be provided using a NeuronsDatasetConfig object.

    sessions: subjects.ProcessData
        collection object of sessions

    data: dict[neurpy.Neurons]

    edge_times: dict[pd.DataFrame(
                            start, stop, total_time_hours, effective_time_hours, label)]
        data and edge_times has the same key list.
    """

    # TODO move per-segment per-neuron data to this level
    class NeuronSegmentStats(Savable):
        firing_rates: np.ndarray

    data: dict[Neurons]
    edge_times: dict[pd.DataFrame]
    segment_stats: dict[NeuronSegmentStats]
    segment_firing_rates: dict[np.ndarray]
    _conf: NeuronsDatasetConfig

    def __init__(
        self,
        sessions,
        conf: NeuronsDatasetConfig,
    ):
        super().__init__()
        self.data = {}
        self.edge_times = {}
        self.segment_firing_rates = {}  # 1:1 with edge_times
        self._conf = conf
        self._sessions = sessions  # keep reference for lazy ProbeGroup loading
        self._probegroups = {}     # session Key → ProbeGroup
        self.prep(sessions)

    @property
    def probegroups(self):
        """Lazy-load ProbeGroups if not yet populated."""
        if not getattr(self, '_probegroups', None):
            self._probegroups = {}
            sessions = _san(getattr(self, '_sessions', None) or [])
            for session in sessions:
                session_name = self._short_session_name(session)
                key = Key(session=session_name)
                pg_files = sorted(session.basepath.glob('*.probegroup.npy'))
                if pg_files:
                    from neuropy.core.probe import ProbeGroup
                    pg = ProbeGroup.from_file(pg_files[0])
                    recinfo = getattr(session, 'recinfo', None)
                    skipped = getattr(recinfo, 'skipped_channels', None)
                    if skipped is not None and len(skipped) > 0:
                        mask = pg._data['channel_id'].isin(np.asarray(skipped))
                        pg._data.loc[mask, 'connected'] = False
                    self._probegroups[key] = pg
        return self._probegroups

    def __str__(self):
        s = ''
        cnt = 0
        for k, v in self.data.items():
            s += f"{k}\t{str(v)}"
            cnt += 1
        return f"NeuronsDataset #sessions = {cnt}\n{s}"

    def prep(self, sessions):
        """
        Filter neurons by behavioral epochs and type
        Set segment edge timing data
        """
        c = self.conf
        self._probegroups = {}  # session Key → ProbeGroup

        sessions = _san(sessions)
        for session in sessions:
            session_name = self._short_session_name(session)
            self.conf.session_names.append(session_name)
            key = Key(session=session_name)

            # Temporal filtering on neurons
            neurons = self._time_filter(session)

            # Store edge times
            self.edge_times[key] = self._get_edge_times(key, neurons, session)
            # Store firing rates
            self.segment_firing_rates[key] = self._get_firing_rates_by_segment(
                self.edge_times[key], neurons)

            # Zero spike times
            if c.zero_spike_times:
                neurons = neurons.zero_spike_times()
                for _, et in self.edge_times.items():
                    et[['start', 'stop']] -= neurons.t_start

            # Store filtered neurons
            self.data[key] = neurons

            # Load ProbeGroup for this session
            pg_files = sorted(session.basepath.glob('*.probegroup.npy'))
            if pg_files:
                from neuropy.core.probe import ProbeGroup
                pg = ProbeGroup.from_file(pg_files[0])
                # Mark skipped channels as disconnected
                recinfo = getattr(session, 'recinfo', None)
                skipped = getattr(recinfo, 'skipped_channels', None)
                if skipped is not None and len(skipped) > 0:
                    mask = pg._data['channel_id'].isin(np.asarray(skipped))
                    pg._data.loc[mask, 'connected'] = False
                self._probegroups[key] = pg

    def _short_session_name(self, session):
        """
        Get a printable session name in the format of ANIMAL_DayX
        """
        sess_name = session.filePrefix.parts[-1].split('_')[:2]
        sess_name = '_'.join(sess_name)
        return sess_name

    def _time_filter(self, session):
        neurons = session.neurons_stable
        neurons.metadata['intervals'] = np.array(
            [[neurons.t_start, neurons.t_stop]])  #TODO move elsewhere
        if self.conf.neuron_types is not None:
            neurons = neurons.get_neuron_type(self.conf.neuron_types)

        if self.conf.epochs is not None:
            neurons = neurons.behav_slice(behav_times=session.paradigm,
                                          labels=self.conf.epochs)

        if self.conf.sleep is not None:
            neurons = neurons.behav_slice(behav_times=session.brainstates,
                                          labels=self.conf.sleep.labels,
                                          discard=self.conf.sleep.discard,
                                          min_dur=self.conf.sleep.min_dur)

        if self.conf.ripple is not None:
            neurons = neurons.behav_slice(behav_times=session.ripple,
                                          labels=None,
                                          discard=self.conf.ripple.discard,
                                          min_dur=self.conf.ripple.min_dur)
        return neurons

    def _get_edge_times(self, key: Key, neurons: Neurons, session):
        """
        Get the start and end of each segment. The edge timing are processed by epoch
        A segment is the smallest time period in the
        dataset where analysis will be performed (e.g. data used to calculate one CCG). There
        can be many overlapping segments within a dataset depending on configuration.

        Define segment edges of each neurons group
        see neurons.py:
            _edges_time_split   time_split
            _edges_time_window  time_windows
            _edges_spikecount   spikecount_split

        passing in key because we need to generate finer keys for objects in edge_times
        """

        dfs = []
        ivs = neurons.metadata['intervals']
        for i, e in enumerate(_san(self.conf.epochs, wrap_none=True)):
            if not e in session.paradigm.labels:
                print(f"{key.session} doesn't have epoch {e}")
                continue

            k = key.add(epoch=e)
            t_start, t_stop = session.paradigm.timing_by_label(e) if e \
                                else (neurons.t_start, neurons.t_stop)

            if self.conf.seg_spikecount is not None:
                # TODO spikecount segmentation code is not maintained
                neus = neurons.time_slice(t_start, t_stop)
                for i in range(neus.n_neurons):
                    k = key.add(epoch=e, ref_ind=i)
                    starts, stops = neus._edges_spikecount(
                        i=i, n=self.conf.seg_spikecount, discard_tail=False)
            elif self.conf.seg_stride is not None and self.conf.seg_len is not None:
                starts, stops = neurons._edges_time_window(
                    stride=self.conf.seg_stride,
                    seg_len=self.conf.seg_len,
                    t_start=t_start,
                    t_stop=t_stop)
            elif self.conf.n_segments is not None and self.conf.n_segments[
                    i] > 1:
                starts, stops = neurons._edges_time_split(
                    n_segments=self.conf.n_segments,
                    t_start=t_start,
                    t_stop=t_stop)
            else:
                starts, stops = np.array([t_start]), np.array([t_stop])
            """
            Calculate total/actual time lengths of each segment
            """
            edges = pd.DataFrame({
                "start": starts,
                "stop": stops,
                "key": [k.add(segment=i) for i in range(len(starts))],
                "label": [e + str(i) for i in range(len(starts))],
                "total_time_hours": (stops - starts) / 3600,
            })
            #TODO does not work for spikecount edges yet

            eths = []
            for row in edges.itertuples(index=False):
                start, stop, tth = row.start, row.stop, row.total_time_hours

                # find intervals that overlap the edge
                overlap_mask = (ivs[:, 1] > start) & (ivs[:, 0] < stop)
                overlapping_ivs = ivs[overlap_mask]

                # clip intervals to edge boundaries
                clipped_start = np.clip(overlapping_ivs[:, 0], start, stop)
                clipped_stop = np.clip(overlapping_ivs[:, 1], start, stop)

                # compute effective time in hours
                effective_hours = np.sum(clipped_stop - clipped_start) / 3600
                eths.append(min(effective_hours, tth))
            edges['effective_time_hours'] = np.array(eths)
            dfs.append(edges)
        return pd.concat(dfs, axis=0)

    def _get_firing_rates_by_segment(self, edge_times: pd.DataFrame,
                                     neurons: Neurons):
        """
            Calculate and store segment-specific firing rates
        """
        x = np.zeros((edge_times.shape[0], neurons.n_neurons))
        for i, (t_start,
                t_end) in enumerate(zip(edge_times['start'],
                                        edge_times['stop'])):
            x[i] = neurons.time_slice(t_start, t_end).firing_rate
        return x

    def frate_stats(self, key, alpha=0.05):
        """
        Generate a stats description of firing rates
        """
        from scipy.stats import describe, ttest_ind

        edge_times = self.edge_times[key]
        frates = self.segment_firing_rates[key]
        neuron_types = self.data[key].neuron_type
        labels = edge_times['label'].values
        stats_name = "firing rate"
        neuron_type_conf = self._conf.neuron_types

        for neuron_type in neuron_type_conf:
            print(f"{stats_name} stats {neuron_type}")
            print(
                f"segment | num | mean | iqr | min | max | variance | skew | kurt"
            )
            for i, (vi) in enumerate(edge_times.itertuples()):
                fr = frates[i][neuron_types == neuron_type]
                mean = np.mean(fr)
                iqr = np.percentile(fr, 75) - np.percentile(fr, 25)
                desc = describe(fr)
                print(
                    f"{str(i)+':'+str(vi.label):7} | {desc.nobs} | {mean:.2f} | {iqr:.2f} | {desc.minmax[0]:.2f} | {desc.minmax[1]:.2f} | {desc.variance:.2f} | {desc.skewness:.2f} | {desc.kurtosis:.2f}"
                )
            print("\n")

            print(f"Difference in mean {stats_name} P VALUES")
            printstr = ""
            decimal_places = int(-np.floor(np.log10(alpha)))

            # print p-value matrix
            printstr = f'{neuron_type:{decimal_places+3}}|'
            for i in range(len(labels)):
                printstr += f"{str(i):{decimal_places+3}}|"
            printstr += "\n"
            for i, (vi) in enumerate(frates):
                printstr += f"{str(i):{decimal_places+3}}|"
                for j, (vj) in enumerate(frates):
                    fri = frates[i][neuron_types == neuron_type]
                    frj = frates[j][neuron_types == neuron_type]
                    if j >= i:
                        continue
                    if len(fri) < 5:
                        printstr += f"{'-':{decimal_places}}|"
                    else:
                        p = ttest_ind(fri, frj, equal_var=True).pvalue
                        printstr += f"{p:.{decimal_places}f}{'*' if p<=alpha else ' '}|"
                        # Standard t-test,  check if mean firing rate changes per cell type
                printstr += f"{labels[i]}"
                printstr += "\n"
            print(printstr)
