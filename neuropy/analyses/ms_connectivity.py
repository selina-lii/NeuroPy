"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

from neuropy.io import NeuroscopeIO
import numpy as np
try:
    import cupy as cp
except ImportError:
    print("Error importing CuPy")
    cp = None

import neuropy.analyses.correlations as correlations
from neuropy.analyses.utils import _san, _san_np, _hasvalue, Config, AnalysisDataset, Savable, SetOp, ConfigOption
from neuropy.core.neurons import Neurons
from scipy.signal import windows
from scipy.stats import poisson, ttest_ind, ttest_1samp, describe
from scipy import ndimage
from typing import Union, Optional, Dict, Any, Tuple
import h5py
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field, replace
from collections import defaultdict
from copy import deepcopy
import imageio
import neuropy.plotting.ccg as plot_ccg
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil

# TODO
CHANNELS_PER_SHANK = 16
SAVE_ROOT = "~/Documents/Neuropy/outputs"


class IgnoreLevel(ConfigOption):
    """Config for CCG and other analysis
    Do we ignore neurons on the same peak channel / shank?"""
    NONE = 0
    SAME_CHANNEL = 1
    SAME_SHANK = 2


class NormalizeBy(ConfigOption):
    """Config for CCG and other analysis
    Do we ignore neurons on the same peak channel / shank?"""
    NONE = 0
    REF_FRATE = 1
    REF_SPKS = 2
    TARGET_FRATE = 3
    TARGET_SPKS = 4
    TIME_SPAN = 5


class ConnStrengthMethod(ConfigOption):
    PEAKSIZE = 0
    TAILED = 1


def example(var: dict):
    """
    Get an example from a dictionary
    """
    k, v = next(iter(var.items()))
    return v


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


@dataclass(frozen=True)
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

        self.ch_per_shank = CHANNELS_PER_SHANK

        if epochs is not None and n_segments is not None:
            assert len(self.n_segments) == len(self.epochs)

        # self.sleep = EpochSlicingConfig(labels=['REM','NREM'],
        #                           min_dur=120,
        #                           discard=False,)
        # self.ripple = EpochSlicingConfig(min_dur=0,
        #                           discard=True,)


class CCGConfig(Config):
    """
    All the details of CCG computation need to be predefined, kinda like a batch run
    """

    def __init__(
        self,
        name="default",
        conn_types_E: Union[list[list], list] = [('pyr', 'pyr'),
                                                 ('pyr', 'inter')],
        conn_types_I: Union[list[list], list] = [('inter', 'inter'),
                                                 ('inter', 'pyr')],
        duration: float = 20e-3,
        bin_size: float = 1e-3,
        conv_window: float = 5e-3,
        alpha: float = 0.05,
        alpha2: float = 0.1,
        min_lag: float = 1e-3,
        max_lag: float = 3e-3,
        min_spkcount=2.5,
        spkcount_scope=12e-3,
        mc_method: str = None,
        ignore=IgnoreLevel.SAME_CHANNEL,
        use_acceleration=True,
        symmetrize_ccg=True,
        normalize_methods: list[NormalizeBy] = NormalizeBy.NONE,
        conn_strength_method: ConnStrengthMethod = ConnStrengthMethod.PEAKSIZE,
    ):
        super().__init__()
        self.name = name

        self.conn_types_E = conn_types_E
        self.conn_types_I = conn_types_I
        self.duration = duration
        self.bin_size = bin_size
        self.conv_window = conv_window
        self.alpha = alpha
        self.alpha2 = alpha2
        self.mc_method = mc_method
        self.center_bin = int(self.duration / self.bin_size // 2)
        self.nbins = int(self.duration / self.bin_size) + 1  # NOTE

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_spkcount = min_spkcount
        self.spkcnt_scope = spkcount_scope
        self.spkcnt_bins = int(self.spkcnt_scope / self.bin_size)
        self.ignore = ignore

        self.min_lag_bin = self.center_bin + int(
            self.min_lag / self.bin_size)  # leftmost bin for p value test
        self.max_lag_bin = self.center_bin + int(
            self.max_lag / self.bin_size) + 1  # rightmost bin for p value test
        self.min_spkcnt_bin = self.center_bin - self.spkcnt_bins // 2  # leftmost bin requiring minimum spike count
        self.max_spkcnt_bin = self.center_bin + self.spkcnt_bins // 2 + 1  # rightmost bin requiring minimum spike count

        self.use_acceleration = use_acceleration
        self.symmetrize_ccg = symmetrize_ccg

        self.normalize_methods = _san(normalize_methods)
        self.conn_strength_method = conn_strength_method

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key}: {val}\n"
        s += f"config file: {self.filepath}\n"
        return s

    @property
    def conn_types(self):
        return {'E': self.conn_types_E, 'I': self.conn_types_I}

    @property
    def conn_types_flat(self):
        return self.conn_types_E + self.conn_types_I

    @property
    def conv_window_bins(self):
        return self.conv_window / self.bin_size

    def time2bin(self, x):
        """time in SECONDS to bin#"""
        return x / self.bin_size

    def bin2time(self, x):
        """bin# to time in SECONDS"""
        return x * self.bin_size

    @property
    def filepath(self):
        return f"{SAVE_ROOT}/{self.name}.ccg.meta.h5"


class NeuronsDataset(AnalysisDataset):
    """
    A collection of neurons wrapped for analysis
    Arguments of the analysis should be provided using a NeuronsDatasetConfig object

    sessions: subjects.ProcessData
        collection object of sessions

    data: dict[neurpy.Neurons]

    edge_times: dict[pd.DataFrame(
                            start, stop, total_time_hours, effective_time_hours, label)]
        data and edge_times has the same key list.
    """

    # TODO move per-segement per-neuron data to this level
    class NeuronSegmentStats(Savable):
        firing_rates: np.ndarray

    data: dict[Neurons]
    edge_times: dict[pd.DataFrame]
    segment_stats: dict[NeuronSegmentStats]
    segment_firing_rates: dict[np.ndarray]
    # TODO segment_firing_rates needs to account for when a neuron is missing from the segment
    # maybe segment_stats as a class is better?
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
        self.prep(sessions)

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

    def _short_session_name(self, session):
        """
        Get a printable session name in the format of ANIMAL_DayX
        """
        sess_name = session.filePrefix.parts[-1].split('_')[:2]
        sess_name = '_'.join(sess_name)
        return sess_name

    def _time_filter(self, session):
        neurons = session.neurons
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


class CCGPointer(Savable):
    """
    A positional pointer to CCGdata locations
    """

    # TODO 4 CCG() has two modes: _optimize_memory or no optimization
    # optimized CCG stores its own CCG values while unoptimized indexes a CCGraw object, avoiding recomputation at higher memory costs
    # load CCGraw as needed.
    # wrt to p-values: jitter needs to be run as sparsely as possible while eran_conv can be done for the whole dataset easily.
    # should optim CCG use sparse matrices with certain rows filled?
    # there should be two versions of CCG, but they should share some common methods,
    #           using sparse and dense matrix representations.
    # filtered: ccg shape = ngroups,nccg,nbins     ids shape = nccg,2
    # flat:     ccg shape = (ngroups),nneu,nneu,nbins   ids shape = nneu
    # since segments will be abolished, ngroups dimension should always exist, even when redundant
    # segments is a dimension such that all items in the list can have the same operations applied on them
    #   as segments are purely for comparison analysis
    def __init__(
        self,
        key,
        inds,
        conf: CCGConfig = None,
        significant=None,
        edge_times=None,
    ):
        super().__init__()
        self.key = key
        self._inds = _san_np(inds)
        self.edge_times = edge_times
        self.conf = conf
        self.significant = significant

    # def __repr__(self):
    #     printstr = "CCG name: " + str(self.key) + "\n===\n"
    #     printstr += "Legend\n" + self.segment_labels + "\n===\n"
    #     conn = self.connectivity
    #     if conn is None: return 'No connectivity'
    #     printstr += f"{'Pair indices':<15}\n"
    #     for (x, y) in sorted(conn.keys()):
    #         printstr += f"{f'({x}, {y})':<15}\tIn segments {str(conn[(x,y)]):<20}\t"
    #         printstr += "\n"
    #     return printstr

    @property
    def connectivity_array(self):
        conn = self.connectivity
        inds = self.inds2
        a = pd.DataFrame(0,
                         index=pd.MultiIndex.from_arrays(inds.T),
                         columns=range(self.n_segments))
        for ind in inds:
            a.loc[tuple(ind), conn[tuple(ind)]] = 1
        return a

    @property
    def segment_labels(self):
        printstr = ""
        for i, l in enumerate(self.edge_times.label.values):
            printstr += f"{str(i):2}:{l:10}"
            if (i + 1) % 5 == 0:
                printstr += "\n"
        return printstr

    @property
    def stored_by_segment(self):
        if not _hasvalue(self._inds):
            return False
        return self._inds.shape[-1] == 3  #otherwise 2

    @property
    def connectivity(self):
        if not _hasvalue(self.inds):
            return None
        d = defaultdict(list)
        if self.stored_by_segment:
            d = defaultdict(list)
            for i, x, y in self.inds:
                d[(x, y)].append(i)
        else:
            for x, y in self.inds:
                d[(x, y)].append(list(np.arange(self.n_segments)))
        return d

    def __str__(self):
        s = 'CCG Pointer\n'
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray) or isinstance(val, list):
                s += f"{key}\tshape={np.array(val).shape}"
                sval = "\n".join(str(val[0:2]).splitlines()[:3])
                s += f"\tval={sval}...\n"
            elif isinstance(val, dict):
                k, v = next(iter(val.items()))
                s += f"{key} dict keys={k}...\n"
                item_str = str(v)
                for line in item_str.splitlines()[:3]:
                    s += f"\t\t{line}\n"
            elif key != '_conf':
                sval = "\n".join(str(val).splitlines()[:3])
                s += f"{key}: {sval}\n"
        return s

    @property
    def inds2(self):
        if self.stored_by_segment:
            return SetOp.unique(self._inds[:, -2:])
        else:
            return self.inds

    @property
    def inds3(self):
        if self.stored_by_segment:
            return self._inds
        else:
            x = np.arange(self.n)
            yz = self.inds[:, 1:]
            return np.column_stack(
                [np.repeat(x, yz.shape[0]),
                 np.tile(yz, (x.shape[0], 1))])

    @property
    def indsplit(self) -> list[np.ndarray]:
        """Indices listed by each segment"""
        return [
            self.inds[np.where(self.inds[:, 0] == i)[0]][:, 1:]
            for i in range(self.n_segments)
        ]

    @property
    def n_pairs(self):
        if not _hasvalue(self._inds):
            return 0
        if self.stored_by_segment:
            return self.inds2.shape[0]
        else:
            return self.inds.shape[0]

    def n_pairs_segment(self, i: int):
        if self.stored_by_segment:
            return np.sum(np.where(self.inds[:, 0] == i)[0])
        else:
            assert i < self.n_segments
            return self.n_pairs

    @property
    def inds(self):
        return self._inds

    @property
    def ref_inds(self):
        return self.inds[:, -2]

    @property
    def target_inds(self):
        return self.inds[:, -1]

    @property
    def ref_ind(self):
        return self.inds[-2]

    @property
    def unique_inds(self):
        """Returns list of unique neuron inds, not pairs!"""
        return np.unique(self.inds)

    @property
    def n_segments(self):
        return self.edge_times.shape[0]

    def get_segment(self, i: int) -> 'CCGPointer':
        if self._inds is None:
            inds = None
        elif self.stored_by_segment is False:
            assert i < self.n_segments
            inds = np.hstack([np.zeros(self.n_pairs), self.inds])
        else:
            inds = self.inds[np.where(self.inds[:, 0] == i)[0]][:, 1:]
        return CCGPointer(
            key=self.key.add(segment=i),
            inds=inds,
            edge_times=self.edge_times.iloc[i],
        )

    def split(self) -> list['CCGPointer']:
        return [self.get_segment(i) for i in range(self.edge_times.shape[0])]

    def print_connectivity(self):
        grouped = defaultdict(list)
        for seg_i, x, y in self.inds:
            grouped[(x, y)].append(seg_i)
        for (x, y), seg_inds in grouped.items():
            print(f"[{x},{y}] appearing in [{', '.join(map(str, seg_inds))}]")

    def plotdir(self, root):
        root = os.path.expanduser(root)
        tag = ''
        for v in [self.key.epoch, self.key.segment]:
            if v is not None:
                tag += v
        if tag != '':
            tag = '-' + tag
        if self.key.conn_type is None:
            return f"{root}/{self.key.session}/{self.key.session}{tag}/{self.key.excitability}_any"
        return f"{root}/{self.key.session}/{self.key.session}{tag}/{self.key.excitability}_{self.key.conn_type[0]}-{self.key.conn_type[1]}"


@dataclass
class CCGData(Savable):
    """
    Stores the whole CCG array and its p values

        ccg         [N, Np, Nbins]
        N = number of data segments
        Np = number of neuron pairs
        Nbins = number of bins per CCG 
        ccg_null    [N, Np, Nbins]
    """
    key: Key
    _conf: CCGConfig

    # [n_seg, n_ref, n_tgt, n_bins]
    ccg: np.ndarray
    ccg_null: np.ndarray
    pval: np.ndarray
    qval: np.ndarray
    pval_corrected: np.ndarray
    qval_corrected: np.ndarray
    significant: np.ndarray
    norm_factors: list[np.ndarray]

    # [n_seg, n_ref, n_tgt]
    conn_strength: np.ndarray

    def __post_init__(self):
        super().__init__()  # initializes parent attributes

    @property
    def conf(self):
        return self._conf

    @property
    def n_segment(self):
        return self.ccg.shape[0]

    @staticmethod
    def get_autocorr_locations(shape):
        """
        Genearte a mask of autocorrelation locations shaped (ngroups, nneurons, nneurons)
        """
        n_auto = min(shape[-3], shape[-2])
        auto_mask = np.eye(n_auto, dtype=bool)
        auto_mask = np.pad(auto_mask,
                           ((0, shape[-3] - n_auto), (0, shape[-2] - n_auto)))
        autocorr_locations = np.broadcast_to(auto_mask, shape[:-1])
        return autocorr_locations

    def get_conn_strength(self, method: ConnStrengthMethod, **kwargs):
        """
        Wrapper
        """
        if self.ccg is None:
            return  # no connection
        if method == ConnStrengthMethod.PEAKSIZE:
            self.__get_conn_strength_peaksize(**kwargs)
        elif method == ConnStrengthMethod.TAILED:
            self.__get_conn_strength_tailed(**kwargs)
        else:
            return NotImplementedError("Unknown connection strength method")

    @property
    def conn_strength_change(self):
        return self.conn_strength[-1, ...] - self.conn_strength[0, ...]

    def get_normalize_factors(self, edge_times=None, frates=None):
        """
        Divide CCG by normalization factors
        
        :param edge_times: shape [n_segment,...]
        :param frates: shape [n_segment,n_neurons]
        """

        axes = [0] if NormalizeBy.TIME_SPAN else []
        axes += [1] if NormalizeBy.REF_FRATE else []
        axes += [2] if NormalizeBy.TARGET_FRATE else []

        self.norm_factors = []
        for axis in axes:
            shape = [1] * self.ccg.ndim
            shape[0] = self.n_segment
            shape[axis] = -1 if axis == 0 else frates.shape[1]

            arr = edge_times.effective_time_hours.values if axis == 0 else frates
            reshaped = arr.reshape(shape)
            self.norm_factors.append(reshaped)

    def save_plots(self,
                   pt: CCGPointer,
                   neurons: Neurons,
                   neurons_config: NeuronsDatasetConfig,
                   frates_cut: np.ndarray,
                   plotdir: str,
                   split_all_plots=False,
                   overwrite=False,
                   overlay_normalized=False):

        if not os.path.exists(plotdir):
            os.makedirs(plotdir, exist_ok=True)

        if overlay_normalized:
            normalize_info = []
            if NormalizeBy.REF_FRATE in self.conf.normalize_methods:
                normalize_info.append(
                    'normalized by reference neuron firing rate')
            if NormalizeBy.TARGET_FRATE in self.conf.normalize_methods:
                normalize_info.append('normalized by target neuron firing rate')
            if NormalizeBy.TIME_SPAN in self.conf.normalize_methods:
                normalize_info.append('normalized by total time (hours)')
            normalize_info = '\n'.join(normalize_info)
        else:
            normalize_info = None

        if self.n_segment > 1 or split_all_plots:
            self.__save_gif(
                plotdir=plotdir,
                pt=pt,
                neurons=neurons,
                neurons_config=neurons_config,
                frates_cut=frates_cut,
                overwrite=overwrite,
                normalize_info=normalize_info,
                overlay_normalized=overlay_normalized,
            )
        else:
            self.__save_img()

    def __get_conn_strength_peaksize(self, norm_factor: np.ndarray = None):
        """
        Connection strength:
        
            Area under CCG curve minus baseline, within temporal ROI
            The ROI is by default the same as the interval tested for peak/trough signficance
            Can be negative
        """
        auc = self.ccg - self.ccg_null  # area under curve
        cs = np.sum(auc[..., self.conf.min_lag_bin:self.conf.max_lag_bin],
                    axis=-1)  # (inds,)
        if norm_factor is not None:
            cs = cs / norm_factor  # e.g. presynaptic element firing rate
        self.conn_strength = cs

    def __get_conn_strength_tailed(self,
                                   nspks: list,
                                   norm_factor: np.ndarray = False):
        """
        Connection strength:

                Area under CCG curve minus a 'tailed' baseline after deconvolving autocorrelograms
        
        Can be negative
        TODO testing
        """

        # Remove autocorrelograms (ACG) from CCG
        # target/reference is set to true if corresponding ACG is to be removed
        remove_target = True
        remove_ref = True

        def _deconv_autocorr(ccg,
                             acg1=None,
                             nspks1=None,
                             acg2=None,
                             nspks2=None):
            """
            Deconvolve acgs from ccg using FFT-based method.
            Translated from MATLAB
            https://github.com/EranStarkLab/CCH-deconvolution/cchdeconv.m
            
            Parameters
            ----------
            acg1, acg2 : ndarray
                Autocorrelograms for neurons 1 and 2
            nspks1, nspks2 : int or float
                Number of spikes for neurons 1 and 2
            
            Returns
            -------
            dcccg : ndarray
                Deconvolved cross-correlogram
            """
            # Preparations
            m = ccg.shape[-1]
            assert m % 2 == 1  # CCG must have an odd number of bins
            hw = (m - 1) // 2  # midpoint

            # Scale acg1
            acg1 = (acg1 - np.mean(acg1)
                   ) / nspks1  # remove mean of clipped, divide by nspks1
            hidx = np.concatenate([np.arange(hw),
                                   np.arange(hw + 1, m)])  # [0:hw, (hw+1):m]
            acg1[hw] = 1 - np.sum(acg1[hidx])  # set zero-lag bin s.t. sum of 1
            den = np.fft.fft(acg1)

            if acg2 is not None:
                # Scale acg2
                acg2 = (acg2 - np.mean(acg2)
                       ) / nspks2  # remove mean of clipped, divide by nspks2
                hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])
                acg2[hw] = 1 - np.sum(
                    acg2[hidx])  # set zero-lag bin s.t. sum of 1
                den = den * np.fft.fft(acg2)

            # Deconvolve acgs from the ccg
            dcccg = np.real(np.fft.ifft(np.fft.fft(ccg) / den))

            # Set my CCG to deconvCCG
            dcccg = np.concatenate([dcccg[1:], [dcccg[0]]])  # shift DC to end
            dcccg[dcccg < 0] = 0  # clip negatives to zero
            return dcccg

        acgs = self.ccg[self.get_autocorr_locations()]
        for i, (ref, tgt) in enumerate(self.inds):
            if remove_ref and remove_target:
                self.ccg[i] = _deconv_autocorr(self.ccg[i], acgs[ref],
                                               nspks[ref], acgs[tgt],
                                               nspks[tgt])
            elif remove_ref:
                self.ccg[i] = _deconv_autocorr(self.ccg[i], acgs[ref],
                                               nspks[ref])
            elif remove_target:
                self.ccg[i] = _deconv_autocorr(self.ccg[i], acgs[tgt],
                                               nspks[tgt])
            else:
                Warning("_deconv_autocorr: No effect")
                return

        # Baseline by 'tail' (|t|>11ms)
        # 'Tail' is accurate when coupled with autocorr deconvolution
        l = self.conf.time2bin(-11e-3)
        r = self.conf.time2bin(11e-3)
        baseline = np.mean([self.ccg[:l], self.ccg[r + 1:]])
        self.ccg_null = np.ones_like(self.ccg) * baseline

        # area under curve
        auc = self.ccg - self.ccg_null
        cs = np.sum(
            auc[self.conf.min_lag_bin:self.conf.max_lag_bin])  # inds,nbins
        if norm_factor:
            cs /= norm_factor
        self.conn_strength = cs

    def __save_gif(
        self,
        plotdir,
        pt: CCGPointer,
        neurons: Neurons,
        neurons_config: NeuronsDatasetConfig,
        frates_cut: np.ndarray,
        overwrite=False,
        overlay_normalized=False,
        normalize_info=None,
    ):
        if pt.inds is None:
            print(f"nothing to plot: {pt}")
            return
        
        if overlay_normalized:
            normalized_ccg = self.ccg.astype(float)
            normalized_ccg_null = self.ccg_null.astype(float)
            for nf in self.norm_factors:
                normalized_ccg /= nf
                normalized_ccg_null /= nf

        for i, inds in enumerate(pt.inds2):
            sig = self.significant[:,
                                   *inds] if self.significant is not None else None
            where_sig = np.where(sig)[0]
            save_path = f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}-{'-'.join(['sig'+str(s) for s in where_sig])}.gif"
            if not overwrite and os.path.exists(save_path):
                print(f"{save_path} already exists")
                continue

            figs = []
            ymin, ymax, ymin2, ymax2 = [], [], [], []
            print(i, inds)
            for i_seg in range(self.n_segment):
                loc = (i_seg, *inds)
                fig = plot_ccg.plot_ccg_figure(
                    inds=inds,
                    ids=neurons.ind2id(inds),
                    neuron_types=neurons.neuron_type[inds],
                    frates_cut=frates_cut[i_seg][inds],
                    frates_all=neurons.firing_rate[inds],
                    waveforms=neurons.waveforms[inds],
                    shank_ids=neurons.shank_ids[inds],
                    discarded_channels=neurons_config.recinfo.skipped_channels,
                    ch_per_shank=neurons_config.ch_per_shank,
                    save_path=None,
                    window_size=self.conf.duration * 1e3,
                    bin_size=self.conf.bin_size * 1e3,
                    ccg=self.ccg[loc],
                    ccg_null=self.ccg_null[loc]
                    if self.ccg_null is not None else None,
                    pval=self.pval[loc] if self.pval is not None else None,
                    pval_corrected=self.pval_corrected[loc]
                    if self.pval_corrected is not None else None,
                    alpha=self.conf.alpha
                    if self.conf.alpha is not None else None,
                    is_significant_pair=self.significant[loc]
                    if self.significant is not None else None,
                    show=False,
                    save=False,
                    segment_id=i_seg,
                    normalize_info=normalize_info,
                    overlay_normalized=overlay_normalized,
                    normalized_ccg=normalized_ccg[loc]
                    if overlay_normalized else None,
                    normalized_ccg_null=normalized_ccg_null[loc]
                    if overlay_normalized else None,
                )
                figs.append(fig) 
                _ymin, _ymax = fig.axes[0].get_ylim()
                ymin.append(_ymin)
                ymax.append(_ymax)
                if overlay_normalized:
                    _ymin, _ymax = fig.axes[1].get_ylim()
                    ymin2.append(_ymin)
                    ymax2.append(_ymax)
            ymin, ymax = min(ymin), max(ymax)
            if overlay_normalized: 
                ymin2, ymax2 = min(ymin2), max(ymax2)
            frames = []
            for fig in figs:
                fig.axes[0].set_ylim(ymin, ymax)
                if overlay_normalized:
                    fig.axes[1].set_ylim(ymin2, ymax2)
                fig.canvas.draw_idle()
                frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
                plt.close(fig)

            imageio.mimsave(save_path, frames, duration=0.8)

        print("done saving plots")


class CCGDataset(AnalysisDataset):
    """
    Data and operations on CCGs from an experiment.

    Requires a NeuronsDataset to be processed first, and a configuration object
    (see :class:`CCGConfig`).

    Tests CCGs and stores them separately by significance criteria.

    Attributes
    ----------
    data : dict[CCGPointer, ...]
        Neuron pairs that are significant connections.
    spurious : dict[CCGPointer, ...]
        Neuron pairs that passed rough significance checks but do not belong to a
        certain connection type.
    conf : CCGConfig
        Configuration.
    nd : NeuronsDataset
        Source neurons for the CCGs.
    """

    _ccg: dict[CCGData]
    data: dict[CCGPointer]
    spurious: dict[CCGPointer]
    _conf: CCGConfig
    nd: NeuronsDataset

    def __init__(
        self,
        conf=None,
        nd=None,
    ):
        super().__init__(conf)
        self.nd = nd
        self._ccg = {}
        self.data = {}
        self.spurious = {}
        self.get_ccg()

    @property
    def filepath(self):
        return ''

    def get_ccg(self, baseline_method="eran_conv", use_segments=True):
        """
        main function of the class
        """
        if baseline_method == "eran_conv":
            conv = EranConv(self.conf)
            for key, edge_times in self.nd.edge_times.items():
                self.__ccg_eranconv(key=key,
                                    conv=conv,
                                    edge_times=edge_times,
                                    use_segments=use_segments)
                self.get_normalize_factors()
        elif baseline_method == "jitter":
            NotImplementedError("CCG jitter must be run in the Jitter object, " \
            "since it generates a ton of extra data. Nothing is run...")
        else:
            ValueError("Unknown method")

    def copy(self) -> "CCGDataset":
        """Copy only conf and nd (nd is a shallow reference)"""
        new = self.__class__(conf=self._conf)
        new.nd = self.nd
        return new

    def merge_CCGs(self, merge_level='epoch'):
        # TODO
        groups = self.groupby(merge_level, source='_ccg')  #TODO
        self._ccg = {}
        for key, group in groups:
            self._ccg[key] = CCGData.merge(group)

    def split_CCG(self, level='epoch'):
        # TODO
        for key, ccg in self._ccg:
            ccg.split(level=level)

    def change_timescale(self,
                         bin_size,
                         duration=None,
                         jscale=None) -> 'CCGDataset':
        """
        Run CCG and convolution based significance test for all neurons

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        # change timescale in my configurations.
        print(
            f"recalculated CCG from binsize={self._conf.bin_size} to binsize={bin_size}"
        )
        self._conf.bin_size = bin_size
        if duration:
            self._conf.duration = duration
        if jscale:
            self._conf.jscale = jscale
        self.get_ccg()
        print("rescale completed")

    def save_plots(self,
                   root="~/Documents/ms_synchrony/NeuroPy/images/ccg_plots_tmp",
                   source='data',
                   overwrite=False,
                   overlay_normalized=False,
                   **filters):

        root = os.path.expanduser(root)
        assert os.path.isdir(root)
        print(f"Saving plots under {root} reload")

        if root == os.path.expanduser(
                "~/Documents/ms_synchrony/NeuroPy/images/ccg_plots_tmp"):
            for f in os.listdir(root):
                p = os.path.join(root, f)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

        itergroup = self.filter(attrname=source, **filters)
        if itergroup is None:
            return

        for key, ccg_pointer in itergroup.items():
            print(f"ccg {key.session} {key.conn_type}")

            self._ccg[key.nd()].save_plots(
                pt=ccg_pointer,
                neurons=self.nd.data[key.nd()],
                frates_cut=self.nd.segment_firing_rates[key.nd()],
                neurons_config=self.nd.conf,
                plotdir=ccg_pointer.plotdir(root),
                overwrite=overwrite,
                overlay_normalized=overlay_normalized,
            )
        print("done")

    def get_normalize_factors(self):
        for key, ccg in self._ccg.items():
            ccg.get_normalize_factors(
                edge_times=self.nd.edge_times[key.nd()],
                frates=self.nd.segment_firing_rates[key.nd()])

    def get_connection_strengths(self, method=ConnStrengthMethod.PEAKSIZE):
        """
        Set connection_strengths value for ccg data based on given method.
        Values can be found in self._ccg[key].conn_strengths.
        """
        for key, ccg_data in self._ccg.items():
            if method == ConnStrengthMethod.TAILED:
                frate = self.nd.segment_firing_rates[key]
                total_time = self.nd.edge_times[key]['effective_time_hours']
                spikecount = np.round(frate * total_time)  #TODO precision
                ccg_data.get_conn_strength(method=method, spikecount=spikecount)
            elif method == ConnStrengthMethod.PEAKSIZE:
                ccg_data.get_conn_strength(method=method)
            else:
                raise NotImplementedError()

    def plot_connection_strengths(
            self,
            n_segments_threshold=None,
            norm_by_n_sess=False,
            norm_by_total_strength=False,
            zero_first_timepoint=False,
            show_legend=False,
            skips={},
            save=False,
            root='~/Documents/NeuroPy/images/conn_strengths',
            debug=False):

        def filter(self, inds, min_n_segment, skips):
            if skips is not None:
                inds = [
                    int(i)
                    for i, (x, y) in enumerate(inds)
                    if not ((x in skips[:, 0]) & (y in skips[:, 1])) and
                    (np.sum(self.significant[:, x, y]) >= min_n_segment)
                ]
            else:
                inds = np.where(
                    np.sum(self.significant[(
                        slice(None),
                        *inds.T)], axis=1) >= min_n_segment)[0].astype(int)
            return inds

        for k, cp in self.data.items():
            skip_k = skips.get(k)
            pairs = filter(cp.inds,
                           min_n_segment=n_segments_threshold,
                           skips=skip_k)
            ccg_data = self._ccg[k.nd()]
            plot_ccg.plot_strength(
                key=k,
                n_segments_threshold=n_segments_threshold,
                plot_data=ccg_data.conn_strength[:, pairs[:, 0], pairs[:, 1]],
                pairs=pairs,
                significant=ccg_data.significant[:, pairs[:, 0], pairs[:, 1]],
                n_segments=cp.n_segments,
                save=save,
                root=root,
                norm_by_n_sess=norm_by_n_sess,
                norm_by_total_strength=norm_by_total_strength,
                zero_first_timepoint=zero_first_timepoint,
                show_legend=show_legend,
                has_skips=skip_k is not None,
                debug=debug)

    def __ccg_eranconv(self, key, conv, edge_times, use_segments=True):
        """
        Run CCG and generate a convolution-based baseline for all neurons in my NeuronsDataset.
        Run significance tests.
        Store results in objects:
            self._ccg
            self.data
            self.spurious.

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        print("EranConv significant pairs")

        neurons = self.nd.data[key.nd()]

        ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(neurons.n_neurons),  # all
            bin_size=self.conf.bin_size,
            window_size=self.conf.duration,
            use_acceleration=self.conf.use_acceleration,
            symmetrize=self.conf.symmetrize_ccg,
            edge_times=edge_times if use_segments else None,
        )

        if self.conf.ignore == IgnoreLevel.SAME_CHANNEL:
            neuron_locations = neurons.peak_channels
        elif self.conf.ignore == IgnoreLevel.SAME_SHANK:
            neuron_locations = neurons.shank_ids
        else:
            neuron_locations = None

        pvals, pred, qvals, ccg_pointers, spur_pointers, printstr = conv.eranconv(
            neurons_key=key,
            ccg=ccg,
            edge_times=edge_times,
            neuron_locations=neuron_locations,
            neuron_type=neurons.neuron_type,
            conf=self.conf)

        ccg_data = CCGData(key=key,
                           _conf=self.conf,
                           ccg=ccg,
                           ccg_null=pred,
                           pval=pvals,
                           qval=qvals,
                           pval_corrected=conv._pvals,
                           qval_corrected=conv._qvals,
                           significant=conv._significant,
                           conn_strength=None,
                           norm_factors=None)

        self._ccg[key] = ccg_data
        self._attr_append(key, ccg_pointers, 'data')
        self._attr_append(key, spur_pointers, 'spurious')

        tt = edge_times.total_time_hours.values
        et = edge_times.effective_time_hours.values

        s = f"======={key.session}=======\n"
        s += f"Segment(s) are {[f'{_:.2f}' for _ in et]} hours long"
        if self.nd.conf.sleep is not None:
            s += f"\nand contain {[f'{_:.2f}' for _ in et]} hours of actual sleep "
        for _ in self.nd.conf.neuron_types:
            s += f"{_}={neurons.get_neuron_type(_).n_neurons} "
        s += "\n"

        print(s + printstr)


class EranConv:
    """
    A device for running EranConv and other significance tests    
    """

    def __init__(self, conf):
        self._pvals = []
        self._qvals = []
        self._significant = []  # final filtering results
        self.conf = conf

    @staticmethod
    def _conv(ccg, W=5, wintype="gauss", hollow_frac=None):
        """
        Estimate chance-level correlations using convolution method from Stark and Abeles (2009, J. Neuro Methods).
        Referencing MATLAB script EranConv.m written by the authors

        Parameters
        ----------
        ccg: np.array. 
            1D or 2D. (CCGs in columns)
            If 2D, elements in the first dimension are individual ccgs and second dimension are bins.
        W: 
            defines the width (unit: ms) of the convolution window, should be same as size of jitter window if were to use one
            `gauss`: W is standard deviation (sigma). Total window length will be 
            `rect`: Half size of window = W, total length is always odd
            `triang`: Window length is W rounded up to the nearest odd number

        wintype: ["gauss", "rect", "triang"]
            Type of convolution window.
            `gauss`: Gaussian kernel
            `rect`: rectangular kernel
            `triang`: triangular kernel

        hollow_frac: weight of the current bin
        
        Returns
        -------
        pvals: p-values (bin-wise)
        pred: predictor (expected values) 
        qvals: p-values (bin-wise) for inhibition
        """
        if len(ccg.shape) == 1:
            ccg = ccg[np.newaxis, ...]

        assert wintype in ["gauss", "rect", "triang"]
        assert W <= ccg.shape[-1]

        # Auto-assign appropriate hollow fraction if not specified
        # generate window
        # get center indices of window
        if wintype == "gauss":
            hollow_frac = hollow_frac or 0.6
            sigma = W / 2
            W = int(6 * sigma + (2 if W % 2 else 1))
            center = int(3 * sigma + (0.5 if W % 2 else 0))
            window = windows.gaussian(W, std=sigma) / (2 * np.pi * sigma)
        elif wintype == "rect":
            hollow_frac = hollow_frac or 0.42
            if W % 2 == 0:
                W += 1
            center = W // 2
            window = windows.boxcar(W)
        elif wintype == "triang":
            hollow_frac = hollow_frac or 0.63
            W = 2 * W + (-1 if W % 2 else 1)
            center = W // 2
            window = windows.triang(W)

        # hollow and normalize window
        window[center] *= (1 - hollow_frac)
        window /= np.sum(window)
        # padding
        ccg_pad = np.concatenate(
            [ccg[..., :W][..., ::-1], ccg, ccg[..., -W:][..., ::-1]], axis=-1)

        # convolve window with ccg
        pred = ndimage.convolve1d(ccg_pad, window, axis=-1)
        pred = pred[..., W:-W]

        # mid-p Poisson test: P( val<=pred ) + half of P ( val==pred )
        pvals = 1 - poisson.cdf(ccg - 1, pred) - poisson.pmf(ccg, pred) * 0.5
        qvals = 1 - pvals
        return pvals, pred, qvals

    @staticmethod
    def multiple_correction(
            pvals,
            alpha):  # correct for number of bins
        """
        Hierarchical FDR as described in https://doi.org/10.1093/biomet/asaa086, 4.1
        Also see statsmodels.stats.multitest.multipletests.
        """        
        def simes_1d(p):
            return np.min(p.size * np.sort(p) / np.arange(1, p.size + 1))

        p0 = np.apply_along_axis(simes_1d, axis=-1, arr=pvals)
        s0, pc0, _, _ = multipletests(p0.ravel(), alpha, method='fdr_bh')
        s0 = s0.reshape(p0.shape)
        pc0 = pc0.reshape(p0.shape)

        alpha1 = np.sum(s0)/ np.prod(p0.shape)
        p1 = pvals
        pc1=np.full_like(p1,fill_value=np.nan)
        s1=np.full_like(p1,fill_value=np.nan)
        s1[s0], pc1[s0] = np.apply_along_axis(multipletests,axis=-1,arr=p1[s0],\
                                     alpha=alpha1,method='fdr_bh')
        return s1, pc1

    def spkcount_mask(self, ccg):
        min_bin = self.conf.min_spkcnt_bin
        max_bin = self.conf.max_spkcnt_bin
        threshold = self.conf.min_spkcount
        pair_inds = np.argwhere((ccg[..., min_bin:max_bin]
                                 >= threshold).all(axis=-1))
        # NOTE right now it's the same criteria for exctiation/inhibition
        return pair_inds

    def significance_mask(self, p, excitability):
        conf = self.conf
        if excitability == 'E':
            sig, self._pvals = EranConv.multiple_correction(
                p, conf.alpha, method=self.conf.mc_method)
            has_valid_peak = sig[...,
                                 conf.min_lag_bin:conf.max_lag_bin].any(axis=-1)
            # sig2, _ = EranConv.multiple_correction(p,
            #                                        conf.alpha2,
            #                                        method=self.conf.mc_method)
            # has_bad_peak = sig2[..., :conf.min_lag_bin].any(-1) | sig2[
            #     ..., conf.max_lag_bin:].any(-1)
            pair_inds = np.argwhere(has_valid_peak)  # & ~has_bad_peak)
        elif excitability == 'I':
            sig1, self._qvals = EranConv.multiple_correction(
                p, conf.alpha, method=self.conf.mc_method)
            sig2, _ = EranConv.multiple_correction(p,
                                                   conf.alpha2,
                                                   method=self.conf.mc_method)
            neighbor = sig1 & (
                np.roll(sig2, 1, -1) | np.roll(sig2, -1, -1)
            )  # significant bins must have a significant-ish neighbor
            pair_inds = np.argwhere(neighbor.any(-1))
        return pair_inds

    def _autocorr_mask(self, pair_inds):
        if _hasvalue(pair_inds):
            pair_inds = np.array(
                [inds for inds in pair_inds if inds[-2] != inds[-1]])
        return pair_inds

    def _cell_type_mask(self, pair_inds, neuron_type, conn_types):
        sig_pairs = {}
        # Conn types with no pairs are marked with None
        if not _hasvalue(pair_inds):
            for ct in conn_types:
                sig_pairs[ct] = None

        # Condition 3: Ref/Target are specific cell types
        for ct in conn_types:
            inds = np.where(
                np.isin(pair_inds[:, -2], np.where(neuron_type == ct[0])) &
                np.isin(pair_inds[:, -1], np.where(neuron_type == ct[1])))[0]
            sig_pairs[ct] = pair_inds[inds] if inds.shape[0] else None
        return sig_pairs

    def _probe_loc_mask(self, pair_inds, neuron_locations):
        """
        Filter out pairs that are too close by
        """
        if _hasvalue(neuron_locations) and _hasvalue(pair_inds):
            # Check by locations
            x, y = pair_inds[:, -2], pair_inds[:, -1]
            inds = np.where(neuron_locations[x] != neuron_locations[y])[0]
            return pair_inds[inds]
        return pair_inds

    def eranconv(
        self,
        neurons_key: Key,
        ccg,
        edge_times: pd.DataFrame,
        neuron_locations,
        neuron_type,
        conf: CCGConfig,
    ):
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
        print("running eranconv (1st pass)")
        key = neurons_key
        self.conf = conf
        self.n_segments = edge_times.shape[0]

        pvals, pred, qvals = EranConv._conv(ccg,
                                            W=conf.conv_window_bins,
                                            wintype="gauss",
                                            hollow_frac=None)

        def build_inds(p, EI, conn_types):
            rough_inds = SetOp.intersect(self.significance_mask(p, EI),
                                         self.spkcount_mask(ccg))
            rough_inds = self._autocorr_mask(rough_inds)
            inds = self._probe_loc_mask(rough_inds, neuron_locations)
            inds = self._cell_type_mask(inds, neuron_type, conn_types)
            return rough_inds, inds

        # [n_seg, n_pair, 2]
        rough_inds_E, inds_E = build_inds(pvals, 'E', conf.conn_types_E)
        rough_inds_I, inds_I = build_inds(qvals, 'I', conf.conn_types_I)

        # Record a global map of significant pairs
        self._significant = np.zeros(ccg.shape[:3], dtype=bool)
        for inds in [inds_E, inds_I]:
            if inds is None:
                continue
            for k, v in inds.items():
                if v is None:
                    continue
                self._significant[tuple(v.T)] = True

        ccg_inds_by_type, spur_inds_by_type = {}, {}

        # Force CCG to be 4D
        if ccg.ndim == 3:
            ccg = ccg[None]
            pred = pred[None]
            for attr in (
                    "_pvals",
                    "_qvals",
            ):
                setattr(self, attr, getattr(self, attr)[None])

        count = np.zeros((edge_times.shape[0], len(self.conf.conn_types_flat)),
                         dtype=int)
        j = 0
        # Update return values
        for EI in ['E', 'I']:
            spurious = rough_inds_E if EI == 'E' else rough_inds_I  # initialize spurious pairs

            for conn_type in self.conf.conn_types[EI]:
                inds = inds_E[conn_type] if EI == 'E' else inds_I[conn_type]
                ccg_key = key.add(conn_type=conn_type, excitability=EI)
                ccg_pointer = CCGPointer(key=ccg_key,
                                         conf=self.conf,
                                         inds=inds if _hasvalue(inds) else None,
                                         edge_times=edge_times)
                for i, ccg in enumerate(ccg_pointer.split()):
                    count[i, j] = ccg.n_pairs if ccg is not None else 0
                ccg_inds_by_type[ccg_key] = ccg_pointer
                spurious = SetOp.setdiff(
                    spurious, inds)  # remove these pairs from spurious
                j += 1

            spur_key = key.add(excitability=EI)
            spur_inds_by_type[spur_key] = CCGPointer(
                key=spur_key,
                conf=self.conf,
                inds=spurious if _hasvalue(spurious) else None,
                edge_times=edge_times)

        printstr = ''
        for i, (segment_i, edge_time) in enumerate(edge_times.iterrows()):
            N_totalE = (rough_inds_E[:, 0] == i).sum()
            N_totalI = (rough_inds_I[:, 0] == i).sum()
            printstr += f"{edge_time['label']:10}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "

            EIs = ['E', 'E', 'I', 'I']  #TODO
            for N, (ref, target), EI in zip(count[i], self.conf.conn_types_flat,
                                            EIs):
                printstr += f"{ref}-{target}/{EI} {f'{N:02d}' if N>0 else ' -'} | "
            printstr += '\n'

        print("eranconv done")

        return pvals, pred, qvals, ccg_inds_by_type, spur_inds_by_type, printstr


def routine_eranconv_connection_info(info,
                                     nd: NeuronsDataset,
                                     cd: CCGDataset,
                                     epoch_id=0):
    """
    Print aggregated information of eranconv_pairs() outputs

    info: eranconv_pairs outputs
    """
    results = {'E': {}, 'I': {}}
    total_by_conntype = {'E': {}, 'I': {}}
    total_by_EI = {'E': 0, 'I': 0}
    for EI in ['E', 'I']:
        for conn_type in cd.conf.conn_types[EI]:
            results[EI][conn_type] = {'sig_conv': 0, 'list': []}
            total_by_conntype[EI][conn_type] = 0

    neuron_types = nd.conf.neuron_types
    epoch = nd.conf.epochs[epoch_id]

    for key, neurons in nd.data.items():
        n = {}
        for _ in neuron_types:
            n[_] = neurons.get_neuron_type(_).n_neurons

        total_by_EI['E'] += neurons['E']['total']
        total_by_EI['I'] += neurons['I']['total']

        for EI in ['E', 'I']:
            for conn_type in nd.conf.conn_types[EI]:
                try:
                    ccgs = key + (EI, conn_type)
                    n_sig = len(ccgs[0]['inds'])  # Only has one session
                except Exception as e:
                    n_sig = 0
                ref, target = conn_type
                if ref == target:
                    total_by_conntype[EI][conn_type] += n[ref] * (n[ref] - 1)
                else:
                    total_by_conntype[EI][conn_type] += n[ref] * n[target]
                results[EI][conn_type]['sig_conv'] += n_sig
                results[EI][conn_type]['list'].append(n_sig)

    overview_str = f"||______name_______||_sig___|_mean__|_std___|_mean/0|_std/0_||_EI____|_%_____||ref-tgt|_%_____||\n"
    for EI in ['E', 'I']:
        for conn_type in nd.conf.conn_types[EI]:
            typename = f"{conn_type[0]}-{conn_type[1]}/{EI}"
            ls = np.array(results[EI][conn_type]['list'])
            tEI = total_by_EI[EI]
            tConn = total_by_conntype[EI][conn_type]
            sig = results[EI][conn_type]['sig_conv']
            mean = np.mean(ls)
            std = np.std(ls)
            print(ls)
            meanN0 = np.mean(ls[ls != 0])
            stdN0 = np.std(ls[ls != 0])
            pEI = sig / tEI * 100
            pConn = sig / tConn * 100
            d = {
                f"total_{EI}": tEI,
                f"total_{conn_type[0]}_{conn_type[1]}": tConn,
                'sig_conv': sig,
                'list': ls,
                f'total_{EI}_percentage': pEI,
                f'total_{conn_type[0]}_{conn_type[1]}_percentage': pConn,
            }
            results[EI][conn_type] = d
            overview_str += f"|| {typename:>15} || {sig:>5} | {mean:5.2f} | {std:5.2f} | {meanN0:5.2f} | {stdN0:5.2f} || {tEI:>5} | {pEI:5.2f} || {tConn:>5} | {pConn:5.2f} || \n"
    print(overview_str)
    return results
