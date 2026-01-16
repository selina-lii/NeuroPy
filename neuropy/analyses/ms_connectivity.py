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
from neuropy.analyses.utils import _san, Config, AnalysisDataset
from neuropy.core.neurons import Neurons
from scipy.signal import windows
from scipy.stats import poisson, ttest_ind, ttest_1samp, describe
from scipy import ndimage
from typing import Union, Optional, Dict, Any, Tuple
import h5py
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field, replace
from collections import defaultdict
from enum import Enum
from copy import deepcopy
import imageio
import neuropy.plotting.ccg as plot_ccg
import pandas as pd


# TODO
CHANNELS_PER_SHANK=16
SAVE_ROOT="~/Documents/Neuropy/outputs"

class IgnoreLevel(Enum):
    """Config for CCG and other analysis
    Do we ignore neurons on the same peak channel / shank?"""
    NONE = 0
    SAME_CHANNEL = 1
    SAME_SHANK = 2


class NormalizeBy(Enum):
    """Config for CCG and other analysis
    Do we ignore neurons on the same peak channel / shank?"""
    NONE = 0
    REF_FRATE = 1
    REF_SPKS = 2
    TARGET_FRATE = 3
    TARGET_SPKS = 4 
    BOTH_FRATE = 5
    BOTH_SPKS = 6


def _short_session_name(session):
    """
    Get a printable session name in the format of ANIMAL_DayX
    """
    sess_name = session.filePrefix.parts[-1].split('_')[:2]
    sess_name='_'.join(sess_name)
    return sess_name


def example(var:dict):
    """
    Get an example from a dictionary
    """
    k,v = next(iter(var.items()))
    return v


def __set_op(x,y,f):
    """
    Perform set operation of two N-dim arrays by their row elements.
    x,y: np.ndarray of shape [...,k]
    ravel_dims: (d1,...,dk), each d is sufficiently large

    Ravels row values to v = v1*d1+...+vn*dn for comparison and then conver back
   
    """
    ax=tuple(np.arange(len(x.shape)-1))
    ravel_dims=np.max(np.vstack([x.max(axis=ax),y.max(axis=ax)]),axis=0)
    xr, yr = np.ravel_multi_index(x.T, ravel_dims), np.ravel_multi_index(y.T, ravel_dims)
    res = f(xr, yr)
    return np.array(np.unravel_index(res, ravel_dims)).T


def intersect(x, y):
    """
    Intersect two N-dim arrays by their row elements
    """
    if x is None or y is None: return np.array([])
    return __set_op(x,y,np.intersect1d)


def setdiff(x, y):#n2=None
    """
    X minus Y for two N-dim arrays by their row elements
    """
    # Set difference of coordinate lists
    if x is None or y is None: return x if x is not None else np.array([])
    return __set_op(x,y,np.setdiff1d)


def union(x, y):#n2=None
    """
    Union two N-dim arrays by their row elements
    """
    # Set difference of coordinate lists
    if x is None: return y if y is not None else np.array([])
    elif y is None: return x if x is not None else np.array([])
    return __set_op(x,y,np.union1d)


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
    conn_type: Optional[tuple[str,str]] = None

    def __str__(self):
        parts = [
            f"sess_{self.session}" if self.session else "",
            f"epoch_{self.epoch}" if self.epoch else "",
            f"ref_{self.ref_ind}" if self.ref_ind is not None else "",
            f"tgt_{self.target_ind}" if self.target_ind is not None else "",
            f"seg_{self.segment}" if self.segment is not None else "",
            f"ex_{self.excitability}" if self.excitability else "",
            f"type_{self.conn_type[0]}-{self.conn_type[1]}" if self.conn_type else ""
        ]
        return ".".join(filter(None, parts)) or "root"
        
    def __eq__(self,other):
        try:
            return all(getattr(self, f) == getattr(other, f) for f in self.__dataclass_fields__)
        except:
            return False # NOTE no type check, allows comparison with other 'key' classes

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
 
    def remove(self,*dimensions) -> 'Key':
        return Key(**{f: getattr(self, f) for f in self.__dataclass_fields__ if f not in dimensions})

    def add(self, **kwargs) -> 'Key':
        for k, v in kwargs.items(): assert getattr(self, k) is None, f"{k} is not None"
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
    def __init__(self,labels:Union[list[str], str, None]=None,min_dur=0,discard=False):
        self.labels=_san(labels)
        self.min_dur=min_dur
        self.discard=discard

    def __str__(self):
        return self.__class__ + ': ' + '\n'.join([f"{key}={val}" for key, val in self.__dict__.items()])


class NeuronsDatasetConfig(Config):
    """
    Metadata of NeuronsDataset

    tight_time: bool
    if true, try to shrink start and end of epoch to where brainstates are happening 

    n_segments: int
    Splits session time axis into equal-lengthed blocks if >1

    """
    def __init__(self,
                 name:str = "default",
                 neuron_types:Union[list[str], str, None]=['pyr','inter'], 
                 epochs:Union[list[str], str, None]=['pre','maze','post','re-maze'], 
                 sleep:Union[EpochSlicingConfig, None]=None, 
                 ripple:Union[EpochSlicingConfig, None]=None, 
                 n_segments:Union[list[int], int]=None, 
                 seg_stride:int=None,
                 seg_len:int=None,
                 seg_spikecount:int=None,
                 zero_spike_times=False,
                 recinfo:NeuroscopeIO=None):
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
            assert len(self.n_segments)==len(self.epochs)

        # self.sleep = EpochSlicingConfig(labels=['REM','NREM'],
        #                           min_dur=120,
        #                           discard=False,)
        # self.ripple = EpochSlicingConfig(min_dur=0,
        #                           discard=True,)

    # def get_segs_from_epoch(self,epoch):
    #     idx = self.epochs.index(epoch)
    #     return self.n_segments[idx]


class CCGConfig(Config):
    """
    All the details of CCG computation need to be predefined, kinda like a batch run
    """
    def __init__(self, 
                name="default",
                conn_types_E:Union[list[list], list]=[('pyr','pyr'), ('pyr','inter')],
                conn_types_I:Union[list[list], list]=[('inter','inter'), ('inter','pyr')],
                duration:float = 20e-3,
                bin_size:float = 1e-3,
                conv_window:float = 5e-3,
                alpha:float = 0.05,
                alpha2:float = 0.1,
                min_lag:float = 1e-3,
                max_lag:float = 3e-3,
                min_spkcount = 2.5,
                spkcount_scope = 12e-3,
                multiple_correction_method:str = None,
                ignore:IgnoreLevel = IgnoreLevel.SAME_CHANNEL,
                use_acceleration = True,
                symmetrize_ccg = True,
                normalize:NormalizeBy = NormalizeBy.NONE,
                ):
        self.name = name

        self.conn_types_E = conn_types_E
        self.conn_types_I = conn_types_I
        self.duration = duration
        self.bin_size = bin_size
        self.conv_window = conv_window
        self.alpha = alpha
        self.alpha2 = alpha2
        self.use_multiple_correction = multiple_correction_method is not None
        self.mc_method = multiple_correction_method
        self.center_bin = int(self.duration/self.bin_size//2)
        self.nbins = int(self.duration/self.bin_size)+1 # NOTE

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_spkcount = min_spkcount
        self.spkcnt_scope = spkcount_scope
        self.spkcnt_bins = int(self.spkcnt_scope/self.bin_size)
        self.ignore = ignore

        self.min_lag_bin = self.center_bin+int(self.min_lag/self.bin_size) # leftmost bin for p value test
        self.max_lag_bin = self.center_bin+int(self.max_lag/self.bin_size)+1 # rightmost bin for p value test
        self.min_spkcnt_bin = self.center_bin-self.spkcnt_bins//2 # leftmost bin requiring minimum spike count 
        self.max_spkcnt_bin = self.center_bin+self.spkcnt_bins//2+1 # rightmost bin requiring minimum spike count

        self.use_acceleration = use_acceleration
        self.symmetrize_ccg = symmetrize_ccg

        self.normalize = normalize

        # if self.use_multiple_correction: 
        #     self.corrected_alpha=alpha/(n**2-n)/self.nbins # local threshold
        #     self.corrected_alpha2=alpha2/(n**2-n)/self.nbins

    def __str__(self):
        s=""
        for key,val in self.__dict__.items():
            s+=f"{key}: {val}\n"
        s+=f"config file: {self.filepath}\n"
        return s
        
    @property
    def conn_types(self):
        return {'E':self.conn_types_E, 
                'I':self.conn_types_I}
    @property
    def conn_types_flat(self):
        return self.conn_types_E+self.conn_types_I
    
    @property
    def conv_window_bins(self):
        return self.conv_window/self.bin_size
    
    def time2bin(self,x):
        """time in SECONDS to bin#"""
        return x/self.bin_size
    
    def bin2time(self,x):
        """bin# to time in SECONDS"""
        return x*self.bin_size

    @property
    def filepath(self):
        return f"{SAVE_ROOT}/{self.name}.ccg.meta.h5"
    
    def save(self):
        with h5py.File(self.filepath, "w") as f:
            for k, v in self.__dict__.items():
                try:
                    f.create_dataset(k, data=v)
                except TypeError:
                    f.attrs[k] = str(v)  # fallback for non-array data

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)  # bypass __init__
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                obj.__dict__[k] = np.array(v)
            for k, v in f.attrs.items():
                obj.__dict__[k] = v
        return obj


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
    def __init__(self, sessions, conf:NeuronsDatasetConfig):
        self._conf = conf        
        self.prep(sessions)

    def __str__(self): 
        s=''
        cnt=0
        for k,v in self.data.items():
            s+=f"{k}\t{str(v)}"
            cnt+=1
        return f"NeuronsDataset #sessions = {cnt}\n{s}"
        
    def prep(self, sessions):
        """
        Filter neurons by behavioral epochs and type 
        Set segment edge timing data
        """
        self.data={}
        self.edge_times=defaultdict(pd.DataFrame)
        c = self.conf

        sessions = _san(sessions)
        for s in sessions:
            name = _short_session_name(s)
            key = Key(session=name)
            self.conf.session_names.append(name)
            neurons = s.neurons
            neurons.metadata['intervals']=np.array([[neurons.t_start,neurons.t_stop]])

            # Filter neurons
            if c.neuron_types is not None:
                neurons = neurons.get_neuron_type(c.neuron_types)

            if c.epochs is not None:
                neurons = neurons.behav_slice(behav_times=s.paradigm, 
                                        labels=c.epochs)
                
            if c.sleep is not None:
                neurons = neurons.behav_slice(behav_times=s.brainstates, 
                                        labels=c.sleep.labels, 
                                        discard=c.sleep.discard,
                                        min_dur=c.sleep.min_dur)
                
            if c.ripple is not None:
                neurons = neurons.behav_slice(behav_times=s.ripple, 
                                        labels=None, 
                                        discard=c.ripple.discard,
                                        min_dur=c.ripple.min_dur)
            
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
            """
            
            dfs=[]
            ivs = neurons.metadata['intervals']
            for i, e in enumerate(_san(c.epochs,wrap_none=True)):
                k = key.add(epoch=e)
                t = s.paradigm.timing_by_label(e) if e else (neurons.t_start, neurons.t_stop)
                t_start, t_stop = t
                
                if c.seg_spikecount is not None:
                    # TODO spikecount segmentation code is not maintained
                    neus = neurons.time_slice(t_start,t_stop)
                    for i in range(neus.n_neurons):
                        k = key.add(epoch=e, ref_ind=i)
                        starts, stops = neus._edges_spikecount(i=i,
                                                    n=c.seg_spikecount,
                                                    discard_tail=False)
                elif c.seg_stride is not None and c.seg_len is not None:
                    starts, stops = neurons._edges_time_window(stride=c.seg_stride, 
                                                                seg_len=c.seg_len,
                                                                t_start=t_start,
                                                                t_stop=t_stop) 
                elif c.n_segments is not None and c.n_segments[i] > 1:
                    starts, stops = neurons._edges_time_split(n_segments=c.n_segments,
                                                                t_start=t_start,
                                                                t_stop=t_stop)
                else:
                    starts, stops = np.array([t_start]),np.array([t_stop])


                """
                Calculate total/actual time lengths of each segment
                """
                edges = pd.DataFrame({"start": starts,
                                    "stop":  stops,
                                    "total_time_hours": (stops-starts)/3600,
                                    "key": [k.add(segment=i) for i in range(len(starts))],
                                    "epoch": [e for i in range(len(starts))],
                                    })
                #TODO does not work for spikecount edges yet
                
                eths = []
                for row in edges.itertuples(index=False):
                    start, stop, tth = row.start, row.stop, row.total_time_hours

                    # find intervals that overlap the edge
                    overlap_mask = (ivs[:,1] > start) & (ivs[:,0] < stop)
                    overlapping_ivs = ivs[overlap_mask]

                    # clip intervals to edge boundaries
                    clipped_start = np.clip(overlapping_ivs[:,0], start, stop)
                    clipped_stop  = np.clip(overlapping_ivs[:,1], start, stop)

                    # compute effective time in hours
                    effective_hours = np.sum(clipped_stop - clipped_start) / 3600
                    eths.append(min(effective_hours, tth))
                edges['effective_time_hours']=np.array(eths)
                dfs.append(edges)
            self.edge_times[key] = pd.concat(dfs, axis=0)

            if c.zero_spike_times:
                neurons = neurons.zero_spike_times()
                for _, v in self.edge_times.items(): v[['start','stop']] -= neurons.t_start

            # Store filtered neurons
            self.data[key] = neurons

    def frate_stats(self):
        @dataclass
        class FrateStat:
            def __init__(self,n_neurons,neu_type):
                self.n_neurons=n_neurons
                self.neu_type=neu_type
                self.keys=[]
                self.frates=[]
                self.effective_time=[]
                self.n_neurons=[]
                self.neu_type=[]
                self.i=[]
            
            def append(self,key,frates,effective_time,n_neurons,i):
                self.keys.append(key)
                self.frates.append(frates)
                self.effective_time.append(effective_time)
                self.n_neurons.append(n_neurons)
                self.i.append(i)

            @property
            def iqr(self):
                return np.percentile(self.frates, 75)-np.percentile(self.frates, 25)
            
            @property
            def describe(self):
                return [describe(self.frates)]
            
            def __str__(self):
                overview_str=""
                overview_str+=f"{self.i}. {self.neu_type}\t"
                overview_str+=f"n={int(self.n_neurons)}\t"
                overview_str+=f"mean firing rates (Hz)|effective time (h)\n"
                for ts,mfrs in zip(self.effective_time,self.mean_firing_rates):
                    for t,mfr in zip(ts,mfrs):
                        overview_str+=f"{mfr:.02f}|{t:.02f}  "
                overview_str+="\n"
                if self.n_neurons<2:
                    overview_str+="Too few neurons in this category\n"
                return overview_str

        print("Mean firing rates P VALUES")
        stats = {}
        for k,edges in self.edge_times.items():
            for epoch in self.conf.epochs:
                overview_str = f"======={k.session}-{epoch}=======\n"
                neurons=self.data[k.nd()]
                for i,neu_type in enumerate(self.conf.neuron_types):
                    neus = neurons.get_neuron_type(neu_type)
                    for s,e in zip(edges[0],edges[1]):
                        _neus = neus.time_slice(s,e)
                        if _neus.n_neurons>0:
                            stats[k] = FrateStat(
                                i=i,
                                neu_type=neu_type,
                                frates=_neus.firing_rate,
                                effective_time = _neus.effective_time_hours,
                                n_neurons=_neus.n_neurons,
                                )
                        else:
                            stats[k] = FrateStat(n_neurons=0)
                    labels += [f"{epoch.capitalize()}{i+1}" for i in range(len(edges[0]))]
        if neus.n_neurons>5:
            pass
        return stats


                # decimal_places=int(2+-np.floor(np.log10(alpha)))
                # frates = [xx for x in frates for xx in x]
                # flag = False
                # for j in range(total_n_segments):
                #     for k in range(j):
                #         p = ttest_ind(frates[k],frates[j],equal_var=True).pvalue
                #         if p<alpha:
                #             flag = True
                #             overview_str+=f"{labels[k]} VS SLEEP{labels[j]}\tp={p:.{decimal_places}f}\n"
                #         # Standard t-test,  check if mean firing rate changes per cell type
                # if not flag: overview_str+="No significant difference between segments\n"            


class ACG:
    """Like Neurons, but for auto-correlograms"""
    def __init__(self, key, acg, inds,
                 conf:CCGConfig=None):
        self.key=key
        self.inds=inds
        self.acg=acg
        self._conf=conf


class CCG:
    """
    Collection of CCGs
    ccg         [N, Np, Nbins]
        N = number of data segments
        Np = number of neuron pairs
        Nbins = number of bins per CCG 
    ccg_null    [N, Np, Nbins]
    inds        [Np, 2]

    """
    # TODO
    # TODO 1 CCGs are np.float16
    # TODO 2 CCG of a session is saved locally in hdf5
    # TODO 3 class CCGraw() stores the whole CCG array and its p values, with metadata of CCGConfig, 
    #  contains slicing functions
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
    def __init__(self, key, ccg, inds, 
                 ccg_null=None, pval=None, conn_strength=None,
                 conf:CCGConfig=None,significant=True,
                 edge_times=None):
        self.key=key
        self.inds=_san(inds,as_np=True)
        self.ccg=ccg
        self.ccg_null=ccg_null # 'baseline' chance level CCG
        self.pval=pval
        self.conn_strength = conn_strength # 
        self.conf=conf
        self.edge_times=edge_times
        self.frates=None
        n=self.inds.shape[0]
        if type(significant)==bool:
            self.significant=np.full((n,),significant) # by default, all ccgs in this variable are significant
        else:
            # assert significant.shape==(n,)#TODO
            self.significant=significant
#def __init__(self, exist:np.ndarray[bool], strength:np.ndarray[float]):

    def set_firing_rates(self, neurons:Neurons):
        # Obtain the firing rates during the time period used to compute this CCG
        if (self.edge_times is not None) and (neurons.firing_rate is not None):
            n_seg, n_pairs = self.ccg.shape[0], self.inds.shape[0]
            self.frates = np.zeros((n_seg,n_pairs,2))
            for i,(t_start, t_end) in enumerate(zip(self.edge_times[0], self.edge_times[1])):
                neu=neurons.time_slice(t_start,t_end)
                self.frates[i] = neu.firing_rate[self.inds] #TODO other cases

    def __repr__(self):
        s = str(self.key) + "\n"
        for i,inds in enumerate(self.inds):
            s += f"{str(inds):<15}\tIn segments {str(np.where(self.significant[i])[0]):<20}\tstrengths: {self.conn_strength[i]}\n"
        return s
        
    def __str__(self):
        s='CCG connectivity\n'
        for key, val in self.__dict__.items():
            if isinstance(val,np.ndarray) or isinstance(val,list):
                s+=f"{key}\tshape={np.array(val).shape}"
                sval="\n".join(str(val[0:2]).splitlines()[:3])
                s+=f"\tval={sval}...\n"
            elif isinstance(val,dict):
                k,v = next(iter(val.items()))
                s+=f"{key} dict keys={k}...\n"
                item_str = str(v)
                for line in item_str.splitlines()[:3]:
                    s+=f"\t\t{line}\n"
            elif key!='_conf':
                sval="\n".join(str(val).splitlines()[:3])
                s+=f"{key}: {sval}\n"
        return s
    
    @property
    def ref_inds(self):
        return self.inds[:,-2]
    
    @property
    def target_inds(self):
        return self.inds[:,-1]
    
    @property
    def ref_ind(self):
        return self.inds[-2]
    
    @property
    def unique_inds(self):
        return np.unique(self.inds)
    
    @property
    def total(self):
        return self.ccg.shape[-2]

    @property
    def conn_strength_change(self):
        r=self.conn_strength
        return (np.polyfit(range(r.shape[0]), r, 1)[0] if len(r.shape)>1 else np.full(r.shape,np.nan))

    @property
    def n_segments(self):
        return self.ccg.shape[0]
    
    def _set_cs_eranconv(self, norm_factor:np.ndarray=None):
        """
        Connection strength:
        
            Area under CCG curve minus baseline, within temporal ROI
            The ROI is by default the same as the interval tested for peak/trough signficance

        Can be negative
        """
        auc = self.ccg-self.ccg_null # area under curve
        cs = np.sum(auc[...,self.conf.min_lag_bin:self.conf.max_lag_bin],axis=-1) # (inds,)
        if norm_factor is not None: 
            shape=cs.shape
            cs = cs.astype(float).reshape(shape) / norm_factor # e.g. presynaptic element firing rate
        self.conn_strength = cs.squeeze() # divided by presynaptic firing rate

    def _set_cs_eranconv_compound(self, norm_factor:float=None):
        """
        Connection strength:
        
            Area under CCG curve minus baseline, within temporal ROI
            The ROI is by default the same as the interval tested for peak/trough signficance

        Can be negative
        """
        auc = self.ccg-self.ccg_null # area under curve
        cs = np.sum(auc[:,:,self.conf.min_lag_bin:self.conf.max_lag_bin],axis=(1,2)) # (inds,)
        if norm_factor is not None: cs = cs.astype(float) / norm_factor # e.g. presynaptic element firing rate
        self.conn_strength = cs # divided by presynaptic firing rate

    def _set_cs_tail(self, acgs:ACG, nspks:list, norm_factor:np.ndarray=False): #TODO untested
        """
        Connection strength:

                Area under CCG curve minus a 'tailed' baseline after deconvolving autocorrelograms
        
        Can be negative
        """
        self.deconv_autocorr(acgs, nspks, target=True, ref=True)
        self._set_baseline_by_tail()
        auc = self.ccg-self.ccg_null # area under curve
        cs = np.sum(auc[self.conf.min_lag_bin:self.conf.max_lag_bin]) # inds,nbins 
        if norm_factor: cs /= norm_factor
        self.conn_strength = cs
    
    def _set_baseline_by_tail(self):
        """
        Baseline by 'tail' (|t|>11ms)
        'Tail' is accurate when coupled with deconv_autocorr
        """
        l = self.conf.time2bin(-11e-3)
        r = self.conf.time2bin(11e-3)
        baseline = np.mean([self.ccg[:l],self.ccg[r+1:]])
        self.ccg_null = np.ones_like(self.ccg)*baseline

    def plotdir(self, root):
        root=os.path.expanduser(root)
        if self.key.conn_type is None:
            return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.segment}/{self.key.excitability}_any"
        return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.segment}/{self.key.excitability}_{self.key.conn_type[0]}-{self.key.conn_type[1]}"    

    def save_plots(self, ids, neuron_types, shank_ids, waveforms, frates_all, root, discarded_channels=None,ch_per_shank=None):
        assert self.ccg is not None
        plotdir = self.plotdir(root)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir,exist_ok=True)

        idx_dict = {n: i for i, n in enumerate(np.unique(self.inds))}
        s=np.argsort(self.inds[:,-2]) #[np.random.random_integers(0,inds.shape[0]-1,5)]
        n_seg=self.ccg.shape[0]

        if len(self.ccg.shape)==3:
            for i,inds in enumerate(self.inds[s]):
                figs = []
                ymin,ymax=[],[]
                print(i,inds)
                idx=np.array([idx_dict[inds[0]],idx_dict[inds[1]]])
                for i_seg in range(n_seg):
                    fig = plot_ccg.plot_ccg_figure(ids=ids,
                                    inds=inds,
                                    neuron_types=neuron_types[idx],
                                    frates_cut=self.frates[i_seg][s][i],
                                    frates_all=frates_all[idx],
                                    waveforms=waveforms[idx] if waveforms is not None else None,
                                    shank_ids=shank_ids[idx],
                                    discarded_channels=discarded_channels,
                                    ch_per_shank=ch_per_shank,
                                    ccg=self.ccg[i_seg][s][i], 
                                    plotdir=plotdir, 
                                    window_size=self.conf.duration*1e3,
                                    bin_size=self.conf.bin_size*1e3,
                                    pval=self.pval[i_seg][s][i] if self.pval is not None else None,
                                    ccg_null=self.ccg_null[i_seg][s][i] if self.ccg_null is not None else None,
                                    significant=self.significant[i_seg][s][i] if self.significant else None,
                                    show=False,save=False,
                                    segment_id=i_seg)
                    _ymin, _ymax =fig.axes[0].get_ylim()
                    ymin.append(_ymin);ymax.append(_ymax)
                    figs.append(fig)
                ymin=min(ymin);ymax=max(ymax)
                frames=[]
                for fig in figs:
                    fig.axes[0].set_ylim(ymin, ymax)
                    fig.canvas.draw_idle()
                    frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
                    plt.close(fig)
                imageio.mimsave(
                    f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.gif",
                    frames,
                    duration=0.5
                )
        print("done saving plots")

    def deconv_autocorr(self, acgs:ACG, nspks, target=True, ref=True):
        """
        Remove auto-correlograms (ACG) from CCG
        target/ref is set to true if corresponding ACG is to be removed
        """

        def _deconv_autocorr(ccg, acg1=None, nspks1=None, acg2=None, nspks2=None):
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
            assert m%2==1 # CCG must have an odd number of bins
            hw = (m - 1) // 2 # midpoint
            
            # Scale acg1
            acg1 = (acg1 - np.mean(acg1)) / nspks1  # remove mean of clipped, divide by nspks1
            hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])  # [0:hw, (hw+1):m]
            acg1[hw] = 1 - np.sum(acg1[hidx])  # set zero-lag bin s.t. sum of 1
            den = np.fft.fft(acg1)

            if acg2 is not None:
                # Scale acg2
                acg2 = (acg2 - np.mean(acg2)) / nspks2  # remove mean of clipped, divide by nspks2
                hidx = np.concatenate([np.arange(hw), np.arange(hw + 1, m)])
                acg2[hw] = 1 - np.sum(acg2[hidx])  # set zero-lag bin s.t. sum of 1
                den = den * np.fft.fft(acg2)

            # Deconvolve acgs from the ccg
            dcccg = np.real(np.fft.ifft(np.fft.fft(ccg) / den))
            
            # Set my CCG to deconvCCG
            dcccg = np.concatenate([dcccg[1:], [dcccg[0]]])  # shift DC to end
            dcccg[dcccg < 0] = 0  # clip negatives to zero
            return dcccg
    
        for i,(ref,tgt) in enumerate(self.inds):
            if ref and target:
                self.ccg[i] = _deconv_autocorr(self.ccg[i], acgs[ref], nspks[ref], acgs[tgt], nspks[tgt])
            elif ref:
                self.ccg[i] = _deconv_autocorr(self.ccg[i], acgs[ref], nspks[ref])
            elif target:
                self.ccg[i] = _deconv_autocorr(self.ccg[i], acgs[tgt], nspks[tgt])
            else:
                Warning("_deconv_autocorr: No effect")
                return

    def normalize_by_ref_rate(self):
        shape=self.ccg.shape
        frates=self.frates[...,-2][...,np.newaxis] 
        self.ccg = self.ccg.astype(float).reshape(shape) / frates
        self.ccg_null = self.ccg_null.astype(float).reshape(shape) / frates

    def normalize_by_target_rate(self):
        shape=self.ccg.shape
        frates=self.frates[...,-1][...,np.newaxis] 
        self.ccg = self.ccg.astype(float).reshape(shape) / frates
        self.ccg_null = self.ccg_null.astype(float).reshape(shape) / frates        
    
    def filter(self,min_n_segment,skips):
        if skips is not None: 
            inds = [int(i) for i,(v,e) in enumerate(zip(self.inds,self.significant)) if 
                    not ((v[0] in skips[:,0]) & (v[1] in skips[:,1])) 
                    and (np.sum(e)>=min_n_segment)]
        else:
            inds = np.where(np.sum(self.significant,axis=1)>=min_n_segment)[0].astype(int)
        return inds


class CCGIndexSource(Enum):
    SIGNIFICANT=0
    SPURIOUS=1
    SIGNIFICANT_ANY=2


# TODO two storage views? one w excitability in mind, the other view is all CCGs.
# flat view is memory intensive. 
class CCGDataset(AnalysisDataset):
    """
    Data and operations on CCGs from an experiment
    Requires a NeuronsDataset to be processed first, and a configuration object (see CCGConfig)
    Test CCGs and stores them separately by significance criteria
    """
    def __init__(self, nd:NeuronsDataset, conf:CCGConfig=None):
        self.nd = nd # neurons
        self.data={} # CCGs of interest
        self._conf=conf or CCGConfig() # config
        self.spurious={} # rest of pairwise CCG that failed the significance checks
        self.auto={} # autocorrelograms 
        self.connectivity={}

    def get_ccg(self, method="eran_conv"):
        """
        main function of the class
        """
        if method=="eran_conv":
            self._ccg_eranconv()
        elif method=="jitter":
            NotImplementedError("CCG jitter must be run in the Jitter object, " \
            "since it generates a ton of extra data. Nothing is run...")
        else:
            ValueError("Unknown method")
    
    def filter_excitability(self, E):
        return self.filter(excitability=E)

    def filter_ref(self, ref):
        return {k: v for k, v in self.data.items() if k.conn_type and k.conn_type[0] == ref}

    def filter_target(self, target):
        return {k: v for k, v in self.data.items() if k.conn_type and k.conn_type[1] == target}

    def _attr_append(self, base_key: Key, inputs: Dict[Key, CCG], attrname:str='data'):
        getattr(self, attrname).update({Key(**{**base_key.__dict__, **k.__dict__}): v for k, v in inputs.items()})

    def merge_CCGs(self, merge_levels):
        # TODO
        groups=self.groupby(*merge_levels)
        # self.data={}
        for merge_key, grouped_ccgs in groups.items():
            # [dev] ccg list items must share a common key that indexes the neurons dataset
            n_bins = example(grouped_ccgs).ccg.shape[-1]
            n_items = len(grouped_ccgs)
            keys = list(grouped_ccgs.keys())

            inds = np.unique(np.concatenate([c.inds for _,c in grouped_ccgs.items()],axis=0),axis=0)
            ids = self.nd.data[merge_key.nd()].ind2id(inds)
            n_uniqinds=inds.shape[0]
            # TODO 
            # PICK UP HERE!!!!! i was debugging merge_CCGs, combining CCGs of different epochs into one
            # sort_inds should be by row; make row matching a helper function
            sort_inds = [np.where(np.isin(c.inds, inds))[0]  for _,c in grouped_ccgs.items()]
            ccg=np.full((n_uniqinds,n_items,n_bins),np.nan) # what if ccg list has none items..
            pval=np.full_like(ccg,np.nan)
            significant=np.full_like(ccg,np.nan)
            conn_strength=np.full((n_uniqinds,n_items),np.nan)
            significant=np.full_like(conn_strength,np.nan)
            # i need to define how jsig and significant are different, or give them different names

            for i,(_,c) in enumerate(grouped_ccgs.items()):
                if c.ccg is not None: ccg[sort_inds[i],i,:]=c.ccg
                if c.pval is not None: pval[sort_inds[i],i,:]=c.pval
                if c.significant is not None: significant[sort_inds[i],i,:]=c.significant
                if c.conn_strength is not None: conn_strength[sort_inds[i],i]=c.conn_strength
                if c.significant is not None: significant[sort_inds[i],i]=c.significant
            
            # check if there's any non-nan values
            pass
            
            ccg = CCG(key=merge_key,
                ccg=ccg,
                ids=ids,
                inds=inds,
                pval=pval,
                significant=significant,
                conn_strength=conn_strength,
                conf=grouped_ccgs[0].conf,)
            ccg.keys = keys
            self.data[merge_key]=ccg

    def split_CCG(self):
        pass

    @property
    def filepath(self):
        return ''
        
    def save(self):
        self.conf.save()
        with h5py.File(self.filepath, "w") as f:
            for k in ['ccg','spurious']:
                grp = f.create_group(k)
                for kk, vv in self.__dict__[k].items():
                    grp.create_dataset(str(kk), data=vv)

    def load(self):
        try:
            self.conf.load()
            with h5py.File(self.filepath, "r") as f:
                for k in ['ccg','spurious']:
                    grp = f[k]
                    self.__dict__[k] = {kk: grp[kk][:] for kk in grp.keys()}
        except Exception as e:
            print(f"Load failed: {e}")

    def _ccg_eranconv(self):
        """
        Run CCG and convolution based significance test for all neurons

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        print("EranConv significant pairs")
        def _s(key, neurons, total_time_hours, effective_time_hours): # print helper
            s = f"======={key.session}-{key.epoch}=======\n"
            s+=f"Segment(s) are {total_time_hours[0]:.2f}h each "
            if self.nd.conf.sleep is not None:
                s+=f"and contain {[f'{_:.2f}' for _ in effective_time_hours]} hours of actual sleep "
            for _ in self.nd.conf.neuron_types:
                s+=f"{_}={neurons.get_neuron_type(_).n_neurons} "
            s+="\n"
            return s

        conv = EranConv()
        for key, edge_times in self.nd.edge_times.items():
            neurons = self.nd.data[key.nd()]
            if edge_times is None: 
                edge_times = [np.array([neurons.t_start]),np.array([neurons.t_stop])]

            ccgs, spurs, autos, printstr = conv.eranconv_split(key=key, 
                                        neurons=neurons, 
                                        edge_times=edge_times, 
                                        conf=self.conf)
            
            self._attr_append(key, ccgs, 'data')
            self._attr_append(key, spurs, 'spurious')
            self._attr_append(key, autos, 'auto')

            tt=self.nd.total_time_hours[key]
            et=self.nd.effective_time_hours[key]
            print(_s(key,neurons,tt,et)+printstr)

    def _reCCG(self,key:Key,
               indices_source=CCGIndexSource.SIGNIFICANT,
               inds=None,
               ):
        """
        Rerun CCG given list of indices

        Call one of the wrappers instead
        """
        edge_times=None
        
        # groups=self.groupby('session','epoch',source='connectivity')
        if inds is not None:
            pass

        elif key.ref_ind is not None:
            # REMOVE Key(session=key.session,epoch=key.epoch,ref_ind=key.ref_ind,segment=key.segment)
            edge_times = self.nd.edge_times[key.remove('segment')]

        elif indices_source==CCGIndexSource.SIGNIFICANT:
            inds = self.data[key].inds

        elif indices_source==CCGIndexSource.SIGNIFICANT_ANY:
            # group by
            inds = self.data[key].inds

        elif indices_source==CCGIndexSource.AUTOCORRELOGRAMS:
            inds = np.vstack([self.auto[key].inds,self.auto[key].inds]).T    

        conv = EranConv()
        ccg_dict = conv.eranconv_merge(key=key, 
                            neurons=self.nd.data[key.nd()], # session, epoch, conn_type
                            pair_inds=inds,
                            edge_times=edge_times or self.nd.edge_times[key.get('session','epoch')], 
                            conf=self.conf)
        return ccg_dict

    def reCCG_timescale(self,bin_size,duration=None,jscale=None):
        """
        Run CCG and convolution based significance test for all neurons

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        # change timescale in my configurations.
        old_bin_size = self._conf.bin_size
        self._conf.bin_size = bin_size
        if duration: self._conf.duration = duration
        if jscale: self._conf.jscale = jscale
        print(f"recalculated CCG from binsize={old_bin_size} to binsize={bin_size}")

        for key, ccg in self.data.items():
            ccg_dict = self._reCCG(indices_source=CCGIndexSource.SIGNIFICANT,key=key)
            for k,ccg in ccg_dict.items():
                    # inherit original significant markers
                ccg.significant = self.data[key].exist
                self.data[k] = ccg
        print("rescale completed")

    def reCCG_pair_inds_1conntype(self,conn_type=('pyr','pyr'),external_inds=None,_new_data=None,_group=None):
        """
        modifies _new_data
        """
        SET_SELF_DATA=_new_data is None
        if _group is None:
            _group = self.groupby('conn_type',source='connectivity')[Key(conn_type=conn_type)]
        if _new_data is None:
            _new_data={}
            
        for key, old_ccg in _group.items():
            if old_ccg is not None:
                new_ccgs = self._reCCG(indices_source=CCGIndexSource.SIGNIFICANT_ANY,
                                       inds=external_inds,
                                       key=key)
                for k,ccg in new_ccgs.items():
                     # TODO inherit original significant markers
                    if external_inds is not None:
                        ccg.significant=np.zeros((old_ccg.n_segments,external_inds.shape[0]))
                        m = (external_inds[:,None] == old_ccg.inds).all(-1)
                        i_ext, i_old = np.where(m)
                        ccg.significant[:,i_ext] = self.data[k].exist[i_old].T
                    else:
                        ccg.significant = self.connectivity[k].exist.T
                    _new_data[k] = ccg

        if SET_SELF_DATA: 
            for k, v in _new_data.items():
                self.data[k]=v

    def reCCG_pair_inds(self,external_inds_by_type:dict[np.ndarray]=None):
        """
        Rerun CCG with a list of pair indices
        The significance of the pairs are set by whether they were significant before the rerun

        overwrites self.data
        """
        new_data = {}
        groups = self.groupby('conn_type',source='connectivity')
        for k, group in groups.items():
            inds = external_inds_by_type.get(k.conn_type) if external_inds_by_type is not None else None
            self.reCCG_pair_inds_1conntype(conn_type=k.conn_type, 
                                              external_inds=inds,
                                              _new_data=new_data,
                                              _group=group)
        self.data = new_data
        print("recomputed CCG for pairs that had been significant in any segment")

    def reCCG_pair_inds_pairwise(self,conn_type=('pyr','pyr')):
        """
        TODO group by reference
        """
        group = self.groupby('conn_type',source='connectivity')[Key(conn_type=conn_type)]
        self.data={}
        for key, ccg in group.items():
            if ccg is not None:
                for i,pair in enumerate(ccg.inds):
                    print(f"reCCG: {i}/{ccg.inds.shape[0]}")
                    # REMOVE Key(session=key.session,epoch=key.epoch,ref_ind=pair[0],target_ind=pair[1],
                    #         segment=key.segment,excitability=key.excitability,conn_type=key.conn_type)
                    new_ccgs = self._reCCG(key=key.add(ref_ind=pair[0],target_ind=pair[1]),
                                           inds=pair)
                    for k,new_ccg in new_ccgs.items():
                        self.data[k] = new_ccg
        print("recomputed CCG using spike count segments")

    def save_plots(self, root="~/Documents/NeuroPy/images/ccg_plots_tmp",
                   conn_types:list=None):
        assert os.path.isdir(os.path.expanduser(root))
        keys = self.filter_keys(conn_type=conn_types) if conn_types else self.data.keys()
        print(keys)
        print(f"Saving plots under {root}")
        for key in keys:
            ccg = self.data[key]
            neurons = self.nd.data[key.nd()]
            inds = ccg.unique_inds

            print(f"ccg {key.session} {key.conn_type}")

            ccg.save_plots(
                neuron_types=neurons.neuron_type[inds],
                waveforms=None if neurons.waveforms is None else neurons.waveforms[inds],
                shank_ids=None if neurons.shank_ids is None else neurons.shank_ids[inds],
                frates_all=None if neurons.firing_rate is None else neurons.firing_rate[inds],
                discarded_channels=self.nd.conf.recinfo.skipped_channels,
                ch_per_shank=self.nd.conf.ch_per_shank,
                root=root,
            )
        print("done")

    def normalize(self):
        for _, ccg in self.data.items():
            if self.conf.normalize.name == NormalizeBy.REF_FRATE.name:
                ccg.normalize_by_ref_rate()
            elif self.conf.normalize.name == NormalizeBy.TARGET_FRATE.name:
                ccg.normalize_by_target_rate()
            elif self.conf.normalize.name == NormalizeBy.BOTH_FRATE.name:
                ccg.normalize_by_ref_rate()
                ccg.normalize_by_target_rate()

    def set_connection_strengths(self, method="eran_conv"):
        """
        Set value for each CCG() object of self.data based on given method.
        Values can be found in self.data[i].conn_strengths.
        """
        #TODO untested
        for key, ccg in self.data.items():
            k=key.nd()
            if ccg is None: continue # no connection
            if method=="eran_conv":
                ccg._set_cs_eranconv()
            elif method=="tail":
                spikecount = self.nd[k].n_spikes
                acg = self.auto[k]
                ccg._set_cs_tail(acg,spikecount)
                return NotImplementedError("Unknown connection strength method")


class EranConv:
    """
    A device for running EranConv and other significance tests    
    """
    rough_mask={} # used for eranconv_split
    mask={}

    _pvals=[]
    _qvals=[]
    _pval_corrected=[]
    _qval_corrected=[]
    _qval_corrected2=[]
    _ccg=[]
    pred=[]
    conf=None

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
        if len(ccg.shape)==1:
            ccg=ccg[np.newaxis,...]

        assert wintype in ["gauss", "rect", "triang"]
        assert W<=ccg.shape[-1]

        # Auto-assign appropriate hollow fraction if not specified
        # generate window
        # get center indices of window
        if wintype == "gauss":
            hollow_frac = hollow_frac or 0.6
            sigma = W/2
            W = int(6*sigma + (2 if W%2 else 1))
            center = int(3*sigma + (0.5 if W%2 else 0))
            window = windows.gaussian(W,std=sigma)/(2*np.pi*sigma)
        elif wintype == "rect":
            hollow_frac = hollow_frac or 0.42
            if W%2==0: W+=1
            center = W//2
            window = windows.boxcar(W)
        elif wintype == "triang":
            hollow_frac = hollow_frac or 0.63
            W = 2*W+(-1 if W%2 else 1)
            center = W//2
            window = windows.triang(W)

        # hollow and normalize window
        window[center]*=(1-hollow_frac)
        window /= np.sum(window)
        # padding
        ccg_pad=np.concatenate([ccg[...,:W][...,::-1],ccg,ccg[...,-W:][...,::-1]],axis=-1)

        # convolve window with ccg
        pred = ndimage.convolve1d(ccg_pad, window, axis=-1)
        pred=pred[...,W:-W]

        # mid-p Poisson test: P( val<=pred ) + half of P ( val==pred )
        pvals = 1 - poisson.cdf(ccg-1, pred) - poisson.pmf(ccg, pred)*0.5
        qvals = 1 - pvals
        return pvals, pred, qvals

    @staticmethod
    def get_autocorr_locations(shape):
        """
        Genearte a mask of autocorrelation locations shaped (ngroups, nneurons, nneurons)
        """
        n_auto = min(shape[-3], shape[-2])
        auto_mask = np.eye(n_auto, dtype=bool)
        auto_mask = np.pad(auto_mask, ((0,shape[-3]-n_auto),(0,shape[-2]-n_auto)))
        autocorr_locations = np.broadcast_to(auto_mask, shape[:-1])
        return autocorr_locations
    
    @staticmethod
    def multiple_correction(pvals,alpha,method='bonferroni'): # correct for number of bins
        # NOTE should bump this to utils or something
        # methods: 'fdr_bh', 'bonferroni'
        sig = np.empty_like(pvals, dtype=bool)
        p_corr = np.empty_like(pvals, dtype=float)

        for idx in np.ndindex(pvals.shape[:-3]):
            subarray = pvals[idx]  # shape = last 3 dims
            flat = subarray.ravel()
            s, pc, _, _ = multipletests(flat, alpha=alpha, method=method)
            sig[idx] = s.reshape(subarray.shape)
            p_corr[idx] = pc.reshape(subarray.shape)
        return sig,p_corr
    
    def spkcount_mask(self):
        min_bin = self.conf.min_spkcnt_bin
        max_bin = self.conf.max_spkcnt_bin
        threshold = self.conf.min_spkcount
        pair_inds = np.argwhere((self._ccg[...,min_bin:max_bin]>=threshold).all(axis=-1)) 
        # NOTE right now it's the same criteria for exctiation/inhibition
        return pair_inds
    
    def significance_mask(self,p,excitability):
        conf = self.conf
        if excitability=='E':
            sig, self._pval_corrected = EranConv.multiple_correction(p, conf.alpha)
            pair_inds = np.argwhere((sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1))
        elif excitability=='I':
            sig1, self._qval_corrected = EranConv.multiple_correction(p, conf.alpha)
            sig2, self._qval_corrected2 = EranConv.multiple_correction(p, conf.alpha2)
            neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1))  # significant bins must have a significant-ish neighbor
            pair_inds = np.argwhere(neighbor.any(-1))
        return pair_inds
    
    def _autocorr_mask(self,pair_inds):
        pair_inds = np.array([inds for inds in pair_inds if inds[-2] != inds[-1]])
        return pair_inds

    def _cell_type_mask(self,pair_inds,neurons,conn_types):
        sig_pairs = {}
        # Conn types with no pairs are marked with None
        if pair_inds.shape[0]==0:
            for ct in conn_types: sig_pairs[ct]=None

        # Condition 3: Ref/Target are specific cell types
        for ct in conn_types:
            inds=np.where(np.isin(pair_inds[:,-2],np.where(neurons.neuron_type==ct[0])) & 
                                np.isin(pair_inds[:,-1],np.where(neurons.neuron_type==ct[1])))[0]
            sig_pairs[ct]=pair_inds[inds] if inds.shape[0] else None
        return sig_pairs
    
    def _probe_loc_mask(self,pair_inds,neurons):
        # Condition 2: Ref/Target are not too close by
        ignore = self.conf.ignore
        x,y = pair_inds[:,-2],pair_inds[:,-1]
        if ignore.name==IgnoreLevel.SAME_CHANNEL.name:
            assert neurons.peak_channels is not None
            inds=np.where(neurons.peak_channels[x]!=neurons.peak_channels[y])[0]
        elif ignore.name==IgnoreLevel.SAME_SHANK.name:
            assert neurons.shank_ids is not None
            inds=np.where(neurons.shank_ids[x]!=neurons.shank_ids[y])[0]
        return pair_inds[inds]
    
    # Update print string 
    def _printstr_sig(self, pairs_dict, EI, s=""):
        # if any type of connection under consideration has a non-zero count, print a summary
        nonempty = any(v is not None for v in pairs_dict.values())
        if nonempty:
            for (ref,target), pairs in pairs_dict.items():
                s+=f"{ref}-{target}/{EI} {f'{pairs.shape[0]:02d}' if pairs is not None else '-'} | "
        else:
            s=f"no {'excitatory' if EI=='E' else 'inhbitory'} connections  "
        return s, nonempty

    def eranconv_split(self, key:Key, neurons:Neurons, edge_times:pd.DataFrame, conf:CCGConfig,):
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
        print("running eranconv (1st pass)")
        self.conf = conf
        self.n_segments = len(edge_times['start'])

        self._ccg = correlations.spike_correlations(
                neurons=neurons,
                neuron_inds=np.arange(neurons.n_neurons), # all
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                edge_times=edge_times,
            )

        pvals, self._pred, qvals = EranConv._conv(self._ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        def _hasvalue(x):
            return x is not None and x.size > 0

        def build_inds(p, EI, conn_types):
            rough_inds = intersect(self.significance_mask(p, EI), self.spkcount_mask())
            inds = self._autocorr_mask(rough_inds) if _hasvalue(rough_inds) else None
            inds = self._probe_loc_mask(inds, neurons) if _hasvalue(inds) else None
            inds = self._cell_type_mask(inds, neurons, conn_types) if _hasvalue(inds) else None
            return rough_inds, inds

        # [n_seg, n_pair, 2[]
        rough_inds_E, inds_E = build_inds(pvals, 'E', conf.conn_types_E)
        rough_inds_I, inds_I = build_inds(qvals, 'I', conf.conn_types_I)

        def process_output(key:Key, neurons:Neurons):
            """
            Post processor organizing eranconv outputs into mergeable formats for CCGDataset
            """

            def regroup_masks():
                groups = defaultdict(lambda: defaultdict(list))
                for i in range(self.n_segments): 
                    for EI in ['E','I']:
                        groups[i][EI]=[]
                for EI in ['E','I']:
                    for row in rough_mask[EI]:
                        groups[row[0]][EI].append(row[1:])
                for i in range(self.n_segments): 
                    for EI in ['E','I']:
                        groups[i][EI]=np.array(groups[i][EI])
                rough_mask = groups

                groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                for i in range(self.n_segments): 
                    for EI in ['E','I']:
                        for conn_type in self.conf.conn_types[EI]:
                            groups[i][EI][conn_type]=[]
                for EI in ['E','I']:
                    for conn_type, pair_vals in self.mask[EI].items():
                        if pair_vals is None: continue
                        for row in pair_vals:
                            groups[row[0]][EI][conn_type].append(row[1:])
                for i in range(self.n_segments): 
                    for EI in ['E','I']:
                        for conn_type in self.conf.conn_types[EI]:
                            groups[i][EI][conn_type]=np.array(groups[i][EI][conn_type])
                self.mask = groups
            regroup_masks()

            ccgs_by_type,spurs_by_type,acgs = {},{},{} # 1 neuron group -> many connection types

            if len(self._ccg.shape)==3: 
                self._ccg=self._ccg[np.newaxis,...]
                self._pred=self._pred[np.newaxis,...]
                self._pval_corrected=self._pval_corrected[np.newaxis,...]
                self._qval_corrected=self._qval_corrected[np.newaxis,...]

            # Update return values
            for seg in range(self.n_segments):
                for EI in ['E','I']:
                    pairs=self.mask[seg][EI]
                    spairs = rough_mask[seg][EI] # initialize spurious pairs
                    p = self._pval_corrected if EI=='E' else self._qval_corrected # TODO TODO not storing corrected p-vals. make it an option!

                    for conn_type, prs in pairs.items():
                        new_key = key.add(segment=seg if self.n_segments>1 else None,
                                conn_type=conn_type,
                                excitability=EI)
                        if prs is None or len(prs)==0: ccgs_by_type[new_key] = None
                        else:
                            x,y = prs[:,-2],prs[:,-1]
                            ccgs_by_type[new_key] = CCG(key=new_key,conf=self.conf, 
                                                inds=prs, 
                                                ccg=self._ccg[seg,x,y], 
                                                ccg_null=self._pred[seg,x,y], 
                                                pval=p[seg,x,y], 
                                                edge_times=edge_times
                                                )
                            spairs = setdiff(spairs,prs) # remove these pairs from spurious

                    new_key = key.add(segment=seg if self.n_segments>1 else None,
                                      excitability=EI)

                    if isinstance(spairs, np.ndarray) and spairs.any():
                        x,y = spairs[:,-2],spairs[:,-1] # x,y = pairs[...,0],pairs[...,1]
                        spurs_by_type[new_key] = CCG(key=new_key,conf=self.conf, 
                                            inds=spairs, ids=neurons.ind2id(spairs), 
                                            ccg=self._ccg[seg,x,y], ccg_null=self._pred[seg,x,y], pval=p[seg,x,y], 
                                            )
                    else:
                        spurs_by_type[new_key] = None

            autocorr_locations = EranConv.get_autocorr_locations(self._ccg.shape)            
            new_key = key
            acgs[new_key] = ACG(key=new_key,
                        acg=self._ccg[:,autocorr_locations[0]],
                        inds=np.arange(neurons.n_neurons),
                        ids=neurons.ind2id(np.arange(neurons.n_neurons)),
                        conf=self.conf)

            return ccgs_by_type, spurs_by_type, acgs
        ccgs_by_type, spurs_by_type, acgs = process_output(key, neurons)

        overview_str=""
        for i in range(self.n_segments):
            E_str, hasE = self._printstr_sig(self.mask[i]['E'], 'E')
            I_str, hasI = self._printstr_sig(self.mask[i]['I'], 'I')
            overview_str += f"SLEEP{i}: E/I pairs {rough_mask[i]['E'].shape[0]:03d} / {rough_mask[i]['I'].shape[0]:03d} | "
            overview_str=overview_str+E_str+I_str+"\n" if (hasE or hasI) else overview_str+"no connections\n"
        print("eranconv (1st pass) done")

        return ccgs_by_type, spurs_by_type, acgs, overview_str

    def eranconv_merge(self, key:list[Key], neurons:Neurons, pair_inds:dict[list], edge_times:np.ndarray, conf:CCGConfig):
        # ref and target indices should be organized by conn type
        print(f"running eranconv (2nd pass): {key}")

        self.conf = conf
        self.n_segments = len(edge_times[0])

        ccg_dict = {}
        neuron_inds = np.unique(pair_inds)
        self._ccg = correlations.spike_correlations(
                neurons=neurons,
                neuron_inds=neuron_inds,
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                edge_times=edge_times,
            )
        idx = {n: i for i, n in enumerate(neuron_inds)}
        slicer = (...,
                [idx[a] for a in pair_inds[:, -2]],
                [idx[b] for b in pair_inds[:, -1]],
                slice(None))
        self._ccg=self._ccg[slicer]

        self._pvals, self._pred, self._qvals = EranConv._conv(self._ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        if key.excitability=='E':
            sig, self._pval_corrected = EranConv.multiple_correction(self._pvals, conf.alpha)
            sig_mask = (sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1)
        elif key.excitability=='I':
            sig1, self._qval_corrected = EranConv.multiple_correction(self._qvals, conf.alpha)
            sig2, self._qval_corrected2 = EranConv.multiple_correction(self._qvals, conf.alpha2)
            neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1))  # significant bins must have a significant-ish neighbor
            sig_mask = neighbor.any(-1)

        k=key.remove('segment')
        ccg_dict[k]=CCG(inds=pair_inds, 
                ids=neurons.ind2id(pair_inds), 
                ccg=self._ccg, 
                ccg_null=self._pred, 
                pval=self._pval_corrected if key.excitability=='E' else self._qval_corrected, 
                conf=self.conf,
                significant=sig_mask,
                key=k,
                edge_times=edge_times)
        ccg_dict[k].set_firing_rates(neurons)

        print("done")
        return ccg_dict


def routine_eranconv_connection_info(info, nd:NeuronsDataset, cd:CCGDataset, epoch_id=0):
    """
    Print aggregated information of eranconv_pairs() outputs

    info: eranconv_pairs outputs
    """
    results = {'E':{},'I':{}}
    total_by_conntype = {'E':{},'I':{}}
    total_by_EI = {'E':0,'I':0}
    for EI in ['E','I']:
        for conn_type in cd.conf.conn_types[EI]:
            results[EI][conn_type]={'sig_conv':0,'list':[]}
            total_by_conntype[EI][conn_type] = 0
        
    neuron_types = nd.conf.neuron_types
    epoch = nd.conf.epochs[epoch_id]

    for key,neurons in nd.data.items():        
        n = {}
        for _ in neuron_types: 
            n[_] = neurons.get_neuron_type(_).n_neurons
                
        total_by_EI['E'] += neurons['E']['total']
        total_by_EI['I'] += neurons['I']['total']

        for EI in ['E','I']:
            for conn_type in nd.conf.conn_types[EI]:
                try:
                    ccgs = key+(EI,conn_type)
                    n_sig=len(ccgs[0]['inds']) # Only has one session
                except Exception as e:
                    n_sig=0
                ref,target=conn_type
                if ref==target:
                    total_by_conntype[EI][conn_type] += n[ref]*(n[ref]-1)
                else:
                    total_by_conntype[EI][conn_type] += n[ref]*n[target]
                results[EI][conn_type]['sig_conv']+=n_sig
                results[EI][conn_type]['list'].append(n_sig)
    
    overview_str = f"||______name_______||_sig___|_mean__|_std___|_mean/0|_std/0_||_EI____|_%_____||ref-tgt|_%_____||\n"
    for EI in ['E','I']:
        for conn_type in nd.conf.conn_types[EI]:
                typename = f"{conn_type[0]}-{conn_type[1]}/{EI}"
                ls = np.array(results[EI][conn_type]['list'])
                tEI = total_by_EI[EI]
                tConn = total_by_conntype[EI][conn_type]
                sig = results[EI][conn_type]['sig_conv']
                mean = np.mean(ls)
                std = np.std(ls)
                print(ls)
                meanN0 = np.mean(ls[ls!=0])
                stdN0 = np.std(ls[ls!=0])
                pEI = sig/tEI*100
                pConn = sig/tConn*100
                d = {f"total_{EI}": tEI,
                    f"total_{conn_type[0]}_{conn_type[1]}": tConn,
                    'sig_conv': sig,
                    'list': ls,
                    f'total_{EI}_percentage': pEI,
                    f'total_{conn_type[0]}_{conn_type[1]}_percentage': pConn,
                    }
                results[EI][conn_type]=d
                overview_str += f"|| {typename:>15} || {sig:>5} | {mean:5.2f} | {std:5.2f} | {meanN0:5.2f} | {stdN0:5.2f} || {tEI:>5} | {pEI:5.2f} || {tConn:>5} | {pConn:5.2f} || \n"
    print(overview_str)
    return results

