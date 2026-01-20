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
from neuropy.analyses.utils import _san, Config, AnalysisDataset, Savable
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


@dataclass(frozen=True)
class CCGData(Savable):
    """
    Stores the whole CCG array and its p values
    """
    key: Key
    ccg: np.ndarray
    ccg_null: np.ndarray
    pval: np.ndarray
    qval: np.ndarray # TODO
    conn_strength: np.ndarray

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

    def _set_cs_tail(self, acgs:np.ndarray, nspks:list, norm_factor:np.ndarray=False): #TODO untested
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

    def deconv_autocorr(self, acgs:np.ndarray, nspks, target=True, ref=True):
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
    
    def filter(self,min_n_segment,skips):
        if skips is not None: 
            inds = [int(i) for i,(v,e) in enumerate(zip(self.inds,self.significant)) if 
                    not ((v[0] in skips[:,0]) & (v[1] in skips[:,1])) 
                    and (np.sum(e)>=min_n_segment)]
        else:
            inds = np.where(np.sum(self.significant,axis=1)>=min_n_segment)[0].astype(int)
        return inds
    
    @property
    def conn_strength_change(self):
        r=self.conn_strength
        return (np.polyfit(range(r.shape[0]), r, 1)[0] if len(r.shape)>1 else np.full(r.shape,np.nan))


class CCGPointer(Savable):
    """
    Collection of CCGs
    ccg         [N, Np, Nbins]
        N = number of data segments
        Np = number of neuron pairs
        Nbins = number of bins per CCG 
    ccg_null    [N, Np, Nbins]
    inds        [Np, 2]

    """
    # TODO 3 class CCGraw() sig, 
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
    def __init__(self, key, inds, 
                 conf:CCGConfig=None,significant=True,
                 edge_times=None):
        self.key=key
        self.inds=_san(inds,as_np=True)
        self.edge_times=edge_times
        self.frates=None
        self.conf=conf
        n=self.inds.shape[0]
        if type(significant)==bool:
            self.significant=np.full((n,),significant)
        else:
            # assert significant.shape==(n,)#TODO
            self.significant=significant

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
        s='CCG Pointer\n'
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
    def n(self):
        return self.ccg.shape[-2]

    @property
    def n_segments(self):
        return self.ccg.shape[0]
    
    def get_segment(self,segment:int)->'CCGPointer':
        idx = np.where(self.inds[:, 0] == segment)[0]
        return CCGPointer(
            key=self.key.add(segment=segment),
            inds=self.inds[idx][:,1:],
            edge_times=self.edge_times.iloc[segment],
        )

    def split(self)->list['CCGPointer']:
        return [self.get_segment(i) for i in range(self.edge_times.shape[0])]


class CCGIndexSource(Enum):
    SIGNIFICANT=0
    SPURIOUS=1
    SIGNIFICANT_ANY=2


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
    data: dict[CCGPointer] = {}
    spurious: dict[CCGPointer] = {}
    __ccg: dict[CCGData] = {}
    conf: CCGConfig
    nd: NeuronsDataset
    
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

    def _attr_append(self, base_key: Key, inputs: Dict[Key, CCGPointer], attrname:str='data'):
        getattr(self, attrname).update({Key(**{**base_key.__dict__, **k.__dict__}): v for k, v in inputs.items()})

    def merge_CCGs(self, merge_levels):
        pass

    def split_CCG(self):
        pass

    @property
    def filepath(self):
        return ''
        
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

            ccg = correlations.spike_correlations(
                    neurons=neurons,
                    neuron_inds=np.arange(neurons.n_neurons), # all
                    bin_size=self.conf.bin_size,
                    window_size=self.conf.duration,
                    use_acceleration=self.conf.use_acceleration,
                    symmetrize=self.conf.symmetrize_ccg,
                    edge_times=edge_times,
                )

            pvals, pred, qvals, ccg_inds, spur_inds, printstr = conv.eranconv(neurons_key=key, 
                                        ccg=ccg, 
                                        edge_times=edge_times, 
                                        peak_channels=neurons.peak_channels,
                                        shank_ids=neurons.shank_ids,
                                        neuron_type=neurons.neuron_type,
                                        conf=self.conf)

            ccg_data = CCGData(key=key,
                      ccg=ccg,
                      ccg_null=pred,
                      pval=pvals,
                      qvals=qvals,
                      conn_strength=None)
            
            self._attr_append(key, ccg_data, '__ccg')       
            self._attr_append(key, ccg_inds, 'data')
            self._attr_append(key, spur_inds, 'spurious')

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

        self.data[key] = CCGPointer(key=key
                                    inds=inds,
                                    edge_times=self.data[key].edge_times,
                                    conf=self.data[key].conf,
                                    )
        
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
        self.get_ccg()
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
                        ccg.significant[:,i_ext] = old_ccg[k].significant[i_old].T
                    else:
                        ccg.significant = self._ccg[k].significant.T
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
        for key, ccg in self._ccg.items():
            k=key.nd()
            if ccg is None: continue # no connection
            if method=="eran_conv":
                ccg._set_cs_eranconv()
            elif method=="tail":
                spikecount = self.nd[k].n_spikes
                #TODO acg
                ccg._set_cs_tail(acg,spikecount)
                return NotImplementedError("Unknown connection strength method")


class EranConv:
    """
    A device for running EranConv and other significance tests    
    """

    _pval_corrected=[]
    _qval_corrected=[]
    _qval_corrected2=[]
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
    
    def spkcount_mask(self, ccg):
        min_bin = self.conf.min_spkcnt_bin
        max_bin = self.conf.max_spkcnt_bin
        threshold = self.conf.min_spkcount
        pair_inds = np.argwhere((ccg[...,min_bin:max_bin]>=threshold).all(axis=-1)) 
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

    def _cell_type_mask(self,pair_inds,neuron_type,conn_types):
        sig_pairs = {}
        # Conn types with no pairs are marked with None
        if pair_inds.shape[0]==0:
            for ct in conn_types: sig_pairs[ct]=None

        # Condition 3: Ref/Target are specific cell types
        for ct in conn_types:
            inds=np.where(np.isin(pair_inds[:,-2],np.where(neuron_type==ct[0])) & 
                                np.isin(pair_inds[:,-1],np.where(neuron_type==ct[1])))[0]
            sig_pairs[ct]=pair_inds[inds] if inds.shape[0] else None
        return sig_pairs
    
    def _probe_loc_mask(self,pair_inds,peak_channels,shank_ids):
        # Condition 2: Ref/Target are not too close by
        ignore = self.conf.ignore
        x,y = pair_inds[:,-2],pair_inds[:,-1]
        if ignore.name==IgnoreLevel.SAME_CHANNEL.name:
            assert peak_channels is not None
            inds=np.where(peak_channels[x]!=peak_channels[y])[0]
        elif ignore.name==IgnoreLevel.SAME_SHANK.name:
            assert shank_ids is not None
            inds=np.where(shank_ids[x]!=shank_ids[y])[0]
        return pair_inds[inds]

    def eranconv(self, neurons_key:Key, ccg, edge_times:pd.DataFrame,
                       peak_channels, shank_ids, neuron_type, conf:CCGConfig,):
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
        print("running eranconv (1st pass)")
        key = neurons_key
        self.conf = conf
        self.n_segments = edge_times.shape[0]

        pvals, pred, qvals = EranConv._conv(ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        def _hasvalue(x):
            return x is not None and x.size > 0

        def build_inds(p, EI, conn_types):
            rough_inds = intersect(self.significance_mask(p, EI), self.spkcount_mask(ccg))
            inds = self._autocorr_mask(rough_inds) if _hasvalue(rough_inds) else None
            inds = self._probe_loc_mask(inds, peak_channels, shank_ids) if _hasvalue(inds) else None
            inds = self._cell_type_mask(inds, neuron_type, conn_types) if _hasvalue(inds) else None
            return rough_inds, inds

        # [n_seg, n_pair, 2]
        rough_inds_E, inds_E = build_inds(pvals, 'E', conf.conn_types_E)
        rough_inds_I, inds_I = build_inds(qvals, 'I', conf.conn_types_I)

        ccg_inds_by_type, spur_inds_by_type = {},{}

        # Force CCG to be 4D
        if ccg.ndim==3: 
            ccg = ccg[None]
            pred = self._pred[None]
            for attr in ("_pval_corrected",
                         "_qval_corrected",
                         "_qval_corrected2"):
                setattr(self, attr, getattr(self, attr)[None])
        
        # Update return values
        for EI in ['E','I']:
            all_inds = inds_E if EI=='E' else inds_I
            spurious = rough_inds_E if EI=='E' else rough_inds_I # initialize spurious pairs

            for conn_type, inds in all_inds.items():
                    ccg_key = key.add(conn_type=conn_type, excitability=EI)
                    ccg_inds_by_type[ccg_key] = CCGPointer(
                                            key=ccg_key,
                                            conf=self.conf, 
                                            inds=inds if _hasvalue(inds) else None, 
                                            edge_times=edge_times
                                            )
                    spurious = setdiff(spurious,inds) # remove these pairs from spurious

            spur_key = key.add(excitability=EI)
            spur_inds_by_type[spur_key] = CCGPointer(
                                key=spur_key,
                                conf=self.conf, 
                                inds=spurious if _hasvalue(spurious) else None,
                                edge_times=edge_times
                                )

        count = np.zeros((len(self.conf.conn_types_flat),edge_times.shape[0]))
        for i,T in enumerate(self.conf.conn_types_flat):
            for j,ccg in enumerate(ccg_inds_by_type[T].split()):
                count[i,j]=ccg.n if ccg is not None else 0
        
        printstr=''
        for i, N_per_segment in enumerate((count.T)):
            label = edge_times['label'][i]
            N_totalE = (rough_inds_E[:,0]==i).sum()
            N_totalI = (rough_inds_I[:,0]==i).sum()
            printstr += f"{label}: E/I pairs {N_totalE:03d} / {N_totalI:03d} | "
            for EI in ['E','I']:
                for N,(ref,target) in zip(N_per_segment,self.conf.conn_types[EI]):
                    printstr += f"{ref}-{target}/{EI} {f'{N:02d}' if N>0 else '-'} | "

        print("eranconv (1st pass) done")

        return pvals, pred, qvals, ccg_inds_by_type, spur_inds_by_type, printstr


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

