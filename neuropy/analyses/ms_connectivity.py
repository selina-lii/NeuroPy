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



# TODO
CHANNELS_PER_SHANK=16

def _short_session_name(session):
    """get short printable session name in the format of ANIMAL_DayX"""
    sess_name = session.filePrefix.parts[-1].split('_')[:2]
    sess_name='_'.join(sess_name)
    return sess_name


def _san(var,as_np=False):
    """Sanitize """
    if var is None: return var
    if not isinstance(var, list): var = [var]
    if as_np: var = np.array(var)
    return var


def example(var:dict):
    """Get an example from a dictionary"""
    k,v = next(iter(var.items()))
    return v

@dataclass(frozen=True)
class Key:
    session: Optional[str] = None
    epoch: Optional[str] = None
    ref_ind: Optional[int] = None
    target_ind: Optional[int] = None
    segment: Optional[int] = None
    excitability: Optional[str] = None
    conn_type: Optional[tuple[str,str]] = None

    """

    key.session      # Top level
    key.epoch        # Middle level  
    key.segment          # Finest time division
    key.conn_type    # Connection properties
    key.ref_ind      # Source neuron
    key.target_ind   # Target neuron

    Dependencies
    tuple(session, epoch, ... , segment) should alway be present
    conn_type -> excitability
    ref_ind hyperfocus on the relations between ONE neuron and the rest of the population.
        Do not set a ref_ind unless focusing on analysis relative to one reference neuron
        All other neurons in a NeuronDataset are targets; remove irrelevant neurons before constructing NeuronDataset.
    """

    def __str__(self):
        parts = []
        if self.session: parts.append(f"sess_{self.session}")
        if self.epoch: parts.append(f"epoch_{self.epoch}")
        if self.ref_ind is not None: parts.append(f"ref_{self.ref_ind}")
        if self.target_ind is not None: parts.append(f"tgt_{self.target_ind}")
        if self.segment is not None: parts.append(f"seg_{self.segment}")
        if self.excitability: parts.append(f"ex_{self.excitability}")
        if self.conn_type: parts.append(f"type_{self.conn_type[0]}-{self.conn_type[1]}")
        return ".".join(parts) if parts else "root"
    
    def __eq__(self,other):
        try:
            return all(getattr(self, f) == getattr(other, f) for f in self.__dataclass_fields__)
        except:
            return False # TODO no type check. allows comparison with other 'key' classes

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


class AnalysisDataset:
    """
    Container for all analysis data with flexible indexing
    """
    def __init__(self):
        self.data: Dict[Key, Any] = {}
    
    # def __setitem__(self, key: Key, value: Any):
    #     """Store data with a key"""
    #     self.data[key] = value
    
    # def __getitem__(self, key: Key) -> Any:
    #     """Retrieve data by key"""
    #     return self.data[key]
    
    # def get(self, key: Key, default=None) -> Any:
    #     """Safe retrieval with default"""
    #     return self.data.get(key, default)
    
    def example(self,field=None) -> Any:
        """Get an example from data or another field"""
        if field:
                item = next(iter(self.__dict__[field].keys()))
                return self.__dict__[field].get(item,None)
        return self.data.get(next(iter(self.data.keys())),None)

    def filter(self, **criteria) -> Dict[Key, Any]:
        """
        Filter data by any combination of key attributes.
        
        Example:
            dataset.filter(session='s1', epoch='pre')
            dataset.filter(analysis_type='correlogram', neuron_type='pyramidal')
        """
        return {k: v for k, v in self.data.items() if k.matches(**criteria)}
    
    def keys_matching(self, **criteria) -> list[Key]:
        """Get all keys matching criteria"""
        return [k for k in self.data.keys() if k.matches(**criteria)]
    
    def group_by(self, *dimensions, source='data') -> Dict[Key, Dict[Key, Any]]:
        """
        Group data by specified dimensions.
        
        Example:
            dataset.group_by('session', 'epoch')
            # Returns: {('s1', 'pre'): {key: data, ...}, ('s1', 'post'): {...}}
        """
        items = getattr(self, source)
        groups = defaultdict(dict)
        for key, value in items.items():
            # group_key = tuple(getattr(key, dim, None) for dim in dimensions)
            groups[key.get(*dimensions)][key] = value
        return dict(groups)
    
    def iter_sessions(self):
        """Iterate over unique sessions"""
        sessions = {k.session for k in self.data.keys() if k.session}
        for session in sorted(sessions):
            yield session, self.filter(session=session)
    
    def iter_epochs(self, session: Optional[str] = None):
        """Iterate over epochs, optionally filtered by session"""
        criteria = {'session': session} if session else {}
        epochs = {k.epoch for k in self.data.keys() if k.epoch and k.matches(**criteria)}
        for epoch in sorted(epochs):
            yield epoch, self.filter(epoch=epoch, **criteria)
        
    def __len__(self):
        return len(self.data)
    
    @property
    def keys(self):
        for k in self.data.keys():
            print(k)

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self,conf):
        ans = input("Clear all datafields with the new config? [y/n]").lower()
        if ans=='y':
            self.data={}
            self.spurious={}
            self.auto={}
            self.connectivity={}
            print(f'{self.__class__.__name__}: all data fields are cleared')
        self._conf = conf
        print(f"{self.__class__.__name__}Config changed, which might create inconsistencies between existing data and config. Rerun if necessary.")


class EpochSelect:
    def __init__(self,labels:Union[list[str], str, None]=None,min_dur=0,discard=False):
        self.labels=_san(labels)
        self.min_dur=min_dur
        self.discard=discard

    def __str__(self):
        return self.__class__ + ': ' + '\n'.join([f"{key}={val}" for key, val in self.__dict__.items()])


class NeuronsDatasetConfig:
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
                 sleep:Union[EpochSelect, None]=None, 
                 ripple:Union[EpochSelect, None]=None, 
                 n_segments:Union[list[int], int]=None, 
                 seg_stride:int=None,
                 seg_len:int=None,
                 seg_spikecount:int=None,
                 zero_spike_times=False,
                 recinfo:NeuroscopeIO=None):
        self.name = name
        self.session_names = []
        self.neuron_types = _san(neuron_types)
        self.sleep = sleep
        self.ripple = ripple
        self.n_segments = _san(n_segments)
        self.seg_stride = seg_stride
        self.seg_len = seg_len
        self.seg_spikecount = seg_spikecount
        self.zero_spike_times = zero_spike_times
        self.recinfo = recinfo

        self.ch_per_shank = 16 

        self.epochs = epochs
        if epochs is not None and n_segments is not None:
            assert len(self.n_segments)==len(self.epochs)

        # self.sleep = EpochSelect(labels=['REM','NREM'],
        #                           min_dur=120,
        #                           discard=False,)
        # self.ripple = EpochSelect(min_dur=0,
        #                           discard=True,)

    def __str__(self):
        s=""
        for key, val in self.__dict__.items():
            s+=f"{key}: {val}\n"
        s+=f"config file: {self.filepath}\n"
        return s

    @property
    def filepath(self):
        return f"~/Documents/jitter_out/{self.name}.nd.meta.h5"

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

    def get_segs_from_epoch(self,epoch):
        idx = self.epochs.index(epoch)
        return self.n_segments[idx]


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


class CCGConfig:
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
        return f"~/Documents/jitter_out/{self.name}.ccg.meta.h5"
    
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


class JitterType(Enum):
    SPIKE_TIMING = 0
    INTERVAL = 1


class JitterConfig:
    def __init__(self, ccg:CCGConfig,njitter:int=100, jitter_type:JitterType=JitterType.INTERVAL, jscale:float=5e-3, alpha:float=.05, use_acceleration=True,):
        """
        Parameters
        ----------
        njitter : int
            number of jitters
        jscale: float
            maximum spiking time shift in seconds (default is +-5ms)
        use_acceleration: bool, optional
            whether or not to use gpu acceleration
        """
        self.njitter = njitter
        self.jitter_type = jitter_type
        self.jscale = jscale
        self.alpha = alpha
        self.use_acceleration = use_acceleration
        self.ccg = ccg

    def __str__(self):
        return f"njitter:{self.njitter}, jitter_type:{self.jitter_type}, jscale:{self.jscale}, p:{self.alpha}, use_acceleration:{self.use_acceleration}"

    @property
    def jscale_ms(self):
        return self.jscale*1e3
    
    @property
    def jscale_bins(self):
        return self.jscale/self.ccg.bin_size


class NeuronsDataset(AnalysisDataset):
    """
    A collection of neurons wrapped for analysis
    Arguments of the analysis should be provided using a NeuronsDatasetConfig object

    sessions: subjects.ProcessData
        collection object of sessions
    """
    def __init__(self, sessions, conf:NeuronsDatasetConfig):
        self._conf = conf        
        self.data={}
        self.edge_times=defaultdict(lambda: defaultdict(list))
        self.effective_time_hours=defaultdict(lambda: defaultdict(list))
        self.total_time_hours=defaultdict(lambda: defaultdict(list))
        self.prep(sessions)

    def __str__(self): 
        s=''
        cnt=0
        for k,v in self.data.items():
            s+=f"{k}\t{str(v)}"
            cnt+=1
        return f"NeuronsDataset #sessions = {cnt}\n{s}"
    
    @property
    def conf(self):
        return self._conf
    
    def prep(self, sessions):
        c = self.conf
        sessions = _san(sessions)
        for s in sessions:
            name = _short_session_name(s)
            key = Key(session=name)
            self.conf.session_names.append(name)
            neurons = s.neurons
            neurons.metadata['intervals']=np.array([[neurons.t_start,neurons.t_stop]])

            # Filter neurons
            # -type
            if self.conf.neuron_types is not None:
                neurons = neurons.get_neuron_type(c.neuron_types)
            # -paradigm label
            if self.conf.epochs is not None:
                neurons = neurons.behav_slice(behav_times=s.paradigm, 
                                        labels=c.epochs)
            # -sleep states
            if self.conf.sleep is not None:
                neurons = neurons.behav_slice(behav_times=s.brainstates, 
                                        labels=c.sleep.labels, 
                                        discard=c.sleep.discard,
                                        min_dur=c.sleep.min_dur)
            # -ripple states
            if self.conf.ripple is not None:
                neurons = neurons.behav_slice(behav_times=s.ripple, 
                                        labels=None, 
                                        discard=c.ripple.discard,
                                        min_dur=c.ripple.min_dur)

            """
            Get the start and end of each segment. A segment is the smallest time period in the 
            dataset where analysis will be performed (e.g. data used to calculate one CCG). There 
            can be many overlapping segments within a dataset depending on configuration.
            
            Define segment edges of each neurons group
            see neurons.py: 
                _edges_time_split   time_split
                _edges_time_window  time_windows
                _edges_spikecount   spikecount_split
            """
            
            for i, e in enumerate(_san(c.epochs) or [None]):
                k = key.add(epoch=e)
                t = s.paradigm.timing_by_label(e) if e else (neurons.t_start, neurons.t_stop)
                t_start, t_stop = t
                
                if c.seg_spikecount is not None:
                    neus = neurons.time_slice(t_start,t_stop)
                    for i in range(neus.n_neurons):
                        k = key.add(epoch=e, ref_ind=i)
                        edges = neus._edges_spikecount(i=i,
                                                    n=c.seg_spikecount,
                                                    discard_tail=False)
                        self.edge_times[k]=edges 
                elif c.seg_stride is not None and c.seg_len is not None:
                    self.edge_times[k] = neurons._edges_time_window(stride=c.seg_stride, 
                                                                seg_len=c.seg_len,
                                                                t_start=t_start,
                                                                t_stop=t_stop) 
                elif c.n_segments is not None and c.n_segments[i] > 1:
                    self.edge_times[k] = neurons._edges_time_split(n_segments=c.n_segments,
                                                                t_start=t_start,
                                                                t_stop=t_stop)
                else:
                    self.edge_times[k] = [np.array([t_start]),np.array([t_stop])]
                
                # Calculate total/actual time length of each segment
                #TODO does not work for spikecount edges yet
                intervals = neurons.metadata['intervals']
                seg_edges=self.edge_times[k]
                effective_time_hours, total_time_hours = [],[]
                for t_start, t_stop in zip(seg_edges[0],seg_edges[1]):
                    iv = intervals[(intervals[:,1]>=t_start)&(intervals[:,0]<=t_stop)]
                    tth = (t_stop-t_start)/3600
                    total_time_hours.append(tth)
                    effective_time_hours.append(min(tth,np.sum(iv[:,1]-iv[:,0])/3600))
                self.effective_time_hours[k]=np.array(effective_time_hours)
                self.total_time_hours[k]=np.array(total_time_hours)

            if c.zero_spike_times:
                t_start = neurons.t_start
                neurons = neurons.zero_spike_times()
                for k,v in self.edge_times.items():
                    self.edge_times[k]=(v[0]-t_start,v[1]-t_start)

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
    """Like Neurons, but for auto-correlograms
    Static dataclass, not mean for reuse"""
    def __init__(self, key, acg, ids, inds,
                 conf:CCGConfig=None):
        self.key=key
        self.ids=ids
        self.inds=inds
        self.acg=acg
        self._conf=conf


class CCG:
    """Like Neurons, but for CCGs
    * Static dataclass, not mean for reuse
    * Shouldn't really be called on its own. Wrap in CCGDataset!"""
    # TODO TODO
    # there should be two versions of CCG, but they should share some common methods, 
    #           using sparse and dense matrix representations.
    # filtered: ccg shape = ngroups,nccg,nbins     ids shape = nccg,2
    # flat:     ccg shape = (ngroups),nneu,nneu,nbins   ids shape = nneu
    # since segments will be abolished, ngroups dimension should always exist, even when redundant
    # segments is a dimension such that all items in the list can have the same operations applied on them
    #   as segments are purely for comparison analysis
    def __init__(self, key, ccg, ids, inds, 
                 ccg_null=None, pval=None, j_sig=None, conn_strength=None,
                 conf:CCGConfig=None,significant=True,
                 seg_edges=None):
        self.key=key
        self.ids=ids
        self.inds=inds
        self.ccg=ccg
        self.ccg_null=ccg_null # 'baseline', or jittered, chance level CCG
        self.pval=pval
        self.j_sig=j_sig
        self.conn_strength = conn_strength # 
        self._conf=conf
        self.seg_edges=seg_edges
        self.frates=None
        n=self.inds.shape[0]
        if type(significant)==bool:
            self.significant=np.full((n,),significant) # by default, all ccgs in this variable are significant
        else:
            # assert significant.shape==(n,)#TODO
            self.significant=significant

    def set_firing_rates(self, neurons:Neurons):
        # Obtain the firing rates during the time period used to compute this CCG
        if (self.seg_edges is not None) and (neurons.firing_rate is not None):
            n_seg, n_pairs = self.ccg.shape[0], self.inds.shape[0]
            self.frates = np.zeros((n_seg,n_pairs,2))
            for i,(t_start, t_end) in enumerate(zip(self.seg_edges[0], self.seg_edges[1])):
                neu=neurons.time_slice(t_start,t_end)
                self.frates[i] = neu.firing_rate[self.inds] #TODO other cases

    def __str__(self):
        s=''
        for key, val in self.__dict__.items():
            if isinstance(val,np.ndarray) or isinstance(val,list):
                sval="\n".join(str(val[0:2]).splitlines()[:3])
                s+=f"{key}\tshape={np.array(val).shape}\n\tval={sval}...\n"
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
    def conf(self):
        return self._conf

    @property
    def total(self):
        return self.ccg.shape[-2]

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

    def sample_plot(self, inds):
        assert inds in self.inds
        pval = self.pval[inds] if self.pval else None
        ccg_null = self.ccg_null[inds] if self.ccg_null else None
        j_sig = self.j_sig[inds] if self.j_sig else None
        plot_ccg_only(
            ccg=self.ccg[inds], 
            ids=self.ids[inds], 
            inds=inds, 
            window_size=self.conf.duration, 
            bin_size=self.conf.bin_size, 
            pval=pval, ccg_null=ccg_null, j_sig=j_sig,
        )

    def save_plots(self, neuron_types, shank_ids, waveforms, frates_all, root, discarded_channels=None,ch_per_shank=None):
        assert self.ccg is not None
        plotdir = self.plotdir(root)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir,exist_ok=True)

        idx_dict = {n: i for i, n in enumerate(np.unique(self.inds))}
        s=np.argsort(self.inds[:,-2]) #[np.random.random_integers(0,inds.shape[0]-1,5)]
        n_seg=self.ccg.shape[0]

        if len(self.ccg.shape)==3:
            for i,(inds,ids) in enumerate(zip(self.inds[s],self.ids[s])):
                figs = []
                ymin,ymax=[],[]
                print(i,inds)
                idx=np.array([idx_dict[inds[0]],idx_dict[inds[1]]])
                for i_seg in range(n_seg):
                    fig = plot_ccg_figure(ids=ids,
                                    inds=inds,
                                    neuron_types=neuron_types[idx] if neuron_types is not None else None,
                                    frates_cut=self.frates[i_seg][s][i] if self.frates is not None else None,
                                    frates_all=frates_all[idx] if frates_all is not None else None,
                                    waveforms=waveforms[idx] if waveforms is not None else None,
                                    shank_ids=shank_ids[idx] if shank_ids is not None else None,
                                    discarded_channels=discarded_channels,
                                    ch_per_shank=ch_per_shank,
                                    ccg=self.ccg[i_seg][s][i], 
                                    plotdir=plotdir, 
                                    window_size=self.conf.duration*1e3,
                                    bin_size=self.conf.bin_size*1e3,
                                    pval=self.pval[i_seg][s][i] if self.pval is not None else None,
                                    ccg_null=self.ccg_null[i_seg][s][i] if self.ccg_null is not None else None,
                                    j_sig=self.j_sig[i_seg][s][i] if self.j_sig else None,
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
        else:
            for i,(inds,ids) in enumerate(zip(self.inds[s],self.ids[s])):
                print(i,inds)
                idx=np.array([idx_dict[inds[0]],idx_dict[inds[1]]])
                plot_ccg_figure(ids=ids,
                                inds=inds,
                                neuron_types=neuron_types[idx] if neuron_types is not None else None,
                                frates_cut=self.frates[idx] if self.frates is not None else None,
                                frates_all=frates_all[idx] if frates_all is not None else None,
                                waveforms=waveforms[idx] if waveforms is not None else None,
                                shank_ids=shank_ids[idx] if shank_ids is not None else None,
                                discarded_channels=discarded_channels,
                                ch_per_shank=ch_per_shank,
                                ccg=self.ccg[s][i], 
                                plotdir=plotdir, 
                                window_size=self.conf.duration*1e3,
                                bin_size=self.conf.bin_size*1e3,
                                pval=self.pval[s][i] if self.pval is not None else None,
                                ccg_null=self.ccg_null[s][i] if self.ccg_null is not None else None,
                                j_sig=self.j_sig[s][i] if self.j_sig else None,
                                show=False,save=True)
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


class Connectivity:
    def __init__(self, key, n_segments, ccgs, inds, ids, exist:np.ndarray[bool], strength:np.ndarray[float]):
        self.key=key
        self.inds=_san(inds,as_np=True)
        self.ccgs=ccgs
        self.exist=exist # has to be a separate field, cannot be inferred from strength because a strength might not be significant
        self.strength = strength # 
        self.coordinates = None #TODO physical neuron coordinates
        self.n_segments = n_segments
        self.ids = ids
        
    def __repr__(self):
        s = str(self.key) + "\n"
        for i,inds in enumerate(self.inds):
            s += f"{str(inds):<15}\tIn segments {str(np.where(self.exist[i])[0]):<20}\tstrengths: {self.strength[i]}\n"
        return s
        
    def __str__(self):
        s='Connectivity\n'
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
        return s
    
    def filter(self,min_n_segment,skips):
        if skips is not None: 
            inds = [int(i) for i,(v,e) in enumerate(zip(self.inds,self.exist)) if 
                    not ((v[0] in skips[:,0]) & (v[1] in skips[:,1])) 
                    and (np.sum(e)>=min_n_segment)]
        else:
            inds = np.where(np.sum(self.exist,axis=1)>=min_n_segment)[0].astype(int)
        return inds

    def plot_strength(self,
                    n_segments_threshold=None,
                    norm_by_n_sess=False,
                    norm_by_total_strength=False,
                    zero_first_timepoint=False,
                    show_legend=False,
                    skips=None,
                    save=False,
                    root=None,
                    debug=False):
        # show all pairs by default
        n_segments_threshold=n_segments_threshold if n_segments_threshold is not None else 0
        plt.figure()
        inds = self.filter(min_n_segment=n_segments_threshold,skips=skips)
        plot_data = self.strength[inds]
        significant=self.exist[inds]
        n_significant=np.sum(significant,axis=1,keepdims=True)
        pairs = self.inds[inds]
        if pairs.shape[0]==0: 
            print(f"{self.key}: No pairs fit the criteria min_n_segment={n_segments_threshold}, nothing is plotted")
            return
        
        ylabel = "connection strength"
        if skips is not None:
            ylabel+="\nremoving outliers"
        if norm_by_total_strength:
            plot_data/=np.nansum(plot_data,axis=1,keepdims=True)
            ylabel=ylabel+" \nnormalized by total strength"
        if norm_by_n_sess: # normalize by the inverse of number of sessions where this pair appeared
            plot_data=plot_data*n_significant/self.n_segments
            ylabel=ylabel+" \n(normalized by number of sessions)"
        if zero_first_timepoint:
            # dmax = np.nanmax(plot_data,axis=1,keepdims=True)
            # dmin = np.nanmin(plot_data,axis=1,keepdims=True)
            plot_data= (plot_data-plot_data[:,0:1])
            ylabel=ylabel+" \naligning the first timepoint"
        colors = plt.cm.hsv(np.linspace(0, 1, plot_data.shape[0]))
        legend_keys = []
        
        if debug:
            max_pairs=np.max(plot_data,axis=1).argsort()[-3:][::-1]
            min_pairs=np.min(plot_data,axis=1).argsort()[:3]
            print("max",pairs[max_pairs],"min",pairs[min_pairs])
        for i, (pair, v, c, sig) in enumerate(zip(pairs,plot_data,colors,significant)):
            plt.plot(v,c=c,alpha=0.3)  # normalized
            x_sig = np.where(sig)[0]
            plt.scatter(x_sig, v[x_sig], s=8, c=c,label="_nolegend_")
            if show_legend: legend_keys.append(f"{i}:{pair}")
        plt.title(f"{self.key}")
        plt.xlabel("time segment")
        plt.xticks(np.arange(self.n_segments),np.arange(self.n_segments))
        plt.ylabel(ylabel)
        if show_legend: 
            # spacing
            ncol = 1+int(i//25)
            i_per_col=i//ncol
            offset = -.3-.5*(i_per_col/25)
            plt.legend(legend_keys,loc='right', bbox_to_anchor=(1, offset), ncol=ncol)
        
        if save:
            assert os.path.isdir(os.path.expanduser(root))
            plt.savefig(f"{os.path.expanduser(root)}/{self.key}.png", bbox_inches='tight')
        else:
            plt.show()

        mean, pvals = ttest_1samp(plot_data,0,axis=0)
        print("pvals",pvals[1:],'threshold',0.05/len(pvals[1:]),"\n")
        print("mean values",mean[1:],"\n")

    def plot_network(self):
        pass


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

    def append_ccg(self, base_key: Key, ccgs: Dict[Key, CCG]):
        self.data.update({Key(**{**base_key.__dict__, **k.__dict__}): v for k, v in ccgs.items()})

    def append_spurious(self, base_key: Key, spurs: Dict[Key, CCG]):
        self.spurious.update({Key(**{**base_key.__dict__, **k.__dict__}): v for k, v in spurs.items()})

    def append_auto(self, base_key: Key, spurs: Dict[Key, CCG]):
        self.auto.update({Key(**{**base_key.__dict__, **k.__dict__}): v for k, v in spurs.items()})

    def merge_CCGs(self, merge_levels):
        # TODO
        groups=self.group_by(*merge_levels)
        self.data={}
        for merge_key, ccg_list in groups.items():
            # [dev] ccg list items must share a common key that indexes the neurons dataset
            n_bins = ccg_list[0].shape[-1]
            n_segments = len(ccg_list)
            keys = np.array([ccg.key for ccg in ccg_list])

            inds = np.unique(np.concatenate([c.inds for c in ccg_list],axis=0),axis=0)
            ids = self.nd[merge_key.nd()].ind2id(inds)
            n_uniqinds=inds.shape[0]
            sort_inds = [np.where(inds,c.inds) for c in ccg_list]
            ccg=np.full((n_uniqinds,n_segments,n_bins),np.nan) # what if ccg list has none items..
            pval=np.full_like(ccg,np.nan)
            j_sig=np.full_like(ccg,np.nan)
            conn_strength=np.full((n_uniqinds,n_segments),np.nan)
            significant=np.full_like(conn_strength,np.nan)
            # i need to define how jsig and significant are different, or give them different names

            for i,c in enumerate(ccg_list):
                if c.ccg is not None: ccg[sort_inds[i],i,:]=c.ccg
                if c.pval is not None: pval[sort_inds[i],i,:]=c.pval
                if c.j_sig is not None: j_sig[sort_inds[i],i,:]=c.j_sig
                if c.conn_strength is not None: conn_strength[sort_inds[i],i]=c.conn_strength
                if c.significant is not None: significant[sort_inds[i],i]=c.significant
            
            # check if there's any non-nan values
            pass
            
            ccg = CCG(key=merge_key,
                ccg=ccg,
                ids=ids,
                inds=inds,
                pval=pval,
                j_sig=j_sig,
                conn_strength=conn_strength,
                conf=ccg_list[0].conf,
                significant=significant)
            ccg.keys = keys
            self.data[merge_key]=ccg


    def split_CCG(self):
        pass

    @property
    def filepath(self):
        return f"~/Documents/jitter_out/{self.conf.name}.ccg.h5"
        
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
        for key, seg_edges in self.nd.edge_times.items():
            neurons = self.nd.data[key.nd()]
            if seg_edges is None: 
                seg_edges = [np.array([neurons.t_start]),np.array([neurons.t_stop])]

            ccgs, spurs, autos, printstr = conv.eranconv_split(key=key, 
                                        neurons=neurons, 
                                        seg_edges=seg_edges, 
                                        conf=self.conf)
            
            self.append_ccg(key, ccgs)
            self.append_spurious(key, spurs)
            self.append_auto(key, autos)

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
        seg_edges=None
        
        # groups=self.group_by('session','epoch',source='connectivity')
        if inds is not None:
            pass

        elif key.ref_ind is not None:
            # REMOVE Key(session=key.session,epoch=key.epoch,ref_ind=key.ref_ind,segment=key.segment)
            seg_edges = self.nd.edge_times[key.remove('segment')]

        elif indices_source==CCGIndexSource.SIGNIFICANT:
            inds = self.data[key].inds

        elif indices_source==CCGIndexSource.SIGNIFICANT_ANY:
            inds = self.connectivity[key].inds

        elif indices_source==CCGIndexSource.AUTOCORRELOGRAMS:
            inds = np.vstack([self.auto[key].inds,self.auto[key].inds]).T    

        conv = EranConv()
        ccgs = conv.eranconv_merge(key=key, 
                            neurons=self.nd.data[key.nd()], # session, epoch, conn_type
                            pair_inds=inds,
                            seg_edges=seg_edges or self.nd.edge_times[key.get('session','epoch')], 
                            conf=self.conf)
        return ccgs

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
            new_ccgs = self._reCCG(indices_source=CCGIndexSource.SIGNIFICANT,key=key)
            for k,ccg in new_ccgs.items():
                    # inherit original significant markers
                ccg.significant = self.connectivity[key].exist
                self.data[k] = ccg
        print("rescale completed")

    def reCCG_pair_inds_1conntype(self,conn_type=('pyr','pyr'),external_inds=None,_new_data=None,_group=None):
        """
        modifies _new_data
        """
        SET_SELF_DATA=_new_data is None
        if _group is None:
            _group = self.group_by('conn_type',source='connectivity')[Key(conn_type=conn_type)]
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
                        ccg.significant[:,i_ext] = self.connectivity[k].exist[i_old].T
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
        groups = self.group_by('conn_type',source='connectivity')
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
        group = self.group_by('conn_type',source='connectivity')[Key(conn_type=conn_type)]
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

    def save_plots(self, jd = 'JitterDataset', root="~/Documents/NeuroPy/images/ccg_plots_tmp",
                   conn_types:list=None):
        assert os.path.isdir(os.path.expanduser(root))
        if isinstance(jd,str): jd = None # TODO ugly. to avoid circular imports
        keys = self.keys_matching(conn_type=conn_types) if conn_types else self.data.keys()
        print(keys)
        print(f"Saving plots under {root}")
        for key in keys:
            ccg = self.data[key]
            neurons = self.nd.data[key.nd()]
            inds = ccg.unique_inds

            print(f"ccg {key.session} {key.conn_type}")
            # try:
            if jd is not None:
                ccg.j_sig = jd.data[key].significant
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

    def set_connectivity(self, conn_types=None, sessions=None, epochs=None):
        """
        Create Connectivity objects within this class from connection strengths.
        Sess class Connecitivity().
        """
        conn_types = _san(conn_types) or self.conf.conn_types_flat
        sessions = _san(sessions) or self.nd.conf.session_names
        epochs = _san(epochs) or self.nd.conf.epochs

        grouped = self.group_by('conn_type','session','epoch','excitability')
        for k, group in grouped.items():
            exist = {}
            strength = {}
            pairs = []
            k2 = k.get('session','epoch')
            n_seg = 1 if self.nd.edge_times[k2] is None else len(self.nd.edge_times[k2][0])
            for keyy, _ in grouped[k].items():
                i_seg = keyy.segment or 0
                if keyy.conn_type in conn_types:
                    ccg=self.data[keyy]
                    if ccg is None: continue # no connections
                    for ind,val,sig in zip(ccg.inds,ccg.conn_strength.T,ccg.significant.T):#TODO how to not use transpose?
                        p = tuple(ind)
                        if p not in pairs:
                            pairs.append(p)
                            exist[p]=np.full(n_seg,False)
                            strength[p]=np.full(n_seg,np.nan)
                        if isinstance(sig,np.ndarray):
                            exist[p]=sig
                            strength[p]=val
                        else:
                            exist[p][i_seg]=sig
                            strength[p][i_seg]=val
            conn_key = Key(k.session,k.epoch,conn_type=k.conn_type,excitability=k.excitability)
            if len(pairs)==0: 
                self.connectivity[conn_key]=None
                return
            exist_arr = np.zeros((len(pairs),n_seg))
            strength_arr = np.zeros((len(pairs),n_seg))
            for i,p in enumerate(pairs):
                exist_arr[i]=exist[p]
                strength_arr[i]=strength[p]
            neurons = self.nd.data[k.nd()]
            self.connectivity[conn_key] = Connectivity(key=conn_key, n_segments=n_seg, ccgs=group,
                                                       inds=pairs, exist=exist_arr, strength=strength_arr,
                                                       ids = neurons.ind2id(pairs))

    # def set_connectivity_compound(self):
    #     """
    #     Create Connectivity objects within this class from connection strengths.
    #     Sess class Connecitivity().
    #     """
    #     for k, ccg in self.data.items():
    #         segment_key = Key(session=k.session, epoch=k.epoch, ref_ind=k.ref_ind)
    #         n = len(self.nd.edge_times[segment_key][0])
    #         neurons = self.nd.data[Key(k.session,k.epoch)]
    #         self.connectivity[k] = Connectivity(key=k, n_segments=n, ccgs=ccg,
    #                                                    inds=ccg.inds, exist=ccg.significant, strength=ccg.conn_strength,
    #                                                    ids = neurons.ind2id(ccg.inds))

    # TODO move to plotting
    def plot_connection_strengths(self,n_segments_threshold=None,
                                            norm_by_n_sess=False,
                                            norm_by_total_strength=False,
                                            zero_first_timepoint=False,
                                            show_legend=False,
                                            skips={},
                                            save=False,
                                            root='~/Documents/NeuroPy/images/conn_strengths',
                                            debug=False):
        for k1, conn in self.connectivity.items():
            print(k1,skips.get(k1))
            conn.plot_strength(n_segments_threshold=n_segments_threshold,
                                           save=save,
                                           root=root,
                                           norm_by_n_sess=norm_by_n_sess,
                                           norm_by_total_strength=norm_by_total_strength,
                                           zero_first_timepoint=zero_first_timepoint,
                                           show_legend=show_legend,
                                           skips=skips.get(k1),
                                           debug=debug)

    def plot_connection_strengths_compound(self,n_segments_threshold=None,save=False,
                                            norm_by_n_sess=False,
                                            norm_by_total_strength=False,
                                            zero_first_timepoint=False,
                                            show_legend=False,
                                            skips={},
                                            legend_group_size=25):
            self.plot_strength_compound(n_segments_threshold=n_segments_threshold,
                                           save=save,
                                           norm_by_n_sess=norm_by_n_sess,
                                           norm_by_total_strength=norm_by_total_strength,
                                           zero_first_timepoint=zero_first_timepoint,
                                           show_legend=show_legend,
                                           legend_group_size=legend_group_size
                                           )

    def plot_strength_compound(self,
                      n_segments_threshold=None,save=False,
                        norm_by_n_sess=False,
                        norm_by_total_strength=False,
                        zero_first_timepoint=False,
                        show_legend=False,
                        legend_group_size=25,
                        ):
        inds=[]
        strength=[]
        x_coords=[]
        sig=[]
        for k, ccg in self.data.items():
            edges = self.nd.edge_times[k.get('session','epoch','ref_ind')]
            xcoord = np.array([(ts+te)/7200 for ts,te in zip(edges[0],edges[1])])
            inds.append(ccg.inds)
            strength.append(ccg.conn_strength)
            sig.append(ccg.significant)
            x_coords.append(xcoord)
        inds=np.array(inds)
        x_ticks = list(np.arange(13))

        # n_segments_threshold=n_segments_threshold if n_segments_threshold is not None else self.n_segments
        plt.figure()
        # inds = self.filter(min_n_segment=n_segments_threshold,skips=skips)
        plot_data = strength
        pairs = inds
        if pairs.shape[0]==0: 
            print(f": No pairs fit the criteria min_n_segment={n_segments_threshold}, nothing is plotted")
            return

        if norm_by_total_strength:
            plot_data=[x/np.nansum(x) for x in plot_data]
        if norm_by_n_sess:
            plot_data=[x*np.count_nonzero(si)/len(x) for x,si in zip(plot_data,sig)]
        if zero_first_timepoint:
            plot_data=[(x - np.nanmin(x)) / np.nanmean(x) for x in plot_data]
        colors = plt.cm.hsv(np.linspace(0, 1, len(plot_data)))
        legend_keys = []
        for i, (pair, x,y, c,si) in enumerate(zip(pairs,x_coords,plot_data,colors,sig)):
            plt.plot(x,y,c=c,alpha=0.3)
            si=np.array(si)
            if si.any():plt.scatter(x[si], y[si], s=8,c=c,label="_nolegend_")
            if show_legend: legend_keys.append(f"{i}:{pair}({len(x)})")
        plt.title(f"{k.get('session','epoch','excitability','conn_type')}")
        plt.xlabel("time (hours)")
        plt.xticks(x_ticks,x_ticks)
        plt.ylabel("normalized connection strength")
        if show_legend: 
            # spacing
            ncol = 1+int(i//legend_group_size)
            i_per_col=i//ncol
            offset = -.3-1*(i_per_col/legend_group_size)
            plt.legend(legend_keys,loc='right', bbox_to_anchor=(1, offset), ncol=ncol)
        plt.show()
        #TODO save


class DataState(Enum):
    DEFAULT = 0
    READONLY = 1
    #TODO hopefully get rid of this


# class Jitterlet:
#     def __init__(self, key:Key, neurons:Neurons, noj_inds:Union[int,list[int]], j_ind:int, conf:JitterConfig):
#         """
#         A set of neuronal pairs defined for efficient computation. 

#         Pairs are required to have:

#         1. The same target neuron
#             (we jitter the target neuron spiketrain)

#         2. The same reference neuronal type 
#             Multiple references are okay.
#             this is to ensure significance criteria are uniform 
#             i.e. we test for either excitation or inhibition.
        
#         You should never have to create instances of this class because it will be 
#         taken care of by Jitter().
#         """
#         self.key = key
#         self.noj_inds = _san(noj_inds)
#         self.j_ind = j_ind
#         self.neurons = neurons
#         self.j_spktrains = []
#         self.j_ccg = []
#         self.conf = conf
#         self.__datastate = DataState.DEFAULT
#         assert len(set(neurons.neuron_type[self.noj_inds]))==1
    
#     @property
#     def datastate(self):
#         return self.__datastate
#     @datastate.setter
#     def datastate(self,v:DataState):
#         if self.__datastate==DataState.READONLY: return
#         self.__datastate=v
        
#     @property
#     def inds(self):
#         return np.concatenate([self.noj_inds, self.j_ind])
#     @property
#     def ref_inds(self):
#         return self.noj_inds
#     @property
#     def target_ind(self):
#         return self.j_ind

#     @property
#     def n_ref(self):
#         return len(self.noj_inds)
    
#     @property
#     def target_type(self):
#         return self.neurons.neuron_type[self.j_ind][0]
    
#     @property
#     def target_id(self):
#         return self.neurons.neuron_ids[self.j_ind]

#     @property
#     def ref_type(self):
#         return self.neurons.neuron_type[self.noj_inds[0]][0]

#     def add_jitter(self):
#         if self.conf.jitter_type == JitterType.INTERVAL:
#             self.add_interval_jitter()
#         else: # JitterType.SPIKE_TIMING
#             self.add_jitter_spike_timing()

#     def add_jitter_spike_timing(self):
#         """
#         Spike timing jitter.
#         Randomly shift each spike in target spike train
#         """
#         b = self.j_ind
#         target_nspikes = self.neurons.n_spikes[b]
#         target_spiketrain = self.neurons.spiketrains[b]
#         sampling_rate = self.neurons.sampling_rate

#         if self.conf.use_acceleration:
#             jittertrains = (
#                 cp.round(
#                     (
#                         cp.array(target_spiketrain)
#                         + 2 * self.conf.jscale * cp.random.rand(self.conf.njitter,target_nspikes)
#                         - 1 * self.conf.jscale
#                     )
#                     * sampling_rate
#                 )
#                 / sampling_rate
#             ).get()
#         else:
#             jittertrains = (
#                 np.round(
#                     (
#                         target_spiketrain
#                         + 2 * self.conf.jscale * np.random.rand(self.conf.njitter,target_nspikes)
#                         - 1 * self.conf.jscale
#                     )
#                     * sampling_rate
#                 )
#                 / sampling_rate
#             )
#         self.j_spktrains = list(jittertrains)
#         if self.conf.use_acceleration: cp.get_default_memory_pool().free_all_blocks()
    
#     def add_interval_jitter(self):        
#         sampling_rate = self.neurons.sampling_rate
#         b = self.j_ind
#         target_nspikes = self.neurons.n_spikes[b]
#         jscale_samples = int(self.conf.jscale * sampling_rate)
#         # example: jscale_ms = 5ms, sampling rate = 30KHz, jscale in samples = 150
        
#         # from https://github.com/aamarasingham/bjitter/blob/master/Figure2.m
#         if self.conf.use_acceleration:
#             jittertrains = (
#                 cp.sort(cp.floor(
#                     (cp.floor(
#                         cp.round(cp.array(self.neurons.spiketrains[b]) * sampling_rate) 
#                         / jscale_samples
#                     ) + cp.random.rand(self.conf.njitter,target_nspikes)) * jscale_samples 
#                 ))
#                 / sampling_rate
#             ).get()
#         else:
#             jittertrains = (
#                 np.sort(np.floor(
#                     (np.floor(
#                         np.round(np.array(self.neurons.spiketrains[b]) * sampling_rate) 
#                         / jscale_samples
#                     ) + np.random.rand(self.conf.njitter,target_nspikes)) * jscale_samples 
#                 ))
#                 / sampling_rate
#             )            
#         self.j_spktrains = list(jittertrains)
#         if self.conf.use_acceleration: cp.get_default_memory_pool().free_all_blocks()

#     def run_ccg_jitter(self):
#         """
#         CCGs are shaped (N0,1,nbins)
#         """
#         print("debug",self.noj_inds,self.j_ind)

#         neurons = self.neurons.neuron_slice(neuron_inds=self.noj_inds)
#         j = Neurons(spiketrains=self.j_spktrains,
#             t_start=self.neurons.t_start,
#             t_stop=self.neurons.t_stop,
#             neuron_ids=[self.target_id]*self.conf.njitter,
#             neuron_type=[self.target_type]*self.conf.njitter
#             ) # TODO not copying over other fields
#         neurons.merge(j)
        
#         self.j_ccg=correlations.spike_correlations(
#                 neurons=neurons,
#                 ref_neuron_inds=np.arange(self.n_ref),
#                 neuron_inds=self.n_ref+np.arange(self.conf.njitter),
#                 bin_size=self.conf.ccg.bin_size,
#                 window_size=self.conf.ccg.duration,
#                 use_acceleration=self.conf.ccg.use_acceleration,
#                 symmetrize=self.conf.ccg.symmetrize_ccg,
#             )
#         # Debugging - 'debug' should be all zeros (two methods are identical)
#         # orig = correlations.spike_correlations(
#         #         neurons=neurons,
#         #         neuron_inds=np.arange(neurons.n_neurons),
#         #         bin_size=bin_size,
#         #         window_size=duration,
#         #         use_acceleration=use_acceleration,
#         #         symmetrize=True,
#         #     )
#         # debug = orig[0,len(noj_inds):]-ccg_all[0]
#         # print(debug)

#     def jitter_significance(self, EI):
#         """
#                 EI: if 'E', use p-vals for peaks, else use q-vals for troughs
#         # TODO
#         # ccg_all: (N0, njitter+1, Nbins)
#         # pval = (N0, Nbins) where real data is ranked among fake data. conservative when there are ties
#         # thresholds = (N0, Nbins)

#         """
#         if EI=='E':
#             pval = np.argsort(np.argsort(-self.j_ccg,axis=1,kind="stable"),axis=1)[:,-2]/self.conf.njitter
#             thresholds = np.percentile(self.j_ccg[:,1:], 100*(1-self.conf.alpha), axis=1)
#         else:
#             pval = np.argsort(np.argsort(self.ccg,axis=1,kind="stable"),axis=1)[:,-2]/self.njitter
#             thresholds = np.percentile(self.ccg[:,1:], 100*(self.alpha), axis=1)

#         self.j_sig = pval<=self.conf.alpha
#         self.thresholds=thresholds

#     def jbsi(self,real_ccg):
#         """
#         Jitter-based synchrony index  Agmon (2012)
#         """
#         assert self.j_ccg is not None

#         j_ccg_avg = np.mean(self.j_ccg,axis=1) # (N0, Nbins) averaged over Njitter columns
#         n1 = np.minimum(self.neurons.firing_rate[self.j_ind],
#                             self.neurons.firing_rate[self.noj_inds])[..., None] # (N0,1) or (1,1)

#         ts = self.conf.ccg.bin_size
#         tj = self.conf.jscale

#         b = tj/(tj-ts) if tj/ts>2 else 2
#         JBSI =  b/n1*(real_ccg - j_ccg_avg) # (N0, Nbins) or (1, Nbins)
#         return JBSI
    
#     def spktrain_path(self): # TODO
#         get_path_from_key(self.key)
#         pass

#     def ccg_path(self):
#         get_path_from_key(self.key)
#         pass

#     def save(self):
#         with h5py.File(self.spktrain_path, "a") as f:
#             f.create_dataset(self.j_ind, data=self.j_spktrains)
#             print(f"saved jitter spiketrains {self.key}:{self.j_ind}")
#         with h5py.File(self.ccg_path, "a") as f:
#             f.create_dataset(self.j_ind, data=self.j_ccg)
#             print(f"saved jitter ccgs {self.key}:{self.j_ind}")

#     def load(self):
#         # loaded data 
#         with h5py.File(self.spktrain_path, "r") as f:
#             self.j_spktrains = f[self.j_ind][:]
#             print(f"loaded jitter spiketrains {self.key}:{self.j_ind}")
#         with h5py.File(self.ccg_path, "r") as f:
#             self.ccg_path = f[self.j_ind][:]
#             print(f"loaded jitter ccgs {self.key}:{self.j_ind}")
#         self.neurons = None
#         self.conf = None
#         self.datastate=DataState.READONLY # data was loaded not computed. cannot recompute bc neurons/conf will not be saved


# class Jitter:
#     def __init__(self, key:Key, neurons: Neurons, conf:JitterConfig, ccg:CCG, root:str=None):
#         """Single session/epoch jitters
#         Note: Jitter computation is time and memory consuming!"""
#         self.key = key
#         self.conf = conf

#         self.jref_inds = []
#         self.jtgt_inds = []
#         self.pos = {} # TODO name
#         self.pval = []
#         self.significant = []
#         self.threshold = []
#         self.JBSI = []
#         self.jitterlets = {}

#         self.neurons = neurons
#         self.ccg = ccg

#         self.root = root or f"~/Documents/jitter_out"
#         self.get_jitter_inputs()
    
#     @property
#     def n_inds(self):
#         return len(self.ccg.inds)

#     def get_jitter_inputs(self):
#         """Reshape coordinates of (ref,target) pairs into most efficient format for jittering
#         grouped by target indices"""
#         keys, inv = np.unique(self.ccg.inds[:,-1], return_inverse=True)
#         self.jref_inds = [self.ccg.inds[inv==i,0].tolist() for i in range(len(keys))]
#         self.jtgt_inds = keys
#         self.pos = {k: np.where(inv == i)[0] for i, k in enumerate(keys)}
    
#     def get(self,ref,tgt,field='jitters'):
#         # TODO untested
#         return getattr(self.data[tgt], field)[np.where(self.data[tgt].noj_inds==ref)[0]]

#     def run(self,save_progress=False):
#         self.JBSI = np.zeros((self.n_inds,self.ccg.conf.nbins))
#         for refs, tgt in zip(self.jref_inds,self.jtgt_inds):
#             self.jitterlets[tgt] = Jitterlet(key=self.key,
#                                         neurons=self.neurons,
#                                        noj_inds=refs,
#                                        j_ind=tgt,
#                                        conf=self.conf)
#         for tgt,j in self.jitterlets.items():
#             j.add_jitter()
#             j.run_ccg_jitter()
#             self.JBSI[self.pos[tgt]] = j.jbsi(self.ccg.ccg[self.pos[tgt]]) # TODO indexing
#         if save_progress:
#             self.save()

#     @property
#     def filepath(self):
#         return f'{self.root}/jitter-{str(self.key)}.h5'

#     def save(self,intermediates=False):
#         if intermediates:
#             for k,v in self.jitterlets.items():
#                 v.save()
#         with h5py.File(self.filepath, "a") as f:
#                 f.create_dataset((self.session_name,field), data=getattr(self, field, None))
#         print(f"saved {self.key}")

#     def load(self,intermediates=False):
#         if intermediates:
#             for k,v in self.jitterlets.items():
#                 v.load()
#         with h5py.File(self.filepath, "r") as f:
#             for field in ['key','pval','JBSI','conf']:
#                 d = f[(self.session_name,field)][:]
#                 setattr(self, field, d)
#         print(f"loaded jitter data {self.key}")
   

# class JitterDataset(AnalysisDataset):
#     def __init__(self, nd: Neurons, cd: CCGDataset, conf:JitterConfig):
#         """Note that jitter dataset stores single session/epoch data because jitter computation is memory consuming"""
#         self._conf = conf
#         self.conf.ccg=cd.conf
#         self.nd = nd
#         self.cd = cd
#         self.data = {} # data key is target index

#     @property
#     def filepath(self):
#         return f'~/Documents/jitter_out/'

#     def save(self):
#         for k,v in self.data.items():
#             v.save(root=self.filepath)

#     def load(self):
#         for k,v in self.data.items():
#             v.load(root=self.filepath)

#     def run_jitter(self,save_progress=True):
#         for key, ccg in self.cd.data.items():
#             if ccg is None: 
#                 self.data[key] = None
#             else:
#                 neurons = self.nd.data[key.nd()]
#                 self.data[key]=Jitter(key=key,
#                                     neurons=neurons,
#                                     conf=self.conf,
#                                     ccg=ccg)
#         for _,j in self.data.items():
#             if j is not None: j.run(save_progress=save_progress)


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


class EranConv:
    """
    A device for running EranConv and other significance tests    
    """
    rough_mask={} # used for eranconv_split
    mask={}

    pvals=[]
    qvals=[]
    pval_corrected=[]
    qval_corrected=[]
    qval_corrected2=[]
    ccg=[]
    acg=[]
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


    def intersect(coords1, coords2, n, n_groups=1):
        # Intersection of coordinate lists
        if coords1 is None or coords2 is None: return np.array([])
        ravel_dims = (n,n) if coords1.shape[-1]==2 else (n_groups,n,n) #TODO
        coords1_flat = np.ravel_multi_index(coords1.T, ravel_dims)
        coords2_flat = np.ravel_multi_index(coords2.T, ravel_dims)
        coords_flat = np.intersect1d(coords1_flat, coords2_flat)
        coords = np.array(np.unravel_index(coords_flat, ravel_dims)).T
        return coords

    @staticmethod
    def setdiff(coords1,coords2, n, n_groups=1):#n2=None
        # Set difference of coordinate lists
        if coords1 is None or coords2 is None: 
            return coords1 if coords1 is not None else np.array([])
        ravel_dims = (n,n) if coords1.shape[-1]==2 else (n_groups,n,n)
        flat1 = np.ravel_multi_index(coords1.T, ravel_dims)
        flat2 = np.ravel_multi_index(coords2.T, ravel_dims)
        flat  = np.setdiff1d(flat1, flat2)
        coords = np.array(np.unravel_index(flat, ravel_dims)).T
        return coords


    @staticmethod
    def union(coords1, coords2, n, n_groups=1):#n2=None
        # Set difference of coordinate lists
        if coords1 is None: return coords2 if coords2 is not None else np.array([])
        elif coords2 is None: return coords1 if coords1 is not None else np.array([])
        ravel_dims = (n,n) if coords1.shape[-1]==2 else (n_groups,n,n)
        flat1 = np.ravel_multi_index(coords1.T, ravel_dims)
        flat2 = np.ravel_multi_index(coords2.T, ravel_dims)
        flat  = np.union1d(flat1, flat2)
        coords = np.array(np.unravel_index(flat, ravel_dims)).T
        return coords

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
        conf=self.conf
        pair_inds = np.argwhere((self.ccg[...,conf.min_spkcnt_bin:conf.max_spkcnt_bin]>=conf.min_spkcount).all(axis=-1)) # NOTE right now it's the same criteria for E/I
        return pair_inds
    
    def significance_mask(self,excitability):
        conf = self.conf
        if excitability=='E':
            sig, self.pval_corrected = EranConv.multiple_correction(self.pvals, conf.alpha)
            pair_inds = np.argwhere((sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1))
        elif excitability=='I':
            sig1, self.qval_corrected = EranConv.multiple_correction(self.qvals, conf.alpha)
            sig2, self.qval_corrected2 = EranConv.multiple_correction(self.qvals, conf.alpha2)
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
    
    @staticmethod
    def count_significant_pairs():
        """
        Create a tally of significant neuronal connectoins by type
        Currently, the type is defined as 
            reference-target/[E,I]
        where reference is presynaptic, and target is postsynaptic neuronal type, 
        and E/I indicates the connection being excitatory or inhibitory

        SL: If this helper function seems messy it's probably because 
        it pertains to our specific definition of significant pairs (see Diba 2014, Pairwise connections.)
        """
        pass

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

    def eranconv_split(self, key:Key, neurons:Neurons, seg_edges:np.ndarray, conf:CCGConfig,):
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
        print("running eranconv (1st pass)")
        self.conf = conf
        self.n = neurons.n_neurons
        self.n_segments = len(seg_edges[0])

        self.ccg = correlations.spike_correlations(
                neurons=neurons,
                neuron_inds=np.arange(neurons.n_neurons), # all
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                seg_edges=seg_edges,
            )

        self.pvals, self.pred,self.qvals = EranConv._conv(self.ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        self.mask={'E':{},'I':{}}
        self.rough_mask={'E':{},'I':{}}

        for EI in ['E','I']:
            self.rough_mask[EI] = EranConv.intersect(self.significance_mask(EI), self.spkcount_mask(),self.n,self.n_segments)
            val = self.rough_mask.get(EI)
            if isinstance(val, np.ndarray) and val.any():
                self.rough_mask[EI] = self._autocorr_mask(self.rough_mask[EI])
            val = self.rough_mask.get(EI)
            if isinstance(val, np.ndarray) and val.any():
                self.mask[EI] = self._probe_loc_mask(self.rough_mask[EI],neurons)
            val = self.mask.get(EI)
            if isinstance(val, np.ndarray) and val.any():
                self.mask[EI] = self._cell_type_mask(self.mask[EI],neurons,conf.conn_types[EI])

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
                    for row in self.rough_mask[EI]:
                        groups[row[0]][EI].append(row[1:])
                for i in range(self.n_segments): 
                    for EI in ['E','I']:
                        groups[i][EI]=np.array(groups[i][EI])
                self.rough_mask = groups

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

            if len(self.ccg.shape)==3: 
                self.ccg=self.ccg[np.newaxis,...]
                self.pred=self.pred[np.newaxis,...]
                self.pval_corrected=self.pval_corrected[np.newaxis,...]
                self.qval_corrected=self.qval_corrected[np.newaxis,...]

            # Update return values
            for seg in range(self.n_segments):
                for EI in ['E','I']:
                    pairs=self.mask[seg][EI]
                    spairs = self.rough_mask[seg][EI] # initialize spurious pairs
                    p = self.pval_corrected if EI=='E' else self.qval_corrected # TODO TODO not storing corrected p-vals. make it an option!

                    for conn_type, prs in pairs.items():
                        new_key = key.add(segment=seg if self.n_segments>1 else None,
                                conn_type=conn_type,
                                excitability=EI)
                        if prs is None or len(prs)==0: ccgs_by_type[new_key] = None
                        else:
                            x,y = prs[:,-2],prs[:,-1]
                            ccgs_by_type[new_key] = CCG(key=new_key,conf=self.conf, 
                                                inds=prs, ids=neurons.ind2id(prs), 
                                                ccg=self.ccg[seg,x,y], ccg_null=self.pred[seg,x,y], pval=p[seg,x,y], 
                                                )
                            spairs = EranConv.setdiff(spairs,prs,self.n,self.n_segments) # remove these pairs from spurious

                    new_key = key.add(segment=seg if self.n_segments>1 else None,
                                      excitability=EI)

                    if isinstance(spairs, np.ndarray) and spairs.any():
                        x,y = spairs[:,-2],spairs[:,-1] # x,y = pairs[...,0],pairs[...,1]
                        spurs_by_type[new_key] = CCG(key=new_key,conf=self.conf, 
                                            inds=spairs, ids=neurons.ind2id(spairs), 
                                            ccg=self.ccg[seg,x,y], ccg_null=self.pred[seg,x,y], pval=p[seg,x,y], 
                                            )
                    else:
                        spurs_by_type[new_key] = None

            autocorr_locations = EranConv.get_autocorr_locations(self.ccg.shape)            
            new_key = key
            acgs[new_key] = ACG(key=new_key,
                        acg=self.ccg[:,autocorr_locations[0]],
                        inds=np.arange(neurons.n_neurons),
                        ids=neurons.ind2id(np.arange(neurons.n_neurons)),
                        conf=self.conf)

            return ccgs_by_type, spurs_by_type, acgs
        ccgs_by_type, spurs_by_type, acgs = process_output(key, neurons)

        overview_str=""
        for i in range(self.n_segments):
            E_str, hasE = self._printstr_sig(self.mask[i]['E'], 'E')
            I_str, hasI = self._printstr_sig(self.mask[i]['I'], 'I')
            overview_str += f"SLEEP{i}: E/I pairs {self.rough_mask[i]['E'].shape[0]:03d} / {self.rough_mask[i]['I'].shape[0]:03d} | "
            overview_str=overview_str+E_str+I_str+"\n" if (hasE or hasI) else overview_str+"no connections\n"
        print("eranconv (1st pass) done")

        return ccgs_by_type, spurs_by_type, acgs, overview_str

    def eranconv_merge(self, key:list[Key], neurons:Neurons, pair_inds:dict[list], seg_edges:np.ndarray, conf:CCGConfig):
        # ref and target indices should be organized by conn type
        print(f"running eranconv (2nd pass): {key}")

        self.conf = conf
        self.n=neurons.n_neurons
        if seg_edges is None: 
            seg_edges = [np.array([neurons.t_start]),np.array([neurons.t_stop])]
        self.n_segments = len(seg_edges[0])

        ccgs = {}
        neuron_inds = np.unique(pair_inds)
        self.ccg = correlations.spike_correlations(
                neurons=neurons,
                neuron_inds=neuron_inds,
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                seg_edges=seg_edges,
            )
        idx = {n: i for i, n in enumerate(neuron_inds)}
        slicer = (...,
                [idx[a] for a in pair_inds[:, -2]],
                [idx[b] for b in pair_inds[:, -1]],
                slice(None))
        self.ccg=self.ccg[slicer]

        self.pvals, self.pred, self.qvals = EranConv._conv(self.ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        conf = self.conf
        if key.excitability=='E':
            sig, self.pval_corrected = EranConv.multiple_correction(self.pvals, conf.alpha)
            sig_mask = (sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1)
        elif key.excitability=='I':
            sig1, self.qval_corrected = EranConv.multiple_correction(self.qvals, conf.alpha)
            sig2, self.qval_corrected2 = EranConv.multiple_correction(self.qvals, conf.alpha2)
            neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1))  # significant bins must have a significant-ish neighbor
            sig_mask = neighbor.any(-1)

        k=key.remove('segment')
        ccgs[k]=CCG(inds=pair_inds, 
                ids=neurons.ind2id(pair_inds), 
                ccg=self.ccg, 
                ccg_null=self.pred, 
                pval=self.pval_corrected if key.excitability=='E' else self.qval_corrected, 
                conf=self.conf,
                significant=sig_mask,
                key=k,
                seg_edges=seg_edges)
        ccgs[k].set_firing_rates(neurons)

        # for i in range(self.n_segments):
        #     k=key.change(segment=i)
        #     ccgs[k]=CCG(inds=pair_inds, 
        #             ids=neurons.ind2id(pair_inds), 
        #             ccg=self.ccg[i], 
        #             ccg_null=self.pred[i], 
        #             pval=self.pval_corrected[i] if key.excitability=='E' else self.qval_corrected[i], 
        #             conf=self.conf,
        #             significant=sig_mask[i],
        #             key=k)
        print("done")
        return ccgs

    def eranconv_ref(self, key:list[Key], neurons:Neurons, pair_inds:dict[list], seg_edges:np.ndarray, conf:CCGConfig):
        # TODO
        print(f"running eranconv (2nd pass): {key}")

        self.conf = conf
        self.n=neurons.n_neurons
        if seg_edges is None: 
            seg_edges = [np.array([neurons.t_start]),np.array([neurons.t_stop])]
        self.n_segments = len(seg_edges[0])

        ccgs = {}

        self.ccg = correlations.spike_correlations(
                neurons=neurons,
                ref_neuron_inds=pair_inds[0],
                neuron_inds=pair_inds[1:],
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                seg_edges=seg_edges,
            )[:,0,1:2] #TODO
        self.pvals, self.pred, self.qvals = EranConv._conv(self.ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        self.rough_mask = EranConv.intersect(self.significance_mask(key.excitability), self.spkcount_mask(),self.n,self.n_segments)
        if self.rough_mask.any(): self.rough_mask = self._probe_loc_mask(self.mask,neurons)
        
        significant = np.full((self.ccg.shape[-3],self.ccg.shape[-2]),False) 
        if self.rough_mask.any(): significant[self.rough_mask[:,0],self.rough_mask[:,1]] = True
        significant=significant.squeeze()

        # REMOVE Key(session=key.session,epoch=key.epoch,segment=None,ref_ind=key.ref_ind,target_ind=key.target_ind,
        #       excitability=key.excitability,conn_type=key.conn_type)
        ccgs[k]=CCG(inds=pair_inds, 
                ids=neurons.ind2id(pair_inds), 
                ccg=self.ccg, 
                ccg_null=self.pred, 
                pval=self.pval_corrected if key.excitability=='E' else self.qval_corrected, 
                conf=self.conf,
                significant=sig,
                key=key.remove('segment')) # merged ccg

        print("done")
        return ccgs
    

# NOTE move to plotting in the future!
import seaborn as sns
import matplotlib.pyplot as plt
import os
import neuropy.plotting.probe as probe


def plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, 
                   pval=None, ccg_null=None, j_sig=None,segment_id=None):
    """Single CCG plot into provided axis"""
    bins = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)

    ax.bar(bins, ccg, width=bin_size, alpha=0.5, label="ccg")
    if ccg_null is not None:
        ax.bar(bins, ccg_null, width=bin_size, alpha=0.5, label="ccg-smooth")
    ax2 = ax.twinx()
    ylim=ax.get_ylim()[1]*0.8
    if pval is not None:
        ax2.plot(bins, pval/pval.max() * ylim, label='p',alpha=0.3, color='gray')
    if j_sig is not None:
        ax2.plot(bins, j_sig/j_sig.max() * ylim, label='j-significance')
    # Set ticks to show original pval values
    ticks_scaled = np.linspace(0, ylim, len(ax.get_yticks()))  # positions in scaled axis
    ticks_original = np.round(ticks_scaled /ylim*pval.max(), 2) #TODO sometimes p val looks weird, is it a problem w p val calculation?
    ax2.set_yticks(ticks_scaled)
    ax2.set_yticklabels(ticks_original)
    ax2.set_ylabel("p-value")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    X, Y = ids; x, y = inds
    ax.set_title(f"CCG{segment_id}, neuron_ids=[{X},{Y}], indices=[{x},{y}]")
    ax.legend()
    sns.despine(ax=ax)
    sns.despine(ax=ax2)
    # return ax, ax2 #TODO read return vars?

def plot_waveform_panel(ax, waveform, neuron_type, neuron_id, 
                        frate_all=None, frates_cut=None, n_shanks=None, ch_per_shank=None, discarded_channels=None):
    """Single waveform panel into provided axis"""
    n_shanks = n_shanks or 12
    ch_per_shank = ch_per_shank or CHANNELS_PER_SHANK # TODO put hardcoded values elsewhere?
    max_ch = waveform.shape[0]
    ax.imshow(waveform.astype(float))
    ax.set_title(f"{neuron_type}{neuron_id}")
    xlabel = ""
    if frate_all is not None:
        xlabel += f"{frate_all:.2f}Hz all "
    elif frates_cut is not None:
        xlabel += f"{frates_cut:.2f}Hz cut "
    ax.set_xlabel(xlabel)
    
    edges = (np.array(range(n_shanks))+1)*ch_per_shank+1
    if discarded_channels is not None:
        shanks = discarded_channels // ch_per_shank
        edges = edges - np.cumsum(np.histogram(shanks,np.arange(n_shanks))[0])
        
    for k in edges:
        ax.axhline(k, c='w', alpha=0.5, linestyle='dashed')
    
    return ax


def plot_ccg_figure(ccg, ids, inds, neuron_types, waveforms, 
                    window_size, bin_size, pval=None, ccg_null=None, j_sig=None, 
                    shank_ids=None,
                    frates_all=None, frates_cut=None, n_shanks=None, ch_per_shank=None,
                    discarded_channels=None,
                    show=True, save=False, plotdir=None,
                    waveform_plot_type="channel",
                    segment_id=None):
    """Full figure: CCG + 2 waveforms"""
    if waveform_plot_type=='channel':
        fig, axs = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

    # labels = ['ref', 'target']

    plot_ccg_panel(axs[0], ccg, ids, inds, window_size, bin_size, pval, ccg_null, j_sig,segment_id)
    if waveform_plot_type=='channel' and shank_ids is not None:
        def get_filled_waveforms(shank_id,wf):
            channel_ids = ch_per_shank*shank_id+np.arange(ch_per_shank)
            mask = ~np.isin(channel_ids, discarded_channels)
            start = ch_per_shank*shank_id-np.sum(discarded_channels<16*shank_id)
            length = np.sum(mask,axis=0)
            clean = np.full((ch_per_shank,wf.shape[-1]),np.nan)
            clean[mask]=wf[start:start+length]
            return clean

        ref_waveform = get_filled_waveforms(shank_ids[0],waveforms[0])
        tgt_waveform = get_filled_waveforms(shank_ids[1],waveforms[1])

        xlabel = ""
        if frates_all is not None:
            xlabel += f"ref {frates_all[0]:.2f}Hz | tgt {frates_all[1]:.2f} all \n"
        if frates_cut is not None:
            xlabel += f"ref {frates_cut[0]:.2f}Hz | tgt {frates_cut[1]:.2f} cut "
        axs[1] = probe.plot_waveform_on_channel(ref_waveform, shank_ids[0], 
                                                tgt_waveform, shank_ids[1], 
                                                footnote=xlabel, amplitude_limit=True,
                                                ax=axs[1],
                                                color='green' if shank_ids[0]!=shank_ids[1] else 'orange')
        sns.despine(ax=axs[1])
    else:
        for i in range(2):
            axs[1+i] = plot_waveform_panel(axs[1+i], waveforms[i], neuron_types[i], ids[i],
                                frates_all[i] if frates_all is not None else None,
                                frates_cut[i] if frates_cut is not None else None,
                                n_shanks=n_shanks,ch_per_shank=ch_per_shank,
                                discarded_channels=discarded_channels)

    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
        assert os.path.exists(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png") #TODO why do we need this?
        plt.close(fig)
    if show:
        plt.show()
        plt.close(fig)
    return fig


def plot_ccg_only(ccg, ids, inds, window_size, bin_size, pval=None, ccg_null=None, j_sig=None, 
                  show=True, save=False, plotdir=None):
    """Save only the CCG plot without waveforms"""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, pval, ccg_null, j_sig)
    
    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
    if show:
        plt.show()
    plt.close(fig)
    return fig
