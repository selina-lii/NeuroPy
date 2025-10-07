"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

import numpy as np
try:
    import cupy as cp
except ImportError:
    print("Error importing CuPy")
    cp = None
import neuropy.analyses.correlations as correlations
from neuropy.core.neurons import Neurons
from scipy.signal import windows, convolve
from scipy.stats import poisson
from scipy import ndimage
import time
from scipy.stats import ttest_ind
from typing import Union, Optional, Dict, Any, Tuple
from datetime import datetime
import h5py
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field
from collections import defaultdict


def _short_session_name(session):
    """get short printable session name in the format of ANIMAL_DayX"""
    sess_name = session.filePrefix.parts[-1].split('_')[:2]
    sess_name='_'.join(sess_name)
    return sess_name


def _san(var):
    """Sanitize """
    if var is None: return var
    if not isinstance(var, list): var = [var]
    return var


def _intersect2d(coords1,coords2,n):
    # Intersection of coordinate lists
    coords1 = coords1[:,0]*n+coords1[:,1]
    coords2 = coords2[:,0]*n+coords2[:,1]
    coords=np.intersect1d(coords1,coords2)
    coords=np.array([[x//n,x%n] for x in coords])
    return coords


def _setdiff2d(coords1,coords2,n):
    # Set difference of coordinate lists
    coords1 = coords1[:,0]*n+coords1[:,1]
    coords2 = coords2[:,0]*n+coords2[:,1]
    coords=np.setdiff1d(coords1,coords2)
    coords=np.array([[x//n,x%n] for x in coords])
    return coords


@dataclass(frozen=True)
class Key:
    session: Optional[str] = None
    epoch: Optional[str] = None
    chunk: Optional[int] = None
    excitability: Optional[tuple[str,str]] = None
    conn_type: Optional[str] = None

    """
    Dependencies
    tuple(session, epoch, chunk) should alway be present
    conn_type -> excitability
    """
    
    def __str__(self):
        parts = []
        if self.session: parts.append(f"{self.session}")
        if self.epoch: parts.append(f"{self.epoch}")
        if self.chunk is not None: parts.append(f"c:{self.chunk}")
        if self.excitability: parts.append(f"{self.excitability}")
        if self.conn_type: parts.append(f"{self.conn_type[0]}-{self.conn_type[1]}")
        return "_".join(parts) if parts else "root"

    def matches(self, **kwargs) -> bool:
        """Check if this key matches given criteria (for filtering)"""
        for k, v in kwargs.items():
            if v is not None and getattr(self, k, None) != v:
                return False
        return True
    
    def parent(self) -> 'Key':
        """Get parent key (one level up in hierarchy)"""
        if self.excitability is not None: # conn_type goes with excitability
            return Key(self.session, self.epoch, self.chunk)
        if self.chunk is not None:
            return Key(self.session, self.epoch)
        if self.epoch is not None:
            return Key(self.session)
        return Key()


class AnalysisDataset:
    """
    Container for all analysis data with flexible indexing
    """
    def __init__(self):
        self.data: Dict[Key, Any] = {}
    
    def __setitem__(self, key: Key, value: Any):
        """Store data with a key"""
        self.data[key] = value
    
    def __getitem__(self, key: Key) -> Any:
        """Retrieve data by key"""
        return self.data[key]
    
    def get(self, key: Key, default=None) -> Any:
        """Safe retrieval with default"""
        return self.data.get(key, default)
    
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
    
    def group_by(self, *dimensions) -> Dict[Tuple, Dict[Key, Any]]:
        """
        Group data by specified dimensions.
        
        Example:
            dataset.group_by('session', 'epoch')
            # Returns: {('s1', 'pre'): {key: data, ...}, ('s1', 'post'): {...}}
        """
        groups = defaultdict(dict)
        for key, value in self.data.items():
            group_key = tuple(getattr(key, dim, None) for dim in dimensions)
            groups[group_key][key] = value
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
    
    def __str__(self):
        lines = [f"AnalysisDataset with {len(self.data)} entries:"]
        
        # Group by session and epoch for readable output
        by_session = self.group_by('session')
        for session, session_data in sorted(by_session.items()):
            lines.append(f"  Session {session[0]}:")
            by_epoch = defaultdict(list)
            for key in session_data.keys():
                by_epoch[key.epoch].append(key)
            for epoch, keys in sorted(by_epoch.items()):
                lines.append(f"    Epoch {epoch}: {len(keys)} entries")
        
        return "\n".join(lines)


class NeuronsDatasetConfig:
    """
    Metadata of NeuronsDataset

    tight_time: bool
    if true, try to shrink start and end of epoch to where brainstates are happening 

    chunks_per_session: int
    Splits session time axis into equal-lengthed blocks if >1

    """
    def __init__(self,
                 name:str = "default",
                 neuron_types:Union[list[str], str] = ['pyr', 'inter'], 
                 epochs:Union[list[str], str]="post", 
                 chunks_per_session:Union[list[int], int]=1, 
                 sleep_selection:Union[list[str], str]=["REM","NREM"], 
                 ripple_selection=None, tight_epoch=False):
        self.name = name
        self.session_names = []
        self.neuron_types = _san(neuron_types)
        self.epochs = _san(epochs)
        self.sleep_selection = _san(sleep_selection)
        self.ripple_selection = ripple_selection
        self.chunks_per_session = _san(chunks_per_session)
        self.tight_epoch = tight_epoch

        assert len(self.chunks_per_session)==len(self.epochs)

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

    def get_chunks_from_epoch(self,epoch):
        idx = self.epochs.index(epoch)
        return self.chunks_per_session[idx]


class CCGConfig:
    def __init__(self, 
                name="default",
                conn_types_E:Union[list[list], list]=[('pyr','pyr'), ('pyr','inter')],
                conn_types_I:Union[list[list], list]=[('inter','inter'), ('inter','pyr')],
                duration:float=20*1e-3,
                bin_size:float=1*1e-3,
                bin_align = 'center',
                jscale = 5,
                alpha:float = 0.05,
                alpha2:float = 0.1,
                min_lag:float = 1*1e-3,
                max_lag:float = 3*1e-3,
                min_spkcount = 2.5,
                spkcount_scope = 12*1e-3,
                multiple_correction_method:str = None,
                ignore_same_electrodes = True,
                use_cupy = True,
                symmetrize_ccg = True,
                ):
        self.name = name

        self.conn_types_E = conn_types_E
        self.conn_types_I = conn_types_I
        self.duration = duration
        self.bin_size = bin_size
        self.jscale = jscale # 
        self.alpha = alpha
        self.alpha2 = alpha2
        self.use_multiple_correction = multiple_correction_method is not None
        self.mc_method = multiple_correction_method
        self.center_bin = int(self.duration/self.bin_size//2)
        self.nbins = int(self.duration/self.bin_size)

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_spkcount = min_spkcount
        self.spkcnt_scope = spkcount_scope
        self.spkcnt_bins = int(self.spkcnt_scope/self.bin_size)
        self.ignore_same_electrodes = ignore_same_electrodes

        self.min_lag_bin = self.center_bin+int(self.min_lag/self.bin_size) # leftmost bin for p value test
        self.max_lag_bin = self.center_bin+int(self.max_lag/self.bin_size)+1 # rightmost bin for p value test
        self.min_spkcnt_bin = self.center_bin-self.spkcnt_bins//2 # leftmost bin requiring minimum spike count 
        self.max_spkcnt_bin = self.center_bin+self.spkcnt_bins//2+1 # rightmost bin requiring minimum spike count

        self.use_cupy = use_cupy
        self.symmetrize_ccg = symmetrize_ccg
        self.bin_align = bin_align # 'center', 'edge'

        # if self.use_multiple_correction: 
        #     self.corrected_alpha=alpha/(n**2-n)/self.nbins # local threshold
        #     self.corrected_alpha2=alpha2/(n**2-n)/self.nbins


        # example configs
        msconn_args = {
            'min_lag':0,
            'max_lag':1,
            'min_spkcount':2.5,
            'spkcount_scope':12,
            'ignore_same_electrodes':False,
            'ref_type':'pyr',
            'target_type':'pyr',
            'p':0.05,
        }
        excit_args = {
            'min_lag':1,
            'max_lag':3,
            'min_spkcount':2.5,
            'spkcount_scope':12,
            'ignore_same_electrodes':True,
            'ref_type':'pyr',
            'target_type':['pyr','inter'],
            'p':0.05,
        }
        inhib_args = {
            'min_lag':1,
            'max_lag':3,
            'min_span':2,
            'min_spkcount':2.5,
            'spkcount_scope':12,
            'ignore_same_electrodes':False,
            'ref_type':'inter',
            'target_type':['pyr','inter'],
            'p':0.05,
            'p2':0.1
        }

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


class JitterConfig:
    def __init__(self, njitter:int=None, jitter_type:str=None, jscale:int=None, alpha:float=.05, use_cupy=True,
                 ):
        """
        Parameters
        ----------
        njitter : int
            number of jitters
        jscale: int
            maximum spiking time shift in ms (default is +-5ms)
        use_cupy: bool, optional
            whether or not to use gpu acceleration
        """
        self.njitter = njitter
        self.jitter_type = jitter_type
        self.jscale = jscale
        self.alpha = alpha
        self.use_cupy = use_cupy

    def __str__(self):
        return f"njitter:{self.njitter}, jitter_type:{self.jitter_type}, jscale:{self.jscale}, p:{self.alpha}, use_cupy:{self.use_cupy}"


class KeySlicing:
    data = None
    @property
    def _key_list(self):
        assert isinstance(self.data, dict)
        return list(self.data.keys())

    def session_keys(self, session):
        keys = self._key_list
        if len(keys)==0:
            return []
        if isinstance(keys[0],str):
            return [k for k in keys if k == session]
        else:
            return [k for k in keys if k[0] == session]

    def epoch_keys(self, epoch):
        keys = self._key_list
        if len(keys)==0:
            return []
        assert len(keys[0])>=2
        return [k for k in keys if k[1] == epoch]
        
    def chunk_keys(self, chunk_id, epoch:Optional[Union[str,list[str]]]=None):
        keys = self._key_list
        if len(keys)==0:
            return []
        assert len(keys[0])>=3
        if epoch:
            epochs = _san(epoch)
            keys = [k for k in keys if k[1] in epochs] # all sessions should have that chunk
            for e in epochs:
                # check if the epoch has this many chunks
                assert epoch in self.conf.epochs
                assert self.conf.get_chunks_from_epoch(epoch)>chunk_id
        return keys

    def excitability_keys(self, excitability):
        keys = self._key_list
        if len(keys)==0:
            return []
        assert len(keys[0])>=2
        return [k for k in keys if k[-2] == excitability]
        
    def conn_type_keys(self, conn_type, excitability=None):
        keys = self._key_list
        if len(keys)==0:
            return []
        assert len(keys[0])>=2
        keys = [k for k in keys if k[-1] == conn_type]  
        if excitability:
            keys = [k for k in keys if k[-2] == excitability]  
        return keys


class NeuronsDataset(AnalysisDataset):
    """
    A collection of neurons created for analysis
    Arguments of the analysis should be provided in an NeuronsDatasetConfig instance

    sessions: subjects.ProcessData
        collection object of sessions
    """
    def __init__(self, sessions, conf:NeuronsDatasetConfig):
        
        self.conf = conf
        self.data = {}
        
        sessions = _san(sessions)
        
        for s in sessions:
            ssn = _short_session_name(s)
            self.conf.session_names.append(ssn)
            
            for e, n_chunks in zip(self.conf.epochs, self.conf.chunks_per_session):
                p = s.paradigm.label_slice(e)
                neus = s.neurons.get_neuron_type(self.conf.neuron_types) \
                    .time_slice(p.starts[0], p.stops[0])
                
                if self.conf.sleep_selection is not None:
                    neus = neus.behav_slice(s.brainstates, self.conf.sleep_selection, 
                                            tighten=self.conf.tight_epoch)
                
                if self.conf.ripple_selection is not None:
                    neus = neus.behav_slice(s.ripples, self.conf.ripple_selection, 
                                            tighten=self.conf.tight_epoch)
                
                if n_chunks > 1:
                    neus_list = neus.time_split(n_chunks=n_chunks)  # Returns list
                    # Store each chunk separately
                    for chunk_id, chunk_neus in enumerate(neus_list):
                        key = Key(session=ssn, epoch=e, chunk=chunk_id)
                        self[key] = chunk_neus
                else:
                    key = Key(session=ssn, epoch=e,chunk=0)
                    self[key] = neus

    def __str__(self):
        s = str(self.conf) + "\ndata:\n"
        # Group by session for organized display
        by_session = self.group_by('session')
        for (session,), session_data in sorted(by_session.items()):
            s += f"  Session {session}:\n"
            by_epoch = self.group_by('epoch')
            for (epoch,), epoch_data in sorted(by_epoch.items()):
                matching = [k for k in session_data.keys() if k.epoch == epoch]
                if matching:
                    s += f"    Epoch {epoch}: {len(matching)} entries\n"
        return s

    def get_neurons(self, session: str = None, epoch: str = None, 
                    chunk: int = None):
        """
        Convenience method to get neurons with optional filtering.
        Returns single Neurons object or dict of matching entries.
        """
        results = self.filter(session=session, epoch=epoch, chunk=chunk, 
                            analysis_type='neurons')
        
        if len(results) == 1:
            return list(results.values())[0]
        return results
    

class Jitter:
    def __init__(self, neurons:Neurons, ref_inds:Union[int,list[int]], target_ind:int,
                 jscale=5,njitter=100):
        self.ref_inds = _san(ref_inds)
        self.target_ind = target_ind
        self.neurons = neurons
        self.jitters = []
        self.jsigs = []
        self.real_ccg = []
        self.jitter_ccg = []
        self.pval = 0
        self.significances = 0
        self.thresholds = 0
        self.JBSI = 0
        self.jscale = jscale
        self.njitter = njitter
        self.jtype = "interval"

        # All reference neurons should be somewhat 'similar',
        # meaning the same operation can be applied on them.
        # Here we require them to have the same neuronal type
        assert len(set(neurons.neuron_type[self.ref_inds]))==1
    
    def print_config(self):
        print(self._conf)

    def clear(self):
        self.jitters = None
        self.jsigs = None
        self.real_ccg = None
        self.jitter_ccg = None
        self.pval = None
        self.significances = None
        self.thresholds = None
        self.JBSI=None

    @property
    def inds(self):
        return np.concatenate([self.ref_inds, self.target_ind])

    @property
    def n_ref(self):
        return len(self.ref_inds)
    
    @property
    def target_type(self):
        return self.neurons.neuron_type[self.target_ind][0]
    
    @property
    def target_id(self):
        return self.neurons.neuron_ids[self.target_ind]

    @property
    def ref_type(self):
        return self.neurons.neuron_type[self.ref_inds[0]][0]

    def routine(self, ccg_conf):
        if self.jtype == 'interval':
            self.add_interval_jitter()
        else:
            self.add_jitter()
        self.run_ccg_jitter(ccg_conf)
        self.jbsi(ccg_conf)

    def add_jitter(self):
        """
        Spike timing jitter.
        Randomly shift each spike in target spike train
        """
        b = self.target_ind
        target_nspikes = self.neurons.n_spikes[b]
        target_spiketrain = self.neurons.spiketrains[b]
        sampling_rate = self.neurons.sampling_rate

        if self.use_cupy:
            jittertrains = (
                cp.round(
                    (
                        cp.array(target_spiketrain)
                        + 2 * self.jscale * 1e-3 * cp.random.rand(self.njitter,target_nspikes)
                        - 1 * self.jscale * 1e-3
                    )
                    * sampling_rate
                )
                / sampling_rate
            ).get()
        else:
            jittertrains = (
                np.round(
                    (
                        target_spiketrain
                        + 2 * self.jscale * 1e-3 * np.random.rand(self.njitter,target_nspikes)
                        - 1 * self.jscale * 1e-3
                    )
                    * sampling_rate
                )
                / sampling_rate
            )
        self.jitters = list(jittertrains)
        cp.get_default_memory_pool().free_all_blocks()
    
    def add_interval_jitter(self):        
        sampling_rate = self.neurons.sampling_rate
        b = self.target_ind
        target_nspikes = self.neurons.n_spikes[b]
        jscale_samples = int(self.jscale * sampling_rate)
        # example: jscale = 5ms, sampling rate = 30KHz, jscale in samples = 150
        
        # from https://github.com/aamarasingham/bjitter/blob/master/Figure2.m
        if self.use_cupy:
            jittertrains = (
                cp.sort(cp.floor(
                    (cp.floor(
                        cp.round(cp.array(self.neurons.spiketrains[b]) * sampling_rate) 
                        / jscale_samples
                    ) + cp.random.rand(self.njitter,target_nspikes)) * jscale_samples 
                ))
                / sampling_rate
            ).get()
        else:
            jittertrains = (
                np.sort(np.floor(
                    (np.floor(
                        np.round(np.array(self.neurons.spiketrains[b]) * sampling_rate) 
                        / jscale_samples
                    ) + np.random.rand(self.njitter,target_nspikes)) * jscale_samples 
                ))
                / sampling_rate
            ).get()            
        self.jitters = list(jittertrains)

    def run_ccg_jitter(self, cconf:CCGConfig):
        """
        CCGs are shaped (N0,1,nbins)
        """
        print("debug",self.ref_inds,self.target_ind)
        
        neurons = self.neurons.neuron_slice(neuron_inds=self.ref_inds)
        j = Neurons(spiketrains=self.jitters,
            t_start=self.neurons.t_start,
            t_stop=self.neurons.t_stop,
            neuron_ids=[self.target_id]*self.njitter,
            neuron_type=[self.target_type]*self.njitter
            ) # TODO not copying over other fields
        neurons.merge(j)

        # run ccg
        self.real_ccg=correlations.spike_correlations(
                neurons=self.neurons,
                ref_neuron_inds=self.ref_inds,
                neuron_inds=self.target_ind,
                bin_size=cconf.bin_size,
                window_size=cconf.duration,
                use_cupy=cconf.use_cupy,
                symmetrize=cconf.symmetrize_ccg,
                bin_mode=cconf.bin_align,
            )
        
        self.jitter_ccg=correlations.spike_correlations(
                neurons=neurons,
                ref_neuron_inds=np.arange(self.n_ref),
                neuron_inds=self.n_ref+np.arange(self.njitter),
                bin_size=cconf.bin_size,
                window_size=cconf.duration,
                use_cupy=cconf.use_cupy,
                symmetrize=cconf.symmetrize_ccg,
                bin_mode=cconf.bin_align,
            )
        # Debugging - 'debug' should be all zeros (two methods are identical)
        # orig = correlations.spike_correlations(
        #         neurons=neurons,
        #         neuron_inds=np.arange(neurons.n_neurons),
        #         bin_size=bin_size,
        #         window_size=duration,
        #         use_cupy=use_cupy,
        #         symmetrize=True,
        #         bin_mode=bin_mode,
        #     )
        # debug = orig[0,len(ref_inds):]-ccg_all[0]
        # print(debug)

    def jitter_significance(self, EI):
        """
                EI: if 'E', use p-vals for peaks, else use q-vals for troughs
        # TODO
        # ccg_all: (N0, njitter+1, Nbins)
        # pval = (N0, Nbins) where real data is ranked among fake data. conservative when there are ties
        # thresholds = (N0, Nbins)

        """
        if EI=='E':
            pval = np.argsort(np.argsort(-self.ccg,axis=1,kind="stable"),axis=1)[:,0]/self.njitter
            thresholds = np.percentile(self.ccg[:,1:], 100*(1-self.alpha), axis=1)
        else:
            pval = np.argsort(np.argsort(self.ccg,axis=1,kind="stable"),axis=1)[:,0]/self.njitter
            thresholds = np.percentile(self.ccg[:,1:], 100*(self.alpha), axis=1)

        self.significances = pval<=self.alpha
        self.thresholds=thresholds

    def jbsi(self,cconf:CCGConfig):
        """
        Jitter-based synchrony index  Agmon (2012)
        """
        assert self.real_ccg is not None and self.jitter_ccg is not None

        if self.n_ref>1:
            jitter_ccg_avg = np.mean(self.jitter_ccg[:,1:],axis=1) # (N0, Nbins) averaged over Njitter columns
        else:
            jitter_ccg_avg = np.mean(self.jitter_ccg[1:],axis=0) # (1, Nbins) averaged over Njitter rows
        n1 = np.minimum(self.neurons.firing_rate[self.target_ind],
                            self.neurons.firing_rate[self.ref_inds])[..., None] # (N0,1) or (1,1)

        ts = cconf.bin_size
        tj = self.jscale
        b = tj/(tj-ts) if tj/ts>2 else 2
        self.JBSI =  b/n1*(self.real_ccg - jitter_ccg_avg) # (N0, Nbins) or (1, Nbins)

class JitterDataset:
    def __init__(self, neurons: Neurons, session_name:str, inds:np.ndarray, conf:JitterConfig, 
                 epoch:str=None, chunk_id:int=1, EI:str=None, conn_type:tuple=None,
                 ):
        """Note that jitter dataset stores single session/epoch data because jitter computation is memory consuming"""
        self.neurons = neurons
        self.session_name = session_name
        self.epoch = epoch
        self.chunk_id = chunk_id
        self.EI = EI
        self.conn_type = conn_type
        self.inds = inds # (n,2)
        self.significance = []
        self._conf = conf
        self.data = [] # data key is target index
        self.jref_inds = []
        self.jtgt_inds = []
        self.get_jitter_inputs()

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self,conf):
        ans = input("Changing configuration will remove existing jitters. Proceed? [y/n]").lower()
        if ans=='n' or ans=='no':
            print("Aborted")
            return
        self._conf = conf
        self.significance = []
        self.data = []
        self.jref_inds = []
        self.jtgt_inds = []

    @property
    def filepath(self):
        return f'~/Documents/jitter_out/jitter-{self.session_name}_{self.epoch}{self.chunk_id}_{self.conn_type}-{self.EI}.h5'

    def save(self):
        with h5py.File(self.filepath, "a") as f:
            f.create_dataset(self.session_name, data=self.data)
            print(f"saved {self.conn_type}")

    def load(self):
        with h5py.File(self.filepath, "r") as f:
            self.data = f[self.session_name][:]
            print(f"loaded jitter data {self.conn_type}")

    @property
    def get_jitter_inputs(self):
        """Reshape coordinates of (ref,target) pairs into most efficient format for jittering
        grouped by target indices"""
        keys, inv = np.unique(self.inds[:,1], return_inverse=True)
        self.jref_inds = [self.inds[inv==i,0].tolist() for i in range(len(keys))]
        self.jtgt_inds = keys
    
    def run_jitter(self):
        for refs,tgt in zip(self.jref_inds,self.jtgt_inds):
            jd = Jitter(neurons=self.neurons,
                            ref_inds=refs,
                            target_ind=tgt,
                            conf=self.conf).routine()
            self.data.append(jd)


class CCG:
    """Like Neurons, but for CCGs
    Static dataclass, not mean for reuse"""
    def __init__(self, key, ccg, ids, inds, pred=None, pval=None, jsig=None,
                 conf:CCGConfig=None):
        self.key=key
        self.ids=ids
        self.inds=inds
        self.ccg=ccg
        self.pred=pred
        self.pval=pval
        self.jsig=jsig
        self._conf=conf

    def __str__(self):
        s = self.conf.__str__()
        for key, val in self.__dict__.items():
            if isinstance(val,np.ndarray) or isinstance(val,list):
                s+=f"{key}: {val[0]}...\n"
            else:
                s+=f"{key}: {val}\n"
        return s
    
    @property
    def conf(self):
        return self._conf

    @property
    def total(self):
        return self.ccg.shape[0]
    
    def plotdir(self, root):
        if self.key.conn_type is None:
            return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.chunk}/{self.key.excitability}_any"
        return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.chunk}/{self.key.excitability}_{self.key.conn_type[0]}-{self.key.conn_type[1]}"    

    def sample_plot(self,inds):
        assert inds in self.inds
        pval = self.pval[inds] if self.pval else None
        pred = self.pred[inds] if self.pred else None
        jsig = self.jsig[inds] if self.jsig else None
        plot_ccg_only(
            ccg=self.ccg[inds], 
            ids=self.ids[inds], 
            inds=inds, 
            window_size=self.conf.duration, 
            bin_size=self.conf.bin_size, 
            pval=pval, pred=pred, jsig=jsig,
            mode='even' if self.conf.bin_align=='edge' else 'odd',
        )

    def save_plots(self, neuron_types, waveforms, firing_rate, frates_all, root):
        assert self.ccg is not None
        plotdir = self.plotdir(root)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir,exist_ok=True)

        s=np.argsort(self.inds[:,1]) #[np.random.random_integers(0,coords.shape[0]-1,5)]
        for i,inds in enumerate(self.inds[s]):
            #TODO self doesn't have all the indices so the they're overflowing
            # might be better to use ids?
            plot_ccg_figure(ids=self.ids[s][i],
                            inds=inds,
                            neuron_types=neuron_types[s][i] if neuron_types is not None else None,
                            frate_cut=firing_rate[s][i] if firing_rate is not None else None,
                            frates_all=frates_all[s][i] if frates_all is not None else None,
                            waveforms=waveforms[s][i] if waveforms is not None else None,
                            ccg=np.array(self.ccg)[s][i], 
                            plotdir=plotdir, 
                            window_size=self.conf.duration*1e3,
                            bin_size=self.conf.bin_size*1e3,
                            pval=self.pval[s][i] if self.pval is not None else None,
                            pred=self.pred[s][i] if self.pred is not None else None,
                            jsig=self.jsig[s][i] if self.jsig else None,
                            mode='odd',
                            show=False,save=True)


class CCGDataset(AnalysisDataset):
    def __init__(self, nd:NeuronsDataset, conf:CCGConfig=None):
        self.nd = nd
        self.data={}
        self._conf=conf
        self.spurious={}

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self,conf):
        ans = input("Changing configuration will remove existing CCG data. Proceed? [y/n]").lower()
        if ans=='n' or ans=='no':
            print("Aborted")
            return
        self._conf = conf
        self.data={}
        self.spurious={}
    
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

    def run_eranconv(self):
        """
        Run CCG and convolution based significance test for all neurons

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        print("EranConv significant pairs")
        def _s(sess_name, neurons):
            neurons=_san(neurons)
            s = f"======={sess_name}=======\n"
            s+=f"Chunk(s) are {neurons[0].total_time_hours:.2f}h each and contain {[f'{_.effective_time_hours:.2f}' for _ in neurons]} hours of actual sleep "
            for _ in self.nd.conf.neuron_types: # sleep chunk
                s+=f"{_}={neurons[0].get_neuron_type(_).n_neurons} "
            s+="\n"
            return s
                
        for key, neurons in self.nd.data.items():
            ccgs, spurs, printstr = eranconv_group(neurons=neurons,conf=self.conf,key=key)
            self.append_ccg(key, ccgs)
            self.append_spurious(key, spurs)
            print(_s(key.session,neurons)+printstr)

    def rescale_eranconv(self,bin_size,duration=None,jscale=None,include_spurious=False,):
        """
        Run CCG and convolution based significance test for all neurons

        Params
        -----
        ccg_config: Parameters for CCG, contains all configurations
        nd: Neurons dataset, contains all input data
        """
        old_bin_size = self._conf.bin_size
        self._conf.bin_size = bin_size
        if duration: self._conf.duration = duration
        if jscale: self._conf.jscale = jscale # 
        print(f"recalculated CCG from binsize={old_bin_size} to binsize={bin_size}")

        print("EranConv significant pairs")
        def _s(sess_name, neurons):
            neurons=_san(neurons)
            s = f"======={sess_name}=======\n"
            s+=f"Chunk(s) are {neurons[0].total_time_hours:.2f}h each and contain {[f'{_.effective_time_hours:.2f}' for _ in neurons]} hours of actual sleep "
            for _ in self.nd.conf.neuron_types: # sleep chunk
                s+=f"{_}={neurons[0].get_neuron_type(_).n_neurons} "
            s+="\n"
            return s
        
        for key, ccg in self.data.items():
            neurons = self.nd.data[key.parent()]
            ccgs = eranconv_rescale(neurons=neurons,conf=self.conf,inds=ccg.inds)
            self.data[key].ccg = [ccgs[i,i] for i in range(ccgs.shape[0])] #TODO right not it only replaces ccg, not p values and conv
            self.data[key].pval=None
            self.data[key].pred=None
            self.data[key].jsig=None # TODO these are temporary

        print("rescale completed")

        if not include_spurious:
            return

        for key, spur in self.spurious.items():
            neurons = self.nd.data[key.parent()]
            ccgs = eranconv_rescale(neurons=neurons,conf=self.conf,inds=spur.inds)
            self.spur[key].ccg = ccgs
            self.data[key].pval=None
            self.data[key].pred=None
            self.data[key].jsig=None # TODO these are temporary
        print("rescale of spurious CCG completed")
        
    def save_plots(self, jd: JitterDataset = None,
                   root="/home/selinali/Documents/NeuroPy/images/ccg_plots",
                   frates_all=None):
        for key, ccg in self.data.items():
            neurons = self.nd.data[key.parent()]
            frates = frates_all[key.parent()] if frates_all else None
            for EI in ['E', 'I']:
                for ct in self.conf.conn_types[EI]:
                    try:
                        if jd is not None:
                            ccg.jsig = jd.data[key].significance
                        ccg.save_plots(
                            neuron_types=neurons.neuron_type[ccg.inds],
                            waveforms=neurons.waveforms[ccg.inds] if neurons.waveforms is not None else None,
                            firing_rate=neurons.firing_rate[ccg.inds] if neurons.firing_rate is not None else None,
                            frates_all=frates[ccg.inds] if frates is not None else None,
                            root=root
                        )
                    except Exception as e:
                        print(f"{key.session}: No {ct} connections {e}")
                        continue
        print("done")

    def save_plots_spurious(self, root="/home/selinali/Documents/NeuroPy/images/ccg_plots", frates_all=None):
        for key, ccg in self.spurious.items():
            neurons = self.nd[key.parent()]
            frates = frates_all[key.parent()] if frates_all else None
            for EI in ['E', 'I']:
                for ct in self.conf.conn_types[EI]:
                    try:
                        ccg.save_plots(
                            neuron_types=neurons.neuron_type[ccg.inds],
                            waveforms=neurons.waveforms[ccg.inds] if neurons.waveforms is not None else None,
                            firing_rate=neurons.firing_rate[ccg.inds] if neurons.firing_rate is not None else None,
                            frates_all=frates[ccg.inds] if frates is not None else None,
                            root=root)
                    except Exception as e:
                        print(f"{key.session}: No {ct} connections {e}")
                        continue
        print("done")


def routine_mean_firing_rates(nd:NeuronsDataset):
    n_chunks=nd.conf.chunks_per_session
    epochs=nd.conf.epochs
    total_n_chunks = np.sum(n_chunks)
    neuron_types = nd.conf.neuron_types
    ntypes = len(neuron_types)
    alpha = 0.05

    print("Mean firing rates P VALUES")
    for key,sess_neurons in nd.data.items():
        
        overview_str=f"======={key.session}=======\n"
        for epoch, n_chunk in zip(epochs, n_chunks):
            labels+=[f"{epoch.capitalize()}{i+1}" for i in range(n_chunk)]

        for itype in range(ntypes):
            ################ UPDATE RETURN VALUES #################       
            nneurons = 0
            mean_firing_rates,\
            sd_firing_rates,\
            iqr,\
            frates,\
            effective_time = [],[],[],[],[]
            ############# END OF UPDATE RETURN VALUES ##############

            ### start of epochs loop ###
            for ie, epoch, n_chunk in enumerate(zip(epochs, n_chunks)):
                ################ UPDATE RETURN VALUES #################       
                mean_firing_rates.append([])
                sd_firing_rates.append([])
                iqr.append([])
                frates.append([])
                effective_time.append([])
                ############# END OF UPDATE RETURN VALUES ##############

                ### start of chunks loop ###
                for ic in range(n_chunk):
                    neus = sess_neurons[epoch][ic]
                    neus = neus.get_neuron_type(neuron_types[itype])
                    nneurons=neus.n_neurons
                    frate = neus.firing_rate if nneurons>0 else 0
                    frates[ie].append(frate)
                    mean_firing_rates[ie].append(np.mean(frate))
                    sd_firing_rates[ie].append(np.std(frate))
                    effective_time[ie].append(neus.effective_time_hours) # time in hours
                    if neus.n_neurons>5:
                        iqr[ie].append(np.percentile(frate, 75)-np.percentile(frate, 25))
                ### end of chunks loop ###

            ################ UPDATE PRINT STRING #################
            overview_str+=f"{itype+1}. {neuron_types[itype]}\t"
            overview_str+=f"n={int(nneurons)}\t"
            overview_str+=f"mean firing rates (Hz)|effective time (h)\n"
            for ts,mfrs in zip(effective_time,mean_firing_rates):
                for t,mfr in zip(ts,mfrs):
                    overview_str+=f"{mfr:.02f}|{t:.02f}  "
            overview_str+="\n"
            if nneurons<2:
                overview_str+="Too few neurons in this category\n"
            else:
                decimal_places=int(2+-np.floor(np.log10(alpha)))
                frates = [xx for x in frates for xx in x]
                flag = False
                for j in range(total_n_chunks):
                    for k in range(j):
                        p = ttest_ind(frates[k],frates[j],equal_var=True).pvalue
                        if p<alpha:
                            flag = True
                            overview_str+=f"{labels[k]} VS SLEEP{labels[j]}\tp={p:.{decimal_places}f}\n"
                        # Standard t-test,  check if mean firing rate changes over sleep per cell type
                if not flag: overview_str+="No significant difference between chunks\n"
            ############# END OF UPDATE PRINT STRING ##############
            ### end of celltype loop ###
            
        print(overview_str)
        ### end of sessions loop ###
    return effective_time, mean_firing_rates, sd_firing_rates, iqr, frates
    ### end of function ###
 

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


def routine_jitter_pairs_after_conv_wrapper(cd:CCGDataset,nd:NeuronsDataset,njitter=100,jitter_type='spike_time',use_cupy=True,jscale=None,filename=None):
    # filename = filename or datetime.now().strftime("%Y-%m-%d-%H-%M")
    jconf = JitterConfig(njitter=njitter,
                            jitter_type=jitter_type,
                            jscale=jscale or cd.conf.jscale,
                            use_cupy=use_cupy)

    for key, ccgs in cd.data.items():
        if isinstance(key,tuple) and len(key)==4:
            sess_name, epoch, EI, conn_type = key
            nkey = (sess_name, epoch)
        elif isinstance(key,tuple) and len(key)==3:
            sess_name, EI, conn_type = key
            epoch = nd.conf.epochs[0]
            nkey = sess_name
        else:
            sess_name = key
            epoch = nd.conf.epochs[0]
            conn_type = cd.conf.conn_types['E'][0] or cd.conf.conn_types['I'][0]
            EI = 'E' if len(cd.conf.conn_types['E'][0])>0 else 'I'
            nkey = None

        print("jitter",sess_name)
        neurons = nd.data[nkey] if nkey else nd.data
        _routine_jitter_pairs_after_conv(sess_name,epoch,EI,conn_type,neurons,conf=jconf,inds=ccgs.inds,ccg_conf=ccgs.conf)


def _routine_jitter_pairs_after_conv(sess_name,epoch,EI,conn_type,neurons:Neurons,conf:JitterConfig,inds,ccg_conf:CCGConfig):
        neurons=_san(neurons)
        for c,neu in enumerate(neurons):
            jd = JitterDataset(neurons=neu,
                            session_name=sess_name,
                            inds=inds,
                            conf=conf,
                            epoch=epoch,
                            chunk_id=c,
                            EI=EI,
                            conn_type=conn_type,)
            jd.routine(ccg_conf)
            jd.save()


def eranconv(ccg, W=5, wintype="gauss", hollow_frac=None):
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

def eranconv_rescale(neurons:Neurons, conf:CCGConfig, inds):
    print(inds[:,0],inds[:,1])
    if inds is not None:
        ccgs = correlations.spike_correlations(
                neurons=neurons,
                ref_neuron_inds=inds[:,0],
                neuron_inds=inds[:,1],
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_cupy=conf.use_cupy,
                symmetrize=conf.symmetrize_ccg,
                bin_mode='odd' if conf.bin_align=='center' else 'even',
            )
    print("completed rerun")
    print(ccgs.shape)
    return ccgs

def eranconv_group(key:Key, neurons:Neurons, conf:CCGConfig):
    """
    Helper of eranconv

    chunked_neurons: list[Neurons] or Neurons, sliced from the same session.
    """
    overview_str = ""
    ccgs,spurs = {},{}
    n=neurons.n_neurons
    inds_by_type = {}
    for t in (['pyr','inter']): #TODO hardcoded
        inds_by_type[t]=np.where(neurons.neuron_type==t)

    ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(n), # all
            bin_size=conf.bin_size,
            window_size=conf.duration,
            use_cupy=conf.use_cupy,
            symmetrize=conf.symmetrize_ccg,
            bin_mode='odd' if conf.bin_align=='center' else 'even',
        )
    
    pvals, pred, qvals = eranconv(ccg,W=conf.jscale,wintype="gauss",hollow_frac=None)

    def _multiple_correction(pvals,alpha,method='bonferroni'):
        # TODO should bump this to utils or something
        # methods: 'fdr_bh', 'bonferroni'
        p_flat = pvals.ravel()
        sig, p_corr, _, _ = multipletests(p_flat, alpha=alpha, method=method)
        sig = sig.reshape(pvals.shape)
        p_corr = p_corr.reshape(pvals.shape)   
        # sig = (pvals[...,C+start:C+end+1] < corrected_alpha)
        return sig,p_corr
    
    sig, p_correct = _multiple_correction(pvals, conf.alpha)
    coords_excitatory = np.argwhere((sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1))

    sig1, q_correct1 = _multiple_correction(qvals, conf.alpha)
    sig2, q_correct2 = _multiple_correction(qvals, conf.alpha2)
    neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1)) # significant bins must have a semi-significant neighbor
    coords_inhitibitory = np.argwhere(neighbor.any(-1))

    coords_spkcount = np.argwhere((ccg[...,conf.min_spkcnt_bin:conf.max_spkcnt_bin]>=conf.min_spkcount).all(axis=-1))
    
    coordsE = _intersect2d(coords_excitatory, coords_spkcount, n)
    coordsI = _intersect2d(coords_inhitibitory, coords_spkcount, n)

    def _count_significant_pairs(coords,neurons,conn_types,ignore_same_electrodes=True):
        """
        Create a tally of significant neuronal connectoins by type
        Currently, the type is defined as 
            reference-target/[E,I]
        where reference is presynaptic, and target is postsynaptic neuronal type, 
        and E/I indicates the connection being excitatory or inhibitory

        SL: If this helper function seems messy it's probably because 
        it pertains to our specific definition of significant pairs (see Diba 2014, Pairwise connections.)
        """
        all_sig_pairs = []
        list_empty = True
        if coords.shape[0]:
            # Condition 1: Ref/Target are never on the same electrode
            if ignore_same_electrodes:
                diff_channel=np.where(neurons.peak_channels[coords[:,0]]!=neurons.peak_channels[coords[:,1]])[0]
                coords = coords[diff_channel]
            # Condition 2: Specify Ref/Target cell types
            for (ref,target) in conn_types:
                sig_pairs=np.where(np.isin(coords[:,0],inds_by_type[ref]) & 
                                    np.isin(coords[:,1],inds_by_type[target]))[0]
                all_sig_pairs.append(coords[sig_pairs])
            list_empty = not np.any([_.shape[0] for _ in all_sig_pairs])
        return all_sig_pairs,list_empty

    ### start of celltype loop ###
    excitatory_pairs,list_emptyE = _count_significant_pairs(coordsE,neurons,conf.conn_types_E,ignore_same_electrodes=conf.ignore_same_electrodes)
    inhibitory_pairs,list_emptyI = _count_significant_pairs(coordsI,neurons,conf.conn_types_I,ignore_same_electrodes=conf.ignore_same_electrodes)
    ### end of celltype loop ###
    n_E, n_I = coordsE.shape[0], coordsI.shape[0]

    ################ UPDATE PRINT STRING #################
    def _printstr_sig(all_sig_pairs, conn_types, EI):
        # if any type of connection under consideration has a non-zero count, print a summary
        s=""
        if np.any([_.shape[0] for _ in all_sig_pairs]):
            for sig_pairs,(ref,target) in zip(all_sig_pairs,conn_types):
                s+=f"{ref}-{target}/{EI} {f'{sig_pairs.shape[0]:02d}' if sig_pairs.shape[0] else '-'} | "
        if s=="":
            s=f"no {'excitatory' if EI=='E' else 'inhbitory'} connections  "
        return s
    sE = _printstr_sig(excitatory_pairs, conf.conn_types_E, 'E')
    sI = _printstr_sig(inhibitory_pairs, conf.conn_types_I, 'I')
    overview_str += f"SLEEP{key.chunk}: E/I pairs {n_E:03d} / {n_I:03d} | "
    if list_emptyE and list_emptyI:
        overview_str+="no connections\n"
    else:
        overview_str=overview_str+sE+sI+"\n"

    ################ UPDATE RETURN VALUES #################
    for excitability, conn_types, pair_inds, coords in zip(['E','I'],
                                                           [conf.conn_types_E,conf.conn_types_I],
                                                           [excitatory_pairs,inhibitory_pairs],
                                                           [coordsE,coords_inhitibitory]):
        significance = pvals if excitability=='E' else qvals
        skey=Key(session=key.session,epoch=key.epoch,chunk=key.chunk,excitability=excitability)

        if not pair_inds:
            for conn_type,inds in zip(conn_types,pair_inds):
                ckey=Key(session=key.session,epoch=key.epoch,chunk=key.chunk,conn_type=conn_type,excitability=excitability)
                ccgs[ckey] = None
            spurs[skey] = None
            continue

        spurious_inds = coords
        for conn_type,inds in zip(conn_types,pair_inds):
            ckey=Key(session=key.session,epoch=key.epoch,chunk=key.chunk,conn_type=conn_type,excitability=excitability)

            spurious_inds = _setdiff2d(spurious_inds,inds,n)

            if inds is None or len(inds)==0:
                ccgs[ckey] = None
            else:
                ccgs[ckey] = CCG(inds=inds, 
                                ids=neurons.ind2id(inds), 
                                ccg=ccg[inds[:,0],inds[:,1]], 
                                pred=pred[inds[:,0],inds[:,1]], 
                                pval=significance[inds[:,0],inds[:,1]], 
                                conf=conf,
                                key=ckey)                

        inds = np.asarray(spurious_inds)
        if inds is None or len(inds)==0:
            spurs[skey] = None
        else:
            spurs[skey] = CCG(inds=inds, 
                            ids=neurons.ind2id(inds), 
                            ccg=ccg[inds[:,0],inds[:,1]], 
                            pred=pred[inds[:,0],inds[:,1]], 
                            pval=pvals[inds[:,0],inds[:,1]], 
                            conf=conf,
                            key=skey)

    ### end of chunks loop ###
    return ccgs, spurs, overview_str


# TODO: move to plotting in the future!
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, 
                   pval=None, pred=None, jsig=None, mode='even'):
    """Single CCG plot into provided axis"""
    if mode == 'even':
        bins = np.arange(-window_size / 2, window_size / 2, bin_size) + bin_size / 2
    else:
        bins = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)

    ax.bar(bins, ccg, width=bin_size, alpha=0.5, label="ccg")
    if pred is not None:
        ax.bar(bins, pred, width=bin_size, alpha=0.5, label="ccg-smooth")
    if pval is not None:
        ax.plot(bins, pval * np.max(ccg), label='p')
    if jsig is not None:
        ax.plot(bins, jsig * np.max(ccg), label='j-significance')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    X, Y = ids; x, y = inds
    ax.set_title(f"CCG, neuron_ids=[{X},{Y}], indices=[{x},{y}]")
    ax.legend()
    sns.despine(ax=ax)
    return ax


def plot_waveform_panel(ax, waveform, neuron_type, neuron_id, 
                        frate_all=None, frate_cut=None, ch_per_shank=16):
    """Single waveform panel into provided axis"""
    max_ch = waveform.shape[0]
    ax.imshow(waveform.astype(float))
    ax.set_title(f"{neuron_type}{neuron_id}")
    xlabel = ""
    if frate_all is not None:
        xlabel += f"{frate_all:.2f}Hz all "
    elif frate_cut is not None:
        xlabel += f"{frate_cut:.2f}Hz cut "
    ax.set_xlabel(xlabel)
    for k in range(max_ch // ch_per_shank):
        ax.axhline((k + 1) * ch_per_shank, c='w', alpha=0.5, linestyle='dashed')
    return ax


def plot_ccg_figure(ccg, ids, inds, neuron_types, waveforms,
                    window_size, bin_size, pval=None, pred=None, jsig=None, 
                    frates_all=None, frate_cut=None, mode='even', 
                    show=True, save=False, plotdir=None):
    """Full figure: CCG + 2 waveforms"""
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

    plot_ccg_panel(axs[0], ccg, ids, inds, window_size, bin_size, pval, pred, jsig, mode)
    labels = ['ref', 'target']
    for i in range(2):
        plot_waveform_panel(axs[1+i], waveforms[i], neuron_types[i], ids[i],
                            frates_all[i] if frates_all is not None else None,
                            frate_cut[i] if frate_cut is not None else None)

    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
    if show:
        plt.show()
    plt.close(fig)
    return fig

def plot_ccg_only(ccg, ids, inds, window_size, bin_size, pval=None, pred=None, jsig=None, 
                  show=True, save=False, plotdir=None, mode='even'):
    """Save only the CCG plot without waveforms"""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, pval, pred, jsig, mode)
    
    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
    if show:
        plt.show()
    plt.close(fig)
    return fig
