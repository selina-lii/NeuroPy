"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

import numpy as np
# try:
#     import cupy as cp
# except ImportError:
#     print("Error importing CuPy")
#     cp = None

import neuropy.analyses.correlations as correlations
from neuropy.core.neurons import Neurons
from scipy.signal import windows
from scipy.stats import poisson, ttest_ind
from scipy import ndimage
from typing import Union, Optional, Dict, Any, Tuple
import h5py
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


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


@dataclass(frozen=True)
class Key:
    session: Optional[str] = None
    epoch: Optional[str] = None
    ref_ind: Optional[int] = None
    target_ind: Optional[int] = None
    chunk: Optional[int] = None
    excitability: Optional[str] = None
    conn_type: Optional[tuple[str,str]] = None

    """
    Dependencies
    tuple(session, epoch, ... , chunk) should alway be present
    conn_type -> excitability
    ref_ind hyperfocus on the relations between ONE neuron and the rest of the population.
        Do not set a ref_ind unless focusing on analysis relative to one reference neuron
        All other neurons in a NeuronDataset are targets; remove irrelevant neurons before constructing NeuronDataset.
    """

    def __str__(self):
        parts = []
        if self.session: parts.append(f"{self.session}")
        if self.epoch: parts.append(f"{self.epoch}")
        if self.ref_ind is not None: parts.append(f"ref{self.ref_ind}")
        if self.target_ind is not None: parts.append(f"_tgt{self.target_ind}")
        if self.chunk is not None: parts.append(f"c{self.chunk}")
        if self.excitability: parts.append(f"{self.excitability}")
        if self.conn_type: parts.append(f"{self.conn_type[0]}-{self.conn_type[1]}")
        return "_".join(parts) if parts else "root"
    
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

    def parent(self) -> 'Key':
        """Get parent key (one level up in hierarchy)"""
        if self.excitability is not None: # conn_type goes with excitability
            return Key(self.session, self.epoch, chunk=self.chunk)
        if self.chunk is not None:
            return Key(self.session, self.epoch, self.ref_ind, self.target_ind)
        if self.ref_ind is not None:
            return Key(self.session, self.epoch)
        if self.epoch is not None:
            return Key(self.session)
        return Key()

    def subkey(self, *dimensions) -> 'Key':
        return Key(**{dim: getattr(self, dim, None) for dim in dimensions})
 
    def remove(self,*dimensions) -> 'Key':
        return Key(**{f: getattr(self, f) for f in self.__dataclass_fields__ if f not in dimensions})


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
            groups[key.subkey(*dimensions)][key] = value
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


class Toggle(Enum):
    """
    Set operation on a variable during filtering of neural data
    """
    NONE = 0
    SELECT = 1
    REMOVE = 2


class NeuronsDatasetConfig:
    """
    Metadata of NeuronsDataset

    tight_time: bool
    if true, try to shrink start and end of epoch to where brainstates are happening 

    n_chunks: int
    Splits session time axis into equal-lengthed blocks if >1

    """
    def __init__(self,
                 name:str = "default",
                 neuron_types:Union[list[str], str, None] = ['pyr', 'inter'], 
                 epochs:Union[list[str], str, None]="post", 
                 n_chunks:Union[list[int], int]=1, 
                 chunk_stride:int=None,
                 chunk_len:int=None,
                 spikecount_per_group:int=None,
                 sleep_labels:Union[list[str], str]=["REM","NREM"], 
                 ripple:Toggle=Toggle.NONE, tight_epoch=False):
        self.name = name
        self.session_names = []
        self.neuron_types = _san(neuron_types)
        self.epochs = _san(epochs) or [None] # each epoch gets their own neurons object
        self.sleep_labels = _san(sleep_labels)
        self.ripple = ripple
        self.n_chunks = _san(n_chunks)
        self.tight_epoch = tight_epoch
        self.chunk_stride = chunk_stride
        self.chunk_len = chunk_len
        self.spikecount_per_group = spikecount_per_group

        assert len(self.n_chunks)==len(self.epochs)

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
        return self.n_chunks[idx]


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
    # TGT_FRATE = 3
    # TGT_SPKS = 4 # I don't think we're doing those yet


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


#TODO am i using this?
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
    A collection of neurons wrapped for analysis
    Arguments of the analysis should be provided using a NeuronsDatasetConfig object

    sessions: subjects.ProcessData
        collection object of sessions
    """
    def __init__(self, sessions, conf:NeuronsDatasetConfig):
        self.conf = conf        
        self.data={}
        self.edge_timestamps={} # TODO n_chunks is used for normalizing connection strengths

        self.prep(sessions)

    def __str__(self): #TODO untested
        s = str(self.conf) + "\ndata:\n"
        by_session = self.group_by('session')
        for k1, session_data in sorted(by_session.items()):
            s += f"  Session {k1.session}:\n"
            by_epoch = session_data.group_by('epoch')
            for k2, epoch_data in sorted(by_epoch.items()):
                s += f"    Epoch {k2.epoch}: {len(epoch_data)} entries\n"
        return s

    def get_neurons(self, session: str = None, epoch: str = None):
        """
        Convenience method to get neurons with optional filtering.
        Returns single Neurons object or dict of matching entries.
        """
        results = self.filter(session=session, epoch=epoch, analysis_type='neurons')
        
        if len(results) == 1:
            return list(results.values())[0]
        return results
    
    def prep(self, sessions):
        sessions = _san(sessions)
        for s in sessions:
            ssn = _short_session_name(s)
            self.conf.session_names.append(ssn)

            for i, e in enumerate(self.conf.epochs):
                neus = s.neurons

                # Filter neurons
                # -type
                if self.conf.neuron_types is not None:
                    neus = neus.get_neuron_type(self.conf.neuron_types)
                # -paradigm label
                if self.conf.epochs[i] is not None:
                    p = s.paradigm.label_slice(e)
                    neus = neus.time_slice(p.starts[0], p.stops[0])
                # -sleep states
                if self.conf.sleep_labels is not None:
                    neus = neus.behav_slice(s.brainstates, self.conf.sleep_labels, 
                                            tighten=self.conf.tight_epoch)
                # -ripple states
                if self.conf.ripple==Toggle.SELECT:
                    neus = neus.behav_slice(s.ripple,
                                            tighten=self.conf.tight_epoch,
                                            min_dur=0) # NOTE not selecting ripple duration for now
                elif self.conf.ripple==Toggle.REMOVE:
                    non_ripple = s.ripple.time_invert_selection(t_start=p.starts[0],t_stop=p.stops[0])
                    neus = neus.behav_slice(non_ripple,
                                            tighten=self.conf.tight_epoch,
                                            min_dur=0) # NOTE not selecting ripple duration for now

                # Store filtered neurons
                key = Key(session=ssn, epoch=e)
                self.data[key] = neus
                N = neus.n_neurons

                # Define how to segment each neurons group for comparative analysis
                # see neurons.py: 
                #    _edges_time_split  time_split
                #   _edges_time_window  time_windows
                #   _edges_spikecount   spikecount_split

                if self.conf.spikecount_per_group is not None:
                    for i in range(N):
                        k = Key(session=ssn, epoch=e, ref_ind=i)
                        edges = neus._edges_spikecount(i=i,
                                                          n=self.conf.spikecount_per_group,
                                                          discard_tail=False)
                        self.edge_timestamps[k]=edges
                elif self.conf.chunk_stride is not None and self.conf.chunk_len is not None:
                    self.edge_timestamps[key] = neus._edges_time_window(stride=self.conf.chunk_stride, 
                                                                        chunk_len=self.conf.chunk_len) 
                elif self.conf.n_chunks is not None and self.conf.n_chunks[i] > 1:
                    self.edge_timestamps[key] = neus._edges_time_split(n_chunks=self.conf.n_chunks)
                else:
                    self.edge_timestamps[key] = None


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
    # since chunks will be abolished, ngroups dimension should always exist, even when redundant
    # chunks is a dimension such that all items in the list can have the same operations applied on them
    #   as chunks are purely for comparison analysis
    def __init__(self, key, ccg, ids, inds, 
                 ccg_null=None, pval=None, j_sig=None, conn_strength=None,
                 conf:CCGConfig=None,significant=True):
        self.key=key
        self.ids=ids
        self.inds=inds
        self.ccg=ccg
        self.ccg_null=ccg_null # 'baseline', or jittered, chance level CCG
        self.pval=pval
        self.j_sig=j_sig
        self.conn_strength = conn_strength # 
        self._conf=conf
        n=self.inds.shape[0]
        if type(significant)==bool:
            self.significant=np.full((n,),significant) # by default, all ccgs in this variable are significant
        else:
            # assert significant.shape==(n,)#TODO
            self.significant=significant

    def __str__(self):
        s = self.conf.__str__()
        for key, val in self.__dict__.items():
            if isinstance(val,np.ndarray) or isinstance(val,list):
                s+=f"{key}: {val[0]}...\n"
            else:
                s+=f"{key}: {val}\n"
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
        cs = np.sum(auc[:,self.conf.min_lag_bin:self.conf.max_lag_bin],axis=1) # (inds,)
        if norm_factor is not None: cs = (cs.astype(float) / norm_factor.T).ravel() # e.g. presynaptic element firing rate
        self.conn_strength = cs # divided by presynaptic firing rate

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
            return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.chunk}/{self.key.excitability}_any"
        return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.chunk}/{self.key.excitability}_{self.key.conn_type[0]}-{self.key.conn_type[1]}"    

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

    def save_plots(self, neuron_types, waveforms, firing_rate, frates_all, root):
        assert self.ccg is not None
        plotdir = self.plotdir(root)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir,exist_ok=True)

        s=np.argsort(self.inds[:,-1]) #[np.random.random_integers(0,inds.shape[0]-1,5)]
        for i,inds in enumerate(self.inds[s]):
            print(i,inds)
            #TODO self doesn't have all the indices so the they're overflowing
            # might be better to use ids?
            # TODO Think I fixed this, but I don't remember the fix
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
                            ccg_null=self.ccg_null[s][i] if self.ccg_null is not None else None,
                            j_sig=self.j_sig[s][i] if self.j_sig else None,
                            show=False,save=True)

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


class Connectivity:
    def __init__(self, key, n_chunks, ccgs, inds, ids, exist:np.ndarray[bool], strength:np.ndarray[float]):
        self.key=key
        self.inds=_san(inds,as_np=True)
        self.ccgs=ccgs
        self.exist=exist # has to be a separate field, cannot be inferred from strength because a strength might not be significant
        self.strength = strength # 
        self.coordinates = None #TODO physical neuron coordinates
        self.n_chunks = n_chunks
        self.ids = ids
        
    def __repr__(self):
        s = str(self.key) + "\n"
        for i,inds in enumerate(self.inds):
            s += f"{str(inds):<15}\tIn chunks {str(np.where(self.exist[i])[0]):<20}\tstrengths: {self.strength[i]}\n"
        return s
        

    def filter(self,min_n_chunk=None,skips=None):
        if skips is not None: 
            inds = [int(i) for i,(v,e) in enumerate(zip(self.inds,self.exist)) if (v not in skips) and (np.sum(e)>=min_n_chunk)]
        else:
            inds = np.where(np.sum(self.exist,axis=1)>=min_n_chunk)[0].astype(int)
        return inds

    def plot_strength(self,
                      n_chunks_threshold=None,save=False,
                        norm_by_n_sess=False,
                        norm_by_total_strength=False,
                        z_score=False,
                        show_legend=False,
                        skips=None):
        n_chunks_threshold=n_chunks_threshold if n_chunks_threshold is not None else self.n_chunks
        plt.figure()
        inds = self.filter(min_n_chunk=n_chunks_threshold,skips=skips)
        plot_data = self.strength[inds]
        significant=self.exist[inds]
        pairs = self.inds[inds]
        if pairs.shape[0]==0: 
            print(f"{self.key}: No pairs fit the criteria min_n_chunk={n_chunks_threshold}, nothing is plotted")
            return

        if norm_by_total_strength:
            plot_data/=np.nansum(plot_data,axis=1,keepdims=True)
        if norm_by_n_sess:
            plot_data=plot_data*np.count_nonzero(~np.isnan(plot_data),axis=1,keepdims=True)/self.n_chunks
        if z_score:
            plot_data=(plot_data - np.nanmin(plot_data,axis=1,keepdims=True)) / np.nanmean(plot_data,axis=1,keepdims=True)
        colors = plt.cm.hsv(np.linspace(0, 1, plot_data.shape[0]))
        legend_keys = []
        for i, (pair, v, c, sig) in enumerate(zip(pairs,plot_data,colors,significant)):
            plt.plot(v,c=c,alpha=0.3)  # normalized
            x_sig = np.where(sig)[0]
            plt.scatter(x_sig, v[x_sig], s=8, c=c,label="_nolegend_")
            if show_legend: legend_keys.append(f"{i}:{pair}")
        plt.title(f"{self.key}")
        plt.xlabel("epoch id")
        plt.xticks(np.arange(self.n_chunks),np.arange(self.n_chunks))
        plt.ylabel("normalized connection strength")
        if show_legend: 
            # spacing
            ncol = 1+int(i//25)
            i_per_col=i//ncol
            offset = -.3-.5*(i_per_col/25)
            plt.legend(legend_keys,loc='right', bbox_to_anchor=(1, offset), ncol=ncol)
        plt.show()
        #TODO save

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

    def get_CCG_with_baseline(self, method="eran_conv"):
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

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self,conf):
        # ans = input("Changing configuration will remove existing CCG data. Proceed? [y/n]").lower()
        # if ans=='n' or ans=='no':
        #     print("Aborted")
        #     return
        self._conf = conf
        # self.data={}
        # self.spurious={}
        # self.auto={}
        # self.connectivity={}
    
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
    
    def merge_CCGs(self, cd_list: list):
        # TODO
        new_dataset = CCGDataset(conf=self.conf)
        for cd in cd_list:
            (cd.ccgs.items())# split into key list and values list
            new_dataset.append_ccg(base_key, cd.ccgs.items())
            new_dataset.append_spurious(base_key, cd.spurious)
            new_dataset.append_spurious(base_key, cd.auto) # do not append if none
        return new_dataset

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
        def _s(sess_name, neurons): # print helper
            neurons=_san(neurons)
            s = f"======={sess_name}=======\n"
            s+=f"Chunk(s) are {neurons[0].total_time_hours:.2f}h each and contain {[f'{_.effective_time_hours:.2f}' for _ in neurons]} hours of actual sleep "
            for _ in self.nd.conf.neuron_types: # sleep chunk
                s+=f"{_}={neurons[0].get_neuron_type(_).n_neurons} "
            s+="\n"
            return s

        conv = EranConv()
        for key, neurons in self.nd.data.items():
            ccgs, spurs, autos, printstr = conv.eranconv_1st_pass(key=key, 
                                        neurons=neurons, 
                                        chunk_edges=self.nd.edge_timestamps[key], 
                                        conf=self.conf)
            self.append_ccg(key, ccgs)
            self.append_spurious(key, spurs)
            self.append_auto(key, autos)
            print(_s(key.session,neurons)+printstr)

    def _reCCG(self,key:Key,
               indices_source=CCGIndexSource.SIGNIFICANT,
               inds=None,
               ):
        """
        Rerun CCG given list of indices

        Call one of the wrappers instead
        """
        chunk_edges=None
        
        # groups=self.group_by('session','epoch',source='connectivity')
        if inds is not None:
            k = Key(session=key.session,epoch=key.epoch,ref_ind=key.ref_ind,chunk=key.chunk)
            chunk_edges = self.nd.edge_timestamps[k]

        elif indices_source==CCGIndexSource.SIGNIFICANT:
            inds = self.data[key].inds

        elif indices_source==CCGIndexSource.SPURIOUS:
            inds = self.spurious[key].inds

        elif indices_source==CCGIndexSource.SIGNIFICANT_ANY:
            inds = self.connectivity[key].inds

        elif indices_source==CCGIndexSource.AUTOCORRELOGRAMS:
            inds = np.vstack([self.auto[key].inds,self.auto[key].inds]).T                  

        conv = EranConv()
        data_key = key.subkey('session','epoch')
        ccgs = conv.eranconv_2nd_pass(key=key, 
                            neurons=self.nd.data[data_key], # session, epoch, conn_type
                            pair_inds=inds,
                            chunk_edges=chunk_edges or self.nd.edge_timestamps[data_key], 
                            conf=self.conf)
        return ccgs

    def reCCG_timescale(self,bin_size,duration=None,jscale=None,include_spurious=False):
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
            if ccg is not None:
                self._reCCG(indices_source=CCGIndexSource.SIGNIFICANT,ccg_key=key)
        print("rescale completed")

        if include_spurious: 
            for key, spur in self.spurious.items():
                if spur is not None:
                    self._reCCG(indices_source=CCGIndexSource.SPURIOUS,ccg_key=key)
        print("rescale of spurious CCG completed")

    def reCCG_connectivity(self):
        """
        Rerun CCG with list of pairs that had been significant in any chunk
        """
        for key, ccg in self.connectivity.items():
            if ccg is not None:
                new_ccgs = self._reCCG(indices_source=CCGIndexSource.SIGNIFICANT_ANY,key=key)
                for k,ccg in new_ccgs.items():
                     # inherit original significant markers
                    ccg.significant = self.connectivity[key].exist[:,k.chunk]
                    self.data[k] = ccg
        print("recomputed CCG for pairs that had been significant in any chunk")
    
    def reCCG_connectivity_pairwise(self,conn_type=('pyr','pyr')):
        """
        TODO group by reference
        """
        group = self.group_by('conn_type',source='connectivity')[Key(conn_type=conn_type)]
        self.data={}
        for key, ccg in group.items():
            if ccg is not None:
                for i,pair in enumerate(ccg.inds):
                    print(f"reCCG: {i}/{ccg.inds.shape[0]}")
                    k = Key(session=key.session,epoch=key.epoch,ref_ind=pair[0],target_ind=pair[1],
                            chunk=key.chunk,excitability=key.excitability,conn_type=key.conn_type)
                    new_ccgs = self._reCCG(key=k,
                                           inds=pair)
                    for k,new_ccg in new_ccgs.items():
                        self.data[k] = new_ccg
        print("recomputed CCG using spike count chunks")

    def save_plots(self, jd = 'JitterDataset', root="~/Documents/NeuroPy/images/ccg_plots",
                   conn_types:list=None):
        assert os.path.isdir(os.path.expanduser(root))
        if isinstance(jd,str): jd = None # TODO ugly. to avoid circular imports
        keys = self.keys_matching(conn_type=conn_types) if conn_types else self.data.keys()
        print(keys)
        print(f"Saving plots under {root}")
        for key in keys:
            ccg = self.data[key]
            neurons = self.nd[key.subkey('session','epoch')]
            chunk_edges = self.nd.edge_timestamps[key.subkey('session','epoch')]
            if chunk_edges is not None:
                t_start= chunk_edges[0][key.chunk]
                t_end=chunk_edges[1][key.chunk]
                firing_rate = None if neurons.firing_rate is None else neurons.time_slice(t_start,t_end).firing_rate[ccg.inds]
            else:
                firing_rate = None if neurons.firing_rate is None else neurons.firing_rate[ccg.inds]
            # frates = frates_all[key.subkey('session','epoch')] if frates_all else None
            print(f"ccg {key.session} {key.conn_type}")
            # try:
            if jd is not None:
                ccg.j_sig = jd.data[key].significant
            ccg.save_plots(
                neuron_types=neurons.neuron_type[ccg.inds],
                waveforms=None if neurons.waveforms is None else neurons.waveforms[ccg.inds],
                frates_all=None if neurons.firing_rate is None else neurons.firing_rate[ccg.inds],
                firing_rate=firing_rate,
                root=root,
            )
                # firing_rate=None if frates is None else frates[ccg.inds],
            # except Exception as e:
            #     print(f"No {key.conn_type} connections: {e}")
            #     continue
        print("done")

    def save_plots_spurious(self, root="~/Documents/NeuroPy/images/ccg_plots", frates_all=None, EI:list=None):
        assert os.path.isdir(root)
        keys = self.keys_matching(excitability=EI) if EI else self.spurious.keys()
        print(f"Saving plots under {root}")
        for key in keys:
            ccg = self.spurious[key]
            neurons = self.nd[key.parent()]
            if EI is not None and key.excitability not in EI: continue
            print(f"spurious {key.session} {key.conn_type}")
            frates = frates_all[key.parent()] if frates_all else None
            try:
                ccg.save_plots(
                    neuron_types=neurons.neuron_type[ccg.inds],
                    waveforms=None if neurons.waveforms is None else neurons.waveforms[ccg.inds],
                    firing_rate=None if neurons.firing_rate is None else neurons.firing_rate[ccg.inds],
                    frates_all=None if frates is None else frates[ccg.inds],
                    root=root)
            except Exception as e:
                print(f"{key.session}: No {key.excitability} spurious connections {e}")
                continue
        print("done")

    def normalize_by_ref(self):
        """
        Normalize CCGs by reference firing rate (or the number of reference spikes?)
        """
        # TODO modifies data TODO untested TODO normalize to reference firing rate
        for d in self.data:
            d.normalize(self.nd.data[d.key.parent()].n_spikes[d.inds[:,-2]])
        for d in self.spurious:
            d.normalize(self.nd.data[d.key.parent()].n_spikes[d.inds[:,-2]])

    def set_connection_strengths(self, method="eran_conv"):
        """
        Set value for each CCG() object of self.data based on given method.
        Values can be found in self.data[i].conn_strengths.
        """
        #TODO untested
        for key, ccg in self.data.items():
            k=Key(key.session,key.epoch)
            if ccg is None: continue # no connection
            if self.conf.normalize.name == NormalizeBy.REF_FRATE.name:
                norm_factors = self.nd[k].firing_rate[ccg.ref_inds][...,np.newaxis]
            elif self.conf.normalize.name==NormalizeBy.REF_SPKS.name:
                norm_factors = self.nd[k].n_spikes[ccg.ref_inds][...,np.newaxis]
            else:
                norm_factors = None

            if method=="eran_conv":
                ccg._set_cs_eranconv(norm_factors)
            elif method=="tail":
                spikecount = self.nd[key.parent()].n_spikes
                acg = self.auto[key.parent()]
                ccg._set_cs_tail(acg,spikecount,norm_factors=norm_factors)
                return NotImplementedError("Unknown connection strength method")

    def set_connection_strengths_compound(self, method="eran_conv"):
        """
        Set value for each CCG() object of self.data based on given method.
        Values can be found in self.data[i].conn_strengths.
        """
        for key, ccg in self.data.items():
            k=Key(key.session,key.epoch)
            if ccg is None: continue # no connection
            if self.conf.normalize.name == NormalizeBy.REF_FRATE.name:
                norm_factor = self.nd[k].firing_rate[ccg.ref_ind]
            elif self.conf.normaliz.name==NormalizeBy.REF_SPKS.name:
                norm_factor = self.nd[k].n_spikes[ccg.ref_ind]
            else:
                norm_factor = None
            ccg._set_cs_eranconv_compound(norm_factor)

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
            n = 1 if self.nd.edge_timestamps[k.parent()] is None else len(self.nd.edge_timestamps[k.parent()][0])
            for keyy, _ in grouped[k].items():
                i_chunk = keyy.chunk
                if keyy.conn_type in conn_types:
                    ccg=self.data[keyy]
                    if ccg is None: continue # no connections
                    for ind,val,sig in zip(ccg.inds,ccg.conn_strength,ccg.significant):
                        p = tuple(ind)
                        if p not in pairs:
                            pairs.append(p)
                            exist[p]=np.full(n,False)
                            strength[p]=np.full(n,np.nan)
                        exist[p][i_chunk]=sig
                        strength[p][i_chunk]=val
            exist_arr = np.zeros((len(pairs),n))
            strength_arr = np.zeros((len(pairs),n))
            for i,p in enumerate(pairs):
                exist_arr[i]=exist[p]
                strength_arr[i]=strength[p]
            conn_key = Key(k.session,k.epoch,conn_type=k.conn_type,excitability=k.excitability)
            neurons = self.nd.data[Key(k.session,k.epoch)]
            self.connectivity[conn_key] = Connectivity(key=conn_key, n_chunks=n, ccgs=group,
                                                       inds=pairs, exist=exist_arr, strength=strength_arr,
                                                       ids = neurons.ind2id(pairs))

    # def set_connectivity_compound(self):
    #     """
    #     Create Connectivity objects within this class from connection strengths.
    #     Sess class Connecitivity().
    #     """
    #     for k, ccg in self.data.items():
    #         chunk_key = Key(session=k.session, epoch=k.epoch, ref_ind=k.ref_ind)
    #         n = len(self.nd.edge_timestamps[chunk_key][0])
    #         neurons = self.nd.data[Key(k.session,k.epoch)]
    #         self.connectivity[k] = Connectivity(key=k, n_chunks=n, ccgs=ccg,
    #                                                    inds=ccg.inds, exist=ccg.significant, strength=ccg.conn_strength,
    #                                                    ids = neurons.ind2id(ccg.inds))

    # TODO move to plotting
    def plot_connection_strengths(self,n_chunks_threshold=None,save=False,
                                            norm_by_n_sess=False,
                                            norm_by_total_strength=False,
                                            z_score=False,
                                            show_legend=False,
                                            skips={}):
        for k1, conn in self.connectivity.items():
            conn.plot_strength(n_chunks_threshold=n_chunks_threshold,
                                           save=save,
                                           norm_by_n_sess=norm_by_n_sess,
                                           norm_by_total_strength=norm_by_total_strength,
                                           z_score=z_score,
                                           show_legend=show_legend,
                                           skips=skips.get(k1))
            print(k1,skips.get(k1))

    def plot_connection_strengths_compound(self,n_chunks_threshold=None,save=False,
                                            norm_by_n_sess=False,
                                            norm_by_total_strength=False,
                                            z_score=False,
                                            show_legend=False,
                                            skips={},
                                            legend_group_size=25):
            self.plot_strength_compound(n_chunks_threshold=n_chunks_threshold,
                                           save=save,
                                           norm_by_n_sess=norm_by_n_sess,
                                           norm_by_total_strength=norm_by_total_strength,
                                           z_score=z_score,
                                           show_legend=show_legend,
                                           legend_group_size=legend_group_size
                                           )

    def plot_strength_compound(self,
                      n_chunks_threshold=None,save=False,
                        norm_by_n_sess=False,
                        norm_by_total_strength=False,
                        z_score=False,
                        show_legend=False,
                        legend_group_size=25,
                        ):
        inds=[]
        strength=[]
        x_coords=[]
        sig=[]
        for k, ccg in self.data.items():
            edges = self.nd.edge_timestamps[k.subkey('session','epoch','ref_ind')]
            xcoord = np.array([(ts+te)/7200 for ts,te in zip(edges[0],edges[1])])
            inds.append(ccg.inds)
            strength.append(ccg.conn_strength)
            sig.append(ccg.significant)
            x_coords.append(xcoord)
        inds=np.array(inds)
        x_ticks = list(np.arange(13))

        # n_chunks_threshold=n_chunks_threshold if n_chunks_threshold is not None else self.n_chunks
        plt.figure()
        # inds = self.filter(min_n_chunk=n_chunks_threshold,skips=skips)
        plot_data = strength
        pairs = inds
        if pairs.shape[0]==0: 
            print(f": No pairs fit the criteria min_n_chunk={n_chunks_threshold}, nothing is plotted")
            return

        if norm_by_total_strength:
            plot_data=[x/np.nansum(x) for x in plot_data]
        if norm_by_n_sess:
            plot_data=[x*np.count_nonzero(si)/len(x) for x,si in zip(plot_data,sig)]
        if z_score:
            plot_data=[(x - np.nanmin(x)) / np.nanmean(x) for x in plot_data]
        colors = plt.cm.hsv(np.linspace(0, 1, len(plot_data)))
        legend_keys = []
        for i, (pair, x,y, c,si) in enumerate(zip(pairs,x_coords,plot_data,colors,sig)):
            plt.plot(x,y,c=c,alpha=0.3)
            si=np.array(si)
            if si.any():plt.scatter(x[si], y[si], s=8,c=c,label="_nolegend_")
            if show_legend: legend_keys.append(f"{i}:{pair}({len(x)})")
        plt.title(f"{k.subkey('session','epoch','excitability','conn_type')}")
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


class Jitterlet:
    def __init__(self, key:Key, neurons:Neurons, noj_inds:Union[int,list[int]], j_ind:int, conf:JitterConfig):
        """
        A set of neuronal pairs defined for efficient computation. 

        Pairs are required to have:

        1. The same target neuron
            (we jitter the target neuron spiketrain)

        2. The same reference neuronal type 
            Multiple references are okay.
            this is to ensure significance criteria are uniform 
            i.e. we test for either excitation or inhibition.
        
        You should never have to create instances of this class because it will be 
        taken care of by Jitter().
        """
        self.key = key
        self.noj_inds = _san(noj_inds)
        self.j_ind = j_ind
        self.neurons = neurons
        self.j_spktrains = []
        self.j_ccg = []
        self.conf = conf
        self.__datastate = DataState.DEFAULT
        assert len(set(neurons.neuron_type[self.noj_inds]))==1
    
    @property
    def datastate(self):
        return self.__datastate
    @datastate.setter
    def datastate(self,v:DataState):
        if self.__datastate==DataState.READONLY: return
        self.__datastate=v
        
    @property
    def inds(self):
        return np.concatenate([self.noj_inds, self.j_ind])
    @property
    def ref_inds(self):
        return self.noj_inds
    @property
    def target_ind(self):
        return self.j_ind

    @property
    def n_ref(self):
        return len(self.noj_inds)
    
    @property
    def target_type(self):
        return self.neurons.neuron_type[self.j_ind][0]
    
    @property
    def target_id(self):
        return self.neurons.neuron_ids[self.j_ind]

    @property
    def ref_type(self):
        return self.neurons.neuron_type[self.noj_inds[0]][0]

    def add_jitter(self):
        if self.conf.jitter_type == JitterType.INTERVAL:
            self.add_interval_jitter()
        else: # JitterType.SPIKE_TIMING
            self.add_jitter_spike_timing()

    def add_jitter_spike_timing(self):
        """
        Spike timing jitter.
        Randomly shift each spike in target spike train
        """
        b = self.j_ind
        target_nspikes = self.neurons.n_spikes[b]
        target_spiketrain = self.neurons.spiketrains[b]
        sampling_rate = self.neurons.sampling_rate

        if self.conf.use_acceleration:
            jittertrains = (
                cp.round(
                    (
                        cp.array(target_spiketrain)
                        + 2 * self.conf.jscale * cp.random.rand(self.conf.njitter,target_nspikes)
                        - 1 * self.conf.jscale
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
                        + 2 * self.conf.jscale * np.random.rand(self.conf.njitter,target_nspikes)
                        - 1 * self.conf.jscale
                    )
                    * sampling_rate
                )
                / sampling_rate
            )
        self.j_spktrains = list(jittertrains)
        if self.conf.use_acceleration: cp.get_default_memory_pool().free_all_blocks()
    
    def add_interval_jitter(self):        
        sampling_rate = self.neurons.sampling_rate
        b = self.j_ind
        target_nspikes = self.neurons.n_spikes[b]
        jscale_samples = int(self.conf.jscale * sampling_rate)
        # example: jscale_ms = 5ms, sampling rate = 30KHz, jscale in samples = 150
        
        # from https://github.com/aamarasingham/bjitter/blob/master/Figure2.m
        if self.conf.use_acceleration:
            jittertrains = (
                cp.sort(cp.floor(
                    (cp.floor(
                        cp.round(cp.array(self.neurons.spiketrains[b]) * sampling_rate) 
                        / jscale_samples
                    ) + cp.random.rand(self.conf.njitter,target_nspikes)) * jscale_samples 
                ))
                / sampling_rate
            ).get()
        else:
            jittertrains = (
                np.sort(np.floor(
                    (np.floor(
                        np.round(np.array(self.neurons.spiketrains[b]) * sampling_rate) 
                        / jscale_samples
                    ) + np.random.rand(self.conf.njitter,target_nspikes)) * jscale_samples 
                ))
                / sampling_rate
            )            
        self.j_spktrains = list(jittertrains)
        if self.conf.use_acceleration: cp.get_default_memory_pool().free_all_blocks()

    def run_ccg_jitter(self):
        """
        CCGs are shaped (N0,1,nbins)
        """
        print("debug",self.noj_inds,self.j_ind)

        neurons = self.neurons.neuron_slice(neuron_inds=self.noj_inds)
        j = Neurons(spiketrains=self.j_spktrains,
            t_start=self.neurons.t_start,
            t_stop=self.neurons.t_stop,
            neuron_ids=[self.target_id]*self.conf.njitter,
            neuron_type=[self.target_type]*self.conf.njitter
            ) # TODO not copying over other fields
        neurons.merge(j)
        
        self.j_ccg=correlations.spike_correlations(
                neurons=neurons,
                ref_neuron_inds=np.arange(self.n_ref),
                neuron_inds=self.n_ref+np.arange(self.conf.njitter),
                bin_size=self.conf.ccg.bin_size,
                window_size=self.conf.ccg.duration,
                use_acceleration=self.conf.ccg.use_acceleration,
                symmetrize=self.conf.ccg.symmetrize_ccg,
            )
        # Debugging - 'debug' should be all zeros (two methods are identical)
        # orig = correlations.spike_correlations(
        #         neurons=neurons,
        #         neuron_inds=np.arange(neurons.n_neurons),
        #         bin_size=bin_size,
        #         window_size=duration,
        #         use_acceleration=use_acceleration,
        #         symmetrize=True,
        #     )
        # debug = orig[0,len(noj_inds):]-ccg_all[0]
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
            pval = np.argsort(np.argsort(-self.j_ccg,axis=1,kind="stable"),axis=1)[:,-2]/self.conf.njitter
            thresholds = np.percentile(self.j_ccg[:,1:], 100*(1-self.conf.alpha), axis=1)
        else:
            pval = np.argsort(np.argsort(self.ccg,axis=1,kind="stable"),axis=1)[:,-2]/self.njitter
            thresholds = np.percentile(self.ccg[:,1:], 100*(self.alpha), axis=1)

        self.j_sig = pval<=self.conf.alpha
        self.thresholds=thresholds

    def jbsi(self,real_ccg):
        """
        Jitter-based synchrony index  Agmon (2012)
        """
        assert self.j_ccg is not None

        j_ccg_avg = np.mean(self.j_ccg,axis=1) # (N0, Nbins) averaged over Njitter columns
        n1 = np.minimum(self.neurons.firing_rate[self.j_ind],
                            self.neurons.firing_rate[self.noj_inds])[..., None] # (N0,1) or (1,1)

        ts = self.conf.ccg.bin_size
        tj = self.conf.jscale

        b = tj/(tj-ts) if tj/ts>2 else 2
        JBSI =  b/n1*(real_ccg - j_ccg_avg) # (N0, Nbins) or (1, Nbins)
        return JBSI
    
    def spktrain_path(self): # TODO
        get_path_from_key(self.key)
        pass

    def ccg_path(self):
        get_path_from_key(self.key)
        pass

    def save(self):
        with h5py.File(self.spktrain_path, "a") as f:
            f.create_dataset(self.j_ind, data=self.j_spktrains)
            print(f"saved jitter spiketrains {self.key}:{self.j_ind}")
        with h5py.File(self.ccg_path, "a") as f:
            f.create_dataset(self.j_ind, data=self.j_ccg)
            print(f"saved jitter ccgs {self.key}:{self.j_ind}")

    def load(self):
        # loaded data 
        with h5py.File(self.spktrain_path, "r") as f:
            self.j_spktrains = f[self.j_ind][:]
            print(f"loaded jitter spiketrains {self.key}:{self.j_ind}")
        with h5py.File(self.ccg_path, "r") as f:
            self.ccg_path = f[self.j_ind][:]
            print(f"loaded jitter ccgs {self.key}:{self.j_ind}")
        self.neurons = None
        self.conf = None
        self.datastate=DataState.READONLY # data was loaded not computed. cannot recompute bc neurons/conf will not be saved


class Jitter:
    def __init__(self, key:Key, neurons: Neurons, conf:JitterConfig, ccg:CCG, root:str=None):
        """Single session/epoch jitters
        Note: Jitter computation is time and memory consuming!"""
        self.key = key
        self.conf = conf

        self.jref_inds = []
        self.jtgt_inds = []
        self.pos = {} # TODO name
        self.pval = []
        self.significant = []
        self.threshold = []
        self.JBSI = []
        self.jitterlets = {}

        self.neurons = neurons
        self.ccg = ccg

        self.root = root or f"~/Documents/jitter_out"
        self.get_jitter_inputs()
    
    @property
    def n_inds(self):
        return len(self.ccg.inds)

    def get_jitter_inputs(self):
        """Reshape coordinates of (ref,target) pairs into most efficient format for jittering
        grouped by target indices"""
        keys, inv = np.unique(self.ccg.inds[:,-1], return_inverse=True)
        self.jref_inds = [self.ccg.inds[inv==i,0].tolist() for i in range(len(keys))]
        self.jtgt_inds = keys
        self.pos = {k: np.where(inv == i)[0] for i, k in enumerate(keys)}
    
    def get(self,ref,tgt,field='jitters'):
        # TODO untested
        return getattr(self.data[tgt], field)[np.where(self.data[tgt].noj_inds==ref)[0]]

    def run(self,save_progress=False):
        self.JBSI = np.zeros((self.n_inds,self.ccg.conf.nbins))
        for refs, tgt in zip(self.jref_inds,self.jtgt_inds):
            self.jitterlets[tgt] = Jitterlet(key=self.key,
                                        neurons=self.neurons,
                                       noj_inds=refs,
                                       j_ind=tgt,
                                       conf=self.conf)
        for tgt,j in self.jitterlets.items():
            j.add_jitter()
            j.run_ccg_jitter()
            self.JBSI[self.pos[tgt]] = j.jbsi(self.ccg.ccg[self.pos[tgt]]) # TODO indexing
        if save_progress:
            self.save()

    @property
    def filepath(self):
        return f'{self.root}/jitter-{str(self.key)}.h5'

    def save(self,intermediates=False):
        if intermediates:
            for k,v in self.jitterlets.items():
                v.save()
        with h5py.File(self.filepath, "a") as f:
                f.create_dataset((self.session_name,field), data=getattr(self, field, None))
        print(f"saved {self.key}")

    def load(self,intermediates=False):
        if intermediates:
            for k,v in self.jitterlets.items():
                v.load()
        with h5py.File(self.filepath, "r") as f:
            for field in ['key','pval','JBSI','conf']:
                d = f[(self.session_name,field)][:]
                setattr(self, field, d)
        print(f"loaded jitter data {self.key}")
   

class JitterDataset:
    def __init__(self, nd: Neurons, cd: CCGDataset, conf:JitterConfig):
        """Note that jitter dataset stores single session/epoch data because jitter computation is memory consuming"""
        self._conf = conf
        self.conf.ccg=cd.conf
        self.nd = nd
        self.cd = cd
        self.data = {} # data key is target index

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self,conf):
        # ans = input("Changing configuration will remove existing jitters. Proceed? [y/n]").lower()
        # if ans=='n' or ans=='no':
        #     print("Aborted")
        #     return
        self._conf = conf
        # self.data = []

    @property
    def filepath(self):
        return f'~/Documents/jitter_out/'

    def save(self):
        for k,v in self.data.items():
            v.save(root=self.filepath)

    def load(self):
        for k,v in self.data.items():
            v.load(root=self.filepath)

    def run_jitter(self,save_progress=True):
        for key, ccg in self.cd.data.items():
            if ccg is None: 
                self.data[key] = None
            else:
                neurons = self.nd.data[key.parent()]
                self.data[key]=Jitter(key=key,
                                    neurons=neurons,
                                    conf=self.conf,
                                    ccg=ccg)
        for _,j in self.data.items():
            if j is not None: j.run(save_progress=save_progress)


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
    mask_basic={}
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

    def intersect(self, coords1, coords2):
        # Intersection of coordinate lists
        ravel_dims = (self.n,self.n) if coords1.shape[-1]==2 else (self.n_chunks,self.n,self.n) #TODO
        coords1_flat = np.ravel_multi_index(coords1.T, ravel_dims)
        coords2_flat = np.ravel_multi_index(coords2.T, ravel_dims)
        coords_flat = np.intersect1d(coords1_flat, coords2_flat)
        coords = np.array(np.unravel_index(coords_flat, ravel_dims)).T
        return coords

    def setdiff(self, coords1,coords2):
        # Set difference of coordinate lists
        ravel_dims = (self.n,self.n) if coords1.shape[-1]==2 else (self.n_chunks,self.n,self.n)
        flat1 = np.ravel_multi_index(coords1.T, ravel_dims)
        flat2 = np.ravel_multi_index(coords2.T, ravel_dims)
        flat  = np.setdiff1d(flat1, flat2)
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
    def multiple_correction(pvals,alpha,method='bonferroni'):
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
    
    def significance_mask(self,excitability):
        conf = self.conf
        autocorr_locations = EranConv.get_autocorr_locations(self.pvals.shape)
        if excitability=='E':
            sig, self.pval_corrected = EranConv.multiple_correction(self.pvals, conf.alpha)
            sig_mask = np.argwhere((sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1) & ~autocorr_locations)
        elif excitability=='I':
            sig1, self.qval_corrected = EranConv.multiple_correction(self.qvals, conf.alpha)
            sig2, self.qval_corrected2 = EranConv.multiple_correction(self.qvals, conf.alpha2)
            neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1))  # significant bins must have a significant-ish neighbor
            sig_mask = np.argwhere(neighbor.any(-1) & ~autocorr_locations)
        return sig_mask

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

    def _regroup_basic(self,pairs):
        groups = defaultdict(lambda: defaultdict(list))
        for i in range(self.n_chunks): 
            for EI in ['E','I']:
                groups[i][EI]=[]

        for EI in ['E','I']:
            for row in pairs[EI]:
                if len(row)>2:
                    groups[row[0]][EI].append(row[1:])
                else:
                    groups[0][EI].append(row)

        for i in range(self.n_chunks): 
            for EI in ['E','I']:
                groups[i][EI]=np.array(groups[i][EI])
        return groups

    def __regroup_mask(self,pairs):
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i in range(self.n_chunks): 
            for EI in ['E','I']:
                for conn_type in self.conf.conn_types[EI]:
                    groups[i][EI][conn_type]=[]

        for EI in ['E','I']:
            for conn_type, pair_vals in pairs[EI].items():
                if pair_vals is None: continue
                for row in pair_vals:
                    if len(row)>2:
                        groups[row[0]][EI][conn_type].append(row[1:])
                    else:
                        groups[0][EI][conn_type].append(row)

        for i in range(self.n_chunks): 
            for EI in ['E','I']:
                for conn_type in self.conf.conn_types[EI]:
                    groups[i][EI][conn_type]=np.array(groups[i][EI][conn_type]) 
        return groups
    
    def _group_by_chunk(self):
        self.mask = self.__regroup_mask(self.mask)
        self.mask_basic = self._regroup_basic(self.mask_basic)

    def eranconv_1st_pass(self, key:Key, neurons:Neurons, chunk_edges:np.ndarray, conf:CCGConfig):
        """
        Main function for CCG computatinon
        Call from CCGDataset
        """
        print("running eranconv (1st pass)")
        self.conf = conf
        self.n=neurons.n_neurons
        if chunk_edges is not None:
            self.n_chunks = len(chunk_edges[0])
        else: 
            self.n_chunks = 1

        self.ccg = correlations.spike_correlations(
                neurons=neurons,
                neuron_inds=np.arange(neurons.n_neurons), # all
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                chunk_edges=chunk_edges,
            )

        spkcount_mask = np.argwhere((self.ccg[...,conf.min_spkcnt_bin:conf.max_spkcnt_bin]>=conf.min_spkcount).all(axis=-1)) # NOTE right now it's the same criteria for E/I

        self.pvals, self.pred,self.qvals = EranConv._conv(self.ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

        # Universal criteria of significance
        # These indices pairs are kept as a baseline; anything not fitting downstream criteria is put under 'spurious'

        # Multiple corrections
        self.mask={'E':{},'I':{}}
        self.mask_basic={'E':{},'I':{}}

        self.mask_basic['E'] = self.intersect(self.significance_mask('E'), spkcount_mask)
        self.mask['E'] = self._probe_loc_mask(self.mask_basic['E'],neurons)
        self.mask['E'] = self._cell_type_mask(self.mask['E'],neurons,conf.conn_types_E)

        self.mask_basic['I'] = self.intersect(self.significance_mask('I'), spkcount_mask)
        self.mask['I'] = self._probe_loc_mask(self.mask_basic['I'],neurons)
        self.mask['I'] = self._cell_type_mask(self.mask['I'],neurons,conf.conn_types_I)

        ccgs_by_type, spurs_by_type, acgs = self.process_output(key, neurons)

        overview_str=""
        for i in range(self.n_chunks):
            E_str, hasE = self._printstr_sig(self.mask[i]['E'], 'E')
            I_str, hasI = self._printstr_sig(self.mask[i]['I'], 'I')
            overview_str += f"SLEEP{i}: E/I pairs {self.mask_basic[i]['E'].shape[0]:03d} / {self.mask_basic[i]['I'].shape[0]:03d} | "
            overview_str=overview_str+E_str+I_str+"\n" if (hasE or hasI) else overview_str+"no connections\n"
        print("eranconv (1st pass) done")

        return ccgs_by_type, spurs_by_type, acgs, overview_str

    def process_output(self, key, neurons):
        """
        Post processor organizing eranconv outputs into mergeable formats for CCGDataset
        """
        
        self._group_by_chunk()
        if len(self.ccg.shape)==3: 
            self.ccg=self.ccg[np.newaxis,...]
            self.pred=self.pred[np.newaxis,...]
            self.pval_corrected=self.pval_corrected[np.newaxis,...]
            self.qval_corrected=self.qval_corrected[np.newaxis,...]

        ccgs_by_type,spurs_by_type,acgs = {},{},{} # 1 neuron group -> many connection types

        # Update return values
        for chunk_id in range(self.n_chunks):
            for EI in ['E','I']:
                filtered_pairs=self.mask[chunk_id][EI]
                all_pairs=self.mask_basic[chunk_id][EI]
                p = self.pval_corrected if EI=='E' else self.qval_corrected # TODO TODO not storing corrected p-vals. make it an option!

                spurious_pairs = all_pairs
                for conn_type, pairs in filtered_pairs.items():
                    key = Key(session=key.session,
                            epoch=key.epoch,
                            chunk=chunk_id,
                            conn_type=conn_type,
                            excitability=EI)
                    if pairs is None or len(pairs)==0:
                        ccgs_by_type[key] = None
                    else:
                        spurious_pairs = self.setdiff(spurious_pairs,pairs)
                        x,y = pairs[:,-2],pairs[:,-1]
                        ccgs_by_type[key] = CCG(inds=pairs, 
                                            ids=neurons.ind2id(pairs), 
                                            ccg=self.ccg[chunk_id,x,y], 
                                            ccg_null=self.pred[chunk_id,x,y], 
                                            pval=p[chunk_id,x,y], 
                                            conf=self.conf,
                                            key=key)

                key = Key(session=key.session,
                        epoch=key.epoch,
                        chunk=chunk_id,
                        excitability=EI)
                pairs = np.asarray(spurious_pairs)
                if pairs is None or len(pairs)==0:
                    spurs_by_type[key] = None
                else:
                    x,y = pairs[...,0],pairs[...,1]
                    spurs_by_type[key] = CCG(inds=pairs, 
                                    ids=neurons.ind2id(pairs), 
                                    ccg=self.ccg[chunk_id,x,y], 
                                    ccg_null=self.pred[chunk_id,x,y], 
                                    pval=p[chunk_id,x,y], 
                                    conf=self.conf,
                                    key=key)
        
        autocorr_locations = EranConv.get_autocorr_locations(self.ccg.shape)            
        if self.n_chunks:
            for chunk_id in range(self.n_chunks):
                key = Key(session=key.session,
                        epoch=key.epoch,
                        chunk=chunk_id)
                acgs[key] = ACG(key=key,
                            acg=self.ccg[chunk_id,autocorr_locations[0]],
                            inds=np.arange(neurons.n_neurons),
                            ids=neurons.ind2id(np.arange(neurons.n_neurons)),
                            conf=self.conf)
        else:
            key = Key(session=key.session,
                        epoch=key.epoch,
                        chunk_id=0)
            acgs[key] = ACG(key=key,
                        acg=self.ccg[chunk_id,autocorr_locations[0]],
                        inds=np.arange(neurons.n_neurons),
                        ids=neurons.ind2id(np.arange(neurons.n_neurons)),
                        conf=self.conf)

        return ccgs_by_type, spurs_by_type, acgs
    
    def eranconv_2nd_pass(self, key:Key, neurons:Neurons, pair_inds:list, chunk_edges:np.ndarray, conf:CCGConfig):
        # ref and target indices should be organized by conn type
        print(f"running eranconv (2nd pass): {key}")

        self.conf = conf
        self.n=neurons.n_neurons
        if chunk_edges is not None:
            self.n_chunks = len(chunk_edges[0])
        else: 
            self.n_chunks = 1

        self.conf = conf
        ccgs = {}
        if key.ref_ind is None:
            self.ccg = correlations.spike_correlations(
                    neurons=neurons,
                    neuron_inds=np.arange(neurons.n_neurons),
                    bin_size=conf.bin_size,
                    window_size=conf.duration,
                    use_acceleration=conf.use_acceleration,
                    symmetrize=conf.symmetrize_ccg,
                    chunk_edges=chunk_edges,
                )
            slicer = (..., pair_inds[:, -2], pair_inds[:, -1], slice(None))
            self.ccg=self.ccg[slicer]

            self.pvals, self.pred, self.qvals = EranConv._conv(self.ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)
            self.significance_mask(key.excitability)
            for i in range(self.n_chunks):
                k=Key(session=key.session,epoch=key.epoch,chunk=i,excitability=key.excitability,conn_type=key.conn_type)
                ccgs[k]=CCG(inds=pair_inds, 
                        ids=neurons.ind2id(pair_inds), 
                        ccg=self.ccg[i], 
                        ccg_null=self.pred[i], 
                        pval=self.pval_corrected[i] if key.excitability=='E' else self.qval_corrected[i], 
                        conf=self.conf,
                        key=k)
        else:
            self.ccg = correlations.spike_correlations(
                    neurons=neurons,
                    neuron_inds=pair_inds,
                    bin_size=conf.bin_size,
                    window_size=conf.duration,
                    use_acceleration=conf.use_acceleration,
                    symmetrize=conf.symmetrize_ccg,
                    chunk_edges=chunk_edges,
                )[:,0,1:2] #TODO
            self.pvals, self.pred, self.qvals = EranConv._conv(self.ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

            spkcount_mask = np.argwhere((self.ccg[...,conf.min_spkcnt_bin:conf.max_spkcnt_bin]>=conf.min_spkcount).all(axis=-1)) # NOTE right now it's the same criteria for E/I
            self.mask = self.intersect(self.significance_mask(key.excitability), spkcount_mask)
            if self.mask.any(): self.mask = self._probe_loc_mask(self.mask,neurons)
            sig = np.full((self.ccg.shape[-3],self.ccg.shape[-2]),False) 
            if self.mask.any(): sig[self.mask[:,0],self.mask[:,1]] = True

            k=Key(session=key.session,epoch=key.epoch,chunk=None,ref_ind=key.ref_ind,target_ind=key.target_ind,
                  excitability=key.excitability,conn_type=key.conn_type)
            ccgs[k]=CCG(inds=pair_inds, 
                    ids=neurons.ind2id(pair_inds), 
                    ccg=self.ccg, 
                    ccg_null=self.pred, 
                    pval=self.pval_corrected if key.excitability=='E' else self.qval_corrected, 
                    conf=self.conf,
                    significant=sig.squeeze(),
                    key=k) # merged ccg

        print("done")
        return ccgs
    

# NOTE move to plotting in the future!
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, 
                   pval=None, ccg_null=None, j_sig=None):
    """Single CCG plot into provided axis"""
    bins = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)

    ax.bar(bins, ccg, width=bin_size, alpha=0.5, label="ccg")
    if ccg_null is not None:
        ax.bar(bins, ccg_null, width=bin_size, alpha=0.5, label="ccg-smooth")
    ax2 = ax.twinx()
    if pval is not None:
        ax2.plot(bins, pval * np.max(ccg), label='p')
    if j_sig is not None:
        ax2.plot(bins, j_sig * np.max(ccg), label='j-significance')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    X, Y = ids; x, y = inds
    ax.set_title(f"CCG, neuron_ids=[{X},{Y}], indices=[{x},{y}]")
    ax.legend()
    sns.despine(ax=ax)
    return ax, ax2 #TODO read return vars?


def plot_waveform_panel(ax, waveform, neuron_type, neuron_id, 
                        frate_all=None, frate_cut=None, n_shanks=None, ch_per_shank=None, discard_channels=None):
    """Single waveform panel into provided axis"""
    n_shanks = n_shanks or 12
    ch_per_shank = ch_per_shank or 16 # TODO put hardcoded values elsewhere?
    max_ch = waveform.shape[0]
    ax.imshow(waveform.astype(float))
    ax.set_title(f"{neuron_type}{neuron_id}")
    xlabel = ""
    if frate_all is not None:
        xlabel += f"{frate_all:.2f}Hz all "
    elif frate_cut is not None:
        xlabel += f"{frate_cut:.2f}Hz cut "
    ax.set_xlabel(xlabel)
    
    edges = (np.array(range(n_shanks))+1)*ch_per_shank+1
    if discard_channels is not None:
        shanks = discard_channels // ch_per_shank
        edges = edges - np.cumsum(np.histogram(shanks,np.arange(n_shanks))[0])
        
    for k in edges:
        ax.axhline(k, c='w', alpha=0.5, linestyle='dashed')
    return ax


def plot_ccg_figure(ccg, ids, inds, neuron_types, waveforms,
                    window_size, bin_size, pval=None, ccg_null=None, j_sig=None, 
                    frates_all=None, frate_cut=None, n_shanks=None, ch_per_shank=None,
                    show=True, save=False, plotdir=None):
    """Full figure: CCG + 2 waveforms"""
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

    plot_ccg_panel(axs[0], ccg, ids, inds, window_size, bin_size, pval, ccg_null, j_sig)
    labels = ['ref', 'target']
    for i in range(2):
        axs[1+i] = plot_waveform_panel(axs[1+i], waveforms[i], neuron_types[i], ids[i],
                            frates_all[i] if frates_all is not None else None,
                            frate_cut[i] if frate_cut is not None else None,
                            n_shanks=n_shanks,ch_per_shank=ch_per_shank)

    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
    if show:
        plt.show()
    plt.close(fig)
    assert os.path.exists(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png") #TODO why do we need this?
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


# will fall under WindowSweep
def routine_mean_firing_rates(nd:NeuronsDataset):
    n_chunks=nd.conf.n_chunks
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
 
# routine_connection_strength


