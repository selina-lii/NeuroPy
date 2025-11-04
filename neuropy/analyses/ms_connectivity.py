"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

import numpy as np
try:
    # import os, platform
    # if platform.system() == "Darwin":
    #     os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import jax.numpy as jnp
    import jax.random as jr
except ImportError:
    print("Error importing JAX. No GPU acceleration available.") # Was CuPy
    jnp = None
    jr = None

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
from enum import Enum
import time

def RAND_KEY():
    return jr.PRNGKey(np.random.randint(0,1e10))

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
    excitability: Optional[str] = None
    conn_type: Optional[tuple[str,str]] = None
    sliding_window: Optional[int] = None

    """
    Dependencies
    tuple(session, epoch, chunk) should alway be present
    conn_type -> excitability
    """

    def __str__(self):
        parts = []
        if self.session: parts.append(f"{self.session}")
        if self.epoch: parts.append(f"{self.epoch}")
        if self.chunk is not None: parts.append(f"c{self.chunk}")
        if self.excitability: parts.append(f"{self.excitability}")
        if self.conn_type: parts.append(f"{self.conn_type[0]}-{self.conn_type[1]}")
        if self.sliding_window: parts.append(f"sw{self.sliding_window}")
        return "_".join(parts) if parts else "root"

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
        if self.sliding_window is not None:
            return Key(self.session, self.epoch, self.chunk, self.excitability, self.conn_type)            
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

    chunks_per_session: int
    Splits session time axis into equal-lengthed blocks if >1

    """
    def __init__(self,
                 name:str = "default",
                 neuron_types:Union[list[str], str] = ['pyr', 'inter'], 
                 epochs:Union[list[str], str]="post", 
                 chunks_per_session:Union[list[int], int]=1, 
                 sleep_labels:Union[list[str], str]=["REM","NREM"], 
                 ripple:Toggle=Toggle.NONE, tight_epoch=False):
        self.name = name
        self.session_names = []
        self.neuron_types = _san(neuron_types)
        self.epochs = _san(epochs)
        self.sleep_labels = _san(sleep_labels)
        self.ripple = ripple
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


        # example configs
        msconn_args = {
            'min_lag':0,
            'max_lag':1,
            'min_spkcount':2.5,
            'spkcount_scope':12,
            'ignore':IgnoreLevel.NONE,
            'ref_type':'pyr',
            'target_type':'pyr',
            'p':0.05,
        }
        excit_args = {
            'min_lag':1,
            'max_lag':3,
            'min_spkcount':2.5,
            'spkcount_scope':12,
            'ignore':IgnoreLevel.NONE,
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
            'ignore':IgnoreLevel.NONE,
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
        
        self.prep(sessions)

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
    
    def prep(self, sessions):
        sessions = _san(sessions)
        for s in sessions:
            ssn = _short_session_name(s)
            self.conf.session_names.append(ssn)
            
            for e, n_chunks in zip(self.conf.epochs, self.conf.chunks_per_session):
                p = s.paradigm.label_slice(e)
                neus = s.neurons.get_neuron_type(self.conf.neuron_types) \
                    .time_slice(p.starts[0], p.stops[0])
                
                if self.conf.sleep_labels is not None:
                    neus = neus.behav_slice(s.brainstates, self.conf.sleep_labels, 
                                            tighten=self.conf.tight_epoch)
                
                if self.conf.ripple==Toggle.SELECT:
                    neus = neus.behav_slice(s.ripple,
                                            tighten=self.conf.tight_epoch,
                                            min_dur=0) # NOTE not selecting ripple duration for now
                elif self.conf.ripple==Toggle.REMOVE:
                    non_ripple = s.ripple.time_invert_selection(t_start=p.starts[0],t_stop=p.stops[0])
                    neus = neus.behav_slice(non_ripple,
                                            tighten=self.conf.tight_epoch,
                                            min_dur=0) # NOTE not selecting ripple duration for now
                
                if n_chunks > 1:
                    neus_list = neus.time_fracture(n_chunks=n_chunks)  # Returns list 
                    # Store each chunk separately
                    for chunk_id, chunk_neus in enumerate(neus_list):
                        key = Key(session=ssn, epoch=e, chunk=chunk_id)
                        self[key] = chunk_neus
                else:
                    key = Key(session=ssn, epoch=e,chunk=0)
                    self[key] = neus


class NeuronsDatasetChange(NeuronsDataset):
    def change_within_epoch(self):
        # self.conf.n_chunks
        pass # TODO maybe in a new class that has all 3 types of data, bc jitter and ccg are involved


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
    * Shouldn't really be called on its own. Wrap in a CCGDataset!"""
    def __init__(self, key, ccg, ids, inds, pred=None, pval=None, j_sig=None,
                 conf:CCGConfig=None):
        self.key=key
        self.ids=ids
        self.inds=inds
        self.ccg=ccg
        self.pred=pred # 'baseline', or jittered, chance level CCG
        self.pval=pval
        self.j_sig=j_sig
        self._conf=conf
        self.conn_strength
 
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
        return self.inds[:,0]
    
    @property
    def target_inds(self):
        return self.inds[:,1]
    
    @property
    def conf(self):
        return self._conf

    @property
    def total(self):
        return self.ccg.shape[0]

    def _set_cs_eranconv(self, norm_factor:np.ndarray=None):
        """
        Connection strength:
        
            Area under CCG curve minus baseline, within temporal ROI
            The ROI is by default the same as the interval tested for peak/trough signficance

        Can be negative
        """
        auc = self.ccg-self.pred # area under curve
        cs = np.sum(auc[self.conf.min_lag_bin:self.conf.max_lag_bin]) # inds,nbins
        if norm_factor: cs /= norm_factor # e.g. presynaptic element firing rate
        self.conn_strength = cs

    def _set_cs_tail(self, acgs:ACG, nspks:list, norm_factor:np.ndarray=False):
        """
        Connection strength:

                Area under CCG curve minus a 'tailed' baseline after deconvolving autocorrelograms
        
        Can be negative
        """
        self.deconv_autocorr(acgs, nspks, target=True, ref=True)
        self._set_baseline_by_tail()
        auc = self.ccg-self.pred # area under curve
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
        self.pred = np.ones_like(self.ccg)*baseline

    def plotdir(self, root):
        if self.key.conn_type is None:
            return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.chunk}/{self.key.excitability}_any"
        return f"{root}/{self.key.session}/{self.key.session}-{self.key.epoch}{self.key.chunk}/{self.key.excitability}_{self.key.conn_type[0]}-{self.key.conn_type[1]}"    

    def sample_plot(self, inds):
        assert inds in self.inds
        pval = self.pval[inds] if self.pval else None
        pred = self.pred[inds] if self.pred else None
        j_sig = self.j_sig[inds] if self.j_sig else None
        plot_ccg_only(
            ccg=self.ccg[inds], 
            ids=self.ids[inds], 
            inds=inds, 
            window_size=self.conf.duration, 
            bin_size=self.conf.bin_size, 
            pval=pval, pred=pred, j_sig=j_sig,
        )

    def save_plots(self, neuron_types, waveforms, firing_rate, frates_all, root):
        assert self.ccg is not None
        plotdir = self.plotdir(root)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir,exist_ok=True)

        s=np.argsort(self.inds[:,1]) #[np.random.random_integers(0,inds.shape[0]-1,5)]
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
                            pred=self.pred[s][i] if self.pred is not None else None,
                            j_sig=self.j_sig[s][i] if self.j_sig else None,
                            show=False,save=True)

    def deconv_autocorr(self, acgs, nspks, target=True, ref=True):
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


class CCGDataset(AnalysisDataset):
    """
    Data and operations on CCGs from an experiment
    Requires a NeuronsDataset to be processed first, and a configuration object (see CCGConfig)
    """
    def __init__(self, nd:NeuronsDataset, conf:CCGConfig=None):
        self.nd = nd # neurons
        self.data={} # CCGs
        self._conf=conf # config
        self.spurious={} # rest of pairwise CCG that failed the significance checks
        self.auto={} # autocorrelograms 

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
        self.auto={}
    
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
            ccgs, spurs, autos, printstr = eranconv_group(neurons=neurons,conf=self.conf,key=key)
            self.append_ccg(key, ccgs)
            self.append_spurious(key, spurs)
            self.append_auto(key, autos)
            print(_s(key.session,neurons)+printstr)

    def time_rescale(self,bin_size,duration=None,jscale=None,include_spurious=False,run_conv=False):
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
        if jscale: self._conf.jscale = jscale
        print(f"recalculated CCG from binsize={old_bin_size} to binsize={bin_size}")

        for key, ccg in self.data.items():
            if ccg is not None:
                neurons = self.nd.data[key.parent()]
                self.data[key] = rescale_ccg(neurons=neurons,conf=self.conf,inds=ccg.inds,ccg_key=key,run_conv=run_conv)
        print("rescale completed")

        if not include_spurious: return

        for key, spur in self.spurious.items():
            if spur is not None:
                neurons = self.nd.data[key.parent()]
                self.spur[key] = rescale_ccg(neurons=neurons,conf=self.conf,inds=spur.inds,ccg_key=key,run_conv=run_conv)
        print("rescale of spurious CCG completed")
        
    def save_plots(self, jd = 'JitterDataset', root="~/Documents/NeuroPy/images/ccg_plots",
                   frates_all:dict=None,
                   conn_types:list=None):
        assert os.path.isdir(root)
        if isinstance(jd,str): jd == None # TODO ugly. to avoid circular imports
        keys = self.keys_matching(conn_type=conn_types) if conn_types else self.data.keys()
        print(keys)
        print(f"Saving plots under {root}")
        for key in keys:
            ccg = self.data[key]
            neurons = self.nd.data[key.parent()]
            frates = frates_all[key.parent()] if frates_all else None
            print(f"ccg {key.session} {key.conn_type}")
            try:
                if jd is not None:
                    ccg.j_sig = jd.data[key].significance
                ccg.save_plots(
                    neuron_types=neurons.neuron_type[ccg.inds],
                    waveforms=None if neurons.waveforms is None else neurons.waveforms[ccg.inds],
                    firing_rate=None if neurons.firing_rate is None else neurons.firing_rate[ccg.inds],
                    frates_all=None if frates is None else frates[ccg.inds],
                    root=root
                )
            except Exception as e:
                print(f"No {key.conn_type} connections: {e}")
                continue
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
            d.normalize(self.nd.data[d.key.parent].n_spikes[d.inds[:,0]])
        for d in self.spurious:
            d.normalize(self.nd.data[d.key.parent].n_spikes[d.inds[:,0]])

    def set_connection_strengths(self,method="eranconv"):
        #TODO untested
        for key, ccg in self.data.items():
            if self.conf.normalize==NormalizeBy.REF_FRATE:
                norm_factors = self.nd[key.parent].firing_rate[ccg.ref_inds][...,np.newaxis]
            elif self.conf.normalize==NormalizeBy.REF_SPKS:
                norm_factors = self.nd[key.parent].n_spikes[ccg.ref_inds][...,np.newaxis]
            else:
                norm_factors = None

            if method=="eranconv":
                ccg._set_cs_eranconv(norm_factors)
            elif method=="tail":
                spikecount = self.nd[key.parent].n_spikes
                acg = self.auto[key.parent]
                ccg._set_cs_tail(acg,spikecount,norm_factors=norm_factors)
                return NotImplementedError("Unknown connection strength method")

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
                jnp.round(
                    (
                        jnp.array(target_spiketrain)
                        + 2 * self.conf.jscale *jr.uniform(RAND_KEY(),(self.conf.njitter,target_nspikes))
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
        if self.conf.use_acceleration: jnp.get_default_memory_pool().free_all_blocks()
    
    def add_interval_jitter(self):        
        sampling_rate = self.neurons.sampling_rate
        b = self.j_ind
        target_nspikes = self.neurons.n_spikes[b]
        jscale_samples = int(self.conf.jscale * sampling_rate)
        # example: jscale_ms = 5ms, sampling rate = 30KHz, jscale in samples = 150
        
        # from https://github.com/aamarasingham/bjitter/blob/master/Figure2.m
        if self.conf.use_acceleration:
            jittertrains = (
                jnp.sort(jnp.floor(
                    (jnp.floor(
                        jnp.round(jnp.array(self.neurons.spiketrains[b]) * sampling_rate) 
                        / jscale_samples
                    ) + jr.uniform(RAND_KEY(),(self.conf.njitter,target_nspikes))) * jscale_samples 
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
        if self.conf.use_acceleration: jnp.get_default_memory_pool().free_all_blocks()

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
            pval = np.argsort(np.argsort(-self.j_ccg,axis=1,kind="stable"),axis=1)[:,0]/self.conf.njitter
            thresholds = np.percentile(self.j_ccg[:,1:], 100*(1-self.conf.alpha), axis=1)
        else:
            pval = np.argsort(np.argsort(self.ccg,axis=1,kind="stable"),axis=1)[:,0]/self.njitter
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
        self.significance = []
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
        keys, inv = np.unique(self.ccg.inds[:,1], return_inverse=True)
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
        ans = input("Changing configuration will remove existing jitters. Proceed? [y/n]").lower()
        if ans=='n' or ans=='no':
            print("Aborted")
            return
        self._conf = conf
        self.data = []

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


def rescale_ccg(ccg_key:Key, neurons:Neurons, conf:CCGConfig, inds:list[int], run_conv=False):
    ccg = correlations.spike_correlations(
                neurons=neurons,
                ref_neuron_inds=inds[:,0],
                neuron_inds=inds[:,1],
                bin_size=conf.bin_size,
                window_size=conf.duration,
                use_acceleration=conf.use_acceleration,
                symmetrize=conf.symmetrize_ccg,
                paired=True
            )
    print("completed rerun")
    if run_conv: 
        # TODO i think W should be number of bins. not actual jitter scale
        pvals, pred, qvals = eranconv(ccg,W=conf.jscale_bins,wintype="gauss",hollow_frac=None)
        # significance = pvals if ccg_key.excitability=='E' else qvals
        print("completed conv")
        return CCG(inds=inds,
                ids=neurons.ind2id(inds), 
                ccg=ccg, 
                pred=pred, 
                pval=None, 
                conf=conf,
                key=ccg_key)
    else:
        return CCG(inds=inds,
                ids=neurons.ind2id(inds), 
                ccg=ccg, 
                pred=None, 
                pval=None, 
                conf=conf,
                key=ccg_key)


def _multiple_correction(pvals,alpha,method='bonferroni'):
    # NOTE should bump this to utils or something
    # methods: 'fdr_bh', 'bonferroni'
    p_flat = pvals.ravel()
    sig, p_corr, _, _ = multipletests(p_flat, alpha=alpha, method=method)
    sig = sig.reshape(pvals.shape)
    p_corr = p_corr.reshape(pvals.shape)   
    # sig = (pvals[...,C+start:C+end+1] < corrected_alpha)
    return sig,p_corr


def eranconv_group(key:Key, neurons:Neurons, conf:CCGConfig):
    """
    Helper of eranconv

    chunked_neurons: list[Neurons] or Neurons, sliced from the same session.
    """
    overview_str = ""
    ccgs_by_type,spurs_by_type,autos = {},{},{} # 1 neuron group -> many connection types
    n=neurons.n_neurons

    ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=np.arange(n), # all
            bin_size=conf.bin_size,
            window_size=conf.duration,
            use_acceleration=conf.use_acceleration,
            symmetrize=conf.symmetrize_ccg,
        )
    
    pvals, pred, qvals = eranconv(ccg,W=conf.conv_window_bins,wintype="gauss",hollow_frac=None)

    # Universal, basic criteria for significance
    # These indices pairs are kept as a baseline; anything not fitting downstream criteria is put under 'spurious'
    autocorr_locations = np.eye(pvals.shape[0], dtype=bool)
    autos[key] = ccg[autocorr_locations]
    
    sig, p_correct = _multiple_correction(pvals, conf.alpha)
    pairs_E = np.argwhere((sig[...,conf.min_lag_bin:conf.max_lag_bin]).any(axis=-1) & ~autocorr_locations)

    sig1, q_correct1 = _multiple_correction(qvals, conf.alpha)
    sig2, q_correct2 = _multiple_correction(qvals, conf.alpha2)
    neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1))  # significant bins must have a significant-ish neighbor
    pairs_I = np.argwhere(neighbor.any(-1) & ~autocorr_locations)
    
    def _count_significant_pairs(pairs,neurons,conn_types,ignore:IgnoreLevel=IgnoreLevel.SAME_CHANNEL):
        """
        Create a tally of significant neuronal connectoins by type
        Currently, the type is defined as 
            reference-target/[E,I]
        where reference is presynaptic, and target is postsynaptic neuronal type, 
        and E/I indicates the connection being excitatory or inhibitory

        SL: If this helper function seems messy it's probably because 
        it pertains to our specific definition of significant pairs (see Diba 2014, Pairwise connections.)
        """
        sig_pairs = {}
        # Conn types with no pairs are marked with None
        if pairs.shape[0]==0:
            for ct in conn_types: sig_pairs[ct]=None
            return sig_pairs

        # Condition 2: Ref/Target are not too close by
        x,y = pairs[:,0],pairs[:,1]
        if ignore==IgnoreLevel.SAME_CHANNEL:
            assert neurons.peak_channels is not None
            inds=np.where(neurons.peak_channels[x]!=neurons.peak_channels[y])[0]
            pairs = pairs[inds]
        elif ignore==IgnoreLevel.SAME_SHANK:
            assert neurons.shank_ids is not None
            inds=np.where(neurons.shank_ids[x]!=neurons.shank_ids[y])[0]
            pairs = pairs[inds]
            
        # Condition 3: Ref/Target are specific cell types
        for ct in conn_types:
            inds=np.where(np.isin(pairs[:,0],np.where(neurons.neuron_type==ct[0])) & 
                                np.isin(pairs[:,1],np.where(neurons.neuron_type==ct[1])))[0]
            sig_pairs[ct]=pairs[inds] if inds.shape[0] else None
        return sig_pairs

    # Celltype specific crieria for significance
    # Condition 1: Spikecounts of bins within a certain time range are pass threshold
    pairs_spkcount = np.argwhere((ccg[...,conf.min_spkcnt_bin:conf.max_spkcnt_bin]>=conf.min_spkcount).all(axis=-1)) # NOTE right now it's the same criteria for E/I
    pairs_E = _intersect2d(pairs_E, pairs_spkcount, n)
    pairs_I = _intersect2d(pairs_I, pairs_spkcount, n)

    pairs_E_filtered = _count_significant_pairs(pairs_E,neurons,conf.conn_types_E,ignore=conf.ignore)
    pairs_I_filtered = _count_significant_pairs(pairs_I,neurons,conf.conn_types_I,ignore=conf.ignore)

    # Update print string 
    def _printstr_sig(pairs_dict, EI, s=""):
        # if any type of connection under consideration has a non-zero count, print a summary
        nonempty = any(v is not None for v in pairs_dict.values())
        if nonempty:
            for (ref,target), pairs in pairs_dict.items():
                s+=f"{ref}-{target}/{EI} {f'{pairs.shape[0]:02d}' if pairs is not None else '-'} | "
        else:
            s=f"no {'excitatory' if EI=='E' else 'inhbitory'} connections  "
        return s, nonempty
    
    overview_str += f"SLEEP{key.chunk}: E/I pairs {pairs_E.shape[0]:03d} / {pairs_I.shape[0]:03d} | "
    sE, hasE = _printstr_sig(pairs_E_filtered, 'E')
    sI, hasI = _printstr_sig(pairs_I_filtered, 'I')
    overview_str=overview_str+sE+sI+"\n" if (hasE or hasI) else overview_str+"no connections\n"
    
    # Update return values
    for EI, filtered_pairs, all_pairs in zip(['E','I'],
                                            [pairs_E_filtered,pairs_I_filtered],
                                            [pairs_E,pairs_I]):
        significance = pvals if EI=='E' else qvals # NOTE not storing corrected p-vals. This is just for display purposes

        spurious_pairs = all_pairs
        for conn_type, pairs in filtered_pairs.items():
            key = Key(session=key.session,
                    epoch=key.epoch,
                    chunk=key.chunk,
                    conn_type=conn_type,
                    excitability=EI)
            if pairs is None or len(pairs)==0:
                ccgs_by_type[key] = None
            else:
                spurious_pairs = _setdiff2d(spurious_pairs,pairs,n)
                x,y = pairs[:,0],pairs[:,1]
                ccgs_by_type[key] = CCG(inds=pairs, 
                                    ids=neurons.ind2id(pairs), 
                                    ccg=ccg[x,y], 
                                    pred=pred[x,y], 
                                    pval=significance[x,y], 
                                    conf=conf,
                                    key=key)

        key = Key(session=key.session,
                epoch=key.epoch,
                chunk=key.chunk,excitability=EI)
        pairs = np.asarray(spurious_pairs)
        if pairs is None or len(pairs)==0:
            spurs_by_type[key] = None
        else:
            x,y = pairs[:,0],pairs[:,1]
            spurs_by_type[key] = CCG(inds=pairs, 
                            ids=neurons.ind2id(pairs), 
                            ccg=ccg[x,y], 
                            pred=pred[x,y], 
                            pval=significance[x,y], 
                            conf=conf,
                            key=key)

    ### end of chunks loop ###
    return ccgs_by_type, spurs_by_type, autos, overview_str


# NOTE move to plotting in the future!
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, 
                   pval=None, pred=None, j_sig=None):
    """Single CCG plot into provided axis"""
    bins = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)

    ax.bar(bins, ccg, width=bin_size, alpha=0.5, label="ccg")
    if pred is not None:
        ax.bar(bins, pred, width=bin_size, alpha=0.5, label="ccg-smooth")
    if pval is not None:
        ax.plot(bins, pval * np.max(ccg), label='p')
    if j_sig is not None:
        ax.plot(bins, j_sig * np.max(ccg), label='j-significance')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    X, Y = ids; x, y = inds
    ax.set_title(f"CCG, neuron_ids=[{X},{Y}], indices=[{x},{y}]")
    ax.legend()
    sns.despine(ax=ax)
    return ax


def plot_waveform_panel(ax, waveform, neuron_type, neuron_id, 
                        frate_all=None, frate_cut=None, n_shanks=12, ch_per_shank=16, discard_channels=None):
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
    
    edges = (np.array(range(n_shanks))+1)*ch_per_shank+1
    if discard_channels is not None:
        shanks = discard_channels // ch_per_shank
        edges = edges - np.cumsum(np.histogram(shanks,np.arange(n_shanks))[0])
        
    for k in edges:
        ax.axhline(k * ch_per_shank, c='w', alpha=0.5, linestyle='dashed')
    return ax


def plot_ccg_figure(ccg, ids, inds, neuron_types, waveforms,
                    window_size, bin_size, pval=None, pred=None, j_sig=None, 
                    frates_all=None, frate_cut=None, n_shanks=None, ch_per_shank=None,
                    show=True, save=False, plotdir=None):
    """Full figure: CCG + 2 waveforms"""
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

    plot_ccg_panel(axs[0], ccg, ids, inds, window_size, bin_size, pval, pred, j_sig)
    labels = ['ref', 'target']
    for i in range(2):
        plot_waveform_panel(axs[1+i], waveforms[i], neuron_types[i], ids[i],
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


def plot_ccg_only(ccg, ids, inds, window_size, bin_size, pval=None, pred=None, j_sig=None, 
                  show=True, save=False, plotdir=None):
    """Save only the CCG plot without waveforms"""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, pval, pred, j_sig)
    
    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
    if show:
        plt.show()
    plt.close(fig)
    return fig


class WindowSweep:
    """
    Wrapper over a CCG, a neuron dataset, etc to split each item of the object into several windows of arbitrary step size
    and compare statistics on them. 

    Probably specific to the CCG project only, TBD how large the scope is
    """
    data = {}
    # sweep by window size, or number pre-syn spikes, 



# will fall under WindowSweep
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
 
routine_connection_strength
