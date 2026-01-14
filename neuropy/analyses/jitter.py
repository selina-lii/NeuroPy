
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
   

class JitterDataset(AnalysisDataset):
    def __init__(self, nd: Neurons, cd: CCGDataset, conf:JitterConfig):
        """Note that jitter dataset stores single session/epoch data because jitter computation is memory consuming"""
        self._conf = conf
        self.conf.ccg=cd.conf
        self.nd = nd
        self.cd = cd
        self.data = {} # data key is target index

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
                neurons = self.nd.data[key.nd()]
                self.data[key]=Jitter(key=key,
                                    neurons=neurons,
                                    conf=self.conf,
                                    ccg=ccg)
        for _,j in self.data.items():
            if j is not None: j.run(save_progress=save_progress)

