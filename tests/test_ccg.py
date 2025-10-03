import neuropy.analyses.ms_connectivity as msconn
import numpy as np
from neuropy.analyses import correlations 

class Test:
    def __init__(self):
        
        # Universal example neurons
        self.neurons=subjects.nsd.allsess[5].neurons

        # Indices sorted by length of spiketrain
        self.indices = np.argsort([self.neurons.spiketrains[_].shape[0] for _ in range(self.neurons.n_neurons)])
        
        # Short testing neurons
        self.short_neurons = self.neurons.time_slice(t_start=10000,t_stop=10550)

    def test_ccg(self):
        # test equivalence of ccgs. 
        # Use a very small bin size. 
        # when bin size approaches the finest possible scale, ccgsE===ccgsO
        # when bin size is large, ccgsE can be slightly larger due to extra time from the two edge bins
        bin_size=1
        window_size=10
        window_width=1 # long window length doesn't work? pred and ccg shape incompatible
        ind=(0,1)
        pvalsE, ccgsE, predE, qvalsE=msconn.pairwise_conn_fast(self.neurons,
            neuron_inds=self.indices[100,14],
            bin_size=bin_size,
            duration=window_size,
            window_width=window_width,
            wintype="gauss", 
            hollow_frac=None,
            alpha=0.05,
            use_multi_correction=True,
            use_cupy=True,
            bin_mode='even',
        )
        print(ccgsE[ind].sum(),ccgsE[ind])

        pvalsO, ccgsO, predO, qvalsO=msconn.pairwise_conn_fast(self.neurons,
            neuron_inds=self.indices[100,14],
            bin_size=bin_size,
            duration=window_size,
            window_width=window_width,
            wintype="gauss", 
            hollow_frac=None,
            alpha=0.05,
            use_multi_correction=True,
            use_cupy=True,
            bin_mode='odd',
        )
        print(ccgsO[ind].sum(),ccgsO[ind])

    def sess_info():
        sess = subjects.nsd.allsess[5]
        print("data sampling rate\t", sess.recinfo.dat_sampling_rate)
        print("eeg sampling rate\t", sess.recinfo.eeg_sampling_rate)
        print("n_channels\t\t", sess.recinfo.n_channels)
        print("signal dtype\t\t", sess.recinfo.sig_dtype)
        print("skipped channels\t", sess.recinfo.skipped_channels)
        print("discarded channels\t", sess.recinfo.discarded_channels)
        print("source file\t\t", sess.recinfo.source_file)
        # print(sess.recinfo.to_dict())
        print("basepath\t\t", sess.basepath)
        print("file prefix\t\t", sess.filePrefix)
        print("probegroup\t\t", sess.probegroup)
        print("channel groups\t\t",sess.recinfo.channel_groups)
    
    def list_sess():
        print(subjects.nsd.allsess)

    def test_ccg_plot(tmpdir):
        neurons=subjects.nsd.allsess[5].neurons.get_neuron_type(['pyr','inter'])[[182,10]]
        ccg=correlations.spike_correlations(neurons,[0,1],bin_size=0.001,window_size=0.02,use_cupy=True)[0,1]
        pval=np.random.random(ccg.shape[0])
        pred=np.clip(ccg-max(ccg)//10,min(ccg),np.inf)
        frates_all=[1,2]
        msconn.plot_ccg_eranconv(neurons, ccg, inds=[0,1], plotdir=tmpdir, window_size=0.02, bin_size=0.001, 
                            pval=pval, pred=pred, frates_all=frates_all, jsig=None, mode='even')
        
    def test_ccg_2group():
        neurons=subjects.nsd.allsess[5].neurons.get_neuron_type(['pyr','inter'])[[182,10]]
        # for use_cupy in ['True','False']        
        for symmetrize in ['True','False']:
            for bin_mode in ['even', 'odd']:
                ccg=correlations.spike_correlations(neurons,[0,1],bin_size=0.001,window_size=0.02,use_cupy=True,symmetrize=symmetrize,bin_mode='even')[0,1]
                pred=correlations.spike_correlations(neurons,ref_neuron_inds=[0],neuron_inds=[1],bin_size=0.001,window_size=0.02,use_cupy=True,bin_mode='even')[0,0]
                pval=np.random.random(ccg.shape[0])
                frates_all=[1,2]
                msconn.plot_ccg_eranconv(neurons, ccg, inds=[0,1], plotdir="./tests", window_size=0.02, bin_size=0.001, 
                                    pval=pval, pred=pred, frates_all=frates_all, jsig=None, mode='even')
                
    def test_example_inter_spike_interval_jitter():
        # i came up w this method (histogram) but Han's code was more efficient
        # do we allow spikes with identical times?
        start = time.time()
        njitters = 1
        nbins = 10
        x = 5
        size = np.random.randint(nbins*x//3,2*nbins*x//3)
        # assume no duplicate spikes
        simulated_spikes = cp.random.choice(nbins*x,size=size,replace=False)
        hist,_ = cp.histogram(simulated_spikes, bins=edges)
        print(hist)
        # hist = cp.random.randint(0, x, size=nbins, dtype=int)
        edges = cp.arange(0, (nbins+1)*x, x)
        jittertrains = cp.zeros((njitters,size))
        for i in range(njitters):
            perms = cp.array([cp.random.permutation(x) for _ in range(nbins)])
            print([perms[j,:int(hist[j])]+j*x for j in range(nbins)])
            tmp = cp.sort(cp.concatenate([perms[j,:int(hist[j])]+j*x for j in range(nbins)]))
            print(cp.histogram(tmp, bins=edges)[0]-hist)
            jittertrains[i] = tmp
        end = time.time()
        print("Execution time:", end - start, "seconds")
        """
                    max_edge = x*((target_spiketrain.max()+x-1)//x)
                    edges = cp.arange(0, max_edge+x, x)
                    n_bins = edges.shape[0]-1
                    hist,_ = cp.histogram(target_spiketrain, bins=edges)
                    indices = [cp.random.permutation(x) for _ in range(n_bins*self.njitter)]
                    jittertrains = indices
                    bin_indices = np.random.choice(len(hist),size=target_nspikes,p=hist/target_nspikes)
        """