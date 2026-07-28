import neuropy.analyses.ms_connectivity as msconn
import numpy as np
from neuropy.analyses import correlations 

class Test:
    def __init__(self,neurons):
        
        # Universal example neurons
        self.neurons=neurons#subjects.nsd.allsess[5].neurons

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
            use_acceleration=True,
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
            use_acceleration=True,
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
        ccg=correlations.spike_correlations(neurons,[0,1],bin_size=0.001,window_size=0.02,use_acceleration=True)[0,1]
        pval=np.random.random(ccg.shape[0])
        pred=np.clip(ccg-max(ccg)//10,min(ccg),np.inf)
        frates_all=[1,2]
        msconn.plot_ccg_eranconv(neurons, ccg, inds=[0,1], plotdir=tmpdir, window_size=0.02, bin_size=0.001, 
                            pval=pval, pred=pred, frates_all=frates_all, jsig=None, mode='even')
        
    def test_ccg_2group():
        neurons=subjects.nsd.allsess[5].neurons.get_neuron_type(['pyr','inter'])[[182,10]]
        # for use_acceleration in ['True','False']        
        for symmetrize in ['True','False']:
            for bin_mode in ['even', 'odd']:
                ccg=correlations.spike_correlations(neurons,[0,1],bin_size=0.001,window_size=0.02,use_acceleration=True,symmetrize=symmetrize,bin_mode='even')[0,1]
                pred=correlations.spike_correlations(neurons,ref_neuron_inds=[0],neuron_inds=[1],bin_size=0.001,window_size=0.02,use_acceleration=True,bin_mode='even')[0,0]
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

    def test_frate_stats(nd:NeuronsDataset):
        edge_times = nd.edge_times[key]
        frates = nd.segment_firing_rates[key]
        neuron_types = nd.neurons_for(key).neuron_type
        labels = edge_times['label'].values
        stats_name = "firing rate"
        from scipy.stats import ttest_ind

        # test code here

        """
        Example output

        firing rate stats pyr
        segment | num | mean | iqr | min | max | variance | skew | kurt
        0:pre0  | 174 | 1.05 | 1.21 | 0.00 | 5.76 | 1.25 | 1.76 | 3.37
        1:maze0 | 174 | 1.28 | 1.68 | 0.00 | 11.45 | 2.65 | 2.77 | 10.84
        2:post0 | 174 | 0.69 | 0.64 | 0.00 | 4.36 | 0.46 | 2.23 | 6.71
        3:post1 | 174 | 0.65 | 0.63 | 0.00 | 3.78 | 0.38 | 2.10 | 5.89
        4:post2 | 174 | 0.64 | 0.61 | 0.00 | 3.43 | 0.34 | 1.85 | 4.20
        5:post3 | 174 | 0.65 | 0.63 | 0.00 | 3.39 | 0.34 | 1.70 | 3.34
        6:post4 | 174 | 0.65 | 0.65 | 0.00 | 3.37 | 0.34 | 1.69 | 3.25
        7:post5 | 174 | 0.66 | 0.64 | 0.00 | 3.37 | 0.35 | 1.66 | 3.07
        8:post6 | 174 | 0.66 | 0.62 | 0.00 | 3.40 | 0.36 | 1.67 | 3.06
        9:post7 | 174 | 0.67 | 0.62 | 0.00 | 3.40 | 0.37 | 1.66 | 2.88
        10:post8 | 174 | 0.68 | 0.64 | 0.00 | 3.44 | 0.40 | 1.71 | 2.95
        11:re-maze0 | 174 | 0.94 | 1.25 | 0.00 | 4.10 | 0.91 | 1.21 | 0.74
        
        Mean firing rate P VALUES
        pyr  |0    |1    |2    |3    |4    |5    |6    |7    |8    |9    |10   |11   |
        0    |pre0
        1    |0.12 |maze0
        2    |0.00*|0.00*|post0
        3    |0.00*|0.00*|0.56 |post1
        4    |0.00*|0.00*|0.44 |0.85 |post2
        5    |0.00*|0.00*|0.54 |0.98 |0.86 |post3
        6    |0.00*|0.00*|0.53 |0.97 |0.87 |0.98 |post4
        7    |0.00*|0.00*|0.64 |0.90 |0.74 |0.88 |0.86 |post5
        8    |0.00*|0.00*|0.72 |0.81 |0.66 |0.79 |0.78 |0.91 |post6
        9    |0.00*|0.00*|0.79 |0.74 |0.59 |0.71 |0.70 |0.83 |0.92 |post7
        10   |0.00*|0.00*|0.95 |0.59 |0.46 |0.57 |0.56 |0.67 |0.76 |0.84 |post8
        11   |0.33 |0.02*|0.00*|0.00*|0.00*|0.00*|0.00*|0.00*|0.00*|0.00*|0.00*|re-maze0

        firing rate stats inter
        segment | num | mean | iqr | min | max | variance | skew | kurt
        0:pre0  | 14 | 26.30 | 25.74 | 0.10 | 58.00 | 328.18 | 0.16 | -1.15
        1:maze0 | 14 | 27.28 | 28.97 | 0.13 | 58.85 | 355.17 | 0.03 | -1.20
        2:post0 | 14 | 21.13 | 15.89 | 9.40 | 37.26 | 108.70 | 0.29 | -1.44
        3:post1 | 14 | 20.64 | 14.47 | 9.35 | 35.53 | 99.32 | 0.32 | -1.41
        4:post2 | 14 | 20.40 | 13.11 | 9.12 | 35.16 | 93.09 | 0.29 | -1.38
        5:post3 | 14 | 20.65 | 14.23 | 8.77 | 34.86 | 92.68 | 0.19 | -1.45
        6:post4 | 14 | 20.67 | 12.94 | 8.11 | 34.51 | 88.50 | 0.14 | -1.41
        7:post5 | 14 | 21.14 | 12.77 | 7.07 | 34.89 | 89.49 | 0.04 | -1.34
        8:post6 | 14 | 21.40 | 12.40 | 6.19 | 35.02 | 88.45 | -0.05 | -1.22
        9:post7 | 14 | 21.68 | 12.97 | 5.79 | 35.08 | 89.34 | -0.14 | -1.16
        10:post8 | 14 | 22.11 | 13.95 | 6.19 | 35.36 | 90.95 | -0.17 | -1.19
        11:re-maze0 | 14 | 27.59 | 23.51 | 3.44 | 48.70 | 200.52 | -0.28 | -1.21

        Mean firing rate P VALUES
        inter|0    |1    |2    |3    |4    |5    |6    |7    |8    |9    |10   |11   |
        0    |pre0
        1    |0.89 |maze0
        2    |0.36 |0.30 |post0
        3    |0.32 |0.25 |0.90 |post1
        4    |0.29 |0.23 |0.85 |0.95 |post2
        5    |0.31 |0.25 |0.90 |1.00 |0.95 |post3
        6    |0.31 |0.25 |0.90 |0.99 |0.94 |0.99 |post4
        7    |0.35 |0.29 |1.00 |0.89 |0.84 |0.89 |0.90 |post5
        8    |0.38 |0.31 |0.94 |0.84 |0.78 |0.84 |0.84 |0.94 |post6
        9    |0.41 |0.33 |0.88 |0.78 |0.73 |0.78 |0.78 |0.88 |0.94 |post7
        10   |0.45 |0.37 |0.80 |0.69 |0.64 |0.69 |0.69 |0.79 |0.84 |0.91 |post8
        11   |0.84 |0.96 |0.18 |0.15 |0.13 |0.14 |0.14 |0.17 |0.19 |0.21 |0.24 |re-maze0
        """