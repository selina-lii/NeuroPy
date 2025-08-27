import neuropy.analyses.ms_connectivity as msconn
import numpy as np
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
            symmetrize_mode='even',
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
            symmetrize_mode='odd',
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
        orubt(subjects.nsd.allsess)