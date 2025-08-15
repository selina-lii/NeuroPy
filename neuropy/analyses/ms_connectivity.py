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


def eran_conv(ccg, W=5, wintype="gauss", hollow_frac=None):
    """
    Estimate chance-level correlations using convolution method from Stark and Abeles (2009, J. Neuro Methods).
    Referencing MATLAB script EranConv.m written by the authors

    Parameters
    ----------
    ccg: np.array. 
        1D or 2D. (CCGs in columns)
        If 2D, elements in the first dimension are individual ccgs and second dimension are bins.
    W: 
        defines the width of the convolution window, should be equivalent to size of jitter window (in milliseconds)
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

    # two-tailed one-sample test of whether the two values are significantly different in Poisson distribution
    pvals = 1 - poisson.cdf(ccg-1, pred) - poisson.pmf(ccg, pred)*0.5
    qvals = 1 - pvals
    return pvals, pred, qvals

# Analyes
def add_jitter(neurons: Neurons, njitter, neuron_inds, jscale, use_cupy=False):
    """
    Spike timing jitter.
    Randomly shift each spike in non-reference spike train
    Maximum shift is +/-(jscale) milliseconds

    Parameters
    ----------
    njitter : int
        number of jitters
    neuron_inds : list
        [a,b]
        a: index of reference neuron
        b: index of non-reference neuron
    jscale: int
        defines maximum time shift of a spike in seconds
    use_cupy: bool, optional
        whether or not to use gpu acceleration

    Returns
    -------
    neurons: Neurons=
        a Neurons object containing (njitter+2) neurons, with indices 0...njitter.
        first neuron is the reference cell, index=0
        second neuron is the non-reference cell with index=1
        the next (njitter) neurons are jitters of the non-reference cell
    """
    neurons = neurons.get_by_id(neuron_inds)

    neurons.neuron_ids[0]=0 # ref
    neurons.neuron_ids[1]=1 # non-ref

    nonref_nspikes = neurons.n_spikes[1]
    nonref_type = neurons.neuron_type[1]
    nonref_spiketrain = neurons.spiketrains[1]

    if use_cupy:
        jittertrains = (
            cp.round(
                (
                    cp.array(nonref_spiketrain)
                    + 2 * jscale * cp.random.rand(njitter,nonref_nspikes)
                    - 1 * jscale
                )
                * neurons.sampling_rate
            )
            / neurons.sampling_rate
        ).get()
    else:
        jittertrains = (
            np.round(
                (
                    nonref_spiketrain
                    + 2 * jscale * np.random.rand(njitter,nonref_nspikes)
                    - 1 * jscale
                )
                * neurons.sampling_rate
            )
            / neurons.sampling_rate
        )

    # Asign indices sequentially
    jittertrains = list(jittertrains)
    jittered = Neurons(spiketrains=jittertrains,
        t_stop=neurons.t_stop,
        neuron_ids=np.arange(njitter)+len(neuron_inds),
        neuron_type=[nonref_type]*njitter
        ) # TODO not copying over other fields
    neurons.merge(jittered)
    return neurons
    
def add_jitter_ISI(neurons: Neurons, njitter, neuron_inds, jscale, use_cupy=False):
    """
    Inter-spike intervals jitter.
    Randomly shuffled the spike time intervals in non-reference spike train
    within local windows of +/-(jscale) intervals

    Parameters
    ----------
    njitter : int
        number of jitters
    neuron_inds : list
        [a,b]
        a: index of reference neuron
        b: index of non-reference neuron
    jscale: int
        defines window within which intervals are grouped and shuffled
    use_cupy: bool, optional
        whether or not to use gpu acceleration

    Returns
    -------
    neurons: Neurons
        a Neurons object containing (njitter+2) neurons, with indices 0...njitter.
        first neuron is the reference cell, index=0
        second neuron is the non-reference cell with index=1
        the next (njitter) neurons are jitters of the non-reference cell
    """
    neurons = neurons.get_by_id(neuron_inds)

    neurons.neuron_ids[0]=0 # ref
    neurons.neuron_ids[1]=1 # non-ref

    nonref_nspikes = neurons.n_spikes[1]
    nonref_type = neurons.neuron_type[1]
    nonref_spiketrain = neurons.spiketrains[1]
    intervals = cp.diff(nonref_spiketrain)


    if use_cupy:
        jittertrains = (
            cp.round(
                (
                    cp.array(nonref_spiketrain)
                    + 2 * jscale * cp.random.rand(njitter,nonref_nspikes)
                    - 1 * jscale
                )
                * neurons.sampling_rate
            )
            / neurons.sampling_rate
        ).get()
    else:
        jittertrains = (
            np.round(
                (
                    nonref_spiketrain
                    + 2 * jscale * np.random.rand(njitter,nonref_nspikes)
                    - 1 * jscale
                )
                * neurons.sampling_rate
            )
            / neurons.sampling_rate
        )

    # Asign indices sequentially
    jittertrains = list(jittertrains)
    jittered = Neurons(spiketrains=jittertrains,
        t_stop=neurons.t_stop,
        neuron_ids=np.arange(njitter)+len(neuron_inds),
        neuron_type=[nonref_type]*njitter
        ) # TODO not copying over other fields
    neurons.merge(jittered)
    return neurons

def ccg_jitter(neurons: Neurons,
    neuron_inds,
    sample_rate=30000,
    bin_size=0.001,
    duration=0.02,
    jscale=5,
    njitter=100,
    alpha=0.05,
    use_cupy=False,
):

    # TODO: make this take the same inputs as correlograms? e.g. spikes from both clusters sorted by time with corresponding cluster ids?
    # # Make spike trains into 1d numpy array
    # spikes0 = spike_trains[0]
    # spikes1 = spike_trains[1]

    # set up variables
    # halfbins = ( TODO what are halfbins for...
    #     cp.round(duration / bin_size / 2) if cuda else np.round(duration / bin_size / 2)
    # )
    # spikes_sorted, clus_sorted = ccg_spike_assemble(spike_trains)
    # spikes1 = spikes_sorted[
    #     clus_sorted == 1
    # ]  # keep all spike times from cluster 1 for easy manipulation during jitter step


    # Now run on jittered spike-trains!
    # TODO: implement this in ALL cupy and compare times...does it matter if the spike jitter code is in numpy? Answer: it does 16ms with numpy vs 1 with cupy.
    # nspikes1 = len(spikes1)
    neuronsj = add_jitter(neurons=neurons,
            njitter=njitter,
            neuron_inds=neuron_inds,
            jscale=jscale,
            use_cupy=use_cupy,
        )
        # Jitter spikes in second cluster
    # NRK TODO: start debugging here!
    # spikes_sorted, clus_sorted = ccg_spike_assemble(spike_trains)

    # re-run ccg
    ccg_all=correlations.spike_correlations_2group(
            neurons=neuronsj,
            ref_neuron_inds=0,
            neuron_inds=1+np.arange(njitter+1),
            sample_rate=sample_rate,
            bin_size=bin_size,
            window_size=duration,
            use_cupy=use_cupy,
            symmetrize=True,
        )[0] # get row1 since there's only one reference neuron

    # Debugging - 'debug' should be all zeros (two methods are identical)
    orig = correlations.spike_correlations(
            neurons=neuronsj,
            neuron_inds=np.arange(njitter+2),
            sample_rate=sample_rate,
            bin_size=bin_size,
            window_size=duration,
            use_cupy=use_cupy,
            symmetrize=True,
        )[0,1:]
    debug = orig-ccg_all
   
    # import copy
    
    # ccg_all = copy.deepcopy(ccgj)
    # ccg_all.append(correlograms)
    # ccg_all=np.array(ccg_all)
    # P value is where the real data is ranked among fake data; conservative when there are ties
    pval = np.argsort(np.argsort(-ccg_all,axis=0,kind="stable"),axis=0)[0]/njitter
    thresholds = np.percentile(ccg_all[1:],100*(1-alpha))
    significances = ccg_all[0] > thresholds

    # significances = correlograms > thresholds

    return ccg_all, pval, significances, orig


def pairwise_conn_fast(neurons: Neurons,
    neuron_inds=None,
    sample_rate=30000,
    bin_size=0.001,
    duration=0.02,
    window_width=5,
    wintype="gauss", 
    hollow_frac=None,
    alpha=0.05,
    use_multi_correction=False,
    use_cupy=False,
    symmetrize_mode='even'):

    """
    A quick, rough screening for neuronal pairs with significant CCG peaks
    Uses eran_conv

    window_width:
        window witdth of the convolution kernel
        unit is milliseconds
        should be the same as `jscale` that you'd use later for jittering
    """
    neuron_inds = neuron_inds or np.arange(neurons.n_neurons)
    ccg = correlations.spike_correlations(
            neurons=neurons,
            neuron_inds=neuron_inds,
            sample_rate=sample_rate,
            bin_size=bin_size,
            window_size=duration,
            use_cupy=use_cupy,
            symmetrize=True,
            symmetrize_mode=symmetrize_mode,
        )
    W = window_width*1e-3/bin_size # align conv kernel size to jitter timescale
    pvals, pred, qvals = eran_conv(ccg,
                                   W=W,
                                   wintype=wintype,
                                   hollow_frac=hollow_frac)
    if use_multi_correction: alpha=alpha/(len(neuron_inds)**2)
    
    # pvals=pvals.flatten()
    # pred=pred.flatten()
    # qvals=qvals.flatten()

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
        'target_type':['pyr','int'],
        'p':0.05,
    }
    inhib_args = {
        'min_lag':1,
        'max_lag':3,
        'min_span':2,
        'min_spkcount':2.5,
        'spkcount_scope':12,
        'ignore_same_electrodes':False,
        'ref_type':'int',
        'target_type':'pyr',
        'p':0.05,
        'p2':0.1
    }

    # C=int(duration/bin_size//2) # center bin
    # start=int(args['min_lag']/bin_size)
    # end=int(args['max_lag']/bin_size)
    # coords_sig = np.argwhere((pvals<alpha).any(axis=-1))

    return pvals, ccg, pred, qvals #, p_sig, q_sig




# def ccg_spike_assemble(spike_trains):
#     """Assemble an array of sorted spike times and cluIDs for the input cluster ids the list clus_use """
#     spikes_all, clus_all = [], []
#     for ids, spike_train in enumerate(spike_trains):
#         spikes_all.append(spike_train),
#         clus_all.append(np.ones_like(spike_train) * ids)
#     if cuda:
#         spikes_all, clus_all = cp.concatenate(spikes_all), cp.concatenate(clus_all)
#     else:
#         spikes_all, clus_all = np.concatenate(spikes_all), np.concatenate(clus_all)
#     spikes_sorted = spikes_all[spikes_all.argsort()]
#     clus_sorted = clus_all[spikes_all.argsort()]

#     return spikes_sorted, clus_sorted.astype("int")
