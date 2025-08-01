"""Calculate and test millisecond-scale connectivity between neurons a la Diba et al. (2014) and English/McKenzie
 et al. (2017)"""

import numpy as np
import neuropy.analyses.correlations as correlations
from neuropy.core.neurons import Neurons
import cupy as cp
# TODO ^
# TODO was this outdated?
try:
    import ccg_gpu as ccg
    import cupy as cp

    cuda = True
except ModuleNotFoundError:
    # import ccg

    cuda = False
cuda = False

import time

def eran_conv(ccg, window_width=5, wintype="gauss", hollow_frac=None):
    """Estimate chance-level correlations using convolution method from Stark and Abeles (2009, J. Neuro Methods).

    :param ccg:
    :param window_width:
    :param wintype:
    :param hollow_frac:
    :return: pvals:
             pred:
             qvals:
    """
    pvals, pred, qvals = None
    assert wintype in ["gauss", "rect", "triang"]

    # Auto-assign appropriate hollow fraction if not specified
    if hollow_frac is None:
        if wintype == "gauss":
            hollow_frac = 0.6
        elif wintype == "rect":
            hollow_frac = 0.42
        elif wintype == "triang":
            hollow_frac = 0.63

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

    ref_nspikes = neurons.n_spikes[0]
    ref_type = neurons.neuron_type[0]
    ref_spiketrain = neurons.spiketrains[0]

    if use_cupy:
        jittertrains = (
            cp.round(
                (
                    cp.array(ref_spiketrain)
                    + 2 * jscale * cp.random.rand(njitter,ref_nspikes)
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
                    ref_spiketrain
                    + 2 * jscale * np.random.rand(njitter,ref_nspikes)
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
        neuron_type=[ref_type]*njitter
        ) # TODO not copying over other fields
    neurons.merge(jittered)
    return neurons
    

def ccg_jitter(
    neurons: Neurons,
    neuron_inds,
    SampleRate=30000,
    binsize=0.0005,
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
    #     cp.round(duration / binsize / 2) if cuda else np.round(duration / binsize / 2)
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
            sample_rate=SampleRate,
            bin_size=binsize,
            window_size=duration,
            use_cupy=use_cupy,
            symmetrize=False,
        )[0] # get row1 since there's only one reference neuron

    # # Debugging
    # debug = correlations.spike_correlations(
    #         neurons=neuronsj,
    #         neuron_inds=np.arange(njitter+2),
    #         sample_rate=SampleRate,
    #         bin_size=binsize,
    #         window_size=duration,
    #         use_cupy=use_cupy,
    #         symmetrize=False,
    #     )[0,1:]-ccg_all
   
    # import copy
    
    # ccg_all = copy.deepcopy(ccgj)
    # ccg_all.append(correlograms)
    # ccg_all=np.array(ccg_all)
    # P value is where the real data is ranked among fake data; conservative when there are ties
    pval = np.argsort(np.argsort(-ccg_all,axis=0,kind="stable"),axis=0)[-1]/njitter
    thresholds = np.percentile(ccg_all[:-1],100*(1-alpha))
    significances = ccg_all[-1] > thresholds

    # significances = correlograms > thresholds

    return ccg_all, pval, significances


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
