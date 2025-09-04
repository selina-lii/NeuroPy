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
from typing import Union
from datetime import datetime
import h5py
from statsmodels.stats.multitest import multipletests

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

def add_jitter(neurons: Neurons, njitter, neuron_inds, jscale=5, use_cupy=False):
    """
    Spike timing jitter.
    Randomly shift each spike in target spike train

    Parameters
    ----------
    njitter : int
        number of jitters
    neuron_inds : list
        [a1, ..., an, b]
        a1~an: indices of reference neurons (total of N0)
        b: index of target neuron (just one)
    jscale: int
        maximum spiking time shift in ms (default is +-5ms)
    use_cupy: bool, optional
        whether or not to use gpu acceleration

    Returns
    -------
    neurons: Neurons=
        a Neurons object containing (N0+1+njitter) neurons, with ids 0,1,2,....
        reference: 0, 1, ..., N0-1
        target: N0
        jitter: N0+1, ..., N0+njitter
    """
    b = neuron_inds[-1]
    id = neurons.neuron_ids[b]
    target_nspikes = neurons.n_spikes[b]
    target_type = neurons.neuron_type[b]
    target_spiketrain = neurons.spiketrains[b]

    if use_cupy:
        jittertrains = (
            cp.round(
                (
                    cp.array(target_spiketrain)
                    + 2 * jscale * 1e-3 * cp.random.rand(njitter,target_nspikes)
                    - 1 * jscale * 1e-3
                )
                * neurons.sampling_rate
            )
            / neurons.sampling_rate
        ).get()
    else:
        jittertrains = (
            np.round(
                (
                    target_spiketrain
                    + 2 * jscale * 1e-3 * np.random.rand(njitter,target_nspikes)
                    - 1 * jscale * 1e-3
                )
                * neurons.sampling_rate
            )
            / neurons.sampling_rate
        )
    jittertrains = list(jittertrains)
    jittered = Neurons(spiketrains=jittertrains,
        t_start=neurons.t_start,
        t_stop=neurons.t_stop,
        neuron_ids=[id]*njitter,
        neuron_type=[target_type]*njitter
        ) # TODO not copying over other fields
    neurons.neuron_slice(neuron_inds=neuron_inds) \
            .merge(jittered)
    cp.get_default_memory_pool().free_all_blocks()
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
                    + 2 * jscale * 1e-3 * cp.random.rand(njitter,nonref_nspikes)
                    - 1 * jscale * 1e-3
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
                    + 2 * jscale * 1e-3 * np.random.rand(njitter,nonref_nspikes)
                    - 1 * jscale * 1e-3
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
    cp.get_default_memory_pool().free_all_blocks()
    return neurons

def ccg_jitter(neurons: Neurons,
    neuron_inds,
    bin_size=0.001,
    duration=0.02,
    jscale=5,
    njitter=100,
    alpha=0.05,
    use_cupy=False,
    bin_mode='even',
    EI = 'E'
):
    """
    CCGs are shaped (N0,1,nbins)

    EI: if 'E', use p-vals for peaks, else use q-vals for troughs
    """

    # SL: These were comments from Nat I guess - 
    # most of these are naturally fixed as I update the function

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

    # Add jitter to the last neuron in the list, which is the target neuron
    neuronsj = add_jitter(neurons=neurons,
            njitter=njitter,
            neuron_inds=neuron_inds,
            jscale=jscale,
            use_cupy=use_cupy,
        )
        # Jitter spikes in second cluster

    # run ccg
    N0 = len(neuron_inds)-1
    ccg_all=correlations.spike_correlations(
            neurons=neuronsj,
            ref_neuron_inds=np.arange(N0),
            neuron_inds=N0+np.arange(njitter+1),
            bin_size=bin_size,
            window_size=duration,
            use_cupy=use_cupy,
            symmetrize=True,
            bin_mode=bin_mode,
        )
    # Debugging - 'debug' should be all zeros (two methods are identical)
    # orig = correlations.spike_correlations(
    #         neurons=neuronsj,
    #         neuron_inds=np.arange(neuronsj.n_neurons),
    #         bin_size=bin_size,
    #         window_size=duration,
    #         use_cupy=use_cupy,
    #         symmetrize=True,
    #         bin_mode=bin_mode,
    #     )
    # debug = orig[0,len(ref_inds):]-ccg_all[0]
    # print(debug)
    
    # TODO
    # ccg_all: (N0, njitter+1, Nbins)
    # pval = (N0, Nbins) where real data is ranked among fake data. conservative when there are ties
    # thresholds = (N0, Nbins)
    if EI=='E':
        pval = np.argsort(np.argsort(-ccg_all,axis=1,kind="stable"),axis=1)[:,0]/njitter
        thresholds = np.percentile(ccg_all[:,1:], 100*(1-alpha), axis=1)
    else:
        pval = np.argsort(np.argsort(ccg_all,axis=1,kind="stable"),axis=1)[:,0]/njitter
        thresholds = np.percentile(ccg_all[:,1:], 100*(alpha), axis=1)
    significances = pval<=alpha
    jbsi_vals = jbsi(ccg_all,
                     frates = neurons.firing_rate[neuron_inds],
                     bin_size=bin_size,
                     jscale=jscale)
    jitter_out = {'neuronsj':neuronsj,
                  'ccg_all':ccg_all,
                  'pval':pval,
                  'significances':significances,
                  'thresholds':thresholds,
                  'JBSI':jbsi_vals
                  }
    return jitter_out # orig    

def jbsi(ccgs,frates,bin_size,jscale,):
    """
    Can be batched, but only for one neuron vs group of neurons
    frates = [ref1, ref2, ..., refN0, target]
    ccgs = (N0, Njitters+1, Nbins) or (Njitters+1, Nbins)
    Real ccg should be at the beginning of the array, before jitters

    return shape = (N0, Nbins) or (Nbins,)
    """
    if len(ccgs.shape)==3:
        N0 = ccgs.shape[0] 
        Nreal =ccgs[:,0] # (N0, Nbins)
        Nj_avg = np.mean(ccgs[:,1:],axis=1) # (N0, Nbins) averaged over Njitter columns
    else:
        N0 = 1
        Nreal =ccgs[0] # (1, Nbins)
        Nj_avg = np.mean(ccgs[1:],axis=0) # (1, Nbins) averaged over Njitter rows
    assert len(frates)==N0+1
    comp_frates = np.zeros((2,N0))
    comp_frates[0]=frates[-1]
    comp_frates[1]=frates[:-1]
    n1 = np.min(comp_frates,axis=0)[...,np.newaxis] # (N0,1) or (1,1)

    ts = bin_size
    tj = jscale
    b = tj/(tj-ts) if tj/ts>2 else 2
    return b/n1*(Nreal - Nj_avg) # (N0, Nbins) or (1, Nbins)
    
def _short_session_name(session):
    """get short printable session name in the format of ANIMAL_DayX"""
    sess_name = session.filePrefix.parts[-1].split('_')[:2]
    sess_name='_'.join(sess_name)
    return sess_name

def routine_jitter(neurons,jitter_inputs):
    """
    Return neuronal pairs with valid CCG peaks (slow)
    Uses jittering

    """

    # TODO not a routine yet
    # only returns p values
    # group by target neuron, return list of inputs properly formatted for add_jitter()
    sigs=[]
    for ji in jitter_inputs:
        sigs.append(ccg_jitter(neurons,neuron_inds=ji,use_cupy=True,
                               njitter=100,alpha=1e-6,bin_mode='odd')[2])
    jitter_significances=np.array([xx for x in sigs for xx in x])#.astype(int)
    return jitter_significances

def prep_connectivity_dict(data_struct, session_names, epochs, conn_types):
    # |_ session name:
    #   |_ epoch:
    #       |_ excitability:
    #           |_ connection type:
    #               |_ [connectivity data 1, ...]
    # |_ metadata1
    # |_ metadata2
    # ...
    for sn in session_names:
        data_struct[sn]={}
        for e in epochs:
            data_struct[sn][e]={}
            for EI in ['E','I']:
                data_struct[sn][e][EI]={}
                for ct in conn_types[EI]: 
                    data_struct[sn][e][EI][ct]=[]
    return data_struct

def prep_connectivity_plot_folders(root, session_names, epochs, n_chunks, conn_types):
    # |_ session name:
    #   |_ epoch:
    #       |_ excitability:
    #           |_ connection type:
    #               |_ [connectivity data 1, ...]
    os.makedirs(root, exist_ok=True)
    for sn in session_names:
        sn_root = f"{root}/{sn}"
        os.makedirs(sn_root,exist_ok=True)
        for e,n_chunk in zip(epochs,n_chunks):
            for c in range(n_chunk):
                e_root = f"{root}/{sn}/{sn}-{e}{c}"
                for EI in ['E','I']:
                    for ct in conn_types[EI]: 
                        c_root = f"{root}/{sn}/{sn}-{e}{c}/{sn}-{e}{c}-{ct[0]}-{ct[1]}-{EI}"
                        os.makedirs(c_root,exist_ok=True)

def get_connectivity_plot_folder(root, session_name, epoch, chunk_id, conn_type, EI):
    return f"{root}/{session_name}/{session_name}-{epoch}{chunk_id}/{session_name}-{epoch}{chunk_id}-{conn_type[0]}-{conn_type[1]}-{EI}"

def routine_prep_neurons(sessions,
                         neuron_types:Union[list[str], str] = ['pyr', 'inter'], 
                         epochs:Union[list[str], str]="post",
                         brainstates:Union[list[str], str]=["REM","NREM"],
                         n_chunks:Union[list[str], str]=3,tight_epoch=False,):
    """
    sessions: subjects.ProcessData
        collection object of sessions

    tight_time: bool
        if true, try to shrink start and end of epoch to where brainstates are happening 
    """

    if not isinstance(neuron_types, list): neuron_types = [neuron_types]
    if not isinstance(sessions, list): sessions = [sessions]
    if not isinstance(epochs, list): epochs = [epochs]
    if not isinstance(brainstates, list): brainstates = [brainstates]
    if not isinstance(n_chunks, list): n_chunks = [n_chunks]
    assert len(n_chunks)==len(epochs)

    neurons_dict={"session_names":[],
                  'neuron_types':neuron_types,
                  "epochs":epochs,
                  "brainstates":brainstates,
                  'n_chunks':n_chunks,
                  'tight_epoch':tight_epoch,
                  }

    for sess in sessions: 
        sess_name = _short_session_name(sess)
        neurons_dict['session_names'].append(sess_name)
        neurons_dict[sess_name]={}
        for epoch,n_chunk in zip(epochs,n_chunks):
            p=sess.paradigm.label_slice(epoch)
            sess_neurons = sess.neurons.get_neuron_type(neuron_types) \
                    .time_slice(p.starts[0], p.stops[0]) \
                    .behav_slice(sess.brainstates,brainstates,tighten=tight_epoch) \
                    .time_split(n_chunks=n_chunk) # is a list  
            neurons_dict[sess_name][epoch]=sess_neurons
    return neurons_dict

"""
# def ccg_spike_assemble(spike_trains):
    #     Assemble an array of sorted spike times and cluIDs for the input cluster ids the list clus_use
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
"""

def routine_mean_firing_rates(neurons_dict):
    n_chunks=neurons_dict['n_chunks']
    epochs=neurons_dict['epochs']
    total_n_chunks = np.sum(n_chunks)
    neuron_types = neurons_dict['neuron_types'] # has to be 'inter'
    ntypes = len(neuron_types)
    alpha = 0.05

    print("Mean firing rates P VALUES")
    for sess_name in neurons_dict['session_names']:
        sess_neurons = neurons_dict[sess_name]

        overview_str=f"======={sess_name}=======\n"
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
                    effective_time[ie].append(neus.effective_time/60/60) # time in hours
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

def routine_eranconv_pairs(neurons_dict,
                           conn_types_E= [('pyr','pyr'), ('pyr','inter')],
                           conn_types_I = [('inter','inter'), ('inter','pyr')]
                           ):
    """
    Return neuronal pairs with valid CCG peaks (fast)
    Uses eran_conv

    window_width
        window width of the convolution kernel
        unit is milliseconds
        should be the same as `jscale` that you'd use for jittering
    """
    print("EranConv significant pairs")
    neuron_types =  neurons_dict['neuron_types']
    session_names = neurons_dict['session_names']
    n_chunks = neurons_dict['n_chunks']
    epochs = neurons_dict['epochs']
    conn_types = {'E':conn_types_E, 'I':conn_types_I}

    ################ CONFIG #################
    duration=20*1e-3 # 20ms - must be multiples of bin_size
    bin_size=1*1e-3 # 1ms
    window_width = 5 # 5ms
    alpha = 0.05
    alpha2 = 0.1
    use_multiple_correction = True
    C=int(duration/bin_size//2) # center bin
    nbins = int(duration/bin_size)

    min_lag = 1*1e-3 # 1ms
    max_lag = 3*1e-3 # 3ms
    min_spkcount = 2.5
    spkcount_scope = 12*1e-3 # 12ms total
    ignore_same_electrodes = True

    start=int(min_lag/bin_size)
    end=int(max_lag/bin_size)
    spkcount=int(spkcount_scope/2/bin_size)
    ############# END OF CONFIG ##############

    ################ UPDATE RETURN VALUES #################
    out = {'session_names':session_names,
                    'neuron_types':neuron_types,
                    'conn_types': conn_types,
                    'epochs': epochs,
                    'n_chunks': n_chunks,
                    'args': {
                        'duration': duration,
                        'bin_size': bin_size,
                        'window_width': window_width,
                        'alpha': alpha,
                        'alpha2': alpha2,
                        'use_multiple_correction': use_multiple_correction,
                        'min_lag': min_lag,
                        'max_lag': max_lag,
                        'min_lag_bin': start,
                        'max_lag_bin': end,
                        'min_spkcount': min_spkcount,
                        'spkcount_scope': spkcount_scope,
                        'spkcount_scope_bin': spkcount,
                        'ignore_same_electrodes': ignore_same_electrodes,
                        'nbins': nbins,
                        'center_bin': C,
                    }
                    }
    out = prep_connectivity_dict(out, session_names, epochs, conn_types)
    ############# END OF UPDATE RETURN VALUES ##############

    ### start of session loop ###
    for sess_name in session_names:
        for n_chunk, epoch in zip(n_chunks,epochs):
            sess_neurons = neurons_dict[sess_name][epoch]

            c_len_ef = [sess_neurons[_].effective_time/3600 for _ in range(n_chunk)]
            c_len = (sess_neurons[0].t_stop-sess_neurons[0].t_start)/3600

            ################ UPDATE PRINT STRING #################
            overview_str = f"======={sess_name}=======\n"
            overview_str+=f"Chunks are {c_len:.2f}h each and contain {[f'{cl:.2f}' for cl in c_len_ef]} hours of actual sleep "
            for _ in neuron_types: # sleep chunk
                overview_str+=f"{_}={sess_neurons[0].get_neuron_type(_).n_neurons} "
            overview_str+="\n"
            ############# END OF UPDATE PRINT STRING ##############

            ### start of chunks loop ###
            for c in range(n_chunk):
                ################ COMPUTATION #################
                neurons = sess_neurons[c]
                inds_by_type = {
                    'pyr':np.where(neurons.neuron_type=='pyr'),
                    'inter':np.where(neurons.neuron_type=='inter')
                }
                n = neurons.n_neurons
                if use_multiple_correction: 
                    corrected_alpha=alpha/(n**2-n)/nbins # local threshold
                    corrected_alpha2=alpha2/(n**2-n)/nbins

                ccg = correlations.spike_correlations(
                        neurons=neurons,
                        neuron_inds=np.arange(n), # all
                        bin_size=bin_size,
                        window_size=duration,
                        use_cupy=True,
                        symmetrize=True,
                        bin_mode='odd',
                    )
                
                pvals, pred, qvals = eran_conv(ccg,W=window_width,wintype="gauss",hollow_frac=None)
                print(pvals.shape)
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
                    'target_type':'pyr',
                    'p':0.05,
                    'p2':0.1
                }

                def _multiple_correction(pvals,alpha):
                    p_flat = pvals.ravel()
                    method = 'bonferroni' #'fdr_bh'
                    sig, p_corr, _, _ = multipletests(p_flat, alpha=alpha, method=method)
                    sig = sig.reshape(pvals.shape)
                    p_corr = p_corr.reshape(pvals.shape)   
                    return sig,p_corr
                
                sig, p_corr = _multiple_correction(pvals, alpha)
                coords_excitatory = np.argwhere((sig[...,C+start:C+end+1]).any(axis=-1))
                # coords_inhitibitory = np.argwhere((qvals[...,C+start:C+end+1]<alpha).any(axis=-1))

                sig1, q_corr1 = _multiple_correction(qvals, alpha)
                sig2, q_corr2 = _multiple_correction(qvals, alpha2)

                # sig1 = (qvals[...,C+start:C+end+1] < corrected_alpha)
                # sig2 = (qvals[...,C+start:C+end+1] < corrected_alpha2)
                neighbor = sig1 & (np.roll(sig2,1,-1)|np.roll(sig2,-1,-1)) # significant bins might have a neighbor who's kinda significant
                coords_inhitibitory = np.argwhere(neighbor.any(-1))

                coords_spkcount = np.argwhere((ccg[...,C-spkcount:C+spkcount+1]>=min_spkcount).all(axis=-1))
                def _intersect2d(n,coords1,coords2):
                    # Intersection of coordinate lists
                    coords1 = coords1[:,0]*n+coords1[:,1]
                    coords2 = coords2[:,0]*n+coords2[:,1]
                    coords=np.intersect1d(coords1,coords2)
                    coords=np.array([[x//n,x%n] for x in coords])
                    return coords
                coordsE = _intersect2d(n, coords_excitatory, coords_spkcount)
                coordsI = _intersect2d(n, coords_inhitibitory, coords_spkcount)

                def _count_significant_pairs(coords,neurons,conn_types,EI="E",ignore_same_electrodes=True):
                    """
                    Create a tally of significant neuronal connectoins by type
                    Currently, the type is defined as 
                        reference-target/[E,I]
                    where reference is presynaptic, and target is postsynaptic neuronal type, 
                    and E/I indicates the connection being excitatory or inhibitory

                    SL: If this helper function seems messy it's probably because 
                    it pertains to our specific definition of significant pairs (see Diba 2014, Pairwise connections.)

                    """
                    s=""
                    list_empty=True 
                    significant_pairs_ids = []
                    if coords.shape[0]:
                        # Condition 1: Ref/Target are never on the same electrode
                        if ignore_same_electrodes:
                            diff_channel=np.where(neurons.peak_channels[coords[:,0]]!=neurons.peak_channels[coords[:,1]])[0]
                            coords = coords[diff_channel]
                        # Conditoin 2: Specify Ref/Target cell types
                        for (ref,target) in conn_types:
                            sig_pairs=np.where(np.isin(coords[:,0],inds_by_type[ref]) & 
                                            np.isin(coords[:,1],inds_by_type[target]))[0]
                            sig_pairs=coords[sig_pairs]
                            significant_pairs_ids.append(sig_pairs)
                        # if any type of connection under consideration has a non-zero count, print a summary
                        if np.any([_.shape[0] for _ in significant_pairs_ids]):
                            list_empty=False 
                            for sig_pairs,(ref,target) in zip(significant_pairs_ids,conn_types):
                                s+=f"{ref}-{target}/{EI} {f'{sig_pairs.shape[0]:02d}' if sig_pairs.shape[0] else '-'} | "
                    if s=="":
                        s=f"no {'excitatory' if EI=='E' else 'inhbitory'} connections  "
                    return significant_pairs_ids,s,list_empty
                ### start of celltype loop ###
                excitatory_pairs, sE, list_emptyE = _count_significant_pairs(coordsE,neurons,conn_types_E,EI="E",ignore_same_electrodes=ignore_same_electrodes)
                inhibitory_pairs, sI, list_emptyI = _count_significant_pairs(coordsI,neurons,conn_types_I,EI="I",ignore_same_electrodes=ignore_same_electrodes)
                ### end of celltype loop ###
                n_E, n_I = coordsE.shape[0], coordsI.shape[0]
                ################ END OF COMPUTATION #################

                ################ UPDATE PRINT STRING #################
                overview_str += f"SLEEP{c}: E/I pairs {n_E:03d} / {n_I:03d} | "
                if list_emptyE and list_emptyI:
                    overview_str+="no connections\n"
                else:
                    overview_str=overview_str+sE+sI+"\n"
                ############# END OF UPDATE PRINT STRING ##############

                ################ UPDATE RETURN VALUES #################       
                out[sess_name][epoch]["per_chunk_len"] = c_len
                out[sess_name][epoch]["effective_chunk_lens"] = c_len_ef

                out[sess_name][epoch]['E']['total']=n_E
                for conn_type,ep in zip(conn_types_E,excitatory_pairs):
                    if ep.any(): 
                        ep_id = neurons.ind2id(ep)
                    else: ep_id = ep
                    out[sess_name][epoch]['E'][conn_type].append({
                        'ref':conn_type[0],'target':conn_type[1],
                        'ids':ep_id,
                        'inds':ep,
                        'ccg':ccg[ep[:,0],ep[:,1]],
                        'pred':pred[ep[:,0],ep[:,1]],
                        'pval':pvals[ep[:,0],ep[:,1]],
                    }) 
                out[sess_name][epoch]['I']['total']=n_I
                for conn_type,ip in zip(conn_types_I,inhibitory_pairs):
                    if ip.any(): 
                        ip_id = neurons.ind2id(ip)
                    else: ip_id = ip
                    out[sess_name][epoch]['I'][conn_type].append({
                        'ref':conn_type[0],'target':conn_type[1],
                        'ids':ip_id,
                        'inds':ip,
                        'ccg':ccg[ip[:,0],ip[:,1]],
                        'pred':pred[ip[:,0],ip[:,1]],
                        'pval':qvals[ip[:,0],ip[:,1]], # use qvals for inhibitory
                    })
                ############# END OF UPDATE RETURN VALUES ##############

                ### end of chunks loop ###
            print(overview_str)
            ### end of sessions loop ###
    return out
    ### end of function ###

def routine_eranconv_connection_info(info, neurons_dict, epoch_id=0,chunk_id=0):
    """
    Print aggregated information of eranconv_pairs() outputs

    info: eranconv_pairs outputs
    """
    results = {'E':{},'I':{}}
    total_by_conntype = {'E':{},'I':{}}
    total_by_EI = {'E':0,'I':0}
    for EI in ['E','I']:
        for conn_type in info['conn_types'][EI]:
            results[EI][conn_type]={'sig_conv':0,'list':[]}
            total_by_conntype[EI][conn_type] = 0
        
    neuron_types = info['neuron_types']
    epoch = neurons_dict['epochs'][epoch_id]

    for sess_name in info['session_names']:
        x = info[sess_name][epoch]
        neurons = neurons_dict[sess_name][epoch][chunk_id]

        n = {}
        for _ in neuron_types: 
            n[_] = neurons.get_neuron_type(_).n_neurons
                
        total_by_EI['E'] += x['E']['total']
        total_by_EI['I'] += x['I']['total']

        for EI in ['E','I']:
            for conn_type in info['conn_types'][EI]:
                try:
                    n_sig=len(x[EI][conn_type][0]['inds']) # Only has one session
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
        for conn_type in info['conn_types'][EI]:
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

def get_jitter_inputs(coords):
    """
    Reshape coordinates of (ref,target) pairs into most efficient format for jittering
    grouped by target indices

    coords: np array with shape (n,2)
    """
    coords=np.argsort(coords[:,1])
    keys, inv = np.unique(coords[:,1], return_inverse=True)
    jitter_inputs = [coords[inv==i,0].tolist()+[k] for i,k in enumerate(keys)]
    return jitter_inputs

def routine_jitter_pairs_after_conv(eranconv_out,neurons_dict,njitter=100,jitter_type='spike_time',filename=None):
    now = datetime.now()
    filename = filename or now.strftime("%Y-%m-%d-%H-%M")
    datafolder = '~/Documents/jitter_out'
    epochs = neurons_dict['epochs']
    n_chunks = neurons_dict['n_chunks']
    sess_names = neurons_dict['session_names']
    conn_types = eranconv_out['conn_types']

    for sess_name in sess_names:
        for epoch, n_chunk in zip(epochs,n_chunks):
            data_struct = {'njitter':njitter,
                        'jitter_type':jitter_type,}
            data_struct = prep_connectivity_dict(data_struct, sess_name, epochs, conn_types)
            sess_coords = eranconv_out[sess_name][epoch]
            sess_neurons = neurons_dict[sess_name][epoch]
            for EI in ['E','I']:
                for conn_type in conn_types[EI]:
                    for c in range(n_chunk):
                        coords = sess_coords[EI][conn_type][c]['inds']
                        neurons = sess_neurons[c]
                        j_inds = get_jitter_inputs(coords)
                        for _,j in enumerate(j_inds):
                            jitter_out = ccg_jitter(neurons=neurons,
                                                neuron_inds=j,
                                                njitter=njitter,
                                                use_cupy=True,
                                                bin_mode='odd')
                            data_struct[sess_name][EI][conn_type].append(jitter_out)
        with h5py.File(f"{datafolder}/jitter-{filename}.h5", "a") as f:
            f.create_dataset(f"{sess_name}", data=data_struct)   

# TODO: move to plotting in the future!
import seaborn as sns
import matplotlib.pyplot as plt
import os

def routine_eranconv_save_plots(neurons_dict,sessions,conv_dict,jitter_dict=None,
                                window_size = 0.02,bin_size = 0.001,
                                session_names=None,
                                chunk_ids=None,
                                conn_types=None,
                                epochs=None,
                                ):
    """
    TODO jitter p values are plotted, which should be similar to eran_conv p values.
    no way to specify which field of jitter_dict is plotted yet
    """
    images_folder = f"/home/selinali/Documents/NeuroPy/images/ccg_plots/test"
    epochs = epochs or neurons_dict['epochs']
    n_chunks = chunk_ids or neurons_dict['n_chunks']
    conn_types = conn_types or conv_dict['conn_types']
    session_names = session_names or neurons_dict['session_names']
    prep_connectivity_plot_folders(images_folder, session_names, epochs, n_chunks, conn_types)

    # TODO pass from previous functions
    for s, sess_name in enumerate(session_names):
        print(sess_name)
        if jitter_dict is not None:
            jitters = jitter_dict[sess_name]
        sess_neurons = neurons_dict[sess_name]
        conv = conv_dict[sess_name]
        frates_all = sessions[s].neurons.get_neuron_type(neurons_dict['neuron_types']).firing_rate
        for EI in ['E','I']:
            for conn_type in conn_types[EI]:
                for epoch,n_chunk in zip(epochs,n_chunks):
                    neurons = sess_neurons[epoch]
                    for c in range(n_chunk):
                        try:
                            k=conv[epoch][EI][conn_type][c]
                        except:
                            print(f"{sess_name}: No {conn_type} connections")
                            continue
                        if jitter_dict is not None:
                            jitter = jitters[EI][conn_type]
                        else:
                            jitter = None
                        coords=k['inds']#[np.random.random_integers(0,coords.shape[0]-1,5)]
                        s=np.argsort(coords[:,1])
                        coords=coords[s]
                        plotdir = get_connectivity_plot_folder(images_folder,sess_name,epoch,c,conn_type,EI)
                        for i,cd in enumerate(coords):
                            j = jitter[i] if jitter is not None else None
                            neurons_slice = neurons[c][cd]
                            plot_ccg_eranconv(neurons = neurons_slice,
                                            inds=cd,
                                            ccg=k['ccg'][s][i], 
                                            plotdir=plotdir, 
                                            window_size=window_size*1e3,
                                            bin_size=bin_size*1e3,
                                            pval=k['pval'][s][i],
                                            pred=k['pred'][s][i],
                                            jsig=j,
                                            frates_all=frates_all[cd],
                                            mode='odd')

def get_global_band(ccgs):
    if len(ccgs.shape==2):
        return np.min(ccgs[1:],axis=-1),np.max(ccgs[1:],axis=-1)
    elif len(ccgs.shape==3):
        return np.min(ccgs[:,1:],axis=-1),np.max(ccgs[:,1:],axis=-1)
    else:
        ValueError("CCGs must be 2D or 3D to calculate global significance band")

def plot_ccg_eranconv(neurons: Neurons, ccg, inds, plotdir, window_size, bin_size, 
                      pval, pred, frates_all, jsig=None, mode='even', ch_per_shank=16):
    """
    One CCG plot
    TODO This is defined in such a way that all plotting parameters are expanded. should i collapse and only pass in neurons?
    """
    fig, axs = plt.subplots(1,3,figsize=(10,5),gridspec_kw={'width_ratios':[2,1,1]})
    # generating even-numbered bins
    if mode=='even':
        bins = np.arange(-window_size / 2, window_size / 2, bin_size)+bin_size/2
        # bins = np.arange(-window_size / 2-bin_size, window_size / 2+bin_size/2, bin_size)+bin_size/2
    else:
        bins = np.arange(-window_size / 2, window_size / 2+bin_size, bin_size)
    ax=axs[0]
    ax.bar(bins, ccg, width=bin_size,alpha=0.5,label="ccg")
    ax.bar(bins, pred, width=bin_size,alpha=0.5,label='ccg-smooth')
    ax.plot(bins, pval*np.max(ccg), label='p')
    if jsig is not None:
        ax.plot(bins, jsig*np.max(ccg),label='j-significance')
    ax.set_xlabel("Time (millisecond)")
    ax.set_ylabel("Count")
    x,y=inds
    X,Y=neurons.neuron_ids
    ax.set_title(f"CCG, neuron_ids=[{X},{Y}], indices=[{x},{y}]")
    ax.legend()
    
    sns.despine(ax=ax)
    l = ['ref','target']
    for i in range(2):
        max_ch = neurons.waveforms[i].shape[0]
        ax=axs[1+i]
        ax.imshow(neurons.waveforms[i].astype(float))
        ax.set_title(f"{l[i]}: {neurons.neuron_type[i]}{neurons.neuron_ids[i]}")
        ax.set_xlabel(f"{frates_all[i]:.2f}Hz all | {neurons.firing_rate[i]:.2f}Hz sleep")
        # TODO shanks. ignoring that some channels may have been discarded
        for k in range(max_ch//ch_per_shank):
            ax.axhline((k+1)*ch_per_shank,c='w',alpha=0.5,linestyle='dashed')
    fig.savefig(f"{plotdir}/ccg-inds{x}-{y}.png")
    fig.tight_layout()
    plt.close(fig)
