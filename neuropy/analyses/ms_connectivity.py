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
from neuropy.core.epoch import Epoch
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
    Randomly shift each spike in target spike train
    Maximum shift is +/-(jscale) milliseconds

    Parameters
    ----------
    njitter : int
        number of jitters
    neuron_inds : list
        [a1, ..., an, b]
        a1~an: indices of reference neurons
        b: index of target neuron
    jscale: int
        defines maximum time shift of a spike in seconds
    use_cupy: bool, optional
        whether or not to use gpu acceleration

    Returns
    -------
    neurons: Neurons=
        a Neurons object containing (njitter+2) neurons, with indices 0...njitter.
        first neuron is the reference cell, index=0
        second neuron is the target cell with index=1
        the next (njitter) neurons are jitters of the target cell
    """
    neurons = neurons.get_by_id(neuron_inds)
    nref=len(neuron_inds)-1
    new_ref_inds=np.arange(nref)
    new_target_ind=nref
    jitter_inds=nref+1+np.arange(njitter)
    for i in range(nref+1):
        neurons.neuron_ids[i]=i # ref

    target_nspikes = neurons.n_spikes[new_target_ind]
    target_type = neurons.neuron_type[new_target_ind]
    target_spiketrain = neurons.spiketrains[new_target_ind]

    if use_cupy:
        jittertrains = (
            cp.round(
                (
                    cp.array(target_spiketrain)
                    + 2 * jscale * cp.random.rand(njitter,target_nspikes)
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
                    target_spiketrain
                    + 2 * jscale * np.random.rand(njitter,target_nspikes)
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
        neuron_ids=jitter_inds,
        neuron_type=[target_type]*njitter
        ) # TODO not copying over other fields
    neurons.merge(jittered)
    return neurons, new_ref_inds, new_target_ind, jitter_inds
    
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
    bin_size=0.001,
    duration=0.02,
    jscale=5,
    njitter=100,
    alpha=0.05,
    use_cupy=False,
    symmetrize_mode='even',
):

    # SL: These were comments from Nat I guess - 
    # most of these should naturally be fixed as I update the function
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
    neuronsj,ref_inds,target_ind,jitter_inds = add_jitter(neurons=neurons,
            njitter=njitter,
            neuron_inds=neuron_inds,
            jscale=jscale,
            use_cupy=use_cupy,
        )
        # Jitter spikes in second cluster
    # NRK TODO: start debugging here!
    # spikes_sorted, clus_sorted = ccg_spike_assemble(spike_trains)

    # re-run ccg
    ccg_all=correlations.spike_correlations(
            neurons=neuronsj,
            ref_neuron_inds=ref_inds,
            neuron_inds=np.concatenate([[target_ind],jitter_inds]),
            bin_size=bin_size,
            window_size=duration,
            use_cupy=use_cupy,
            symmetrize=True,
            symmetrize_mode=symmetrize_mode,
        )

    # Debugging - 'debug' should be all zeros (two methods are identical)
    # orig = correlations.spike_correlations(
    #         neurons=neuronsj,
    #         neuron_inds=np.arange(neuronsj.n_neurons),
    #         bin_size=bin_size,
    #         window_size=duration,
    #         use_cupy=use_cupy,
    #         symmetrize=True,
    #         symmetrize_mode=symmetrize_mode,
    #     )
    # debug = orig[0,len(ref_inds):]-ccg_all[0]
    # print(debug)
    
    #TODO fix this tomorrow!
    # pval = where real data is ranked among fake data. conservative when there are ties
    pval = np.argsort(np.argsort(-ccg_all,axis=1,kind="stable"),axis=1)[:,0]/njitter
    # threshold = 
    thresholds = np.percentile(ccg_all[:,1:],100*(1-alpha), axis=0)
    print(ccg_all[:,0].shape,thresholds.shape)
    significances = pval<alpha

    # significances = correlograms > thresholds

    return ccg_all, pval, significances, # orig


def pairwise_conn_fast(neurons: Neurons,
    neuron_inds=None,
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


def _short_session_name(session):
    """get short printable session name in the format of ANIMAL_DayX"""
    sess_name = session.filePrefix.parts[-1].split('_')[:2]
    sess_name='_'.join(sess_name)
    return sess_name

def _split_session(n_chunks, start,stop, neurons: Neurons):
    """
    Evenly divide session into n chunks by recording time
    """
    chunk_starts = np.histogram_bin_edges([],bins=n_chunks,range=(start,stop))[:-1]
    chunk_stops = np.histogram_bin_edges([],bins=n_chunks,range=(start,stop))[1:]
    chunked = [neurons.time_slice(s,e,zero_spike_times=True) for s,e in zip(chunk_starts, chunk_stops)]
    return chunked

def _split_session_brainstate(bs_timing:Epoch, labels, neurons:Neurons, shrink=False):
    brainstates = bs_timing.label_slice(labels).duration_slice(min_dur=120) # "QW","AW","REM","NREM"
    intervals = list(zip(brainstates.starts,brainstates.stops))
    state_neurons=neurons.time_multislices(intervals,shrink=shrink)
    return state_neurons

def routine_eranconv_pairs(sessions,epoch="post",brainstates=["REM","NREM"],n_chunks=3,tight_bounds=False,return_neurons=False):
    """
    sessions: subjects.ProcessData, collection object of sessions
    """
    print("EranConv significant pairs")
    neuron_types = ['pyr','inter'] # has to be 'inter'
    conn_types_E = [('pyr','pyr'), ('pyr','int')]
    conn_types_I = [('int','int'), ('int','pyr')]
    all_connections = {}
    if not isinstance(sessions, list):
        sessions = [sessions]
    for sess in sessions:
        # Get data
        ind = np.where(sess.paradigm.labels==epoch)[0][0]
        start, stop=sess.paradigm.starts[ind],sess.paradigm.stops[ind]
        overview_str = f"======={_short_session_name(sess)}=======\n"
        sess_neurons = sess.neurons.get_neuron_type(neuron_types) \
                                    .time_slice(start, stop)
        sess_neurons = _split_session_brainstate(sess.brainstates,brainstates,sess_neurons,shrink=tight_bounds)
        sess_neurons = _split_session(n_chunks,start,stop,sess_neurons)
        
        chunk_len = sess_neurons[0].effective_time/60/60
        overview_str+=f"Each chunk is {chunk_len:.2f} hours  "
        sess_connections = {"t":chunk_len,"E":{},"I":{}}
        for conn_type in conn_types_E:
            sess_connections['E'][conn_type]=[]
        for conn_type in conn_types_I:
            sess_connections['I'][conn_type]=[]

        ################ CONFIG #################
        duration=20*1e-3 # 20ms
        bin_size=1*1e-3 # 1ms
        window_width = 5
        alpha = 0.05
        C=int(duration/bin_size//2) # center bin

        min_lag = 1*1e-3 # 1ms
        max_lag = 3*1e-3 # 3ms
        min_spkcount = 2.5
        spkcount_scope = 12*1e-3 # 12ms total
        ignore_same_electrodes = True

        start=int(min_lag/bin_size)
        end=int(max_lag/bin_size)
        spkcount=int(spkcount_scope/2/bin_size)
        ############# END OF CONFIG ##############

        for i in neuron_types: # sleep chunk
            overview_str+=f"{i}={sess_neurons[0].get_neuron_type(i).n_neurons} "
        overview_str+="\n"

        for c in range(n_chunks):
            neurons = sess_neurons[c]
            ids_by_type = {
                'pyr':neurons.get_neuron_type('pyr').neuron_ids,
                'int':neurons.get_neuron_type('inter').neuron_ids
            }
            n = neurons.n_neurons
            corrected_alpha=alpha/(n**2) # multipl comparison

            pvals, ccg, pred, qvals=pairwise_conn_fast(neurons,
                neuron_inds=None,
                bin_size=bin_size,
                duration=duration,
                window_width=window_width,
                wintype="gauss", 
                hollow_frac=None,
                alpha=corrected_alpha,
                use_multi_correction=True,
                use_cupy=True,
                symmetrize_mode='odd',
            )

            coords_excitatory = np.argwhere((pvals[...,C+start:C+end+1]<corrected_alpha).any(axis=-1))
            coords_inhitibitory = np.argwhere((qvals[...,C+start:C+end+1]<corrected_alpha).any(axis=-1))
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

            def _count_significant_pairs(coords,neurons,conn_types_E,EI="E",ignore_same_electrodes=True):
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
                significant_pairs = []
                if coords.shape[0]:
                    # Condition 1: Ref/Target are never on the same electrode
                    if ignore_same_electrodes:
                        diff_channel=np.where(neurons.peak_channels[coords[:,0]]!=neurons.peak_channels[coords[:,1]])[0]
                        coords = coords[diff_channel]
                    # Conditoin 2: Specify Ref/Target cell types
                    for (ref,target) in conn_types_E:
                        sig_pairs=np.where(np.isin(coords[:,0],ids_by_type[ref]) & 
                                        np.isin(coords[:,1],ids_by_type[target]))[0]
                        sig_pairs=coords[sig_pairs]
                        significant_pairs.append(sig_pairs)
                    # if any type of connection under consideration has a non-zero count, print a summary
                    if np.any([_.shape[0] for _ in significant_pairs]):
                        list_empty=False 
                        for sig_pairs,(ref,target) in zip(significant_pairs,conn_types_E):
                            s+=f"{ref}-{target}/{EI} {f'{sig_pairs.shape[0]:02d}' if sig_pairs.shape[0] else '-'} | "
                if s=="":
                    s=f"no {'excitatory' if EI=='E' else 'inhbitory'} connections  "
                return significant_pairs,s,list_empty
            ### start of celltype loop ###
            excitatory_pairs, sE, list_emptyE = _count_significant_pairs(coordsE,neurons,conn_types_E,EI="E",ignore_same_electrodes=ignore_same_electrodes)
            inhibitory_pairs, sI, list_emptyI = _count_significant_pairs(coordsI,neurons,conn_types_I,EI="I",ignore_same_electrodes=ignore_same_electrodes)
            ### end of celltype loop ###
            overview_str += f"SLEEP{c}: E/I pairs {coordsE.shape[0]:03d} / {coordsI.shape[0]:03d} | "
            if list_emptyE and list_emptyI:
                overview_str+="no connections\n"
            else:
                overview_str=overview_str+sE+sI+"\n"

            ## format values for return ##
            for conn_type,ep in zip(conn_types_E,excitatory_pairs):
                sess_connections['E'][conn_type].append({
                    'ref':conn_type[0],'target':conn_type[1],
                    't':chunk_len,
                    'ids':ep,
                    'ccg':ccg[ep[:,0],ep[:,1]],
                    'pred':pred[ep[:,0],ep[:,1]],
                    'pval':pvals[ep[:,0],ep[:,1]],
                    'qval':qvals[ep[:,0],ep[:,1]],
                    'neurons':{
                        'ids':np.unique(ep),
                        'frates':neurons.firing_rate[np.unique(ep)]
                    }
                }) 
            for conn_type,ip in zip(conn_types_I,inhibitory_pairs):
                sess_connections['I'][conn_type].append({
                    'ref':conn_type[0],'target':conn_type[1],
                    't':chunk_len,
                    'ids':ip,
                    'ccg':ccg[ip[:,0],ip[:,1]],
                    'pred':pred[ip[:,0],ip[:,1]],
                    'pval':pvals[ip[:,0],ip[:,1]],
                    'qval':qvals[ip[:,0],ip[:,1]],
                    'neurons':{
                        'ids':np.unique(ep),
                        'frates':neurons.firing_rate[np.unique(ep)]
                    }
                })
            if return_neurons:
                sess_connections['neurons']=neurons
            ### end of chunks loop ###
        all_connections[_short_session_name(sess)]=sess_connections
        print(overview_str)
        ### end of sessions loop ###
    return all_connections
    ### end of function ###

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



from scipy.stats import ttest_ind
def routine_mean_firing_rates(sessions, epochs = "post", n_chunks = 3, 
                                 brainstates=["REM","NREM"], alpha=0.05):
    if isinstance(epochs,str):
        epochs=[epochs]
    if isinstance(n_chunks,int):
        n_chunks=[n_chunks]
    assert len(n_chunks)==len(epochs), "incomplete parameters"
    total_n_chunks = np.sum(n_chunks)
    neuron_types = ['pyr','inter'] # has to be 'inter'
    ntypes = len(neuron_types)
    types_printname = ['Pyramidal neurons',
                        'Interneurons\t',]

    print("Mean firing rates P VALUES")
    for sess in sessions:
        sess_name = _short_session_name(sess)
        overview_str=f"======={sess_name}=======\n"
        neurons = sess.neurons.get_neuron_type(neuron_types)
        neurons=_split_session_brainstate(sess.brainstates,brainstates,neurons)
        neurons_chunked,labels=[],[]
        for epoch_name,n_chunk in zip(epochs,n_chunks):
            p=sess.paradigm.label_slice(epoch_name)
            _ = _split_session(n_chunk, p.starts[0],p.stops[0], neurons)
            neurons_chunked.append(_)
            labels+=[f"{epoch_name.capitalize()}{i+1}" for i in range(n_chunk)]

        for i_type in range(ntypes):
            # initialize data structures
            nneurons = 0
            mean_firing_rates,\
            sd_firing_rates,\
            iqr,\
            frates,\
            effective_time = [],[],[],[],[]

            for i_epoch in range(len(epochs)):
                mean_firing_rates.append([])
                sd_firing_rates.append([])
                iqr.append([])
                frates.append([])
                effective_time.append([])

                for i_chunk in range(n_chunks[i_epoch]):
                    neus = neurons_chunked[i_epoch][i_chunk]
                    neus = neus.get_neuron_type(neuron_types[i_type])
                    nneurons=neus.n_neurons
                    frate = neus.firing_rate if nneurons>0 else 0
                    frates[i_epoch].append(frate)
                    mean_firing_rates[i_epoch].append(np.mean(frate))
                    sd_firing_rates[i_epoch].append(np.std(frate))
                    effective_time[i_epoch].append(neus.effective_time/60/60) # time in hours
                    if neus.n_neurons>5:
                        iqr[i_epoch].append(np.percentile(frate, 75)-np.percentile(frate, 25))
                ### end of chunks loop ###
            
            overview_str+=f"{i_type+1}. {types_printname[i_type]}\t"
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
            ### end of celltype loop ###
            
        print(overview_str)
        ### end of sessions loop ###
    return effective_time, mean_firing_rates, sd_firing_rates, iqr, frates
    ### end of function ###