import numpy as np
try:
    import cupy as cp
except ImportError:
    print("Error importing CuPy")
    cp = None

# Define acceptable dtypes
_ACCEPTED_ARRAY_DTYPES = (
    float,
    int,
    bool,
)

def _san(var):
    """Sanitize """
    if var is None: return var
    if not (isinstance(var, list) or isinstance(var, np.ndarray)): var = [var]
    return var

def __cp_clean():
    """Call at the end of each cupy procedure to free resources"""
    cp.cuda.Stream.null.synchronize() # ensure gpu ops are complete
    cp._default_memory_pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

# Assemble Spike Arrayss
def _np_assemble_spike_arrays(neurons):
    """
    Assemble spike arrays for neurons from neurons object using NumPy.

    Returns
    -------
    spike_times: 1D array, spike timing in given time units
    spike_clusters: 1D array, index of neuron where spike came from
    spike_samples: 1D array, spike timing converted to sample indices

    """
    # Get spike times from neurons
    spike_times = np.concatenate(neurons.spiketrains)

    # Get neuron clusters
    spike_clusters = np.concatenate([
        np.full(len(spiketrain), cluster_id)
        for spiketrain, cluster_id in zip(neurons.spiketrains, neurons.neuron_ids)
    ])

    # Sort spike times and neuron clusters
    sort_ind = np.argsort(spike_times)

    # Get all sorted arrays
    spike_times = spike_times[sort_ind]
    spike_clusters = spike_clusters[sort_ind]
    spike_samples = (spike_times * neurons.sampling_rate).astype(int)

    # Debug: effect of removing duplicate spike timings
    # spike_samples=np.unique(spike_samples)
    # spks,count=np.unique(spike_samples,return_counts=True)
    # assert spks.shape[0]==count.sum()
    return spike_times, spike_clusters, spike_samples


def _cp_assemble_spike_arrays(neurons):
    """
    Assemble spike arrays for neurons from neurons object using CuPy.

    Returns
    -------
    spike_times: 1D array, spike timing in given time units
    spike_clusters: 1D array, index of neuron where spike came from
    spike_samples: 1D array, spike timing converted to sample indices

    """
    spike_times = cp.concatenate([cp.asarray(spiketrain) for spiketrain in neurons.spiketrains])

    # Get neuron clusters
    spike_clusters = cp.concatenate([
        cp.full(len(spiketrain), cluster_id, dtype=cp.int32)
        for spiketrain, cluster_id in zip(neurons.spiketrains, neurons.neuron_ids)
    ])

    # Sort spike times and neuron clusters
    sort_ind = cp.argsort(spike_times)

    # Get all sorted arrays
    spike_times = spike_times[sort_ind]
    spike_clusters = spike_clusters[sort_ind]
    spike_samples = (spike_times * neurons.sampling_rate).astype(cp.int32)

    return spike_times, spike_clusters, spike_samples


def _np_assemble_segment_ids_array(neurons,t_starts,t_ends):
    # t_starts, t_ends: specified time periods

    # create segments. 
    # the segments are another form of representation of t_starts and t_ends. 
    # as there might be overlaps between time periods, a segment can map to multiple time periods,
    #  hence the lookup table to map them back to time period ids.
    # segment ids are used to label spike identity for rolling computation of multiple CCGs.
    segments = np.unique(np.concatenate([t_starts,t_ends])) 
    # maps segment ids back to t_starts/t_ends ids
    lookup_table = [
        np.where((t_starts <= s) & (t_ends > s))[0]
        for s in segments[:-1]
    ]

    # Example:
    # suppose our time chunks of interest are 0~3, 1~5, and 2~3.
    # t_starts=[0, 1, 2], t_ends=[3, 5, 3]
    # segments = [0,1,2,3,5]
    # lookup_table = [[1],[1,2],[1,2,3],[2]]

    spike_segment_ids = np.concatenate([
        np.searchsorted(segments, spiketrain)-1 for spiketrain in neurons.spiketrains
    ])
    
    return spike_segment_ids, lookup_table


def _cp_assemble_segment_ids_array(neurons,t_starts,t_ends):
    # t_starts, t_ends: specified time periods

    edges = np.concatenate([[0], np.cumsum([n.shape[0] for n in neurons.spiketrains])])
    t_starts,t_ends=cp.array(t_starts),cp.array(t_ends)
    spike_segment_ids=cp.full((len(t_starts),int(edges[-1])),False)
    for i in range(len(neurons.spiketrains)):
        st = cp.array(neurons.spiketrains[i])[:,None]
        x = (st < t_ends) & (st >= t_starts)
        spike_segment_ids[:,edges[i]:edges[i+1]]=x.T 
    print(cp.sum(spike_segment_ids,axis=1))
    return spike_segment_ids

# spike orders are wrong somehow?


# Create Arrays
def _np_as_array(arr, dtype=None):
    """
    Convert an object to a numerical NumPy array.
    Avoid a copy if possible.
    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError(
            "'arr' seems to have an invalid dtype: " "{0:s}".format(str(out.dtype))
        )
    return out


def _cp_as_array(arr, dtype=None):
    """
    Convert an object to a numerical CuPy array.
    """
    if arr is None:
        return None
    if isinstance(arr, cp.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = cp.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    # Check for accepted CuPy dtypes
    accepted_dtypes = (cp.float32, cp.float64, cp.int32, cp.int64, cp.bool_)
    if out.dtype not in accepted_dtypes:
        raise ValueError(
            f"'arr' seems to have an invalid dtype: {out.dtype}"
        )
    return out


# Create Index array via lookup
def _np_index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.
    Implicitely assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _cp_index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.
    Implicitly assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements of arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Convert lookup to a CuPy array of int32
    lookup = cp.asarray(lookup, dtype=cp.int32)

    # Determine the size of the temporary array
    m = (lookup.max().item() if len(lookup) else 0) + 1  # Convert to Python int

    # Create the temporary array on the GPU
    tmp = cp.zeros(int(m + 1), dtype=cp.int32)  # Ensure size is an integer

    # Ensure that -1 values are kept
    tmp[-1] = -1

    # Map lookup values to their indices
    if len(lookup):
        tmp[lookup] = cp.arange(len(lookup), dtype=cp.int32)

    # Convert arr to CuPy array and return mapped indices
    arr = cp.asarray(arr, dtype=cp.int32)
    return tmp[arr]


# TODO interesting...
def _np_unique(x):
    """Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    if x is None or len(x) == 0:
        return np.array([], dtype=int)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _np_as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]


def _cp_unique(x):
    """Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    """
    CuPy implementation of _np_unique
    """
    if x is None or len(x) == 0:
        return cp.array([], dtype=cp.int32)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _cp_as_array(x)
    x = x[x >= 0]
    bc = cp.bincount(x)
    return cp.nonzero(bc)[0]


def _np_increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _np_as_array(arr)
    indices = _np_as_array(indices)
    bbins = np.bincount(indices)
    arr[: len(bbins)] += bbins
    return arr


def _cp_increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _cp_as_array(arr)
    indices = _cp_as_array(indices)
    bbins = cp.asarray(
        np.bincount(cp.asnumpy(indices))
    )  # NRK can you make this cupy? Maybe add in try/except statement?
    arr[: len(bbins)] += bbins
    return arr


def _np_diff_shifted(arr, steps=1):
    arr = _np_as_array(arr)
    return arr[steps:] - arr[: len(arr) - steps]


def _cp_diff_shifted(arr, steps=1):
    arr = _cp_as_array(arr)
    return arr[steps:] - arr[: len(arr) - steps]


def _np_create_correlograms_array(n_clusters, winsize_bins, n_groups=None):
    full_shape = (n_groups, n_clusters, n_clusters, winsize_bins // 2 + 1)
    return np.zeros(full_shape, dtype=np.int32) if n_groups else np.zeros(full_shape[1:], dtype=np.int32)


def _cp_create_correlograms_array(n_clusters, winsize_bins, n_groups=None):
    """Create an empty correlograms array using CuPy."""
    full_shape = (n_groups, n_clusters, n_clusters, winsize_bins // 2 + 1)
    return cp.zeros(full_shape, dtype=np.int32) if n_groups else cp.zeros(full_shape[1:], dtype=cp.int32)


def _np_create_correlograms_array_2groups(n_clusters1,n_clusters2, winsize_bins, symmetrize, n_groups=None):
    nbins = winsize_bins+1 if symmetrize else winsize_bins//2+1
    full_shape = (n_groups, n_clusters1, n_clusters2, nbins)
    return np.zeros(full_shape, dtype=np.int32) if n_groups else np.zeros(full_shape[1:], dtype=cp.int32)


def _cp_create_correlograms_array_2groups(n_clusters1,n_clusters2, winsize_bins, symmetrize, n_groups=None):
    nbins = winsize_bins+1 if symmetrize else winsize_bins//2+1
    full_shape = (n_groups, n_clusters1, n_clusters2, nbins)
    return cp.zeros(full_shape, dtype=cp.int32) if n_groups else cp.zeros(full_shape[1:], dtype=cp.int32)


def _np_create_correlograms_array_paired(n, winsize_bins, symmetrize, n_groups=None):
    nbins = winsize_bins+1 if symmetrize else winsize_bins//2+1
    full_shape = (n_groups, n, nbins)
    return np.zeros(full_shape, dtype=np.int32) if n_groups else np.zeros(full_shape[1:], dtype=cp.int32)


def _cp_create_correlograms_array_paired(n, winsize_bins, symmetrize, n_groups=None):
    nbins = winsize_bins+1 if symmetrize else winsize_bins//2+1
    full_shape = (n_groups, n, nbins)
    return cp.zeros(full_shape, dtype=cp.int32) if n_groups else cp.zeros(full_shape[1:], dtype=cp.int32)


def _np_symmetrize_correlograms(c):
    """Return the symmetrized version of the CCG arrays."""
    # We symmetrize c[..., i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    # correlograms[..., 0] = np.maximum(correlograms[..., 0], correlograms[..., 0].T)

    nbins = c.shape[-1]

    # sym first bin
    # SL & NRK: Phy's implementation of taking the maximum is wrong; either use half-bins and sum them, or do not combine the values
    c0 = c[..., 0]                 # view, no copy
    c0 += c0.swapaxes(-2, -1)

    # sym remaining bins: reverse k and transpose i,j in-place
    sym = c[..., 1:][..., ::-1].swapaxes(-2, -3)         # view, no copy

    # preallocate for speed
    shape = list(c.shape)
    shape[-1] = nbins*2-1
    out = np.empty(tuple(shape), dtype=c.dtype)
    out[..., :nbins-1] = sym
    out[..., nbins-1:] = c

    return out


def _cp_symmetrize_correlograms(c):
    """Return the symmetrized version of the CCG arrays."""
    # We symmetrize c[..., i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    # correlograms[..., 0] = np.maximum(correlograms[..., 0], correlograms[..., 0].T)

    nbins = c.shape[-1]

    # sym first bin
    # SL & NRK: Phy's implementation of taking the maximum is wrong; either use half-bins and sum them, or do not combine the values
    c0 = c[..., 0]                 # view, no copy
    c0 += c0.swapaxes(-2, -1)

    # sym remaining bins: reverse k and transpose i,j in-place
    sym = c[..., 1:][..., ::-1].swapaxes(-2, -3)         # view, no copy

    # preallocate for speed
    shape = list(c.shape)
    shape[-1] = nbins*2-1
    out = cp.empty(tuple(shape), dtype=c.dtype)
    out[..., :nbins-1] = sym
    out[..., nbins-1:] = c

    return out


# currently unused
def firing_rate(spike_clusters, cluster_ids=None, bin_size=None, duration=None):
    """Compute the average number of spikes per cluster per bin."""

    # Take the cluster order into account.
    if cluster_ids is None:
        cluster_ids = _np_unique(spike_clusters)
    else:
        cluster_ids = _np_as_array(cluster_ids)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _np_index_of(spike_clusters, cluster_ids)

    assert bin_size > 0
    bc = np.bincount(spike_clusters_i)
    # Handle the case where the last cluster(s) are empty.
    if len(bc) < len(cluster_ids):
        n = len(cluster_ids) - len(bc)
        bc = np.concatenate((bc, np.zeros(n, dtype=bc.dtype)))
    assert bc.shape == (len(cluster_ids),)
    return bc * np.c_[bc] * (bin_size / (duration or 1.0))


def np_spike_correlations(
        neurons,
        neuron_inds,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute all pairwise cross-correlations among neurons(clusters) given in neurons class.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    # Convert to array if int
    neuron_inds = _san(neuron_inds)

    neurons = neurons.neuron_slice(neuron_inds=neuron_inds)

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _np_assemble_spike_arrays(neurons)

    # Get binsize
    bin_size = np.clip(bin_size, 1e-5, 1e5)
    binsize = int(neurons.sampling_rate * bin_size)
    assert binsize >= 1, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}"

    # Get window-size dependent bins
    window_size = np.clip(window_size, 1e-5, 1e5)
    winsize_bins = 2 * int(0.5 * window_size / bin_size)
    assert winsize_bins >= 1
    # assert winsize_bins % 2 == 1 # TODO SL: winsize_bins will never be an odd number

    # Get unique neuron clusters
    clusters = _np_unique(spike_clusters)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _np_index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # each side has half+1 bins
    max_d = winsize_bins//2+1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_samples, dtype=bool)
    correlograms = _np_create_correlograms_array(n_clusters, winsize_bins,)


    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _np_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b >= max_d] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()

        # Update the masks given the clusters to update.
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index(
            (
                spike_clusters_i[:-shift][m],
                spike_clusters_i[+shift:][m],
                d
            ),
            correlograms.shape
        )

        # Increment the matching spikes in the correlograms array.
        _np_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)

    if symmetrize:
        correlograms=_np_symmetrize_correlograms(correlograms)

    # Fill in neurons with zero spikes
    n_neurons=neurons.n_neurons
    idxs = [np.where(neurons.neuron_ids == c)[0][0] for c in clusters]
    out = np.zeros((n_neurons, n_neurons, correlograms.shape[-1]), dtype=correlograms.dtype)
    out[np.ix_(idxs, idxs)] = correlograms

    return out


def cp_spike_correlations(
        neurons,
        neuron_inds,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute all pairwise cross-correlations among neurons(clusters) given in neurons class.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    # Convert to array if int
    neuron_inds = _san(neuron_inds)

    neurons = neurons.neuron_slice(neuron_inds=neuron_inds)

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _cp_assemble_spike_arrays(neurons)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    print(f"Bin size is {binsize}")
    assert binsize >= 1, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}"

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size)

    # Get unique neuron clusters
    clusters = _cp_unique(spike_clusters)
    n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    # 'center' mode has one more bin at the center
    max_d = winsize_bins//2+1

    mask = cp.ones_like(spike_samples, dtype=cp.bool_)
    correlograms = _cp_create_correlograms_array(n_clusters, winsize_bins)
    print(spike_samples)
    print(spike_clusters_i)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _cp_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b >= max_d] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # # Update the masks given the clusters to update.
        # m0 = cp.in1d(spike_clusters[:-shift], clusters)
        # m = m & m0
        # d = spike_diff_b[m]
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = cp.ravel_multi_index(
            (spike_clusters_i[:-shift][m], spike_clusters_i[+shift:][m], d),
            correlograms.shape,
        )

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)
        # print(indices.shape[0])

        shift += 1

    if symmetrize:
        correlograms=_cp_symmetrize_correlograms(correlograms).get()
    else:
        correlograms=correlograms.get()

    n_neurons=neurons.n_neurons
    idxs = [np.where(neurons.neuron_ids == c)[0][0] for c in clusters.get()]
    out = np.zeros((n_neurons, n_neurons, correlograms.shape[-1]), dtype=correlograms.dtype)
    out[np.ix_(idxs, idxs)] = correlograms

    print("shift:", shift)

    __cp_clean()
    return out


def np_spike_correlations_2groups(
        neurons,
        ref_inds,
        target_inds,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute pairwise cross-correlations between reference neuron(s) and all
    non-reference neurons(clusters) given by indices.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    assert bin_size>=1/neurons.sampling_rate, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}. Bins must be longer than one sampling interval"

    # Convert to array if int
    target_inds = _san(target_inds)
    ref_inds = _san(ref_inds)
    all_inds = np.concatenate([np.array(ref_inds),np.array(target_inds)])

    # SL: get the threshold of which neuron indices are group0 vs group1
    N0=len(ref_inds)
    N1=len(target_inds)
    
    # TODO! makeshift solution, shouldn't need to relabel neurons. Find out what's wrong-
    # Reindex neurons. References are 0...N0-1 and targets are N0...N0+N1-1    
    neurons = neurons.neuron_slice(neuron_inds=all_inds)
    for i in range(neurons.n_neurons):
        neurons.neuron_ids[i]=i

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _np_assemble_spike_arrays(neurons)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    
    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) # total number of bins
    
    # Get unique neuron clusters
    clusters = _np_unique(spike_clusters)
    # n_clusters = len(clusters)

    spike_clusters_i = _np_index_of(spike_clusters, clusters)

    shift = 1

    mask = np.ones_like(spike_samples, dtype=np.bool_)
    correlograms = _np_create_correlograms_array_2groups(N0, N1, winsize_bins, symmetrize)
    
    # 'center' mode has one more bin at the center
    max_d = winsize_bins//2+1
    center = winsize_bins//2

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    # SL: shift walks over spike indices, not time lags. shift+=1 discards the last spike from the train.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _np_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        # SL: spike_diff_b is which bin the spike_diff falls in 
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        # SL: If there are no matching spikes now, there wouldn't be any with a larger shift.
        mask[:-shift][spike_diff_b >= max_d] = False # SL exclude spike pairs that fall outside the ccg window after shifting

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m] # SL: get which bins need to be incremented

        # SL: This function only computes intergroup ccgs between
        #   reference (group0) and target (group1)
        #   even tho all neurons are pooled in one spiketrain in which
        #   the first N clusters are group0 and the others are group1

        # SL: group0->group1 forward connections. create an intergroup mask
        ref=spike_clusters_i[:-shift][m]
        target=spike_clusters_i[+shift:][m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        gm = (ref < N0) & (target >= N0)
        indices = np.ravel_multi_index(
            (ref[gm], target[gm]-N0, d[gm]+(center if symmetrize else 0)),
            correlograms.shape,
        )

        if symmetrize:
            gm_sym = (ref >= N0) & (target < N0)
            indices_sym= np.ravel_multi_index(
                (target[gm_sym], ref[gm_sym]-N0, center-d[gm_sym]),
                correlograms.shape,
            )
            indices = np.concatenate([indices,indices_sym])

        # Increment the matching spikes in the correlograms array.
        _np_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)
    correlograms=correlograms
    return correlograms


def cp_spike_correlations_2groups(
        neurons,
        ref_inds,
        target_inds,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute pairwise cross-correlations between reference neuron(s) and all
    non-reference neurons(clusters) given by indices.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    assert bin_size>=1/neurons.sampling_rate, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}. Bins must be longer than one sampling interval"

    # Convert to array if int
    target_inds = _san(target_inds)
    ref_inds = _san(ref_inds)
    all_inds = np.concatenate([np.array(ref_inds),np.array(target_inds)])

    # SL: get the threshold of which neuron indices are group0 vs group1
    N0=len(ref_inds)
    N1=len(target_inds)
    
    # TODO! makeshift solution, shouldn't need to relabel neurons. Find out what's wrong-
    # Reindex neurons. References are 0...N0-1 and targets are N0...N0+N1-1    
    neurons = neurons.neuron_slice(neuron_inds=all_inds)
    for i in range(neurons.n_neurons):
        neurons.neuron_ids[i]=i

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _cp_assemble_spike_arrays(neurons)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    
    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) # total number of bins
    
    # Get unique neuron clusters
    clusters = _cp_unique(spike_clusters)
    # n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    mask = cp.ones_like(spike_samples, dtype=cp.bool_)
    correlograms = _cp_create_correlograms_array_2groups(N0, N1, winsize_bins, symmetrize)
    
    # 'center' mode has one more bin at the center
    max_d = winsize_bins//2+1
    center = winsize_bins//2

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    # SL: shift walks over spike indices, not time lags. shift+=1 discards the last spike from the train.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _cp_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        # SL: spike_diff_b is which bin the spike_diff falls in 
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        # SL: If there are no matching spikes now, there wouldn't be any with a larger shift.
        mask[:-shift][spike_diff_b >= max_d] = False # SL exclude spike pairs that fall outside the ccg window after shifting

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m] # SL: get which bins need to be incremented

        # SL: This function only computes intergroup ccgs between
        #   reference (group0) and target (group1)
        #   even tho all neurons are pooled in one spiketrain in which
        #   the first N clusters are group0 and the others are group1

        # SL: group0->group1 forward connections. create an intergroup mask
        ref=spike_clusters_i[:-shift][m]
        target=spike_clusters_i[+shift:][m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        gm = (ref < N0) & (target >= N0)
        indices = cp.ravel_multi_index(
            (ref[gm], target[gm]-N0, d[gm]+(center if symmetrize else 0)),
            correlograms.shape,
        )

        if symmetrize:
            gm_sym = (ref >= N0) & (target < N0)
            indices_sym= cp.ravel_multi_index(
                (target[gm_sym], ref[gm_sym]-N0, center-d[gm_sym]),
                correlograms.shape,
            )
            indices = cp.concatenate([indices,indices_sym])

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)
    correlograms=correlograms.get()
    __cp_clean()
    return correlograms


#TODO probably gonna remove paired. super slow
def np_spike_correlations_paired(
        neurons,
        ref_inds,
        target_inds,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute pairwise cross-correlations between pairs of 
    reference - non-reference neuron(clusters) given by indices.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, winsize_samples)` array with paired CCGs.
    """
    # TODO test

    assert bin_size>=1/neurons.sampling_rate, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}. Bins must be longer than one sampling interval"

    # Convert to array if int
    target_inds = _san(target_inds)
    ref_inds = _san(ref_inds)
    all_inds = np.concatenate([np.array(ref_inds),np.array(target_inds)])
    assert len(ref_inds)==len(target_inds)
    N = len(ref_inds)

    # Reindex neurons. References are 0...N-1 and targets are N...2N-1    
    neurons = neurons.neuron_slice(neuron_inds=all_inds)
    for i in range(neurons.n_neurons):
        neurons.neuron_ids[i]=i

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _np_assemble_spike_arrays(neurons)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    
    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) # total number of bins
    
    # Get unique neuron clusters
    clusters = _np_unique(spike_clusters)
    # n_clusters = len(clusters)

    spike_clusters_i = _np_index_of(spike_clusters, clusters)

    shift = 1

    mask = np.ones_like(spike_samples, dtype=np.bool_)
    correlograms = _np_create_correlograms_array_paired(N, winsize_bins, symmetrize)
    
    max_d = winsize_bins//2+1
    center = winsize_bins//2

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    # SL: shift walks over spike indices, not time lags. shift+=1 discards the last spike from the train.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _np_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        # SL: spike_diff_b is which bin the spike_diff falls in 
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        # SL: If there are no matching spikes now, there wouldn't be any with a larger shift.
        mask[:-shift][spike_diff_b >= max_d] = False # SL exclude spike pairs that fall outside the ccg window after shifting

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m] # SL: get which bins need to be incremented

        # SL: This function only computes intergroup ccgs between
        #   reference - target pairs in a list
        #   even tho all neurons are pooled in one spiketrain in which
        #   the first N clusters are group0 and the others are group1

        ref=spike_clusters_i[:-shift][m]
        target=spike_clusters_i[+shift:][m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        # SL: group0:group1 forward connections. create an intergroup mask
        gm = (target-ref == N)
        indices = np.ravel_multi_index(
            (ref[gm], d[gm]+(center if symmetrize else 0)),
            correlograms.shape,
        )

        if symmetrize:
            gm = (ref-target == N)
            indices_sym= np.ravel_multi_index(
                (target[gm], center-d[gm]), # TODO verify this
                correlograms.shape,
            )
            indices = np.concatenate([indices,indices_sym])

        # Increment the matching spikes in the correlograms array.
        _np_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)
    correlograms=correlograms
    return correlograms


def cp_spike_correlations_paired(
        neurons,
        ref_inds,
        target_inds,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute pairwise cross-correlations between pairs of 
    reference - non-reference neuron(clusters) given by indices.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, winsize_samples)` array with paired CCGs.
    """
    # TODO test

    assert bin_size>=1/neurons.sampling_rate, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}. Bins must be longer than one sampling interval"

    # Convert to array if int
    target_inds = _san(target_inds)
    ref_inds = _san(ref_inds)
    all_inds = np.concatenate([np.array(ref_inds),np.array(target_inds)])
    assert len(ref_inds)==len(target_inds)
    N = len(ref_inds)

    # Reindex neurons. References are 0...N-1 and targets are N...2N-1    
    neurons = neurons.neuron_slice(neuron_inds=all_inds)
    for i in range(neurons.n_neurons):
        neurons.neuron_ids[i]=i

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _cp_assemble_spike_arrays(neurons)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    
    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) # total number of bins
    
    # Get unique neuron clusters
    clusters = _cp_unique(spike_clusters)
    # n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    mask = cp.ones_like(spike_samples, dtype=cp.bool_)
    correlograms = _cp_create_correlograms_array_paired(N, winsize_bins, symmetrize)
    
    max_d = winsize_bins//2+1
    center = winsize_bins//2

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    # SL: shift walks over spike indices, not time lags. shift+=1 discards the last spike from the train.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _cp_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        # SL: spike_diff_b is which bin the spike_diff falls in 
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        # SL: If there are no matching spikes now, there wouldn't be any with a larger shift.
        mask[:-shift][spike_diff_b >= max_d] = False # SL exclude spike pairs that fall outside the ccg window after shifting

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m] # SL: get which bins need to be incremented

        # SL: This function only computes intergroup ccgs between
        #   reference - target pairs in a list
        #   even tho all neurons are pooled in one spiketrain in which
        #   the first N clusters are group0 and the others are group1

        ref=spike_clusters_i[:-shift][m]
        target=spike_clusters_i[+shift:][m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        # SL: group0:group1 forward connections. create an intergroup mask
        gm = (target-ref == N)
        indices = cp.ravel_multi_index(
            (ref[gm], d[gm]+(center if symmetrize else 0)),
            correlograms.shape,
        )

        if symmetrize:
            gm = (ref-target == N)
            indices_sym= cp.ravel_multi_index(
                (target[gm], center-d[gm]), # TODO verify this
                correlograms.shape,
            )
            indices = cp.concatenate([indices,indices_sym])

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)
    correlograms=correlograms.get()
    __cp_clean()
    return correlograms


def np_spike_correlations_snapshots(
        neurons,
        neuron_inds,
        t_starts,
        t_ends,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute N pairwise cross-correlations at once using multiple subsets from one set of spike trains.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    assert bin_size>=1/neurons.sampling_rate, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}. Bins must be longer than one sampling interval"

    # Convert to array if int
    neuron_inds = _san(neuron_inds)

    neurons = neurons.neuron_slice(neuron_inds=neuron_inds)

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _np_assemble_spike_arrays(neurons)
    spike_segment_ids, segments_lookup_table = _np_assemble_segment_ids_array(neurons,t_starts,t_ends)

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    
    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) # total number of bins
    
    # Get unique neuron clusters
    clusters = _np_unique(spike_clusters)
    n_clusters = len(clusters)

    spike_clusters_i = _np_index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # each side has half+1 bins
    max_d = winsize_bins//2+1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_samples, dtype=bool)
    correlograms = _np_create_correlograms_array(n_clusters, winsize_bins,n_groups=len(t_starts))
    
    # The loop continues as long as there is at least one spike with
    # a matching spike.
    # SL: shift walks over spike indices, not time lags. shift+=1 discards the last spike from the train.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _np_diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        # SL: spike_diff_b is which bin the spike_diff falls in 
        spike_diff_b = (spike_diff+binsize//2) // binsize

        # Spikes with no matching spikes are masked.
        # SL: If there are no matching spikes now, there wouldn't be any with a larger shift.
        mask[:-shift][spike_diff_b >= max_d] = False # SL exclude spike pairs that fall outside the ccg window after shifting

        # Cache the masked spike delays.
        m = mask[:-shift].copy()

        # Update the masks given the clusters to update.
        d = spike_diff_b[m] # SL: get which bins need to be incremented

        ref_=spike_segment_ids[:-shift][m]
        target_=spike_segment_ids[+shift:][m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        for ii,chunks in enumerate(segments_lookup_table):
            m_segment = (ref_ == ii) & (target_ == ii) # time chunk mask
            dm = d[m_segment]
            for i_ccg in chunks:
                indices = np.ravel_multi_index(
                        (
                            np.full(dm.shape,i_ccg), 
                            spike_clusters_i[:-shift][m][m_segment], 
                            spike_clusters_i[+shift:][m][m_segment],
                            dm
                        ),
                        correlograms.shape,
                        )
                # Increment the matching spikes in the correlograms array.
                _np_increment(correlograms.ravel(), indices)
        shift += 1

    if symmetrize:
        correlograms = _np_symmetrize_correlograms(correlograms)

    print("shift", shift)

    # Fill in neurons with zero spikes
    n_neurons=neurons.n_neurons
    idxs = [np.where(neurons.neuron_ids == c)[0][0] for c in clusters]
    out = np.zeros((len(t_starts), n_neurons, n_neurons, correlograms.shape[-1]), dtype=correlograms.dtype)
    for i,c in enumerate(correlograms):
        out[i][np.ix_(idxs,idxs)]=c
    return out


def cp_spike_correlations_snapshots(
        neurons,
        neuron_inds,
        t_starts,
        t_ends,
        bin_size=None,
        window_size=None,
        symmetrize=True,
):
    """
    Compute N pairwise cross-correlations at once using multiple subsets from one set of spike trains.

    Parameters
    ----------
    neurons : core.neurons
        neurons obj containing spiketrains and related info
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    symmetrize : boolean (True)
        Whether the output matrix should be symmetrized or not.

    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise CCGs.
    """

    assert bin_size>=1/neurons.sampling_rate, f"Bin size {bin_size} is too small for sampling rate {neurons.sampling_rate}. Bins must be longer than one sampling interval"

    # Convert to array if int
    neuron_inds = _san(neuron_inds)

    neurons = neurons.neuron_slice(neuron_inds=neuron_inds)

    # Get spike times from neurons
    spike_times, spike_clusters, spike_samples = _cp_assemble_spike_arrays(neurons)
    spike_segment_ids = _cp_assemble_segment_ids_array(neurons,cp.asarray(t_starts),cp.asarray(t_ends))

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    binsize = int(neurons.sampling_rate * bin_size)  # in samples
    
    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds  # NRK can you make this cupy? does it matter?
    winsize_bins = 2 * int(0.5 * window_size / bin_size) # total number of bins
    
    # Get unique neuron clusters
    clusters = _cp_unique(spike_clusters)
    n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    # each side has half+1 bins
    max_d = winsize_bins//2+1
    correlograms = _cp_create_correlograms_array(n_clusters, winsize_bins,n_groups=len(t_starts))

    for i_ccg in range(len(t_starts)):
        print("here",i_ccg)
        # Shift between the two copies of the spike trains.
        shift = 1

        # At a given shift, the mask precises which spikes have matching spikes
        # within the correlogram time window.
        mi = spike_segment_ids[i_ccg] # spikes in this segment
        seg_spike_samples = spike_samples[mi]
        seg_clusters_i = spike_clusters_i[mi]
        print(seg_spike_samples)
        print(seg_clusters_i)
        mask = cp.ones_like(seg_spike_samples, dtype=cp.bool_)

        # The loop continues as long as there is at least one spike with
        # a matching spike.
        # SL: shift walks over spike indices, not time lags. shift+=1 discards the last spike from the train.
        while mask[:-shift].any():
            # Number of time samples between spike i and spike i+shift.
            spike_diff = _cp_diff_shifted(seg_spike_samples, shift)

            # Binarize the delays between spike i and spike i+shift.
            # SL: spike_diff_b is which bin the spike_diff falls in 
            spike_diff_b = (spike_diff+binsize//2) // binsize

            # Spikes with no matching spikes are masked.
            # SL: If there are no matching spikes now, there wouldn't be any with a larger shift.
            mask[:-shift][spike_diff_b >= max_d] = False # SL exclude spike pairs that fall outside the ccg window after shifting

            # Cache the masked spike delays.
            m = mask[:-shift].copy()

            # Update the masks given the clusters to update.
            d = spike_diff_b[m] # SL: get which bins need to be incremented

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike clusters.

            # same segment
            indices = cp.ravel_multi_index(
                    (
                        cp.full_like(d,i_ccg), 
                        seg_clusters_i[:-shift][m], 
                        seg_clusters_i[+shift:][m],
                        d
                    ),
                    correlograms.shape,
                    )
            # Increment the matching spikes in the correlograms array.
            _cp_increment(correlograms.ravel(), indices)
            # print(indices.shape[0])
            shift += 1
        print("shift", shift)

    if symmetrize:
        correlograms = _cp_symmetrize_correlograms(correlograms).get()
    else:
        correlograms = correlograms.get()

    # Fill in neurons with zero spikes
    n_neurons=neurons.n_neurons
    idxs = [np.where(neurons.neuron_ids == c.get())[0][0] for c in clusters]
    out = np.zeros((len(t_starts), n_neurons, n_neurons, correlograms.shape[-1]), dtype=correlograms.dtype)
    for i,c in enumerate(correlograms):
        out[i][np.ix_(idxs,idxs)]=c
    
    __cp_clean()
    return out


def spike_correlations(
        neurons,
        neuron_inds,
        ref_neuron_inds=None,
        bin_size=None,
        window_size=None,
        symmetrize=True,
        use_acceleration=False,
        paired=False,
        chunk_edges=None
):
    """
    Switch between spike correlation cases.

        paired : 
        If True this will compute pairwise (ref_inds[k],target_inds[k]) correlations only

    """
    print("running spike correlations")
    if chunk_edges is not None:
        if use_acceleration:
            correlograms = cp_spike_correlations_snapshots(neurons, 
            neuron_inds = neuron_inds, t_starts=chunk_edges[0], t_ends=chunk_edges[1],
            bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
        else:
            correlograms = np_spike_correlations_snapshots(neurons,
            neuron_inds = neuron_inds, t_starts=chunk_edges[0], t_ends=chunk_edges[1],
            bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
    elif ref_neuron_inds is not None:
        if paired:
            if use_acceleration:
                correlograms = cp_spike_correlations_paired(neurons, ref_inds=ref_neuron_inds, 
                target_inds = neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
            else:
                correlograms=None# TODO write np version
        else:
            if use_acceleration:
                correlograms = cp_spike_correlations_2groups(neurons, ref_inds=ref_neuron_inds,
                target_inds = neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
            else:
                correlograms=None# TODO write np version
    else:
        if use_acceleration:
            correlograms = cp_spike_correlations(neurons, neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
        else:
            correlograms = np_spike_correlations(neurons, neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
    print("spike correlation done")
    return correlograms

