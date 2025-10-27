import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    print("Error importing JAX. No GPU acceleration available.") # Was CuPy
    jnp=None


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

# Assemble Spike Arrays
def _np_assemble_spike_arrays(neurons):
    """
    Assemble spike arrays for neurons from neurons object using NumPy.
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
    spike_times: 1D array
    spike_clusters: 1D array, index of neuron where spike came from
    spike_samples: 1D array, spike timing converted to sample indices

    """
    spike_times = jnp.concatenate([jnp.asarray(spiketrain) for spiketrain in neurons.spiketrains])

    # Get neuron clusters
    spike_clusters = jnp.concatenate([
        jnp.full(len(spiketrain), cluster_id, dtype=jnp.int32)
        for spiketrain, cluster_id in zip(neurons.spiketrains, neurons.neuron_ids)
    ])

    # Sort spike times and neuron clusters
    sort_ind = jnp.argsort(spike_times)

    # Get all sorted arrays
    spike_times = spike_times[sort_ind]
    spike_clusters = spike_clusters[sort_ind]
    spike_samples = (spike_times * neurons.sampling_rate).astype(jnp.int32)

    return spike_times, spike_clusters, spike_samples


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
    if isinstance(arr, jnp.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = jnp.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    # Check for accepted CuPy dtypes
    accepted_dtypes = (jnp.float32, jnp.float64, jnp.int32, jnp.int64, jnp.bool_)
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
    lookup = jnp.asarray(lookup, dtype=jnp.int32)

    # Determine the size of the temporary array
    m = (lookup.max().item() if len(lookup) else 0) + 1  # Convert to Python int

    # Create the temporary array on the GPU
    tmp = jnp.zeros(int(m + 1), dtype=jnp.int32)  # Ensure size is an integer

    # Ensure that -1 values are kept
    tmp[-1] = -1

    # Map lookup values to their indices
    if len(lookup):
        tmp[lookup] = jnp.arange(len(lookup), dtype=jnp.int32)

    # Convert arr to CuPy array and return mapped indices
    arr = jnp.asarray(arr, dtype=jnp.int32)
    return tmp[arr]

# TODO interesting...
# Get unique values
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


def _unique_cupy(x):
    """
    CuPy implementation of _np_unique
    """
    if x is None or len(x) == 0:
        return jnp.array([], dtype=jnp.int32)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _cp_as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]



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
    bbins = jnp.asarray(
        np.bincount(jnp.asnumpy(indices))
    )  # NRK can you make this cupy? Maybe add in try/except statement?
    arr[: len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    arr = _np_as_array(arr)
    return arr[steps:] - arr[: len(arr) - steps]


def _cp_diff_shifted(arr, steps=1):
    arr = _cp_as_array(arr)
    return arr[steps:] - arr[: len(arr) - steps]


def _np_create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=np.int32)


def _cp_create_correlograms_array(n_clusters, winsize_bins):
    """Create an empty correlograms array using CuPy."""
    return jnp.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=jnp.int32)


def _cp_create_correlograms_array_2groups(n_clusters1,n_clusters2, winsize_bins, symmetrize):
    """Create an empty correlograms array using CuPy."""
    nbins = winsize_bins+1 if symmetrize else winsize_bins//2+1
    return jnp.zeros((n_clusters1, n_clusters2, nbins), dtype=jnp.int32)

def _cp_create_correlograms_array_paired(n, winsize_bins, symmetrize):
    """Create an empty correlograms array using CuPy."""
    nbins = winsize_bins+1 if symmetrize else winsize_bins//2+1
    return jnp.zeros((n, nbins), dtype=jnp.int32)

def _np_symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    # correlograms[..., 0] = np.maximum(correlograms[..., 0], correlograms[..., 0].T)

    correlograms[..., 0] = np.add(correlograms[..., 0], correlograms[..., 0].T)
    # Symmetrize the remaining bins
    sym = correlograms[..., 1:][..., ::-1]

    sym = np.transpose(sym, (1, 0, 2))
    return np.dstack((sym, correlograms))


def _cp_symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays using CuPy."""
    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # # Symmetrize correlograms[..., 0]
    # TODO SL & NRK: maximum is wrong; either use half-bins and sum them, or do not combine the two values
    correlograms[..., 0] = jnp.add(correlograms[..., 0], correlograms[..., 0].T)
    # Symmetrize the remaining bins
    sym = correlograms[..., 1:][..., ::-1]

    sym = jnp.transpose(sym, (1, 0, 2))
    correlograms = jnp.dstack((sym, correlograms))
    return correlograms


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
        spike_diff = _diff_shifted(spike_samples, shift)

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

    if symmetrize:
        correlograms=_np_symmetrize_correlograms(correlograms)
    
    return correlograms


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
    clusters = _unique_cupy(spike_clusters)
    n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    # 'center' mode has one more bin at the center
    max_d = winsize_bins//2+1

    mask = jnp.ones_like(spike_samples, dtype=jnp.bool_)
    correlograms = _cp_create_correlograms_array(n_clusters, winsize_bins)

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
        # m0 = jnp.in1d(spike_clusters[:-shift], clusters)
        # m = m & m0
        # d = spike_diff_b[m]
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = jnp.ravel_multi_index(
            (spike_clusters_i[:-shift][m], spike_clusters_i[+shift:][m], d),
            correlograms.shape,
        )

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)

        shift += 1

    if symmetrize:
        correlograms=_cp_symmetrize_correlograms(correlograms).get()
    else:
        correlograms=correlograms.get()

    n_neurons=neurons.n_neurons
    idxs = [np.where(neurons.neuron_ids == c)[0][0] for c in clusters.get()]
    out = np.zeros((n_neurons, n_neurons, correlograms.shape[-1]), dtype=correlograms.dtype)
    out[np.ix_(idxs, idxs)] = correlograms

    jnp.get_default_memory_pool().free_all_blocks()
    return out


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
    paired : 
        If True this will compute pairwise (ref_inds[k],target_inds[k]) correlations only

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
    clusters = _unique_cupy(spike_clusters)
    # n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    mask = jnp.ones_like(spike_samples, dtype=jnp.bool_)
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
        indices = jnp.ravel_multi_index(
            (ref[gm], target[gm]-N0, d[gm]+(center if symmetrize else 0)),
            correlograms.shape,
        )

        if symmetrize:
            gm_sym = (ref >= N0) & (target < N0)
            indices_sym= jnp.ravel_multi_index(
                (target[gm_sym], ref[gm_sym]-N0, center-d[gm_sym]),
                correlograms.shape,
            )
            indices = jnp.concatenate([indices,indices_sym])

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)
    correlograms=correlograms.get()
    jnp.get_default_memory_pool().free_all_blocks()
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
    clusters = _unique_cupy(spike_clusters)
    # n_clusters = len(clusters)

    spike_clusters_i = _cp_index_of(spike_clusters, clusters)

    shift = 1

    mask = jnp.ones_like(spike_samples, dtype=jnp.bool_)
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
        indices = jnp.ravel_multi_index(
            (ref[gm], d[gm]+(center if symmetrize else 0)),
            correlograms.shape,
        )

        if symmetrize:
            gm = (ref-target == N)
            indices_sym= jnp.ravel_multi_index(
                (target[gm], center-d[gm]), # TODO verify this
                correlograms.shape,
            )
            indices = jnp.concatenate([indices,indices_sym])

        # Increment the matching spikes in the correlograms array.
        _cp_increment(correlograms.ravel(), indices)

        shift += 1

    print("shift", shift)
    correlograms=correlograms.get()
    jnp.get_default_memory_pool().free_all_blocks()
    return correlograms


def spike_correlations(
        neurons,
        neuron_inds,
        ref_neuron_inds=None,
        bin_size=None,
        window_size=None,
        symmetrize=True,
        use_acceleration=False,
        paired=False
):
    """
    Switch between spike correlation cases.

        paired : 
        If True this will compute pairwise (ref_inds[k],target_inds[k]) correlations only

    """
    if ref_neuron_inds is not None:
        if paired:
            if use_acceleration:
                correlograms = cp_spike_correlations_paired(neurons, ref_inds=ref_neuron_inds, target_inds = neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
            else:
                pass# TODO write np version
        else:
            if use_acceleration:
                correlograms = cp_spike_correlations_2groups(neurons, ref_inds=ref_neuron_inds, target_inds = neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
            else:
                pass# TODO write np version
        return correlograms
    else:
        if use_acceleration:
            correlograms = cp_spike_correlations(neurons, neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
        else:
            correlograms = np_spike_correlations(neurons, neuron_inds, bin_size=bin_size, window_size=window_size, symmetrize=symmetrize)
        return correlograms

