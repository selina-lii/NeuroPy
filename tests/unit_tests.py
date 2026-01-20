import platform
import numpy as np

import pytest
from neuropy.core.session import ProcessData

cur_sys = platform.system()

if cur_sys == "Linux":
    pass

def get_test_session():
    import subjects
    sess = subjects.nsd.allsess[5]
    return sess

def test_plot_waveform_on_channel(sess:ProcessData):
    import neuropy.plotting.probe as probe
    neuron =sess.neurons
    ind1 = 26
    ind2 = 157
    shank_id1 = neuron.shank_ids[ind1]
    shank_id2 = neuron.shank_ids[ind2]
    ref_waveform=neuron.waveforms[ind1,shank_id1*16:(shank_id1+1)*16]
    traget_waveform=neuron.waveforms[ind2,shank_id1*16:(shank_id2+1)*16]
    x=probe.plot_waveform_on_channel(ref_waveform, shank_id1, traget_waveform, shank_id2, amplitude_limit=True)

def test_spike_correlation_snapshots(sess:ProcessData):
    """
    Validate that CCG results are identical using old/new implementations

    new implementation is about 3x faster
    """
    import neuropy.analyses.ms_connectivity as msconn
    conf=msconn.CCGConfig(use_acceleration=True,normalize=msconn.NormalizeBy.NONE)

    # new implementation, compute all time segments together
    c_newchunk=msconn.NeuronsDatasetConfig(seg_stride=60*60*0.5, seg_len=60*60*5, epochs=['post'], sleep_labels=None,recinfo=sess.recinfo)
    nd_n=msconn.NeuronsDataset(sess,c_newchunk)
    edges = nd_n.edge_times[list(nd_n.edge_times.keys())[0]]
    print("segment edges", edges)
    ccg_n=msconn.CCGDataset(nd_n,conf)
    ccg_n.get_ccg() 

    # old implemention, slice time to make a new neurons object
    c_oldchunk=msconn.NeuronsDatasetConfig(epochs=['post'], sleep_labels=None,recinfo=sess.recinfo)
    nd_o=[msconn.NeuronsDataset(sess,c_oldchunk) for _ in range(len(edges[0]))]
    for i,n in enumerate(nd_o):
        n.data[list(n.data.keys())[0]]=n.data[list(n.data.keys())[0]].time_slice(t_start=edges[0][i],t_stop=edges[1][i])
    ccg_o=[msconn.CCGDataset(n,conf) for n in nd_o]
    for c in ccg_o:
        c.get_ccg()
        c.get_connection_strengths(method="eran_conv")
        c.get_connectivity()
        neu=c.nd.data[list(c.nd.data.keys())[0]]
        print(np.sum([n.shape[0] for n in neu.spiketrains]))

    # should be identical
    print(ccg_o[0].data[list(ccg_o[0].data.keys())[0]].ccg)
    print(ccg_o[0].data[list(ccg_o[0].data.keys())[0]].inds)
    print(ccg_n.data[list(ccg_n.data.keys())[0]].ccg)
    print(ccg_n.data[list(ccg_n.data.keys())[0]].inds)


def test_view_ccg():
    ccg_n.example('connectivity')    
