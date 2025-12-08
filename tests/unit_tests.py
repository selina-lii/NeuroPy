import platform


import pytest
from neuropy import *

cur_sys = platform.system()

if cur_sys == "Linux":
    pass


def test_plot_waveform_on_channel():
    import neuropy.plotting.probe as probe
    sess = subjects.nsd.allsess[5]
    neuron =sess.neurons
    ind1 = 26
    ind2 = 157
    shank_id1 = neuron.shank_ids[ind1]
    shank_id2 = neuron.shank_ids[ind2]
    ref_waveform=neuron.waveforms[ind1,shank_id1*16:(shank_id1+1)*16]
    traget_waveform=neuron.waveforms[ind2,shank_id1*16:(shank_id2+1)*16]
    x=probe.plot_waveform_on_channel(ref_waveform, shank_id1, traget_waveform, shank_id2, amplitude_limit=True)