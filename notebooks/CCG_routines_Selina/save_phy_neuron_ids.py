"""
load existing file with:
phy_neuron_ids=np.load('./phy_neuron_ids.npy',allow_pickle=True)
in neuropy/notebooks.
"""

from neuropy.io.phyio import PhyIO
import subjects
import numpy as np
sess = subjects.nsd.allsess[5]
cluster_path = sorted(sess.basepath.glob("**/params.py"))[1].parent
phydata = PhyIO(cluster_path)
print(phydata.neuron_ids)
np.save('./phy_neuron_ids.npy',phydata.neuron_ids,allow_pickle=True)
