#!/usr/bin/env python3
"""Launch CCG Review UI against pre-migration data in data/project_test2/."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(ROOT, 'notebooks')

os.environ.setdefault(
    'QT_QPA_PLATFORM_PLUGIN_PATH',
    os.path.join(
        sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages', 'PySide6', 'Qt', 'plugins', 'platforms'))

sys.path.insert(0, NOTEBOOKS)
sys.path.insert(0, ROOT)
os.chdir(NOTEBOOKS)

import neuropy.analyses.neurons_dataset as nd_mod
import neuropy.analyses.ms_connectivity as msconn
import subjects

nd = nd_mod.NeuronsDataset(
    subjects.nsd.allsess + subjects.sd.allsess,
    nd_mod.NeuronsDatasetConfig(epochs=['pre', 'maze', 'post', 're-maze', 'sd', 'rs']))
cd = msconn.CCGDataset(
    msconn.CCGConfig(use_acceleration=False, duration=20e-3, alpha=0.05, name='test2'),
    nd)

import pyqtgraph as pg

app = pg.mkQApp('CCG Review (test2)')

from neuropy.ui.ccg_ui import CCGReviewUI

launch_key = CCGReviewUI.default_launch_key(cd, 'RatK_Day2')
win = CCGReviewUI.launch(cd, launch_key)
sys.exit(app.exec())
