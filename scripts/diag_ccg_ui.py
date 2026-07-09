#!/usr/bin/env python3
"""Offscreen diagnostic: type switch pair counts + render logging."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(ROOT, 'notebooks')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, NOTEBOOKS)
sys.path.insert(0, ROOT)
os.chdir(NOTEBOOKS)

import neuropy.analyses.neurons_dataset as nd_mod
import neuropy.analyses.ms_connectivity as msconn
import subjects
import pyqtgraph as pg

pg.mkQApp('ccg diag')

nd = nd_mod.NeuronsDataset(
    subjects.nsd.allsess + subjects.sd.allsess,
    nd_mod.NeuronsDatasetConfig(epochs=['pre', 'maze', 'post', 're-maze', 'sd', 'rs']))
cd = msconn.CCGDataset(
    msconn.CCGConfig(use_acceleration=False, duration=20e-3, alpha=0.05, name='test'),
    nd)

from neuropy.ui.ccg_ui import CCGReviewUI

key = CCGReviewUI.default_launch_key(cd, 'RatK_Day2')
print(f"launch_key={key} label={CCGReviewUI._type_label(key)}", flush=True)
win = CCGReviewUI(cd, key)
win._group_mgr._autoload_session_latest(restore_groups=True)
win._apply_sel_for_key(key)
n0 = len(win._nav.all_inds)
n0_sel = len(win.sel_data.selected_inds)
print(f"after autoload: {n0} pairs sel={n0_sel}", flush=True)
win._ccg_panel.request_render()

# simulate type switch to I INT→PYR
for i in range(win._type_combo.count()):
    tk = win._type_combo.itemData(i)
    lbl = win._type_label(tk)
    if 'INT' in lbl and 'PYR' in lbl and lbl.startswith('I'):
        win._switch_session(win._canonical_ptr_key(tk))
        n = len(win._nav.all_inds)
        ns = len(win.sel_data.selected_inds)
        print(f"switch {lbl}: {n} pairs sel={ns}", flush=True)
        win._ccg_panel.request_render()
        break

# simulate E PYR→PYR
for i in range(win._type_combo.count()):
    tk = win._type_combo.itemData(i)
    lbl = win._type_label(tk)
    if lbl.startswith('E') and 'PYR→PYR' in lbl:
        win._switch_session(win._canonical_ptr_key(tk))
        n = len(win._nav.all_inds)
        ns = len(win.sel_data.selected_inds)
        nu = len(win.sel_data.unselected_inds)
        print(f"switch {lbl}: {n} pairs sel={ns} unsel={nu}", flush=True)
        win._ccg_panel.request_render()
        break

print("diag done", flush=True)
