#!/usr/bin/env python
"""Standalone CCG Review GUI launcher.
!python ./run_ccg_gui.py
Separate process so app.exec() never blocks a Jupyter kernel's heartbeat.
"""
from __future__ import annotations
import os
import PySide6
from neuropy.analyses.ms_connectivity import open_project

# --- CONFIG (edit these) ----------------------------------------------------
CONFIG   = 'test2'
DURATION = 20e-3
ALPHA    = 0.05
SESSION  = None          # None = restore last-used, or e.g. 'RatK_Day2'
import subjects  # notebooks/subjects.py — must run from notebooks/
from neuropy.io.datasets import bapun
sess = subjects.nsd.allsess + subjects.sd.allsess
# ----------------------------------------------------------------------------

neurons, cd, sd = open_project(CONFIG, sess)   # test2 predates headers

"""
Point Qt at PySide6's bundled plugins (a parent kernel can leave QT_* stale/empty).
Fixes qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""
"""
for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    os.environ.pop(var, None)
plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
if os.path.isdir(plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins
    os.environ["QT_PLUGIN_PATH"] = plugins

# Qt-dependent imports must follow the plugin-path fix above.
from pyqtgraph.Qt.QtWidgets import QApplication
from neuropy.ui.ccg_ui import CCGReviewUI

app = QApplication.instance() or QApplication([])
CCGReviewUI.launch(cd, SESSION).raise_()
app.exec()
