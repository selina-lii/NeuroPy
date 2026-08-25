#!/usr/bin/env python
"""Standalone CCG Review GUI launcher.
!python ./run_ccg_gui.py
Separate process so app.exec() never blocks a Jupyter kernel's heartbeat.
"""
from __future__ import annotations
import os
import PySide6
from neuropy.analyses.ms_connectivity import (build_project, open_project,
                                              projects_on_disk)

# --- CONFIG (edit these) ----------------------------------------------------
CONFIG  = None           # None = reopen last-used project, or e.g. 'test2'
SESSION = None           # None = restore last-used, or e.g. 'RatK_Day2'
# ----------------------------------------------------------------------------

import subjects  # notebooks/subjects.py — must run from notebooks/

# Point Qt at PySide6's bundled plugins: a parent kernel can leave QT_* stale or
# empty, and Qt then fails with 'Could not find the Qt platform plugin "cocoa"'.
for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    os.environ.pop(var, None)
plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
if os.path.isdir(plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins
    os.environ["QT_PLUGIN_PATH"] = plugins

# Qt-dependent imports must follow the plugin-path fix above.
from pyqtgraph.Qt.QtWidgets import QApplication
from neuropy.ui.ccg_ui import CCGReviewUI, UIStates
from neuropy.ui.dialogs import AddProjectDialog

# Guard required: jitter spawns worker processes, and spawn re-imports __main__.
# Without it every jitter task re-runs this file — reloading sessions and a second UI.
if __name__ == '__main__':
    sess = subjects.nsd.allsess + subjects.sd.allsess
    app = QApplication.instance() or QApplication([])   # the empty-project dialog needs one
    if not projects_on_disk():
        spec = AddProjectDialog.show(None)
        if spec is None:
            raise SystemExit("No project to open.")
        _, cd, _ = build_project(*spec)
    else:
        _, cd, _ = open_project(CONFIG or UIStates.last_project() or None, sess)
    CCGReviewUI.launch(cd, SESSION).raise_()
    app.exec()
