#!/usr/bin/env python
"""Standalone CCG Review GUI launcher — runs in its own process.

Why a separate process:
  Inside a Jupyter/VSCode kernel, `app.exec()` blocks the kernel's main
  thread so ipykernel can't answer the ZMQ heartbeat → VSCode declares the
  kernel dead and kills it. Running the GUI in a separate process keeps the
  kernel free (and never touches Qt), so it cannot crash the kernel.

Usage
-----
Terminal (blocks this terminal until you close the window):
    conda activate NeuroPy2
    cd notebooks
    python run_ccg_gui.py                 # default session RatK_Day2
    python run_ccg_gui.py --session RatK_Day2

From a notebook cell (NON-blocking — kernel stays alive):
    import subprocess, sys, os
    proc = subprocess.Popen(
        [sys.executable, "run_ccg_gui.py", "--session", "RatK_Day2"],
        cwd=os.path.dirname("notebooks/"))   # or just cwd="notebooks"
    # window opens in its own process; keep using the notebook.
    # proc.terminate() to close it programmatically.

Data is rebuilt from the cached CCG pointers on disk (fast), so this does
not recompute correlograms.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys

faulthandler.enable()  # dump a C traceback to stderr on any segfault


def _fix_qt_plugin_path() -> None:
    """Point Qt at PySide6's bundled platform plugins.

    A parent shell/kernel (e.g. a VSCode-launched notebook spawning this via
    subprocess) can leave QT_QPA_PLATFORM_PLUGIN_PATH / QT_PLUGIN_PATH set to
    "" or a stale value, causing:
        'Could not find the Qt platform plugin "cocoa" in ""'.
    Drop the inherited values (so PySide6 self-configures on import) and then
    pin them explicitly to PySide6's own plugins dir — robust either way.
    """
    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        os.environ.pop(var, None)
    import PySide6
    plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
    if os.path.isdir(plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins
        os.environ["QT_PLUGIN_PATH"] = plugins


_fix_qt_plugin_path()

# --- Defaults mirror CCG_gui.ipynb -----------------------------------------
DEFAULT_CONFIG = 'test2'
DEFAULT_EPOCHS = ['pre', 'maze', 'post', 're-maze', 'sd', 'rs']
DEFAULT_SESSION = 'RatK_Day2'
DEFAULT_DURATION = 20e-3
DEFAULT_ALPHA = 0.05


def build_dataset(config_name: str, epochs: list[str],
                  duration: float, alpha: float):
    """Reconstruct the CCGDataset exactly as the notebook does."""
    import neuropy.analyses.neurons_dataset as neurons_dataset
    import neuropy.analyses.ms_connectivity as msconn
    import subjects  # notebooks/subjects.py — must run from notebooks/

    sess = subjects.nsd.allsess + subjects.sd.allsess
    print(f"[run_ccg_gui] building NeuronsDataset ({len(sess)} sessions)…",
          flush=True)
    ndconf = neurons_dataset.NeuronsDatasetConfig(epochs=epochs)
    nd = neurons_dataset.NeuronsDataset(sess, ndconf)
    cconf = msconn.CCGConfig(use_acceleration=False, duration=duration,
                             alpha=alpha, name=config_name)
    cd = msconn.CCGDataset(cconf, nd)
    print("[run_ccg_gui] dataset ready.", flush=True)
    return cd


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch the CCG Review GUI.")
    ap.add_argument('--config', default=DEFAULT_CONFIG,
                    help='CCGConfig name (cache folder).')
    ap.add_argument('--session', default=DEFAULT_SESSION,
                    help='Session to open, e.g. RatK_Day2.')
    ap.add_argument('--epochs', nargs='+', default=DEFAULT_EPOCHS)
    ap.add_argument('--duration', type=float, default=DEFAULT_DURATION)
    ap.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args()

    cd = build_dataset(args.config, args.epochs, args.duration, args.alpha)

    from pyqtgraph.Qt.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from neuropy.ui.ccg_ui import CCGReviewUI
    print(f"[run_ccg_gui] launching GUI for session {args.session!r}…",
          flush=True)
    win = CCGReviewUI.launch(cd, cd.find(args.session))
    win.raise_()
    win.activateWindow()

    print("[run_ccg_gui] GUI open — close the window to exit.", flush=True)
    app.exec()
    print("[run_ccg_gui] window closed, exiting.", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
