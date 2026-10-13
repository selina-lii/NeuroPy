"""TEMPORARY: export every CCG a stats test's rows use, one grid PNG per (row, resolution, segment)."""
from __future__ import annotations

import datetime
import math
import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pyqtgraph.Qt.QtWidgets import QMessageBox, QPushButton

from neuropy.analyses.neurons_dataset import Key
from neuropy.ui.app_state import _ALL_SEGS
from neuropy.ui.stats_tests_frontend import StatsTestPanel

N_COLS = 8


def row_ccgs(backend, cfg, resolution: str, seg_name: str) -> list:
    """``[(title, lags_ms, ccg)]`` for one row, normalised as the test's display config normalises them."""
    out = []
    for conn_type in cfg.conn_types or ['']:
        ei, ct = backend.cd.conf.parse_conn_type_label(conn_type)
        for sess in cfg.sessions or backend.available_sessions():
            ptr_key = Key(session=sess, excitability=ei, conn_type=ct)
            if ptr_key not in backend.cd.ptr:
                continue
            data_key = ptr_key.change(resolution=resolution, segment=seg_name)
            try:
                if backend.cd.ccg_for(data_key) is None:
                    continue
            except FileNotFoundError:   # segment never computed for this session, as the test skips it
                continue
            seg = backend.cd.segment_index(data_key, seg_name)
            ccg, _ = backend.cd.apply_ccg_transform_for(data_key, backend.norms)
            n = ccg.shape[-1]
            lags = (np.arange(n) - n // 2) * backend.cd.conf.at(resolution).bin_size * 1e3
            for ref, tgt in sorted(backend.pairs_for_rules(cfg.rules, ptr_key)):
                ids = backend.cd.nd.ids_at(Key.pair(sess, int(ref), int(tgt)))
                out.append((f"{sess} {ids[0]}→{ids[1]}", lags, ccg[seg, ref, tgt]))
    return out


def save_grid(ccgs: list, title: str, path: str) -> None:
    n_rows = max(1, math.ceil(len(ccgs) / N_COLS))
    fig = Figure(figsize=(2.2 * N_COLS, 1.8 * n_rows), layout='constrained')
    FigureCanvasAgg(fig)
    fig.suptitle(f"{title}  (n={len(ccgs)})")
    axes = fig.subplots(n_rows, N_COLS, squeeze=False).ravel()
    for ax, (name, lags, ccg) in zip(axes, ccgs):
        ax.bar(lags, ccg, width=lags[1] - lags[0], color='k')
        ax.axvline(0, color='r', lw=0.5)
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(ccgs):]:
        ax.axis('off')
    fig.savefig(path, dpi=100)


def export_group_ccgs(backend, out_dir: str) -> list:
    """Every row's CCGs in each of its resolutions and segments; returns the written paths."""
    backend.resolve_follows()
    backend.sd.ensure_groups_loaded_for(backend.sessions_for())
    backend.ensure_loaded()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, cfg in enumerate(backend.rows, 1):
        for resolution in cfg.resolution or backend.available_resolutions():
            for seg_name in cfg.segments or [_ALL_SEGS]:
                title = f"row{i}_{cfg.name or 'unnamed'}_{resolution}_seg-{seg_name}"
                path = os.path.join(out_dir, "".join(c if c.isalnum() or c in "-_." else "_" for c in title) + ".png")
                save_grid(row_ccgs(backend, cfg, resolution, seg_name), title, path)
                paths.append(path)
    return paths


def _on_export_ccgs_btn(panel: StatsTestPanel) -> None:
    panel._sync_backend()
    out_dir = os.path.join(panel.backend.save_dir, "group_ccgs",
                           datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    paths = export_group_ccgs(panel.backend, out_dir)
    QMessageBox.information(panel, "Export group CCGs", f"{len(paths)} PNG(s) written to\n{out_dir}")


def install() -> None:
    """Add the export button to every stats panel built from now on."""
    build = StatsTestPanel._build

    def _build(self):
        build(self)
        btn = QPushButton("Export group CCGs (temp)")
        btn.clicked.connect(lambda: _on_export_ccgs_btn(self))
        self.layout().addWidget(btn)

    StatsTestPanel._build = _build
