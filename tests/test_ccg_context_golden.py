"""Characterization test: RenderContext output must not change.

Records every numeric field CCGContextBuilder produces across a parameter sweep
into a golden .npz, then compares later runs against it. Refactors are safe only
while this passes.

    python tests/test_ccg_context_golden.py --record   # capture the baseline
    python tests/test_ccg_context_golden.py            # verify against it
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

import PySide6

for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    os.environ.pop(_var, None)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_PLUGINS = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _PLUGINS
os.environ["QT_PLUGIN_PATH"] = _PLUGINS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "notebooks"))

from pyqtgraph.Qt.QtWidgets import QApplication
import subjects
from neuropy.analyses.ms_connectivity import open_project
from neuropy.ui.ccg_ui import CCGReviewUI, UIStates
from neuropy.ui.ccg_panel import CCGContextBuilder

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ccg_context_golden.npz")

# fields worth pinning: every array or scalar a plot is drawn from
ARRAY_FIELDS = ('ccg', 'ccg_null_plot', 'pval', 'pval_corrected',
                'acg_ref', 'acg_tgt')
SCALAR_FIELDS = ('bin_size_eff', 'window_size_eff', 'alpha', 'inds',
                 'min_lag_plot', 'max_lag_plot', 'base_window_ms',
                 'is_significant_pair', 'seg_id_display')


def open_ui():
    """Launch the UI headless on the last-used project, on a key that has pairs."""
    app = QApplication.instance() or QApplication([])
    sessions = subjects.nsd.allsess + subjects.sd.allsess
    _, cd, _ = open_project(UIStates.last_project() or None, sessions)
    ui = CCGReviewUI.launch(cd, None)
    nav = ui.nav
    if len(nav.all_pairs_np) == 0:
        for ptr_key in cd.ptr:
            pairs = {p for p in cd.ptr[ptr_key].pair_set if p[0] != p[1]}
            if not pairs:
                continue
            nav.set_key(nav.key.change(session=ptr_key.session,
                                       conn_type=ptr_key.conn_type,
                                       excitability=ptr_key.excitability))
            nav.active_selections.unselected |= pairs
            if len(nav.all_pairs_np):
                break
    if nav.current_pair_inds is None:
        nav.set_current_pair(0)
    return app, ui


def context_values(ctx) -> dict:
    """Flatten one RenderContext into name -> array, for comparison."""
    if ctx is None:
        return {'__none__': np.array([1])}
    out = {}
    for name in ARRAY_FIELDS:
        value = getattr(ctx, name, None)
        out[name] = (np.asarray([], dtype=float) if value is None
                     else np.asarray(value, dtype=float))
    for name in SCALAR_FIELDS:
        value = getattr(ctx, name, None)
        out[name] = np.asarray([np.nan] if value is None else value, dtype=object)
    return out


def sweep(ui) -> dict:
    """Build contexts over a grid of display settings; returns case -> arrays."""
    nav, panel = ui.nav, ui.mainview
    cor, cs = panel.corr_section, panel.cs_section
    extend_row = cor._extend_rows[0]
    recorded = {}

    def record(case: str, ctx):
        for field, value in context_values(ctx).items():
            recorded[f"{case}::{field}"] = value

    for show_acg in (False, True):
        for btn in (cor.ref_btn, cor.tgt_btn):
            while btn.show != show_acg:
                btn.click()
        for show_baseline in (False, True):
            while cor.baseline_btn.show != show_baseline:
                cor.baseline_btn.click()
            for pvals in (False, True):
                cs.p_btn.setChecked(pvals)
                cs.pc_btn.setChecked(pvals)
                for deconv in (False, True):
                    cor.deconv_ref_btn.setChecked(deconv)
                    cor.deconv_tgt_btn.setChecked(deconv)
                    tag = f"acg{int(show_acg)}_bl{int(show_baseline)}" \
                          f"_p{int(pvals)}_dc{int(deconv)}"
                    record(f"stored_{tag}",
                           CCGContextBuilder.build_context(nav, panel))
                    extend_row.extend_check.setChecked(True)
                    record(f"extend_{tag}",
                           CCGContextBuilder.build_extend_context(
                               nav, panel, ext_view=extend_row))

    extend_row.extend_check.setChecked(True)
    for window_ms in ('5', '20', '50', '200'):
        for bin_ms in ('0.2', '1.0'):
            extend_row._ms_spin.setCurrentText(window_ms)
            extend_row._bin_spin.setCurrentText(bin_ms)
            record(f"grid_{window_ms}_{bin_ms}",
                   CCGContextBuilder.build_extend_context(
                       nav, panel, ext_view=extend_row))

    for method in ('conv', 'global', 'tailed', 'jitter'):
        nav.set_cs_params(method, nav.cs_metric)
        record(f"baseline_{method}",
               CCGContextBuilder.build_context(nav, panel))
    return recorded


def compare(recorded: dict, golden) -> list:
    """Return a list of human-readable differences; empty means identical."""
    problems = []
    missing = set(golden.files) - set(recorded)
    added = set(recorded) - set(golden.files)
    if missing:
        problems.append(f"{len(missing)} case(s) disappeared, e.g. {sorted(missing)[:3]}")
    if added:
        problems.append(f"{len(added)} new case(s), e.g. {sorted(added)[:3]}")
    for key in sorted(set(recorded) & set(golden.files)):
        want, got = golden[key], recorded[key]
        if want.shape != got.shape:
            problems.append(f"{key}: shape {want.shape} -> {got.shape}")
        elif want.dtype == object or got.dtype == object:
            if [str(x) for x in want.ravel()] != [str(x) for x in got.ravel()]:
                problems.append(f"{key}: {want.ravel()} -> {got.ravel()}")
        elif not np.allclose(want, got, rtol=1e-9, atol=1e-9, equal_nan=True):
            worst = np.nanmax(np.abs(want - got)) if want.size else float('nan')
            problems.append(f"{key}: values differ, max |delta| = {worst:.6g}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true',
                        help='write the golden file instead of checking it')
    args = parser.parse_args()

    _, ui = open_ui()
    recorded = sweep(ui)

    if args.record:
        np.savez_compressed(GOLDEN_PATH, **recorded)
        print(f"recorded {len(recorded)} arrays -> {GOLDEN_PATH}")
        return 0

    if not os.path.isfile(GOLDEN_PATH):
        print(f"no golden file at {GOLDEN_PATH}; run with --record first")
        return 2
    problems = compare(recorded, np.load(GOLDEN_PATH, allow_pickle=True))
    if problems:
        print(f"FAIL: {len(problems)} difference(s)")
        for line in problems[:40]:
            print(f"  {line}")
        return 1
    print(f"PASS: {len(recorded)} arrays identical")
    return 0


if __name__ == '__main__':
    sys.exit(main())
