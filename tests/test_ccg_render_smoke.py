"""Smoke test: rendering must draw the same item counts it drew before.

The golden test pins the data feeding the plot; this one pins the drawing.
Records how many items each overlay puts on the scene across a toggle sweep.

    python tests/test_ccg_render_smoke.py --record
    python tests/test_ccg_render_smoke.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ccg_context_golden import open_ui   # sets the Qt offscreen env
import suite_harness as harness

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ccg_render_smoke.json")


def item_counts(plot_widget) -> list:
    """Items on each subplot, its ACG view boxes, p-value box, and waveform overlay."""
    counts = []
    for sub in plot_widget._subplots:
        acg = sum(len(vb.addedItems) for vb, _ in sub.acg_axes)
        counts.append([len(sub.plot.items), acg, len(sub.pval_items),
                       len(sub.wf_vb.addedItems)])
    return counts


def sweep(ui) -> dict:
    """Render under each toggle combination; returns case -> item counts."""
    panel = ui.mainview
    cor, cs = panel.corr_section, panel.cs_section
    recorded = {}
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
                for test_window in (False, True):
                    cs.test_window_btn.setChecked(test_window)
                    for ref_wf in (False, True):
                        cor.ref_wf_btn.setChecked(ref_wf)
                        panel.request_render()
                        tag = (f"acg{int(show_acg)}_bl{int(show_baseline)}"
                               f"_p{int(pvals)}_tw{int(test_window)}"
                               f"_wf{int(ref_wf)}")
                        recorded[tag] = item_counts(panel.plot_widget)
                    cor.ref_wf_btn.setChecked(False)
    return recorded


def main() -> int:
    return harness.run(open_ui, sweep, GOLDEN_PATH, harness.check_json, harness.write_json)


if __name__ == '__main__':
    sys.exit(main())
