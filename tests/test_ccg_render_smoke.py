"""Smoke test: rendering must draw the same item counts it drew before.

The golden test pins the data feeding the plot; this one pins the drawing.
Records how many items each overlay puts on the scene across a toggle sweep.

    python tests/test_ccg_render_smoke.py --record
    python tests/test_ccg_render_smoke.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ccg_context_golden import open_ui   # sets the Qt offscreen env

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    _, ui = open_ui()
    recorded = sweep(ui)

    if args.record:
        with open(GOLDEN_PATH, 'w') as fh:
            json.dump(recorded, fh, indent=1, sort_keys=True)
        print(f"recorded {len(recorded)} cases -> {GOLDEN_PATH}")
        return 0

    if not os.path.isfile(GOLDEN_PATH):
        print(f"no golden file at {GOLDEN_PATH}; run with --record first")
        return 2
    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)
    problems = [f"{case}: {golden.get(case)} -> {counts}"
                for case, counts in sorted(recorded.items())
                if golden.get(case) != counts]
    if problems:
        print(f"FAIL: {len(problems)} case(s) differ")
        for line in problems[:20]:
            print(f"  {line}")
        return 1
    print(f"PASS: {len(recorded)} cases identical")
    return 0


if __name__ == '__main__':
    sys.exit(main())
