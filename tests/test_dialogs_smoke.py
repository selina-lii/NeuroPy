"""Smoke test: dialogs must build with the same widget structure.

Constructs each dialog against the live UI (never exec's one) and records the
widget-class census of its tree. Catches a row that stopped being added or a
control that changed type.

    python tests/test_dialogs_smoke.py --record
    python tests/test_dialogs_smoke.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ccg_context_golden import open_ui   # sets the Qt offscreen env

from neuropy.analyses.view_spec import NEURON_VIEW, PAIR_VIEW
from neuropy.ui.dialogs import (ManageGroupsDialog, PairTagsDialog,
                                ExportOptionsDialog)

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dialogs_smoke.json")


def widget_census(widget) -> dict:
    """Count of every widget class in this widget's tree."""
    counts = Counter()
    stack = [widget]
    while stack:
        node = stack.pop()
        counts[type(node).__name__] += 1
        stack.extend(node.children())
    return dict(sorted(counts.items()))


def sweep(ui) -> dict:
    """Build each dialog; returns case -> widget census."""
    panel = ui.pairs_view.pair_selection
    recorded = {}

    dialog = ManageGroupsDialog(ui.nav.sel_data, panel, parent=None)
    recorded['manage_groups'] = widget_census(dialog)
    dialog.close()

    ui.set_view(NEURON_VIEW)
    neuron_dialog = ManageGroupsDialog(ui.nav.sel_data, panel, parent=None)
    recorded['manage_groups_neuron'] = widget_census(neuron_dialog)
    neuron_dialog.close()
    ui.set_view(PAIR_VIEW)

    tags = PairTagsDialog(0, 1, {'note': 'x'}, parent=None)
    recorded['pair_tags'] = widget_census(tags)
    tags.close()

    export = ExportOptionsDialog(
        ui.nav, ui.nav.cd, ui.nav.sel_data, panel,
        ui.ui_states.panel_state, fmt='png',
        preview_pair=ui.nav.current_pair_inds, parent=None)
    recorded['export_options'] = widget_census(export)
    recorded['export_collect'] = {k: str(v) for k, v in export._collect().items()}
    export.close()
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
    problems = [f"{case}: {golden.get(case)} -> {value}"
                for case, value in sorted(recorded.items())
                if golden.get(case) != value]
    if problems:
        print(f"FAIL: {len(problems)} case(s) differ")
        for line in problems[:20]:
            print(f"  {line}")
        return 1
    print(f"PASS: {len(recorded)} cases identical")
    return 0


if __name__ == '__main__':
    sys.exit(main())
