"""Smoke test: neuron view must list the same neurons and draw the same layers.

Switches views against the live UI and records list length, row text, plot item
counts per control state, and that pair view is untouched by the round trip.

    python tests/test_neuron_view_smoke.py --record
    python tests/test_neuron_view_smoke.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ccg_context_golden import open_ui   # sets the Qt offscreen env

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "neuron_view_smoke.json")


def plot_items(panel) -> int:
    return len(panel.plot_widget.getPlotItem().items)


def sweep(ui) -> dict:
    """Drive the view switch and every control; returns case -> signature."""
    nav = ui.nav
    recorded = {}

    nav.set_current_pair(3)
    pair_before = nav.current_pair
    ui.set_view('neuron')
    view = ui.neuron_view
    recorded['list_rows'] = view.list_panel.list.count()
    recorded['row_0'] = view.list_panel.list.item(0).text()
    recorded['row_last'] = view.list_panel.list.item(
        view.list_panel.list.count() - 1).text()

    plot = view.plot_panel
    recorded['default_items'] = plot_items(plot)
    plot.bin_spin.setValue(1.0)
    recorded['bin_1s_items'] = plot_items(plot)
    plot.rate_btn._cycle(); plot.render()
    recorded['rate_line_items'] = plot_items(plot)
    plot.rate_btn._cycle(); plot.render()
    recorded['rate_hidden_items'] = plot_items(plot)
    plot.rate_btn._cycle(); plot.render()
    recorded['rate_solid_items'] = plot_items(plot)

    view.list_panel.list.setCurrentRow(5)
    recorded['selected_neuron'] = nav.current_item
    recorded['selected_neurons'] = nav.view.neurons_of(nav.current_item)

    plot.wf_btn.setChecked(True)
    recorded['waveform_axes'] = len(plot.wf_panel._fig.axes)
    plot.wf_btn.setChecked(False)

    groups = ui.neuron_groups
    if '_smoke' in groups.defined_groups:      # a previous run may have saved it
        groups.delete_group('_smoke')
    groups.create_group('_smoke')
    view.list_panel.list.setCurrentRow(3)
    view.list_panel.toggle_group('_smoke')
    recorded['tagged'] = len(groups.members_in_group('_smoke', nav.current_session_str))
    view.list_panel.undo()
    recorded['after_undo'] = len(groups.members_in_group('_smoke', nav.current_session_str))
    groups.delete_group('_smoke')
    groups.save()

    nav.set_session_any_mode(True)
    recorded['all_session_items'] = len(nav.view.items(nav.key))
    nav.set_session_any_mode(False)
    recorded['single_session_items'] = len(nav.view.items(nav.key))

    ui.set_view('pair')
    recorded['pair_restored'] = nav.current_pair == pair_before
    recorded['center_widget'] = type(ui.center_stack.currentWidget()).__name__
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
