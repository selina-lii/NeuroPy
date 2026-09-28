"""Run every UI characterization suite against one headless launch.

Each suite pays the same multi-minute `open_ui()`; sharing it turns five
launches into one.

    python tests/run_suites.py                   # verify all
    python tests/run_suites.py neuron_view       # verify some
    python tests/run_suites.py --record neuron_view
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite_harness as harness
from test_ccg_context_golden import open_ui   # sets the Qt offscreen env
from neuropy.analyses.view_spec import PAIR_VIEW
from neuropy.ui.app_state import DisplayConfig
import test_ccg_context_golden as context_golden
import test_ccg_render_smoke as render_smoke
import test_dialogs_smoke as dialogs_smoke
import test_network_smoke as network_smoke
import test_neuron_view_smoke as neuron_view

SUITES = {
    'context_golden': (context_golden, context_golden.check_npz, harness.write_npz),
    'render_smoke':   (render_smoke,   harness.check_json,       harness.write_json),
    'dialogs_smoke':  (dialogs_smoke,  harness.check_json,       harness.write_json),
    'network_smoke':  (network_smoke,  harness.check_json,       harness.write_json),
    'neuron_view':    (neuron_view,    harness.check_json,       harness.write_json),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', metavar='SUITE',
                        help=f"suites to run (default: all of {', '.join(SUITES)})")
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    names = args.names or list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}")

    _, ui = open_ui()
    failed = []
    for name in names:
        module, compare, write = SUITES[name]
        harness.reset_ui(ui, PAIR_VIEW, DisplayConfig())
        recorded = module.sweep(ui)
        if args.record:
            write(module.GOLDEN_PATH, recorded)
            print(f"{name:16s} recorded {len(recorded)} case(s)")
            continue
        print(f"{name:16s} ", end='')
        if harness.report(compare(module.GOLDEN_PATH, recorded), len(recorded)):
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
