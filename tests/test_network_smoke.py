"""Smoke test: the probe network panel must assemble and draw the same thing.

Pins _assemble_data's output and the resulting scene item count across a sweep
of the panel's display toggles.

    python tests/test_network_smoke.py --record
    python tests/test_network_smoke.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ccg_context_golden import open_ui   # sets the Qt offscreen env

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "network_smoke.json")


def data_summary(panel, data) -> dict:
    """Content fingerprint of assembled data plus what the redraw put on the scene."""
    if data is None:
        return {'data': None, 'items': len(panel._pg_items)}
    return {
        'n_neurons': int(data.n_neurons),
        'pos_sum': round(float(np.nansum(data.pos)), 6),
        'peak_channels_sum': int(np.nansum(data.peak_channels)),
        'n_pair_entries': len(data.pair_entries),
        'n_entries_total': sum(len(v) for v in data.pair_entries.values()),
        'session_label': str(data.session_label),
        'items': len(panel._pg_items),
        'nav_edges': len(panel._nav_edges),
    }


def sweep(ui) -> dict:
    """Assemble and redraw under each toggle combination; returns case -> summary."""
    panel = ui.neuron_network
    nd_key = ui.nav.key.nd()
    type_key = ui.nav.key
    recorded = {}
    for hide_channel in (False, True):
        panel._net_hide_same_channel = hide_channel
        for hide_shank in (False, True):
            panel._net_hide_same_shank = hide_shank
            panel._data_cache.clear()
            data = panel._assemble_data(nd_key, type_key, None, False, None)
            panel._draw_impl()
            tag = f"ch{int(hide_channel)}_sh{int(hide_shank)}"
            recorded[tag] = data_summary(panel, data)
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
    problems = [f"{case}: {golden.get(case)} -> {summary}"
                for case, summary in sorted(recorded.items())
                if golden.get(case) != summary]
    if problems:
        print(f"FAIL: {len(problems)} case(s) differ")
        for line in problems[:20]:
            print(f"  {line}")
        return 1
    print(f"PASS: {len(recorded)} cases identical")
    return 0


if __name__ == '__main__':
    sys.exit(main())
