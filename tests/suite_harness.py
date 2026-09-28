"""Record/compare plumbing every UI characterization suite shares."""
from __future__ import annotations
import argparse
import json
import os

import numpy as np


def write_json(path: str, recorded: dict) -> None:
    with open(path, 'w') as fh:
        json.dump(recorded, fh, indent=1, sort_keys=True)


def check_json(path: str, recorded: dict) -> list:
    """Differences against the stored JSON; empty means identical."""
    with open(path) as fh:
        golden = json.load(fh)
    return [f"{case}: {golden.get(case)} -> {value}"
            for case, value in sorted(recorded.items())
            if golden.get(case) != value]


def write_npz(path: str, recorded: dict) -> None:
    np.savez_compressed(path, **recorded)


def reset_ui(ui, pair_view: str, fresh_display) -> None:
    """Undo what a sweep leaves behind, so suites sharing one launch stay independent."""
    for row in ui.mainview.corr_section._extend_rows:
        row.extend_check.setChecked(False)
    ui.nav.apply_display_config(fresh_display)
    ui.set_view(pair_view)
    ui.nav.set_current_pair(0)
    ui.mainview.request_render()


def run(open_ui, sweep, path: str, compare, write, argv=None) -> int:
    """One suite's entry point: sweep the UI, then record or verify."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true',
                        help='write the golden file instead of checking it')
    args = parser.parse_args(argv)

    _, ui = open_ui()
    recorded = sweep(ui)

    if args.record:
        write(path, recorded)
        print(f"recorded {len(recorded)} case(s) -> {path}")
        return 0
    if not os.path.isfile(path):
        print(f"no golden file at {path}; run with --record first")
        return 2
    return report(compare(path, recorded), len(recorded))


def report(problems: list, n_cases: int) -> int:
    if problems:
        print(f"FAIL: {len(problems)} case(s) differ")
        for line in problems[:40]:
            print(f"  {line}")
        return 1
    print(f"PASS: {n_cases} cases identical")
    return 0
