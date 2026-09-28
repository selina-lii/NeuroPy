"""Result 1b: run post10 / post10sleep-only with the binarized on/off label.

Loads each saved stats config, swaps the metric to 'CS binarized (on/off)',
runs it through the backend (no GUI) and prints the report.
"""

import json
import pathlib
import sys
import time

from neuropy.analyses.ccg_transforms import AuxTestConfig, ConnectionStrength
from neuropy.analyses.ms_connectivity import open_project
from neuropy.ui.stats_tests_backend import StatsTestBackend

CONFIGS = ('post10', 'post10sleep only')
METRIC = 'CS binarized (on/off)'
ALL_TESTS = tuple(ConnectionStrength.AUX_TESTS)
OUT = 'design_docs/RESULT1_binarized_stats.txt'


def run_one(cd, sd, name: str, metric: str) -> str:
    path = pathlib.Path(cd.stats_results_dir) / f'{name}.json'
    backend = StatsTestBackend(cd, sd)
    bundle = json.loads(path.read_text())
    backend.from_dict(bundle)
    if bundle.get('display'):
        backend.display.__setstate__(bundle['display'])
    for row in backend.rows:
        row.data_type = metric
    if (err := backend.validate()):
        return f"[{name}] validate failed: {err}"
    t = time.perf_counter()
    results = backend.run()
    head = f"[{name}] metric={metric!r}  {time.perf_counter() - t:.1f}s"
    return head + "\n" + backend.section_stats(results)


def main():
    nd, cd, sd = open_project('test2')
    # the on/off label must come from all five rules, as the sweep wrote them
    cd.aux_test_config = AuxTestConfig(ALL_TESTS)
    sd.ensure_groups_loaded_for([str(k.session) for k in cd.nd.session_keys])
    print(f"aux tests: {cd.aux_test_config.enabled}", flush=True)
    out = []
    for name in CONFIGS:
        for metric in (METRIC, 'Conn Strength'):
            block = '=' * 70 + '\n' + run_one(cd, sd, name, metric)
            print(block, flush=True)
            out.append(block)
    pathlib.Path(OUT).write_text('\n'.join(out))
    print(f'\nwrote {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
