"""Result 4: 1000x jitter over every session, conn-type and screened pair.

Segment 'all'. Saves after each (session, conn-type) so a kill loses one block,
not the run. Skips blocks already in the store.
"""
import sys, time
from neuropy.analyses.ms_connectivity import open_project
from neuropy.analyses.jitter_batch import run_and_store, pairs_for

NJITTER = 1000
SEGMENT = 'all'

def main():
    nd, cd, sd = open_project('test2')
    cd.load_jitter()
    keys = sorted(cd.ptr, key=str)
    t0 = time.perf_counter(); done = 0
    for i, k in enumerate(keys, 1):
        pairs = pairs_for(cd, sd, k, None)
        if not pairs:
            continue
        store = cd._jitter_results.get(k.nd(), {})
        if all((r, t, 'lo', None) in store for r, t in pairs):
            print(f"[{i}/{len(keys)}] {k} — already done, skipping", flush=True)
            continue
        print(f"[{i}/{len(keys)}] {k}: {len(pairs)} pairs x {NJITTER} …", flush=True)
        n = run_and_store(cd, sd, k, SEGMENT, njitter=NJITTER, save=True,
                          progress=lambda s: print(f"   {s}", flush=True))
        done += n
        print(f"   stored {n}; total {done}; elapsed {(time.perf_counter()-t0)/60:.1f} min",
              flush=True)
    print(f"DONE {done} pair-results in {(time.perf_counter()-t0)/3600:.2f} h", flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
