"""Rewrite stored pval/qval arrays as float32, halving what a session faults in.

p-values are only ever compared against alpha, so float32's 7 digits are ample; the
saving is real because CCGData memmaps these files and pages in what it touches.

    python notebooks/fix/pval_float32.py <data_root>            # report only
    python notebooks/fix/pval_float32.py <data_root> --apply    # convert in place
"""
import argparse
import os
from pathlib import Path

import numpy as np

NAMES = ('pval', 'qval')
ALPHAS = (0.05, 0.01, 0.001, 1e-4)


def convert(path: Path) -> tuple[int, int]:
    """Rewrite one .npy as float32 via a temp file; returns (bytes_before, bytes_after)."""
    src = np.load(path, mmap_mode='r')
    before = src.nbytes
    tmp = path.with_suffix('.npy.tmp')
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=src.shape)
    # chunk over dim0-major slabs: never materializes the whole array
    step = max(1, src.shape[0])
    for i in range(0, src.shape[0], step):
        chunk = np.asarray(src[i:i + step])
        for a in ALPHAS:                       # verdicts must survive the narrowing
            assert np.array_equal(chunk <= a, chunk.astype(np.float32) <= a), path
        out[i:i + step] = chunk
    out.flush()
    del out, src
    os.replace(tmp, path)
    return before, before // 2


def main(root: str, apply: bool):
    paths = [p for p in sorted(Path(root).glob('**/*.npy')) if p.stem in NAMES]
    todo = [p for p in paths if np.load(p, mmap_mode='r').dtype == np.float64]
    saved = 0
    for p in todo:
        if apply:
            before, after = convert(p)
            saved += before - after
        else:
            saved += np.load(p, mmap_mode='r').nbytes // 2
    verb = 'reclaimed' if apply else 'reclaimable'
    print(f'{len(todo)}/{len(paths)} files float64; {saved / 1e9:.2f} GB {verb}')
    if not apply:
        print('re-run with --apply to write')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='data root holding project_*/ dirs')
    ap.add_argument('--apply', action='store_true', help='write (default: report only)')
    a = ap.parse_args()
    main(a.root, a.apply)
