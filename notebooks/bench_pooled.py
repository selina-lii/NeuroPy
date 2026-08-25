"""Pooled cross-validation: all sessions together, split at random.

Leave-one-rat-out asks "does this transfer to a new animal". This asks the prior
question — "is the criterion learnable at all" — by ignoring which session and
which neurons a pair came from. The labels describe the CCG alone, so the split
is over pairs alone.

    python notebooks/bench_pooled.py [project] [model] [bias]
"""
from __future__ import annotations

import sys

import numpy as np

from neuropy.analyses.ms_connectivity import CCGConfig, CCGDataset
from neuropy.classifier.dataset import build_multi
from neuropy.classifier.train import cross_validate, report


WELL_SUPPORTED = 200      # positives below this are reported but not chased


def main(project: str = 'test2', model: str = 'kernel', bias: str = 'discover'):
    cd = CCGDataset(CCGConfig(name=project))
    ls = build_multi([cd], min_count=20, highres=True)
    duration = cd.conf.duration
    print(f"{len(ls.samples)} pairs, {len(ls.label_names)} labels, "
          f"model={model}, bias={bias}\n")

    print("--- pooled, random 5-fold")
    out = cross_validate(ls, model, duration=duration, bias=bias)
    print(report(out['scores']))

    # The headline number: a label with 26 examples cannot be fixed by modelling,
    # so averaging it in only hides movement on the labels that can.
    solid = {n: s for n, s in out['scores'].items() if s['n'] >= WELL_SUPPORTED}
    print(f"\n--- labels with n >= {WELL_SUPPORTED} ({len(solid)} of "
          f"{len(out['scores'])})")
    print(f"  mean F1 {np.mean([s['f1'] for s in solid.values()]):.3f}   "
          f"mean AUC {np.mean([s['auc'] for s in solid.values()]):.3f}")


if __name__ == '__main__':
    main(*(sys.argv[1:4] or ['test2']))
