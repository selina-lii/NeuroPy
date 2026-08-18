#!/usr/bin/env python
"""Train and compare CCG classifiers without the GUI.

    python notebooks/run_classifier.py                  # default model, writes figures
    python notebooks/run_classifier.py conv kernel      # compare models
    python notebooks/run_classifier.py --project 001695 # another project
"""
from __future__ import annotations
import argparse
import numpy as np

from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig
from neuropy.classifier import build_labeled_set, leave_one_rat_out, report
from neuropy.classifier.models import MODELS
from neuropy.classifier.run import DEFAULT_MODEL, train_project

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='*', default=[], help=f'default: {DEFAULT_MODEL}')
    ap.add_argument('--project', default='test2')
    ap.add_argument('--head', choices=['mlp', 'gb'], help='override the model default')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    cd = CCGDataset(CCGConfig(name=args.project))
    names = args.models or [DEFAULT_MODEL]

    if len(names) == 1 and not args.head:
        # Single model: full pipeline — trains, scores, saves model + figures.
        r = train_project(cd, names[0], figures=not args.no_figures)
        s = r['summary']
        print(f"\n{s['model']}: {s['n_samples']} pairs / {s['n_rats']} animals")
        print(report(s['scores']))
        print(f"\nwrote {len(r['figures'])} figures to {r['out_dir']}")
    else:
        # Comparison: cross-validate each, no model/figure output.
        ls = build_labeled_set(cd, cd.selections_dir)
        kw = {'head': args.head} if args.head else {}
        for name in names:
            if name not in MODELS:
                raise SystemExit(f"unknown model {name!r}; choose from {list(MODELS)}")
            cv = leave_one_rat_out(ls, name, duration=cd.conf.duration, **kw)
            f1 = np.mean([v['f1'] for v in cv['scores'].values()])
            auc = np.mean([v['auc'] for v in cv['scores'].values()])
            print(f"\n=== {name}{' +' + args.head if args.head else ''}: "
                  f"F1 {f1:.3f}  AUC {auc:.3f}")
            print(report(cv['scores']))
