#!/usr/bin/env python
"""Train and compare CCG classifiers without the GUI.

    python notebooks/run_classifier.py                  # default model, writes figures
    python notebooks/run_classifier.py conv kernel      # compare models
    python notebooks/run_classifier.py --project 001695 # another project (dir dandi001695)
"""
from __future__ import annotations
import argparse
import numpy as np

from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig
from neuropy.classifier import build_labeled_set, leave_one_rat_out, report
from neuropy.classifier.models import BIAS_BETA, MODELS
from neuropy.classifier.run import (DEFAULT_MODEL, apply_model, list_models,
                                    train_project)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='*', default=[], help=f'default: {DEFAULT_MODEL}')
    ap.add_argument('--project', default='test2')
    ap.add_argument('--head', choices=['mlp', 'gb'], help='override the model default')
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--bias", choices=list(BIAS_BETA), default='balanced',
                    help='discover favours recall, accurate favours precision')
    ap.add_argument('--apply', metavar='NAME', help='score with a saved classifier')
    ap.add_argument('--list', action='store_true', help='show saved classifiers')
    args = ap.parse_args()

    cd = CCGDataset(CCGConfig(name=args.project))

    if args.list:
        for m in list_models(cd):
            t = m.get('trained_on', {})
            print(f"{m['name']:<24} {m['model_key']}+{m['head']}  {m['saved_at'][:19]}")
            print(f"    trained on {t.get('project')}: {t.get('n_samples')} pairs / "
                  f"{len(t.get('sessions', []))} sessions / {len(t.get('rats', []))} rats")
            print(f"    F1 {t.get('mean_f1', 0):.3f}  AUC {t.get('mean_auc', 0):.3f}  "
                  f"bins lo={t.get('n_bins_lowres')} hi={t.get('n_bins_highres')}")
        raise SystemExit

    if args.apply:
        out = apply_model(cd, args.apply)
        t = out['meta'].get('trained_on', {})
        print(f"scored {len(out['rows'])} pairs in {args.project} "
              f"with '{args.apply}' (trained on {t.get('project')}, "
              f"{t.get('n_samples')} pairs)")
        if out['skipped']:
            print(f"skipped {len(out['skipped'])} session(s) lacking required CCGs: "
                  f"{', '.join(out['skipped'])}")
        counts: dict[str, int] = {}
        for row in out['rows']:
            for lab in row['labels']:
                counts[lab] = counts.get(lab, 0) + 1
        for lab, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {lab:<12} {n}")
        raise SystemExit

    names = args.models or [DEFAULT_MODEL]

    if len(names) == 1 and not args.head:
        # Single model: full pipeline — trains, scores, saves model + figures.
        r = train_project(cd, names[0], figures=not args.no_figures,
                          min_count=args.min_count, bias=args.bias)
        s = r['summary']
        print(f"\n{s['model']}: {s['n_samples']} pairs / {s['n_rats']} animals")
        print(report(s['scores']))
        print(f"\nwrote {len(r['figures'])} figures to {r['out_dir']}")
    else:
        # Comparison: cross-validate each, no model/figure output.
        ls = build_labeled_set(cd, min_count=args.min_count)
        kw = {'head': args.head} if args.head else {}
        kw['bias'] = args.bias
        for name in names:
            if name not in MODELS:
                raise SystemExit(f"unknown model {name!r}; choose from {list(MODELS)}")
            cv = leave_one_rat_out(ls, name, duration=cd.conf.duration, **kw)
            f1 = np.mean([v['f1'] for v in cv['scores'].values()])
            auc = np.mean([v['auc'] for v in cv['scores'].values()])
            print(f"\n=== {name}{' +' + args.head if args.head else ''}: "
                  f"F1 {f1:.3f}  AUC {auc:.3f}")
            print(report(cv['scores']))
