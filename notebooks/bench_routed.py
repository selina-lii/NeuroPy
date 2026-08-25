"""Is per-label routing worth it? Every strategy pooled, then routed against them.

`fit_routed` picks each label's strategy from leave-one-rat-out scores, but that
split costs ~0.09 mean F1 to per-animal label-rate imbalance — so the routing
decision is made on the weaker evaluation. This scores every strategy on the
pooled split, routes from those numbers instead, and reports what routing gains
over simply using the best single strategy everywhere.

    python notebooks/bench_routed.py [project] [bias]
"""
from __future__ import annotations

import sys

import numpy as np

from bench_pooled import WELL_SUPPORTED
from neuropy.analyses.ms_connectivity import CCGConfig, CCGDataset
from neuropy.classifier.dataset import build_multi
from neuropy.classifier.models import BIAS_BETA, fbeta_from_pr
from neuropy.classifier.train import cross_validate, route_by_score

STRATEGIES = ['rule', 'kernel', 'conv', 'dualres', 'shape']


def mean_over(scores: dict, key: str, floor: int) -> float:
    vals = [s[key] for s in scores.values() if s['n'] >= floor]
    return float(np.mean(vals)) if vals else 0.0


def main(project: str = 'test2', bias: str = 'discover'):
    cd = CCGDataset(CCGConfig(name=project))
    ls = build_multi([cd], min_count=20, highres=True)
    duration = cd.conf.duration
    print(f"{len(ls.samples)} pairs, {len(ls.label_names)} labels, bias={bias}\n")

    cv = {}
    for name in STRATEGIES:
        print(f"--- {name}", flush=True)
        cv[name] = cross_validate(ls, name, duration=duration, bias=bias)
        s = cv[name]['scores']
        print(f"  mean F1 {mean_over(s, 'f1', WELL_SUPPORTED):.3f}   "
              f"AUC {mean_over(s, 'auc', WELL_SUPPORTED):.3f}\n", flush=True)

    print(f"\n{'strategy':<10}{'F1 (n>=200)':>13}{'AUC':>8}{'F1 (all)':>10}")
    print("-" * 41)
    for name, res in sorted(cv.items(),
                            key=lambda kv: -mean_over(kv[1]['scores'], 'f1',
                                                      WELL_SUPPORTED)):
        s = res['scores']
        print(f"{name:<10}{mean_over(s, 'f1', WELL_SUPPORTED):>13.3f}"
              f"{mean_over(s, 'auc', WELL_SUPPORTED):>8.3f}"
              f"{mean_over(s, 'f1', 0):>10.3f}")

    # Routing takes each label's column from its best strategy. Scoring the
    # routes off these same pooled folds is optimistic, so it is an upper bound
    # on what routing could give — if that bound is small, routing is not worth
    # the seven-fold training cost.
    beta = BIAS_BETA.get(bias, 1.0)
    routes = route_by_score(cv, ls.label_names, beta)
    best_single = max(cv, key=lambda k: mean_over(cv[k]['scores'], 'f1',
                                                  WELL_SUPPORTED))

    print(f"\n--- routed (upper bound) vs best single ({best_single})")
    print(f"  {'label':<12}{'n':>5}{'route':>9}{'routed':>8}{'single':>8}{'d':>7}")
    print("  " + "-" * 47)
    gains, solid = [], []
    for name in ls.label_names:
        r = cv[routes[name]]['scores'][name]
        b = cv[best_single]['scores'][name]
        d = fbeta_from_pr(r['precision'], r['recall'], 1.0) - b['f1']
        gains.append(d)
        if b['n'] >= WELL_SUPPORTED:
            solid.append((r['f1'], b['f1']))
        print(f"  {name:<12}{b['n']:>5}{routes[name]:>9}{r['f1']:>8.2f}"
              f"{b['f1']:>8.2f}{d:>+7.2f}")

    if solid:
        rf = float(np.mean([a for a, _ in solid]))
        bf = float(np.mean([b for _, b in solid]))
        print(f"\n  n>=200: routed {rf:.3f}  vs  {best_single} alone {bf:.3f}"
              f"   ({rf - bf:+.3f})")
    n_used = len(set(routes.values()))
    print(f"  routing uses {n_used} of {len(STRATEGIES)} strategies")


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['test2']))
