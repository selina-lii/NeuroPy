"""Propose bulk labels: unlabeled pairs sitting inside a high-purity cluster.

The fastest way to grow a label with real support is not to review pairs one at
a time — it is to confirm a whole cluster at once. A cluster that is 83% rhythm
among its labeled members is a strong claim about its unlabeled ones too, and
confirming it costs one decision instead of forty.

Writes a worklist of (cluster, proposed label, unlabeled members) ordered by how
much each would add to a label that already has enough data to train on.

    python notebooks/cluster_suggest.py [project] [n_clusters] [min_purity]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from neuropy.analyses.ms_connectivity import CCGConfig, CCGDataset
from neuropy.classifier.cluster import cluster_pairs, rule_space
from neuropy.classifier.dataset import build_multi

WELL_SUPPORTED = 200      # a label already worth training on


def suggestions(clusters, ls, min_purity: float, min_labeled: int = 10,
                min_shape_votes: int = 10):
    """Clusters pure enough to label in bulk, richest opportunity first."""
    out = []
    for c in clusters:
        if c.n_labeled < min_labeled:
            continue
        pur = c.purity
        if pur < min_purity or not c.tag_counts:
            continue
        top = c.top_tag
        # One dominant tag among a handful of labeled members is thin evidence
        # for the rest of the cluster, however pure the fraction looks.
        if c.tag_counts[top] < min_shape_votes:
            continue
        blank = [i for i in c.members if not ls.samples[i].labels]
        missing = [i for i in c.members
                   if ls.samples[i].labels and top not in ls.samples[i].labels]
        out.append({'cid': c.cid, 'size': len(c), 'label': top,
                    'purity': round(pur, 3), 'n_labeled': c.n_labeled,
                    'unlabeled': blank, 'labeled_without': missing,
                    'gain': len(blank) + len(missing)})
    return sorted(out, key=lambda s: -s['gain'])


def main(project: str = 'test2', n_clusters: int = 80, min_purity: float = 0.5):
    cd = CCGDataset(CCGConfig(name=project))
    ls = build_multi([cd], min_count=20, highres=True)
    a = ls.arrays(True)
    counts = {n: int(ls.Y[:, j].sum()) for j, n in enumerate(ls.label_names)}

    X = rule_space(a.ccg, a.null, cd.conf.duration, a.acg_ref, a.acg_tgt,
                   a.ccg_hi, a.null_hi)
    clusters = cluster_pairs(X, n_clusters, [s.labels for s in ls.samples])
    sugg = suggestions(clusters, ls, min_purity)

    print(f"{len(sugg)} clusters at purity >= {min_purity}\n")
    print(f"  {'cid':>4} {'size':>5} {'pure':>6} {'label':<14} {'has':>5} "
          f"{'would add':>10}")
    print("  " + "-" * 52)
    total = {}
    for s in sugg:
        print(f"  {s['cid']:>4} {s['size']:>5} {s['purity']:>6.2f} "
              f"{s['label']:<14} {counts.get(s['label'], 0):>5} {s['gain']:>10}")
        total[s['label']] = total.get(s['label'], 0) + s['gain']

    print(f"\n  {'label':<14} {'now':>6} {'+bulk':>7} {'would be':>9}")
    print("  " + "-" * 40)
    for name in sorted(total, key=lambda n: -(counts.get(n, 0) + total[n])):
        now = counts.get(name, 0)
        mark = '' if now + total[name] >= WELL_SUPPORTED else '   (still thin)'
        print(f"  {name:<14} {now:>6} {total[name]:>7} "
              f"{now + total[name]:>9}{mark}")

    out_dir = os.path.join(cd.save_path, 'clusters')
    os.makedirs(out_dir, exist_ok=True)
    doc = [{**s,
            'unlabeled': [[ls.samples[i].session, ls.samples[i].ref,
                           ls.samples[i].tgt] for i in s['unlabeled']],
            'labeled_without': [[ls.samples[i].session, ls.samples[i].ref,
                                 ls.samples[i].tgt] for i in s['labeled_without']]}
           for s in sugg]
    path = os.path.join(out_dir, 'suggestions.json')
    with open(path, 'w') as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nwrote {path}")
    print(f"cluster figures: {out_dir}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test2',
         int(sys.argv[2]) if len(sys.argv) > 2 else 80,
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
