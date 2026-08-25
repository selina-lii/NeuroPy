"""Cluster a project's CCGs and draw each group, for labelling by group.

Writes one PNG per cluster (medoid first, then random members) plus a report of
how the existing hand tags fall across the clusters.

    python notebooks/cluster_ccgs.py [project] [n_clusters]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from neuropy.analyses.ms_connectivity import CCGConfig, CCGDataset
from neuropy.classifier.cluster import (cluster_pairs, cluster_report,
                                        label_agreement, rule_space)
from neuropy.classifier.dataset import build_multi
from neuropy.classifier.features import lag_axis, residual, smooth

N_SHOWN = 9


def draw_cluster(c, ls, lags, out_dir: str, rng) -> str:
    """Medoid plus a random sample of members, on one page."""
    pick = [c.medoid] + [i for i in rng.permutation(c.members)
                         if i != c.medoid][:N_SHOWN - 1]
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), sharex=True)
    for ax, i in zip(axes.ravel(), pick):
        s = ls.samples[i]
        res = smooth(residual(s.ccg[None], s.null[None]), 1.0)[0]
        ax.fill_between(lags, res, 0, color='#4a7', alpha=.7)
        ax.axvline(0, color='k', lw=.5)
        ax.set_title(f"{s.session[:14]} {s.ref}->{s.tgt}\n{','.join(s.labels) or '—'}",
                     fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(pick):]:
        ax.axis('off')
    top = f"{c.top_tag} {c.purity:.0%}" if c.tag_counts else "unlabeled"
    fig.suptitle(f"cluster {c.cid} — {len(c)} pairs, {c.n_labeled} labeled — {top}",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"cluster_{c.cid:03d}_n{len(c)}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(project: str = 'test2', n_clusters: int = 40):
    cd = CCGDataset(CCGConfig(name=project))
    ls = build_multi([cd], min_count=20, highres=True)
    a = ls.arrays(True)
    print(f"{len(ls.samples)} pairs, acg={ls.has_acg}, highres={ls.has_highres}")

    X = rule_space(a.ccg, a.null, cd.conf.duration, a.acg_ref, a.acg_tgt,
                   a.ccg_hi, a.null_hi)
    print(f"rule space: {X.shape[1]} features")
    clusters = cluster_pairs(X, n_clusters, [s.labels for s in ls.samples])
    print(cluster_report(clusters))

    print("\n=== how each label spreads across clusters")
    print(f"  {'label':<14} {'n':>5} {'clusters':>9} {'concentration':>14}")
    agree = label_agreement(clusters, ls.label_names)
    for name in sorted(agree, key=lambda n: -agree[n]['concentration']):
        v = agree[name]
        print(f"  {name:<14} {v['total']:>5} {v['n_clusters']:>9} "
              f"{v['concentration']:>14.2f}")

    out_dir = os.path.join(cd.save_path, 'clusters')
    os.makedirs(out_dir, exist_ok=True)
    lags = lag_axis(a.ccg.shape[1], cd.conf.duration)
    rng = np.random.default_rng(0)
    for c in clusters:
        draw_cluster(c, ls, lags, out_dir, rng)

    doc = {'project': project, 'n_clusters': len(clusters),
           'label_agreement': {k: {kk: vv for kk, vv in v.items()
                                   if kk != 'top_clusters'}
                               for k, v in agree.items()},
           'clusters': [{'cid': c.cid, 'size': len(c), 'n_labeled': c.n_labeled,
                         'purity': round(c.purity, 3), 'top_tag': c.top_tag,
                         'tag_counts': c.tag_counts,
                         'members': [[ls.samples[i].session, ls.samples[i].ref,
                                      ls.samples[i].tgt] for i in c.members]}
                        for c in clusters]}
    path = os.path.join(out_dir, 'clusters.json')
    with open(path, 'w') as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n{len(clusters)} cluster figures + {path}")
    print(f"look in {out_dir}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test2',
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)
