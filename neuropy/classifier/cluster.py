"""Group CCGs that look alike, so labels can be assigned per group.

Tagging pair by pair is slow and drifts: the same shape gets tagged heavily in
one session and lightly in another, and the classifier then learns the drift as
if it were signal. Labelling a *cluster* is one decision applied uniformly to
every member, and it covers pairs nobody ever reviewed.

Similarity is measured in the rule-feature space — peak counts, spacings, named
millisecond windows, ACG quality — rather than over raw traces. This matters: a
clustering is only as good as its notion of "alike", and raw-trace distance is
dominated by amplitude, while the hand-written rules turn on peak *structure*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from neuropy.classifier.features import (acg_features, peak_features,
                                         shape_features, window_features)

# Smallest group worth a reviewing decision, and so also the smallest scope that
# can be clustered at all.
MIN_CLUSTER = 8


@dataclass
class Cluster:
    """One group of similar CCGs, with the evidence needed to name it."""
    cid: int
    members: np.ndarray              # indices into the source arrays
    medoid: int                      # index of the most central member
    tag_counts: dict[str, int] = field(default_factory=dict)
    n_labeled: int = 0

    def __len__(self) -> int:
        return len(self.members)

    @property
    def purity(self) -> float:
        """Share of the labeled members carrying this cluster's commonest tag.

        High purity says the space agrees with the existing hand labels; low
        purity says the cluster mixes shapes the reviewer considers distinct.
        """
        if not self.n_labeled or not self.tag_counts:
            return 0.0
        return max(self.tag_counts.values()) / self.n_labeled

    @property
    def top_tag(self) -> str:
        return max(self.tag_counts, key=self.tag_counts.get) if self.tag_counts else ''


@dataclass
class Clustering:
    """The clusters plus the picture they came from.

    The plot positions are the first two axes of the embedding the clustering
    itself ran in, so a viewer reading distance off the scatter is reading the
    distance Ward used, not a second projection that merely resembles it.
    """
    clusters: list[Cluster]
    coords: np.ndarray               # [n, 2] plot positions, source-array order
    assign: np.ndarray               # [n] cluster id per row, -1 once dropped
    n_asked: int = 0                 # requested count, above n_used when capped
    n_used: int = 0                  # count the scope could actually support

    def __iter__(self):
        return iter(self.clusters)

    def __len__(self) -> int:
        return len(self.clusters)

    def by_id(self, cid: int) -> Cluster | None:
        return next((c for c in self.clusters if c.cid == cid), None)

    @property
    def kept(self) -> np.ndarray:
        """Cluster id per row with the dropped small groups marked ``-1``."""
        keep = {c.cid for c in self.clusters}
        return np.array([a if a in keep else -1 for a in self.assign])


def rule_space(ccg, null, duration: float, acg_ref=None, acg_tgt=None,
               ccg_hi=None, null_hi=None) -> np.ndarray:
    """The feature space clustering runs in — the terms the rules are stated in.

    Deliberately the interpretable blocks only: no kernel bank, no learned
    filters. Those are wide and correlated, which suits a tree but swamps a
    distance metric, where every column counts equally.
    """
    parts = [shape_features(ccg, null, duration),
             peak_features(ccg, null, duration),
             window_features(ccg, null, duration)]
    if ccg_hi is not None:
        parts += [peak_features(ccg_hi, null_hi, duration),
                  window_features(ccg_hi, null_hi, duration)]
    if acg_ref is not None:
        parts += [acg_features(acg_ref, duration), acg_features(acg_tgt, duration)]
    X = np.hstack(parts)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def embed(X: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Standardize then compress — distances need comparable, decorrelated axes."""
    Z = StandardScaler().fit_transform(X)
    k = min(n_components, Z.shape[0], Z.shape[1])
    return PCA(n_components=k, random_state=0).fit_transform(Z)


def cluster_pairs(X: np.ndarray, n_clusters: int = 40,
                  labels: list[list[str]] = None,
                  min_size: int = MIN_CLUSTER) -> 'Clustering':
    """Group rows of *X* into *n_clusters*, most populous first.

    Ward linkage, so clusters are compact in the embedded space rather than
    chained. Groups smaller than *min_size* are dropped: a handful of outliers
    is not worth a reviewing decision, and leaving them in makes the listing
    long without covering more pairs. *labels* (per row, possibly empty) is only
    used to report how the existing hand tags fall across the groups — never to
    form them.
    """
    Z = embed(X)
    # Capped by what min_size can support, not just by the row count: asking for
    # more groups than that splits a narrow scope into singletons and drops all
    # of them, leaving nothing to review.
    asked = n_clusters
    n_clusters = max(1, min(n_clusters, len(Z) // min_size))
    assign = AgglomerativeClustering(n_clusters=n_clusters,
                                     linkage='ward').fit_predict(Z)
    out = []
    for cid in range(n_clusters):
        members = np.flatnonzero(assign == cid)
        if len(members) < min_size:
            continue
        centre = Z[members].mean(axis=0)
        medoid = members[int(np.argmin(((Z[members] - centre) ** 2).sum(axis=1)))]
        counts: dict[str, int] = {}
        n_lab = 0
        for i in members:
            tags = labels[i] if labels is not None else []
            n_lab += bool(tags)
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
        out.append(Cluster(cid=cid, members=members, medoid=int(medoid),
                           tag_counts=counts, n_labeled=n_lab))
    return Clustering(clusters=sorted(out, key=len, reverse=True),
                      coords=Z[:, :2], assign=assign,
                      n_asked=asked, n_used=n_clusters)


SCOPES = ('all pairs', 'untagged only', 'tagged only')


def in_scope(samples: list, scope: str) -> np.ndarray:
    """Row mask for a scope name — which pairs the clustering should run over."""
    if scope == 'untagged only':
        return np.array([not s.labels for s in samples])
    if scope == 'tagged only':
        return np.array([bool(s.labels) for s in samples])
    return np.ones(len(samples), dtype=bool)


def cluster_report(clusters: list[Cluster]) -> str:
    """One line per cluster: size, how many are labeled, and the dominant tag."""
    rows = [f"  {'id':>3} {'size':>5} {'labeled':>8} {'purity':>7}  top tag",
            "  " + "-" * 52]
    for c in clusters:
        rows.append(f"  {c.cid:>3} {len(c):>5} {c.n_labeled:>8} "
                    f"{c.purity:>7.2f}  {c.top_tag}")
    return "\n".join(rows)


def label_agreement(clusters: list[Cluster], label_names: list[str]) -> dict:
    """How each existing label is spread across clusters.

    A label concentrated in one cluster is one the space represents well. One
    smeared over many is either a label covering several visual shapes, or a
    shape the features cannot see — the two cases the reviewer needs told apart.
    """
    where: dict[str, dict[int, int]] = {n: {} for n in label_names}
    for c in clusters:
        for tag, n in c.tag_counts.items():
            if tag in where:
                where[tag][c.cid] = n
    out = {}
    for name, spread in where.items():
        total = sum(spread.values())
        if not total:
            out[name] = {'total': 0, 'n_clusters': 0, 'concentration': 0.0}
            continue
        out[name] = {'total': total, 'n_clusters': len(spread),
                     'concentration': max(spread.values()) / total,
                     'top_clusters': sorted(spread.items(),
                                            key=lambda kv: -kv[1])[:3]}
    return out
