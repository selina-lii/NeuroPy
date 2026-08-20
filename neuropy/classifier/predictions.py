"""Tentative classifier labels, held apart from hand-made group tags.

Predictions are guesses. Writing them straight into ``GroupDataset`` would make
the next training run learn from its own output, so they live here instead and
only reach the groups when the user explicitly accepts them.

The store is keyed exactly like ``GroupDataset`` — ``(session, ref, tgt)`` — so a
pair's predictions and its real tags line up without translation.
"""
from __future__ import annotations

import json
import os

from neuropy.analyses.utils import ADMITTED_PREFIX


def admitted_group(model_name: str) -> str:
    """Marker group recording which model proposed a pair the user accepted.

    ``__special_`` keeps it off the tag chips and out of training, and naming the
    model means a later model can tell its own admissions from another's.
    """
    return ADMITTED_PREFIX + model_name


def prior_admissions(groups, model_name: str, sessions: set = None) -> int:
    """How many pairs *model_name* has already had accepted, within *sessions*."""
    marker = admitted_group(model_name)
    want = sessions if sessions is not None else groups.sessions_for_group(marker)
    return sum(len(groups.pairs_in_group(marker, s)) for s in want)


def by_owner(owner_of: dict, order: list = None) -> list[tuple]:
    """Invert ``{item: owner}`` into ``[(owner, [item, ...]), ...]``.

    Owners appear in the order their first item does, so a cascade's models
    line up with the order they ran in.
    """
    out: dict = {}
    for item in (order if order is not None else owner_of):
        out.setdefault(owner_of[item], []).append(item)
    return list(out.items())


class PredictionStore:
    """Per-pair predicted labels with their scores, plus accept/reject bookkeeping."""

    def __init__(self, model_name: str = '', labels: list[str] = None):
        self.model_name = model_name
        self.labels = list(labels or [])
        self.rows: dict[tuple, dict] = {}      # (sess, ref, tgt) -> {labels, scores}
        # What was accepted is read off the group tags themselves, never copied
        # here; only a rejection has no tag to record it.
        self.rejected: set[tuple] = set()

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pk(session: str, ref: int, tgt: int) -> tuple:
        return (str(session), int(ref), int(tgt))

    def add(self, session: str, ref: int, tgt: int, labels: list[str],
            scores: dict[str, float], type_label: str = '', by: dict = None):
        # type_label is kept so the review list can navigate to a pair that
        # lives under a different conn type than the one currently shown.
        # by maps label -> model that proposed it, empty for a single model.
        self.rows[self._pk(session, ref, tgt)] = {'labels': list(labels),
                                                  'scores': dict(scores),
                                                  'type_label': type_label,
                                                  'by': dict(by or {})}

    def by_model(self, session: str, ref: int, tgt: int,
                 labels: list[str] = None) -> list[tuple]:
        """``[(model, [label, ...]), ...]`` grouped by which head proposed each label.

        One model gives one group; a cascade gives one line per head, in the
        order they ran, so the origin of every chip is visible.
        """
        row = self.rows.get(self._pk(session, ref, tgt))
        if not row:
            return []
        names = self.labels_for(session, ref, tgt) if labels is None else labels
        return by_owner({l: row['by'].get(l, self.model_name) for l in names}, names)

    def labels_for(self, session: str, ref: int, tgt: int) -> list[str]:
        row = self.rows.get(self._pk(session, ref, tgt))
        return list(row['labels']) if row else []

    def scores_for(self, session: str, ref: int, tgt: int) -> dict[str, float]:
        row = self.rows.get(self._pk(session, ref, tgt))
        return dict(row['scores']) if row else {}

    def type_label_for(self, session: str, ref: int, tgt: int) -> str:
        row = self.rows.get(self._pk(session, ref, tgt))
        return row.get('type_label', '') if row else ''

    def top_label(self, session: str, ref: int, tgt: int) -> str:
        labs = self.labels_for(session, ref, tgt)
        return labs[0] if labs else ''

    def confidence(self, session: str, ref: int, tgt: int,
                   label: str = None) -> float:
        """Score for *label*, or the pair's strongest label when none is given.

        The review list ranks and cuts on the same number: filtered to one
        label that is the label's own score, otherwise the top score overall.
        """
        row = self.rows.get(self._pk(session, ref, tgt))
        if not row or not row['labels']:
            return 0.0
        if label is not None:
            return float(row['scores'].get(label, 0.0))
        return max(row['scores'].get(l, 0.0) for l in row['labels'])

    def pairs_for_label(self, label: str, session: str = None) -> list[tuple]:
        """Pairs predicted *label*, most confident first."""
        hits = [(pk, row) for pk, row in self.rows.items()
                if label in row['labels'] and (session is None or pk[0] == session)]
        hits.sort(key=lambda kv: -kv[1]['scores'].get(label, 0.0))
        return [pk for pk, _ in hits]

    def review_order(self, groups, session: str = None) -> list[tuple]:
        """Pairs with a label still to judge, most confident first.

        A partly-accepted pair stays listed: its remaining labels are still open.
        """
        pend = [pk for pk in self.rows
                if self.pending(groups, *pk) and pk not in self.rejected
                and (session is None or pk[0] == session)]
        return sorted(pend, key=lambda pk: -self.confidence(*pk))

    def taken(self, groups, session: str, ref: int, tgt: int) -> list[str]:
        """Labels of this pair already accepted, in the order the model ranked them.

        The tags themselves are the answer, so an accept undone elsewhere shows up
        here as pending again — there is no second copy that could go stale.
        """
        got = groups.groups_for_pair(session, ref, tgt)
        return [l for l in self.labels_for(session, ref, tgt) if l in got]

    def pending(self, groups, session: str, ref: int, tgt: int) -> list[str]:
        """Labels still awaiting judgement — the left half of the row's editor."""
        got = set(self.taken(groups, session, ref, tgt))
        return [l for l in self.labels_for(session, ref, tgt) if l not in got]

    def accept(self, session: str, ref: int, tgt: int, groups,
               only: str = None) -> list[tuple]:
        """Promote a pair's predictions into real group tags.

        *only* tags that one label instead of all of them — a prediction is often
        partly right, and the wrong labels should not ride along with the right one.
        Accepting moves a label from pending to taken rather than retiring the whole
        row, so the rest of the pair's labels stay judgeable.

        Returns the group additions actually made, as ``(group, sess, pair)``, so
        the caller can undo them; a label already taken adds nothing.
        The admitted marker names the model that proposed it, so a later model
        can see which of its own suggestions have already been judged.
        """
        pk = self._pk(session, ref, tgt)
        have = groups.groups_for_pair(*pk)
        new = [l for l in ([only] if only else self.labels_for(*pk)) if l not in have]
        if not new:
            return []
        marker = admitted_group(self.model_name)
        added = [marker] if marker not in have else []   # lands once per pair per model
        for label in new + added:
            groups.add_to_group(label, pk[0], (pk[1], pk[2]))
        self.rejected.discard(pk)
        return [(l, pk[0], (pk[1], pk[2])) for l in new + added]

    def reject(self, session: str, ref: int, tgt: int):
        self.rejected.add(self._pk(session, ref, tgt))

    def summary(self) -> dict[str, int]:
        """Predicted-pair count per label, for the review dialog's overview."""
        out: dict[str, int] = {}
        for row in self.rows.values():
            for label in row['labels']:
                out[label] = out.get(label, 0) + 1
        return out

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        doc = {'model': self.model_name, 'labels': self.labels,
               'rows': [{'session': s, 'ref': r, 'tgt': t, **row}
                        for (s, r, t), row in self.rows.items()],
               'rejected': [list(pk) for pk in sorted(self.rejected)]}
        with open(path, 'w') as fh:
            json.dump(doc, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> 'PredictionStore':
        with open(path) as fh:
            doc = json.load(fh)
        store = cls(doc.get('model', ''), doc.get('labels', []))
        for row in doc.get('rows', []):
            store.add(row['session'], row['ref'], row['tgt'],
                      row['labels'], row['scores'], row.get('type_label', ''),
                      row.get('by'))
        store.rejected = {(s, int(r), int(t)) for s, r, t in doc.get('rejected', [])}
        return store


def store_from_rows(rows: list[dict], model_name: str,
                    labels: list[str]) -> PredictionStore:
    """Build a store from ``run.predict_project`` output."""
    store = PredictionStore(model_name, labels)
    for row in rows:
        store.add(str(row['key'].session), row['ref'], row['tgt'],
                  row['labels'], row['scores'], row['key'].type_label(),
                  row.get('by'))
    return store
