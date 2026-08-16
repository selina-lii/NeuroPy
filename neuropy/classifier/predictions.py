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

PREDICTED_PREFIX = 'pred_'


class PredictionStore:
    """Per-pair predicted labels with their scores, plus accept/reject bookkeeping."""

    def __init__(self, model_name: str = '', labels: list[str] = None):
        self.model_name = model_name
        self.labels = list(labels or [])
        self.rows: dict[tuple, dict] = {}      # (sess, ref, tgt) -> {labels, scores}
        self.accepted: set[tuple] = set()
        self.rejected: set[tuple] = set()

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pk(session: str, ref: int, tgt: int) -> tuple:
        return (str(session), int(ref), int(tgt))

    def add(self, session: str, ref: int, tgt: int, labels: list[str],
            scores: dict[str, float]):
        self.rows[self._pk(session, ref, tgt)] = {'labels': list(labels),
                                                  'scores': dict(scores)}

    def labels_for(self, session: str, ref: int, tgt: int) -> list[str]:
        row = self.rows.get(self._pk(session, ref, tgt))
        return list(row['labels']) if row else []

    def scores_for(self, session: str, ref: int, tgt: int) -> dict[str, float]:
        row = self.rows.get(self._pk(session, ref, tgt))
        return dict(row['scores']) if row else {}

    def top_label(self, session: str, ref: int, tgt: int) -> str:
        labs = self.labels_for(session, ref, tgt)
        return labs[0] if labs else ''

    def confidence(self, session: str, ref: int, tgt: int) -> float:
        """Score of the pair's top predicted label (0 when it abstained)."""
        row = self.rows.get(self._pk(session, ref, tgt))
        if not row or not row['labels']:
            return 0.0
        return max(row['scores'].get(l, 0.0) for l in row['labels'])

    def pairs_for_label(self, label: str, session: str = None) -> list[tuple]:
        """Pairs predicted *label*, most confident first."""
        hits = [(pk, row) for pk, row in self.rows.items()
                if label in row['labels'] and (session is None or pk[0] == session)]
        hits.sort(key=lambda kv: -kv[1]['scores'].get(label, 0.0))
        return [pk for pk, _ in hits]

    def review_order(self, session: str = None) -> list[tuple]:
        """Undecided pairs, least confident first — where review pays off most."""
        pend = [pk for pk in self.rows
                if pk not in self.accepted and pk not in self.rejected
                and (session is None or pk[0] == session)]
        return sorted(pend, key=lambda pk: self.confidence(*pk))

    def accept(self, session: str, ref: int, tgt: int, groups) -> list[str]:
        """Promote a pair's predictions into real group tags."""
        pk = self._pk(session, ref, tgt)
        labels = self.labels_for(*pk)
        for label in labels:
            groups.add_to_group(label, pk[0], (pk[1], pk[2]))
        self.accepted.add(pk)
        self.rejected.discard(pk)
        return labels

    def reject(self, session: str, ref: int, tgt: int):
        pk = self._pk(session, ref, tgt)
        self.rejected.add(pk)
        self.accepted.discard(pk)

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
               'accepted': [list(pk) for pk in sorted(self.accepted)],
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
                      row['labels'], row['scores'])
        store.accepted = {(s, int(r), int(t)) for s, r, t in doc.get('accepted', [])}
        store.rejected = {(s, int(r), int(t)) for s, r, t in doc.get('rejected', [])}
        return store


def store_from_rows(rows: list[dict], model_name: str,
                    labels: list[str]) -> PredictionStore:
    """Build a store from ``run.predict_project`` output."""
    store = PredictionStore(model_name, labels)
    for row in rows:
        store.add(str(row['key'].session), row['ref'], row['tgt'],
                  row['labels'], row['scores'])
    return store
