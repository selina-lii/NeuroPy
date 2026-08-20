"""The classifier workflow as one object, with no user interface attached.

The steps the Classify dialog takes, as calls: set a scope, train or load a
classifier, score the scope, narrow the predictions, and admit what survives —
so the same workflow runs from a notebook or a script.

The dialog does not yet delegate to this; until it does, the two are separate
implementations of the same workflow and changes belong in both.

    cs = ClassifierSession(cd)
    cs.set_scope(sessions=['RatJ_Day1'], conn_types=['pyr-pyr'])
    cs.train(['conv', 'kernel'], save_as='test2_E')
    cs.predict()
    cs.drop_below(0.8)
    cs.admit()
    cs.save_selections()
"""
from __future__ import annotations

from neuropy.analyses.pair_selection_data import SelectionDataset
from neuropy.classifier.predictions import PredictionStore, store_from_rows
from neuropy.classifier.run import (DEFAULT_MODEL, apply_cascade, list_models,
                                    load_model, missing_highres, predict_project,
                                    scope_keys, scorable_keys, train_project)


class Scope:
    """Which pointer keys a run covers, and the questions asked of that set."""

    def __init__(self, cd, sessions: list = None, conn_types: list = None):
        self.cd = cd
        self.sessions = list(sessions or [])
        self.conn_types = list(conn_types or [])

    @property
    def keys(self) -> list:
        return scope_keys(self.cd, self.sessions, self.conn_types)

    def missing_highres(self) -> list[str]:
        """Sessions in scope with no fine-binned CCGs on disk."""
        return missing_highres(self.cd, self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:
        return (f"Scope({len(self)} keys, "
                f"sessions={self.sessions or 'all'}, "
                f"types={self.conn_types or 'all'})")


class ClassifierSession:
    """One pass of the classify workflow: scope → model → predictions → tags."""

    def __init__(self, cd, sd: SelectionDataset = None):
        self.cd = cd
        self.sd = sd if sd is not None else SelectionDataset(cd)
        self.scope = Scope(cd)
        self.model = None
        self.model_name = ''
        self.store = PredictionStore()
        self.summary: dict = {}
        self.skipped: list[str] = []
        self._key_cache: dict = {}

    # ---- scope -----------------------------------------------------------

    def set_scope(self, sessions: list = None, conn_types: list = None) -> Scope:
        """Narrow the run; empty or omitted means every session / every type."""
        self.scope = Scope(self.cd, sessions, conn_types)
        self._key_cache.clear()
        return self.scope

    def complete_sessions(self) -> list[str]:
        """Sessions with at least one slice marked reviewed-exhaustively."""
        out = set()
        for key, sel in self.sd.sessions.items():
            if any(b.complete for b in sel.selections.values()):
                out.add(str(key.session))
        return sorted(out)

    def scope_complete(self) -> Scope:
        """Set the scope to the sessions that were reviewed exhaustively."""
        return self.set_scope(sessions=self.complete_sessions(),
                              conn_types=self.scope.conn_types)

    # ---- models ----------------------------------------------------------

    def available_models(self) -> list[dict]:
        """Saved classifiers with their training provenance, newest first."""
        return list_models(self.cd)

    def load(self, name: str) -> 'ClassifierSession':
        """Adopt a saved classifier without training."""
        self.model, self.model_name = load_model(self.cd, name), name
        self.summary = dict(getattr(self.model, 'meta', {}).get('trained_on', {}))
        return self

    def train(self, models: list = None, save_as: str = None, *,
              bias: str = 'balanced', min_count: int = 20, min_rats: int = 4,
              highres: bool = True, only_labels: list = None,
              extra: list = None, figures: bool = False) -> dict:
        """Fit on the project's saved labels and keep the result on this session."""
        names = list(models or [DEFAULT_MODEL])
        result = train_project(self.cd, names, figures=figures,
                               save_as=save_as, extra=extra, highres=highres,
                               min_count=min_count, min_rats=min_rats, bias=bias,
                               only_labels=only_labels,
                               conn_types=self.scope.conn_types)
        self.model = result['model']
        self.model_name = result['saved_as']
        self.summary = result['summary']
        return result

    def train_cascade(self, models: list, save_as: str, **kw) -> dict:
        """Train one model per strategy and let each claim what the earlier missed."""
        heads = [self.train([name], f'{save_as}.{name}', **kw) for name in models]
        self.model_name = ' → '.join(h['saved_as'] for h in heads)
        out = apply_cascade(self.cd, [h['saved_as'] for h in heads],
                            self.scope.keys)
        self.store = store_from_rows(out['rows'], self.model_name, out['labels'])
        self.skipped = out['skipped']
        return {'heads': [h['summary'] for h in heads], 'store': self.store}

    # ---- scoring ---------------------------------------------------------

    def predict(self, name: str = None) -> PredictionStore:
        """Score every pair in scope, replacing any earlier predictions."""
        if name is not None:
            self.load(name)
        if self.model is None:
            raise ValueError('train or load a classifier before predicting')
        keys, self.skipped = scorable_keys(self.cd, self.model, self.scope.keys)
        if not keys:
            raise ValueError(f"classifier {self.model_name!r} can score no session "
                             f"in this scope: every one lacks the CCGs it needs")
        rows = predict_project(self.cd, self.model, keys)
        self.store = store_from_rows(rows, self.model_name, self.model.label_names)
        return self.store

    def scores(self) -> dict:
        """Per-label F1/AUC/precision/recall from the run that produced the model."""
        return self.summary.get('scores', {})

    def counts(self) -> dict[str, int]:
        """How many pairs carry each predicted label right now."""
        return self.store.summary()

    # ---- narrowing the predictions ---------------------------------------

    def drop_below(self, threshold: float, label: str = None) -> int:
        """Drop predicted labels scoring under *threshold*; returns how many went."""
        return self._filter(lambda l, s: s >= threshold, label)

    def keep_only(self, labels) -> int:
        """Drop every predicted label outside *labels*."""
        want = set(labels)
        return self._filter(lambda l, s: l in want, None)

    def drop_labels(self, labels) -> int:
        """Drop these labels from every prediction, whatever they scored."""
        drop = set(labels)
        return self._filter(lambda l, s: l not in drop, None)

    def _filter(self, keep, label: str = None) -> int:
        """Apply a keep(label, score) rule across the store; returns labels dropped."""
        removed = 0
        for pk, row in list(self.store.rows.items()):
            kept = [l for l in row['labels']
                    if (label is not None and l != label)
                    or keep(l, row['scores'].get(l, 0.0))]
            removed += len(row['labels']) - len(kept)
            if kept:
                row['labels'] = kept
                row['scores'] = {l: v for l, v in row['scores'].items() if l in kept}
            else:
                del self.store.rows[pk]
        return removed

    # ---- admitting -------------------------------------------------------

    def key_of(self, pk: tuple):
        """The nd-key a predicted pair belongs to, memoized across a batch."""
        ident = (pk[0], self.store.type_label_for(*pk))
        if ident not in self._key_cache:
            self._key_cache[ident] = self.cd.find(ident[0], type_label=ident[1],
                                                  strict=False)
        return self._key_cache[ident]

    def admit(self, pairs=None, only: str = None) -> list[tuple]:
        """Turn predictions into real group tags, selecting the pairs as it goes."""
        changes = []
        for pk in (self.store.review_order(self.sd.groups) if pairs is None
                   else list(pairs)):
            for g, s, p in self.store.accept(pk[0], pk[1], pk[2], self.sd.groups,
                                             only=only):
                key = self.key_of(pk)
                if key is not None:
                    self.sd.tag_pair(key, p, g, add=True)
                changes.append((g, s, p, 'add'))
        return changes

    def retract(self, pk: tuple, label: str) -> list[tuple]:
        """Send an admitted label back to pending, untagging the pair."""
        key = self.key_of(pk)
        if key is not None:
            self.sd.tag_pair(key, (pk[1], pk[2]), label, add=False)
        else:
            self.sd.groups.discard_from_group(label, pk[0], (pk[1], pk[2]))
        return [(label, pk[0], (pk[1], pk[2]), 'remove')]

    def pending(self) -> list[tuple]:
        """Pairs with a label still to judge, most confident first."""
        return self.store.review_order(self.sd.groups)

    def admitted(self) -> list[tuple]:
        """Pairs of this run whose labels have all been accepted."""
        return [pk for pk in self.store.rows
                if not self.store.pending(self.sd.groups, *pk)]

    # ---- persistence -----------------------------------------------------

    def save_selections(self, *names: str) -> list[str]:
        """Write the admitted tags to disk, under ``latest`` and/or named snapshots."""
        return self.sd.save_as(*names)

    def report(self) -> str:
        """One-screen account of where this run stands."""
        lines = [f"scope: {self.scope}",
                 f"model: {self.model_name or '(none)'}"]
        if self.summary.get('mean_f1') is not None:
            lines.append(f"  mean F1 {self.summary.get('mean_f1', 0):.3f}  "
                         f"AUC {self.summary.get('mean_auc', 0):.3f}")
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} session(s): "
                         f"{', '.join(self.skipped)}")
        lines.append(f"predictions: {len(self.store)} pairs, "
                     f"{len(self.pending())} pending")
        for label, n in sorted(self.counts().items(), key=lambda kv: -kv[1]):
            lines.append(f"  {label:<14} {n}")
        return '\n'.join(lines)
