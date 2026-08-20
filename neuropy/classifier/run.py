"""One-call training + verification pipeline, shared by the CLI and the UI.

Kept deliberately thin: build the set, cross-validate, fit on everything, write
the figures. Anything cleverer belongs in the model, not here.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from neuropy.analyses.ms_connectivity import CCGConfig, CCGDataset

from neuropy.classifier.dataset import build_multi, loaded_ccg
from neuropy.classifier.models import MODELS, BaseModel, decide
from neuropy.classifier.train import (fit_final, fit_routed, leave_one_rat_out,
                                      report, routed_cv)
from neuropy.classifier.verify import verify_all

DEFAULT_MODEL = 'conv'   # learned local filters + boosted trees; see CCG_CLASSIFIER_RESULTS.md


def library_dir(cd) -> str:
    """Shared classifier store, beside the projects rather than inside one.

    A trained model is meant to be applied to projects other than the one that
    produced it, so it does not belong under a single ``project_*`` directory.
    """
    return os.path.join(str(cd.data_root), 'classifiers')


def model_path(cd, name: str) -> str:
    return os.path.join(library_dir(cd), f'{name}.pkl')


def list_models(cd) -> list[dict]:
    """Saved classifiers with their training provenance, newest first."""
    out = []
    for side in sorted(glob.glob(os.path.join(library_dir(cd), '*.json'))):
        pkl = os.path.splitext(side)[0] + '.pkl'
        if not os.path.isfile(pkl):
            continue
        with open(side) as fh:
            meta = json.load(fh)
        out.append({'name': os.path.basename(pkl)[:-4], 'path': pkl, **meta})
    return sorted(out, key=lambda m: m.get('saved_at', ''), reverse=True)


def delete_model(cd, name: str):
    """Remove a saved classifier — both the weights and their sidecar."""
    for path in (model_path(cd, name), model_path(cd, name)[:-4] + '.json'):
        if os.path.isfile(path):
            os.remove(path)


def rename_model(cd, old: str, new: str):
    """Rename a classifier; the name is its identity, so collisions are refused."""
    new = new.strip()
    if not new or new == old:
        return
    if os.path.isfile(model_path(cd, new)):
        raise ValueError(f"a classifier named {new!r} already exists")
    for ext in ('.pkl', '.json'):
        src = model_path(cd, old)[:-4] + ext
        if os.path.isfile(src):
            os.rename(src, model_path(cd, new)[:-4] + ext)


def open_projects(names: list[str]) -> list:
    """Load sibling projects by name, for pooling their selections into training."""
    return [CCGDataset(CCGConfig(name=n)) for n in names]


def load_model(cd, name: str) -> BaseModel:
    return BaseModel.load(model_path(cd, name))


def train_project(cd, model_names: list = None, out_dir: str = None,
                  figures: bool = True, save_as: str = None,
                  extra: list = None, highres: bool = True,
                  min_count: int = 20, bias: str = 'balanced',
                  only_labels: list = None, conn_types: list = None,
                  min_rats: int = 4) -> dict:
    """Train on saved selections and store the model in the shared library.

    Defaults are the widest useful ones: every labeled pair in the loaded
    project, both resolutions, saved under the project's name. *extra* pools in
    other projects' selections; *save_as* names the classifier.

    With *highres*, any session missing its fine-binned CCGs is computed and
    saved first — a model trained on both resolutions emits a fixed feature
    width and could never score a lowres-only session afterwards.
    """
    out_dir = out_dir or os.path.join(cd.save_path, 'classifier')
    os.makedirs(out_dir, exist_ok=True)
    duration = cd.conf.duration
    datasets = [cd] + list(extra or [])

    # The UI queues any missing high-res compute before calling this, so a
    # session still lacking it here is an error rather than something to fix inline.
    ls = build_multi(datasets, min_count=min_count, min_rats=min_rats,
                     highres=highres, only_labels=only_labels,
                     conn_types=conn_types)
    if not ls.label_names:
        raise ValueError("no label has enough examples to train on — "
                         "tag more pairs before running the classifier")

    # Several strategies -> one routed model: each label answered by whichever
    # strategy cross-validates best for it, since the encoding that suits a
    # quality grade is not the one that suits a rhythm label.
    names = list(model_names or [DEFAULT_MODEL])
    routes = {}
    if len(names) > 1:
        model, per_model = fit_routed(ls, names, duration=duration, bias=bias)
        routes = model.routes
        cv = routed_cv(per_model, routes, ls.label_names)
    else:
        cv = leave_one_rat_out(ls, names[0], duration=duration, bias=bias)
        model = fit_final(ls, names[0], duration=duration, bias=bias)

    summary = {'model': ' + '.join(names), 'bias': bias, 'routes': routes,
               'n_samples': len(ls.samples),
               'n_rats': len(set(ls.rats)), 'labels': ls.label_names,
               'scores': cv['scores'],
               'mean_f1': float(np.mean([s['f1'] for s in cv['scores'].values()])),
               'mean_auc': float(np.mean([s['auc'] for s in cv['scores'].values()]))}

    # Training sources travel with the weights: a model applied to another
    # project is only interpretable if you can see what taught it.
    provenance = {'projects': list(getattr(ls, 'sources', [cd.conf.name])),
                  'project': cd.conf.name,
                  'selections_dir': str(cd.selections_dir),
                  'cross_validation': 'leave-one-rat-out',
                  'conn_types': list(conn_types or []),
                  'mean_f1': summary['mean_f1'], 'mean_auc': summary['mean_auc'],
                  'scores': cv['scores'], 'routes': routes,
                  **ls.provenance()}
    saved_as = save_as or cd.conf.name
    model.save(os.path.join(out_dir, f"{'+'.join(names)}.pkl"), provenance)
    model.save(model_path(cd, saved_as), provenance)
    summary['saved_as'] = saved_as
    with open(os.path.join(out_dir, 'scores.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)

    paths = verify_all(ls, cv, out_dir, duration) if figures else []
    # The per-sample traces and CV score matrix are only needed for the figures;
    # holding them alongside the model doubles the run's peak memory for nothing.
    del cv['proba'], cv['pred'], cv['thresholds'], ls
    return {'model': model, 'summary': summary, 'figures': paths,
            'out_dir': out_dir, 'saved_as': saved_as}


def apply_model(cd, name: str, scope: list = None) -> dict:
    """Score *cd* with a saved classifier — the cross-project path, no training.

    Sessions whose CCGs the model cannot read (a resolution it was trained with
    is missing) are skipped and named in ``skipped``, rather than failing the
    whole run or being silently scored on partial features.
    """
    model = load_model(cd, name)
    problems = [p for p in model.compatible_with(cd) if 'missing' not in p]
    if problems:
        raise ValueError(f"classifier {name!r} does not fit this project: "
                         + "; ".join(problems))
    keys, skipped = scorable_keys(cd, model, scope)
    if not keys:
        raise ValueError(f"classifier {name!r} can score no session in this "
                         f"project: every one lacks the CCGs it needs")
    return {'model': model, 'rows': predict_project(cd, model, keys),
            'meta': getattr(model, 'meta', {}), 'skipped': skipped}


def apply_cascade(cd, names: list[str], scope: list = None) -> dict:
    """Apply models in order, each label decided by the first model to claim it.

    Every model sees every pair. What an earlier model already found on a pair is
    kept; a later model may still add labels the earlier one did not give it, so
    a series of heads discovers more than any one of them alone. Order is a
    priority: the first model to claim a label owns it, so put the model you
    trust most first.

    Each row carries ``by``: ``{label: model}``, so the review UI can show which
    head proposed what.
    """
    rows: dict[tuple, dict] = {}
    labels, skipped = [], set()
    for name in names:
        out = apply_model(cd, name, scope)
        skipped.update(out['skipped'])
        for label in out['model'].label_names:
            if label not in labels:
                labels.append(label)
        for row in out['rows']:
            if not row['labels']:
                continue
            pk = (str(row['key'].session), row['ref'], row['tgt'])
            prev = rows.get(pk)
            if prev is None:
                rows[pk] = {**row, 'by': {l: name for l in row['labels']}}
                continue
            for label in row['labels']:
                if label in prev['by']:      # an earlier head already claimed it
                    continue
                prev['labels'].append(label)
                prev['scores'][label] = row['scores'].get(label, 0.0)
                prev['by'][label] = name
    return {'rows': list(rows.values()), 'labels': labels,
            'skipped': sorted(skipped)}


def scope_keys(cd, sessions: list = None, type_labels: list = None) -> list:
    """Pointer keys inside a scope; an empty or ``None`` list means no narrowing.

    Scope narrows *which* pairs get scored, never how the model works — the
    default is always the widest one the loaded project offers.
    """
    want_sess, want_type = set(sessions or []), set(type_labels or [])
    return [key for key in cd.ptr
            if (not want_sess or str(key.session) in want_sess)
            and (not want_type or key.type_label() in want_type)]


def missing_highres(cd, keys=None) -> list[str]:
    """Sessions in scope with no high-res CCGs on disk."""
    out = set()
    for key in (list(cd.ptr) if keys is None else keys):
        if loaded_ccg(cd, key, 'highres') is None:
            out.add(str(key.session))
    return sorted(out)


def scorable_keys(cd, model, keys=None) -> tuple[list, list[str]]:
    """Split *keys* into those this model can read and the sessions it cannot."""
    ok, skipped = [], set()
    needed = ['lowres'] + (['highres'] if model.uses_highres else [])
    for key in (list(cd.ptr) if keys is None else keys):
        if all(loaded_ccg(cd, key, r) is not None for r in needed):
            ok.append(key)
        else:
            skipped.add(str(key.session))
    return ok, sorted(skipped)


def predict_project(cd, model, keys=None) -> list[dict]:
    """Score every pointer pair in *cd* — the pairs the UI offers for review.

    Each row is ``{key, ref, tgt, labels, scores}``; ``labels`` is already the
    thresholded decision, with ``'?'`` where the model would not commit.
    """
    rows = []
    for key in (keys if keys is not None else list(cd.ptr)):
        pairs = cd.ptr[key.ptr()].pairs if key.ptr() in cd.ptr else []
        if len(pairs) == 0:
            continue
        data = cd.ccg_for(key)
        ccg, null = data.ccg[0], data.ccg_null[0]
        hi = loaded_ccg(cd, key, 'highres') if model.uses_highres else None
        idx = [(int(r), int(t)) for r, t in pairs
               if r < ccg.shape[0] and t < ccg.shape[1]]
        if not idx:
            continue
        X = np.stack([ccg[r, t] for r, t in idx]).astype(float)
        N = np.stack([null[r, t] for r, t in idx]).astype(float)
        X_hi = N_hi = None
        if hi is not None and hi.ccg is not None:
            X_hi = np.stack([hi.ccg[0][r, t] for r, t in idx]).astype(float)
            N_hi = np.stack([hi.ccg_null[0][r, t] for r, t in idx]).astype(float)
        proba = model.predict_proba(X, N, X_hi, N_hi)
        labels = decide(proba, model.label_names, model.thresholds)
        for (r, t), p, lab in zip(idx, proba, labels):
            # Only the scores that cleared their threshold are kept: storing all
            # 13 per pair costs ~10x more for numbers the review never displays.
            rows.append({'key': key, 'ref': r, 'tgt': t, 'labels': lab,
                         'scores': {n: round(float(v), 3)
                                    for n, v in zip(model.label_names, p)
                                    if n in lab}})
        del X, N, X_hi, N_hi, proba
    return rows
