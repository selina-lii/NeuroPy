"""One-call training + verification pipeline, shared by the CLI and the UI.

Kept deliberately thin: build the set, cross-validate, fit on everything, write
the figures. Anything cleverer belongs in the model, not here.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from neuropy.classifier.dataset import build_labeled_set, loaded_ccg
from neuropy.classifier.models import MODELS, BaseModel, decide
from neuropy.classifier.train import fit_final, leave_one_rat_out, report
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


def load_model(cd, name: str) -> BaseModel:
    return BaseModel.load(model_path(cd, name))


def train_project(cd, model_name: str = DEFAULT_MODEL, out_dir: str = None,
                  figures: bool = True, save_as: str = None) -> dict:
    """Train on a project's saved selections and write model + figures.

    The model also lands in the shared library under *save_as* (default: the
    project name plus the model), so it can later be applied to a different
    project without retraining.
    """
    out_dir = out_dir or os.path.join(cd.save_path, 'classifier')
    os.makedirs(out_dir, exist_ok=True)
    duration = cd.conf.duration

    ls = build_labeled_set(cd, cd.selections_dir)
    if not ls.label_names:
        raise ValueError("no label has enough examples to train on — "
                         "tag more pairs before running the classifier")

    cv = leave_one_rat_out(ls, model_name, duration=duration)
    model = fit_final(ls, model_name, duration=duration)

    summary = {'model': model_name, 'n_samples': len(ls.samples),
               'n_rats': len(set(ls.rats)), 'labels': ls.label_names,
               'scores': cv['scores'],
               'mean_f1': float(np.mean([s['f1'] for s in cv['scores'].values()])),
               'mean_auc': float(np.mean([s['auc'] for s in cv['scores'].values()]))}

    # Training sources travel with the weights: a model applied to another
    # project is only interpretable if you can see what taught it.
    provenance = {'project': cd.conf.name,
                  'selections_dir': str(cd.selections_dir),
                  'cross_validation': 'leave-one-rat-out',
                  'mean_f1': summary['mean_f1'], 'mean_auc': summary['mean_auc'],
                  **ls.provenance()}
    model.save(os.path.join(out_dir, f'{model_name}.pkl'), provenance)
    saved_as = save_as or f'{cd.conf.name}_{model_name}'
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


def apply_model(cd, name: str) -> dict:
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
    keys, skipped = scorable_keys(cd, model)
    if not keys:
        raise ValueError(f"classifier {name!r} can score no session in this "
                         f"project: every one lacks the CCGs it needs")
    return {'model': model, 'rows': predict_project(cd, model, keys),
            'meta': getattr(model, 'meta', {}), 'skipped': skipped}


def scorable_keys(cd, model) -> tuple[list, list[str]]:
    """Pointer keys this model can read, plus the sessions it must skip."""
    keys, skipped = [], set()
    for key in cd.ptr:
        needed = ['lowres'] + (['highres'] if model.uses_highres else [])
        if all(loaded_ccg(cd, key, r) is not None for r in needed):
            keys.append(key)
        else:
            skipped.add(str(key.session))
    return keys, sorted(skipped)


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
