"""One-call training + verification pipeline, shared by the CLI and the UI.

Kept deliberately thin: build the set, cross-validate, fit on everything, write
the figures. Anything cleverer belongs in the model, not here.
"""
from __future__ import annotations

import json
import os

import numpy as np

from neuropy.classifier.dataset import build_labeled_set
from neuropy.classifier.models import MODELS, decide
from neuropy.classifier.train import fit_final, leave_one_rat_out, report
from neuropy.classifier.verify import verify_all

DEFAULT_MODEL = 'hybrid'


def train_project(cd, model_name: str = DEFAULT_MODEL, out_dir: str = None,
                  figures: bool = True) -> dict:
    """Train on a project's saved selections and write model + figures.

    Returns the trained model, the cross-validated scores, and the figure paths.
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
    model.save(os.path.join(out_dir, f'{model_name}.pkl'))

    summary = {'model': model_name, 'n_samples': len(ls.samples),
               'n_rats': len(set(ls.rats)), 'labels': ls.label_names,
               'scores': cv['scores'],
               'mean_f1': float(np.mean([s['f1'] for s in cv['scores'].values()])),
               'mean_auc': float(np.mean([s['auc'] for s in cv['scores'].values()]))}
    with open(os.path.join(out_dir, 'scores.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)

    paths = verify_all(ls, cv, out_dir, duration) if figures else []
    # The per-sample traces and CV score matrix are only needed for the figures;
    # holding them alongside the model doubles the run's peak memory for nothing.
    del cv['proba'], cv['pred'], cv['thresholds'], ls
    return {'model': model, 'summary': summary, 'figures': paths,
            'out_dir': out_dir}


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
        idx = [(int(r), int(t)) for r, t in pairs
               if r < ccg.shape[0] and t < ccg.shape[1]]
        if not idx:
            continue
        X = np.stack([ccg[r, t] for r, t in idx]).astype(float)
        N = np.stack([null[r, t] for r, t in idx]).astype(float)
        proba = model.predict_proba(X, N)
        labels = decide(proba, model.label_names, model.thresholds)
        for (r, t), p, lab in zip(idx, proba, labels):
            # Only the scores that cleared their threshold are kept: storing all
            # 13 per pair costs ~10x more for numbers the review never displays.
            rows.append({'key': key, 'ref': r, 'tgt': t, 'labels': lab,
                         'scores': {n: round(float(v), 3)
                                    for n, v in zip(model.label_names, p)
                                    if n in lab}})
        del X, N, proba
    return rows
