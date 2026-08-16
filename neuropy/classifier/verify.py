"""Visual verification: does a predicted label actually look like its name?

Scores alone cannot tell you whether 'best' means "prominent peak". These figures
answer that directly by plotting what the model chose, so a wrong-looking panel is
visible rather than hidden inside an F1 number.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from neuropy.classifier.features import (FEATURE_NAMES, derivative, lag_axis,
                                         residual, shape_features, smooth)


def _panel(ax, ccg, null, duration, title=''):
    """Residual trace with its slope underlaid — the shape the model actually sees."""
    res = smooth(residual(ccg[None], null[None]), 1.0)[0]
    lags = lag_axis(len(res), duration)
    ax.axhline(0, color='0.8', lw=0.6)
    ax.axvline(0, color='0.8', lw=0.6, ls='--')
    ax.plot(lags, res, color='tab:blue', lw=1.2)
    ax.plot(lags, derivative(res[None], 1)[0], color='tab:orange', lw=0.7, alpha=0.7)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)


def plot_label_examples(ls, proba, label, out_dir, n=12, duration=0.02,
                        thresholds=None):
    """Top-scoring pairs for *label*, most confident first.

    If the label means what its name says, the grid should look homogeneous —
    e.g. every 'best' panel showing one prominent peak. A grid that looks mixed
    is the failure this figure exists to expose.
    """
    j = ls.label_names.index(label)
    order = np.argsort(-proba[:, j])[:n]
    ccg, null, Y = ls.X_ccg, ls.X_null, ls.Y
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.7))
    for ax, i in zip(np.ravel(axes), order):
        truth = 'Y' if Y[i, j] else 'n'
        _panel(ax, ccg[i], null[i], duration,
               f"p={proba[i, j]:.2f} true={truth}\n{ls.samples[i].session}")
    for ax in np.ravel(axes)[len(order):]:
        ax.axis('off')
    fig.suptitle(f"'{label}' — highest-scoring pairs (blue=residual, orange=slope)",
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, f'examples_{label}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_feature_separation(ls, out_dir, duration=0.02):
    """Each label's shape descriptors against everything else.

    This is the sanity check that the features carry the meaning their names
    claim: 'msconn' should stand out on ms_peak, 'rhythm' on rhythm_index. A
    label that separates on nothing cannot be learned from these features.
    """
    F = shape_features(ls.X_ccg, ls.X_null, duration)
    Y = ls.Y
    n_lab = len(ls.label_names)
    fig, axes = plt.subplots(n_lab, 1, figsize=(9, 1.5 * n_lab), sharex=True)
    for ax, j, name in zip(np.ravel(axes), range(n_lab), ls.label_names):
        pos, neg = Y[:, j] == 1, Y[:, j] == 0
        d = []
        for c in range(F.shape[1]):
            a, b = F[pos, c], F[neg, c]
            sd = np.sqrt((a.var() + b.var()) / 2) + 1e-9
            d.append((a.mean() - b.mean()) / sd)          # Cohen's d
        ax.bar(range(len(d)), d, color=np.where(np.array(d) > 0, 'tab:red', 'tab:blue'))
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylabel(f'{name}\n(n={int(Y[:, j].sum())})', fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1].set_xticks(range(len(FEATURE_NAMES)))
    axes[-1].set_xticklabels(FEATURE_NAMES, rotation=60, ha='right', fontsize=7)
    fig.suptitle("Feature separation per label (Cohen's d vs all other pairs)",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, 'feature_separation.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_mean_shapes(ls, proba, out_dir, duration=0.02):
    """Mean residual of predicted vs. true members of each label, overlaid.

    If the model learned the label rather than a confound, the two curves have
    the same shape. A predicted mean that looks nothing like the true mean means
    the classifier found some other regularity.
    """
    ccg, null, Y = ls.X_ccg, ls.X_null, ls.Y
    res = smooth(residual(ccg, null), 1.0)
    lags = lag_axis(res.shape[1], duration)
    n_lab = len(ls.label_names)
    cols = 4
    rows = int(np.ceil(n_lab / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.1))
    for ax, j in zip(np.ravel(axes), range(n_lab)):
        true_m = res[Y[:, j] == 1].mean(axis=0)
        top = np.argsort(-proba[:, j])[:max(10, int(Y[:, j].sum()))]
        ax.axhline(0, color='0.8', lw=0.6)
        ax.axvline(0, color='0.8', lw=0.6, ls='--')
        ax.plot(lags, true_m, color='k', lw=1.4, label='labeled')
        ax.plot(lags, res[top].mean(axis=0), color='tab:red', lw=1.2, ls='--',
                label='predicted')
        ax.set_title(ls.label_names[j], fontsize=8)
        ax.tick_params(labelsize=6)
    np.ravel(axes)[0].legend(fontsize=6)
    for ax in np.ravel(axes)[n_lab:]:
        ax.axis('off')
    fig.suptitle('Mean CCG residual: labeled vs. predicted members', fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, 'mean_shapes.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_score_summary(scores, out_dir):
    """F1 and AUC per label, sorted — the one-glance quality overview."""
    names = sorted(scores, key=lambda n: -scores[n]['f1'])
    f1 = [scores[n]['f1'] for n in names]
    auc = [scores[n]['auc'] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1.5))
    ax.barh(y - 0.2, f1, height=0.4, label='F1', color='tab:blue')
    ax.barh(y + 0.2, auc, height=0.4, label='AUC', color='tab:green')
    ax.axvline(0.5, color='0.6', lw=0.8, ls='--')     # AUC 0.5 = chance
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({scores[n]['n']})" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title('Leave-one-rat-out performance per label', fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, 'scores.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def verify_all(ls, cv_result, out_dir, duration=0.02, n_examples=12) -> list[str]:
    """Write the full figure set; returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    proba = cv_result['proba']
    paths = [plot_score_summary(cv_result['scores'], out_dir),
             plot_feature_separation(ls, out_dir, duration),
             plot_mean_shapes(ls, proba, out_dir, duration)]
    for label in ls.label_names:
        paths.append(plot_label_examples(ls, proba, label, out_dir,
                                         n=n_examples, duration=duration))
    return paths
