#!/usr/bin/env python
"""Rule-set tags vs hand groups on PYR→INT pairs, plus screening pie charts in the GUI's dark theme.

    python notebooks/rule_tag_stats.py                   # test2, figures to notebooks/figures/rule_tag_stats
    python notebooks/rule_tag_stats.py --project X --out DIR
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from neuropy.analyses.ms_connectivity import open_project

PYR_INT = 'E PYR→INT'
SYNAP_GROUPS = ['best', 'good', 'ok', 'ok_wide']
SUBMS_GROUPS = ['msconn', 'leak', '2peakms', 'triple', 'wideMs', 'rift']
RHYTHM_BUCKET = ['rhythm', '0rhythm', 'rhy_2peak', 'wideMs', 'ok_wide']
SUBMS_BUCKET = ['msconn', 'leak', '2peakms', 'triple', 'rift']
MONO_BUCKET = ['best', 'good', 'ok', 'ok_wide']
BUCKET_SECTIONS = [('monosynaptic', MONO_BUCKET), ('rhythm', RHYTHM_BUCKET), ('sub-millisecond', SUBMS_BUCKET)]
# pie sections in priority order: (label, groups); a pair lands in the first section it matches
SYNAP_PIE = [('best', ['best']), ('good', ['good']), ('ok', ['ok']),
             ('rhythm', RHYTHM_BUCKET), ('sub-millisecond', SUBMS_BUCKET)]
SUBMS_PIE = [('msconn', ['msconn']), ('leak', ['leak']), ('triple / 2peakms', ['triple', '2peakms']),
             ('wideMs', ['wideMs']), ('rift', ['rift']), ('monosynaptic', MONO_BUCKET),
             ('rhythm', RHYTHM_BUCKET)]
BG, FG, MUTED = '#1e1e1e', '#dddddd', '#888888'   # ccg_panel's dark plot theme
GREEN, GREY = '#4E9A4E', '#555555'
LIGHT_GREEN, DARK_GREEN = '#7fd17f', '#2f5f2f'
FONT = 37
BUCKET_COLORS = {'monosynaptic': '#ff7f0e', 'rhythm': '#f2c94c', 'sub-millisecond': '#1f77b4'}


def collect(cd, sd) -> tuple[pd.DataFrame, dict]:
    """Per (session, conn type): pair sets for all / significant / selected, and per-pair hand groups."""
    tagged = {name: sd.rule_set_pairs(name) for name in ('synap', 'subms')}
    rows, groups = [], {}
    for k, ptr in cd.ptr.items():
        sess, conn = str(k.session), k.type_label()
        bucket = sd.get_selection_by_session(k).selections[k]
        lift = lambda pairs: {(sess, *p) for p in pairs}
        rows.append(dict(session=sess, conn=conn, all=lift(cd.type_pairs(k)),
                         significant=lift(ptr.pair_set),   # admitted pairs too, so Selected nests inside it
                         selected=lift(p for p in bucket.selected if len(p) == 2),
                         **{n: lift(t.get(k, ())) for n, t in tagged.items()}))
        for p, t in bucket.tags.items():
            if len(p) == 2:
                groups[(conn, sess, *p)] = set(t.get('groups') or ())
    return pd.DataFrame(rows), groups


def pooled(df: pd.DataFrame, conn: str, col: str) -> set:
    return set().union(*df.loc[df.conn == conn, col])


def tag_vs_groups(df, groups, tag: str, names: list) -> pd.DataFrame:
    """Within selected PYR→INT pairs: each group's share, its overlap with *tag*, precision and recall."""
    pop = pooled(df, PYR_INT, 'selected')
    tag_pairs = pooled(df, PYR_INT, tag) & pop
    member = lambda g: {p for p in pop if g in groups.get((PYR_INT, *p), ())}
    out = []
    for name in names + [' ∪ '.join(names)]:
        g = set().union(*(member(n) for n in name.split(' ∪ ')))
        both = g & tag_pairs
        out.append(dict(group=name, pairs=len(g), pct_of_selected=100 * len(g) / len(pop),
                        overlap=len(both), precision=100 * len(both) / max(len(tag_pairs), 1),
                        recall=100 * len(both) / max(len(g), 1)))
    print(f"\n{tag}: {len(tag_pairs)} of {len(pop)} selected {PYR_INT} pairs carry the tag")
    return pd.DataFrame(out).set_index('group').round(1)


def composition(pairs: set, groups: dict, sections: list) -> dict:
    """``{section: (count, recall %)}``: each pair goes to the first section holding any of its groups, else 'none'; recall is over every pair in the section's groups."""
    counts = dict.fromkeys([s for s, _ in sections] + ['none'], 0)
    for p in pairs:
        have = groups.get((PYR_INT, *p), set())
        counts[next((s for s, gs in sections if have & set(gs)), 'none')] += 1
    recall = {}
    for s, gs in sections:
        members = {k[1:] for k, have in groups.items() if k[0] == PYR_INT and have & set(gs)}
        recall[s] = 100 * len(members & pairs) / max(len(members), 1)
    return {s: (n, recall.get(s)) for s, n in counts.items()}


def section_colors(sd, labels: list) -> list:
    """Bucket colours shared by every chart, else the (first) group's own tag colour, else matplotlib's cycle; 'none' greys out."""
    cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'][4:])   # past the bucket blue/orange
    reg = sd.groups.registry
    first = lambda lab: lab.split(' / ')[0]
    return [GREY if lab == 'none' else BUCKET_COLORS.get(lab) or
            (reg[first(lab)].ui_color if first(lab) in reg and reg[first(lab)].ui_color else next(cycle))
            for lab in labels]


def pie(ax, counts: dict, colors: list, title: str, *, center=(0, 0), radius: float = 1.0,
        startangle: float = 90, legend: bool = True) -> tuple:
    """Wedges without outlines, title above and legend below; ``(wedges, labels)``; a count may carry a recall % as ``(n, recall)``."""
    keep = [(lab, *(n if isinstance(n, tuple) else (n, None)), c) for (lab, n), c in zip(counts.items(), colors)
            if (n[0] if isinstance(n, tuple) else n)]
    total = sum(n for _, n, _, _ in keep)
    wedges, _ = ax.pie([n for _, n, _, _ in keep], colors=[c for *_, c in keep], startangle=startangle,
                       center=center, radius=radius, wedgeprops=dict(linewidth=0))
    labels = [f"{lab}  {n} ({100 * n / total:.0f}%" + (f", recall={r:.0f}%)" if r is not None else ")")
              for lab, n, r, _ in keep]
    if legend:
        ax.add_artist(ax.legend(wedges, labels, loc='upper center', bbox_to_anchor=(center[0], center[1] - 1.1),
                                bbox_transform=ax.transData, frameon=False, labelcolor=FG,
                                fontsize=FONT)).set_clip_on(False)   # a second pie on this axes keeps the first legend
    if title:
        ax.text(center[0], center[1] + radius + 0.15, f"{title}\nn = {total}", color=FG, fontsize=FONT + 4,
                ha='center', va='bottom')
    return wedges, labels


def pie_of_pie(ax, sd, every: set, sig: set, sel: set, conn: str, groups: dict, gap: float = 2.3):
    """Significant out of all possible, its green wedge opened into selected pairs (split by group bucket on PYR→INT); no inset for PYR→PYR."""
    frac = len(sig) / len(every)
    inset = conn != 'E PYR→PYR'
    main = pie(ax, {'significant': len(sig), 'not significant': len(every - sig)}, [GREEN, GREY],
               None, startangle=-180 * frac, legend=False)   # the green wedge faces the inset
    parts = [main]
    if inset:
        chosen = sel & sig
        if conn == PYR_INT:
            split = {k: n for k, (n, _) in composition(chosen, groups, BUCKET_SECTIONS).items()}
            counts = {**{k: v for k, v in split.items() if k != 'none'}, 'other selected': split['none']}
            colors = [BUCKET_COLORS[k] for k in BUCKET_COLORS] + [LIGHT_GREEN]
        else:
            counts, colors = {'selected': len(chosen)}, [LIGHT_GREEN]
        parts.append(pie(ax, {**counts, 'not selected': len(sig - sel)}, colors + [DARK_GREEN], None,
                         center=(gap, 0), radius=0.7, legend=False))
        w = main[0][0]
        for theta, y in ((w.theta2, 0.7), (w.theta1, -0.7)):
            ax.plot([np.cos(np.radians(theta)), gap], [np.sin(np.radians(theta)), y], color=MUTED, linewidth=2)
    shown = [(w, lab) for wedges, labels in parts for w, lab in zip(wedges, labels) if not lab.startswith('not ')]
    legend = ax.legend(*zip(*shown), loc='upper center', bbox_to_anchor=(0.5, 0), frameon=False,
                       labelcolor=FG, fontsize=FONT)
    legend.set_in_layout(False)   # the pies keep one size whatever the legend holds
    mid = gap / 2 if inset else 0
    ax.text(mid, 1.2, f"{conn}\nn = {len(every)}", color=FG, fontsize=FONT + 4, ha='center', va='bottom')
    ax.set_xlim(-1.1, gap + 0.8 if inset else 1.1)
    ax.set_ylim(-1.05, 1.9)
    return legend


def outline(ax, wedges: list, labels: list, names: set) -> None:
    """A thick white outline around the wedges whose section is in *names* (they come first, so they are contiguous)."""
    picked = [w for w, lab in zip(wedges, labels) if lab.split('  ')[0] in names]
    if picked:
        ax.add_patch(Wedge((0, 0), 1.0, picked[0].theta1, picked[-1].theta2, fill=False, edgecolor='white',
                           linewidth=8))


def dark_figure(cols: int, title: str = None, size=(9, 14)):
    fig, axes = plt.subplots(1, cols, figsize=(size[0] * cols, size[1]), facecolor=BG, layout='constrained')
    if title:
        fig.suptitle(title, color=FG, fontsize=FONT + 8)
    return fig, np.atleast_1d(axes)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='test2')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'figures', 'rule_tag_stats'))
    args = ap.parse_args()
    _, cd, sd = open_project(args.project, [])
    df, groups = collect(cd, sd)
    pd.set_option('display.width', 200)
    print(tag_vs_groups(df, groups, 'synap', SYNAP_GROUPS).to_string())
    print(tag_vs_groups(df, groups, 'subms', SUBMS_GROUPS).to_string())

    os.makedirs(args.out, exist_ok=True)
    for i, conn in enumerate(sorted(df.conn.unique()), 1):
        fig, (ax,) = dark_figure(1, size=(14, 8))
        legend = pie_of_pie(ax, sd, pooled(df, conn, 'all'), pooled(df, conn, 'significant'),
                            pooled(df, conn, 'selected'), conn, groups)
        fig.savefig(os.path.join(args.out, f"1{'abcd'[i - 1]}_significant_and_selected_{conn.replace('→', '-').replace(' ', '_')}.png"),
                    facecolor=BG, dpi=100, bbox_inches='tight', pad_inches=0.4, bbox_extra_artists=[legend])

    fig, axes = dark_figure(2, f'{PYR_INT}: what the rule tags hold', size=(18, 22))
    for ax, (tag, sections, own) in zip(axes, [('synap', SYNAP_PIE, SYNAP_GROUPS), ('subms', SUBMS_PIE, SUBMS_GROUPS)]):
        counts = composition(pooled(df, PYR_INT, tag), groups, sections)
        wedges, labels = pie(ax, counts, section_colors(sd, list(counts)), tag)
        outline(ax, wedges, labels, {s for s, gs in sections if set(gs) <= set(own)})
        ax.set_ylim(-1.2, 1.7)   # room for the title text above the pie
    fig.savefig(os.path.join(args.out, '2_tag_composition.png'), facecolor=BG, dpi=100, bbox_inches='tight', pad_inches=0.4)
    print(f"\nfigures → {args.out}")
