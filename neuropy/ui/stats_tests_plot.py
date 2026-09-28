"""Stats-test figures: pure matplotlib, no widgets and no Qt backend selection."""
from __future__ import annotations

import numpy as np

from neuropy.ui.pair_selection_panel import SelectionData
from neuropy.ui.stats_tests_backend import (
    _BAR_COLORS, StatsTestBackend, StatsTestConfig, _ViewConfig, _star)


def _mean_sem(x) -> tuple:
    if x.size == 0:
        return np.nan, 0.0
    return float(np.mean(x)), (float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0)


def pairwise_pvals(result, groups) -> list:
    if not result or 'error' in result:
        return []
    for key in ('tukey', 'posthoc'):
        rows = result.get(key)
        if rows:
            return [{'a': r.get('a', ''), 'b': r.get('b', ''),
                     'p_adj': r.get('p_adj', 1.0)} for r in rows]
    return []


def _draw_group_axes(ax, groups, title, view: '_ViewConfig', *, log: bool, dark: bool,
                     paired=False, sig_pairs=(), one_sample=False) -> None:
    """One panel of the stats figure: bars or violins, jittered points, pairing lines, brackets."""
    if not groups:
        ax.axis('off')
        return
    fg, bg = ('#ffffff', '#1e1e1e') if dark else ('#000000', 'white')
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=8)
    ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for sp in ax.spines.values():
        sp.set_color('#555555' if dark else '#cccccc')

    n_g = len(groups)
    names = [g.get('name', chr(65 + i)) or chr(65 + i) for i, g in enumerate(groups)]
    arrays = [np.array(g.get('vals', []), dtype=float) for g in groups]
    colors = [g.get('color') or _BAR_COLORS[i % len(_BAR_COLORS)] for i, g in enumerate(groups)]
    pairs_lists = [g.get('pairs', []) for g in groups]
    xs = np.arange(n_g, dtype=float)
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.tick_params(axis='x', labelsize=9); ax.tick_params(axis='y', labelsize=8)
    rng = np.random.default_rng(0)
    all_xpos = [np.full(a.size, xs[i]) + rng.normal(0, .05, a.size) for i, a in enumerate(arrays)]

    if view.violin:
        nonempty = [(xs[i], arr, colors[i]) for i, arr in enumerate(arrays) if arr.size >= 2]
        if nonempty:
            vp = ax.violinplot([a for _, a, _ in nonempty],
                               positions=[x for x, _, _ in nonempty],
                               showmedians=True, showextrema=True, widths=0.6)
            for i, pc in enumerate(vp.get('bodies', [])):
                pc.set_facecolor(nonempty[i][2]); pc.set_alpha(0.85)
                pc.set_edgecolor('#aaa' if dark else '#333'); pc.set_linewidth(0.8)
            med = vp.get('cmedians')
            if med:
                med.set_linewidth(2); med.set_color(fg)
    else:
        stats = [_mean_sem(a) for a in arrays]   # one pass: bar height and its error bar
        ax.bar(xs, [m for m, _ in stats], yerr=[e for _, e in stats],
               capsize=4, color=colors, edgecolor='#333', lw=0.8)

    if paired and n_g >= 2:
        gmaps = [{SelectionData.as_pair_key(p): (all_xpos[gi][j], float(arrays[gi][j]))
                  for j, p in enumerate(pairs_lists[gi])
                  if j < len(arrays[gi])} for gi in range(n_g)]
        common = set(gmaps[0])
        for gm in gmaps[1:]:
            common &= set(gm)
        for pk in common:
            ax.plot([gmaps[gi][pk][0] for gi in range(n_g)],
                    [gmaps[gi][pk][1] for gi in range(n_g)],
                    color='#888', alpha=0.1, lw=0.7, zorder=1)

    for i, (arr, xpos) in enumerate(zip(arrays, all_xpos)):
        if arr.size == 0:
            continue
        out_idx = set(StatsTestBackend.outlier_indices(groups[i], log))
        if view.outliers:
            ax.scatter(xpos, arr, s=14, color='#222', alpha=0.2, lw=0, zorder=2)
            for j in out_idx:
                if j < arr.size and j < len(pairs_lists[i]):
                    p = pairs_lists[i][j]
                    ax.annotate(f"{p.ref}-{p.tgt}", (xpos[j], float(arr[j])), fontsize=6,
                                ha='center', va='bottom', color='#CC0000',
                                xytext=(0, 3), textcoords='offset points')
        else:
            keep = [k for k in range(arr.size) if k not in out_idx]
            ax.scatter(xpos[keep], arr[keep], s=14, color='#222', alpha=0.2, lw=0, zorder=2)

    ax.grid(axis='y', alpha=0.25, lw=0.7)
    if one_sample:
        ax.axhline(0, color='#555', lw=0.8, ls='--', zorder=0)
    if not sig_pairs:
        return
    nm2x = {n: x for n, x in zip(names, xs)}
    starred = [(r['a'], r['b'], s) for r in sig_pairs
               if r['a'] in nm2x and r['b'] in nm2x and (s := _star(r['p_adj'])) is not None]
    if not starred:
        return
    ylo, ytop = ax.get_ylim()
    step = max(abs(ytop - ylo), 1e-6) * 0.13
    tick = step * 0.3
    starred.sort(key=lambda t: abs(nm2x[t[0]] - nm2x[t[1]]))
    ybase = ytop + step * 0.2
    for lv, (a_nm, b_nm, star) in enumerate(starred):
        x0, x1 = nm2x[a_nm], nm2x[b_nm]
        y = ybase + step * lv
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color='#333', lw=0.9, clip_on=False)
        ax.text((x0 + x1) / 2, y + tick * 0.2, star,
                ha='center', va='bottom', fontsize=8, color='#333', clip_on=False)
    ax.set_ylim(top=ybase + step * (len(starred) + 0.8))


def draw_stats_figure(fig, results: list, view: '_ViewConfig',
                      test_config: 'StatsTestConfig | None', *, dark: bool) -> None:
    """Fill *fig* with one panel per result; pure matplotlib, no widgets."""
    fig.clf()
    fig.patch.set_facecolor('#2b2b2b' if dark else 'white')
    if not results:
        return
    log = test_config.log_transform if test_config else False
    is_paired = ((test_config.test_type if test_config else '') == "Pairwise t-test"
                 or any(sr.is_paired for sr in results))
    n = len(results)
    _res_lbl = {'lowres': 'Lo-res', 'highres': 'Hi-res'}
    first_ax = None
    for i, sr in enumerate(results):
        groups = sr.plot_groups or []
        dtype = groups[0].get('data_type', '') if groups else ''
        title = dtype + (f" ({_res_lbl.get(sr.resolution, sr.resolution)})" if n > 1 else '')
        ax = fig.add_subplot(1, n, i + 1, sharey=first_ax)
        first_ax = first_ax if first_ax is not None else ax
        _draw_group_axes(ax, groups, title, view, log=log, dark=dark, paired=is_paired,
                         sig_pairs=pairwise_pvals(sr.res, groups) if view.sig_brackets else (),
                         one_sample=sr.is_one_sample)


def fit_figure_rect(fig, px_w: int, px_h: int, wh_ratio: str) -> None:
    """Figure fills the canvas; *wh_ratio* shapes the centred axes rect via margins."""
    w, h = max(px_w, 50), max(px_h, 50)
    dpi = fig.get_dpi()
    fig.set_size_inches(w / dpi, h / dpi, forward=False)
    try:
        wr, hr = (float(x) for x in wh_ratio.strip().split(':'))
        ratio = max(wr, 0.1) / max(hr, 0.1)
    except ValueError:
        ratio = 3.0
    fig_ar = w / h
    fx, fy = (ratio / fig_ar, 1.0) if fig_ar > ratio else (1.0, fig_ar / ratio)
    pad_x, pad_y = (1 - fx) / 2, (1 - fy) / 2
    fig.subplots_adjust(left=0.10 + pad_x * 0.9, right=1 - (0.03 + pad_x * 0.9),
                        bottom=0.12 + pad_y * 0.9, top=1 - (0.06 + pad_y * 0.9),
                        wspace=0.25)
