from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

import neuropy.plotting.probe as probe


def plot_ccg_panel(
    ax,
    ccg,
    shank_ids,
    inds,
    window_size,
    bin_size,
    pval=None,
    pval_corrected=None,
    alpha=None,
    ccg_null=None,
    j_ccg=None,
    j_pval=None,
    j_sig=None,
    segment_id=None,
    is_significant_pair=None,
    neuron_type=None,
    normalize_info=None,
    grayscale=False,
    min_lag=None,
    max_lag=None,
    acg_ref=None,
    acg_tgt=None,
    acg_yscale_ref=1.0,
    acg_yscale_tgt=1.0,
    acg_match_ccg=False,
    show_ccg=True,
    plot_style='bar',
    line_ccg=False,
    line_baseline=False,
    line_ref=False,
    line_tgt=False,
    line_jitter=False,
    conn_strength_baseline=None,
    ccg_color=None,
    baseline_color=None,
    ccg_alpha=None,
    baseline_alpha=None,
    cs_shade_color=None,
    show_legend=True,
    show_test_window=None,
    test_window_color=None,
    test_window_alpha=None,
    pval_line_color=None,
    alpha_line_color=None,
    title_show_shanks=True,
    title_show_inds=True,
    title_show_type=True,
    title_show_seg=True,
    title_show_norm_details=True,
    title_show_session=False,
    title_session_label="",
    xticks_ms=None,
    mirror_xticks=True,
    min_text_size=None,
):
    """Single CCG plot into provided axis.

    Per-item line flags (line_ccg, line_baseline, line_ref, line_tgt)
    control whether each histogram renders as a step outline
    (ax.plot with drawstyle='steps-mid') or a filled bar plot.
    The legacy ``plot_style`` parameter sets the default for all items
    when ``'line'``, but per-item flags take precedence.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Convert from seconds to milliseconds for display
        bins_s = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)
        bins = bins_s * 1000  # ms
        bin_w = bin_size * 1000  # ms
        if grayscale:
            dark_gray = (0.2, 0.2, 0.2)
            light_gray = (0.8, 0.8, 0.8)
            _prop = [dark_gray, light_gray]
        else:
            _prop = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
        _ccg_color      = ccg_color      if ccg_color      else _prop[0]
        _base_color     = baseline_color if baseline_color else _prop[1]
        _ccg_alpha      = ccg_alpha      if ccg_alpha      is not None else 0.5
        _base_alpha     = baseline_alpha if baseline_alpha is not None else 0.3
        _cs_shade_color = cs_shade_color if cs_shade_color else '#1a6b2e'

        # Resolve per-item line mode (legacy plot_style as fallback)
        _legacy_line = (plot_style == 'line')
        _ccg_line = line_ccg or _legacy_line
        _baseline_line = line_baseline or _legacy_line

        # Always set y-axis range based on CCG data so hiding CCG
        # doesn't zoom into the baseline alone.  When data contains negative
        # values (e.g. baseline-subtracted), show full range with 10% buffer.
        ccg_min = float(np.min(ccg))
        ccg_max = float(np.max(ccg))
        rng = max(ccg_max - ccg_min, 1e-12)
        buf = rng * 0.1
        y_lo = (ccg_min - buf) if ccg_min < 0 else 0
        y_hi = (ccg_max + buf) if ccg_max > 0 else 1
        ax.set_ylim(y_lo, y_hi)
        # set CCG transparency here
        if show_ccg:
            if _ccg_line:
                ax.plot(bins, ccg, drawstyle='steps-mid', label="CCG", alpha=_ccg_alpha,
                        color=_ccg_color, linewidth=1.5)
            else:
                ax.bar(bins, ccg, width=bin_w, alpha=_ccg_alpha, label="CCG", color=_ccg_color)
        if ccg_null is not None:
            if _baseline_line:
                ax.plot(bins, ccg_null, drawstyle='steps-mid', alpha=_base_alpha,
                        label="Baseline", color=_base_color, linewidth=1.5, linestyle='--')
            else:
                ax.bar(bins, ccg_null, width=bin_w, alpha=_base_alpha,
                       label="Baseline", color=_base_color)

        if conn_strength_baseline is not None and min_lag is not None and max_lag is not None:
            min_lag_ms = min_lag * 1000
            max_lag_ms = max_lag * 1000
            bl = np.asarray(conn_strength_baseline)
            if len(bl) == len(bins):
                # Select bins whose bar body [center-bin_w/2, center+bin_w/2]
                # overlaps [min_lag_ms, max_lag_ms] — matches axvspan geometry exactly.
                mask = (bins > min_lag_ms - bin_w / 2 - 1e-9) & (bins < max_lag_ms + bin_w / 2 + 1e-9)
                bottoms = bl[mask]
                heights = np.maximum(ccg[mask] - bottoms, 0)
                ax.bar(bins[mask], heights, width=bin_w, bottom=bottoms,
                       color=_cs_shade_color, alpha=1.0, label='Conn strength', zorder=3)

        _jitter_line = line_jitter or _legacy_line
        if j_ccg is not None:
            if _jitter_line:
                ax.plot(bins, j_ccg, drawstyle='steps-mid', alpha=0.6, label="jitter avg",
                        color="plum", linewidth=1.2)
            else:
                ax.bar(bins, j_ccg, width=bin_w, alpha=0.4,
                       label="jitter avg", color="plum")

        # Correlogram overlays — each gets its own right-side y-axis
        # ACGs render as hollow bar outlines by default; per-item line flag
        # switches to step outline (steps-mid)
        _acg_axis_offset = 0.14  # start past p-value axis (at 1.0)
        for acg_data, acg_color, acg_label, acg_scale, acg_line in [
            (acg_ref, '#007434', 'ACG ref', acg_yscale_ref, line_ref),
            (acg_tgt, '#9638AB', 'ACG tgt', acg_yscale_tgt, line_tgt),
        ]:
            if acg_data is None:
                continue
            ax_acg = ax.twinx()
            ax_acg.spines['right'].set_position(('axes', 1.0 + _acg_axis_offset))
            if acg_line:
                ax_acg.plot(bins, acg_data, drawstyle='steps-mid', alpha=0.6,
                            color=acg_color, linewidth=1.2, label=acg_label)
            else:
                ax_acg.bar(bins, acg_data, width=bin_w, alpha=0.4,
                           color=acg_color, label=acg_label)
            if acg_match_ccg:
                # Match CCG y-axis exactly
                ccg_ylim = ax.get_ylim()
                ax_acg.set_ylim(ccg_ylim)
            else:
                raw_max = np.max(acg_data) if np.max(acg_data) > 0 else 1
                scale = max(acg_scale, 0.01)
                ax_acg.set_ylim(0, raw_max * 1.1 / scale)
            ax_acg.set_ylabel(acg_label, color=acg_color, fontsize=8)
            ax_acg.tick_params(axis='y', colors=acg_color, labelsize=7)
            ax_acg.spines['right'].set_color(acg_color)
            for sp in ('top', 'bottom', 'left'):
                ax_acg.spines[sp].set_visible(False)
            _acg_axis_offset += 0.12

        # show_test_window=None means legacy: show whenever lags are provided.
        _draw_span = (show_test_window if show_test_window is not None
                      else (min_lag is not None and max_lag is not None))
        if _draw_span and min_lag is not None and max_lag is not None:
            ml_ms = min_lag * 1000
            xl_ms = max_lag * 1000
            # Span covers all tested bins: centers from min_lag to max_lag,
            # each bar extends ±bin_w/2 around its center
            _tw_color = test_window_color if test_window_color else 'green'
            _tw_alpha = test_window_alpha if test_window_alpha is not None else 0.12
            ax.axvspan(ml_ms - bin_w/2, xl_ms + bin_w/2, alpha=_tw_alpha,
                       color=_tw_color, label=f'Test window ({ml_ms:.0f}–{xl_ms:.0f} ms)')

        ax.set_xlabel("Time (ms)")
        if normalize_info is not None:
            if title_show_norm_details:
                ylabel = "Count " + normalize_info
            else:
                ylabel = "Count (normalized)"
        else:
            ylabel = "Count"
        ax.set_ylabel(ylabel)

        # Set explicit ms ticks: 0 always, plus standard landmarks within range
        half_w_ms = window_size * 1000 / 2
        candidate_ticks = [0, 1, -1, 3, -3, 5, -5, 10, -10]
        xticks = sorted(t for t in candidate_ticks if abs(t) <= half_w_ms)
        ax.set_xticks(xticks)

        title_parts = []
        if title_show_session and title_session_label:
            title_parts.append(title_session_label)
        if title_show_seg and segment_id:
            title_parts.append(f"{segment_id}:")
        if title_show_shanks and shank_ids is not None:
            shank_str = ' '.join(str(x) for x in shank_ids)
            title_parts.append(f"shank=({shank_str})")
        if title_show_inds and inds is not None:
            inds_str = ' '.join(str(x) for x in inds)
            title_parts.append(f"inds=({inds_str})")
        if title_show_type and neuron_type is not None:
            type_str = '->'.join(str(x) for x in neuron_type)
            title_parts.append(type_str)
        ax.set_title(', '.join(title_parts))
        sns.despine(ax=ax)

        _has_pval = any(x is not None for x in (pval, pval_corrected, j_pval, j_sig))
        if _has_pval:
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            ticks_scaled = np.linspace(0, 1, 21)
            ticks_original = np.round(ticks_scaled, 2)
            ax2.set_yticks(ticks_scaled)
            ax2.set_yticklabels(ticks_original, fontsize=8)
            ax2.set_ylabel("p-value")

            if pval is not None:
                ax2.plot(bins, pval, label='p (EranConv)', alpha=0.6,
                         color=pval_line_color or 'orange')
            if j_sig is not None:
                ax2.plot(bins, j_sig, label='jitter sig',
                         color=pval_line_color or 'brown')
            if pval_corrected is not None:
                ax2.plot(bins, pval_corrected, label='corrected p', alpha=0.4,
                         color=pval_line_color or 'green')
            if j_pval is not None:
                j_pval = np.asarray(j_pval)
                if j_pval.ndim == 0:
                    # Scalar: single empirical p-value (legacy)
                    ax2.axhline(float(j_pval), label=f'jitter p={float(j_pval):.3f}',
                                color=pval_line_color or 'purple',
                                alpha=0.85, linewidth=1.5, linestyle='--')
                else:
                    # Per-bin p-values
                    ax2.plot(bins, j_pval, label='jitter p-corrected',
                             alpha=0.6, color=pval_line_color or 'purple')
            if alpha is not None:
                ax2.axhline(alpha, label=f'α={alpha}', alpha=0.8,
                            color=alpha_line_color or 'red', linestyle='--', linewidth=1.5)
            if show_legend:
                # Combine handles from both axes into one legend on ax2
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2, fontsize=8)
            sns.despine(ax=ax2)
        elif show_legend:
            ax.legend(fontsize=8)


def plot_ccg_simple(ccg, ccg_null=None, bin_size=1e-3, duration=20e-3,
                    min_lag=1e-3, max_lag=3e-3, inds=None, segment=None,
                    ax=None):
    """
    Lightweight single CCG plot for UI display.
    Uses same bin convention as plot_ccg_panel (bins and ccg same length).

    Parameters
    ----------
    ccg, ccg_null : np.ndarray  shape [n_bins]
    bin_size, duration, min_lag, max_lag : float  in seconds
    inds : tuple (ref_ind, tgt_ind), optional
    segment : int, optional
    ax : matplotlib.axes.Axes, optional
        If provided, draw into this axes (for embedding in existing figures).
        If None, a new figure is created and returned.

    Returns
    -------
    fig : matplotlib.figure.Figure  (only when ax is None)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    # Same bin generation as plot_ccg_panel (window_size=duration, bins in ms)
    bins = np.arange(-duration / 2, duration / 2 + bin_size, bin_size) * 1e3

    ax.bar(bins, ccg, width=bin_size * 1e3, alpha=0.5, label='CCG', color='steelblue')

    if ccg_null is not None:
        ax.bar(bins, ccg_null, width=bin_size * 1e3, alpha=0.5,
               label='Expected', color='orange')

    if min_lag is not None and max_lag is not None:
        ax.axvspan(min_lag * 1e3, max_lag * 1e3, alpha=0.1, color='green',
                   label='Test window')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Count')

    title = f"Pair [{inds[0]}, {inds[1]}]" if inds is not None else ""
    if segment is not None:
        title += f" - Segment {segment + 1}"
    ax.set_title(title)

    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    if return_fig:
        plt.tight_layout()
        return fig


def plot_ccg_figure(
    ccg: np.ndarray,
    ids: np.ndarray,
    inds: np.ndarray,
    neuron_types: np.ndarray,
    waveforms: np.ndarray,
    window_size: float,
    bin_size: float,
    pval: np.ndarray = None,
    pval_corrected: np.ndarray = None,
    alpha: float = None,
    ccg_null: np.ndarray = None,
    j_sig: np.ndarray = None,
    is_significant_pair: bool = None,
    shank_ids: tuple = None,
    frates_all: tuple = None,
    frates_cut: tuple = None,
    n_shanks: int = None,
    ch_per_shank: int = None,
    discarded_channels: np.ndarray = None,
    show: bool = True,
    save: bool = False,
    save_path: str = None,
    waveform_plot_type="probe",  # or "all_channels"
    segment_id: int = None,
    fig=None,
):
    """Full figure: CCG (1 or 2 panels) +  waveforms (1 or 2 panels depending on format)"""
    width_ratios = [2, 1]
    if waveform_plot_type == 'all_channels':
        width_ratios += [1]

    if fig is None:
        fig, axs = plt.subplots(1,
                                len(width_ratios),
                                figsize=(np.round(np.sum(width_ratios) * 2.5), 5),
                                gridspec_kw={'width_ratios': width_ratios})
    else:
        fig.clear()
        axs = fig.subplots(1, len(width_ratios), gridspec_kw={'width_ratios': width_ratios})

    current_ax = 0
    plot_ccg_panel(
        ax=axs[current_ax],
        ccg=ccg,
        shank_ids=ids,
        inds=inds,
        window_size=window_size,
        bin_size=bin_size,
        pval=pval,
        pval_corrected=pval_corrected,
        alpha=alpha,
        ccg_null=ccg_null,
        j_sig=j_sig,
        segment_id=segment_id,
        is_significant_pair=is_significant_pair,
        neuron_type=neuron_types,
        grayscale=False,
    )
    current_ax += 1

    if waveform_plot_type == 'probe' and shank_ids is not None:

        def get_filled_waveforms(shank_id, wf):
            if wf.ndim == 1:
                clean = np.tile(wf, (ch_per_shank, 1))
            else:
                channel_ids = ch_per_shank * shank_id + np.arange(ch_per_shank)
                mask = ~np.isin(channel_ids, discarded_channels)
                start = ch_per_shank * shank_id - np.sum(
                    discarded_channels < 16 * shank_id)
                length = np.sum(mask, axis=0)
                clean = np.full((ch_per_shank, wf.shape[-1]), np.nan)
                clean[mask] = wf[start:start + length]
            return clean

        ref_waveform = get_filled_waveforms(shank_ids[0], waveforms[0])
        tgt_waveform = get_filled_waveforms(shank_ids[1], waveforms[1])

        xlabel = ""
        if frates_all is not None:
            xlabel += f"ref {frates_all[0]:.2f}Hz | tgt {frates_all[1]:.2f} all \n"
        if frates_cut is not None:
            xlabel += f"ref {frates_cut[0]:.2f}Hz | tgt {frates_cut[1]:.2f} cut \n"
        axs[current_ax] = probe.plot_waveform_on_channel(
            ref_waveform,
            shank_ids[0],
            tgt_waveform,
            shank_ids[1],
            footnote=xlabel,
            amplitude_limit=True,
            ax=axs[current_ax],
            color='green' if shank_ids[0] != shank_ids[1] else 'orange')
        sns.despine(ax=axs[current_ax])
    else:
        for i in range(2):
            axs[current_ax + i] = plot_waveform_panel(
                axs[current_ax + i],
                waveforms[i],
                neuron_types[i],
                ids[i],
                frate_all=frates_all[i] if frates_all is not None else None,
                frates_cut=frates_cut[i] if frates_cut is not None else None,
                n_shanks=n_shanks,
                ch_per_shank=ch_per_shank,
                discarded_channels=discarded_channels)
        sns.despine(ax=axs[current_ax + i])

    fig.tight_layout()
    if save:
        try:
            fig.savefig(save_path)
            assert os.path.exists(save_path)  #TODO why do we need this?
        except:
            print(f"failed to save to {save_path}")
        plt.close(fig)
    if show:
        plt.show()
        plt.close(fig)
    return fig


def plot_waveform_panel(ax,
                        waveform,
                        neuron_type,
                        neuron_id,
                        frate_all=None,
                        frates_cut=None,
                        n_shanks=None,
                        ch_per_shank=None,
                        discarded_channels=None):
    """Single waveform panel into provided axis"""
    n_shanks = n_shanks or 12
    ch_per_shank = ch_per_shank or 16  # TODO
    max_ch = waveform.shape[0]
    ax.imshow(waveform.astype(float))
    ax.set_title(f"{neuron_type}{neuron_id}")
    xlabel = ""
    if frate_all is not None:
        xlabel += f"{frate_all:.2f}Hz all "
    elif frates_cut is not None:
        xlabel += f"{frates_cut:.2f}Hz cut "
    ax.set_xlabel(xlabel)

    edges = (np.array(range(n_shanks)) + 1) * ch_per_shank + 1
    if discarded_channels is not None:
        shanks = discarded_channels // ch_per_shank
        edges = edges - np.cumsum(np.histogram(shanks, np.arange(n_shanks))[0])

    for k in edges:
        ax.axhline(k, c='w', alpha=0.5, linestyle='dashed')

    return ax


def plot_strength(key,
                  pairs,
                  significant,
                  n_segments,
                  plot_data,
                  n_segments_threshold=None,
                  norm_by_n_sess=False,
                  norm_by_total_strength=False,
                  zero_first_timepoint=False,
                  show_legend=False,
                  save=False,
                  root=None,
                  debug=False):
    # show all pairs by default
    n_segments_threshold = n_segments_threshold if n_segments_threshold is not None else 0
    plt.figure()

    n_significant = np.sum(significant, axis=1, keepdims=True)
    if pairs.shape[0] == 0:
        print(
            f"{key}: No pairs fit the criteria min_n_segment={n_segments_threshold}, nothing is plotted"
        )
        return

    ylabel = "connection strength"
    if norm_by_total_strength:
        plot_data /= np.nansum(plot_data, axis=1, keepdims=True)
        ylabel = ylabel + " \nnormalized by total strength"
    if norm_by_n_sess:  # normalize by the inverse of number of sessions where this pair appeared
        plot_data = plot_data * n_significant / n_segments
        ylabel = ylabel + " \n(normalized by number of sessions)"
    if zero_first_timepoint:
        # dmax = np.nanmax(plot_data,axis=1,keepdims=True)
        # dmin = np.nanmin(plot_data,axis=1,keepdims=True)
        plot_data = (plot_data - plot_data[:, 0:1])
        ylabel = ylabel + " \naligning the first timepoint"
    colors = plt.cm.hsv(np.linspace(0, 1, plot_data.shape[0]))
    legend_keys = []

    if debug:
        max_pairs = np.max(plot_data, axis=1).argsort()[-3:][::-1]
        min_pairs = np.min(plot_data, axis=1).argsort()[:3]
        print("max", pairs[max_pairs], "min", pairs[min_pairs])
    for i, (pair, v, c,
            sig) in enumerate(zip(pairs, plot_data, colors, significant)):
        plt.plot(v, c=c, alpha=0.3)  # normalized
        x_sig = np.where(sig)[0]
        plt.scatter(x_sig, v[x_sig], s=8, c=c, label="_nolegend_")
        if show_legend:
            legend_keys.append(f"{i}:{pair}")
    plt.title(f"{key}")
    plt.xlabel("time segment")
    plt.xticks(np.arange(n_segments), np.arange(n_segments))
    plt.ylabel(ylabel)
    if show_legend:
        # spacing
        ncol = 1 + int(i // 25)
        i_per_col = i // ncol
        offset = -.3 - .5 * (i_per_col / 25)
        plt.legend(legend_keys,
                   loc='right',
                   bbox_to_anchor=(1, offset),
                   ncol=ncol)

    if save:
        assert os.path.isdir(os.path.expanduser(root))
        plt.savefig(f"{os.path.expanduser(root)}/{key}.png",
                    bbox_inches='tight')
    else:
        plt.show()

    # mean, pvals = ttest_1samp(plot_data,0,axis=0)
    # print("pvals",pvals[1:],'threshold',0.05/len(pvals[1:]),"\n")
    # print("mean values",mean[1:],"\n")


def plot_network(self):
    pass


# def plot_connection_strength(key,n_segments_total,
#                              pairs, x_coords, plot_data, significant,
#                              n_segments_threshold=0,
#                     norm_by_n_sess=False,
#                     norm_by_total_strength=False,
#                     zero_first_timepoint=False,
#                     show_legend=False,
#                     skips=None,
#                     save=False,root=None,
#                     legend_column_size=25):
#         # TODO  n_segments_total needs to be per pair for spike count chunking
#         # TODO  x ticks need to be aligned for spike count chunking
#             # x_ticks = list(np.arange(13))
#             # plt.xticks(x_ticks,x_ticks)

#         n_significant=np.sum(significant,axis=1,keepdims=True)
#         plt.figure()
#         if pairs.shape[0]==0:
#             print(f"{key}: No pairs fit the criteria min_n_segment={n_segments_threshold}, nothing is plotted")
#             return

#         # Modifications to connection strength
#         ylabel = "connection strength"
#         if skips is not None:
#             ylabel+="\nremoving outliers"
#         if norm_by_total_strength:
#             plot_data/=np.nansum(plot_data,axis=1,keepdims=True)
#             ylabel=ylabel+" \nnormalized by total strength"
#         if norm_by_n_sess: # normalize by the inverse of number of sessions where this pair appeared
#             plot_data=plot_data*n_significant/n_segments_total
#             ylabel=ylabel+" \n(normalized by number of sessions)"
#         if zero_first_timepoint:
#             # dmax = np.nanmax(plot_data,axis=1,keepdims=True)
#             # dmin = np.nanmin(plot_data,axis=1,keepdims=True)
#             plot_data= (plot_data-plot_data[:,0:1])
#             ylabel=ylabel+" \naligning the first timepoint"
#         colors = plt.cm.hsv(np.linspace(0, 1, plot_data.shape[0]))
#         legend_keys = []

#         max_pairs=np.max(plot_data,axis=1).argsort()[-5:][::-1]
#         min_pairs=np.min(plot_data,axis=1).argsort()[:5]
#         print("max",pairs[max_pairs],"min",pairs[min_pairs])

#         x_coords = x_coords or np.full(pairs.shape[0],None)
#         for i, (pair, x, y, c, sig) in enumerate(zip(pairs,x_coords,plot_data,colors,significant)):
#             x = list(np.arange(n_segments_total)) if x is None else x
#             plt.plot(x,y,c=c,alpha=0.3)  # normalized
#             plt.scatter(x[sig], y[sig], s=8, c=c,label="_nolegend_")
#             if show_legend: legend_keys.append(f"{i}:{pair}")
#         plt.title(f"{key}")
#         plt.xlabel("time segment")
#         plt.xticks(np.arange(n_segments_total),np.arange(n_segments_total))
#         plt.ylabel(ylabel)
#         if show_legend:
#             # spacing
#             ncol = 1+int(i//legend_column_size)
#             i_per_col=i//ncol
#             offset = -.3-.5*(i_per_col/legend_column_size)
#             plt.legend(legend_keys,loc='right', bbox_to_anchor=(1, offset), ncol=ncol)

#         if save:
#             assert os.path.isdir(os.path.expanduser(root))
#             plt.savefig(f"{os.path.expanduser(root)}/{key}.png", bbox_inches='tight')
#         else:
#             plt.show()


def plot_spike_attribution_raster(
    fig,
    ref_spikes: np.ndarray,
    tgt_spikes: np.ndarray,
    ref_t: float,
    tgt_t: float,
    window: float,
    ref_label: str,
    tgt_label: str,
    pair_idx: int,
) -> None:
    """Draw a 2-row eventplot raster around one attributed spike pair.

    Parameters
    ----------
    fig : matplotlib Figure (cleared on entry)
    ref_spikes, tgt_spikes : full spike train arrays (seconds)
    ref_t, tgt_t : ref and tgt spike times for the selected pair
    window : half-width of the raster in seconds
    ref_label, tgt_label : y-axis labels (e.g. "Ref shank0ch3")
    pair_idx : 0-based index used in title
    """
    center = ref_t
    t0, t1 = center - window, center + window
    ref_win = ref_spikes[(ref_spikes >= t0) & (ref_spikes <= t1)]
    tgt_win = tgt_spikes[(tgt_spikes >= t0) & (tgt_spikes <= t1)]

    fig.clear()
    ax_ref = fig.add_subplot(211)
    ax_tgt = fig.add_subplot(212, sharex=ax_ref)

    if len(ref_win):
        ax_ref.eventplot([ref_win - center], lineoffsets=0,
                         linelengths=0.8, colors='#1565C0')
    ax_ref.axvline(0, color='#E53935', lw=1.5, ls='--', alpha=0.7)
    ax_ref.set_ylabel(ref_label, fontsize=9)
    ax_ref.set_yticks([])
    ax_ref.set_title(
        f"Spike pair #{pair_idx + 1}: ref={ref_t:.4f}s  tgt={tgt_t:.4f}s  "
        f"lag={(tgt_t - ref_t) * 1000:.2f}ms",
        fontsize=9)

    if len(tgt_win):
        ax_tgt.eventplot([tgt_win - center], lineoffsets=0,
                         linelengths=0.8, colors='#2E7D32')
    ax_tgt.axvline(tgt_t - center, color='#E53935', lw=1.5, ls='--', alpha=0.7)
    ax_tgt.set_ylabel(tgt_label, fontsize=9)
    ax_tgt.set_yticks([])
    ax_tgt.set_xlabel("Time relative to ref spike (s)", fontsize=9)
    ax_tgt.set_xlim(-window, window)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# RenderContext and sub-dataclasses (moved from neuropy.ui.ccg_renderer)
# ---------------------------------------------------------------------------

@dataclass
class JitterOverlay:
    j_ccg:    Optional[np.ndarray] = None
    j_pval:   Optional[np.ndarray] = None
    j_ccg_lo: Optional[np.ndarray] = None
    j_ccg_hi: Optional[np.ndarray] = None


@dataclass
class PlotStyle:
    """Export/preview style overrides — field names match plot_ccg_panel params exactly."""
    ccg_color:         Optional[str]   = None
    baseline_color:    Optional[str]   = None
    ccg_alpha:         Optional[float] = None
    baseline_alpha:    Optional[float] = None
    cs_shade_color:    Optional[str]   = None
    show_legend:       bool            = True
    xticks_ms:         Optional[list]  = None
    mirror_xticks:     bool            = True
    min_text_size:     Optional[float] = None
    show_test_window:  Optional[bool]  = None
    test_window_color: Optional[str]   = None
    test_window_alpha: Optional[float] = None
    pval_line_color:   Optional[str]   = None
    alpha_line_color:  Optional[str]   = None


@dataclass
class TitleConfig:
    """Title visibility flags — field names match plot_ccg_panel params exactly."""
    title_show_shanks:       bool = True
    title_show_inds:         bool = True
    title_show_type:         bool = True
    title_show_seg:          bool = True
    title_show_norm_details: bool = True
    title_show_session:      bool = False
    title_session_label:     str  = ''


@dataclass
class RenderContext:
    """Processed data and parameters for rendering one CCG PNG."""
    ccg:            np.ndarray
    ccg_null_plot:  Optional[np.ndarray]
    pval:           Optional[np.ndarray]
    pval_corrected: Optional[np.ndarray]
    jitter:          JitterOverlay
    acg_ref:         Optional[np.ndarray]
    acg_tgt:         Optional[np.ndarray]
    wf_peak_ms:      Optional[np.ndarray]
    wf_peak_amp:     Optional[np.ndarray]
    cs_baseline_arg: Optional[np.ndarray]
    window_size_eff: float
    bin_size_eff:    float
    alpha:           float
    norm_info:       Optional[str]
    seg_id_display:  str
    min_lag_plot:    Optional[float]
    max_lag_plot:    Optional[float]
    extend_on:       bool
    cs_annotation_lines: list
    inds:                tuple
    shank_ids:           tuple
    neuron_type:         tuple
    is_significant_pair: bool
    show_ccg:       bool
    line_ccg:       bool
    line_baseline:  bool
    line_ref:       bool
    line_tgt:       bool
    line_jitter:    bool
    acg_yscale_ref: float
    acg_yscale_tgt: float
    acg_match_ccg:  bool
    ylim:           Optional[tuple]
    style:  PlotStyle
    title:  TitleConfig
    dark_mode: bool = False


def _fill_waveform(wf_neuron, shank_id: int, ch_per_shank: int, discarded):
    """Expand a (possibly trimmed) per-neuron waveform to a full (ch_per_shank, T) array."""
    if wf_neuron.ndim == 1:
        return np.tile(wf_neuron, (ch_per_shank, 1))
    sid  = int(shank_id)
    disc = np.asarray(discarded, dtype=int) if discarded is not None else np.empty(0, dtype=int)
    channel_ids = ch_per_shank * sid + np.arange(ch_per_shank)
    mask   = ~np.isin(channel_ids, disc)
    start  = int(ch_per_shank * sid - np.sum(disc < ch_per_shank * sid))
    length = int(np.sum(mask))
    clean  = np.full((ch_per_shank, wf_neuron.shape[-1]), np.nan)
    clean[mask] = wf_neuron[start:start + length]
    return clean


def load_peak_waveform(ref: int, waveforms, peak_channels, shank_ids,
                       ch_per_shank: int, discarded):
    """Extract (t_ms, amp) for neuron *ref*'s peak-channel waveform. Returns (None, None) on failure."""
    if waveforms is None or peak_channels is None or shank_ids is None:
        return None, None
    try:
        peak_ch = int(peak_channels[ref])
        rs      = int(shank_ids[ref])
    except (IndexError, TypeError, ValueError):
        return None, None
    discarded_arr = None if discarded is None else np.asarray(discarded, dtype=int)
    if discarded_arr is not None and discarded_arr.size and np.isin(peak_ch, discarded_arr):
        return None, None
    local_idx = peak_ch - ch_per_shank * rs
    if not (0 <= local_idx < ch_per_shank):
        return None, None
    ref_full = _fill_waveform(waveforms[ref], rs, ch_per_shank, discarded_arr)
    tr = ref_full[local_idx]
    if not np.any(np.isfinite(tr)):
        return None, None
    n = int(tr.shape[0])
    return np.arange(n, dtype=float) - n // 2, np.asarray(tr, dtype=float)


def render_ccg_png(ctx: RenderContext, png_path: str, dpi: int = 100) -> None:
    """Create figure, call plot_ccg_panel, apply post-processing, save PNG."""
    fig = Figure(figsize=(7, 5))
    ax  = fig.add_subplot(111)

    plot_ccg_panel(
        ax                     = ax,
        ccg                    = ctx.ccg,
        shank_ids              = ctx.shank_ids,
        inds                   = ctx.inds,
        neuron_type            = ctx.neuron_type,
        window_size            = ctx.window_size_eff,
        bin_size               = ctx.bin_size_eff,
        pval                   = ctx.pval,
        pval_corrected         = ctx.pval_corrected,
        alpha                  = ctx.alpha,
        ccg_null               = ctx.ccg_null_plot,
        j_ccg                  = ctx.jitter.j_ccg,
        j_pval                 = ctx.jitter.j_pval,
        segment_id             = ctx.seg_id_display,
        is_significant_pair    = ctx.is_significant_pair,
        min_lag                = ctx.min_lag_plot,
        max_lag                = ctx.max_lag_plot,
        normalize_info         = ctx.norm_info,
        acg_ref                = ctx.acg_ref,
        acg_tgt                = ctx.acg_tgt,
        acg_yscale_ref         = ctx.acg_yscale_ref,
        acg_yscale_tgt         = ctx.acg_yscale_tgt,
        acg_match_ccg          = ctx.acg_match_ccg,
        show_ccg               = ctx.show_ccg,
        line_ccg               = ctx.line_ccg,
        line_baseline          = ctx.line_baseline,
        line_ref               = ctx.line_ref,
        line_tgt               = ctx.line_tgt,
        line_jitter            = ctx.line_jitter,
        conn_strength_baseline = ctx.cs_baseline_arg,
        **dataclasses.asdict(ctx.style),
        **dataclasses.asdict(ctx.title),
    )

    if ctx.extend_on:
        try:
            half_ms = float(ctx.window_size_eff) * 1000.0 / 2.0
            nice    = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            step    = next((s for s in nice if (2 * half_ms / s) <= 10.5), nice[-1])
            start   = -np.floor(half_ms / step) * step
            ticks   = np.arange(start, half_ms + 0.5 * step, step)
            if not np.any(np.isclose(ticks, 0.0)):
                ticks = np.sort(np.append(ticks, 0.0))
            ax.set_xticks(ticks)
        except Exception:
            pass

    if ctx.jitter.j_ccg_lo is not None and ctx.jitter.j_ccg_hi is not None:
        try:
            jlo = np.asarray(ctx.jitter.j_ccg_lo, dtype=float)
            jhi = np.asarray(ctx.jitter.j_ccg_hi, dtype=float)
            if len(jlo) == len(ctx.ccg) and len(jhi) == len(ctx.ccg):
                bs     = ctx.bin_size_eff
                ws     = ctx.window_size_eff
                bins_s = np.arange(-ws / 2, ws / 2 + bs, bs)
                bins   = bins_s * 1000.0
                edges  = np.append(bins - bs * 500.0, bins[-1] + bs * 500.0)
                x_step = np.repeat(edges, 2)[1:-1]
                for arr in (jlo, jhi):
                    ax.plot(x_step, np.repeat(arr, 2),
                            color='#C62828', linewidth=1.15,
                            alpha=0.9, linestyle=(0, (4, 3)), zorder=4)
        except Exception:
            pass

    if ctx.ylim is not None:
        ax.set_ylim(ctx.ylim)

    if ctx.style.xticks_ms:
        try:
            ticks = list(ctx.style.xticks_ms)
            if ctx.style.mirror_xticks:
                ticks = sorted(set(ticks + [-t for t in ticks]))
            ax.set_xticks(ticks)
        except Exception:
            pass

    if ctx.cs_annotation_lines:
        try:
            cur_xlabel = ax.get_xlabel() or ''
            ax.set_xlabel(cur_xlabel + '\n' + '\n'.join(ctx.cs_annotation_lines))
        except Exception:
            pass

    if ctx.style.min_text_size is not None:
        try:
            ms = float(ctx.style.min_text_size)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                if item.get_fontsize() < ms:
                    item.set_fontsize(ms)
        except Exception:
            pass

    if ctx.dark_mode:
        _bg, _fg, _sp = '#2b2b2b', 'white', '#666666'
        fig.set_facecolor(_bg)
        ax.set_facecolor(_bg)
        ax.tick_params(colors=_fg)
        ax.xaxis.label.set_color(_fg)
        ax.yaxis.label.set_color(_fg)
        ax.title.set_color(_fg)
        for sp in ax.spines.values():
            sp.set_edgecolor(_sp)
        for _child_ax in fig.get_axes():
            leg = _child_ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(_bg)
                leg.get_frame().set_edgecolor(_sp)
                for _lt in leg.get_texts():
                    _lt.set_color(_fg)

    fig.savefig(png_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def render_ccg_image(ctx: RenderContext, dpi: int = 100) -> 'np.ndarray':
    """Render CCG panel to a numpy RGBA array (no disk I/O)."""
    import io as _io
    buf = _io.BytesIO()
    render_ccg_png(ctx, buf, dpi=dpi)
    buf.seek(0)
    return mpimg.imread(buf)
