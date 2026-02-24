import seaborn as sns
import matplotlib.pyplot as plt
import os
import neuropy.plotting.probe as probe
import numpy as np
import warnings


def plot_ccg_panel(
    ax,
    ccg,
    ids,
    inds,
    window_size,
    bin_size,
    pval=None,
    pval_corrected=None,
    alpha=None,
    ccg_null=None,
    j_sig=None,
    segment_id=None,
    is_significant_pair=None,
    neuron_type=None,
    normalize_info=None,
    grayscale=False,
    min_lag=None,
    max_lag=None,
):
    """Single CCG plot into provided axis"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        bins = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)

        if grayscale:
            dark_gray = (0.2, 0.2, 0.2)
            light_gray = (0.8, 0.8, 0.8)
            colors = [dark_gray, light_gray]
        else:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]

        ax.bar(bins,
               ccg,
               width=bin_size,
               alpha=0.5,
               label="ccg",
               color=colors[0])
        if ccg_null is not None:
            ax.bar(bins,
                   ccg_null,
                   width=bin_size,
                   alpha=0.5,
                   label="ccg-smooth",
                   color=colors[1])

        if min_lag is not None and max_lag is not None:
            ax.axvspan(min_lag-bin_size/2, max_lag-bin_size/2, alpha=0.12, color='green', label='test window')

        ax.set_xlabel("Time (ms)")
        ylabel = "Count"
        if normalize_info is not None:
            ylabel = "Count " + normalize_info
        ax.set_ylabel(ylabel)

        sig_marker = '*' if is_significant_pair else ''
        type_str = f', type={neuron_type}' if neuron_type is not None else ''
        if normalize_info is None:
            ax.set_title(
                f"{segment_id}: ids={ids}, inds={inds}{type_str}{sig_marker}")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ticks_scaled = np.linspace(0, 1, 21)
        ticks_original = np.round(ticks_scaled, 2)
        ax2.set_yticks(ticks_scaled)
        ax2.set_yticklabels(ticks_original, fontsize=8)
        ax2.set_ylabel("p-value")

        if pval is not None:
            ax2.plot(bins, pval, label='p', alpha=0.3, color='gray')
        if j_sig is not None:
            ax2.plot(bins, j_sig, label='jitter sig', color='brown')
            # Set ticks to pval values on a correct scale
        if pval_corrected is not None:
            ax2.plot(bins,
                     pval_corrected,
                     label='corrected p',
                     alpha=0.2,
                     color='green')
        if alpha is not None:
            ax2.axhline(alpha,
                        label='alpha',
                        alpha=0.1,
                        color='red',
                        linestyle='--')
        ax2.legend(fontsize=8)
        sns.despine(ax=ax2)


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

    ax.bar(bins, ccg, width=bin_size * 1e3, alpha=0.7, label='CCG', color='steelblue')

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
    normalize_info: str = None,
    overlay_normalized: bool = False,
    normalized_ccg: np.ndarray = None,
    normalized_ccg_null: np.ndarray = None,
    fig=None,
):
    """Full figure: CCG (1 or 2 panels) +  waveforms (1 or 2 panels depending on format)"""
    width_ratios = [2, 1] if not overlay_normalized else [2, 2, 1]
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
        ids=ids,
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
        normalize_info=None,
        grayscale=False,
    )
    current_ax += 1

    if overlay_normalized:
        plot_ccg_panel(
            ax=axs[current_ax],
            ccg=normalized_ccg,
            ids=ids,
            inds=inds,
            window_size=window_size,
            bin_size=bin_size,
            pval=None,
            pval_corrected=None,
            alpha=None,
            ccg_null=normalized_ccg_null,
            j_sig=None,
            segment_id=segment_id,
            is_significant_pair=is_significant_pair,
            neuron_type=neuron_types,
            normalize_info=normalize_info,
            grayscale=True,
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
                  has_skips=None,
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
    if has_skips:
        ylabel += "\nremoving outliers"
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
