import seaborn as sns
import matplotlib.pyplot as plt
import os
import neuropy.plotting.probe as probe
import numpy as np

def plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, 
                   pval=None, ccg_null=None, j_sig=None,segment_id=None):
    """Single CCG plot into provided axis"""
    bins = np.arange(-window_size / 2, window_size / 2 + bin_size, bin_size)

    ax.bar(bins, ccg, width=bin_size, alpha=0.5, label="ccg")
    if ccg_null is not None:
        ax.bar(bins, ccg_null, width=bin_size, alpha=0.5, label="ccg-smooth")
    
    ax2 = ax.twinx()
    ylim=ax.get_ylim()[1]*0.8
    if pval is not None:
        pval_scale = pval/pval.max() * ylim
        ax2.plot(bins, pval_scale, label='p',alpha=0.3, color='gray')
    if j_sig is not None:
        j_sig_scale = j_sig/j_sig.max() * ylim
        ax2.plot(bins, j_sig_scale, label='jitter significance')
    # Set ticks to pval values on a correct scale
    ticks_scaled = np.linspace(0, ylim, len(ax.get_yticks()))  # positions in scaled axis
    ticks_original = np.round(ticks_scaled /ylim*pval.max(), 2) #TODO sometimes p val looks weird, is it a problem w p val calculation?
    ax2.set_yticks(ticks_scaled)
    ax2.set_yticklabels(ticks_original)
    ax2.set_ylabel("p-value")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    X, Y = ids; x, y = inds
    ax.set_title(f"CCG{segment_id}, neuron_ids=[{X},{Y}], indices=[{x},{y}]")
    ax.legend()
    sns.despine(ax=ax)
    sns.despine(ax=ax2)


def plot_ccg_figure(ccg, ids, inds, neuron_types, waveforms, 
                    window_size, bin_size, pval=None, ccg_null=None, j_sig=None, 
                    shank_ids=None,
                    frates_all=None, frates_cut=None, n_shanks=None, ch_per_shank=None,
                    discarded_channels=None,
                    show=True, save=False, plotdir=None,
                    waveform_plot_type="channel",
                    segment_id=None):
    """Full figure: CCG + 2 waveforms"""
    if waveform_plot_type=='channel':
        fig, axs = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

    # labels = ['ref', 'target']

    plot_ccg_panel(axs[0], ccg, ids, inds, window_size, bin_size, pval, ccg_null, j_sig,segment_id)
    if waveform_plot_type=='channel' and shank_ids is not None:
        def get_filled_waveforms(shank_id,wf):
            channel_ids = ch_per_shank*shank_id+np.arange(ch_per_shank)
            mask = ~np.isin(channel_ids, discarded_channels)
            start = ch_per_shank*shank_id-np.sum(discarded_channels<16*shank_id)
            length = np.sum(mask,axis=0)
            clean = np.full((ch_per_shank,wf.shape[-1]),np.nan)
            clean[mask]=wf[start:start+length]
            return clean

        ref_waveform = get_filled_waveforms(shank_ids[0],waveforms[0])
        tgt_waveform = get_filled_waveforms(shank_ids[1],waveforms[1])

        xlabel = ""
        if frates_all is not None:
            xlabel += f"ref {frates_all[0]:.2f}Hz | tgt {frates_all[1]:.2f} all \n"
        if frates_cut is not None:
            xlabel += f"ref {frates_cut[0]:.2f}Hz | tgt {frates_cut[1]:.2f} cut "
        axs[1] = probe.plot_waveform_on_channel(ref_waveform, shank_ids[0], 
                                                tgt_waveform, shank_ids[1], 
                                                footnote=xlabel, amplitude_limit=True,
                                                ax=axs[1],
                                                color='green' if shank_ids[0]!=shank_ids[1] else 'orange')
        sns.despine(ax=axs[1])
    else:
        for i in range(2):
            axs[1+i] = probe.plot_waveform(axs[1+i], waveforms[i], neuron_types[i], ids[i],
                                frates_all[i] if frates_all is not None else None,
                                frates_cut[i] if frates_cut is not None else None,
                                n_shanks=n_shanks,ch_per_shank=ch_per_shank,
                                discarded_channels=discarded_channels)

    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
        assert os.path.exists(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png") #TODO why do we need this?
        plt.close(fig)
    if show:
        plt.show()
        plt.close(fig)
    return fig


def plot_ccg_only(ccg, ids, inds, window_size, bin_size, pval=None, ccg_null=None, j_sig=None, 
                  show=True, save=False, plotdir=None):
    """Save only the CCG plot without waveforms"""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_ccg_panel(ax, ccg, ids, inds, window_size, bin_size, pval, ccg_null, j_sig)
    
    fig.tight_layout()
    if save and plotdir:
        fig.savefig(f"{plotdir}/ccg-inds{inds[0]}-{inds[1]}.png")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_connection_strength(key,n_segments_total,
                             pairs, x_coords, plot_data, significant,
                             n_segments_threshold=0,
                    norm_by_n_sess=False,
                    norm_by_total_strength=False,
                    zero_first_timepoint=False,
                    show_legend=False,
                    skips=None,
                    save=False,root=None,
                    legend_column_size=25):
        # TODO  n_segments_total needs to be per pair for spike count chunking
        # TODO  x ticks need to be aligned for spike count chunking
            # x_ticks = list(np.arange(13))
            # plt.xticks(x_ticks,x_ticks)

        n_significant=np.sum(significant,axis=1,keepdims=True)
        plt.figure()
        if pairs.shape[0]==0: 
            print(f"{key}: No pairs fit the criteria min_n_segment={n_segments_threshold}, nothing is plotted")
            return
        
        # Modifications to connection strength
        ylabel = "connection strength"
        if skips is not None:
            ylabel+="\nremoving outliers"
        if norm_by_total_strength:
            plot_data/=np.nansum(plot_data,axis=1,keepdims=True)
            ylabel=ylabel+" \nnormalized by total strength"
        if norm_by_n_sess: # normalize by the inverse of number of sessions where this pair appeared
            plot_data=plot_data*n_significant/n_segments_total
            ylabel=ylabel+" \n(normalized by number of sessions)"
        if zero_first_timepoint:
            # dmax = np.nanmax(plot_data,axis=1,keepdims=True)
            # dmin = np.nanmin(plot_data,axis=1,keepdims=True)
            plot_data= (plot_data-plot_data[:,0:1])
            ylabel=ylabel+" \naligning the first timepoint"
        colors = plt.cm.hsv(np.linspace(0, 1, plot_data.shape[0]))
        legend_keys = []
        
        max_pairs=np.max(plot_data,axis=1).argsort()[-5:][::-1]
        min_pairs=np.min(plot_data,axis=1).argsort()[:5]
        print("max",pairs[max_pairs],"min",pairs[min_pairs])

        x_coords = x_coords or np.full(pairs.shape[0],None)
        for i, (pair, x, y, c, sig) in enumerate(zip(pairs,x_coords,plot_data,colors,significant)):
            x = list(np.arange(n_segments_total)) if x is None else x
            plt.plot(x,y,c=c,alpha=0.3)  # normalized
            plt.scatter(x[sig], y[sig], s=8, c=c,label="_nolegend_")
            if show_legend: legend_keys.append(f"{i}:{pair}")
        plt.title(f"{key}")
        plt.xlabel("time segment")
        plt.xticks(np.arange(n_segments_total),np.arange(n_segments_total))
        plt.ylabel(ylabel)
        if show_legend: 
            # spacing
            ncol = 1+int(i//legend_column_size)
            i_per_col=i//ncol
            offset = -.3-.5*(i_per_col/legend_column_size)
            plt.legend(legend_keys,loc='right', bbox_to_anchor=(1, offset), ncol=ncol)
        
        if save:
            assert os.path.isdir(os.path.expanduser(root))
            plt.savefig(f"{os.path.expanduser(root)}/{key}.png", bbox_inches='tight')
        else:
            plt.show()