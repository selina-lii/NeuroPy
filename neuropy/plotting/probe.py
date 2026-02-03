import matplotlib.pyplot as plt
from ..core import ProbeGroup
import matplotlib.patches as patches
import numpy as np

def plot_probe(
    probe: ProbeGroup,
    annotate_channels=None,
    channel_id=True,
    disconnected=True,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        probe.x,
        probe.y,
        s=12,
        marker="o",
        color="gray",
        zorder=1,
        linewidths=0.5,
        alpha=0.5,
    )
    if channel_id:
        for x, y, chan_id in zip(probe.x, probe.y, probe.channel_id):
            ax.annotate(chan_id, (x, y), fontsize=8)

    if disconnected:
        ax.scatter(
            probe.get_disconnected.x,
            probe.get_disconnected.y,
            s=18,
            edgecolors="#FF5252",
            facecolors="none",
        )

    if annotate_channels is not None:
        prb_data = probe.to_dataframe().set_index("channel_id")
        for channel in annotate_channels:
            x, y = (
                prb_data.loc[[channel]].x.values[0],
                prb_data.loc[[channel]].y.values[0],
            )
            ax.scatter(x, y, s=30, edgecolors="g", facecolors="none", linewidths=2)

    # ax.axhline(probe.y_max + 10, probe.x_min, probe.x_max, lw=4)
    ax.axis("off")
    ax.set_title(f"Probe {probe.n_contacts}ch")

    return ax


def plot_waveform_on_channel(ref_waveform,ref_shank,target_waveform=None,target_shank=None,
                             color="orange",amplitude_limit=False,
                             footnote="",ax=None):
    # units are um (micron)
    # For Cambridge Neurotech F-8. Specs are from their brochure
    # and adjusted based on plotting effects.
    n_channels_per_side = 8
    vertical_eletrode_span = 225+75
    shank_width = 33.5 - 14
    interchannel_x = 16.5 - 5
    interchannel_y = 15
    eletrode_size_x = 11 - 5
    eletrode_size_y = 15
    tip_length = 50
    waveform_amp_limit=interchannel_y*3

    # Function description
    # draw a vertical_eletrode_span * shank_width box, but remove the lower left corner by covering with a white triange, 
    # to show a tilted tip. gray fill, no outline

    # draw 8 vertically lined boxes eletrode_size_x by eletrode_size_y on the left side of the large box
    # black outlines, white fill. interchannel_y apart vertically

    # draw 8 vertically lined boxes eletrode_size_x by eletrode_size_y on the right side of the large box,
    # offset them so each is centered in between two boxes on the left vertically,
    # the right side box array is shifted downwards vertically so the boxes are interleaved.
    # the left and right arrays are interchannel_x apart.
    # vertically lowest of all boxes should be right at where the box starts tapering
    # that is they should all be within the gray fill.

    # there are some constant added manually to shift things to look nicely

    # given waveform shaped (n_channels_per_side * 2, 2, window), put 2 lines side by side by the center of corresponding channel.
    # window will be about 50-100 pixels. the first line is black and closer to the center of the plot.
    # the second line is next to that in a given color 
    # the result is that these waveforms flank the illutration of the channel.
    # add channel number at the left edge of the whole plot from 1...n
    # on top of the channel numbers to the left of the gray-fill shank add text "ch #"

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Draw shank (gray)
    shank = patches.Rectangle(
        (-shank_width / 2, 0),
        shank_width,
        vertical_eletrode_span,
        facecolor="lightgray",
        edgecolor="none",
    )
    ax.add_patch(shank)

    # Draw tapered tip (white triangle)
    tip = patches.Polygon(
        [
            (-shank_width / 2-1, tip_length),
            (shank_width / 2-1, 0-1),
            (-shank_width / 2-1, 0-1),
        ],
        facecolor="white",
        edgecolor="none",
    )
    ax.add_patch(tip)

    # Left electrodes
    left_x = -interchannel_x / 2 
    for i in range(n_channels_per_side):
        y = i * (eletrode_size_y + interchannel_y)+50
        rect = patches.Rectangle(
            (left_x - eletrode_size_x / 2, y),
            eletrode_size_x,
            eletrode_size_y,
            facecolor="#eeeeee",
            edgecolor="black",
            lw=1.2,
        )
        ax.add_patch(rect)

    # Right electrodes (interleaved)
    right_x = interchannel_x / 2 
    for i in range(n_channels_per_side):
        y = i * (eletrode_size_y + interchannel_y) - (eletrode_size_y + interchannel_y) / 2+50
        rect = patches.Rectangle(
            (right_x - eletrode_size_x / 2, y),
            eletrode_size_x,
            eletrode_size_y,
            facecolor="#eeeeee",
            edgecolor="black",
            lw=1.2,
        )
        ax.add_patch(rect)

    # waveforms
    window = ref_waveform.shape[1]
    y_scale = 8  # scaling factor for waveform amplitude
    x_scale = 2  # scaling factor for waveform plotting width
    x_offset = shank_width / 2 + 15  # horizontal distance from shank

    for ch in range(n_channels_per_side * 2):
        if ch < n_channels_per_side:
            y_center = 50 + ch * (eletrode_size_y + interchannel_y) + eletrode_size_y / 2
        else:
            i = ch - n_channels_per_side
            y_center = 50 + i * (eletrode_size_y + interchannel_y) + eletrode_size_y / 2 - (eletrode_size_y + interchannel_y) / 2

        # Scale waveform for display
        wf1 = ref_waveform[ch] * y_scale
        if amplitude_limit:
            amp=wf1.max()-wf1.min()
            if amp>waveform_amp_limit: wf1=wf1/amp*waveform_amp_limit

        # Left or right of shank
        if ch < n_channels_per_side:
            x_center = -x_offset-shank_width
        else:
            x_center = x_offset-5
        ax.plot(x_center + np.zeros_like(wf1)+np.arange(window)/x_scale, y_center + wf1,  color="black", lw=1.2)

        if target_waveform is not None:
            # Scale waveform for display
            wf2 = target_waveform[ch] * y_scale
            if amplitude_limit:
                amp=wf2.max()-wf2.min()
                if amp>waveform_amp_limit: wf2=wf2/amp*waveform_amp_limit
            # Left or right of shank
            if ch < n_channels_per_side:
                x_center_ref = x_center-window/x_scale-2
            else:
                x_center_ref = x_center+window/x_scale+2
            ax.plot(x_center_ref + np.zeros_like(wf2)+np.arange(window)/x_scale, y_center + wf2, color=color, lw=1.2)

    wavewidth = window/x_scale*2

    # Channel numbers
    for ch in range(n_channels_per_side*2):
        y_center = tip_length + (n_channels_per_side*2 - ch-2) * interchannel_y + eletrode_size_y / 2
        ax.text(-shank_width-wavewidth-15, y_center, str(ch + 1), ha="center", va="center", fontsize=14)
    ax.text(-shank_width-wavewidth-15, vertical_eletrode_span+10, f"shank#\nch#", ha="center", va="center", fontsize=10)
    ax.text(0, vertical_eletrode_span-5, f"{str(f'{ref_shank:02d}'):{wavewidth/4}s}ref", ha="center", va="center", fontsize=12)
    if target_shank is not None: 
        ax.text(0, vertical_eletrode_span-5, f"{str(f'{target_shank:02d}'):{wavewidth/2}s}tgt", ha="center", va="center", fontsize=12)
    ax.text(-shank_width-wavewidth-15,-interchannel_y-30,footnote,fontsize=10)

    # Set limits and aspect
    ax.set_xlim(-shank_width-wavewidth, shank_width+wavewidth)
    ax.set_ylim(-10, vertical_eletrode_span + 10)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax

