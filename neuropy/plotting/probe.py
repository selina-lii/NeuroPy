from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import numpy as np

from ..core import ProbeGroup


def plot_probe(
    probe: ProbeGroup,
    annotate_channels=None,
    channel_id=True,
    disconnected=True,
    x_scale=1.0,
    y_scale=1.0,
    hidden_shanks=None,
    ax=None,
):
    """Plot probe channel layout.

    Parameters
    ----------
    x_scale, y_scale : float
        Multiply the raw probe x/y coordinates to adjust horizontal (shank)
        and vertical (channel) spacing.  Default 1.0 = micron coordinates.
    hidden_shanks : set or None
        Set of shank_id integers to hide completely from the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    shank_ids_all = probe.shank_id
    if hidden_shanks:
        mask = np.array([int(s) not in hidden_shanks for s in shank_ids_all])
    else:
        mask = np.ones(len(shank_ids_all), dtype=bool)

    px = probe.x[mask] * x_scale
    py = probe.y[mask] * y_scale

    ax.scatter(
        px,
        py,
        s=12,
        marker="o",
        color="gray",
        zorder=1,
        linewidths=0.5,
        alpha=0.5,
    )
    if channel_id:
        for x, y, chan_id in zip(px, py, probe.channel_id[mask]):
            ax.annotate(chan_id, (x, y), fontsize=8)

    if disconnected:
        disc = probe.get_disconnected
        if hidden_shanks and 'shank_id' in disc.columns:
            disc = disc[~disc['shank_id'].apply(lambda s: int(s) in hidden_shanks)]
        ax.scatter(
            disc.x.values * x_scale,
            disc.y.values * y_scale,
            s=18,
            edgecolors="#FF5252",
            facecolors="none",
        )

    if annotate_channels is not None:
        prb_data = probe.to_dataframe().set_index("channel_id")
        for channel in annotate_channels:
            x = prb_data.loc[[channel]].x.values[0] * x_scale
            y = prb_data.loc[[channel]].y.values[0] * y_scale
            ax.scatter(x, y, s=30, edgecolors="g", facecolors="none", linewidths=2)

    ax.axis("off")
    ax.set_title(f"Probe {probe.n_contacts}ch")

    return ax


def plot_waveform_on_channel(
    ref_waveform,
    ref_shank,
    target_waveform=None,
    target_shank=None,
    color="orange",
    amplitude_limit=False,
    footnote="",
    ax=None,
    ch_per_shank=16,
    discarded_channels=None,
    peak_channel_global=None,
):
    # TODO make dataclass
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

    ch_per_shank = int(ch_per_shank)
    ref_shank = int(ref_shank)
    disc_set = (
        set(int(x) for x in np.ravel(discarded_channels))
        if discarded_channels is not None and len(discarded_channels)
        else set()
    )
    peak_g = int(peak_channel_global) if peak_channel_global is not None else None

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

        global_ch = ch_per_shank * ref_shank + ch
        is_peak = peak_g is not None and peak_g == global_ch
        wcol = "#C62828" if is_peak else "black"
        wlw = 2.8 if is_peak else 1.2
        wz = 5 if is_peak else 2

        # Left or right of shank
        if ch < n_channels_per_side:
            x_center = -x_offset-shank_width
        else:
            x_center = x_offset-5
        ax.plot(
            x_center + np.zeros_like(wf1) + np.arange(window) / x_scale,
            y_center + wf1,
            color=wcol,
            lw=wlw,
            zorder=wz,
            solid_capstyle="round",
        )

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

    # Channel numbers — hardware / linear contact index (respects discarded)
    # IMPORTANT: use the SAME y-centers as waveform plotting so labels align.
    for ch in range(n_channels_per_side * 2):
        if ch < n_channels_per_side:
            y_center = 50 + ch * (eletrode_size_y + interchannel_y) + eletrode_size_y / 2
        else:
            i = ch - n_channels_per_side
            y_center = 50 + i * (eletrode_size_y + interchannel_y) + eletrode_size_y / 2 - (eletrode_size_y + interchannel_y) / 2
        gch = ch_per_shank * ref_shank + ch
        if gch in disc_set:
            lbl = "—"
            fgc = "#999999"
        else:
            lbl = str(gch)
            fgc = "#C62828" if peak_g is not None and peak_g == gch else "black"
        ax.text(
            -shank_width - wavewidth - 15,
            y_center,
            lbl,
            ha="center",
            va="center",
            fontsize=14,
            color=fgc,
            fontweight="bold" if peak_g is not None and peak_g == gch else "normal",
        )
    ax.text(-shank_width-wavewidth-15, vertical_eletrode_span+10, f"shank#\nch#", ha="center", va="center", fontsize=10)
    ax.text(0, vertical_eletrode_span-5, f"{str(f'{ref_shank:02d}'):{wavewidth/4}s}ref", ha="center", va="center", fontsize=12)
    if target_shank is not None: 
        ax.text(0, vertical_eletrode_span-5, f"{str(f'{target_shank:02d}'):{wavewidth/2}s}tgt", ha="center", va="center", fontsize=12)
    ax.text(-shank_width-wavewidth-15,-interchannel_y-30,footnote,fontsize=10)

    # Set limits and aspect
    ax.set_xlim(-shank_width-wavewidth, shank_width+wavewidth)
    ax.set_ylim(-10, vertical_eletrode_span + 10)
    # In embedded UI panels, equal-aspect can shrink the drawing drastically.
    # Use auto aspect so the waveform/probe fills available space.
    ax.set_aspect("auto")
    ax.axis("off")

    return ax

