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


# ---------------------------------------------------------------------------
# ProbeNetworkConfig
# ---------------------------------------------------------------------------

@dataclass
class ProbeNetworkConfig:
    """Display toggles for probe network render — no Tk dependency."""
    # Focus
    focused_neuron: int | None = None
    focused_pair: tuple | None = None
    current_pair: tuple | None = None
    show_current_pair: bool = False
    # Display toggles
    show_arrows: bool = True
    hide_unconnected: bool = False
    hide_same_channel: bool = False
    hide_same_shank: bool = False
    # Style
    line_alpha: float = 1.0
    h_scale: float = 1.0
    v_scale: float = 1.0
    hidden_shanks: frozenset = field(default_factory=frozenset)
    enabled_conn_types: frozenset | None = None   # None = all enabled
    # Group filter
    group_pairs: frozenset = field(default_factory=frozenset)
    gf_active: bool = False
    # Appearance
    dark_mode: bool = False
    type_colors: dict = field(default_factory=lambda: {
        ('pyr', 'pyr'):     '#D32F2F',
        ('pyr', 'inter'):   '#DAA520',
        ('inter', 'pyr'):   '#2E7D32',
        ('inter', 'inter'): '#1565C0',
    })
    show_ch_ids: bool = False
    session_label: str = ''
    any_mode: bool = False
    n_sessions: int = 1
    sess_idx: int = 0
    pair_title: str = ''


# ---------------------------------------------------------------------------
# Module-level render helpers (no Tk)
# ---------------------------------------------------------------------------

_NET_DEFAULT_E = '#D32F2F'
_NET_DEFAULT_I = '#1565C0'


def _ct_enabled(ct, cfg: ProbeNetworkConfig) -> bool:
    return cfg.enabled_conn_types is None or ct in cfg.enabled_conn_types


def _should_skip_pair(ref: int, tgt: int, shank_ids, peak_channels,
                      cfg: ProbeNetworkConfig) -> bool:
    if (cfg.hide_same_shank
            and shank_ids is not None
            and ref < len(shank_ids) and tgt < len(shank_ids)
            and int(shank_ids[ref]) == int(shank_ids[tgt])):
        return True
    if (cfg.hide_same_channel
            and peak_channels is not None
            and ref < len(peak_channels) and tgt < len(peak_channels)
            and int(peak_channels[ref]) == int(peak_channels[tgt])):
        return True
    return False


def _arrow_style(is_fp: bool, fp, in_filt: bool,
                 is_cur: bool, is_sel: bool, is_cpair: bool,
                 ec_default: str) -> tuple:
    if is_fp:                 return 1.00, 3.0, 7, ec_default
    if fp is not None:        return 0.12, 0.3, 1, '#CCCCCC'
    if not in_filt:           return 0.20, 0.4, 1, '#CCCCCC'
    if not is_cur and is_sel: return 0.70, 1.4, 3, ec_default
    if not is_cur:            return 0.35, 0.6, 2, ec_default
    if is_cpair:              return 1.00, 3.0, 7, ec_default
    if is_sel:                return 0.90, 1.8, 4, ec_default
    return                          0.55, 0.9, 3, ec_default


def _arc_style(ref: int, tgt: int, current_pair, fp, fn) -> tuple:
    if (ref, tgt) == current_pair:                return 1.0, 2.5
    if fp is not None and (ref, tgt) != fp:        return 0.12, 0.5
    if fn is not None and ref != fn and tgt != fn: return 0.12, 0.5
    return 0.85, 1.4


# ---------------------------------------------------------------------------
# Vectorized position lookup
# ---------------------------------------------------------------------------

def _compute_positions(neurons, pg):
    """Return (x_pos, y_pos, peak_channels) vectorized via pandas reindex."""
    peak_ch = np.asarray(neurons.peak_channels, dtype=int)
    pg_df = pg.to_dataframe().set_index('channel_id')
    x_pos = pg_df['x'].reindex(peak_ch).fillna(0.0).to_numpy(dtype=float)
    y_pos = pg_df['y'].reindex(peak_ch).fillna(0.0).to_numpy(dtype=float)
    return x_pos, y_pos, peak_ch


# ---------------------------------------------------------------------------
# Draw functions — notebook-callable, no Tk dependency
# ---------------------------------------------------------------------------

def _draw_neurons(ax, x_pos, y_pos, peak_channels, shank_ids, neuron_type, n_neurons,
                  cluster_neurons, pair_entries, deleted_pair_entries,
                  cfg: ProbeNetworkConfig):
    """Scatter-plot neurons, applying focus/hide-unconnected at render time."""
    fn            = cfg.focused_neuron
    fp            = cfg.focused_pair
    nt            = neuron_type

    fp_neurons = {fp[0], fp[1]} if fp is not None else set()

    all_involved: set = set()
    for (ref, tgt), entries in pair_entries.items():
        if fn is not None and ref != fn and tgt != fn:
            continue
        if fp is not None and ref not in fp_neurons and tgt not in fp_neurons:
            continue
        if cfg.gf_active and (ref, tgt) not in cfg.group_pairs:
            continue
        for entry in entries:
            if not _ct_enabled(entry['conn_type'], cfg):
                continue
            if _should_skip_pair(ref, tgt, shank_ids, peak_channels, cfg):
                continue
            all_involved.add(ref)
            all_involved.add(tgt)
            break

    _MARKER_SIZE = 50
    _marker_diam_pt = np.sqrt(_MARKER_SIZE / np.pi) * 2.0
    try:
        pt_to_px   = ax.get_figure().dpi / 72.0
        marker_px  = _marker_diam_pt * pt_to_px
        inv        = ax.transData.inverted()
        origin_d   = inv.transform((0.0, 0.0))
        step_d     = inv.transform((marker_px, 0.0))
        _step      = abs(step_d[0] - origin_d[0]) * 0.5
    except Exception:
        _step = 5.0

    _GRAYS = [f'#{int(v*255):02x}{int(v*255):02x}{int(v*255):02x}'
              for v in [0.0, 0.2, 0.4, 0.6, 0.8]]
    _ch_slots = defaultdict(list)
    for i, ch in enumerate(peak_channels):
        _ch_slots[int(ch)].append(i)

    x_spread = x_pos.copy()
    neuron_colors = np.empty(n_neurons, dtype=object)
    neuron_colors[:] = '#000000'
    for ch, idxs in _ch_slots.items():
        n_ch = len(idxs)
        offsets = (np.arange(n_ch) - (n_ch - 1) / 2.0) * _step
        for slot, ni in enumerate(idxs):
            x_spread[ni] += offsets[slot]
            neuron_colors[ni] = _GRAYS[slot % len(_GRAYS)]

    hide_unconnected = cfg.hide_unconnected
    unconnected, connected_list = [], []
    for idx in range(n_neurons):
        if fn is not None and idx == fn:
            continue
        if idx in fp_neurons:
            continue
        if cfg.hidden_shanks and shank_ids is not None and idx < len(shank_ids):
            if int(shank_ids[idx]) in cfg.hidden_shanks:
                continue
        ntype    = nt[idx] if nt is not None else None
        is_inter = (ntype == 'inter')
        if hide_unconnected:
            in_any = idx in all_involved
        else:
            in_any = idx in all_involved or idx in cluster_neurons
        if in_any:
            connected_list.append((idx, is_inter))
        else:
            unconnected.append(idx)

    if not hide_unconnected and unconnected:
        ax.scatter(x_spread[unconnected], y_pos[unconnected],
                   s=_MARKER_SIZE, marker='o', color='#9E9E9E',
                   zorder=1, linewidths=0, edgecolors='none', alpha=0.25)

    _base_alpha = 1.0 if fp is None else 0.3
    for idx, is_inter in connected_list:
        c      = '#9E9E9E' if fp is not None else neuron_colors[idx]
        marker = 'o' if is_inter else '^'
        ax.scatter([x_spread[idx]], [y_pos[idx]],
                   s=_MARKER_SIZE, marker=marker, color=c,
                   zorder=4, linewidths=0, edgecolors='none', alpha=_base_alpha)

    if fn is not None and 0 <= fn < n_neurons:
        fn_ntype  = nt[fn] if nt is not None else None
        fn_marker = 'o' if fn_ntype == 'inter' else '^'
        ax.scatter([x_pos[fn]], [y_pos[fn]], s=140, marker=fn_marker,
                   color='#FF6F00', zorder=6, linewidths=2.0, edgecolors='black')
    if fp is not None:
        for nid, clr in [(fp[0], '#FF6F00'), (fp[1], '#1E88E5')]:
            if 0 <= nid < n_neurons:
                ntype = nt[nid] if nt is not None else None
                m = 'o' if ntype == 'inter' else '^'
                ax.scatter([x_pos[nid]], [y_pos[nid]], s=140, marker=m,
                           color=clr, zorder=6, linewidths=2.0, edgecolors='black')


def _draw_connections(ax, x_pos, y_pos, peak_channels, shank_ids, n_neurons,
                      pair_entries, deleted_pair_entries, cfg: ProbeNetworkConfig):
    """Draw FancyArrowPatch arrows, deleted-pair lines, same-channel arcs, current-pair overlay."""
    fn           = cfg.focused_neuron
    fp           = cfg.focused_pair
    current_pair = cfg.current_pair

    fp_neurons   = {fp[0], fp[1]} if fp is not None else set()
    show_arrows  = cfg.show_arrows
    all_pair_set = set(pair_entries.keys())

    for (ref, tgt), entries in pair_entries.items():
        if not show_arrows:
            break
        if not (0 <= ref < n_neurons and 0 <= tgt < n_neurons):
            continue
        if fn is not None and ref != fn and tgt != fn:
            continue
        if fp is not None and ref not in fp_neurons and tgt not in fp_neurons:
            continue
        if cfg.gf_active and (ref, tgt) not in cfg.group_pairs:
            continue
        if cfg.hidden_shanks and shank_ids is not None:
            if (ref < len(shank_ids) and tgt < len(shank_ids)
                    and (int(shank_ids[ref]) in cfg.hidden_shanks
                         or int(shank_ids[tgt]) in cfg.hidden_shanks)):
                continue

        has_reverse = (tgt, ref) in all_pair_set
        rad = 0.18 if has_reverse else 0.0

        for entry in entries:
            ct       = entry['conn_type']
            ei       = entry['ei']
            is_cur   = entry['is_current']
            in_filt  = entry['in_filter']
            is_sel   = entry['is_selected']
            is_cpair = (is_cur and (ref, tgt) == current_pair)

            if not _ct_enabled(ct, cfg):
                continue
            if _should_skip_pair(ref, tgt, shank_ids, peak_channels, cfg):
                continue

            ec_default = cfg.type_colors.get(
                ct, _NET_DEFAULT_E if ei == 'E' else _NET_DEFAULT_I)
            is_fp = (fp is not None and (ref, tgt) == fp)

            alpha, lw, zo, ec = _arrow_style(
                is_fp, fp, in_filt, is_cur, is_sel, is_cpair, ec_default)
            alpha *= cfg.line_alpha

            mutation = 10 if is_cpair else 7
            pickable = in_filt and alpha >= 0.30

            arrow = FancyArrowPatch(
                (x_pos[ref], y_pos[ref]), (x_pos[tgt], y_pos[tgt]),
                arrowstyle='->', color=ec,
                linewidth=lw, alpha=alpha, mutation_scale=mutation,
                connectionstyle=f'arc3,rad={rad}',
                shrinkA=5, shrinkB=5, zorder=zo,
                picker=6 if pickable else False,
            )
            arrow.set_gid(f"{ref}_{tgt}_{entry['key']}")
            ax.add_patch(arrow)
            if is_cpair:
                ax.add_patch(FancyArrowPatch(
                    (x_pos[ref], y_pos[ref]), (x_pos[tgt], y_pos[tgt]),
                    arrowstyle='->', color='black',
                    linewidth=1.5, alpha=alpha, mutation_scale=mutation,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=5, shrinkB=5, zorder=zo + 1, picker=False,
                ))

    if show_arrows:
        for (ref, tgt) in deleted_pair_entries:
            if fn is not None and ref != fn and tgt != fn:
                continue
            if cfg.hidden_shanks and shank_ids is not None:
                if (ref < len(shank_ids) and tgt < len(shank_ids)
                        and (int(shank_ids[ref]) in cfg.hidden_shanks
                             or int(shank_ids[tgt]) in cfg.hidden_shanks)):
                    continue
            has_reverse = (tgt, ref) in deleted_pair_entries
            rad = 0.18 if has_reverse else 0.0
            ax.add_patch(FancyArrowPatch(
                (x_pos[ref], y_pos[ref]), (x_pos[tgt], y_pos[tgt]),
                arrowstyle='->', color='#333333',
                linewidth=0.5, alpha=0.20 * cfg.line_alpha, mutation_scale=5,
                connectionstyle=f'arc3,rad={rad}',
                shrinkA=5, shrinkB=5, zorder=1, picker=False,
            ))

    if (fp is not None and fp not in pair_entries
            and show_arrows
            and 0 <= fp[0] < n_neurons and 0 <= fp[1] < n_neurons):
        ax.add_patch(FancyArrowPatch(
            (x_pos[fp[0]], y_pos[fp[0]]), (x_pos[fp[1]], y_pos[fp[1]]),
            arrowstyle='->', color='#888888',
            linewidth=1.5, alpha=0.7 * cfg.line_alpha, linestyle='--',
            mutation_scale=8, connectionstyle='arc3,rad=0',
            shrinkA=5, shrinkB=5, zorder=7,
        ))

    cur_pair_on = cfg.show_current_pair
    if (cur_pair_on and current_pair is not None
            and show_arrows
            and 0 <= current_pair[0] < n_neurons
            and 0 <= current_pair[1] < n_neurons):
        _cp_ct = None
        if current_pair in pair_entries:
            _cp_ents = pair_entries[current_pair]
            _cp_e = next((e for e in _cp_ents if e.get('is_current')), _cp_ents[0])
            _cp_ct = _cp_e.get('conn_type')
        _cp_col = cfg.type_colors.get(_cp_ct, '#888888')
        ax.add_patch(FancyArrowPatch(
            (x_pos[current_pair[0]], y_pos[current_pair[0]]),
            (x_pos[current_pair[1]], y_pos[current_pair[1]]),
            arrowstyle='->', color=_cp_col,
            linewidth=5.0, alpha=1.0 * cfg.line_alpha, mutation_scale=10,
            connectionstyle='arc3,rad=0', shrinkA=5, shrinkB=5, zorder=8,
        ))
        ax.add_patch(FancyArrowPatch(
            (x_pos[current_pair[0]], y_pos[current_pair[0]]),
            (x_pos[current_pair[1]], y_pos[current_pair[1]]),
            arrowstyle='->', color='black',
            linewidth=2.5, alpha=1.0 * cfg.line_alpha, mutation_scale=10,
            connectionstyle='arc3,rad=0', shrinkA=5, shrinkB=5, zorder=9,
        ))

    _hide_same_ch    = cfg.hide_same_channel
    _hide_same_shank = cfg.hide_same_shank
    if peak_channels is not None and show_arrows and not _hide_same_ch:
        BASE_R, R_STEP, GAP = 7, 5, 11

        _arc_entry_for: dict = {}
        for (ref, tgt), entries in pair_entries.items():
            _arc_entry_for[(ref, tgt)] = next(
                (e for e in entries if e.get('is_current')), entries[0])

        _chan_entries: dict = {}
        for (ref, tgt), entry in _arc_entry_for.items():
            if not entry.get('in_filter', True):
                continue
            if ref >= n_neurons or tgt >= n_neurons:
                continue
            try:
                if peak_channels[ref] != peak_channels[tgt]:
                    continue
            except (IndexError, TypeError):
                continue
            if not _ct_enabled(entry.get('conn_type'), cfg):
                continue
            if (_hide_same_shank and shank_ids is not None
                    and ref < len(shank_ids) and tgt < len(shank_ids)
                    and int(shank_ids[ref]) == int(shank_ids[tgt])):
                continue
            _chan_entries.setdefault(int(peak_channels[ref]), []).append((ref, tgt, entry))

        for ch, ch_ents in _chan_entries.items():
            ref0 = ch_ents[0][0]
            cx, cy = x_pos[ref0] + GAP, y_pos[ref0]
            for k, (ref, tgt, entry) in enumerate(ch_ents):
                ct = entry.get('conn_type')
                arc_alpha, lw = _arc_style(ref, tgt, current_pair, fp, fn)
                arc_alpha *= cfg.line_alpha
                col = cfg.type_colors.get(ct, '#888888')
                r   = BASE_R + k * R_STEP
                arc = Arc((cx, cy), 2 * r, 2 * r, angle=0, theta1=20, theta2=340,
                          color=col, linewidth=lw, alpha=arc_alpha, zorder=4)
                arc.set_gid(f"{ref}_{tgt}_{entry.get('key', '')}")
                arc.set_picker(3)
                ax.add_patch(arc)
                t_r = math.radians(20)
                px, py = cx + r * math.cos(t_r), cy + r * math.sin(t_r)
                eps = r * 0.22
                ax.annotate('', xy=(px + math.sin(t_r) * eps, py - math.cos(t_r) * eps),
                            xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color=col, lw=1.0, mutation_scale=6),
                            zorder=5)
                lbl_x = cx + r * math.cos(math.radians(90))
                lbl_y = cy + r * math.sin(math.radians(90))
                ax.text(lbl_x, lbl_y, str(k + 1), ha='center', va='center',
                        fontsize=5, color=col, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))


def _draw_labels(ax, x_pos, y_pos, pg, shank_ids, n_neurons, pair_entries,
                 cfg: ProbeNetworkConfig):
    """Draw legend, shank labels, axis limits, session title, dark mode.

    Returns (xs_all, ys_all) so the caller can do zoom init.
    """
    fig = ax.get_figure()
    h_scale      = cfg.h_scale
    v_scale      = cfg.v_scale
    hidden_shanks = cfg.hidden_shanks

    # ── Legend ────────────────────────────────────────────────────────
    shown_types: set = set()
    for entries in pair_entries.values():
        for entry in entries:
            if entry['conn_type'] is not None:
                shown_types.add(entry['conn_type'])
    _ct_label = {
        ('pyr', 'pyr'):     'pyr→pyr',
        ('pyr', 'inter'):   'pyr→int',
        ('inter', 'inter'): 'int→int',
        ('inter', 'pyr'):   'int→pyr',
    }
    legend_handles = [
        Line2D([0], [0], color=cfg.type_colors[ct], lw=2, label=_ct_label[ct])
        for ct in [('pyr', 'pyr'), ('pyr', 'inter'), ('inter', 'inter'), ('inter', 'pyr')]
        if ct in shown_types and _ct_enabled(ct, cfg)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=6, loc='lower left',
                  framealpha=0.75, handlelength=1.4)

    # ── Shank labels ──────────────────────────────────────────────────
    if pg is not None:
        y_top = np.max(pg.y) * v_scale + 20
        for sk in pg._data['shank_id'].unique():
            if int(sk) in hidden_shanks:
                continue
            shank_data = pg._data[pg._data['shank_id'] == sk]
            sx = shank_data['x'].mean() * h_scale
            ax.text(sx, y_top, f"S{int(sk)}", ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#555555')

    # ── Axis limits ───────────────────────────────────────────────────
    xs_all, ys_all = [], []
    if pg is not None:
        df = pg._data
        if hidden_shanks:
            df = df[~df['shank_id'].apply(lambda s: int(s) in hidden_shanks)]
        if not df.empty:
            xs_all.extend((df['x'] * h_scale).tolist())
            ys_all.extend((df['y'] * v_scale).tolist())
    if shank_ids is not None:
        for idx in range(n_neurons):
            if idx < len(shank_ids) and int(shank_ids[idx]) not in hidden_shanks:
                xs_all.append(x_pos[idx])
                ys_all.append(y_pos[idx])
    if xs_all and ys_all:
        pad_x = max((max(xs_all) - min(xs_all)) * 0.08, 20)
        pad_y = max((max(ys_all) - min(ys_all)) * 0.06, 20)
        ax.set_xlim(min(xs_all) - pad_x, max(xs_all) + pad_x)
        ax.set_ylim(min(ys_all) - pad_y, max(ys_all) + pad_y)

    ax.axis('off')
    ax.set_aspect('equal')

    # ── Any-mode session / pair title ─────────────────────────────────
    if cfg.session_label:
        sess_lbl = f"{cfg.session_label}  {cfg.sess_idx + 1}/{cfg.n_sessions}"
        fig.text(0.5, 0.985, sess_lbl,
                 fontsize=8, ha='center', va='top', color='#222')
        if cfg.pair_title:
            fig.text(0.5, 0.002, cfg.pair_title,
                     fontsize=6, ha='center', va='bottom', color='#444')

    # ── Dark mode ─────────────────────────────────────────────────────
    if cfg.dark_mode:
        _bg, _fg = '#2b2b2b', 'white'
        fig.set_facecolor(_bg)
        ax.set_facecolor(_bg)
        ax.tick_params(colors=_fg)
        ax.xaxis.label.set_color(_fg)
        ax.yaxis.label.set_color(_fg)
        for txt in fig.texts:
            txt.set_color(_fg)
        for txt in ax.texts:
            txt.set_color(_fg)
        for sp in ax.spines.values():
            sp.set_edgecolor('#666666')

    return xs_all, ys_all


# ---------------------------------------------------------------------------
# Notebook entry point
# ---------------------------------------------------------------------------

def plot_probe_network(ax, neurons, ptrs, pg, cfg=None):
    """Draw probe network on *ax*. Notebook-callable, no Tk dependency.

    Parameters
    ----------
    neurons : Neurons
    ptrs    : CCGPointer or list of CCGPointer
    pg      : ProbeGroup
    cfg     : ProbeNetworkConfig or None (uses defaults)
    """
    if cfg is None:
        cfg = ProbeNetworkConfig()
    if not isinstance(ptrs, (list, tuple)):
        ptrs = [ptrs]

    x_pos, y_pos, peak_ch = _compute_positions(neurons, pg)
    n = len(x_pos)

    pair_entries: dict = {}
    for ptr in ptrs:
        if ptr is None or ptr.inds is None:
            continue
        ct = getattr(getattr(ptr, 'key', None), 'conn_type', None)
        ei = getattr(getattr(ptr, 'key', None), 'excitability', 'E')
        sel = set(map(tuple, ptr.inds2)) if hasattr(ptr, 'inds2') else set()
        for ref, tgt in map(tuple, ptr.inds[:, -2:]):
            pair_entries.setdefault((ref, tgt), []).append({
                'key': f'{ref}_{tgt}',
                'conn_type': ct, 'ei': ei,
                'is_current': True, 'in_filter': True,
                'is_selected': (ref, tgt) in sel,
            })

    shank_ids   = getattr(neurons, 'shank_ids',  None)
    neuron_type = getattr(neurons, 'neuron_type', None)
    cur_arr = np.vstack([ptr.inds[:, -2:] for ptr in ptrs
                         if ptr is not None and ptr.inds is not None]) \
              if any(p is not None and p.inds is not None for p in ptrs) \
              else np.empty((0, 2), dtype=int)
    cluster_neurons = set(int(v) for v in np.unique(cur_arr))

    _draw_neurons(ax, x_pos, y_pos, peak_ch, shank_ids, neuron_type, n,
                  cluster_neurons, pair_entries, {}, cfg)
    _draw_connections(ax, x_pos, y_pos, peak_ch, shank_ids, n,
                      pair_entries, {}, cfg)
    _draw_labels(ax, x_pos, y_pos, pg, shank_ids, n, pair_entries, cfg)
    return ax

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

