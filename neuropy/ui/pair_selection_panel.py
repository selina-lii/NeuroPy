"""
neuropy/ui/pair_selection_panel.py

Left column of CCGReviewUI: pair selection lists, group management, search,
and the Spike Pairs tab used during spike attribution.

Classes
-------
SelectionData       — single source of truth for pair/group/tag state
SearchBar           — search-bar widget (Ctrl+F entry point)
LeftPanel           — pair selection UI (position-named)
SpikePairsPanel     — Spike Pairs tab content (subsidiary to LeftPanel)
LeftPanelContainer  — thin wrapper that owns both and exposes .widget
"""

from __future__ import annotations

import json
import re
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from collections import defaultdict as _defaultdict
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirrored in ccg_ui.py — keep in sync)
# ---------------------------------------------------------------------------

_ADMITTED_GROUP   = "__admitted__"
_SPECIAL_PREFIX   = "__special_"
_AVAIL_GROUP_HDR  = "__avail_group_hdr__"

# ---------------------------------------------------------------------------
# SelectionData — owns all pair / group / tag state
# ---------------------------------------------------------------------------

class SelectionData:
    """Single source of truth for pair selection and group/tag annotations.

    Owned by CCGReviewUI; passed read-only to ProbeNetworkPanel and
    read-write to LeftPanel. No Tk dependency.

    Notes
    -----
    selected_inds / unselected_inds are sets of (ref, tgt) tuples in
    single-session mode; in any-mode they contain (Key, ref, tgt) triples.
    """

    def __init__(self):
        self.selected_inds:   set = set()
        self.unselected_inds: set = set()

        # Group state  {group_name -> {session_str -> set((ref,tgt))}}
        self._groups: dict = {}
        self._groups.setdefault(_ADMITTED_GROUP, {})

        # {(ref, tgt): {"notes": str, "tags": [str,...], "groups": [...]}}
        self._pair_tags: dict = {}

        # group_name -> hotkey char (e.g. '1', 'a')
        self._group_hotkeys: dict = {}
        # group_name -> notes string
        self._group_notes: dict = {}

        # v4.0 schema: int_id -> {name, hotkey, notes}
        self._group_registry: dict = {}
        self._next_group_id: int = 1

    # ------------------------------------------------------------------
    # Serialization (groups / tags portion only)
    # SelectionData owns serializing its own data.
    # Full session save (selections-by-type etc.) is handled by CCGReviewUI.
    # ------------------------------------------------------------------

    def serialize_groups(self, session_str: str | None = None) -> dict:
        """Return a groups-export.json-compatible dict for this data."""
        groups = {}
        for g, sessions_dict in self._groups.items():
            if isinstance(sessions_dict, set):
                sess = session_str or '__default__'
                groups[g] = {sess: [[int(r), int(c)]
                                    for r, c in sorted(sessions_dict)]}
            else:
                groups[g] = {
                    sess: [[int(r), int(c)] for r, c in sorted(pairs)]
                    for sess, pairs in sessions_dict.items()
                    if pairs
                }
        return {
            'groups':  groups,
            'hotkeys': dict(self._group_hotkeys),
            'notes':   dict(self._group_notes),
        }

    def deserialize_groups(self, data: dict, file_session: str,
                           restore_hotkeys: bool = False):
        """Merge group data from *data* into self._groups (never overwrites).

        Replaces ``CCGReviewUI._restore_groups_from_data``.
        """
        raw_groups = data.get('groups', {})
        for g, val in raw_groups.items():
            if isinstance(val, list):
                pairs = {file_session: set(
                    tuple(int(v) for v in p) for p in val)}
            elif isinstance(val, dict):
                pairs = {
                    sess: set(tuple(int(v) for v in p) for p in pp)
                    for sess, pp in val.items()
                }
            else:
                pairs = {}

            if g not in self._groups:
                self._groups[g] = pairs
            else:
                for sess, sp in pairs.items():
                    if sess not in self._groups[g]:
                        self._groups[g][sess] = sp

        self._groups.setdefault(_ADMITTED_GROUP, {})
        if restore_hotkeys:
            self._group_hotkeys.update(data.get('hotkeys', {}))
        for k, v in data.get('notes', {}).items():
            if k not in self._group_notes:
                self._group_notes[k] = v


# ---------------------------------------------------------------------------
# SearchBar — search widget hidden until Ctrl+F
# ---------------------------------------------------------------------------

class SearchBar:
    """Search bar that filters both pair listboxes.

    ``toggle()`` is the sole Ctrl+F entry point.
    All other actions (next, prev, hide) are dispatched through this class.
    """

    _MATCH_BG = '#fff099'

    def __init__(self, parent: tk.Widget,
                 get_listboxes: Callable,
                 on_style_reset: Callable):
        """
        Parameters
        ----------
        parent          Frame to build the search bar inside.
        get_listboxes   Callable returning (unselected_list, selected_list).
        on_style_reset  Called after clearing highlights to reapply list styles.
        """
        self._get_listboxes = get_listboxes
        self._on_style_reset = on_style_reset

        self._matches: list  = []
        self._cur:     int   = -1
        self._visible: bool  = False

        self._frame = ttk.Frame(parent)
        # not packed — shown on demand via toggle()

        ttk.Label(self._frame, text="🔍").pack(side=tk.LEFT, padx=(0, 2))
        self._var = tk.StringVar()
        self._entry = ttk.Entry(self._frame, textvariable=self._var)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._count_var = tk.StringVar(value="")
        ttk.Label(self._frame, textvariable=self._count_var,
                  width=6, anchor='e').pack(side=tk.LEFT, padx=(3, 0))
        ttk.Button(self._frame, text="▲", width=2,
                   command=lambda: self.go(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(self._frame, text="▼", width=2,
                   command=lambda: self.go(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(self._frame, text="✕", width=2,
                   command=self.hide).pack(side=tk.LEFT, padx=(1, 0))

        self._var.trace_add('write', lambda *_: self._update())
        self._entry.bind('<Return>',       lambda e: self.go(1))
        self._entry.bind('<Shift-Return>', lambda e: self.go(-1))
        self._entry.bind('<Escape>',       lambda e: self.hide())

    def toggle(self):
        """Ctrl+F: show bar and focus entry if hidden; hide otherwise."""
        if self._visible:
            self.hide()
        else:
            self._frame.pack(fill=tk.X, pady=(3, 0))
            self._visible = True
            self._entry.focus_set()
            self._entry.select_range(0, tk.END)

    def hide(self):
        """Clear state and collapse the bar."""
        self._clear()
        self._frame.pack_forget()
        self._visible = False
        try:
            # return focus to root
            self._frame.winfo_toplevel().focus_set()
        except tk.TclError:
            pass

    def go(self, delta: int):
        """Advance to next (+1) or previous (-1) match."""
        if not self._matches:
            return
        n = len(self._matches)
        self._cur = (self._cur + delta) % n
        self._apply_highlights()
        lb, i = self._matches[self._cur]
        lb.see(i)
        self._count_var.set(f'{self._cur + 1}/{n}')

    def _update(self):
        """Rebuild on query change — resets position to first match."""
        self._rebuild(preserve_cur=False)

    def _refresh(self):
        """Rebuild after list refresh — preserves current nth position."""
        self._rebuild(preserve_cur=True)

    def _rebuild(self, preserve_cur: bool = False):
        q = self._var.get().lower()
        self._clear_highlights()
        old_cur = self._cur
        self._matches = []
        self._cur = -1
        if not q:
            self._count_var.set('')
            return
        unsel, sel = self._get_listboxes()
        for lb in (unsel, sel):
            for i in range(lb.size()):
                if q in lb.get(i).lower():
                    self._matches.append((lb, i))
        n = len(self._matches)
        if n == 0:
            self._count_var.set('0/0')
            return
        if preserve_cur and old_cur >= 0:
            # Keep the same nth position (clamped); don't scroll
            self._cur = max(0, min(old_cur, n - 1))
        else:
            # New query: go to first match and scroll to it
            self._cur = 0
            lb, i = self._matches[0]
            lb.see(i)
        self._apply_highlights()
        self._count_var.set(f'{self._cur + 1}/{n}')

    def _apply_highlights(self):
        for lb, i in self._matches:
            try:
                lb.itemconfig(i, background=self._MATCH_BG,
                              selectbackground=self._MATCH_BG)
            except tk.TclError:
                pass
        self._on_style_reset()

    def _clear_highlights(self):
        for lb, i in self._matches:
            try:
                lb.itemconfig(i, background='', selectbackground='')
            except tk.TclError:
                pass
        self._on_style_reset()

    def _clear(self):
        self._clear_highlights()
        self._matches = []
        self._cur = -1
        self._count_var.set('')
        self._var.set('')

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self._var.get())


# ---------------------------------------------------------------------------
# LeftPanel — pair selection UI
# ---------------------------------------------------------------------------

class LeftPanel:
    """Left column of the 3-column CCGReviewUI layout.

    Owns the pair selection lists, sort controls, group management,
    search bar, and the Spike Pairs tab frame (populated by SpikePairsPanel).

    Named by its position in the layout (left), not by content.

    Parameters
    ----------
    parent          tk.Widget to pack the notebook into.
    data            SelectionData — single source of truth.
    ui              Back-reference to CCGReviewUI (transitional; replace with
                    callbacks incrementally in future refactors).
    ui_state_cache  dict of persisted sort-toggle states.
    """

    # Search highlight colors
    _BOOKMARK_LIST_BG    = '#ffcdd2'
    _BOOKMARK_LIST_FG    = '#b71c1c'
    _BOOKMARK_LIST_SELBG = '#ef9a9a'
    _BOOKMARK_LIST_SELFG = '#4a0000'

    def __init__(self, parent: tk.Widget, data: SelectionData,
                 ui, ui_state_cache: dict):
        self.data = data
        self._ui  = ui              # CCGReviewUI back-reference

        self.notebook = ttk.Notebook(parent)

        self._build_pair_selection_tab(ui_state_cache)
        self._build_spike_pairs_tab()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_pair_selection_tab(self, ui_state_cache: dict):
        pair_tab = ttk.Frame(self.notebook)
        self.notebook.add(pair_tab, text="Pair Selection")

        # ── Sort BooleanVars — initialized before refresh_lists() so the first
        # call during construction can safely read them.
        self._sort_selected = tk.BooleanVar(
            value=ui_state_cache.get('sort_selected', False))
        self._sort_by_tag   = tk.BooleanVar(
            value=ui_state_cache.get('sort_by_tag',   False))
        self._sort_by_mean  = tk.BooleanVar(
            value=ui_state_cache.get('sort_by_mean',  False))
        self._sort_by_min_p = tk.BooleanVar(
            value=ui_state_cache.get('sort_by_min_p', False))

        columns_pane = ttk.PanedWindow(pair_tab, orient=tk.HORIZONTAL)
        columns_pane.pack(fill=tk.BOTH, expand=True, pady=6)
        self._pair_list_pane = columns_pane

        # ── Available pairs ─────────────────────────────────────────────
        unsel_frame = ttk.Frame(columns_pane)
        columns_pane.add(unsel_frame, weight=1)
        self._avail_label = tk.StringVar(
            value=f"Available ({len(self.data.unselected_inds)})")
        _avail_hdr = ttk.Frame(unsel_frame)
        _avail_hdr.pack(fill=tk.X)
        ttk.Label(_avail_hdr, textvariable=self._avail_label,
                  font=('Arial', 10)).pack(side=tk.LEFT)
        self._clear_spec_btn = ttk.Button(
            _avail_hdr, text="✕ predictions",
            command=self._clear_speculated, width=12)
        # packed lazily in refresh_lists when speculated pairs exist

        unsel_scroll = ttk.Scrollbar(unsel_frame)
        unsel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.unselected_list = tk.Listbox(
            unsel_frame, yscrollcommand=unsel_scroll.set,
            selectmode=tk.EXTENDED, font=('Courier', 9), activestyle='none',
            exportselection=False, width=1)
        self.unselected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        unsel_scroll.config(command=self.unselected_list.yview)

        self.unselected_list.bind('<ButtonRelease-1>', self.on_pair_select)
        for _seq in ('<Command-Button-1>', '<Control-Button-1>'):
            self.unselected_list.bind(
                _seq, lambda e, lb=self.unselected_list, kind='avail':
                self._pair_list_toggle_select(e, lb, kind))
        self.unselected_list.bind('<Double-Button-1>', self.move_to_selected)
        self.unselected_list.bind('<Return>',          self.move_to_selected)
        self.unselected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))
        self.unselected_list.bind('<Button-2>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))
        self.unselected_list.bind('<KeyRelease-Up>',   self._on_arrow_key)
        self.unselected_list.bind('<KeyRelease-Down>', self._on_arrow_key)
        self.unselected_list.bind('<Left>',
            lambda e: (self._ui.change_segment(-1), 'break')[1])
        self.unselected_list.bind('<Right>',
            lambda e: (self._ui.change_segment(1),  'break')[1])

        # ── Selected pairs ───────────────────────────────────────────────
        sel_frame = ttk.Frame(columns_pane)
        columns_pane.add(sel_frame, weight=1)
        self._sel_label = tk.StringVar(
            value=f"Selected ({len(self.data.selected_inds)})")
        ttk.Label(sel_frame, textvariable=self._sel_label,
                  font=('Arial', 10)).pack()
        sel_scroll = ttk.Scrollbar(sel_frame)
        sel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_list = tk.Listbox(
            sel_frame, yscrollcommand=sel_scroll.set,
            selectmode=tk.EXTENDED, font=('Courier', 9), activestyle='none',
            exportselection=False)
        self.selected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scroll.config(command=self.selected_list.yview)

        self.selected_list.bind('<ButtonRelease-1>', self.on_pair_select)
        for _seq in ('<Command-Button-1>', '<Control-Button-1>'):
            self.selected_list.bind(
                _seq, lambda e, lb=self.selected_list, kind='sel':
                self._pair_list_toggle_select(e, lb, kind))
        self.selected_list.bind('<Double-Button-1>', self.move_to_unselected)
        self.selected_list.bind('<Return>',          self.move_to_unselected)
        self.selected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))
        self.selected_list.bind('<Button-2>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))
        self.selected_list.bind('<KeyRelease-Up>',   self._on_arrow_key)
        self.selected_list.bind('<KeyRelease-Down>', self._on_arrow_key)
        self.selected_list.bind('<Left>',
            lambda e: (self._ui.change_segment(-1), 'break')[1])
        self.selected_list.bind('<Right>',
            lambda e: (self._ui.change_segment(1),  'break')[1])

        # ── Search bar ────────────────────────────────────────────────
        self.search_bar = SearchBar(
            pair_tab,
            get_listboxes=lambda: (self.unselected_list, self.selected_list),
            on_style_reset=self._reapply_bookmark_list_styles,
        )

        self.refresh_lists()

        # ── Sort controls ────────────────────────────────────────────────
        # BooleanVars already initialized above (before refresh_lists call)
        btn_frame = ttk.Frame(pair_tab)
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        ttk.Checkbutton(btn_frame, text="Sort by group",
                        variable=self._sort_selected,
                        command=self._on_sort_by_group_toggle).pack(
            side=tk.LEFT, padx=2)
        ttk.Checkbutton(btn_frame, text="Sort by tag",
                        variable=self._sort_by_tag,
                        command=self._on_sort_by_tag_toggle).pack(
            side=tk.LEFT, padx=2)
        ttk.Checkbutton(btn_frame, text="Sort by mean",
                        variable=self._sort_by_mean,
                        command=self._on_sort_by_mean_toggle).pack(
            side=tk.LEFT, padx=2)
        ttk.Checkbutton(btn_frame, text="Sort by min p-val",
                        variable=self._sort_by_min_p,
                        command=self._on_sort_by_min_p_toggle).pack(
            side=tk.LEFT, padx=2)

        # ── Spike Pairs tab (populated by SpikePairsPanel) ───────────────
        # listbox widget exposed for SpikePairsPanel to bind
        self._spike_pairs_tab_index: int = 1

    def _build_spike_pairs_tab(self):
        # ── Spike Pairs tab ──────────────────────────────────────────────
        spike_tab = ttk.Frame(self.notebook)
        self.notebook.add(spike_tab, text="Spike Pairs", state='disabled')
        self._spike_pairs_tab = spike_tab
        # listbox / count are created by SpikePairsPanel in __init__
        # and stored as self._spike_pairs_listbox / self._spike_pairs_count

    # ------------------------------------------------------------------
    # Sort toggles
    # ------------------------------------------------------------------

    def _on_sort_by_group_toggle(self):
        if self._sort_selected.get():
            self._sort_by_tag.set(False)
            self._sort_by_mean.set(False)
            self._sort_by_min_p.set(False)
        self.refresh_lists()

    def _on_sort_by_tag_toggle(self):
        if self._sort_by_tag.get():
            self._sort_selected.set(False)
            self._sort_by_mean.set(False)
            self._sort_by_min_p.set(False)
        self.refresh_lists()

    def _on_sort_by_mean_toggle(self):
        if self._sort_by_mean.get():
            self._sort_selected.set(False)
            self._sort_by_tag.set(False)
            self._sort_by_min_p.set(False)
        self.refresh_lists()

    def _on_sort_by_min_p_toggle(self):
        if self._sort_by_min_p.get():
            self._sort_selected.set(False)
            self._sort_by_tag.set(False)
            self._sort_by_mean.set(False)
        self.refresh_lists()

    # ------------------------------------------------------------------
    # Sort helpers
    # ------------------------------------------------------------------

    def _pair_mean_ccg(self, inds):
        ccg_data = self._ui.ccg_data
        if ccg_data is None:
            return 0.0
        ref, tgt = int(inds[0]), int(inds[1])
        try:
            seg = min(self._ui.current_segment, ccg_data.ccg.shape[0] - 1)
            return float(np.mean(ccg_data.ccg[seg, ref, tgt, :]))
        except (IndexError, KeyError):
            return 0.0

    def _pair_min_pval(self, inds):
        ccg_data = self._ui.ccg_data
        if ccg_data is None or ccg_data.pval is None:
            return 1.0
        ref, tgt = int(inds[0]), int(inds[1])
        try:
            seg = min(self._ui.current_segment, ccg_data.pval.shape[0] - 1)
            arr = ccg_data.pval[seg, ref, tgt, :]
            conf = ccg_data.conf
            sl = arr[int(conf.min_lag_bin):int(conf.max_lag_bin)]
            if sl.size == 0:
                return 1.0
            m = float(np.nanmin(sl))
            return m if np.isfinite(m) else 1.0
        except (IndexError, KeyError, TypeError, ValueError):
            return 1.0

    # ------------------------------------------------------------------
    # List population
    # ------------------------------------------------------------------

    def refresh_lists(self):
        ui = self._ui
        data = self.data

        try:
            _unsel_top = self.unselected_list.yview()[0]
        except Exception:
            _unsel_top = None
        try:
            _sel_top = self.selected_list.yview()[0]
        except Exception:
            _sel_top = None

        self.unselected_list.delete(0, tk.END)
        self.selected_list.delete(0, tk.END)

        if getattr(ui, '_session_any_mode', False):
            ui._any_rebuild_pair_handles()
            ui._any_sync_selection_from_universe()

        # Focus-mode gray-out set
        gray_out = None
        fn = getattr(ui, '_focused_neuron', None)
        fp = getattr(ui, '_focused_pair', None)
        if fn is not None:
            focus_connected = {(ref, tgt) for ref, tgt in
                               map(tuple, ui.all_inds)
                               if ref == fn or tgt == fn}
            gray_out = lambda inds: inds not in focus_connected
        elif fp is not None:
            gray_out = lambda inds: inds != fp

        hide_same_channel = (hasattr(ui, 'network_panel')
                             and ui.network_panel._net_hide_same_channel_var.get())
        hide_same_shank   = (hasattr(ui, 'network_panel')
                             and ui.network_panel._net_hide_same_shank_var.get())

        _neurons_for_sids = getattr(ui, 'neurons', None)
        _sids = getattr(_neurons_for_sids, 'shank_ids', None) if _neurons_for_sids is not None else None

        _neurons_obj = getattr(ui, 'neurons', None)
        peak_channels = (getattr(_neurons_obj, 'peak_channels', None)
                         if _neurons_obj is not None else None)

        # Per-session channel lookups for any-mode gray-out
        _pchan_by_sess: dict = {}
        _nd_obj = getattr(getattr(ui, 'cd', None), 'nd', None)
        if _nd_obj is not None:
            for _nk, _nobj in getattr(_nd_obj, 'data', {}).items():
                _sk = str(getattr(_nk, 'session', _nk))
                _pc = getattr(_nobj, 'peak_channels', None)
                if _pc is not None:
                    _pchan_by_sess[_sk] = _pc

        def _should_gray(inds):
            if gray_out is not None and gray_out(inds):
                return True
            if getattr(ui, '_session_any_mode', False):
                try:
                    ref2      = int(inds[1])
                    tgt2      = int(inds[2])
                    pchan_use = _pchan_by_sess.get(
                        str(getattr(inds[0], 'session', inds[0])), peak_channels)
                except (IndexError, TypeError, ValueError, AttributeError):
                    return False
                sids_use = _sids  # any-mode uses same session's shank_ids
            else:
                try:
                    ref2 = int(inds[0])
                    tgt2 = int(inds[1])
                except (IndexError, TypeError, ValueError):
                    return False
                sids_use  = _sids
                pchan_use = peak_channels
            if hide_same_shank and sids_use is not None:
                try:
                    if int(sids_use[ref2]) == int(sids_use[tgt2]):
                        return True
                except (IndexError, TypeError, ValueError):
                    pass
            elif hide_same_channel and pchan_use is not None:
                try:
                    if pchan_use[ref2] == pchan_use[tgt2]:
                        return True
                except (IndexError, TypeError):
                    pass
            return False

        # ── Available list ───────────────────────────────────────────────
        self._avail_list_pairs = []   # parallel: (inds, tag) or _AVAIL_GROUP_HDR tuple

        if getattr(ui, '_session_any_mode', False):
            # In All/Any mode, pairs live under Selected group tags only.
            # Keep Available intentionally empty even after expanding/collapsing groups.
            pass
        else:
            # Normal mode: unselected first, then deleted (greyed)
            spec_groups = getattr(ui, '_speculated_groups', {})
            pairs_to_show = sorted(data.unselected_inds)
            for inds in pairs_to_show:
                pred_group = spec_groups.get(tuple(inds))
                lbl_prefix = self._bookmark_label_prefix(inds)
                label = f"{lbl_prefix}[{inds[0]}, {inds[1]}]"
                grp = self._pair_group_label(inds)
                if grp:
                    label += f" {grp}"
                if pred_group:
                    label += f"  ~{pred_group}"
                self.unselected_list.insert(tk.END, label)
                self._avail_list_pairs.append((inds, pred_group))
                if _should_gray(inds):
                    self.unselected_list.itemconfig(
                        self.unselected_list.size() - 1, foreground='#AAAAAA')

            deleted_inds = getattr(ui, 'deleted_inds', set())
            if deleted_inds:
                sep_idx = self.unselected_list.size()
                self.unselected_list.insert(tk.END, '── deleted ──')
                self.unselected_list.itemconfig(
                    sep_idx,
                    foreground='#AAAAAA',
                    selectforeground='#AAAAAA',
                    background='#EEEEEE',
                    selectbackground='#DDDDDD',
                )
                self._avail_list_pairs.append((_AVAIL_GROUP_HDR, 'deleted'))
                for inds in sorted(deleted_inds):
                    label = f"[{inds[0]}, {inds[1]}]"
                    self.unselected_list.insert(tk.END, label)
                    self.unselected_list.itemconfig(
                        self.unselected_list.size() - 1, foreground='#AAAAAA')
                    self._avail_list_pairs.append((inds, 'deleted'))

        # ── Selected list ────────────────────────────────────────────────
        self._sel_list_pairs      = []
        self._sel_list_header_keys = []

        def _pair_group_combo(inds):
            sess, pair = ui._pair_sess_rt(inds)
            pair = tuple(int(x) for x in pair)
            return tuple(sorted(
                g for g in data._groups
                if not g.startswith('__')
                and pair in self._group_pairs(g, session=sess)
            ))

        def _insert_sel_pair(inds):
            if getattr(ui, '_session_any_mode', False):
                sess_lbl = (str(inds[0].session)
                            if hasattr(inds[0], 'session') else str(inds[0]))
                label = (f"{self._bookmark_label_prefix(inds)}"
                         f"{sess_lbl} [{inds[1]}, {inds[2]}]")
            else:
                label = (f"{self._bookmark_label_prefix(inds)}"
                         f"[{inds[0]}, {inds[1]}]")
            grp = self._pair_group_label(inds)
            if grp:
                label += f" {grp}"
            self.selected_list.insert(tk.END, label)
            self._sel_list_pairs.append(inds)
            self._sel_list_header_keys.append(None)
            if _should_gray(inds):
                self.selected_list.itemconfig(
                    self.selected_list.size() - 1, foreground='#AAAAAA')

        _collapsed = getattr(ui, '_sel_collapsed_headers', set())

        def _insert_sel_header(text, count):
            is_collapsed = text in _collapsed
            display = f"── {text} ({count}) ──" + (" >>" if is_collapsed else "")
            hdr_idx = self.selected_list.size()
            self.selected_list.insert(tk.END, display)
            self.selected_list.itemconfig(
                hdr_idx,
                foreground='#444444', selectforeground='#444444',
                background='#CCCCCC', selectbackground='#BBBBBB',
            )
            self._sel_list_pairs.append(None)
            self._sel_list_header_keys.append(text)
            return is_collapsed

        _any = getattr(ui, '_session_any_mode', False)
        sort_group = self._sort_selected.get()
        sort_tag   = self._sort_by_tag.get()
        # sort by mean / min-p require per-session CCG data, skip in All mode
        sort_mean  = self._sort_by_mean.get() and not _any
        sort_minp  = self._sort_by_min_p.get() and not _any

        _total_any_count = 0
        if _any:
            dead      = getattr(ui, 'deleted_inds', set())
            _expanded = getattr(ui, '_any_expanded_group_tags', set())
            all_gnames = ui._any_group_header_names()

            # Count unique pairs across all groups for the tally
            _all_trips: set = set()
            for _gn in all_gnames:
                _all_trips |= (ui._any_triples_in_group(_gn) - dead)
            _total_any_count = len(_all_trips)

            def _any_insert_hdr(hdr_text, n, key=None):
                exp_hdr = hdr_text in _expanded
                hdr = f"── {hdr_text} ({n}) ──" + ("" if exp_hdr else " >>")
                hdr_idx = self.selected_list.size()
                self.selected_list.insert(tk.END, hdr)
                self.selected_list.itemconfig(
                    hdr_idx,
                    foreground='#444444', selectforeground='#444444',
                    background='#CCCCCC', selectbackground='#BBBBBB',
                )
                self._sel_list_pairs.append(None)
                self._sel_list_header_keys.append(key or hdr_text)
                return exp_hdr

            if sort_group:
                # Build pair → tag-combo buckets (combo = sorted tuple of all tags)
                pair_tags: dict = {}
                for gname in all_gnames:
                    for trip in ui._any_triples_in_group(gname):
                        if trip not in dead:
                            pair_tags.setdefault(trip, set()).add(gname)
                combo_buckets: dict = _defaultdict(list)
                for trip, tags in sorted(pair_tags.items()):
                    combo_buckets[tuple(sorted(tags))].append(trip)

                def _combo_sort_key(combo):
                    return (1, []) if not combo else (0, list(combo))

                for combo in sorted(combo_buckets.keys(), key=_combo_sort_key):
                    hdr_text = ', '.join(combo) if combo else '(untagged)'
                    trips_combo = combo_buckets[combo]
                    exp_hdr = _any_insert_hdr(hdr_text, len(trips_combo))
                    if exp_hdr:
                        for sess, r, t in trips_combo:
                            nd_key = ui._nd_key_for_session_str(sess)
                            if nd_key is None:
                                continue
                            ckey = ui._type_key_for_nd(nd_key)
                            if ckey is None:
                                continue
                            _insert_sel_pair((ckey, r, t))
            else:
                # Per-tag headers (sort_tag or no sort)
                for gname in all_gnames:
                    trips_g = ui._any_triples_in_group(gname)
                    n_tag = len(trips_g - dead)
                    exp_hdr = _any_insert_hdr(gname, n_tag)
                    if exp_hdr:
                        for row in ui._any_iter_pairs_for_group(gname):
                            _insert_sel_pair(row)

        elif sort_mean:
            for inds in sorted(data.selected_inds,
                               key=self._pair_mean_ccg, reverse=True):
                _insert_sel_pair(inds)

        elif sort_minp:
            for inds in sorted(data.selected_inds, key=self._pair_min_pval):
                _insert_sel_pair(inds)

        elif sort_group:
            buckets = _defaultdict(list)
            for inds in sorted(data.selected_inds):
                buckets[_pair_group_combo(inds)].append(inds)

            def _combo_sort_key(combo):
                return (1, []) if not combo else (0, list(combo))

            for combo in sorted(buckets.keys(), key=_combo_sort_key):
                pairs_in_combo = buckets[combo]
                hdr_text = ', '.join(combo) if combo else '(untagged)'
                collapsed = _insert_sel_header(hdr_text, len(pairs_in_combo))
                if not collapsed:
                    for inds in pairs_in_combo:
                        _insert_sel_pair(inds)

        elif sort_tag:
            tag_buckets: dict = _defaultdict(list)
            untagged = []
            non_internal = [g for g in data._groups
                            if not g.startswith('__')
                            and not g.startswith(_SPECIAL_PREFIX)]
            for inds in sorted(data.selected_inds):
                _s, _p = ui._pair_sess_rt(inds)
                _p = tuple(int(x) for x in _p)
                tags = [g for g in non_internal
                        if _p in self._group_pairs(g, session=_s)]
                if tags:
                    for t in tags:
                        tag_buckets[t].append(inds)
                else:
                    untagged.append(inds)
            for tag in sorted(tag_buckets.keys()):
                collapsed = _insert_sel_header(tag, len(tag_buckets[tag]))
                if not collapsed:
                    for inds in tag_buckets[tag]:
                        _insert_sel_pair(inds)
            if untagged:
                collapsed = _insert_sel_header('(untagged)', len(untagged))
                if not collapsed:
                    for inds in untagged:
                        _insert_sel_pair(inds)
        else:
            for inds in sorted(data.selected_inds):
                _insert_sel_pair(inds)

        # Sync avail/sel maps back to ui so undo/highlight code can read them
        ui._avail_list_pairs      = self._avail_list_pairs
        ui._sel_list_pairs        = self._sel_list_pairs
        ui._sel_list_header_keys  = self._sel_list_header_keys

        # Tally labels
        n_spec = len({p for p in getattr(ui, '_speculated_groups', {})
                      if p in data.unselected_inds})
        if getattr(ui, '_session_any_mode', False):
            self._avail_label.set("Available (0) — Any mode")
        else:
            parts = [f"Available ({len(data.unselected_inds)}"]
            if n_spec:
                parts.append(f", {n_spec} predicted")
            deleted_inds = getattr(ui, 'deleted_inds', set())
            if deleted_inds:
                parts.append(f", {len(deleted_inds)} deleted")
            parts.append(")")
            self._avail_label.set(''.join(parts))

        if hasattr(self, '_clear_spec_btn'):
            if n_spec and not getattr(ui, '_session_any_mode', False):
                self._clear_spec_btn.pack(side=tk.RIGHT, padx=2)
            else:
                self._clear_spec_btn.pack_forget()

        try:
            if getattr(ui, '_session_any_mode', False):
                self._sel_label.set(f"Selected ({_total_any_count})")
            else:
                sess = ui._current_session_str()
                ct_lbl = ui._conn_type_label(getattr(ui.key, 'conn_type', None))
                n_ct = len(ui._filter_pairs_to_conn_types(
                    sess, data.selected_inds, {ct_lbl}))
                self._sel_label.set(f"Selected ({n_ct})")
        except Exception:
            self._sel_label.set(f"Selected ({len(data.selected_inds)})")

        ui._apply_jitter_list_colors()
        ui._refresh_stats()

        if self.search_bar.active:
            self.search_bar._refresh()
        else:
            self._reapply_bookmark_list_styles()

        try:
            if _unsel_top is not None:
                self.unselected_list.yview_moveto(_unsel_top)
        except Exception:
            pass
        try:
            if _sel_top is not None:
                self.selected_list.yview_moveto(_sel_top)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pair movement
    # ------------------------------------------------------------------

    def move_to_selected(self, event=None):
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            return
        if ui._select_after is not None:
            ui.root.after_cancel(ui._select_after)
            ui._select_after = None
        if event is not None:
            idx = self.unselected_list.nearest(event.y)
        else:
            sel = self.unselected_list.curselection()
            idx = sel[-1] if sel else None
        if idx is None or idx < 0:
            return
        avail_map = self._avail_list_pairs
        if avail_map is not None:
            if idx >= len(avail_map) or avail_map[idx] is None:
                return
            entry = avail_map[idx]
            inds, pred_group = entry
        else:
            sorted_unsel = sorted(self.data.unselected_inds)
            if idx >= len(sorted_unsel):
                return
            inds, pred_group = sorted_unsel[idx], None
        scroll_top = self.unselected_list.yview()[0]
        if inds in self.data.selected_inds:
            if pred_group is not None:
                if getattr(ui, '_session_any_mode', False):
                    self._group_add_pair(pred_group, (inds[1], inds[2]),
                                         session=inds[0])
                else:
                    self._group_add_pair(pred_group, inds)
                self.refresh_lists()
            return
        ui._push_undo()
        self.data.unselected_inds.discard(inds)
        self.data.selected_inds.add(inds)
        if pred_group is not None:
            if getattr(ui, '_session_any_mode', False):
                self._group_add_pair(pred_group, (inds[1], inds[2]),
                                     session=inds[0])
            else:
                self._group_add_pair(pred_group, inds)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        ui._highlight_changed_pairs({inds})
        ui.current_pair_idx = ui.get_pair_index(inds)
        ui.update_plot()
        ui.network_panel.draw()

    def move_to_unselected(self, event=None):
        ui = self._ui
        if ui._select_after is not None:
            ui.root.after_cancel(ui._select_after)
            ui._select_after = None
        if event is not None:
            idx = self.selected_list.nearest(event.y)
        else:
            sel = self.selected_list.curselection()
            idx = sel[-1] if sel else None
        if idx is None or idx < 0:
            return
        sel_map = self._sel_list_pairs
        if sel_map is not None:
            if idx >= len(sel_map) or sel_map[idx] is None:
                hdr_keys = self._sel_list_header_keys
                if (hdr_keys is not None and idx < len(hdr_keys)
                        and hdr_keys[idx] is not None):
                    hkey = hdr_keys[idx]
                    if getattr(ui, '_session_any_mode', False):
                        if hkey in getattr(ui, '_any_expanded_group_tags', set()):
                            ui._any_expanded_group_tags.discard(hkey)
                            scroll_top = self.selected_list.yview()[0]
                            self.refresh_lists()
                            self.selected_list.yview_moveto(scroll_top)
                        else:
                            ui._toggle_any_avail_group(hkey)
                        return
                    collapsed = getattr(ui, '_sel_collapsed_headers', set())
                    if hkey in collapsed:
                        collapsed.discard(hkey)
                    else:
                        collapsed.add(hkey)
                    scroll_top = self.selected_list.yview()[0]
                    self.refresh_lists()
                    self.selected_list.yview_moveto(scroll_top)
                return
            inds = sel_map[idx]
        else:
            sorted_sel = sorted(self.data.selected_inds)
            if idx >= len(sorted_sel):
                return
            inds = sorted_sel[idx]
        if getattr(ui, '_session_any_mode', False):
            return
        scroll_top = self.selected_list.yview()[0]
        ui._push_undo()
        self.data.selected_inds.discard(inds)
        self.data.unselected_inds.add(inds)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        ui._highlight_changed_pairs({inds})
        ui.current_pair_idx = ui.get_pair_index(inds)
        ui.update_plot()
        ui.network_panel.draw()

    def _move_current_pair(self):
        """Hotkey 'm': toggle current pair between Available and Selected."""
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            return
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        row  = ui.all_inds[ui.current_pair_idx]
        inds = tuple(row)
        ui._push_undo()
        if inds in self.data.unselected_inds:
            self.data.unselected_inds.discard(inds)
            self.data.selected_inds.add(inds)
        else:
            self.data.selected_inds.discard(inds)
            self.data.unselected_inds.add(inds)
        next_idx = min(ui.current_pair_idx + 1, len(ui.all_inds) - 1)
        ui.current_pair_idx = next_idx
        self.refresh_lists()
        self._select_pair_in_list(self._pair_at_all_inds_idx(next_idx))
        ui._highlight_changed_pairs({inds})
        ui.update_plot()
        ui.network_panel.draw()

    def _select_pair_in_list(self, inds):
        """Set listbox cursor to inds and scroll it into view."""
        if inds is None:
            return
        if inds in self.data.unselected_inds:
            listbox = self.unselected_list
            avail_map = self._avail_list_pairs
            if avail_map:
                pos = next((i for i, e in enumerate(avail_map)
                            if e is not None and e[0] == inds), None)
                if pos is None:
                    return
            else:
                try:
                    pos = sorted(self.data.unselected_inds).index(inds)
                except ValueError:
                    return
        elif inds in self.data.selected_inds:
            listbox = self.selected_list
            sel_map = self._sel_list_pairs
            if sel_map:
                pos = next((i for i, e in enumerate(sel_map) if e == inds), None)
                if pos is None:
                    return
            else:
                try:
                    pos = sorted(self.data.selected_inds).index(inds)
                except ValueError:
                    return
        else:
            return
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(pos)
        listbox.activate(pos)
        listbox.see(pos)

    # ------------------------------------------------------------------
    # Pair selection events
    # ------------------------------------------------------------------

    def on_pair_select(self, event):
        widget = event.widget
        try:
            if widget is self.unselected_list:
                self.selected_list.selection_clear(0, tk.END)
            elif widget is self.selected_list:
                self.unselected_list.selection_clear(0, tk.END)
        except Exception:
            pass
        sel = widget.curselection()
        idx = sel[-1] if sel else widget.nearest(event.y)
        if idx < 0 or idx >= widget.size():
            return
        inds_from_map = None
        if widget is self.unselected_list:
            mp = getattr(self, '_avail_list_pairs', None)
            if mp and idx < len(mp):
                e = mp[idx]
                if e is not None:
                    if e[0] == _AVAIL_GROUP_HDR:
                        return
                    inds_from_map = e[0]
        elif widget is self.selected_list:
            mp = getattr(self, '_sel_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                inds_from_map = mp[idx]

        if inds_from_map is not None:
            try:
                self._ui.current_pair_idx = self._ui.get_pair_index(inds_from_map)
            except (ValueError, TypeError):
                return
        else:
            item = widget.get(idx)
            m = re.match(
                r'^\s*(?:\U0001F4CC\s*)?(?:[^\[]+\s+)?\[\s*(\d+)\s*,\s*(\d+)\s*\]',
                item)
            if not m:
                return
            inds = (int(m.group(1)), int(m.group(2)))
            try:
                self._ui.current_pair_idx = self._ui.get_pair_index(inds)
            except (ValueError, TypeError):
                return

        if widget is self.selected_list:
            try:
                ui2 = self._ui
                row = ui2.all_inds[ui2.current_pair_idx]
                ref, tgt = int(row[0]), int(row[1])
                id_pos = {int(n): i for i, n in enumerate(ui2.neurons.neuron_ids)}
                rp, tp = id_pos[ref], id_pos[tgt]
                print(f"[on_pair_select] inds=({ref},{tgt}) shanks=({int(ui2.neurons.shank_ids[rp])},{int(ui2.neurons.shank_ids[tp])})")
            except Exception:
                pass

        _ctrl = event.state & 0x4
        _cmd  = event.state & 0x8
        if _ctrl or _cmd:
            return

        ui = self._ui
        if ui._select_after is not None:
            ui.root.after_cancel(ui._select_after)
        ui._select_after = ui.root.after(180, self._do_pair_select_update)

    def _pair_list_toggle_select(self, event, listbox, kind: str):
        """Cmd/Ctrl+click: toggle row selection without clearing others."""
        try:
            if listbox is self.unselected_list:
                self.selected_list.selection_clear(0, tk.END)
            elif listbox is self.selected_list:
                self.unselected_list.selection_clear(0, tk.END)
        except Exception:
            pass
        try:
            idx = int(listbox.nearest(event.y))
        except Exception:
            return 'break'
        if idx < 0 or idx >= listbox.size():
            return 'break'
        try:
            included = listbox.selection_includes(idx)
        except Exception:
            included = False
        if included:
            listbox.selection_clear(idx)
        else:
            listbox.selection_set(idx)
        listbox.activate(idx)
        listbox.see(idx)

        inds = None
        if kind == 'avail':
            mp = getattr(self, '_avail_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                ent = mp[idx]
                if ent[0] == _AVAIL_GROUP_HDR:
                    return 'break'
                inds = ent[0]
        else:
            mp = getattr(self, '_sel_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                inds = mp[idx]
        if inds is not None:
            try:
                self._ui.current_pair_idx = self._ui.get_pair_index(inds)
            except Exception:
                pass
        try:
            self._ui._exit_spike_attribution_view()
        except Exception:
            pass
        try:
            self._ui._focused_pair = None
        except Exception:
            pass
        try:
            self._ui.update_plot()
            self._ui.network_panel.draw()
        except Exception:
            pass
        return 'break'

    def _do_pair_select_update(self):
        ui = self._ui
        ui._select_after = None
        try:
            row = ui.all_inds[ui.current_pair_idx]
            ref, tgt = int(row[0]), int(row[1])
            id_pos = {int(n): i for i, n in enumerate(ui.neurons.neuron_ids)}
            rp, tp = id_pos[ref], id_pos[tgt]
            print(f"[_do_pair_select_update] inds=({ref},{tgt}) shanks=({int(ui.neurons.shank_ids[rp])},{int(ui.neurons.shank_ids[tp])})")
        except Exception:
            pass
        try:
            ui._exit_spike_attribution_view()
        except Exception:
            pass
        ui._mark_jitter_viewed()
        ui._focused_pair = None
        if hasattr(ui, 'network_panel'):
            try:
                ui.network_panel._focus_pair_var.set("")
                ui.network_panel._focus_pair_info_var.set("")
            except Exception:
                pass
        ui.update_plot()
        ui.network_panel.draw()

    def _on_arrow_key(self, event):
        self.on_pair_select(event)

    # ------------------------------------------------------------------
    # Search (Ctrl+F entry point)
    # ------------------------------------------------------------------

    def _search_toggle(self):
        """Ctrl+F entry point — show or hide the search bar."""
        self.search_bar.toggle()

    # ------------------------------------------------------------------
    # Bookmark helpers (called by SearchBar.on_style_reset)
    # ------------------------------------------------------------------

    def _reapply_bookmark_list_styles(self):
        bm = getattr(self._ui, '_bookmarked_pairs', None) or set()
        if not bm:
            return
        av = getattr(self, '_avail_list_pairs', None)
        if av:
            for i, entry in enumerate(av):
                if entry is None:
                    continue
                raw = entry[0]
                if raw == _AVAIL_GROUP_HDR:
                    continue
                if getattr(self._ui, '_session_any_mode', False):
                    ti = (int(raw[1]), int(raw[2]))  # (ref, tgt) — session-agnostic
                else:
                    ti = tuple(int(x) for x in raw[:2])
                if ti not in bm:
                    continue
                try:
                    self.unselected_list.itemconfig(
                        i,
                        background=self._BOOKMARK_LIST_BG,
                        foreground=self._BOOKMARK_LIST_FG,
                        selectbackground=self._BOOKMARK_LIST_SELBG,
                        selectforeground=self._BOOKMARK_LIST_SELFG,
                    )
                except tk.TclError:
                    pass
        sm = getattr(self, '_sel_list_pairs', None)
        if sm:
            for i, entry in enumerate(sm):
                if entry is None:
                    continue
                if getattr(self._ui, '_session_any_mode', False):
                    ti = (int(entry[1]), int(entry[2]))  # (ref, tgt) — session-agnostic
                else:
                    ti = tuple(int(x) for x in entry[:2])
                if ti not in bm:
                    continue
                try:
                    self.selected_list.itemconfig(
                        i,
                        background=self._BOOKMARK_LIST_BG,
                        foreground=self._BOOKMARK_LIST_FG,
                        selectbackground=self._BOOKMARK_LIST_SELBG,
                        selectforeground=self._BOOKMARK_LIST_SELFG,
                    )
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------
    # Bookmark helpers
    # ------------------------------------------------------------------

    def _bookmark_label_prefix(self, inds) -> str:
        bm = getattr(self._ui, '_bookmarked_pairs', None) or set()
        if getattr(self._ui, '_session_any_mode', False):
            t = (int(inds[1]), int(inds[2]))  # (ref, tgt) — session-agnostic
        else:
            t = tuple(int(x) for x in inds[:2])
        return '\U0001F4CC ' if t in bm else ''

    def _bookmark_toggle_current(self, event=None):
        ui = self._ui
        inds = self._selected_pair_from_lists()
        if inds is None:
            if ui.current_pair_idx >= len(ui.all_inds):
                return
            row = ui.all_inds[ui.current_pair_idx]
            if getattr(ui, '_session_any_mode', False):
                hl = getattr(ui, '_any_pair_handle_list', None) or []
                ci = ui.current_pair_idx
                if 0 <= ci < len(hl):
                    ck, r, t = hl[ci]
                    inds = (int(r), int(t))  # (ref, tgt) — session-agnostic
                else:
                    return
            else:
                inds = tuple(int(x) for x in row[:2])
        else:
            # Normalize: in All mode _selected_pair_from_lists returns (sess, ref, tgt)
            if getattr(ui, '_session_any_mode', False) and len(inds) == 3:
                inds = (int(inds[1]), int(inds[2]))
        bm = ui._bookmarked_pairs
        if inds in bm:
            bm.discard(inds)
        else:
            bm.add(inds)
        self.refresh_lists()

    def _clear_bookmarks(self):
        bm = getattr(self._ui, '_bookmarked_pairs', None)
        if not bm:
            return
        bm.clear()
        self.refresh_lists()

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------

    def _selected_pair_from_lists(self):
        """Return (ref,tgt) or (sess,ref,tgt) in any-mode; else None."""
        ui = self._ui
        for lb, mp in [
            (getattr(self, 'unselected_list', None), getattr(self, '_avail_list_pairs', None)),
            (getattr(self, 'selected_list', None), getattr(self, '_sel_list_pairs', None)),
        ]:
            if lb is None:
                continue
            try:
                sel = list(lb.curselection())
            except Exception:
                sel = []
            if not sel:
                continue
            i = sel[-1]
            if mp is None:
                continue
            try:
                entry = mp[i]
            except Exception:
                continue
            if entry is None:
                continue
            if isinstance(entry, tuple) and entry and entry[0] == _AVAIL_GROUP_HDR:
                continue
            if (isinstance(entry, tuple) and len(entry) == 2
                    and isinstance(entry[0], (tuple, list, np.ndarray))):
                inds = entry[0]
            else:
                inds = entry
            try:
                if getattr(ui, '_session_any_mode', False):
                    sess = (str(inds[0].session) if hasattr(inds[0], 'session')
                            else str(inds[0]))
                    return (sess, int(inds[1]), int(inds[2]))
                return (int(inds[0]), int(inds[1]))
            except Exception:
                continue
        return None

    def _selected_pairs_from_lists(self) -> list:
        """Return all selected (ref,tgt) pairs across both lists (deduped)."""
        ui = self._ui
        out: list = []
        seen: set = set()
        for lb, mp in [
            (getattr(self, 'unselected_list', None), getattr(self, '_avail_list_pairs', None)),
            (getattr(self, 'selected_list', None), getattr(self, '_sel_list_pairs', None)),
        ]:
            if lb is None or mp is None:
                continue
            try:
                sel = list(lb.curselection())
            except Exception:
                sel = []
            for i in sel:
                try:
                    entry = mp[i]
                except Exception:
                    continue
                if entry is None:
                    continue
                if isinstance(entry, tuple) and entry and entry[0] == _AVAIL_GROUP_HDR:
                    continue
                if (isinstance(entry, tuple) and len(entry) == 2
                        and isinstance(entry[0], (tuple, list, np.ndarray))):
                    inds = entry[0]
                else:
                    inds = entry
                try:
                    if getattr(ui, '_session_any_mode', False):
                        sess = (str(inds[0].session) if hasattr(inds[0], 'session')
                                else str(inds[0]))
                        pair = (sess, int(inds[1]), int(inds[2]))
                    else:
                        pair = (int(inds[0]), int(inds[1]))
                except Exception:
                    continue
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(pair)
        out.sort(key=lambda x: (str(x[0]), x[1], x[2]) if len(x) == 3
                               else ('', x[0], x[1]))
        return out

    def _select_all(self):
        """Toggle between Select All and Deselect All."""
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            return
        if self.data.unselected_inds:
            for inds in list(self.data.unselected_inds):
                self.data.selected_inds.add(inds)
            self.data.unselected_inds.clear()
        else:
            for inds in list(self.data.selected_inds):
                self.data.unselected_inds.add(inds)
            self.data.selected_inds.clear()
        self.refresh_lists()
        ui.network_panel.draw()

    def _pair_at_all_inds_idx(self, idx: int):
        """Canonical pair key matching list rows / selection sets."""
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            hl = getattr(ui, '_any_pair_handle_list', None) or []
            if idx < 0 or idx >= len(hl):
                return None
            ck, r, t = hl[idx]
            return (str(ck.session), int(r), int(t))
        row = ui.all_inds[idx]
        return tuple(int(x) for x in row)

    # ------------------------------------------------------------------
    # Context menu helpers
    # ------------------------------------------------------------------

    def _ctx_restore_from_deleted(self, pairs):
        """Context-menu: restore pairs from deleted back to Available."""
        ui = self._ui
        if not pairs:
            return
        scroll_top = self.unselected_list.yview()[0]
        ui._push_undo()
        for p in pairs:
            ui.deleted_inds.discard(p)
            ui.unselected_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        ui._highlight_changed_pairs(set(pairs))
        ui._flush_deleted_to_store()

    def _ctx_delete_pairs(self, pairs):
        """Context-menu: move pairs from Available to deleted."""
        ui = self._ui
        if not pairs:
            return
        scroll_top = self.unselected_list.yview()[0]
        ui._push_undo()
        for p in pairs:
            ui.unselected_inds.discard(p)
            ui.deleted_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        ui._flush_deleted_to_store()

    def _ctx_delete_from_selected(self, pairs):
        """Context-menu: move pairs from Selected to deleted."""
        ui = self._ui
        if not pairs:
            return
        scroll_top = self.selected_list.yview()[0]
        ui._push_undo()
        for p in pairs:
            trip = ui._pair_row_selected_trip(p)
            ui.selected_inds.discard(trip)
            ui.deleted_inds.add(trip)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        ui._flush_deleted_to_store()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _ctx_menu(self, event, widget, action):
        ui = self._ui
        click_idx = widget.nearest(event.y)
        if click_idx not in widget.curselection():
            widget.selection_clear(0, tk.END)
            widget.selection_set(click_idx)
            widget.activate(click_idx)

        if action == 'add':
            avail_map = getattr(self, '_avail_list_pairs', None)
            sel_indices = list(widget.curselection())
            pairs, deleted_pairs = [], []
            for i in sel_indices:
                if avail_map is not None:
                    if i >= len(avail_map) or avail_map[i] is None:
                        continue
                    inds, tag = avail_map[i]
                    if tag == 'deleted':
                        deleted_pairs.append(inds)
                    else:
                        pairs.append(inds)
                else:
                    sorted_unsel = sorted(self.data.unselected_inds)
                    sorted_del   = sorted(getattr(ui, 'deleted_inds', set()))
                    sep_idx = len(sorted_unsel) if sorted_del else None
                    if i < len(sorted_unsel):
                        pairs.append(sorted_unsel[i])
                    elif sep_idx is not None and i > sep_idx:
                        di = i - sep_idx - 1
                        if 0 <= di < len(sorted_del):
                            deleted_pairs.append(sorted_del[di])
        else:
            sel_map = getattr(self, '_sel_list_pairs', None)
            if sel_map is not None:
                pairs = [sel_map[i] for i in widget.curselection()
                         if i < len(sel_map) and sel_map[i] is not None]
            else:
                ss = sorted(self.data.selected_inds)
                pairs = [ss[i] for i in widget.curselection() if i < len(ss)]
            deleted_pairs = []
        n  = len(pairs)
        nd = len(deleted_pairs)

        menu = tk.Menu(ui.root, tearoff=0)
        _any = getattr(ui, '_session_any_mode', False)
        if action == 'add':
            if not _any:
                if pairs:
                    menu.add_command(
                        label=f"Move to Selected ({n})" if n > 1 else "Move to Selected",
                        command=lambda pp=pairs: self._ctx_move_multi_to_selected(pp))
                    menu.add_command(
                        label=f"Move to Deleted ({n})" if n > 1 else "Move to Deleted",
                        command=lambda pp=pairs: self._ctx_delete_pairs(pp))
                if deleted_pairs:
                    menu.add_command(
                        label=f"Restore to Available ({nd})" if nd > 1 else "Restore to Available",
                        command=lambda pp=deleted_pairs: self._ctx_restore_from_deleted(pp))
                if not pairs and not deleted_pairs:
                    menu.add_command(label="(nothing selected)", state='disabled')
            elif not pairs and not deleted_pairs:
                menu.add_command(label="(nothing selected)", state='disabled')
        else:
            if not _any:
                menu.add_command(
                    label=f"Move to Available ({n})" if n > 1 else "Move to Available",
                    command=lambda pp=pairs: self._ctx_move_multi_to_unselected(pp))
                menu.add_command(
                    label=f"Move to Deleted ({n})" if n > 1 else "Move to Deleted",
                    command=lambda pp=pairs: self._ctx_delete_from_selected(pp))
            elif pairs:
                menu.add_command(
                    label=f"Move to Deleted ({n})" if n > 1 else "Move to Deleted",
                    command=lambda pp=pairs: self._ctx_delete_from_selected(pp))

        # Group tag submenu
        menu.add_separator()
        grp_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Group tag", menu=grp_menu)
        grp_menu.add_command(label="Create new group…",
                             command=self._create_group_dialog)
        if self.data._groups:
            grp_menu.add_separator()
        special_items = []
        for gname in sorted(self.data._groups):
            if gname.startswith(_SPECIAL_PREFIX):
                special_items.append(gname)
                continue
            if gname.startswith('__'):
                continue
            if pairs:
                all_in = all(self._pair_in_group(p, gname) for p in pairs)
                label = f"{'✓ ' if all_in else ''}  {gname}"
                grp_menu.add_command(
                    label=label,
                    command=lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))
        if special_items:
            sp_menu = tk.Menu(grp_menu, tearoff=0)
            grp_menu.add_cascade(label="Special", menu=sp_menu)
            for gname in special_items:
                display = gname[len(_SPECIAL_PREFIX):]
                if pairs:
                    all_in = all(self._pair_in_group(p, gname) for p in pairs)
                    label = f"{'✓ ' if all_in else ''}  {display}"
                    sp_menu.add_command(
                        label=label,
                        command=lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))

        # Show Together
        if pairs:
            menu.add_separator()
            max_tog = ui._settings.get('max_show_together', 5)
            tog_tuples = [tuple(x) for x in ui._together_pairs]
            all_pinned = all(tuple(p) in tog_tuples for p in pairs)
            tog_label = ("Remove from 'Show Together'"
                         if all_pinned else "Show Together")
            menu.add_command(label=tog_label,
                             command=lambda pp=pairs: ui._toggle_together(pp))
            if ui._together_pairs:
                menu.add_command(
                    label=f"Clear 'Show Together' ({len(ui._together_pairs)} pairs)",
                    command=ui._clear_together)

        # Pair tags (single pair only)
        if n == 1:
            menu.add_separator()
            p = pairs[0]
            _sess, _rt = ui._pair_sess_rt(p)
            has_tags = _rt in self.data._pair_tags
            menu.add_command(
                label=f"{'✓ ' if has_tags else ''}Pair tags…",
                command=self._pair_tags_dialog)

        menu.add_separator()
        menu.add_command(label="Export view as PNG…",
                         command=lambda: ui._export_current_view('png'))
        menu.add_command(label="Export view as PDF…",
                         command=lambda: ui._export_current_view('pdf'))
        menu.tk_popup(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Pair list move / delete actions
    # ------------------------------------------------------------------

    def _ctx_move_to_selected(self, pair):
        if pair is None: return
        self._ctx_move_multi_to_selected([pair])
        self._ui.current_pair_idx = self._ui.get_pair_index(pair)
        self._ui.update_plot()

    def _ctx_move_multi_to_selected(self, pairs):
        if getattr(self._ui, '_session_any_mode', False):
            return
        if not pairs: return
        scroll_top = self.unselected_list.yview()[0]
        self._ui._push_undo()
        for p in pairs:
            self._ui.unselected_inds.discard(p)
            self._ui.selected_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._ui._highlight_changed_pairs(set(pairs))
        self._ui._draw_network()

    def _ctx_move_to_unselected(self, pair):
        if pair is None: return
        self._ctx_move_multi_to_unselected([pair])
        self._ui.current_pair_idx = self._ui.get_pair_index(pair)
        self._ui.update_plot()

    def _ctx_move_multi_to_unselected(self, pairs):
        if getattr(self._ui, '_session_any_mode', False):
            return
        if not pairs: return
        scroll_top = self.selected_list.yview()[0]
        self._ui._push_undo()
        for p in pairs:
            self._ui.selected_inds.discard(p)
            self._ui.unselected_inds.add(p)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self._ui._highlight_changed_pairs(set(pairs))
        self._ui._draw_network()

    def _on_delete_pair(self, event=None):
        """Delete key: toggle current pair in/out of the Deleted section."""
        ui = self._ui
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        if getattr(ui, '_session_any_mode', False):
            trip = self._pair_at_all_inds_idx(ui.current_pair_idx)
            if trip is None:
                return
            if trip not in ui.selected_inds:
                return
            scroll_top = self.selected_list.yview()[0]
            ui._push_undo()
            ui.selected_inds.discard(trip)
            ui.deleted_inds.add(trip)
            hl_old = list(getattr(ui, '_any_pair_handle_list', ()) or ())
            next_trip = None
            for i in range(ui.current_pair_idx + 1, len(hl_old)):
                ck, r, t = hl_old[i]
                tr = (str(ck.session), int(r), int(t))
                if tr in ui.selected_inds:
                    next_trip = tr
                    break
            if next_trip is None:
                for i in range(ui.current_pair_idx):
                    ck, r, t = hl_old[i]
                    tr = (str(ck.session), int(r), int(t))
                    if tr in ui.selected_inds:
                        next_trip = tr
                        break
            self.refresh_lists()
            if next_trip is not None:
                ui.current_pair_idx = ui.get_pair_index(next_trip)
            elif len(ui.all_inds):
                ui.current_pair_idx = min(
                    ui.current_pair_idx, len(ui.all_inds) - 1)
            else:
                ui.current_pair_idx = 0
            self.selected_list.yview_moveto(scroll_top)
            if len(ui.all_inds):
                self._select_pair_in_list(
                    self._pair_at_all_inds_idx(ui.current_pair_idx))
            ui._flush_deleted_to_store()
            ui.network_panel.draw()
            ui.update_plot()
            return

        inds = tuple(int(x) for x in ui.all_inds[ui.current_pair_idx])

        if inds in ui.selected_inds:
            scroll_top = self.selected_list.yview()[0]
            ui._push_undo()
            ui.selected_inds.discard(inds)
            ui.deleted_inds.add(inds)
            n = len(ui.all_inds)
            next_idx = None
            for i in range(ui.current_pair_idx + 1, n):
                if tuple(ui.all_inds[i]) in ui.selected_inds:
                    next_idx = i; break
            if next_idx is None:
                for i in range(ui.current_pair_idx):
                    if tuple(ui.all_inds[i]) in ui.selected_inds:
                        next_idx = i; break
            if next_idx is None:
                for i in range(ui.current_pair_idx + 1, n):
                    if tuple(ui.all_inds[i]) in ui.unselected_inds:
                        next_idx = i; break
            if next_idx is not None:
                ui.current_pair_idx = next_idx
            self.refresh_lists()
            self.selected_list.yview_moveto(scroll_top)
            if next_idx is not None:
                self._select_pair_in_list(tuple(ui.all_inds[next_idx]))
            ui._flush_deleted_to_store()
            ui.network_panel.draw()
            ui.update_plot()
            return

        scroll_top = self.unselected_list.yview()[0]
        ui._push_undo()
        if inds in ui.deleted_inds:
            ui.deleted_inds.discard(inds)
            ui.unselected_inds.add(inds)
            self.refresh_lists()
            self.unselected_list.yview_moveto(scroll_top)
            self._select_pair_in_list(inds)
        else:
            ui.unselected_inds.discard(inds)
            ui.deleted_inds.add(inds)
            n = len(ui.all_inds)
            next_idx = None
            for i in range(ui.current_pair_idx + 1, n):
                if tuple(ui.all_inds[i]) in ui.unselected_inds:
                    next_idx = i
                    break
            if next_idx is None:
                for i in range(ui.current_pair_idx):
                    if tuple(ui.all_inds[i]) in ui.unselected_inds:
                        next_idx = i
                        break
            if next_idx is not None:
                ui.current_pair_idx = next_idx
            self.refresh_lists()
            self.unselected_list.yview_moveto(scroll_top)
            if next_idx is not None:
                self._select_pair_in_list(tuple(ui.all_inds[next_idx]))
        ui._flush_deleted_to_store()
        ui.network_panel.draw()
        ui.update_plot()

    # ------------------------------------------------------------------
    # Group helpers
    # TODO: GroupsManager — all methods below in this section belong in
    #   a future GroupsManager class that owns group CRUD, hotkeys, and
    #   registry management. LeftPanel will hold a GroupsManager instance.
    # ------------------------------------------------------------------

    def _group_pairs(self, gname, session=None):
        """Return pairs set for group in the given session (default: current)."""
        # TODO: GroupsManager
        g = self.data._groups.get(gname, {})
        if isinstance(g, set):
            return g
        sess = session or self._ui._current_session_str()
        return g.get(sess, set())

    def _group_pairs_all_sessions(self, gname):
        # TODO: GroupsManager
        g = self.data._groups.get(gname, {})
        if isinstance(g, set):
            return g
        all_pairs = set()
        for pairs in g.values():
            all_pairs |= pairs
        return all_pairs

    def _group_add_pair(self, gname, pair, session=None):
        # TODO: GroupsManager
        sess = session or self._ui._current_session_str()
        self.data._groups.setdefault(gname, {}).setdefault(sess, set()).add(pair)

    def _group_discard_pair(self, gname, pair, session=None):
        # TODO: GroupsManager
        sess = session or self._ui._current_session_str()
        g = self.data._groups.get(gname, {})
        if isinstance(g, set):
            g.discard(pair)
        elif sess in g:
            g[sess].discard(pair)

    def _pair_in_group(self, pair, group_name: str) -> bool:
        # TODO: GroupsManager
        sess, p2 = self._ui._pair_sess_rt(pair)
        return p2 in self._group_pairs(group_name, session=sess)

    def _pair_group_label(self, inds) -> str:
        # TODO: GroupsManager
        sess, pair = self._ui._pair_sess_rt(inds)
        pair = tuple(int(x) for x in pair)
        labels = []
        for gname in self.data._groups:
            if pair not in self._group_pairs(gname, session=sess):
                continue
            if gname.startswith(_SPECIAL_PREFIX):
                labels.append('*' + gname[len(_SPECIAL_PREFIX):])
            elif not gname.startswith('__'):
                labels.append(gname)
        pt = self.data._pair_tags.get(pair, {})
        tag_mark = '~' if (pt.get('tags') or pt.get('notes', '').strip()) else ''
        group_str = f"[{','.join(labels)}]" if labels else ""
        return tag_mark + group_str

    def _toggle_pair_group(self, pair, group_name):
        # TODO: GroupsManager
        if group_name not in self.data._groups:
            self.data._groups[group_name] = {}
        sess, p2 = self._ui._pair_sess_rt(pair)
        if p2 in self._group_pairs(group_name, session=sess):
            self._group_discard_pair(group_name, p2, session=sess)
        else:
            self._group_add_pair(group_name, p2, session=sess)
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll   = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    def _toggle_pairs_group(self, pairs, group_name):
        # TODO: GroupsManager
        if group_name not in self.data._groups:
            self.data._groups[group_name] = {}
        all_in = all(self._pair_in_group(p, group_name) for p in pairs)
        if all_in:
            for p in pairs:
                s2, p2 = self._ui._pair_sess_rt(p)
                self._group_discard_pair(group_name, p2, session=s2)
        else:
            for p in pairs:
                s2, p2 = self._ui._pair_sess_rt(p)
                self._group_add_pair(group_name, p2, session=s2)
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll   = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    # ------------------------------------------------------------------
    # Group registry helpers
    # TODO: GroupsManager
    # ------------------------------------------------------------------

    def _ensure_group_registered(self, name: str) -> int:
        # TODO: GroupsManager
        data = self.data
        for gid, g in data._group_registry.items():
            if g['name'] == name:
                return gid
        gid = data._next_group_id
        data._group_registry[gid] = {
            'name':   name,
            'hotkey': data._group_hotkeys.get(name),
            'notes':  data._group_notes.get(name, ''),
        }
        data._next_group_id += 1
        return gid

    def _group_id_for(self, name: str) -> int | None:
        # TODO: GroupsManager
        for gid, g in self.data._group_registry.items():
            if g['name'] == name:
                return gid
        return None

    def _sync_registry_from_groups(self):
        # TODO: GroupsManager
        data = self.data
        for name in list(data._groups.keys()):
            self._ensure_group_registered(name)
        for gid, g in data._group_registry.items():
            name = g['name']
            g['hotkey'] = data._group_hotkeys.get(name)
            g['notes']  = data._group_notes.get(name, '')

    # ------------------------------------------------------------------
    # Group dialogs
    # TODO: GroupsManager
    # ------------------------------------------------------------------

    def _create_group_dialog(self):
        # TODO: GroupsManager
        name = simpledialog.askstring(
            "Create group", "Group name:", parent=self._ui.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self.data._groups:
            messagebox.showinfo("Create group",
                                f"Group '{name}' already exists.")
            return
        self.data._groups[name] = {}
        self._rebuild_groups_menu()
        self.refresh_lists()

    def _create_special_group_dialog(self):
        # TODO: GroupsManager
        name = simpledialog.askstring(
            "Create special group", "Special group name:",
            parent=self._ui.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        full_name = _SPECIAL_PREFIX + name
        if full_name in self.data._groups:
            messagebox.showinfo("Create special group",
                                f"Special group '{name}' already exists.")
            return
        self.data._groups[full_name] = {}
        self._rebuild_groups_menu()
        self.refresh_lists()

    def _manage_groups_dialog(self):
        # TODO: GroupsManager
        data = self.data
        if not data._groups:
            messagebox.showinfo("Manage groups",
                                "No groups yet. Create one first.")
            return
        win = tk.Toplevel(self._ui.root)
        win.title("Manage Groups")
        win.geometry("480x420")
        win.transient(self._ui.root)
        win.grab_set()

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def _add_group_tab(nb, gname, is_special=False):
            display = gname[len(_SPECIAL_PREFIX):] if is_special else gname
            frame = ttk.Frame(nb)
            nb.add(frame, text=display)
            top = ttk.Frame(frame)
            top.pack(fill=tk.X, padx=6, pady=4)
            ttk.Label(top, text="Name:").pack(side=tk.LEFT)
            name_var = tk.StringVar(value=display)
            ttk.Entry(top, textvariable=name_var, width=18).pack(
                side=tk.LEFT, padx=4)
            def _do_rename(old=gname, nv=name_var, sp=is_special):
                new = nv.get().strip()
                if sp:
                    new = _SPECIAL_PREFIX + new
                self._rename_group(old, new, win)
            ttk.Button(top, text="Rename", command=_do_rename).pack(side=tk.LEFT)
            if not is_special:
                hk_frame = ttk.Frame(frame)
                hk_frame.pack(fill=tk.X, padx=6, pady=2)
                ttk.Label(hk_frame, text="Hotkey (1–9/0/a–z):").pack(side=tk.LEFT)
                hk_var = tk.StringVar(value=data._group_hotkeys.get(gname, ''))
                hk_entry = ttk.Entry(hk_frame, textvariable=hk_var, width=6)
                hk_entry.pack(side=tk.LEFT, padx=4)
                ttk.Button(hk_frame, text="Set",
                           command=lambda g=gname, hv=hk_var:
                           self._set_group_hotkey(g, hv.get())).pack(side=tk.LEFT)
            ttk.Label(frame,
                      text="Discussion notes:" if is_special else "Notes:"
                      ).pack(anchor='w', padx=6, pady=(4, 0))
            notes_h = 10 if is_special else 3
            notes_text = tk.Text(frame, height=notes_h, width=40,
                                 font=('Arial', 9), wrap=tk.WORD)
            notes_text.pack(fill=tk.BOTH if is_special else tk.X,
                            expand=is_special, padx=6, pady=2)
            notes_text.insert('1.0', data._group_notes.get(gname, ''))
            notes_text.bind('<KeyRelease>',
                            lambda e, g=gname, t=notes_text:
                            data._group_notes.__setitem__(g, t.get('1.0', 'end-1c')))
            ttk.Label(frame, text="Pairs in group:").pack(anchor='w', padx=6)
            lb_frame = ttk.Frame(frame)
            lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
            sb = ttk.Scrollbar(lb_frame)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb = tk.Listbox(lb_frame, yscrollcommand=sb.set, font=('Courier', 9))
            lb.pack(fill=tk.BOTH, expand=True)
            sb.config(command=lb.yview)
            g = data._groups.get(gname, {})
            if isinstance(g, dict):
                for sess in sorted(g):
                    pairs = g[sess]
                    if not pairs:
                        continue
                    ct_map = self._ui._pairs_by_conn_type(sess, pairs)
                    for ct_label, ct_pairs in ct_map.items():
                        lb.insert(tk.END, f"── {sess} | {ct_label} ──")
                        lb.itemconfig(lb.size() - 1, foreground='#666')
                        for pair in sorted(ct_pairs):
                            lb.insert(tk.END, f"  [{pair[0]:3d}, {pair[1]:3d}]")
            else:
                for pair in sorted(g):
                    lb.insert(tk.END, f"[{pair[0]:3d}, {pair[1]:3d}]")
            btn_row = ttk.Frame(frame)
            btn_row.pack(pady=4)
            if is_special:
                conv_label = "Convert to group"
                def _do_convert(g=gname, d=display):
                    self._rename_group(g, d, win)
            else:
                conv_label = "Convert to special group"
                def _do_convert(g=gname, d=display):
                    self._rename_group(g, _SPECIAL_PREFIX + d, win)
            ttk.Button(btn_row, text=conv_label,
                       command=_do_convert).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_row, text=f"Delete group '{display}'",
                       command=lambda g=gname: self._delete_group(g, win)).pack(
                side=tk.LEFT, padx=4)

        for gname in sorted(data._groups):
            if gname.startswith('__'):
                continue
            _add_group_tab(nb, gname, is_special=False)
        special_names = sorted(g for g in data._groups
                               if g.startswith(_SPECIAL_PREFIX))
        if special_names:
            special_frame = ttk.Frame(nb)
            nb.add(special_frame, text="Special")
            snb = ttk.Notebook(special_frame)
            snb.pack(fill=tk.BOTH, expand=True)
            for gname in special_names:
                _add_group_tab(snb, gname, is_special=True)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)

    def _rename_group(self, old_name, new_name, win=None):
        # TODO: GroupsManager
        data = self.data
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if new_name in data._groups:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")
            return
        try:
            gid = self._group_id_for(old_name)
            if gid is not None and gid in data._group_registry:
                data._group_registry[gid]['name'] = new_name
        except Exception:
            pass
        data._groups[new_name] = data._groups.pop(old_name)
        if old_name in data._group_hotkeys:
            if new_name.startswith(_SPECIAL_PREFIX):
                # Special groups don't use hotkeys — drop it on conversion
                data._group_hotkeys.pop(old_name)
                try:
                    gid = self._group_id_for(new_name)
                    if gid is not None and gid in data._group_registry:
                        data._group_registry[gid]['hotkey'] = None
                except Exception:
                    pass
            else:
                data._group_hotkeys[new_name] = data._group_hotkeys.pop(old_name)
        if old_name in data._group_notes:
            data._group_notes[new_name] = data._group_notes.pop(old_name)
        self._rebuild_groups_menu()
        self.refresh_lists()
        if win:
            win.destroy()
            self._manage_groups_dialog()

    def _delete_group(self, name, win=None):
        # TODO: GroupsManager
        if not messagebox.askyesno("Delete group", f"Delete group '{name}'?"):
            return
        data = self.data
        data._groups.pop(name, None)
        data._group_hotkeys.pop(name, None)
        data._group_notes.pop(name, None)
        self._rebuild_groups_menu()
        self.refresh_lists()
        if win:
            win.destroy()
            if data._groups:
                self._manage_groups_dialog()

    def _merge_groups_dialog(self):
        # TODO: GroupsManager
        data = self.data
        if len(data._groups) < 2:
            messagebox.showinfo("Merge groups", "Need at least 2 groups to merge.")
            return
        win = tk.Toplevel(self._ui.root)
        win.title("Merge Groups")
        win.geometry("340x320")
        win.transient(self._ui.root)
        win.grab_set()
        ttk.Label(win, text="Select groups to merge:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)
        check_vars = {}
        for gname in sorted(data._groups):
            if gname.startswith('__'):
                continue
            var = tk.BooleanVar(value=False)
            check_vars[gname] = var
            ttk.Checkbutton(frame, text=gname, variable=var).pack(anchor='w')
        name_frame = ttk.Frame(win)
        name_frame.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(name_frame, text="Merged group name:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, width=20)
        name_entry.pack(side=tk.LEFT, padx=4)
        def do_merge():
            selected = [g for g, v in check_vars.items() if v.get()]
            if len(selected) < 2:
                messagebox.showwarning("Merge", "Select at least 2 groups.")
                return
            target = name_entry.get().strip() or selected[0]
            if not messagebox.askokcancel(
                    "Merge groups",
                    f"This will merge {len(selected)} groups into '{target}'.\n"
                    "This cannot be undone. Proceed?"):
                return
            merged = {}
            for g in selected:
                g_data = data._groups.get(g, {})
                if isinstance(g_data, set):
                    sess = self._ui._current_session_str()
                    merged.setdefault(sess, set()).update(g_data)
                else:
                    for sess, pairs in g_data.items():
                        merged.setdefault(sess, set()).update(pairs)
                if g != target:
                    data._groups.pop(g, None)
                    data._group_hotkeys.pop(g, None)
                    data._group_notes.pop(g, None)
            data._groups[target] = merged
            self._rebuild_groups_menu()
            self.refresh_lists()
            win.destroy()
        ttk.Button(win, text="Merge", command=do_merge).pack(pady=8)

    # ------------------------------------------------------------------
    # Hotkey methods
    # TODO: GroupsManager
    # ------------------------------------------------------------------

    def _set_group_hotkey(self, group_name, key_str):
        # TODO: GroupsManager
        key_str = key_str.strip().lower()
        if not key_str:
            self.data._group_hotkeys.pop(group_name, None)
            self._rebuild_groups_menu()
            return
        valid_digits = [str(i) for i in range(1, 10)] + ['0']
        if (key_str not in valid_digits
                and not (len(key_str) == 1 and key_str.isalpha())):
            messagebox.showwarning(
                "Hotkey", "Enter a digit 1–9/0 or a single letter a–z.")
            return
        for g, k in list(self.data._group_hotkeys.items()):
            if k == key_str and g != group_name:
                del self.data._group_hotkeys[g]
        self.data._group_hotkeys[group_name] = key_str
        self._rebuild_groups_menu()

    def _rebuild_groups_menu(self):
        # TODO: GroupsManager
        ui = self._ui
        if not hasattr(ui, '_groups_menu'):
            return
        try:
            while ui._groups_menu.index('end') >= 7:
                ui._groups_menu.delete(7)
        except tk.TclError:
            pass
        current_pairs = (set(map(tuple, ui.all_inds))
                         if len(ui.all_inds) else set())
        special_groups = [g for g in self.data._groups if g.startswith(_SPECIAL_PREFIX)]
        if special_groups:
            ui._groups_menu.add_separator()
            special_menu = tk.Menu(ui._groups_menu, tearoff=0)
            for gname in special_groups:
                display = gname[len(_SPECIAL_PREFIX):]
                n = len(self._group_pairs(gname) & current_pairs)
                special_menu.add_command(
                    label=f"{display} ({n})",
                    command=lambda g=gname: self._select_group(g))
            ui._groups_menu.add_cascade(label="Special", menu=special_menu)
        if hasattr(ui, 'network_panel'):
            ui.network_panel.refresh_group_buttons()
        if (hasattr(ui, '_hotkeys_bar')
                and ui._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get()):
            ui._refresh_hotkeys_bar()

    def _select_group(self, group_name):
        # TODO: GroupsManager
        pairs = self._group_pairs(group_name)
        if not pairs:
            return
        first = sorted(pairs)[0]
        self._ui.current_pair_idx = self._ui.get_pair_index(first)
        self._ui.update_plot()
        self._ui._draw_network()

    def _group_hotkey_handler(self, key_str: str, advance: bool = True):
        # TODO: GroupsManager
        """Toggle the current pair in/out of the group assigned to key_str."""
        ui = self._ui
        current_pair = self._selected_pair_from_lists()
        if current_pair is None:
            if ui.current_pair_idx >= len(ui.all_inds):
                return
            if getattr(ui, '_session_any_mode', False):
                trip = self._pair_at_all_inds_idx(ui.current_pair_idx)
                if trip is None:
                    return
                current_pair = trip
            else:
                row = ui.all_inds[ui.current_pair_idx]
                current_pair = tuple(int(x) for x in row)

        for gname, k in self.data._group_hotkeys.items():
            if k != key_str:
                continue
            if not advance:
                highlighted = [current_pair]
                ui._shift_tag_pending_advance = True
            else:
                avail_map = getattr(self, '_avail_list_pairs', None)
                highlighted = []
                for i in self.unselected_list.curselection():
                    if avail_map and i < len(avail_map):
                        entry = avail_map[i]
                        if entry is None or entry[0] == _AVAIL_GROUP_HDR:
                            continue
                        if entry[1] != 'deleted':
                            highlighted.append(entry[0])
                    elif not avail_map:
                        su = sorted(self.data.unselected_inds)
                        if i < len(su):
                            highlighted.append(su[i])
                sel_map = getattr(self, '_sel_list_pairs', None)
                for i in self.selected_list.curselection():
                    if sel_map and i < len(sel_map):
                        inds = sel_map[i]
                        if inds is not None:
                            highlighted.append(inds)
                    elif not sel_map:
                        ss = sorted(self.data.selected_inds)
                        if i < len(ss):
                            highlighted.append(ss[i])
                if not highlighted:
                    highlighted = [current_pair]

            changed = set()
            ui._push_undo()

            for pair in highlighted:
                was_in_group = self._pair_in_group(pair, gname)
                self._toggle_pair_group(pair, gname)
                if getattr(ui, '_session_any_mode', False):
                    changed.add(pair)
                    continue
                if not was_in_group:
                    if pair in self.data.unselected_inds:
                        self.data.unselected_inds.discard(pair)
                        self.data.selected_inds.add(pair)
                        changed.add(pair)
                else:
                    if pair in self.data.selected_inds:
                        has_groups = any(
                            self._pair_in_group(pair, g)
                            for g in self.data._groups
                            if not g.startswith('__')
                        )
                        if not has_groups:
                            self.data.selected_inds.discard(pair)
                            self.data.unselected_inds.add(pair)
                            changed.add(pair)

            self.refresh_lists()
            ui._highlight_changed_pairs(changed or {current_pair})
            if advance:
                next_idx = min(ui.current_pair_idx + 1, len(ui.all_inds) - 1)
                ui.current_pair_idx = next_idx
                self._select_pair_in_list(self._pair_at_all_inds_idx(next_idx))
            else:
                self._select_pair_in_list(current_pair)
            ui.update_plot()
            ui.network_panel.draw()
            return
        ui._show_temp_warning(f"No group assigned to Ctrl+{key_str}")

    def _export_groups(self):
        # TODO: GroupsManager
        data = self.data
        if not data._groups:
            messagebox.showinfo("Export groups", "No groups to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export groups",
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
            initialfile='groups_export.json',
            initialdir=self._ui._sel_save_dir,
        )
        if not path:
            return
        export_data = data.serialize_groups(self._ui._current_session_str())
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=_json_default)
        print(f"[LeftPanel] groups exported → {path}")

    def _import_groups(self):
        # TODO: GroupsManager
        path = filedialog.askopenfilename(
            title="Import groups",
            filetypes=[('JSON files', '*.json')],
            initialdir=self._ui._sel_save_dir,
        )
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            messagebox.showerror("Import groups", f"Failed to read file:\n{exc}")
            return
        self.data.deserialize_groups(
            raw, self._ui._current_session_str(), restore_hotkeys=False)
        for gname, hk in raw.get('hotkeys', {}).items():
            if gname not in self.data._group_hotkeys:
                self.data._group_hotkeys[gname] = hk
        self._rebuild_groups_menu()
        self.refresh_lists()
        print(f"[LeftPanel] groups imported from {path}")

    # ------------------------------------------------------------------
    # Pair tags dialog
    # ------------------------------------------------------------------

    def _pair_tags_dialog(self):
        ui = self._ui
        if ui.current_pair_idx >= len(ui.all_inds):
            messagebox.showinfo("Pair tags", "No pair selected.")
            return
        inds = tuple(ui.all_inds[ui.current_pair_idx])
        ref, tgt = int(inds[0]), int(inds[1])
        tag_data = self.data._pair_tags.get((ref, tgt), {})
        win = tk.Toplevel(ui.root)
        win.title(f"Pair Tags — [{ref}, {tgt}]")
        win.geometry("400x350")
        win.transient(ui.root)
        win.grab_set()
        ttk.Label(win, text="Tags (comma-separated):").pack(
            anchor='w', padx=8, pady=(8, 0))
        tags_var = tk.StringVar(value=', '.join(tag_data.get('tags', [])))
        ttk.Entry(win, textvariable=tags_var, width=50).pack(
            fill=tk.X, padx=8, pady=2)
        ttk.Label(win, text="Notes:").pack(anchor='w', padx=8, pady=(8, 0))
        notes_text = tk.Text(win, height=12, width=50, font=('Arial', 9),
                             wrap=tk.WORD)
        notes_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        notes_text.insert('1.0', tag_data.get('notes', ''))
        def _save():
            tags = [t.strip() for t in tags_var.get().split(',') if t.strip()]
            notes = notes_text.get('1.0', 'end-1c')
            if tags or notes:
                existing_groups = self.data._pair_tags.get((ref, tgt), {}).get('groups', [])
                entry = {'tags': tags, 'notes': notes}
                if existing_groups:
                    entry['groups'] = existing_groups
                self.data._pair_tags[(ref, tgt)] = entry
            elif (ref, tgt) in self.data._pair_tags:
                del self.data._pair_tags[(ref, tgt)]
            self.refresh_lists()
            self._select_pair_in_list((ref, tgt))
            win.destroy()
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_frame, text="Save", command=_save).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=win.destroy).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Speculated predictions
    # ------------------------------------------------------------------

    def _clear_speculated(self):
        """Discard all pending speculated predictions and refresh."""
        getattr(self._ui, '_speculated_groups', {}).clear()
        self.refresh_lists()


# ---------------------------------------------------------------------------
# SpikePairsPanel — Spike Pairs tab content
# ---------------------------------------------------------------------------

class SpikePairsPanel:
    """Content of the 'Spike Pairs' tab.

    Subsidiary to LeftPanel: shares the notebook and the disabled tab frame
    that LeftPanel built. Owns the listbox widget and fires callbacks on click.

    Never use 'sa' abbreviations — always spell out 'spike_pairs'.
    """

    def __init__(self, notebook: ttk.Notebook,
                 spike_pairs_tab: ttk.Frame,
                 data: SelectionData,
                 ui):
        self.data = data
        self._ui  = ui
        self._notebook         = notebook
        self._tab              = spike_pairs_tab
        self._spike_pairs_tab_index = 1   # matches LeftPanel._spike_pairs_tab_index

        self._spike_pairs: list = []
        self._selected_idx: int = -1

        self._build()

    def _build(self):
        # ── Spike Pairs tab ──────────────────────────────────────────────
        self._spike_pairs_count = tk.StringVar(value="")
        ttk.Label(self._tab, textvariable=self._spike_pairs_count,
                  font=('Courier', 9)).pack(side=tk.TOP, anchor='w',
                                            padx=4, pady=2)
        scroll = ttk.Scrollbar(self._tab)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._spike_pairs_listbox = tk.Listbox(
            self._tab, yscrollcommand=scroll.set,
            selectmode=tk.BROWSE, font=('Courier', 9), activestyle='none')
        self._spike_pairs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self._spike_pairs_listbox.yview)
        self._spike_pairs_listbox.bind(
            '<<ListboxSelect>>', self._on_spike_pair_click)

    def populate(self, spike_pairs: list):
        """Fill the listbox with computed spike pairs.

        Called by CCGReviewUI after _compute_spike_pairs.
        """
        self._spike_pairs    = spike_pairs
        self._selected_idx   = -1
        self._spike_pairs_listbox.delete(0, tk.END)
        for i, (rt, tt) in enumerate(spike_pairs):
            lag_ms = (tt - rt) * 1000.0
            self._spike_pairs_listbox.insert(
                tk.END,
                f"{i+1:>5}  ref {rt:10.4f}  tgt {tt:10.4f}  lag {lag_ms:+6.2f}ms")
        self._spike_pairs_count.set(f"{len(spike_pairs)} spike pairs")

    def clear(self):
        """Disable tab and clear list (called when spike attribution is toggled off)."""
        self._notebook.tab(self._spike_pairs_tab_index, state='disabled')
        self._notebook.select(0)
        self._spike_pairs    = []
        self._selected_idx   = -1
        self._spike_pairs_count.set("")

    def activate(self):
        """Enable tab and switch to it."""
        self._notebook.tab(self._spike_pairs_tab_index, state='normal')
        self._notebook.select(self._spike_pairs_tab_index)

    def _on_spike_pair_click(self, _event=None):
        sel = self._spike_pairs_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._spike_pairs):
            return
        self._selected_idx = idx
        self._ui.center_container.spike_attribution_panel._draw_spike_pairs_raster(idx, self._spike_pairs)


# ---------------------------------------------------------------------------
# LeftPanelContainer — wires LeftPanel and SpikePairsPanel
# ---------------------------------------------------------------------------

class LeftPanelContainer:
    """Thin wrapper that owns LeftPanel and SpikePairsPanel.

    Exposes ``.widget`` (the shared notebook) for CCGReviewUI to pack.
    Access sub-panels via ``left_panel`` and ``spike_pairs``.
    """

    def __init__(self, parent: tk.Widget, data: SelectionData,
                 ui, ui_state_cache: dict):
        self.data = data
        self.left_panel = LeftPanel(parent, data, ui, ui_state_cache)
        self.spike_pairs = SpikePairsPanel(
            self.left_panel.notebook,
            self.left_panel._spike_pairs_tab,
            data, ui)
        self.widget = self.left_panel.notebook


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON encoder for numpy scalar types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')
