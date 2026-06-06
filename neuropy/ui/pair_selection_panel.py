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

import random
import re
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox, filedialog
from collections import defaultdict as _defaultdict
from typing import Callable

import numpy as np

from neuropy.ui.utils import (
    _SPECIAL_PREFIX, _SEPARATOR_ROW,
    is_special_group, is_separator_row,
    CollapseState, group_header_label, SelectionCommand, BiIndex,
)
from neuropy.analyses.utils import Savable
from neuropy.ui.dialogs import (PairTagsDialog, MergeGroupsDialog,
                                QuickSaveDialog, LoadSelectionDialog, MissingPairsDialog,
                                CreateGroupDialog, ExportGroupsDialog, ImportGroupsDialog)

@dataclass
class Group:
    name: str
    hotkey: str = ''
    notes: str = ''


class SelectionData(Savable, BiIndex):
    """Pair selection and group/tag state.

    Extends BiIndex: forward key = gname, value = (sess, ref, tgt).
    self.forward(gname)          → set of (sess, ref, tgt) across all sessions
    self.inverse((sess, ref, tgt)) → set of gnames  — O(1)
    """

    _save_format = 'json'

    def __init__(self):
        Savable.__init__(self)
        BiIndex.__init__(self)
        self.selected_inds:   set = set()
        self.unselected_inds: set = set()
        self.deleted_inds:    set = set()
        self._pair_tags: dict = {}
        self._group_metadata: dict[str, Group] = {}

    def add_to_group(self, gname: str, sess: str, pair: tuple) -> None:
        self.add(gname, (sess, int(pair[0]), int(pair[1])))
        self.get_group_metadata(gname)

    def discard_from_group(self, gname: str, sess: str, pair: tuple) -> None:
        self.discard(gname, (sess, int(pair[0]), int(pair[1])))

    def groups_for_pair(self, sess: str, ref: int, tgt: int) -> set:
        """O(1) inverse — which groups contain (sess, ref, tgt)."""
        return self.inverse((sess, int(ref), int(tgt)))

    def pairs_in_group(self, gname: str, sess: str) -> set:
        """(ref,tgt) pairs for group+session. Filters forward index by sess."""
        return {(r, t) for s, r, t in self.forward(gname) if s == sess}

    def sessions_for_group(self, gname: str) -> set:
        """All sessions that have pairs in this group."""
        return {s for s, *_ in self.forward(gname)}

    @property
    def groups(self) -> dict:
        """Forward index: gname → set((sess, ref, tgt)). Read-only alias for _fwd."""
        return self._fwd

    def pack(self) -> dict:
        """Full serializable state: selections + groups + pair_tags."""
        groups: dict = {}
        for gname, members in self._fwd.items():
            for sess, ref, tgt in members:
                groups.setdefault(gname, {}).setdefault(sess, []).append([ref, tgt])
        return {
            'groups':    groups,
            'hotkeys':   {n: g.hotkey for n, g in self._group_metadata.items() if g.hotkey},
            'notes':     {n: g.notes  for n, g in self._group_metadata.items() if g.notes},
            'selected':  [list(p) for p in sorted(self.selected_inds)],
            'unselected':[list(p) for p in sorted(self.unselected_inds)],
            'deleted':   [list(p) for p in sorted(self.deleted_inds)],
            'pair_tags': [[list(k), v] for k, v in self._pair_tags.items()],
        }

    def __setstate__(self, state: dict):
        self.__init__()
        self.selected_inds   = {tuple(int(x) for x in p)
                                 for p in state.get('selected', [])}
        self.unselected_inds = {tuple(int(x) for x in p)
                                 for p in state.get('unselected', [])}
        self.deleted_inds    = {tuple(int(x) for x in p)
                                 for p in state.get('deleted', [])}
        for k, v in state.get('pair_tags', []):
            self._pair_tags[tuple(int(x) for x in k)] = v
        for g, val in state.get('groups', {}).items():
            if isinstance(val, list):
                sess = state.get('file_session', '')
                for pair in val:
                    self.add_to_group(g, sess, pair)
            elif isinstance(val, dict):
                for sess, pp in val.items():
                    for pair in pp:
                        self.add_to_group(g, sess, pair)
            self.get_group_metadata(g)
        for name, hk in state.get('hotkeys', {}).items():
            self.get_group_metadata(name).hotkey = hk
        for name, note in state.get('notes', {}).items():
            grp = self.get_group_metadata(name)
            if not grp.notes:
                grp.notes = note

    def set_state(self, pair: tuple, state: str):
        pair = tuple(pair)
        self.selected_inds.discard(pair)
        self.unselected_inds.discard(pair)
        self.deleted_inds.discard(pair)
        if state == 'sel':
            self.selected_inds.add(pair)
        elif state == 'unsel':
            self.unselected_inds.add(pair)
        elif state == 'del':
            self.deleted_inds.add(pair)

    def populate(self, all_pairs, selected=(), deleted=()):
        """Rebuild pair state from scratch for a new session."""
        all_set  = {tuple(p) for p in all_pairs}
        sel_set  = {tuple(p) for p in selected}
        del_set  = {tuple(p) for p in deleted}
        self.selected_inds   = sel_set & all_set
        self.deleted_inds    = del_set & all_set
        self.unselected_inds = all_set - self.selected_inds - self.deleted_inds

    def get_group_metadata(self, name: str) -> Group:
        """Return Group metadata for name, creating one if absent."""
        if name not in self._group_metadata:
            self._group_metadata[name] = Group(name=name)
        return self._group_metadata[name]

    def rename_group(self, old_name: str, new_name: str) -> None:
        """Rename a group: updates BiIndex key and Group.name in _group_metadata."""
        self.rename_key(old_name, new_name)
        grp = self._group_metadata.pop(old_name, None)
        if grp is not None:
            grp.name = new_name
            if is_special_group(new_name):
                grp.hotkey = ''
            self._group_metadata[new_name] = grp

    def delete_group(self, name: str) -> None:
        """Remove a group from BiIndex and _group_metadata."""
        self.delete_key(name)
        self._group_metadata.pop(name, None)

    def set_hotkey(self, name: str, key_str: str) -> None:
        """Assign hotkey to group, clearing any prior holder of that key."""
        for grp in self._group_metadata.values():
            if grp.hotkey == key_str and grp.name != name:
                grp.hotkey = ''
        self.get_group_metadata(name).hotkey = key_str


class GroupManager:
    """Group + selection management, persistence, and undo/redo for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui._pair_deleted_store = {}

    def _valid_group_pairs_as_triples(self, gname: str) -> set[tuple]:
        """All (session, ref, tgt) in *gname* for the current connection type."""
        ui = self._ui
        lbl = ui._type_label(ui.key)
        out: set[tuple] = set()
        for k in ui.cd.ptr.keys():
            if ui._type_label(k) != lbl:
                continue
            sess = str(k.session)
            valid = ui._all_inds_set_for_ptr(ui.cd.ptr.get(k))
            if not valid:
                continue
            for pair in ui._sel_data.pairs_in_group(gname, sess):
                r, t = int(pair[0]), int(pair[1])
                if (r, t) in valid:
                    out.add((sess, r, t))
        return out

    def _any_nd_keys_for_group(self, gname: str) -> list:
        """Neuron-dataset keys that have ≥1 pair in *gname* (current type)."""
        ui = self._ui
        lbl = ui._type_label(ui.key)
        seen, seen_id = [], set()
        for nk in ui._sess_mgr._real_nd_keys_ordered():
            ckey = ui._sess_mgr._type_key_for_nd(nk)
            if ckey is None or ui._type_label(ckey) != lbl:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.ptr.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            if any((int(a), int(b)) in valid
                   for a, b in ui._sel_data.pairs_in_group(gname, sess)):
                nid = id(nk)
                if nid not in seen_id:
                    seen.append(nk)
                    seen_id.add(nid)
        return seen

    def _iter_group_pairs(self, gname: str):
        """Yield ``(ckey, r, t)`` for *gname* by scanning sessions (expanded tag only)."""
        ui = self._ui
        if gname not in ui._any_expanded_group_tags:
            return
        lbl = ui._type_label(ui.key)
        dead = ui.deleted_inds
        for nk in ui._sess_mgr._real_nd_keys_ordered():
            ckey = ui._sess_mgr._type_key_for_nd(nk)
            if ckey is None or ui._type_label(ckey) != lbl:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.ptr.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            pairs = ui._sel_data.pairs_in_group(gname, sess)
            if not pairs:
                continue
            # Precompute CCG bounds to skip stale pairs (pointer may outlive data)
            _cd = ui.cd.ccg.get(ckey.nd()) if hasattr(ui.cd, 'ccg') else None
            _ccg_sh = (_cd.ccg.shape if _cd is not None and hasattr(_cd, 'ccg')
                       and _cd.ccg is not None else None)
            for r, t in sorted((int(a), int(b)) for a, b in pairs):
                if r == t:
                    continue
                if (r, t) not in valid:
                    continue
                if (sess, r, t) in dead:
                    continue
                if (_ccg_sh is not None and len(_ccg_sh) >= 4
                        and (r >= _ccg_sh[1] or t >= _ccg_sh[2])):
                    continue
                yield ckey, r, t

    def _toggle_any_avail_group(self, gname: str):
        """Expand/collapse a group tag (Any mode); load CCG for involved sessions."""
        ui = self._ui
        if gname in ui._any_expanded_group_tags:
            ui._any_expanded_group_tags.discard(gname)
            ui.refresh_lists()
            return

        nds = ui._group_mgr._any_nd_keys_for_group(gname)

        def _finish_expand():
            ui._any_expanded_group_tags.add(gname)
            ui.refresh_lists()

        if not nds:
            _finish_expand()
            return

        def _chain(idx: int):
            if idx >= len(nds):
                _finish_expand()
                return
            ui._sess_mgr._ensure_session_loaded(nds[idx], on_loaded=lambda: _chain(idx + 1))

        _chain(0)

    def _select_group(self, group_name):
        """Navigate to the first pair in the group."""
        ui = self._ui
        sess = ui._setup_mgr._current_session_str()
        pairs = ui._sel_data.pairs_in_group(group_name, sess)
        if not pairs:
            return
        first = sorted(pairs)[0]
        ui.current_pair_idx = ui._plot_mgr.get_pair_index(first)
        ui._plot_mgr.update_plot()
        ui.network_panel.draw()

    def _rename_group(self, old_name, new_name, win=None):
        ui = self._ui
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if new_name in ui._sel_data.groups:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")
            return
        ui._sel_data.rename_group(old_name, new_name)
        ui._group_mgr._rebuild_groups_menu()
        ui.refresh_lists()
        if win:
            win.destroy()
            ui._manage_groups_dialog()

    def _delete_group(self, name, win=None):
        ui = self._ui
        if not messagebox.askyesno("Delete group",
                                   f"Delete group '{name}'?"):
            return
        ui._sel_data.delete_group(name)
        ui._group_mgr._rebuild_groups_menu()
        ui.refresh_lists()
        if win:
            win.destroy()
            # Reopen if there are remaining groups
            if ui._sel_data.groups:
                ui._manage_groups_dialog()

    def setup_groups_menu(self, menubar):
        """Groups menu: create / manage pair groups."""
        ui = self._ui
        ui._groups_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Groups", menu=ui._groups_menu)
        ui._groups_menu.add_command(label="Create group…",
                                      command=lambda: CreateGroupDialog.show(ui))
        ui._groups_menu.add_command(label="Manage groups…",
                                      command=ui._manage_groups_dialog)
        ui._groups_menu.add_command(label="Merge groups…",
                                      command=lambda: MergeGroupsDialog.show(ui))
        ui._groups_menu.add_command(label="Export groups…",
                                      command=lambda: ExportGroupsDialog.show(ui))
        ui._groups_menu.add_command(label="Import groups…",
                                      command=lambda: ImportGroupsDialog.show(ui))
        ui._groups_menu.add_separator()
        # Dynamic group entries added in _rebuild_groups_menu()

    def _rebuild_groups_menu(self):
        """Refresh the dynamic part of the Groups menu."""
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._rebuild_groups_menu()
            return
        # Pre-container fallback (during early setup before left_container exists)
        if not hasattr(ui, '_groups_menu'):
            return
        try:
            while ui._groups_menu.index('end') >= 7:
                ui._groups_menu.delete(7)
        except tk.TclError:
            pass
        ui.network_panel.refresh_group_buttons()
        if (hasattr(ui, 'hotkeys_bar') and
                ui._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get()):
            ui.hotkeys_bar.refresh()

    def setup_group_hotkeys_bar(self):
        """Create HotkeysBar and pack if panel is visible."""
        ui = self._ui
        ui.hotkeys_bar = HotkeysBar(ui.root, ui)
        if ui._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get():
            ui.hotkeys_bar.frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2))

    def _collect_highlighted_pairs(self, current_pair: tuple) -> list:
        """Return pairs highlighted in both listboxes, falling back to current_pair."""
        ui = self._ui
        _lp = getattr(ui.left_container, 'left_panel', None)
        avail_map = getattr(_lp, '_avail_list_pairs', None)
        sel_map   = getattr(_lp, '_sel_list_pairs',   None)
        result = []
        for i in ui.unselected_list.curselection():
            if avail_map and i < len(avail_map):
                entry = avail_map[i]
                if entry is not None and not is_separator_row(entry) and entry[1] != 'deleted':
                    result.append(entry[0])
            elif not avail_map:
                su = sorted(ui.unselected_inds)
                if i < len(su):
                    result.append(su[i])
        for i in ui.selected_list.curselection():
            if sel_map and i < len(sel_map):
                inds = sel_map[i]
                if inds is not None:
                    result.append(inds)
            elif not sel_map:
                ss = sorted(ui.selected_inds)
                if i < len(ss):
                    result.append(ss[i])
        return result or [current_pair]

    def _group_hotkey_handler(self, key_str: str, advance: bool = True):
        """Toggle current/highlighted pairs in/out of the group bound to key_str."""
        ui = self._ui
        current_pair = ui._selected_pair_from_lists()
        if current_pair is None:
            if ui.current_pair_idx >= len(ui.all_inds):
                return
            if getattr(ui, '_session_any_mode', False):
                current_pair = ui._pair_at_all_inds_idx(ui.current_pair_idx)
                if current_pair is None:
                    return
            else:
                current_pair = tuple(int(x) for x in ui.all_inds[ui.current_pair_idx])

        for _grp in ui._sel_data._group_metadata.values():
            gname, k = _grp.name, _grp.hotkey
            if not k or k != key_str:
                continue

            if not advance:
                highlighted = [current_pair]
                ui._shift_tag_pending_advance = True
            else:
                highlighted = self._collect_highlighted_pairs(current_pair)

            changed = set()
            pair_changes, group_changes = {}, []
            any_mode = getattr(ui, '_session_any_mode', False)

            for pair in highlighted:
                old = ('sel' if pair in ui._sel_data.selected_inds
                       else 'del' if pair in ui._sel_data.deleted_inds
                       else 'unsel')
                sess, p2 = ui._pair_sess_rt(pair)
                was_in = p2 in ui._sel_data.pairs_in_group(gname, sess)
                group_changes.append((gname, sess, p2, 'remove' if was_in else 'add'))
                if was_in:
                    ui._sel_data.discard_from_group(gname, sess, p2)
                else:
                    ui._sel_data.add_to_group(gname, sess, p2)
                if any_mode:
                    changed.add(pair)
                    continue
                if not was_in and pair in ui.unselected_inds:
                    ui._sel_data.set_state(pair, 'sel')
                    pair_changes[pair] = (old, 'sel')
                    changed.add(pair)
                elif was_in and pair in ui.selected_inds:
                    has_groups = any(
                        p2 in ui._sel_data.pairs_in_group(g, sess)
                        for g in ui._sel_data.groups
                        if not g.startswith('__')
                    )
                    if not has_groups:
                        ui._sel_data.set_state(pair, 'unsel')
                        pair_changes[pair] = (old, 'unsel')
                        changed.add(pair)

            ui._group_mgr._push_undo(SelectionCommand(pair_changes, group_changes))
            ui.refresh_lists()
            ui._plot_mgr._highlight_changed_pairs(changed or {current_pair})
            if advance:
                next_idx = min(ui.current_pair_idx + 1, len(ui.all_inds) - 1)
                ui.current_pair_idx = next_idx
                ui._select_pair_in_list(ui._pair_at_all_inds_idx(next_idx))
            else:
                ui._select_pair_in_list(current_pair)
            ui._plot_mgr.update_plot()
            ui.network_panel.draw()
            return
        ui._group_mgr._show_temp_warning(f"No group assigned to Ctrl+{key_str}")

    def _set_group_hotkey(self, group_name, key_str):
        """Assign hotkey: single digit 1–9/0 or single letter a–z."""
        ui = self._ui
        key_str = key_str.strip().lower()
        valid_digits = [str(i) for i in range(1, 10)] + ['0']
        if key_str and key_str not in valid_digits and not (len(key_str) == 1 and key_str.isalpha()):
            messagebox.showwarning("Hotkey", "Enter a digit 1–9/0 or a single letter a–z.")
            return
        ui._sel_data.set_hotkey(group_name, key_str)
        ui._group_mgr._rebuild_groups_menu()

    def _show_hotkeys_dialog(self):
        """Show a dialog listing all keyboard shortcuts."""
        ui = self._ui
        hotkeys_text = (
            "Ctrl+E    Toggle between waveform and CCG\n"
            "Ctrl+R    Toggle resolution (hi / lo)\n"
            "Ctrl+S    Save selection\n"
            "Ctrl+B    Toggle bookmark on current pair (pin + highlight in lists)\n"
            "Ctrl+Z    Undo\n"
            "Ctrl+Y    Redo\n"
            "\n"
            "1..0          Assign group + advance cursor\n"
            "Shift+1..0    Assign group(s) to current pair (no advance)\n"
            "Ctrl+Delete / Ctrl+Backspace   Move current pair to Deleted"
        )
        messagebox.showinfo("Keyboard Shortcuts", hotkeys_text)

    def setup_file_menu(self, menubar):
        """Selections menu: save / load selection versions."""
        ui = self._ui
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Selections", menu=file_menu)
        file_menu.add_command(label="Save selection…",
                              command=lambda: QuickSaveDialog.show(self._ui))
        file_menu.add_command(label="Load selection…",
                              command=lambda: LoadSelectionDialog.show(self._ui))
        file_menu.add_separator()
        file_menu.add_command(label="Export as PNG…",
                              command=ui._export_mgr._export_current_view)
        file_menu.add_separator()
        file_menu.add_command(label="Clear bookmarks",
                              command=self._clear_bookmarks)

    def _selections_menu_close(self):
        ui = self._ui
        ui._bookmarked_pairs.clear()
        ui.root.destroy()

    def _push_undo(self, cmd: 'SelectionCommand') -> None:
        ui = self._ui
        ui._undo_stack.append(cmd)
        if len(ui._undo_stack) > ui._UNDO_LIMIT:
            ui._undo_stack.pop(0)
        ui._redo_stack.clear()

    def _apply_cmd(self, cmd: 'SelectionCommand', reverse: bool = False) -> None:
        ui = self._ui
        for pair, (old, new) in cmd.pair_changes.items():
            target = old if reverse else new
            if target is None:
                ui._sel_data.unselected_inds.discard(pair)
            else:
                ui._sel_data.set_state(pair, target)
        for g, s, p, op in cmd.group_changes:
            effective = ('remove' if op == 'add' else 'add') if reverse else op
            if effective == 'add':
                ui._group_mgr._group_add_pair(g, p, s)
            else:
                ui._group_mgr._group_discard_pair(g, p, s)
        changed = set(cmd.pair_changes.keys())
        if changed:
            ui._left_panel.refresh_lists()
            ui._plot_mgr._highlight_changed_pairs(changed)
            ui._draw_network()
        ui._flush_deleted_to_store()
        ui._plot_mgr.update_plot()
        ui.network_panel.draw()
        ui._refresh_stats()

    # Highlight color for undo/redo indicators (matches CCG baseline orange)
    _UNDO_HIGHLIGHT = '#ff7f0e'

    def _undo(self, event=None):
        ui = self._ui
        if not ui._undo_stack:
            return
        cmd = ui._undo_stack.pop()
        ui._redo_stack.append(cmd)
        self._apply_cmd(cmd, reverse=True)

    def _redo(self, event=None):
        ui = self._ui
        if not ui._redo_stack:
            return
        cmd = ui._redo_stack.pop()
        ui._undo_stack.append(cmd)
        self._apply_cmd(cmd, reverse=False)

    def _clear_undo_highlight(self):
        """Remove undo/redo highlight from all list items."""
        ui = self._ui
        for listbox in (ui.unselected_list, ui.selected_list):
            for idx in range(listbox.size()):
                listbox.itemconfig(idx, background='', foreground='')

    def _reapply_bookmark_list_styles(self):
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._reapply_bookmark_list_styles()

    def _bookmark_toggle_current(self, event=None):
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._bookmark_toggle_current(event)

    def _clear_bookmarks(self):
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._clear_bookmarks()

    def _history_dir(self) -> str:
        ui = self._ui
        return os.path.join(ui._sel_save_dir, ui._HISTORY_SUBDIR)

    def _save_to_history(self, data: dict, suffix: str) -> str:
        """Write data dict to .history/{session}__{ts}{suffix}.json and git-commit."""
        ui = self._ui
        hdir = self._history_dir()
        os.makedirs(hdir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        sess = getattr(ui.key, 'session', 'sess')
        fname = f"{sess}__{ts}{suffix}.json"
        path = os.path.join(hdir, fname)
        ui._atomic_write_json(path, data)
        ui._git_commit_paths([path], f'[history] {fname}')
        return path

    def _purge_history(self):
        """Delete .history/ files older than 3 days and commit the deletion."""
        ui = self._ui
        hdir = self._history_dir()
        if not os.path.isdir(hdir):
            return
        cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
        removed = []
        for fname in os.listdir(hdir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(hdir, fname)
            try:
                if datetime.datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
                    os.remove(fpath)
                    removed.append(fpath)
            except OSError:
                pass
        if removed:
            repo = os.path.abspath(os.path.join(ui._sel_save_dir, '..', '..'))
            for p in removed:
                subprocess.run(['git', 'rm', '--cached', '-f',
                                os.path.relpath(p, repo)],
                               cwd=repo, capture_output=True)
            subprocess.run(['git', 'commit', '--no-gpg-sign', '-m',
                            f'[history] purge {len(removed)} files older than 7 days'],
                           cwd=repo, capture_output=True)
            print(f"[CCGReviewUI] purged {len(removed)} history files older than 7 days")

    def _save_autosnapshot(self):
        """Periodic 15-min autosave to .history/ as .autosaved.json."""
        ui = self._ui
        if ui.ccg_ptr is None or getattr(ui, '_closing', False):
            return
        try:
            data = self._build_save_dict(
                datetime.datetime.now().isoformat(), 'autosaved')
            self._save_to_history(data, '.autosaved')
            print(f"[CCGReviewUI] autosnapshot saved")
        except Exception as exc:
            print(f"[CCGReviewUI] autosnapshot failed: {exc}")

    def _schedule_autosnapshot(self):
        ui = self._ui
        def _do():
            self._save_autosnapshot()
            ui.root.after(ui._AUTOSAVE_INTERVAL_MS, _do)
        ui.root.after(ui._AUTOSAVE_INTERVAL_MS, _do)

    def _autoload_session_latest(self, restore_groups: bool = False):
        """Load the latest selection file for the current session.

        By default only restores pair selections. Pass restore_groups=True on
        first launch to also load groups from groups_export.json (not the
        per-session file, which may have stale group entries).
        """
        ui = self._ui
        latest_path = self._sel_version_path('latest')
        if not os.path.isfile(latest_path):
            # Even without a session file, try to load groups
            if restore_groups:
                self._load_groups_from_export()
            return
        try:
            # Always load pair selections from the session-specific file;
            # never load groups from it (they may be stale).
            self._load_selection_from_file(latest_path,
                                           restore_groups=False,
                                           _skip_redraw=True)
        except Exception as exc:
            print(f"[CCGReviewUI] failed to autoload latest: {exc}")
        if restore_groups:
            self._load_groups_from_export()

    def _autosave_current(self):
        """Silently save current state to a fixed-name action-autosave file.

        Writes to .history/{session}-last_action.autosaved.json (overwrite, no
        new file per call).  Does NOT touch __latest.json — that is only written
        on explicit user save.
        """
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            try:
                self._autosave_all_sessions_for_current_type()
            except Exception as exc:
                print(f"[CCGReviewUI] any-session autosave failed: {exc}")
            try:
                if ui._sel_data.groups:
                    self._save_groups_export()
            except Exception:
                traceback.print_exc()
            try:
                ui._settings_mgr.save_ui_state()
            except Exception:
                traceback.print_exc()
            return
        if ui.ccg_ptr is None:
            return
        self._autosave_to_history_fixed()

    def _save_groups_export(self):
        """Write groups_export via SelectionData.save()."""
        ui = self._ui
        export_base = os.path.join(ui._sel_save_dir, 'groups_export')
        ui._sel_data.save(export_base)
        print(f"[CCGReviewUI] groups_export saved → {export_base}.json")

    def _load_groups_from_export(self):
        """Load SelectionData state from groups_export.json via Savable.load()."""
        ui = self._ui
        export_base = os.path.join(ui._sel_save_dir, 'groups_export')
        export_path = export_base + '.json'
        if not os.path.isfile(export_path):
            latest_path = self._sel_version_path('latest')
            if os.path.isfile(latest_path):
                try:
                    ui._sel_data.load(os.path.splitext(latest_path)[0])
                except Exception as exc:
                    print(f"[CCGReviewUI] failed to load groups from session file: {exc}")
            return
        try:
            ui._sel_data.load(export_base)
            n_groups = len(ui._sel_data.groups)
            n_pairs = sum(len(v) for v in ui._sel_data.groups.values())
            print(f"[CCGReviewUI] groups loaded: {n_groups} groups, "
                  f"{n_pairs} pair-session entries")
            try:
                ui._group_mgr._rebuild_groups_menu()
            except Exception:
                pass
            try:
                if hasattr(ui, 'hotkeys_bar'):
                    ui.hotkeys_bar.refresh()
            except Exception:
                pass
        except Exception as exc:
            print(f"[CCGReviewUI] failed to load groups_export.json: {exc}")

    def _autosave_to_history_fixed(self):
        """Write selections to .history/{session}-last_action.autosaved.json (overwrite)."""
        ui = self._ui
        try:
            data = self._build_save_dict(datetime.datetime.now().isoformat(), 'action_autosave')
            sess = getattr(ui.key, 'session', 'sess')
            hdir = self._history_dir()
            os.makedirs(hdir, exist_ok=True)
            path = os.path.join(hdir, f"{sess}-last_action.autosaved.json")
            ui._atomic_write_json(path, data)
        except Exception as exc:
            print(f"[CCGReviewUI] action autosave failed: {exc}")

    def _autosave_all_sessions_for_current_type(self):
        """Write fixed action-autosave file for each physical session (any-mode).

        Writes to .history/{session}-last_action.autosaved.json per session
        (overwrite).  Does NOT touch __latest.json.
        """
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            ui._sess_mgr._flush_any_selections_to_pointers()
        ui._sess_mgr._flush_any_deleted_to_stores()
        lbl = ui._type_label(ui.key)
        saved_sess: set[str] = set()
        old_key = ui.key
        old_ptr = ui.ccg_ptr
        old_cd = ui.ccg_data
        old_neurons = ui.neurons
        old_ns = ui.n_segments
        try:
            for nk in ui._sess_mgr._real_nd_keys_ordered():
                ckey = ui._sess_mgr._type_key_for_nd(nk)
                if ckey is None or ui._type_label(ckey) != lbl:
                    continue
                sess = str(ckey.session)
                if sess in saved_sess:
                    continue
                saved_sess.add(sess)
                ui._sess_mgr._bind_context_to_type_key(ckey)
                try:
                    self._autosave_to_history_fixed()
                except Exception as exc:
                    print(f"[CCGReviewUI] any-session autosave failed for {sess}: {exc}")
        finally:
            ui.key = old_key
            ui.ccg_ptr = old_ptr
            ui.ccg_data = old_cd
            ui.neurons = old_neurons
            ui.n_segments = old_ns
            # The save loop binds every session; restoring ``old_*`` can leave
            # ``ccg_*`` on session A while ``current_pair_idx`` still points at
            # a pair row for session B → IndexError in ``_resolve_segment_data``.
            if getattr(ui, '_session_any_mode', False):
                idx = ui.current_pair_idx
                hl = getattr(ui, '_any_pair_handle_list', None) or []
                if (getattr(ui.network_panel, '_focused_pair', None) is None
                        and 0 <= idx < len(hl)):
                    ui._sess_mgr._sync_any_plot_context(idx)

    def _sel_version_path(self, name: str) -> str:
        ui = self._ui
        safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        session_tag = getattr(ui.key, 'session', 'sess')
        return os.path.join(ui._sel_save_dir, f"{session_tag}__{safe}.json")

    def _save_selection_version(self, name: str) -> str:
        """Persist selections + pair annotations to a JSON file.

        Writes to the named version path and copies to .history/ for recovery.
        Uses atomic write to prevent partial saves from corrupting the file.
        """
        ui = self._ui
        saved_at = datetime.datetime.now().isoformat()
        data = self._build_save_dict(saved_at, name)
        path = self._sel_version_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ui._atomic_write_json(path, data)
        n_tags = len(data['pair_tags'])
        print(f"[CCGReviewUI] saved → {os.path.basename(path)}  "
              f"({n_tags} pair_tags, {len(data['selections'])} type keys)")
        try:
            self._save_to_history(data, '')
        except Exception as exc:
            print(f"[CCGReviewUI] history copy failed: {exc}")
        return path

    def _list_selection_versions(self) -> list:
        """Return list of (name, path, saved_at, is_valid, is_history) tuples.

        Main versions (in data/selections/) are listed first; then .history/
        entries (backups + autosaves) follow, newest first, marked is_history=True.
        """
        ui = self._ui
        session_tag = getattr(ui.key, 'session', 'sess')
        prefix = session_tag + '__'
        versions = []

        def _read_entry(fpath, fname, is_hist):
            is_autosaved = fname.endswith('.autosaved.json')
            try:
                with open(fpath, encoding='utf-8') as f:
                    meta = json.load(f)
                raw_name = meta.get('name', fname)
                if is_hist:
                    kind = '[autosaved]' if is_autosaved else '[backup]'
                    display = f"{kind} {meta.get('saved_at', '')[:16]}"
                else:
                    display = raw_name
                return (display, fpath, meta.get('saved_at', ''), True, is_hist)
            except Exception:
                return (fname, fpath, '⚠ corrupted', False, is_hist)

        if os.path.isdir(ui._sel_save_dir):
            for fname in sorted(os.listdir(ui._sel_save_dir)):
                if not fname.startswith(prefix) or not fname.endswith('.json'):
                    continue
                versions.append(_read_entry(
                    os.path.join(ui._sel_save_dir, fname), fname, False))

        hdir = self._history_dir()
        hist_entries = []
        if os.path.isdir(hdir):
            for fname in os.listdir(hdir):
                if not fname.startswith(prefix) or not fname.endswith('.json'):
                    continue
                fpath = os.path.join(hdir, fname)
                hist_entries.append(_read_entry(fpath, fname, True))
        # Sort history newest first
        hist_entries.sort(key=lambda e: e[2], reverse=True)
        versions.extend(hist_entries)
        return versions

    def _load_selection_from_file(self, path: str, restore_groups: bool = True,
                                   _skip_redraw: bool = False):
        """Load selection from a JSON file (v1.0, v3.x, or v4.0).

        If restore_groups is False, only pair selections are loaded — groups,
        hotkeys, and notes are left untouched.  Used by autoload on session
        switch (groups are shared across sessions and loaded from groups_export).
        """
        ui = self._ui
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON parse error in {os.path.basename(path)}.\n\n"
                f"The file is likely corrupted from a previous save attempt.\n"
                f"Technical detail: {exc}\n\n"
                f"Please delete this file and re-save your selection."
            ) from exc


        selections_by_type = data.get('selections', {})
        type_keys = ui._sess_mgr._available_type_keys(ui.key.nd())
        saved_keys = set(selections_by_type.keys())
        for tk_ in type_keys:
            ptr = ui.cd.ptr.get(tk_)
            if ptr is None:
                continue
            pairs = selections_by_type.get(str(tk_), [])
            if pairs:
                ptr.selected_inds = np.array(
                    [[int(r), int(c)] for r, c in pairs], dtype=int)
            else:
                ptr.selected_inds = None
        cur_sel = selections_by_type.get(str(ui.key), [])
        selected = set(tuple(int(v) for v in p) for p in cur_sel)

        current_available = set(map(tuple, ui.all_inds))
        _before = {p: ('sel' if p in ui._sel_data.selected_inds
                        else 'del' if p in ui._sel_data.deleted_inds
                        else 'unsel')
                   for p in current_available}
        missing = selected - current_available
        if missing and restore_groups:
            action = MissingPairsDialog.show(self._ui, missing)
            if action == 'cancel':
                return
            elif action == 'partial':
                selected = selected & current_available
            elif action == 'admit_all':
                for pair in missing:
                    _admit_pair(ui._sel_data, pair)
                current_available = set(map(tuple, ui.all_inds))
        elif missing:
            _cd = ui.cd.ccg.get(ui.key.nd()) if hasattr(ui.cd, 'ccg') else ui.ccg_data
            _n = (_cd.ccg.shape[1] if _cd is not None
                  and getattr(_cd, 'ccg', None) is not None else None)
            for pair in missing:
                if _n is not None and (pair[0] >= _n or pair[1] >= _n):
                    continue
                _admit_pair(ui._sel_data, pair)
            current_available = set(map(tuple, ui.all_inds))

        ui._pair_deleted_store = {}
        dbtype = data.get('deleted_by_type') or {}
        for k_str, plist in dbtype.items():
            ui._pair_deleted_store[k_str] = {
                tuple(int(v) for v in p) for p in plist}
        deleted = set(ui._pair_deleted_store.get(str(ui.key), set())) & current_available
        ui._sel_data.populate(current_available, selected=selected, deleted=deleted)
        _after = {p: ('sel' if p in ui._sel_data.selected_inds
                       else 'del' if p in ui._sel_data.deleted_inds
                       else 'unsel')
                  for p in current_available}
        _pair_changes = {p: (_before[p], _after[p])
                         for p in current_available if _before.get(p) != _after[p]}
        self._push_undo(SelectionCommand(_pair_changes, []))

        # Load pair_tags for this session — always reset to avoid stale cross-session tags
        ui._sel_data._pair_tags = {}
        raw_tags = data.get('pair_tags', {})
        cur_sess = ui._setup_mgr._current_session_str()
        for key_str, tdata in raw_tags.items():
            parts = key_str.split(',')
            if len(parts) != 2:
                continue
            pair = (int(parts[0]), int(parts[1]))
            entry = dict(tdata) if isinstance(tdata, dict) else {'notes': str(tdata)}
            ui._sel_data._pair_tags[pair] = entry
            if 'groups' in entry:
                for gitem in entry['groups']:
                    gname = gitem if isinstance(gitem, str) else None
                    if gname:
                        ui._sel_data.add_to_group(str(gname), cur_sess, pair)
        if not _skip_redraw:
            ui._post_load_refresh()

    def _do_save(self, name: str):
        """Core save logic: persist all types' selections + groups."""
        ui = self._ui
        if not self._save_all_state(name, silent=False):
            return

        # Count total selections across all types
        type_keys = ui._sess_mgr._available_type_keys(ui.key.nd())
        total = sum(
            len(ui.cd.ptr[tk_].selected_inds)
            for tk_ in type_keys
            if ui.cd.ptr.get(tk_) is not None
            and getattr(ui.cd.ptr[tk_], 'selected_inds', None) is not None
        )

        # Groups were exported via _save_all_state; keep message for UI feedback.
        groups_msg = f"\nGroups exported ({len(ui._sel_data.groups)} groups)." if ui._sel_data.groups else ""

        messagebox.showinfo(
            "Saved",
            f"Saved {total} pairs across {len(type_keys)} types as '{name}'.{groups_msg}",
            parent=ui.root)

    def _show_temp_warning(self, msg: str, duration_ms: int = 2000):
        """Show a temporary warning label at the top of the window that auto-disappears."""
        ui = self._ui
        lbl = tk.Label(ui.root, text=msg, bg='#FFF3CD', fg='#856404',
                       font=('Arial', 10, 'bold'), padx=8, pady=4)
        lbl.place(relx=0.5, y=4, anchor='n')
        ui.root.after(duration_ms, lbl.destroy)

    def _build_save_dict(self, saved_at: str, name: str = '') -> dict:
        """Build the serializable dict for a session save (v4.0 format).

        Flushes current type's selections to the pointer, then collects all
        type keys + pair_tags (including group membership) + deleted pairs.
        Does NOT write any files.
        """
        ui = self._ui
        if ui.ccg_ptr is None:
            raise RuntimeError("Cannot save: CCG data not yet loaded")
        # Flush current type's selections to pointer
        if getattr(ui, '_session_any_mode', False):
            ui._sess_mgr._flush_any_selections_to_pointers()
        else:
            ui.ccg_ptr.selected_inds = (
                np.array(sorted(ui.selected_inds), dtype=int)
                if ui.selected_inds else None
            )
        # Collect selections for every type key in this session
        type_keys = ui._sess_mgr._available_type_keys(ui.key.nd())
        selections_by_type = {}
        for tk_ in type_keys:
            ptr = ui.cd.ptr.get(tk_)
            if ptr is None:
                continue
            sel = getattr(ptr, 'selected_inds', None)
            selections_by_type[str(tk_)] = (
                [[int(r), int(c)] for r, c in sorted(map(tuple, sel))]
                if sel is not None and len(sel) > 0 else []
            )
        # Serialize pair_tags: include group membership (names) for this session.
        # NOTE: We intentionally store group NAMES (not numeric IDs) in session save
        # files. IDs remain internal to groups_export.json only.
        cur_sess = ui._setup_mgr._current_session_str()
        pair_tags_ser: dict = {}
        # Collect all pairs that have either tags/notes OR group membership
        all_annotated = set(ui._sel_data._pair_tags.keys())
        for gname in ui._sel_data.groups:
            all_annotated |= ui._sel_data.pairs_in_group(gname, cur_sess)
        for pair in sorted(all_annotated):
            r, t = int(pair[0]), int(pair[1])
            existing = dict(ui._sel_data._pair_tags.get(pair, {}))
            group_names = [g for g in ui._sel_data.groups_for_pair(cur_sess, r, t)
                           if not g.startswith('__')]
            if group_names or existing.get('notes') or existing.get('tags'):
                entry: dict = {}
                if group_names:
                    entry['groups'] = sorted(set(group_names))
                if existing.get('notes'):
                    entry['notes'] = existing['notes']
                if existing.get('tags'):
                    entry['tags'] = existing['tags']
                pair_tags_ser[f"{r},{t}"] = entry
        ui._flush_deleted_to_store()
        deleted_by_type = {}
        for tk_ in type_keys:
            ptr = ui.cd.ptr.get(tk_)
            valid = ui._all_inds_set_for_ptr(ptr)
            raw = set(ui._pair_deleted_store.get(str(tk_), set())) & valid
            deleted_by_type[str(tk_)] = [[int(r), int(c)] for r, c in sorted(raw)]
        return {
            'name': name,
            'saved_at': saved_at,
            'session': getattr(ui.key, 'session', 'sess'),
            'nd_key': str(ui.key.nd()),
            'neuron_ids': [int(x) for x in ui.neurons.neuron_ids],
            'selections': selections_by_type,
            'pair_tags': pair_tags_ser,
            'deleted_by_type': deleted_by_type,
        }

    @staticmethod
    def _pair_tag_has_labels(entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        groups = entry.get('groups', []) or []
        tags = entry.get('tags', []) or []
        notes = str(entry.get('notes', '') or '').strip()
        return bool(groups or tags or notes)

    def _save_all_state(self, selection_name: str | None = None, *, silent: bool = True) -> bool:
        """Single saving pathway for selections + groups + ui_state.

        - If selection_name is provided: writes that selection version.
        - Always attempts to write groups export (if any groups exist).
        - Always writes ui_state.json (panel + display button state, resolution, etc.).
        """
        ui = self._ui
        if selection_name is not None:
            try:
                self._save_selection_version(selection_name)
            except Exception as exc:
                traceback.print_exc()
                if not silent:
                    messagebox.showerror("Save error",
                                         f"Failed to save selection:\n{exc}",
                                         parent=ui.root)
                return False
        try:
            if ui._sel_data.groups:
                self._save_groups_export()
        except Exception:
            # never block save on groups export
            traceback.print_exc()
        try:
            ui._settings_mgr.save_ui_state()
        except Exception:
            traceback.print_exc()
        return True

    def run(self):
        ui = self._ui
        if ui._owns_mainloop:
            ui.root.mainloop()
        else:
            # Another Tk root owns the mainloop; just wait for this window
            ui.root.wait_window(ui.root)

    def _current_filter_state(self) -> dict:
        ui = self._ui
        toggles = getattr(ui.time_slider, '_ts_legend_toggles', {})
        return {
            'theme': ui.time_slider._current_theme,
            'labels': {str(lbl): bool(v.get()) for lbl, v in toggles.items()},
        }

class HotkeysBar:
    """Horizontal chip-bar showing Ctrl+key → group-name mappings."""

    _SLOT_ORDER = [str(i) for i in range(1, 10)] + ['0'] + list('abcdefghijklmnopqrstuvwxyz')

    def __init__(self, parent: tk.Widget, ui: 'CCGReviewUI'):
        self._ui = ui
        self._labels: list[tk.Label] = []
        self._scroller = ArrowScroller()
        self.frame = ttk.Frame(parent, relief=tk.GROOVE, borderwidth=1)
        self.refresh()

    def refresh(self):
        """Rebuild chip labels inside the bar."""
        ui = self._ui
        for w in self.frame.winfo_children():
            w.destroy()
        self._labels.clear()

        ttk.Label(self.frame, text="Groups:",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(6, 4))
        tk.Label(self.frame, text="Del/⌫: deleted",
                 font=('Courier', 9), padx=6, pady=1,
                 relief=tk.RIDGE, borderwidth=1, fg='#888888').pack(
            side=tk.LEFT, padx=2, pady=2)

        hk_to_group = {grp.hotkey: grp.name
                       for grp in ui._sel_data._group_metadata.values()
                       if grp.hotkey}
        all_chips = [(k, hk_to_group[k]) for k in self._SLOT_ORDER if k in hk_to_group]

        if not all_chips:
            ttk.Label(self.frame, text="(no hotkeys assigned)",
                      font=('Arial', 9), foreground='#888').pack(side=tk.LEFT, padx=4)
            return

        self._scroller.install(self.frame, len(all_chips), lambda _: self.refresh())
        for key_str, gname in all_chips[self._scroller.offset:]:
            lbl = tk.Label(self.frame, text=f"{key_str}: {gname}",
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1)
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            lbl.bind('<Button-1>',
                     lambda e, g=gname: ui._group_mgr._select_group(g))
            lbl.bind('<Double-Button-1>',
                     lambda e, g=gname: self._on_chip_double_click(g))
            self._labels.append(lbl)

    def _on_chip_double_click(self, group_name: str):
        """Navigate to a random pair from group_name; flash chip red if empty."""
        ui = self._ui
        pairs = ui._sel_data.pairs_in_group(group_name,
                                             ui._setup_mgr._current_session_str())
        if not pairs:
            for lbl in self._labels:
                if group_name in lbl.cget('text'):
                    orig = lbl.cget('fg')
                    lbl.config(fg='red')
                    ui.root.after(300, lambda l=lbl, c=orig: l.config(fg=c))
            return
        chosen = random.choice(sorted(pairs))
        ui.current_pair_idx = ui._plot_mgr.get_pair_index(chosen)
        ui._plot_mgr.update_plot()
        ui.network_panel.draw()


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
        self._entry.bind('<Return>',       lambda _: self.go(1))
        self._entry.bind('<Shift-Return>', lambda _: self.go(-1))
        self._entry.bind('<Escape>',       lambda _: self.hide())

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

    @property
    def active(self) -> bool:
        return bool(self._var.get())


def pair_label(inds, *, bookmarked: bool, group_names: list,
               pair_tags: dict, any_mode: bool = False) -> str:
    """Format the display string for one pair row."""
    if any_mode:
        sess = str(inds[0].session) if hasattr(inds[0], 'session') else str(inds[0])
        base = f"{sess} [{inds[1]}, {inds[2]}]"
    else:
        base = f"[{inds[0]}, {inds[1]}]"
    prefix = '\U0001F4CC ' if bookmarked else ''
    fmt_groups = []
    for n in group_names:
        if n.startswith('__'):
            continue
        fmt_groups.append('*' + n[len(_SPECIAL_PREFIX):] if is_special_group(n) else n)
    pt = pair_tags if isinstance(pair_tags, dict) else {}
    tag_mark = '~' if (pt.get('tags') or pt.get('notes', '').strip()) else ''
    group_str = f"[{','.join(fmt_groups)}]" if fmt_groups else ''
    suffix = (' ' + tag_mark + group_str) if (tag_mark or group_str) else ''
    return prefix + base + suffix


def sort_key_mean(inds, ccg_data, segment: int) -> float:
    if ccg_data is None:
        return 0.0
    try:
        seg = min(segment, ccg_data.ccg.shape[0] - 1)
        return float(np.mean(ccg_data.ccg[seg, int(inds[0]), int(inds[1]), :]))
    except (IndexError, KeyError, TypeError):
        return 0.0


def sort_key_min_pval(inds, ccg_data, segment: int) -> float:
    if ccg_data is None or ccg_data.pval is None:
        return 1.0
    try:
        seg = min(segment, ccg_data.pval.shape[0] - 1)
        arr = ccg_data.pval[seg, int(inds[0]), int(inds[1]), :]
        conf = ccg_data.conf
        sl = arr[int(conf.min_lag_bin):int(conf.max_lag_bin)]
        if sl.size == 0:
            return 1.0
        m = float(np.nanmin(sl))
        return m if np.isfinite(m) else 1.0
    except (IndexError, KeyError, TypeError, ValueError):
        return 1.0


def _bm_key(inds, any_mode: bool) -> tuple:
    """Normalize pair inds to the bookmark key used in _bookmarked_pairs."""
    if any_mode:
        return (int(inds[1]), int(inds[2]))
    return tuple(int(x) for x in inds[:2])


def _group_names_for_pair(data, ui, inds) -> list:
    """Return user-visible group names that contain inds for its session. O(1)."""
    sess, pair = ui._pair_sess_rt(inds)
    ref, tgt = int(pair[0]), int(pair[1])
    return [g for g in data.groups_for_pair(sess, ref, tgt)
            if not g.startswith('__')]


class LeftPanelContextMenu:
    """Context-menu actions mixed into LeftPanel. Assumes self._ui, self.data,
    self.selected_list, self.unselected_list, self.refresh_lists."""

    def _ctx_bulk_move(self, pairs, old_state: str, new_state: str, scroll_lb):
        """Apply a uniform state transition to *pairs*, push undo, refresh."""
        ui = self._ui
        if not pairs:
            return
        scroll_top = scroll_lb.yview()[0]
        changes = {p: (old_state, new_state) for p in pairs}
        ui._group_mgr._push_undo(SelectionCommand(changes, []))
        for p in changes:
            ui._sel_data.set_state(p, new_state)
        self.refresh_lists()
        scroll_lb.yview_moveto(scroll_top)
        ui._plot_mgr._highlight_changed_pairs(set(pairs))
        ui._flush_deleted_to_store()

    def _ctx_restore_from_deleted(self, pairs):
        self._ctx_bulk_move(pairs, 'del', 'unsel', self.unselected_list)

    def _ctx_delete_pairs(self, pairs):
        self._ctx_bulk_move(pairs, 'unsel', 'del', self.unselected_list)

    def _ctx_delete_from_selected(self, pairs):
        self._ctx_bulk_move(pairs, 'sel', 'del', self.selected_list)

    def _ctx_move_to_selected(self, pair):
        if pair is None: return
        self._ctx_move_multi_to_selected([pair])
        self._ui.current_pair_idx = self._ui._plot_mgr.get_pair_index(pair)
        self._ui._plot_mgr.update_plot()

    def _ctx_move_multi_to_selected(self, pairs):
        if getattr(self._ui, '_session_any_mode', False):
            return
        data = self._ui._sel_data
        changes = {}
        for p in pairs:
            if p in data.unselected_inds:   changes[p] = ('unsel', 'sel')
            elif p in data.deleted_inds:    changes[p] = ('del',   'sel')
        if not changes: return
        scroll_top = self.unselected_list.yview()[0]
        self._ui._group_mgr._push_undo(SelectionCommand(changes, []))
        for p, (_, new) in changes.items():
            data.set_state(p, new)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._ui._plot_mgr._highlight_changed_pairs(set(changes))
        self._ui._draw_network()

    def _ctx_move_to_unselected(self, pair):
        if pair is None: return
        self._ctx_move_multi_to_unselected([pair])
        self._ui.current_pair_idx = self._ui._plot_mgr.get_pair_index(pair)
        self._ui._plot_mgr.update_plot()

    def _ctx_move_multi_to_unselected(self, pairs):
        if getattr(self._ui, '_session_any_mode', False):
            return
        data = self._ui._sel_data
        changes = {p: ('sel', 'unsel') for p in pairs if p in data.selected_inds}
        if not changes: return
        scroll_top = self.selected_list.yview()[0]
        self._ui._group_mgr._push_undo(SelectionCommand(changes, []))
        for p in changes:
            data.set_state(p, 'unsel')
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self._ui._plot_mgr._highlight_changed_pairs(set(changes))
        self._ui._draw_network()

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

        menu.add_separator()
        grp_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Group tag", menu=grp_menu)
        grp_menu.add_command(label="Create new group…",
                             command=lambda: CreateGroupDialog.show(self._ui))
        if self.data.groups:
            grp_menu.add_separator()
        special_items = []
        for gname in sorted(self.data.groups):
            if is_special_group(gname):
                special_items.append(gname)
                continue
            if gname.startswith('__'):
                continue
            if pairs:
                all_in = all(p2 in self._ui._sel_data.pairs_in_group(gname, s2)
                             for p in pairs for s2, p2 in (self._ui._pair_sess_rt(p),))
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
                    all_in = all(p2 in self._ui._sel_data.pairs_in_group(gname, s2)
                                 for p in pairs for s2, p2 in (self._ui._pair_sess_rt(p),))
                    label = f"{'✓ ' if all_in else ''}  {display}"
                    sp_menu.add_command(
                        label=label,
                        command=lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))

        if pairs:
            menu.add_separator()
            tog_tuples = [tuple(x) for x in ui._together_pairs]
            all_pinned = all(tuple(p) in tog_tuples for p in pairs)
            tog_label = ("Remove from 'Show Together'" if all_pinned else "Show Together")
            menu.add_command(label=tog_label,
                             command=lambda pp=pairs: ui._toggle_together(pp))
            if ui._together_pairs:
                menu.add_command(
                    label=f"Clear 'Show Together' ({len(ui._together_pairs)} pairs)",
                    command=ui._clear_together)

        if n == 1:
            menu.add_separator()
            p = pairs[0]
            _sess, _rt = ui._pair_sess_rt(p)
            has_tags = _rt in self.data._pair_tags
            menu.add_command(
                label=f"{'✓ ' if has_tags else ''}Pair tags…",
                command=lambda: PairTagsDialog.show(self._ui))

        menu.add_separator()
        menu.add_command(label="Export view as PNG…",
                         command=lambda: ui._export_mgr._export_current_view('png'))
        menu.add_command(label="Export view as PDF…",
                         command=lambda: ui._export_mgr._export_current_view('pdf'))

        sort_group = getattr(self, '_sort_selected', None)
        sort_tag   = getattr(self, '_sort_by_tag', None)
        if (sort_group and sort_group.get()) or (sort_tag and sort_tag.get()):
            all_names = [g for g in self.data.groups
                         if not g.startswith('__')]
            menu.add_separator()
            menu.add_command(label="Collapse all groups",
                             command=lambda: (self._collapse_state.collapse_all(all_names),
                                             self.refresh_lists()))
            menu.add_command(label="Expand all groups",
                             command=lambda: (self._collapse_state.expand_all(),
                                             self.refresh_lists()))

        menu.tk_popup(event.x_root, event.y_root)


class LeftPanel(LeftPanelContextMenu):
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

    @staticmethod
    def _combo_sort_key(combo):
        """Sort key: empty combo (untagged) last, then alphabetical."""
        return (1, []) if not combo else (0, list(combo))

    # Search highlight colors
    _BOOKMARK_LIST_BG    = '#ffcdd2'
    _BOOKMARK_LIST_FG    = '#b71c1c'
    _BOOKMARK_LIST_SELBG = '#ef9a9a'
    _BOOKMARK_LIST_SELFG = '#4a0000'

    def __init__(self, parent: tk.Widget, data: SelectionData,
                 ui, ui_state_cache: dict):
        self.data = data
        self._ui  = ui              # CCGReviewUI back-reference
        self._collapse_state = CollapseState()

        self.notebook = ttk.Notebook(parent)

        self._build_pair_selection_tab(ui_state_cache)
        self._build_spike_pairs_tab()

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
        self._sort_mean_cb = ttk.Checkbutton(btn_frame, text="Sort by mean",
                                              variable=self._sort_by_mean,
                                              command=self._on_sort_by_mean_toggle)
        self._sort_mean_cb.pack(side=tk.LEFT, padx=2)
        self._sort_minp_cb = ttk.Checkbutton(btn_frame, text="Sort by min p-val",
                                              variable=self._sort_by_min_p,
                                              command=self._on_sort_by_min_p_toggle)
        self._sort_minp_cb.pack(side=tk.LEFT, padx=2)

        # ── Spike Pairs tab (populated by SpikePairsPanel) ───────────────
        self._spike_pairs_tab_index: int = 1

    def _build_spike_pairs_tab(self):
        # ── Spike Pairs tab ──────────────────────────────────────────────
        spike_tab = ttk.Frame(self.notebook)
        self.notebook.add(spike_tab, text="Spike Pairs", state='disabled')
        self._spike_pairs_tab = spike_tab
        # and stored as self._spike_pairs_listbox / self._spike_pairs_count

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

    def _pair_mean_ccg(self, inds):
        return sort_key_mean(inds, self._ui.ccg_data, self._ui.current_segment)

    def _pair_min_pval(self, inds):
        return sort_key_min_pval(inds, self._ui.ccg_data, self._ui.current_segment)

    def _save_scroll_positions(self):
        try: unsel = self.unselected_list.yview()[0]
        except Exception: unsel = None
        try: sel = self.selected_list.yview()[0]
        except Exception: sel = None
        return unsel, sel

    def _restore_scroll_positions(self, unsel_top, sel_top):
        try:
            if unsel_top is not None:
                self.unselected_list.yview_moveto(unsel_top)
        except Exception:
            pass
        try:
            if sel_top is not None:
                self.selected_list.yview_moveto(sel_top)
        except Exception:
            pass

    def _build_should_gray(self, ui):
        """Return a predicate `should_gray(inds) -> bool` for the current render state."""
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
        peak_channels = (getattr(_neurons_for_sids, 'peak_channels', None)
                         if _neurons_for_sids is not None else None)

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
                sids_use = _sids
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

        return _should_gray

    def _populate_avail_list(self, ui, data, should_gray):
        """Populate the Available (unselected) listbox."""
        self._avail_list_pairs = []
        if getattr(ui, '_session_any_mode', False):
            return  # any-mode: available list intentionally empty
        bm = getattr(ui, '_bookmarked_pairs', None) or set()
        for inds in sorted(data.unselected_inds):
            _, _pair = ui._pair_sess_rt(inds)
            pair_t = tuple(int(x) for x in _pair)
            gnames = _group_names_for_pair(data, ui, inds)
            label = pair_label(inds,
                               bookmarked=tuple(int(x) for x in inds[:2]) in bm,
                               group_names=gnames,
                               pair_tags=data._pair_tags.get(pair_t, {}))
            self.unselected_list.insert(tk.END, label)
            self._avail_list_pairs.append((inds, None))
            if should_gray(inds):
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
            self._avail_list_pairs.append((_SEPARATOR_ROW, 'deleted'))
            for inds in sorted(deleted_inds):
                label = f"[{inds[0]}, {inds[1]}]"
                self.unselected_list.insert(tk.END, label)
                self.unselected_list.itemconfig(
                    self.unselected_list.size() - 1, foreground='#AAAAAA')
                self._avail_list_pairs.append((inds, 'deleted'))

    def _sel_insert_pair(self, inds, should_gray, any_mode):
        """Insert one pair row into the selected listbox."""
        ui = self._ui
        bm = getattr(ui, '_bookmarked_pairs', None) or set()
        bm_key = _bm_key(inds, any_mode)
        _, _pair = ui._pair_sess_rt(inds)
        pair_t = tuple(int(x) for x in _pair)
        gnames = _group_names_for_pair(self.data, ui, inds)
        label = pair_label(inds,
                           bookmarked=bm_key in bm,
                           group_names=gnames,
                           pair_tags=self.data._pair_tags.get(pair_t, {}),
                           any_mode=any_mode)
        self.selected_list.insert(tk.END, label)
        self._sel_list_pairs.append(inds)
        self._sel_list_header_keys.append(None)
        if should_gray(inds):
            self.selected_list.itemconfig(
                self.selected_list.size() - 1, foreground='#AAAAAA')

    def _sel_insert_header(self, text, count, collapsed_set):
        """Insert a collapsible group header into the selected listbox. Returns is_collapsed."""
        is_collapsed = text in collapsed_set
        display = group_header_label(text, count, is_collapsed)
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

    def _populate_selected_list(self, ui, data, should_gray):
        """Populate the Selected listbox for all sort modes. Returns total_any_count."""
        self._sel_list_pairs       = []
        self._sel_list_header_keys = []

        any_mode   = getattr(ui, '_session_any_mode', False)
        sort_group = self._sort_selected.get()
        sort_tag   = self._sort_by_tag.get()
        sort_mean  = self._sort_by_mean.get() and not any_mode
        sort_minp  = self._sort_by_min_p.get() and not any_mode

        collapsed_set = self._collapse_state.as_set()

        def _ins(inds):
            self._sel_insert_pair(inds, should_gray, any_mode)

        def _ins_hdr(text, count):
            return self._sel_insert_header(text, count, collapsed_set)

        def _pair_group_combo(inds):
            sess, pair = ui._pair_sess_rt(inds)
            pair = tuple(int(x) for x in pair)
            return tuple(sorted(
                g for g in data.groups
                if not g.startswith('__')
                and pair in self._ui._sel_data.pairs_in_group(g, sess)
            ))

        total_any_count = 0
        if any_mode:
            dead      = getattr(ui, 'deleted_inds', set())
            _expanded = getattr(ui, '_any_expanded_group_tags', set())
            def _gname_sort_key(n):
                try: return (0, int(n), '')
                except (ValueError, TypeError): return (1, 0, n)
            all_gnames = sorted((g for g in ui._sel_data.groups if not g.startswith('__')),
                                key=_gname_sort_key)

            _all_trips: set = set()
            for _gn in all_gnames:
                _all_trips |= (ui._group_mgr._valid_group_pairs_as_triples(_gn) - dead)
            total_any_count = len(_all_trips)

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
                pair_tags: dict = {}
                for gname in all_gnames:
                    for trip in ui._group_mgr._valid_group_pairs_as_triples(gname):
                        if trip not in dead:
                            pair_tags.setdefault(trip, set()).add(gname)
                combo_buckets: dict = _defaultdict(list)
                for trip, tags in sorted(pair_tags.items()):
                    combo_buckets[tuple(sorted(tags))].append(trip)
                for combo in sorted(combo_buckets.keys(), key=self._combo_sort_key):
                    hdr_text = ', '.join(combo) if combo else '(untagged)'
                    trips_combo = combo_buckets[combo]
                    exp_hdr = _any_insert_hdr(hdr_text, len(trips_combo))
                    if exp_hdr:
                        for sess, r, t in trips_combo:
                            nd_key = ui._sess_mgr._nd_key_for_session_str(sess)
                            if nd_key is None:
                                continue
                            ckey = ui._sess_mgr._type_key_for_nd(nd_key)
                            if ckey is None:
                                continue
                            _ins((ckey, r, t))
            else:
                for gname in all_gnames:
                    trips_g = ui._group_mgr._valid_group_pairs_as_triples(gname)
                    n_tag = len(trips_g - dead)
                    exp_hdr = _any_insert_hdr(gname, n_tag)
                    if exp_hdr:
                        for row in ui._group_mgr._iter_group_pairs(gname):
                            _ins(row)

        elif sort_mean:
            for inds in sorted(data.selected_inds,
                               key=self._pair_mean_ccg, reverse=True):
                _ins(inds)

        elif sort_minp:
            for inds in sorted(data.selected_inds, key=self._pair_min_pval):
                _ins(inds)

        elif sort_group:
            buckets = _defaultdict(list)
            for inds in sorted(data.selected_inds):
                buckets[_pair_group_combo(inds)].append(inds)
            for combo in sorted(buckets.keys(), key=self._combo_sort_key):
                pairs_in_combo = buckets[combo]
                hdr_text = ', '.join(combo) if combo else '(untagged)'
                if not _ins_hdr(hdr_text, len(pairs_in_combo)):
                    for inds in pairs_in_combo:
                        _ins(inds)

        elif sort_tag:
            tag_buckets: dict = _defaultdict(list)
            untagged = []
            non_internal = [g for g in data.groups
                            if not g.startswith('__')
                            and not is_special_group(g)]
            for inds in sorted(data.selected_inds):
                _s, _p = ui._pair_sess_rt(inds)
                _p = tuple(int(x) for x in _p)
                tags = [g for g in non_internal
                        if _p in self._ui._sel_data.pairs_in_group(g, _s)]
                if tags:
                    for t in tags:
                        tag_buckets[t].append(inds)
                else:
                    untagged.append(inds)
            for tag in sorted(tag_buckets.keys()):
                if not _ins_hdr(tag, len(tag_buckets[tag])):
                    for inds in tag_buckets[tag]:
                        _ins(inds)
            if untagged:
                if not _ins_hdr('(untagged)', len(untagged)):
                    for inds in untagged:
                        _ins(inds)
        else:
            for inds in sorted(data.selected_inds):
                _ins(inds)

        return total_any_count

    def _update_list_labels(self, ui, data, total_any_count):
        """Update avail/selected count labels, jitter colors, stats."""
        if getattr(ui, '_session_any_mode', False):
            self._avail_label.set("Available (0) — Any mode")
        else:
            parts = [f"Available ({len(data.unselected_inds)}"]
            deleted_inds = getattr(ui, 'deleted_inds', set())
            if deleted_inds:
                parts.append(f", {len(deleted_inds)} deleted")
            parts.append(")")
            self._avail_label.set(''.join(parts))

        try:
            if getattr(ui, '_session_any_mode', False):
                self._sel_label.set(f"Selected ({total_any_count})")
            else:
                sess = ui._setup_mgr._current_session_str()
                ct_lbl = ui._conn_type_label(getattr(ui.key, 'conn_type', None))
                n_ct = len(ui._filter_pairs_to_conn_types(
                    sess, data.selected_inds, {ct_lbl}))
                self._sel_label.set(f"Selected ({n_ct})")
        except Exception:
            self._sel_label.set(f"Selected ({len(data.selected_inds)})")

        ui.jitter_controller.apply_list_colors()
        ui._refresh_stats()

    def refresh_lists(self):
        ui, data = self._ui, self.data
        unsel_top, sel_top = self._save_scroll_positions()
        self.unselected_list.delete(0, tk.END)
        self.selected_list.delete(0, tk.END)

        if getattr(ui, '_session_any_mode', False):
            ui._sess_mgr._any_rebuild_pair_handles()
            ui._sess_mgr._any_sync_selection_from_universe()

        should_gray = self._build_should_gray(ui)
        _cb_state = 'disabled' if getattr(ui, '_session_any_mode', False) else 'normal'
        for _cb in (getattr(self, '_sort_mean_cb', None),
                    getattr(self, '_sort_minp_cb', None)):
            if _cb is not None:
                _cb.config(state=_cb_state)
        self._populate_avail_list(ui, data, should_gray)
        total_any = self._populate_selected_list(ui, data, should_gray)
        self._update_list_labels(ui, data, total_any)

        if self.search_bar.active:
            self.search_bar._refresh()
        else:
            self._reapply_bookmark_list_styles()

        self._restore_scroll_positions(unsel_top, sel_top)

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
                    self._ui._sel_data.add_to_group(pred_group, inds[0], (inds[1], inds[2]))
                else:
                    self._ui._sel_data.add_to_group(pred_group, ui._setup_mgr._current_session_str(), inds)
                self.refresh_lists()
            return
        ui._group_mgr._push_undo(SelectionCommand({inds: ('unsel', 'sel')}, []))
        self.data.set_state(inds, 'sel')
        if pred_group is not None:
            if getattr(ui, '_session_any_mode', False):
                self._ui._sel_data.add_to_group(pred_group, inds[0], (inds[1], inds[2]))
            else:
                self._ui._sel_data.add_to_group(pred_group, ui._setup_mgr._current_session_str(), inds)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        ui._plot_mgr._highlight_changed_pairs({inds})
        ui.current_pair_idx = ui._plot_mgr.get_pair_index(inds)
        ui._plot_mgr.update_plot()
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
                            ui._group_mgr._toggle_any_avail_group(hkey)
                        return
                    self._collapse_state.toggle(hkey)
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
        ui._group_mgr._push_undo(SelectionCommand({inds: ('sel', 'unsel')}, []))
        self.data.set_state(inds, 'unsel')
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        ui._plot_mgr._highlight_changed_pairs({inds})
        ui.current_pair_idx = ui._plot_mgr.get_pair_index(inds)
        ui._plot_mgr.update_plot()
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
        old = 'unsel' if inds in self.data.unselected_inds else 'sel'
        new = 'sel' if old == 'unsel' else 'unsel'
        ui._group_mgr._push_undo(SelectionCommand({inds: (old, new)}, []))
        self.data.set_state(inds, new)
        next_idx = min(ui.current_pair_idx + 1, len(ui.all_inds) - 1)
        ui.current_pair_idx = next_idx
        self.refresh_lists()
        self._select_pair_in_list(self._pair_at_all_inds_idx(next_idx))
        ui._plot_mgr._highlight_changed_pairs({inds})
        ui._plot_mgr.update_plot()
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
                    if is_separator_row(e):
                        return
                    inds_from_map = e[0]
        elif widget is self.selected_list:
            mp = getattr(self, '_sel_list_pairs', None)
            if mp and idx < len(mp):
                if mp[idx] is None:
                    return
                inds_from_map = mp[idx]

        if inds_from_map is not None:
            try:
                self._ui.current_pair_idx = self._ui._plot_mgr.get_pair_index(inds_from_map)
            except (ValueError, TypeError) as e:
                print(f"[ops] exception: {e}")
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
                self._ui.current_pair_idx = self._ui._plot_mgr.get_pair_index(inds)
            except (ValueError, TypeError):
                return

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
                if is_separator_row(ent):
                    return 'break'
                inds = ent[0]
        else:
            mp = getattr(self, '_sel_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                inds = mp[idx]
        if inds is not None:
            try:
                self._ui.current_pair_idx = self._ui._plot_mgr.get_pair_index(inds)
            except Exception:
                pass
        try:
            self._ui._exit_spike_attribution_view()
        except Exception:
            pass
        try:
            self._ui.network_panel._focused_pair = None
        except Exception:
            pass
        try:
            self._ui._plot_mgr.update_plot()
            self._ui.network_panel.draw()
        except Exception:
            pass
        return 'break'

    def _do_pair_select_update(self):
        ui = self._ui
        ui._select_after = None
        try:
            ui._exit_spike_attribution_view()
        except Exception:
            pass
        try:
            ui.jitter_controller.mark_viewed()
        except Exception:
            pass
        if hasattr(ui, 'network_panel'):
            ui.network_panel._focused_pair = None
        if hasattr(ui, 'network_panel'):
            try:
                ui.network_panel._focus_pair_var.set("")
                ui.network_panel._focus_pair_info_var.set("")
            except Exception:
                pass
        ui._plot_mgr.update_plot()

    def _on_arrow_key(self, event):
        self.on_pair_select(event)

    def _search_toggle(self):
        """Ctrl+F entry point — show or hide the search bar."""
        self.search_bar.toggle()

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
                if is_separator_row(raw):
                    continue
                ti = _bm_key(raw, getattr(self._ui, '_session_any_mode', False))
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
                ti = _bm_key(entry, getattr(self._ui, '_session_any_mode', False))
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

    def _bookmark_label_prefix(self, inds) -> str:
        bm = getattr(self._ui, '_bookmarked_pairs', None) or set()
        t = _bm_key(inds, getattr(self._ui, '_session_any_mode', False))
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
                inds = _bm_key(row, False)
        else:
            # Normalize: in All mode _selected_pair_from_lists returns (sess, ref, tgt)
            if getattr(ui, '_session_any_mode', False) and len(inds) == 3:
                inds = _bm_key(inds, True)
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

    def _resolve_list_entry(self, entry):
        """Resolve one listbox map entry to a pair tuple, or None if invalid."""
        if entry is None or is_separator_row(entry):
            return None
        inds = (entry[0] if isinstance(entry, tuple) and len(entry) == 2
                and isinstance(entry[0], (tuple, list, np.ndarray)) else entry)
        try:
            if getattr(self._ui, '_session_any_mode', False):
                sess = str(inds[0].session) if hasattr(inds[0], 'session') else str(inds[0])
                return (sess, int(inds[1]), int(inds[2]))
            return (int(inds[0]), int(inds[1]))
        except Exception:
            return None

    def _selected_pair_from_lists(self):
        """Return (ref,tgt) or (sess,ref,tgt) in any-mode; else None."""
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
            if not sel:
                continue
            try:
                pair = self._resolve_list_entry(mp[sel[-1]])
            except Exception:
                continue
            if pair is not None:
                return pair
        return None

    def _selected_pairs_from_lists(self) -> list:
        """Return all selected (ref,tgt) pairs across both lists (deduped)."""
        out, seen = [], set()
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
                    pair = self._resolve_list_entry(mp[i])
                except Exception:
                    continue
                if pair is None or pair in seen:
                    continue
                seen.add(pair)
                out.append(pair)
        out.sort(key=lambda x: (str(x[0]), x[1], x[2]) if len(x) == 3 else ('', x[0], x[1]))
        return out

    def _select_all(self):
        """Toggle between Select All and Deselect All."""
        if getattr(self._ui, '_session_any_mode', False):
            return
        if self.data.unselected_inds:
            for inds in list(self.data.unselected_inds):
                self.data.set_state(inds, 'sel')
        else:
            for inds in list(self.data.selected_inds):
                self.data.set_state(inds, 'unsel')
        self.refresh_lists()
        self._ui.network_panel.draw()

    def _pair_at_all_inds_idx(self, idx: int):
        """Canonical pair key matching list rows / selection sets."""
        if getattr(self._ui, '_session_any_mode', False):
            hl = getattr(self._ui, '_any_pair_handle_list', None) or []
            if idx < 0 or idx >= len(hl):
                return None
            ck, r, t = hl[idx]
            return (str(ck.session), int(r), int(t))
        row = self._ui.all_inds[idx]
        return tuple(int(x) for x in row)

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
            ui._group_mgr._push_undo(SelectionCommand({trip: ('sel', 'del')}, []))
            ui._sel_data.set_state(trip, 'del')
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
                ui.current_pair_idx = ui._plot_mgr.get_pair_index(next_trip)
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
            ui._plot_mgr.update_plot()
            return

        inds = tuple(int(x) for x in ui.all_inds[ui.current_pair_idx])

        if inds in ui.selected_inds:
            scroll_top = self.selected_list.yview()[0]
            ui._group_mgr._push_undo(SelectionCommand({inds: ('sel', 'del')}, []))
            ui._sel_data.set_state(inds, 'del')
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
            ui._plot_mgr.update_plot()
            return

        scroll_top = self.unselected_list.yview()[0]
        if inds in ui.deleted_inds:
            ui._group_mgr._push_undo(SelectionCommand({inds: ('del', 'unsel')}, []))
            ui._sel_data.set_state(inds, 'unsel')
            self.refresh_lists()
            self.unselected_list.yview_moveto(scroll_top)
            self._select_pair_in_list(inds)
        else:
            ui._group_mgr._push_undo(SelectionCommand({inds: ('unsel', 'del')}, []))
            ui._sel_data.set_state(inds, 'del')
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
        ui._plot_mgr.update_plot()

    def _toggle_pairs_group(self, pairs, group_name):
        all_in = True
        for p in pairs:
            s2, p2 = self._ui._pair_sess_rt(p)
            if p2 not in self._ui._sel_data.pairs_in_group(group_name, s2):
                all_in = False
                break
        if all_in:
            for p in pairs:
                s2, p2 = self._ui._pair_sess_rt(p)
                self._ui._sel_data.discard_from_group(group_name, s2, p2)
        else:
            for p in pairs:
                s2, p2 = self._ui._pair_sess_rt(p)
                self._ui._sel_data.add_to_group(group_name, s2, p2)
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll   = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    def _set_group_hotkey(self, group_name, key_str):
        self._ui._group_mgr._set_group_hotkey(group_name, key_str)

    def _rebuild_groups_menu(self):
        ui = self._ui
        if not hasattr(ui, '_groups_menu'):
            return
        try:
            while ui._groups_menu.index('end') >= 7:
                ui._groups_menu.delete(7)
        except tk.TclError:
            pass
        if hasattr(ui, 'network_panel'):
            ui.network_panel.refresh_group_buttons()
        if (hasattr(ui, 'hotkeys_bar')
                and ui._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get()):
            ui.hotkeys_bar.refresh()

    def _select_group(self, group_name):
        sess = self._ui._setup_mgr._current_session_str()
        pairs = self._ui._sel_data.pairs_in_group(group_name, sess)
        if not pairs:
            return
        first = sorted(pairs)[0]
        self._ui.current_pair_idx = self._ui._plot_mgr.get_pair_index(first)
        self._ui._plot_mgr.update_plot()
        self._ui._draw_network()

    def _group_hotkey_handler(self, key_str: str, advance: bool = True):
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

        for _grp in self.data._group_metadata.values():
            gname, k = _grp.name, _grp.hotkey
            if not k or k != key_str:
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
                        if entry is None or is_separator_row(entry):
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

            pair_changes = {}
            group_changes = []
            for pair in highlighted:
                sess, p2 = self._ui._pair_sess_rt(pair)
                was_in = p2 in self._ui._sel_data.pairs_in_group(gname, sess)
                group_changes.append((gname, sess, p2, 'remove' if was_in else 'add'))
                if getattr(ui, '_session_any_mode', False):
                    continue
                if not was_in and pair in self.data.unselected_inds:
                    pair_changes[pair] = ('unsel', 'sel')
                elif was_in and pair in self.data.selected_inds:
                    remaining = [g for g in self.data.groups
                                 if g != gname and not g.startswith('__')
                                 and p2 in self._ui._sel_data.pairs_in_group(g, sess)]
                    if not remaining:
                        pair_changes[pair] = ('sel', 'unsel')
            ui._group_mgr._push_undo(SelectionCommand(pair_changes, group_changes))

            changed = set()
            for pair in highlighted:
                sess, p2 = self._ui._pair_sess_rt(pair)
                was_in_group = p2 in self._ui._sel_data.pairs_in_group(gname, sess)
                if was_in_group:
                    self._ui._sel_data.discard_from_group(gname, sess, p2)
                else:
                    self._ui._sel_data.add_to_group(gname, sess, p2)
                if getattr(ui, '_session_any_mode', False):
                    changed.add(pair)
                    continue
                if pair in pair_changes:
                    self.data.set_state(pair, pair_changes[pair][1])
                    changed.add(pair)

            self.refresh_lists()
            ui._plot_mgr._highlight_changed_pairs(changed or {current_pair})
            if advance:
                next_idx = min(ui.current_pair_idx + 1, len(ui.all_inds) - 1)
                ui.current_pair_idx = next_idx
                self._select_pair_in_list(self._pair_at_all_inds_idx(next_idx))
            else:
                self._select_pair_in_list(current_pair)
            ui._plot_mgr.update_plot()
            ui.network_panel.draw()
            return
        ui._group_mgr._show_temp_warning(f"No group assigned to Ctrl+{key_str}")


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

