"""Qt widget replacement for pair_selection_panel.py.

Pure data/logic classes (SelectionData, GroupManager, CollapseState, …)
remain in pair_selection_panel.py unchanged and are imported here.

Tk var shims (CheckboxVar, LabelVar, LineEditVar from utils) expose .get()/.set()
so CCGReviewUI aliases (ui._sort_selected_var, etc.) keep working without edits.
"""

from __future__ import annotations

import re
import random
from collections import defaultdict as _defaultdict
from typing import TYPE_CHECKING, Callable

import numpy as np

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, QTimer, QObject, Signal
from pyqtgraph.Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QCheckBox,
    QTabWidget, QFrame, QLineEdit, QPushButton,
    QMenu, QApplication, QMessageBox,
    QSizePolicy,
)
from pyqtgraph.Qt.QtGui import QColor, QFont, QBrush, QAction

from neuropy.ui.ui_common import (
    _SPECIAL_PREFIX, _SEPARATOR_ROW, is_special_group, is_separator_row,
    group_header_label, SelectionCommand,
    pair_label, group_names_for_pair, bm_key,
)
from neuropy.ui.dialogs import (
    PairTagsDialog as PairTagsDialog,
    CreateGroupDialog as CreateGroupDialog,
    MissingPairsDialog as MissingPairsDialog,
)
from neuropy.analyses.utils import JsonSavable

if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI as CCGReviewUI

_ROLE_PAIR = Qt.UserRole        # tuple | None
_ROLE_TAG  = Qt.UserRole + 1   # 'deleted' | 'header' | 'sep' | None
_ROLE_HKEY = Qt.UserRole + 2   # str | None  (group key for collapsible headers)

# Shift remaps the digit row to symbols (Shift+1 → "!"), so event.key() is the symbol,
# not the digit. Map the shifted symbols back to their digit char (US layout).
_SHIFT_DIGITS = {
    Qt.Key_Exclam: '1', Qt.Key_At: '2', Qt.Key_NumberSign: '3', Qt.Key_Dollar: '4',
    Qt.Key_Percent: '5', Qt.Key_AsciiCircum: '6', Qt.Key_Ampersand: '7',
    Qt.Key_Asterisk: '8', Qt.Key_ParenLeft: '9', Qt.Key_ParenRight: '0',
}

_C_GRAY_FG   = QColor('#AAAAAA')
_C_HDR_FG    = QColor('#444444')
_C_HDR_BG    = QColor('#CCCCCC')
_C_SEP_FG    = QColor('#AAAAAA')
_C_SEP_BG    = QColor('#EEEEEE')
_C_SEARCH_BG = QColor('#fff099')
_C_BM_FG     = QColor('#b71c1c')
_C_BM_BG     = QColor('#ffcdd2')
_C_BM_SEL_FG = QColor('#4a0000')
_C_BM_SEL_BG = QColor('#ef9a9a')
_C_ACTED_BG  = QColor('#b3e5fc')


import datetime
import json
import os
import re
import traceback
from collections import defaultdict as _defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuropy.ui.ui_common import (
    _SPECIAL_PREFIX, _SEPARATOR_ROW,
    is_special_group, is_separator_row,
    CollapseState, group_header_label, SelectionCommand, BiIndex,
)
from neuropy.analyses.utils import Savable, Autosave, UndoRedo
from neuropy.analyses.neurons_dataset import Key
from neuropy.ui.utils import (
    CheckboxVar, ExclusiveButtonSet, LabelVar, LineEditVar, PairListWidget,
)

from pyqtgraph.Qt.QtWidgets import QMessageBox
from pyqtgraph.Qt.QtCore import QTimer as _QTimer
if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI as CCGReviewUI

@dataclass
class Group(JsonSavable):
    def __init__(self, name: str = '', hotkey: str = '', notes: str = ''):
        JsonSavable.__init__(self)
        self.name    = name
        self.hotkey  = hotkey
        self.notes   = notes


class Groups(QObject, JsonSavable, BiIndex):
    changed = Signal()

    _custom_types = {'registry': Group}

    def __init__(self):
        QObject.__init__(self)
        JsonSavable.__init__(self, ignored_attrs=['ui'])
        BiIndex.__init__(self)
        self.registry: dict[str, Group] = {}
        self.ui = None
        self._save_dir: str = ''

    def bind(self, ui: 'CCGReviewUI') -> None:
        self.ui = ui

    def __setstate__(self, state: dict) -> None:
        # groups.json keys each Group by name but omits "name" from the value dict,
        # so restore it from the registry key — the single source of truth.
        JsonSavable.__setstate__(self, state)
        for name, grp in self.registry.items():
            grp.name = name

    def __bool__(self) -> bool:
        return bool(self._fwd) or bool(self.registry)

    @property
    def defined_groups(self) -> list[str]:
        # add_to_group always registers metadata, so _fwd ⊆ registry
        return sorted(self.registry.keys())

    @property
    def groups(self) -> list[str]:
        return sorted(g for g in self.defined_groups if not is_special_group(g))

    def special_groups(self) -> list[str]:
        return sorted(g for g in self.defined_groups if is_special_group(g))

    def get_group_metadata(self, name: str) -> Group:
        if name not in self.registry:
            self.registry[name] = Group(name=name)
        return self.registry[name]

    def save_path(self, **_) -> str | None:
        d = self._save_dir or (self.ui.sd.save_dir if self.ui is not None else '')
        return os.path.join(d, 'groups') if d else None

    def add_to_group(self, gname: str, sess: str, pair: tuple) -> None:
        self.add(gname, (sess, int(pair[0]), int(pair[1])))
        self.get_group_metadata(gname)

    def discard_from_group(self, gname: str, sess: str, pair: tuple) -> None:
        self.discard(gname, (sess, int(pair[0]), int(pair[1])))

    def pairs_in_group(self, gname: str, sess: str) -> set:
        return {(r, t) for s, r, t in self.forward(gname) if s == sess}

    def groups_for_pair(self, sess: str, ref: int, tgt: int) -> set:
        return self.inverse((sess, int(ref), int(tgt)))

    def sessions_for_group(self, gname: str) -> set:
        return {s for s, *_ in self.forward(gname)}

    def create_group(self, full_name: str) -> Group:
        if full_name in self.registry or full_name in self._fwd:
            raise ValueError(f"group '{full_name}' already exists")
        return self.get_group_metadata(full_name)

    def rename_group(self, old_name: str, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if new_name in self._fwd:
            raise ValueError(f"'{new_name}' already exists")
        self.rename_key(old_name, new_name)
        grp = self.registry.pop(old_name)
        grp.name = new_name
        if is_special_group(new_name):
            grp.hotkey = ''
        self.registry[new_name] = grp

    def delete_group(self, name: str) -> None:
        self.delete_key(name)
        self.registry.pop(name)

    def set_group_hotkey(self, name: str, key_str: str) -> None:
        for grp in self.registry.values():
            if grp.hotkey == key_str and grp.name != name:
                grp.hotkey = ''
        self.get_group_metadata(name).hotkey = key_str

    def pairs_in_group_by_session(self, gname: str) -> set[tuple]:
        ui = self.ui
        lbl = ui.key.type_label()
        out: set[tuple] = set()
        for k in ui.cd.ptr.keys():
            if k.type_label() != lbl:
                continue
            sess = str(k.session)
            valid = ui._all_inds_set_for_ptr(ui.cd.ptr.get(k))
            for pair in self.pairs_in_group(gname, sess):
                r, t = int(pair[0]), int(pair[1])
                if (r, t) in valid:
                    out.add((sess, r, t))
        return out

    def nd_keys_for_group(self, gname: str) -> list:
        ui = self.ui
        lbl = ui.key.type_label()
        seen, seen_id = [], set()
        for nk in ui.real_nd_keys():
            ckey = ui.type_key_for_nd(nk)
            if ckey is None or ckey.type_label() != lbl:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.ptr.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            if any((int(a), int(b)) in valid
                   for a, b in self.pairs_in_group(gname, sess)):
                nid = id(nk)
                if nid not in seen_id:
                    seen.append(nk)
                    seen_id.add(nid)
        return seen

    def iter_pairs(self, gname: str):
        ui = self.ui
        if gname not in ui.any_expanded_group_tags:
            return
        lbl = ui.key.type_label()
        dead = ui.active_selections.deleted
        for nk in ui.real_nd_keys():
            ckey = ui.type_key_for_nd(nk)
            if ckey is None or ckey.type_label() != lbl:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.ptr.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            pairs = self.pairs_in_group(gname, sess)
            if not pairs:
                continue
            for r, t in sorted((int(a), int(b)) for a, b in pairs):
                if r == t or (r, t) not in valid or (sess, r, t) in dead:
                    continue
                yield ckey, r, t

    def header_names(self) -> list[str]:
        def _gname_sort_key(n):
            try:
                return (0, int(n), '')
            except (ValueError, TypeError):
                return (1, 0, n)
        return sorted(self.registry, key=_gname_sort_key)

    def toggle_any_avail(self, gname: str) -> None:
        ui = self.ui
        if gname in ui.any_expanded_group_tags:
            ui.any_expanded_group_tags.discard(gname)
            ui.refresh_lists()
            return

        nds = self.nd_keys_for_group(gname)

        def _finish_expand():
            ui.any_expanded_group_tags.add(gname)
            ui.refresh_lists()

        if not nds:
            _finish_expand()
            return

        res = 'highres' if ui.resolution in ('hi', 'lo_hi') else 'lowres'

        def _chain(idx: int):
            if idx >= len(nds):
                _finish_expand()
                return
            ui.root._ensure_loaded(nds[idx], res, on_loaded=lambda: _chain(idx + 1))

        _chain(0)

    def set_hotkey_ui(self, group_name: str, key_str: str) -> None:
        key_str = key_str.strip().lower()
        valid_digits = [str(i) for i in range(1, 10)] + ['0']
        if key_str and key_str not in valid_digits and not (len(key_str) == 1 and key_str.isalpha()):
            QMessageBox.warning(None, "Hotkey",
                                "Enter a digit 1–9/0 or a single letter a–z.")
            return
        self.set_group_hotkey(group_name, key_str)
        self.changed.emit()

    def hotkey_handler(self, key_str: str, advance: bool = True,
                       collect_highlighted=None) -> None:
        nav   = self.ui
        panel = nav.root.pairs_view.pair_selection
        current_pair = nav.current_pair
        if current_pair is None:
            nav.root.show_transient_banner("Select a pair before using a group hotkey")
            return

        for grp in self.registry.values():
            gname, k = grp.name, grp.hotkey
            if not k or k != key_str:
                continue

            print(f"[hk] handler key={key_str!r} advance={advance} matched group={gname!r}")
            if not advance:
                highlighted = [current_pair]
            else:
                highlighted = (collect_highlighted(current_pair)
                               if collect_highlighted else [current_pair])

            changed = set()
            pair_changes, group_changes = {}, []
            any_mode = nav.session_any_mode

            for pair in highlighted:
                old = ('sel' if pair in nav.active_selections.selected
                       else 'del' if pair in nav.active_selections.deleted
                       else 'unsel')
                sess, p2 = nav.pair_sess_rt(pair)
                was_in = p2 in self.pairs_in_group(gname, sess)
                group_changes.append((gname, sess, p2, 'remove' if was_in else 'add'))
                if was_in:
                    self.discard_from_group(gname, sess, p2)
                else:
                    self.add_to_group(gname, sess, p2)
                if any_mode:
                    changed.add(pair)
                    continue
                if not was_in and pair in nav.active_selections.unselected:
                    nav.active_selections.set_pair_state(pair, 'sel')
                    pair_changes[pair] = (old, 'sel')
                    changed.add(pair)
                elif was_in and pair in nav.active_selections.selected:
                    has_groups = any(
                        p2 in self.pairs_in_group(g, sess)
                        for g in self
                        if not is_special_group(g)
                    )
                    if not has_groups:
                        nav.active_selections.set_pair_state(pair, 'unsel')
                        pair_changes[pair] = (old, 'unsel')
                        changed.add(pair)

            panel.push_undo(SelectionCommand(pair_changes, group_changes))
            nav.refresh_lists()
            if advance:
                next_idx = min(nav.current_pair_idx + 1, len(nav.all_inds) - 1)
                nav.set_current_pair(next_idx)
                panel._select_pair_in_list(panel._pair_at_all_inds_idx(next_idx))
            else:
                panel._select_pair_in_list(current_pair)
            nav.root.request_redraw()
            nav.root.neuron_network.draw()
            return
        print(f"[hk] handler key={key_str!r} NO group matched")
        nav.root.show_transient_banner(f"'{key_str}' is not assigned to any group hotkey")


class _SelectionData(JsonSavable):
    """Selections for one conn-type within a session."""

    def __init__(self):
        JsonSavable.__init__(self)
        self.selected:   set = set()
        self.unselected: set = set()
        self.deleted:    set = set()
        self.tags:       dict = {}   # {(ref,tgt): {groups,notes,tags}}

    def __setstate__(self, state: dict):
        def _to_set(v) -> set:
            if isinstance(v, set):
                return v
            if isinstance(v, dict) and '__set__' in v:
                v = v['__set__']
            return {tuple(x) if isinstance(x, list) else x for x in (v or [])}

        def _to_tuple_key_dict(v) -> dict:
            if isinstance(v, dict):
                if '__dict__' in v:
                    return {(tuple(k) if isinstance(k, list) else k): val
                            for k, val in v['__dict__']}
                result = {}
                for dk, dv in v.items():
                    key = (tuple(int(i) for i in dk.split(','))
                           if isinstance(dk, str) and ',' in dk else dk)
                    result[key] = dv
                return result
            return {}

        self.selected   = _to_set(state.get('selected', []))
        self.unselected = _to_set(state.get('unselected', []))
        self.deleted    = _to_set(state.get('deleted', []))
        self.tags       = _to_tuple_key_dict(state.get('tags', {}))

    def set_pair_state(self, pair: tuple, state: str):
        pair = tuple(pair)
        self.selected.discard(pair)
        self.unselected.discard(pair)
        self.deleted.discard(pair)
        if state == 'sel':
            self.selected.add(pair)
        elif state == 'unsel':
            self.unselected.add(pair)
        elif state == 'del':
            self.deleted.add(pair)

    def reset(self, all_pairs, selected=(), deleted=()):
        """Rebuild this bucket's state from scratch."""
        all_set  = {tuple(p) for p in all_pairs}
        sel_set  = {tuple(p) for p in selected}
        del_set  = {tuple(p) for p in deleted}
        self.selected   = sel_set & all_set
        self.deleted    = del_set & all_set
        self.unselected = all_set - self.selected - self.deleted


class SelectionData(JsonSavable):
    """Per-session, one _SelectionData per conn-type Key."""

    _custom_types = {'selections': (Key, _SelectionData)}

    def __init__(self, *, save_dir: str = '', nd_key: Key = None):
        JsonSavable.__init__(self)
        self.selections: dict[Key, _SelectionData] = _defaultdict(_SelectionData)
        self._save_dir = save_dir
        self._nd_key = nd_key

    def save_path(self, **_) -> str | None:
        if self._save_dir and self._nd_key is not None:
            return os.path.join(self._save_dir, str(self._nd_key.session))
        return None

    @staticmethod
    def as_pair_key(pair, session: str | None = None) -> Key:
        """Normalize a pair to Key(session, ref, tgt) for dict/set matching."""
        if isinstance(pair, Key) and pair.ref is not None and pair.tgt is not None:
            return pair
        p = tuple(pair)
        if len(p) >= 3:
            return Key.pair(p[0], p[1], p[2])
        if session is None:
            raise ValueError(f"session required for pair {p!r}")
        return Key.pair(session, p[0], p[1])

    @staticmethod
    def pairs_vals_map(pairs, vals) -> dict[Key, float]:
        return {SelectionData.as_pair_key(p): float(v) for p, v in zip(pairs or [], vals or [])}


class SelectionDataset(JsonSavable, Autosave):
    """Project-level owner of groups + per-session SelectionData."""

    def __init__(self, save_dir: str = ''):
        JsonSavable.__init__(self)
        self.groups = Groups()
        self.groups._save_dir = save_dir
        self.sessions: dict[Key, SelectionData] = {}
        self.save_dir = save_dir

    def save_path(self, **_) -> str:
        return os.path.join(self.save_dir, 'selection_dataset')

    def __setstate__(self, state: dict):
        self.save_dir = state.get('save_dir', self.save_dir)
        self.groups._save_dir = self.save_dir
        groups_v = state.get('groups', {})
        if isinstance(groups_v, dict) and '__ref__' in groups_v:
            path = groups_v['__ref__'][:-5]
            self.groups.load(path)
        else:
            self.groups.__setstate__(groups_v)
        self.sessions = {}
        for key_str, sd_v in state.get('sessions', {}).items():
            nd = Key.from_str(key_str)
            sd = SelectionData(save_dir=self.save_dir, nd_key=nd)
            if isinstance(sd_v, dict) and '__ref__' in sd_v:
                sd.load(sd_v['__ref__'][:-5])
            else:
                sd.__setstate__(sd_v)
            self.sessions[nd] = sd

    def get_selection_by_session(self, key: Key) -> SelectionData:
        nd = key.nd()
        sd = self.sessions.get(nd)
        if sd is None:
            sd = SelectionData(save_dir=self.save_dir, nd_key=nd)
            self.sessions[nd] = sd
        return sd


class SearchBar(QWidget):
    """Qt search bar — toggle via Ctrl+F.  Replaces SearchBar."""

    _MATCH_BG = QColor('#fff099')

    def __init__(self, parent_widget: QWidget,
                 get_listboxes: Callable,
                 on_style_reset: Callable):
        super().__init__(parent_widget)
        self._get_listboxes  = get_listboxes
        self._on_style_reset = on_style_reset
        self._matches: list  = []
        self._cur:    int    = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        layout.addWidget(QLabel('🔍'))
        self._entry = QLineEdit()
        layout.addWidget(self._entry)

        self._count_lbl = QLabel('')
        self._count_lbl.setFixedWidth(42)
        layout.addWidget(self._count_lbl)

        for txt, delta in [('▲', -1), ('▼', 1)]:
            btn = QPushButton(txt)
            btn.setFixedWidth(26)
            btn.clicked.connect(lambda _, d=delta: self.go(d))
            layout.addWidget(btn)

        close_btn = QPushButton('✕')
        close_btn.setFixedWidth(26)
        close_btn.clicked.connect(self.hide)
        layout.addWidget(close_btn)

        self._entry.textChanged.connect(lambda _: self._rebuild(preserve_cur=False))
        self._entry.returnPressed.connect(lambda: self.go(1))
        self._entry.installEventFilter(self)

        self.setVisible(False)

    @property
    def active(self) -> bool:
        return bool(self._entry.text())

    def eventFilter(self, obj, event):
        if (obj is self._entry
                and event.type() == QtCore.QEvent.KeyPress
                and event.key() == Qt.Key_Escape):
            self.hide()
            return True
        return super().eventFilter(obj, event)

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.setVisible(True)
            self._entry.setFocus()
            self._entry.selectAll()

    def hide(self):
        self._clear()
        self.setVisible(False)

    def go(self, delta: int):
        if not self._matches:
            return
        n = len(self._matches)
        self._cur = (self._cur + delta) % n
        self._apply_highlights()
        lb, i = self._matches[self._cur]
        lb.scrollToItem(lb.item(i))
        self._count_lbl.setText(f'{self._cur + 1}/{n}')

    def _rebuild(self, preserve_cur: bool = False):
        q = self._entry.text().lower()
        self._clear_highlights()
        old_cur = self._cur
        self._matches = []
        self._cur = -1
        if not q:
            self._count_lbl.setText('')
            return
        unsel, sel = self._get_listboxes()
        for lb in (unsel, sel):
            for i in range(lb.count()):
                it = lb.item(i)
                if q in it.text().lower():
                    self._matches.append((lb, i))
        n = len(self._matches)
        if n == 0:
            self._count_lbl.setText('0/0')
            return
        if preserve_cur and old_cur >= 0:
            self._cur = max(0, min(old_cur, n - 1))
        else:
            self._cur = 0
            lb, i = self._matches[0]
            lb.scrollToItem(lb.item(i))
        self._apply_highlights()
        self._count_lbl.setText(f'{self._cur + 1}/{n}')

    def _apply_highlights(self):
        for lb, i in self._matches:
            lb.item(i).setBackground(QBrush(self._MATCH_BG))
        self._on_style_reset()

    def _clear_highlights(self):
        for lb, i in self._matches:
            lb.item(i).setBackground(QBrush())
        self._on_style_reset()

    def _clear(self):
        self._clear_highlights()
        self._matches = []
        self._cur = -1
        self._count_lbl.setText('')
        self._entry.setText('')


class PairSelectionPanel(QWidget, UndoRedo):
    """Qt pair-selection panel."""

    _COMBO_SORT_KEY = staticmethod(
        lambda combo: (1, []) if not combo else (0, list(combo)))

    def __init__(self, parent: QWidget, data: SelectionData,
                 ui: 'CCGReviewUI', ui_state_cache: dict):
        QWidget.__init__(self, parent)
        self.ui               = ui
        self._collapsed_groups   = CollapseState()
        self.avail_list_pairs: list = []
        self.sel_list_pairs:   list = []
        self.sel_list_header_keys: list = []
        self._syncing_sel      = False
        # Debounce timer: prevents redrawing while scrolling quickly through the list.
        self._select_timer     = QTimer(self)
        self._select_timer.setSingleShot(True)
        self._select_timer.timeout.connect(self._do_pair_select_update)
        self._next_focus_pair: tuple | None = None
        self.__init_undo__()
        ui.groups.bind(ui)

        self._build(ui_state_cache)

    @property
    def data(self) -> 'SelectionData':
        # Live current-session SelectionData — never a stale capture (fixes cross-session
        # tag leak when switching sessions).
        return self.ui.sel_data

    def _build(self, ui_state_cache: dict):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        splitter = QSplitter(Qt.Horizontal)
        self._pair_list_pane = splitter
        for title, side, list_attr in (
            ('Available (0)', 'avail', 'unselected_list'),
            ('Selected (0)',  'sel',   'selected_list'),
        ):
            col    = QWidget()
            layout = QVBoxLayout(col)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(1)
            lbl_w = QLabel(title)
            setattr(self, f'_{side}_label_widget', lbl_w)
            setattr(self, f'_{side}_label', LabelVar(lbl_w))
            layout.addWidget(lbl_w)
            lst = PairListWidget(self)
            setattr(self, list_attr, lst)
            layout.addWidget(lst)
            splitter.addWidget(col)
        root.addWidget(splitter, stretch=1)

        self.search_bar = SearchBar(
            self,
            get_listboxes=lambda: (self.unselected_list, self.selected_list),
            on_style_reset=self._reapply_bookmark_list_styles,
        )
        root.addWidget(self.search_bar)

        btn_row    = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(2, 2, 2, 2)
        btn_layout.setSpacing(4)
        btn_layout.addWidget(QLabel("Sort by:"))
        self._sort_btns = ExclusiveButtonSet(on_change=self.refresh_lists, parent=btn_row)
        for key, label, cache_key, cb_attr, var_attr in (
            ('group', 'Group',     'sort_selected', '_sort_selected_cb', '_sort_selected'),
            ('tag',   'Tag',       'sort_by_tag',   '_sort_by_tag_cb',   '_sort_by_tag'),
            ('mean',  'Mean',      'sort_by_mean',  '_sort_by_mean_cb',  '_sort_by_mean'),
            ('min_p', 'Min p-val', 'sort_by_min_p', '_sort_by_min_p_cb', '_sort_by_min_p'),
            ('session', 'Session', 'sort_by_session', '_sort_by_session_cb', '_sort_by_session'),
        ):
            btn, var = self._sort_btns.add(
                key, label, checked=ui_state_cache.get(cache_key, False))
            setattr(self, cb_attr, btn)
            setattr(self, var_attr, var)
            btn_layout.addWidget(btn)
        btn_layout.addStretch()
        root.addWidget(btn_row)

        for lst, action, other in (
            (self.unselected_list, 'add',    self.selected_list),
            (self.selected_list,   'remove', self.unselected_list),
        ):
            lst.itemClicked.connect(
                lambda it, w=lst: self._on_item_clicked(w, it))
            lst.itemDoubleClicked.connect(
                lambda it, w=lst: self._on_item_double_clicked(w, it))
            lst.customContextMenuRequested.connect(
                lambda pos, w=lst, a=action: self._ctx_menu(pos, w, a))
            lst.itemPressed.connect(lambda _, o=other: self._clear_other(o))

    def _list_cursor_follows_action(self) -> bool:
        return self.ui.root.settings.list_cursor_follows_action

    def _clear_acted_highlight(self) -> None:
        self._next_focus_pair = None

    def _mark_next_focus_pair(self, inds: tuple) -> None:
        self._next_focus_pair = tuple(int(x) for x in inds)

    def _apply_acted_highlight(self, widget: PairListWidget, item: QListWidgetItem):
        item.setBackground(QBrush(_C_ACTED_BG))

    def _focus_next_pair(self, *, move_cursor: bool = True) -> None:
        inds = self._next_focus_pair
        if inds is None:
            return
        b = self.ui.active_selections
        if inds in b.selected:
            dest = self.selected_list
        elif inds in b.unselected or inds in self.ui.active_selections.deleted:
            dest = self.unselected_list
        else:
            return
        for i in range(dest.count()):
            it   = dest.item(i)
            pair = it.data(_ROLE_PAIR)
            if pair is None:
                continue
            if pair == inds:
                if move_cursor:
                    dest.setFocus()
                    dest.clearSelection()
                    it.setSelected(True)
                    dest.setCurrentItem(it)
                    dest.scrollToItem(it)
                self._apply_acted_highlight(dest, it)
                return

    def _clear_other(self, other: PairListWidget):
        if self._syncing_sel:
            return
        mods = QApplication.keyboardModifiers()
        if not (mods & (Qt.ControlModifier | Qt.ShiftModifier | Qt.MetaModifier)):
            self._clear_acted_highlight()
            self._syncing_sel = True
            other.clearSelection()
            self._syncing_sel = False

    def _save_scroll_positions(self):
        def _frac(lb):
            sb = lb.verticalScrollBar()
            return sb.value() / sb.maximum() if sb.maximum() else 0.0
        return _frac(self.unselected_list), _frac(self.selected_list)

    def _restore_scroll_positions(self, unsel_frac, sel_frac):
        for lb, frac in [(self.unselected_list, unsel_frac),
                         (self.selected_list,   sel_frac)]:
            sb = lb.verticalScrollBar()
            sb.setValue(int(frac * sb.maximum()))

    def _populate_avail_list(self, ui, data, should_gray):
        self.avail_list_pairs = []
        if ui.session_any_mode:
            return
        bm = ui.bookmarked_pairs
        for inds in sorted(ui.active_selections.unselected):
            _, raw  = ui.pair_sess_rt(inds)
            pair_t  = tuple(int(x) for x in raw)
            gnames  = group_names_for_pair(data, ui, inds)
            label   = pair_label(inds,
                                 bookmarked=bm_key(inds, False) in bm,
                                 group_names=gnames,
                                 pair_tags=data.selections[ui.key].tags.get(pair_t, {}))
            it = QListWidgetItem(label)
            if should_gray(inds):
                it.setForeground(QBrush(_C_GRAY_FG))
            it.setData(_ROLE_PAIR, inds)
            self.unselected_list.addItem(it)
            self.avail_list_pairs.append((inds, None))

        deleted_inds = ui.active_selections.deleted
        if deleted_inds:
            sep = QListWidgetItem('── deleted ──')
            sep.setForeground(QBrush(_C_SEP_FG))
            sep.setBackground(QBrush(_C_SEP_BG))
            sep.setFlags(Qt.ItemIsEnabled)
            self.unselected_list.addItem(sep)
            self.avail_list_pairs.append((_SEPARATOR_ROW, 'deleted'))
            for inds in sorted(deleted_inds):
                it = QListWidgetItem(f"[{inds[0]}, {inds[1]}]")
                it.setForeground(QBrush(_C_GRAY_FG))
                it.setData(_ROLE_PAIR, inds)
                it.setData(_ROLE_TAG, 'deleted')
                self.unselected_list.addItem(it)
                self.avail_list_pairs.append((inds, 'deleted'))

    def _sel_insert_pair(self, inds, should_gray, any_mode):
        ui     = self.ui
        bm     = ui.bookmarked_pairs
        _, raw = ui.pair_sess_rt(inds)
        pair_t = tuple(int(x) for x in raw)
        gnames = group_names_for_pair(self.data, ui, inds)
        label  = pair_label(inds,
                            bookmarked=bm_key(inds, any_mode) in bm,
                            group_names=gnames,
                            pair_tags=self.data.selections[ui.key].tags.get(pair_t, {}),
                            any_mode=any_mode)
        it = QListWidgetItem(label)
        if should_gray(inds):
            it.setForeground(QBrush(_C_GRAY_FG))
        it.setData(_ROLE_PAIR, inds)
        self.selected_list.addItem(it)
        self.sel_list_pairs.append(inds)
        self.sel_list_header_keys.append(None)

    def _sel_insert_header(self, text: str, count: int) -> bool:
        is_col  = self._collapsed_groups.is_collapsed(text)
        display = group_header_label(text, count, is_col)
        it = QListWidgetItem(display)
        it.setForeground(QBrush(_C_HDR_FG))
        it.setBackground(QBrush(_C_HDR_BG))
        it.setFlags(Qt.ItemIsEnabled)
        it.setData(_ROLE_TAG,  'header')
        it.setData(_ROLE_HKEY, text)
        self.selected_list.addItem(it)
        self.sel_list_pairs.append(None)
        self.sel_list_header_keys.append(text)
        return is_col

    def _pair_group_combo(self, inds):
        ui = self.ui
        sess, pair = ui.pair_sess_rt(inds)
        pair = tuple(int(x) for x in pair)
        return tuple(sorted(
            g for g in ui.groups
            if not is_special_group(g)
            and pair in ui.groups.pairs_in_group(g, sess)
        ))

    def _any_metric_values(self, kind: str) -> dict:
        """Metric per cross-session handle for Mean/Min-p sort. Eager-loads every
        session's CCG and captures the value while it is memory-resident."""
        ui  = self.ui
        res = 'highres' if ui.resolution in ('hi', 'lo_hi') else 'lowres'
        by_nd = _defaultdict(list)
        for h in ui.cross_session_handles:
            by_nd[h[0].nd()].append(h)
        ui.root._ensure_sessions_loaded(list(by_nd), res)
        out, worst = {}, (1.0 if kind == 'minp' else -1e18)
        for nd_key, hs in by_nd.items():
            ccgd = ui.cd.ccg_for(nd_key, res)
            has  = ccgd is not None and ccgd.ccg is not None
            seg  = ccgd.ccg.shape[0] if has else 0
            for h in hs:
                out[h] = (worst if not has else
                          ccgd.mean_ccg(int(h[1]), int(h[2]), seg) if kind == 'mean'
                          else ccgd.min_pval(int(h[1]), int(h[2]), seg))
        return out

    def _populate_selected_list(self, ui, should_gray) -> int:
        """Populate selected list for all sort modes. Returns total_any_count."""
        self.sel_list_pairs       = []
        self.sel_list_header_keys = []
        any_mode   = ui.session_any_mode
        sort_group = self._sort_selected_cb.isChecked()
        sort_tag   = self._sort_by_tag_cb.isChecked()
        sort_mean  = self._sort_by_mean_cb.isChecked()
        sort_minp  = self._sort_by_min_p_cb.isChecked()
        sort_session = self._sort_by_session_cb.isChecked()
        def _ins(inds):
            self._sel_insert_pair(inds, should_gray, any_mode)

        def _ins_hdr(text, count) -> bool:
            return self._sel_insert_header(text, count)

        total_any_count = 0
        if any_mode:
            dead      = ui.active_selections.deleted
            _expanded = ui.any_expanded_group_tags

            def _gname_sort_key(n):
                try:
                    return (0, int(n), '')
                except (ValueError, TypeError):
                    return (1, 0, n)
            all_gnames = sorted(
                (g for g in ui.groups if not g.startswith('__')),
                key=_gname_sort_key)

            _all_trips: set = set()
            for _gn in all_gnames:
                _all_trips |= (ui.groups.pairs_in_group_by_session(_gn) - dead)
            total_any_count = len(_all_trips)

            def _any_hdr(hdr_text, n, key=None):
                exp = hdr_text in _expanded
                hdr = f"── {hdr_text} ({n}) ──" + ("" if exp else " >>")
                it  = QListWidgetItem(hdr)
                it.setForeground(QBrush(_C_HDR_FG))
                it.setBackground(QBrush(_C_HDR_BG))
                it.setFlags(Qt.ItemIsEnabled)
                it.setData(_ROLE_TAG,  'header')
                it.setData(_ROLE_HKEY, key or hdr_text)
                self.selected_list.addItem(it)
                self.sel_list_pairs.append(None)
                self.sel_list_header_keys.append(key or hdr_text)
                return exp

            sel_set = ui.active_selections.selected
            sel_handles = [h for h in ui.cross_session_handles
                           if (str(h[0].session), int(h[1]), int(h[2])) in sel_set]

            if (sort_mean or sort_minp) and sel_set:
                vals  = self._any_metric_values('mean' if sort_mean else 'minp')
                order = sorted(sel_handles, key=lambda h: vals[h], reverse=sort_mean)
                for h in order:
                    _ins(h)
                total_any_count = len(order)
            elif sort_session or sort_mean or sort_minp:
                by_sess = _defaultdict(list)
                for h in sel_handles:
                    by_sess[str(h[0].session)].append(h)
                for sess in sorted(by_sess):
                    hs = by_sess[sess]
                    if _any_hdr(sess, len(hs), key=sess):
                        for h in hs:
                            _ins(h)
                total_any_count = len(sel_handles)
            elif sort_group:
                pair_tags: dict = {}
                for gname in all_gnames:
                    for trip in ui.groups.pairs_in_group_by_session(gname):
                        if trip not in dead:
                            pair_tags.setdefault(trip, set()).add(gname)
                buckets = _defaultdict(list)
                for trip, tags in sorted(pair_tags.items()):
                    buckets[tuple(sorted(tags))].append(trip)
                for combo in sorted(buckets.keys(), key=self._COMBO_SORT_KEY):
                    hdr_text = ', '.join(combo) if combo else '(untagged)'
                    trips    = buckets[combo]
                    if _any_hdr(hdr_text, len(trips)):
                        for sess, r, t in trips:
                            nk   = ui.nd_key_for_session(sess)
                            ckey = ui.type_key_for_nd(nk) if nk else None
                            if ckey:
                                _ins((ckey, r, t))
            else:
                for gname in all_gnames:
                    trips_g = ui.groups.pairs_in_group_by_session(gname)
                    n_tag   = len(trips_g - dead)
                    if _any_hdr(gname, n_tag):
                        for row in ui.groups.iter_pairs(gname):
                            _ins(row)

        elif sort_mean:
            seg_idx = self.ui.seg_idx(self.ui.current_segment)
            _ccg_d = self.ui.ccg_data
            if _ccg_d is not None:
                for inds in sorted(ui.active_selections.selected,
                                   key=lambda inds, _s=seg_idx: _ccg_d.mean_ccg(int(inds[0]), int(inds[1]), _s),
                                   reverse=True):
                    _ins(inds)
            else:
                for inds in sorted(ui.active_selections.selected):
                    _ins(inds)
        elif sort_minp:
            seg_idx = self.ui.seg_idx(self.ui.current_segment)
            _ccg_d = self.ui.ccg_data
            if _ccg_d is not None:
                for inds in sorted(ui.active_selections.selected,
                                   key=lambda inds, _s=seg_idx: _ccg_d.min_pval(int(inds[0]), int(inds[1]), _s)):
                    _ins(inds)
            else:
                for inds in sorted(ui.active_selections.selected):
                    _ins(inds)
        elif sort_group:
            buckets = _defaultdict(list)
            for inds in sorted(ui.active_selections.selected):
                buckets[self._pair_group_combo(inds)].append(inds)
            for combo in sorted(buckets.keys(), key=self._COMBO_SORT_KEY):
                pairs_in = buckets[combo]
                hdr_text = ', '.join(combo) if combo else '(untagged)'
                if not _ins_hdr(hdr_text, len(pairs_in)):
                    for inds in pairs_in:
                        _ins(inds)
        elif sort_tag:
            tag_buckets: dict = _defaultdict(list)
            untagged = []
            non_internal = [g for g in ui.groups.defined_groups
                            if not is_special_group(g)]
            for inds in sorted(ui.active_selections.selected):
                _s, _p = ui.pair_sess_rt(inds)
                _p = tuple(int(x) for x in _p)
                tags = [g for g in non_internal
                        if _p in ui.groups.pairs_in_group(g, _s)]
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
            for inds in sorted(ui.active_selections.selected):
                _ins(inds)

        return total_any_count

    def _update_list_labels(self, ui, data, total_any_count):
        if ui.session_any_mode:
            self._avail_label.set("Available (0) — Any mode")
            self._sel_label.set(f"Selected ({total_any_count})")
        else:
            n_avail = len(ui.active_selections.unselected)
            del_suffix = f", {len(ui.active_selections.deleted)} deleted" if ui.active_selections.deleted else ""
            self._avail_label.set(f"Available ({n_avail}{del_suffix})")
            self._sel_label.set(f"Selected ({len(ui.active_selections.selected)})")

        ui.root.jitter_controller.apply_list_colors()
        ui.root.status_bar.refresh()

    def refresh_lists(self):
        ui, data = self.ui, self.data
        ab = ui.active_selections
        unsel_frac, sel_frac = self._save_scroll_positions()
        self.unselected_list.clear()
        self.selected_list.clear()

        if ui.session_any_mode:
            ui.root.any_session.rebuild_pair_handles()
            ui.root.any_session.sync_selection_from_universe()

        net = ui.root.neuron_network
        should_gray = net.build_should_gray(ui.session_any_mode)

        self._sort_by_mean_cb.setEnabled(True)
        self._sort_by_min_p_cb.setEnabled(True)
        self._sort_btns.button('session').setVisible(ui.session_any_mode)

        self._populate_avail_list(ui, data, should_gray)
        total_any = self._populate_selected_list(ui, should_gray)
        self._update_list_labels(ui, data, total_any)

        if self.search_bar.active:
            self.search_bar._rebuild(preserve_cur=True)
        else:
            self._reapply_bookmark_list_styles()

        cursor_follows = self._list_cursor_follows_action()
        if self._next_focus_pair is not None:
            self._focus_next_pair(move_cursor=cursor_follows)
        if self._next_focus_pair is None or not cursor_follows:
            self._restore_scroll_positions(unsel_frac, sel_frac)

    def _on_item_clicked(self, widget: PairListWidget, item: QListWidgetItem):
        """Single click: debounced pair navigation."""
        self._clear_acted_highlight()
        if item.data(_ROLE_TAG) in ('header', 'sep'):
            return
        pair = item.data(_ROLE_PAIR)
        if pair is not None:
            self.ui.set_current_pair(self.ui.get_pair_index(pair))
        # Debounced update
        self._select_timer.stop()
        self._select_timer.start(180)

    def _on_item_double_clicked(self, widget: PairListWidget, item: QListWidgetItem):
        """Double click: shuttle pair or toggle header collapse."""
        tag  = item.data(_ROLE_TAG)
        hkey = item.data(_ROLE_HKEY)

        if tag == 'header' and hkey is not None:
            # Toggle group collapse (selected list) or any-mode expand
            ui = self.ui
            if ui.session_any_mode:
                if hkey in ui.any_expanded_group_tags:
                    ui.any_expanded_group_tags.discard(hkey)
                    self.refresh_lists()
                else:
                    ui.groups.toggle_any_avail(hkey)
            else:
                self._collapsed_groups.toggle(hkey)
                _, sel_frac = self._save_scroll_positions()
                self.refresh_lists()
                self._restore_scroll_positions(0.0, sel_frac)
            return

        if widget is self.unselected_list:
            self.move_to_selected(item)
        else:
            self.move_to_unselected(item)

    def _on_arrow_key(self):
        """Arrow key: update plot after list moves cursor."""
        lb = self.unselected_list if self.unselected_list.hasFocus() else self.selected_list
        cur = lb.currentItem()
        if cur is None:
            return
        pair = cur.data(_ROLE_PAIR)
        if pair is not None and not is_separator_row(pair):
            self.ui.set_current_pair(self.ui.get_pair_index(pair))
        self._select_timer.stop()
        self._select_timer.start(180)

    def _on_enter_key(self, widget: PairListWidget):
        """Return key: same as double-click."""
        cur = widget.currentItem()
        if cur:
            self._on_item_double_clicked(widget, cur)

    def _on_list_key(self, event):
        """Handle key events forwarded from PairListWidget."""
        ui   = self.ui
        key  = event.key()
        mods = event.modifiers()   # per-event modifiers: reliable on macOS, unlike the global state
        ctrl = bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
        shift= bool(mods & Qt.ShiftModifier)
        print(f"[hk] list_key key={key} text={event.text()!r} ctrl={ctrl} shift={shift}")

        if ctrl and key == Qt.Key_B:
            self._bookmark_toggle_current()
            return
        if ctrl and key == Qt.Key_Z:
            self.undo()
            return
        if ctrl and key == Qt.Key_Y:
            self.redo()
            return
        if ctrl and key == Qt.Key_F:
            self.search_bar.toggle()
            return

        # Group hotkey. Bare digit/letter tags the current pair and advances to the
        # next; Shift+digit/letter tags without advancing (multi-tag). Shift remaps the
        # digit row to symbols on macOS (Shift+1 → "!"), so shifted symbols are mapped
        # back to their digit via _SHIFT_DIGITS.
        if not ctrl:
            if Qt.Key_A <= key <= Qt.Key_Z:
                c = chr(key).lower()
            elif Qt.Key_0 <= key <= Qt.Key_9:
                c = chr(key)
            elif key in _SHIFT_DIGITS:
                c = _SHIFT_DIGITS[key]
            else:
                c = None
            if c is not None:
                advance = not shift
                print(f"[hk] -> group hotkey c={c!r} advance={advance}")
                ui.groups.hotkey_handler(
                    c, advance=advance,
                    collect_highlighted=self._collect_highlighted_pairs)
                return

    def _do_pair_select_update(self):
        ui = self.ui
        ui.root.pairs_view.spike_pairs.clear()
        ui.root.jitter_controller.mark_viewed()
        ui.root.neuron_network._focused_pair = None
        ui.root.neuron_network._focus_pair_entry.setText("")
        ui.root.neuron_network._focus_pair_info_label.setText("")
        ui.root.request_redraw()

    def move_to_selected(self, item: QListWidgetItem = None):
        ui = self.ui
        if ui.session_any_mode:
            return
        self._select_timer.stop()

        if item is None:
            item = self.unselected_list.currentItem()
        if item is None:
            return

        entry = self.avail_list_pairs[self.unselected_list.row(item)]
        if is_separator_row(entry):
            return
        inds, pred_group = entry

        if inds in ui.active_selections.selected:
            if pred_group is not None:
                ui.groups.add_to_group(pred_group,
                                          ui.current_session_str, inds)
                self.refresh_lists()
            return
        self.push_undo(SelectionCommand({inds: ('unsel', 'sel')}, []))
        ui.active_selections.set_pair_state(inds, 'sel')
        if pred_group is not None:
            ui.groups.add_to_group(pred_group,
                                      ui.current_session_str, inds)
        self._mark_next_focus_pair(inds)
        self.refresh_lists()
        ui.set_current_pair(ui.get_pair_index(inds))
        ui.root.request_redraw()
        ui.root.neuron_network.draw()

    def move_to_unselected(self, item: QListWidgetItem = None):
        ui = self.ui
        self._select_timer.stop()

        if item is None:
            item = self.selected_list.currentItem()
        if item is None:
            return

        i   = self.selected_list.row(item)
        tag = item.data(_ROLE_TAG)
        if tag == 'header':
            hkey = item.data(_ROLE_HKEY)
            if hkey is not None:
                self._on_item_double_clicked(self.selected_list, item)
            return

        inds = self.sel_list_pairs[i]
        if inds is None:
            return

        if ui.session_any_mode:
            return

        self.push_undo(SelectionCommand({inds: ('sel', 'unsel')}, []))
        ui.active_selections.set_pair_state(inds, 'unsel')
        self._mark_next_focus_pair(inds)
        self.refresh_lists()
        ui.set_current_pair(ui.get_pair_index(inds))
        ui.root.request_redraw()
        ui.root.neuron_network.draw()

    def _select_pair_in_list(self, inds):
        if inds is None:
            return
        data = self.ui.active_selections

        def _focus(widget, idx):
            it = widget.item(idx)
            widget.clearSelection()
            it.setSelected(True)
            widget.setCurrentItem(it)
            widget.scrollToItem(it)

        if inds in data.unselected or inds in self.ui.active_selections.deleted:
            for i, entry in enumerate(self.avail_list_pairs):
                if not is_separator_row(entry) and entry[0] == inds:
                    _focus(self.unselected_list, i)
                    return
        elif inds in data.selected:
            for i, entry in enumerate(self.sel_list_pairs):
                if entry == inds:
                    _focus(self.selected_list, i)
                    return

    def _selected_pair_from_lists(self):
        for lb, mp in [(self.unselected_list, self.avail_list_pairs),
                       (self.selected_list,   self.sel_list_pairs)]:
            items = lb.selectedItems()
            if not items:
                continue
            i = lb.row(items[-1])
            if mp and i < len(mp):
                entry = mp[i]
                if entry is None or is_separator_row(entry):
                    continue
                inds = entry[0] if (isinstance(entry, tuple) and len(entry) == 2
                                    and isinstance(entry[1], str)) else entry
                if self.ui.session_any_mode:
                    return (str(inds[0].session), int(inds[1]), int(inds[2]))
                return (int(inds[0]), int(inds[1]))
        return None

    def _selected_pairs_from_lists(self) -> list:
        any_mode = self.ui.session_any_mode
        out, seen = [], set()

        def _add(inds):
            pair = ((str(inds[0].session), int(inds[1]), int(inds[2])) if any_mode
                    else (int(inds[0]), int(inds[1])))
            if pair not in seen:
                seen.add(pair)
                out.append(pair)

        for it in self.unselected_list.selectedItems():
            entry = self.avail_list_pairs[self.unselected_list.row(it)]
            if not is_separator_row(entry):
                _add(entry[0])
        for it in self.selected_list.selectedItems():
            inds = self.sel_list_pairs[self.selected_list.row(it)]
            if inds is not None:
                _add(inds)
        out.sort(key=lambda x: (str(x[0]), x[1], x[2]) if len(x) == 3 else ('', x[0], x[1]))
        return out

    def _pair_at_all_inds_idx(self, idx: int):
        ui = self.ui
        if ui.session_any_mode:
            hl = ui.cross_session_handles or []
            if idx < 0 or idx >= len(hl):
                return None
            ck, r, t = hl[idx]
            return (str(ck.session), int(r), int(t))
        row = ui.all_inds[idx]
        return tuple(int(x) for x in row)

    def _next_inds_after(self, idx: int, pred) -> int | None:
        inds = self.ui.all_inds
        n = len(inds)
        for i in range(idx + 1, n):
            if pred(tuple(inds[i])):
                return i
        for i in range(idx):
            if pred(tuple(inds[i])):
                return i
        return None

    def _pair_state(self, pair) -> str:
        sel = self.ui.active_selections
        return ('sel' if pair in sel.selected else
                'del' if pair in sel.deleted else 'unsel')

    def _transition(self, pair, new_state: str, *,
                    goto_next_pred=None,
                    scroll_save: tuple = (True, True)) -> None:
        """Set pair state, optionally advance cursor, then redraw."""
        ui = self.ui
        old = self._pair_state(pair)
        unsel_frac, sel_frac = self._save_scroll_positions()
        self.push_undo(SelectionCommand({pair: (old, new_state)}, []))
        ui.active_selections.set_pair_state(pair, new_state)
        next_idx = (self._next_inds_after(ui.current_pair_idx, goto_next_pred)
                    if goto_next_pred else None)
        if next_idx is not None:
            ui.set_current_pair(next_idx)
        self.refresh_lists()
        self._restore_scroll_positions(
            unsel_frac if scroll_save[0] else 0.0,
            sel_frac   if scroll_save[1] else 0.0)
        focus = (self._pair_at_all_inds_idx(next_idx)
                 if next_idx is not None else pair)
        self._select_pair_in_list(focus)
        ui.root.neuron_network.draw()
        ui.root.request_redraw()

    def _transition_many(self, changes: dict) -> None:
        """Bulk state change: {pair: (old, new)}. No cursor jump."""
        if not changes:
            return
        ui = self.ui
        unsel_frac, sel_frac = self._save_scroll_positions()
        self.push_undo(SelectionCommand(changes, []))
        for p, (_, new) in changes.items():
            ui.active_selections.set_pair_state(p, new)
        self.refresh_lists()
        self._restore_scroll_positions(unsel_frac, sel_frac)
        ui.root.neuron_network.draw()

    def _ctx_menu(self, pos, widget: PairListWidget, action: str):
        ui         = self.ui
        global_pos = widget.viewport().mapToGlobal(pos)
        _any       = ui.session_any_mode

        if action == 'add':
            pairs, deleted_pairs = [], []
            for it in widget.selectedItems():
                entry = self.avail_list_pairs[widget.row(it)]
                if is_separator_row(entry):
                    continue
                inds, tag = entry
                (deleted_pairs if tag == 'deleted' else pairs).append(inds)
        else:
            pairs = []
            for it in widget.selectedItems():
                inds = self.sel_list_pairs[widget.row(it)]
                if inds is not None:
                    pairs.append(inds)
            deleted_pairs = []

        n, nd = len(pairs), len(deleted_pairs)
        menu  = QMenu(widget)
        _lbl  = lambda label, count: f"{label} ({count})" if count > 1 else label

        if action == 'add' and not _any:
            if pairs:
                menu.addAction(_lbl("Move to Selected", n),
                               lambda pp=pairs: self._transition_many(
                                   {p: (self._pair_state(p), 'sel') for p in pp
                                    if self._pair_state(p) != 'sel'}))
                menu.addAction(_lbl("Move to Deleted", n),
                               lambda pp=pairs: self._transition_many(
                                   {p: ('unsel', 'del') for p in pp}))
            if deleted_pairs:
                menu.addAction(_lbl("Restore to Available", nd),
                               lambda pp=deleted_pairs: self._transition_many(
                                   {p: ('del', 'unsel') for p in pp}))
        elif action != 'add':
            if not _any:
                menu.addAction(_lbl("Move to Available", n),
                               lambda pp=pairs: self._transition_many(
                                   {p: ('sel', 'unsel') for p in pp}))
            if not _any or pairs:
                menu.addAction(_lbl("Move to Deleted", n),
                               lambda pp=pairs: self._transition_many(
                                   {p: ('sel', 'del') for p in pp}))

        menu.addSeparator()

        grp_menu = QMenu("Group tag", menu)
        grp_menu.addAction("Create new group…",
                           lambda: CreateGroupDialog.show(ui.sel_data, ui.left_container.pair_selection))
        regular = [g for g in sorted(self.ui.groups) if not is_special_group(g)]
        special = [g for g in sorted(self.ui.groups) if is_special_group(g)]
        all_groups = [(g, g) for g in regular] + [(g, g[len(_SPECIAL_PREFIX):]) for g in special]
        if all_groups:
            grp_menu.addSeparator()
        for i, (gname, display) in enumerate(all_groups):
            if i == len(regular) and special and regular:
                grp_menu.addSeparator()
            if pairs:
                all_in = all(p2 in ui.groups.pairs_in_group(gname, s2)
                             for p in pairs for s2, p2 in (ui.pair_sess_rt(p),))
                grp_menu.addAction(f"{'✓ ' if all_in else '  '}{display}",
                                   lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))
        menu.addMenu(grp_menu)

        if pairs:
            menu.addSeparator()
            tog_tuples = [tuple(x) for x in ui.together_pairs]
            all_pinned = all(tuple(p) in tog_tuples for p in pairs)
            menu.addAction("Remove from 'Show Together'" if all_pinned else "Show Together",
                           lambda pp=pairs: ui.toggle_together(pp))
            if ui.together_pairs:
                menu.addAction(
                    f"Clear 'Show Together' ({len(ui.together_pairs)} pairs)",
                    ui._clear_together)

        if n == 1:
            menu.addSeparator()
            p = pairs[0]
            _, _rt = ui.pair_sess_rt(p)
            has_tags = _rt in self.data.selections[ui.key].tags
            def _open_pair_tags():
                inds = ui.current_pair_inds
                ref, tgt = int(inds[0]), int(inds[1])
                pair = (ref, tgt)
                sd = ui.sel_data.selections[ui.key]
                dlg = PairTagsDialog(ref, tgt, sd.tags.get(pair, {}), self)
                if dlg.exec():
                    entry = dlg.result()
                    if entry:
                        sd.tags[pair] = entry
                    else:
                        sd.tags.pop(pair, None)
            menu.addAction(f"{'✓ ' if has_tags else ''}Pair tags…", _open_pair_tags)

        menu.addSeparator()
        menu.addAction("Export view as PNG…",
                       lambda: ui._export_mgr._export_current_view('png'))
        menu.addAction("Export view as PDF…",
                       lambda: ui._export_mgr._export_current_view('pdf'))

        if ui.session_any_mode:
            all_names = ({g for g in ui.groups if not g.startswith('__')}
                         | {str(k.session) for k in ui.real_nd_keys()})
            menu.addSeparator()
            menu.addAction("Collapse all groups",
                           lambda: (ui.any_expanded_group_tags.clear(),
                                    self.refresh_lists()))
            menu.addAction("Expand all groups",
                           lambda: (ui.any_expanded_group_tags.update(all_names),
                                    self.refresh_lists()))
        elif self._sort_selected.get() or self._sort_by_tag.get():
            all_names = [g for g in ui.groups if not g.startswith('__')]
            menu.addSeparator()
            menu.addAction("Collapse all groups",
                           lambda: (self._collapsed_groups.collapse_all(all_names),
                                    self.refresh_lists()))
            menu.addAction("Expand all groups",
                           lambda: (self._collapsed_groups.expand_all(),
                                    self.refresh_lists()))

        menu.exec_(global_pos)

    def _on_delete_pair(self, event=None):
        ui = self.ui
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        if ui.session_any_mode:
            self._on_delete_pair_any()
        elif tuple(int(x) for x in ui.all_inds[ui.current_pair_idx]) in ui.active_selections.selected:
            self._on_delete_pair_single()
        else:
            self._on_toggle_deleted()

    def _on_delete_pair_any(self):
        ui = self.ui
        trip = self._pair_at_all_inds_idx(ui.current_pair_idx)
        if trip is None or trip not in ui.active_selections.selected:
            return
        hl_old = list(ui.cross_session_handles or ())
        next_trip = next(
            ((str(ck.session), int(r), int(t))
             for i, (ck, r, t) in enumerate(hl_old)
             if i != ui.current_pair_idx
             and (str(ck.session), int(r), int(t)) in ui.active_selections.selected),
            None)
        self._transition(trip, 'del', scroll_save=(False, True))
        if next_trip is not None:
            ui.set_current_pair(ui.get_pair_index(next_trip))
        elif ui.all_inds.size:
            ui.set_current_pair(min(ui.current_pair_idx, len(ui.all_inds) - 1))

    def _on_delete_pair_single(self):
        inds = tuple(int(x) for x in self.ui.all_inds[self.ui.current_pair_idx])
        self._transition(inds, 'del',
                         goto_next_pred=lambda p: p in self.ui.active_selections.selected,
                         scroll_save=(False, True))

    def _on_toggle_deleted(self):
        ui = self.ui
        inds = tuple(int(x) for x in ui.all_inds[ui.current_pair_idx])
        going_del = inds not in ui.active_selections.deleted
        self._transition(inds, 'del' if going_del else 'unsel',
                         goto_next_pred=(lambda p: p in ui.active_selections.unselected) if going_del else None,
                         scroll_save=(True, False))

    def _bookmark_toggle_current(self, event=None):
        ui   = self.ui
        inds = self._selected_pair_from_lists()
        if inds is None:
            if ui.current_pair_idx >= len(ui.all_inds):
                return
            inds = self._pair_at_all_inds_idx(ui.current_pair_idx)
            if inds is None:
                return
        inds = bm_key(inds, ui.session_any_mode)
        bm = ui.bookmarked_pairs
        bm.discard(inds) if inds in bm else bm.add(inds)
        self.refresh_lists()

    def _clear_bookmarks(self):
        if self.ui.bookmarked_pairs:
            self.ui.bookmarked_pairs.clear()
            self.refresh_lists()

    def _reapply_bookmark_list_styles(self):
        bm = self.ui.bookmarked_pairs
        if not bm:
            return
        any_mode = self.ui.session_any_mode
        for i, (raw, _) in enumerate(self.avail_list_pairs):
            if not is_separator_row(raw) and bm_key(raw, any_mode) in bm:
                it = self.unselected_list.item(i)
                it.setForeground(QBrush(_C_BM_FG))
                it.setBackground(QBrush(_C_BM_BG))
        for i, entry in enumerate(self.sel_list_pairs):
            if entry is not None and bm_key(entry, any_mode) in bm:
                it = self.selected_list.item(i)
                it.setForeground(QBrush(_C_BM_FG))
                it.setBackground(QBrush(_C_BM_BG))

    def _collect_highlighted_pairs(self, current_pair: tuple) -> list:
        """Return pairs highlighted in both listboxes, falling back to current_pair."""
        result = []
        for it in self.unselected_list.selectedItems():
            entry = self.avail_list_pairs[self.unselected_list.row(it)]
            if not is_separator_row(entry) and entry[1] != 'deleted':
                result.append(entry[0])
        for it in self.selected_list.selectedItems():
            inds = self.sel_list_pairs[self.selected_list.row(it)]
            if inds is not None:
                result.append(inds)
        return result or [current_pair]

    def _show_hotkeys_dialog(self):
        """Show a dialog listing all keyboard shortcuts."""
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
        QMessageBox.information(None, "Keyboard Shortcuts", hotkeys_text)

    # Highlight color for undo/redo indicators (matches CCG baseline orange)
    _UNDO_HIGHLIGHT = '#ff7f0e'

    def apply_command(self, cmd: 'SelectionCommand', reverse: bool = False) -> None:
        ui = self.ui
        for pair, (old, new) in cmd.pair_changes.items():
            target = old if reverse else new
            if target is None:
                ui.active_selections.unselected.discard(pair)
            else:
                ui.active_selections.set_pair_state(pair, target)
        grps = ui.groups
        for g, s, p, op in cmd.group_changes:
            effective = ('remove' if op == 'add' else 'add') if reverse else op
            if effective == 'add':
                grps.add_to_group(g, s, p)
            else:
                grps.discard_from_group(g, s, p)
        changed = set(cmd.pair_changes.keys())
        if changed:
            ui.refresh_lists()
        ui.root.request_redraw()
        ui.root.neuron_network.draw()
        ui.root.status_bar.refresh()

    def _clear_undo_highlight(self):
        for lb in (self.unselected_list, self.selected_list):
            for idx in range(lb.count()):
                it = lb.item(idx)
                if it is not None:
                    it.setBackground(QBrush())
                    it.setForeground(QBrush())

    def _toggle_pairs_group(self, pairs, group_name):
        ui = self.ui
        all_in = all(p2 in ui.groups.pairs_in_group(group_name, s2)
                     for p in pairs for s2, p2 in (ui.pair_sess_rt(p),))
        for p in pairs:
            s2, p2 = ui.pair_sess_rt(p)
            (ui.groups.discard_from_group if all_in else ui.groups.add_to_group)(
                group_name, s2, p2)
        unsel_frac, sel_frac = self._save_scroll_positions()
        self.refresh_lists()
        self._restore_scroll_positions(unsel_frac, sel_frac)


class SpikePairsPanel(QWidget):
    """Qt spike pairs tab content. Replaces SpikePairsPanel."""

    def __init__(self, notebook: QTabWidget, data: SelectionData, ui: 'CCGReviewUI'):
        super().__init__()
        self.data    = data
        self.ui     = ui
        self._notebook = notebook
        self._spike_pairs: list = []
        self._selected_idx: int = -1
        self._spike_pairs_tab_index: int = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._spike_pairs_count = LabelVar(QLabel(""))
        layout.addWidget(self._spike_pairs_count._lbl)

        self._spike_pairs_listbox = QListWidget()
        _mf = QFont(); _mf.setStyleHint(QFont.StyleHint.Monospace); _mf.setPointSize(9)
        self._spike_pairs_listbox.setFont(_mf)
        self._spike_pairs_listbox.itemClicked.connect(self._on_spike_pair_click)
        layout.addWidget(self._spike_pairs_listbox, stretch=1)

        self._raster_ok = False
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._raster_fig = Figure(figsize=(4, 2.5), tight_layout=True)
            self._raster_canvas = FigureCanvasQTAgg(self._raster_fig)
            layout.addWidget(self._raster_canvas, stretch=2)
            self._raster_ok = True
        except Exception:
            pass

        # _tab compat alias
        self._tab = self

    def populate(self, spike_pairs: list):
        self._spike_pairs   = spike_pairs
        self._selected_idx  = -1
        self._spike_pairs_listbox.clear()
        if not spike_pairs:
            self._spike_pairs_count.set("0 spike pairs in this bin")
            if self._raster_ok:
                self._raster_fig.clear()
                self._raster_canvas.draw()
            return
        for i, (rt, tt) in enumerate(spike_pairs):
            lag_ms = (tt - rt) * 1000.0
            self._spike_pairs_listbox.addItem(
                f"{i+1:>5}  ref {rt:10.4f}  tgt {tt:10.4f}  lag {lag_ms:+6.2f}ms")
        self._spike_pairs_count.set(f"{len(spike_pairs)} spike pairs")
        self._spike_pairs_listbox.setCurrentRow(0)
        self._selected_idx = 0
        self._draw_raster(0)

    def clear(self):
        self._notebook.setTabEnabled(self._spike_pairs_tab_index, False)
        self._notebook.setCurrentIndex(0)
        self._spike_pairs   = []
        self._selected_idx  = -1
        self._spike_pairs_count.set("")
        if self._raster_ok:
            self._raster_fig.clear()
            self._raster_canvas.draw()

    def activate(self):
        self._notebook.setTabEnabled(self._spike_pairs_tab_index, True)
        self._notebook.setCurrentIndex(self._spike_pairs_tab_index)

    def _on_spike_pair_click(self, item: QListWidgetItem):
        idx = self._spike_pairs_listbox.row(item)
        if idx >= len(self._spike_pairs):
            return
        self._selected_idx = idx
        self._draw_raster(idx)

    def _draw_raster(self, idx: int):
        """Separate ref/tgt spike raster plots (not on the CCG plot)."""
        if not self._raster_ok or idx >= len(self._spike_pairs):
            return
        ui = self.ui
        inds = ui.current_pair_inds
        if inds is None:
            return
        ref, tgt = int(inds[0]), int(inds[1])
        neurons = ui.neurons
        if (neurons is None or ref >= len(neurons.spiketrains)
                or tgt >= len(neurons.spiketrains)):
            return
        ref_t, tgt_t = self._spike_pairs[idx]
        ref_lbl = f'Ref {ui.root.neuron_network._shank_label(ref)}'
        tgt_lbl = f'Tgt {ui.root.neuron_network._shank_label(tgt)}'
        from neuropy.analyses.spike_attribution import RASTER_WINDOW_SEC
        from neuropy.plotting.ccg import plot_spike_attribution_raster
        plot_spike_attribution_raster(
            self._raster_fig,
            np.asarray(neurons.spiketrains[ref]),
            np.asarray(neurons.spiketrains[tgt]),
            ref_t, tgt_t, window=RASTER_WINDOW_SEC,
            ref_label=ref_lbl, tgt_label=tgt_lbl, pair_idx=idx)
        self._raster_canvas.draw()


class PairSelectionPanelContainer(QWidget, Autosave):
    """Top-level tab container for pair selection.
    Also manages autosaving and session-level persistence."""

    autosave_interval_minutes = 30
    autosave_retain_days      = 7

    def __init__(self, parent: QWidget, data: SelectionData,
                 ui: 'CCGReviewUI', ui_state_cache: dict):
        super().__init__(parent)
        self.ui = ui
        self.data = data
        self._closing = False
        ui.closing.connect(self._on_closing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        tab_widget = QTabWidget(self)
        layout.addWidget(tab_widget)

        # Tab 0: Pair Selection
        pair_tab = QWidget()
        pair_layout = QVBoxLayout(pair_tab)
        pair_layout.setContentsMargins(0, 0, 0, 0)
        self.pair_selection = PairSelectionPanel(pair_tab, data, ui, ui_state_cache)
        pair_layout.addWidget(self.pair_selection)
        tab_widget.addTab(pair_tab, "Pair Selection")

        # Back-wire notebook reference so PairSelectionPanel can expose it
        self.pair_selection.current_left_tab_content = tab_widget

        # Tab 1: Spike Pairs (initially disabled)
        self.spike_pairs = SpikePairsPanel(tab_widget, data, ui)
        tab_widget.addTab(self.spike_pairs, "Spike Pairs")
        tab_widget.setTabEnabled(1, False)

        self.widget = self   # compat: CCGReviewUI packs .widget

    def _on_closing(self):
        self._closing = True

    def autosave_active(self) -> bool:
        return self.ui.ccg_ptr is not None and not self._closing

    def autosave_base_dir(self) -> str:
        return self.ui.sd.save_dir

    def autosave_label(self) -> str:
        return str(self.ui.key.session)

    def schedule_autosave(self) -> None:
        t = _QTimer(self)
        t.setInterval(self.autosave_interval_minutes * 60 * 1000)
        t.timeout.connect(lambda: self.ui.sd.save() if self.autosave_active() else None)
        t.start()

    def _autoload_session_latest(self, restore_groups: bool = False):
        ui = self.ui
        if restore_groups:
            groups_path = ui.groups.save_path()
            ui.groups.load()
        sd = ui.sd.get_selection_by_session(ui.key)
        latest_path = sd.save_path()
        if not latest_path or not os.path.isfile(latest_path + '.json'):
            return
        self._load_selection_from_file(latest_path + '.json',
                                       restore_groups=restore_groups,
                                       _skip_redraw=True)

    def _load_selection_from_file(self, path: str, restore_groups: bool = True,
                                   _skip_redraw: bool = False):
        ui = self.ui
        sd = ui.sd.get_selection_by_session(ui.key)
        p = path[:-5] if path.endswith('.json') else path
        sd.load(p)

        if restore_groups:
            sess = ui.current_session_str
            for gname in list(ui.groups):
                for pair in list(ui.groups.pairs_in_group(gname, sess)):
                    ui.groups.discard_from_group(gname, sess, pair)
            all_tags = {k: v for bucket in sd.selections.values() for k, v in bucket.tags.items()}
            added = []
            for bucket in sd.selections.values():
                for (ref, tgt), entry in bucket.tags.items():
                    for gname in entry.get('groups', []):
                        if isinstance(gname, str) and gname:
                            ui.groups.add_to_group(gname, sess, (ref, tgt))
                            added.append((gname, ref, tgt))
            ui.groups.changed.emit()

        bucket = ui.active_selections
        universe = ui._pairs_for_ptr_key(ui.key)
        selected = bucket.selected & universe
        deleted  = bucket.deleted  & universe

        missing = selected - set(map(tuple, ui.all_inds))
        if missing:
            if restore_groups and not _skip_redraw:
                action = MissingPairsDialog.show(ui, missing)
                if action == 'cancel':
                    return
                if action == 'partial':
                    selected &= set(map(tuple, ui.all_inds))
                else:
                    selected |= missing
                    universe = ui._pairs_for_ptr_key(ui.key) | missing

        _before = {p: self.pair_selection._pair_state(p) for p in universe}
        bucket.reset(universe, selected=selected, deleted=deleted)

        if not _skip_redraw:
            _after = {p: self.pair_selection._pair_state(p) for p in universe}
            self.pair_selection.push_undo(SelectionCommand(
                {p: (_before[p], _after[p]) for p in universe if _before[p] != _after[p]}, []))
            ui._post_load_refresh()

    def _list_selection_versions(self) -> list:
        """Return (name, path, saved_at, is_valid, is_history) tuples for the current
        session's saved versions (<session>__<name>.json) plus any .history/ backups."""
        ui = self.ui
        save_dir = ui.sd.save_dir
        prefix = f"{ui.key.session}__"

        def _entry(fpath: str, fname: str, is_hist: bool) -> tuple:
            name = fname[len(prefix):-len('.json')]
            try:
                with open(fpath, encoding='utf-8') as f:
                    json.load(f)
                saved_at = datetime.datetime.fromtimestamp(
                    os.path.getmtime(fpath)).isoformat()
                return (name, fpath, saved_at, True, is_hist)
            except (OSError, ValueError):
                return (name, fpath, '', False, is_hist)

        versions = []
        if os.path.isdir(save_dir):
            for fname in sorted(os.listdir(save_dir)):
                if fname.startswith(prefix) and fname.endswith('.json'):
                    versions.append(_entry(os.path.join(save_dir, fname), fname, False))
        hdir = os.path.join(save_dir, '.history')
        if os.path.isdir(hdir):
            hist = [_entry(os.path.join(hdir, f), f, True)
                    for f in os.listdir(hdir)
                    if f.startswith(prefix) and f.endswith('.json')]
            hist.sort(key=lambda e: e[2], reverse=True)
            versions.extend(hist)
        return versions

    def _do_save(self, name: str = ''):
        ui = self.ui
        ui.sd.save()   # persist latest (<session>.json + dataset + groups export)
        if name:
            vpath = os.path.join(ui.sd.save_dir, f"{ui.key.session}__{name}")
            ui.sd.get_selection_by_session(ui.key).save(path=vpath)
        type_keys = ui.available_type_keys(ui.key.nd())
        total = sum(
            ui.sd.get_selection_by_session(tk_).selections[tk_].selected.__len__()
            for tk_ in type_keys
        )
        groups_msg = f"\nGroups exported ({len(ui.groups.registry)} groups)." if ui.groups else ""
        vmsg = f"\nVersion: {name}" if name else ""
        QMessageBox.information(
            None, "Saved",
            f"Saved {total} pairs across {len(type_keys)} types.{groups_msg}{vmsg}")

    def _save_all_state(self, **_) -> None:
        ui = self.ui
        ui.sd.save()
        ui._settings_mgr.save_ui_state()

    def _current_filter_state(self) -> dict:
        ui = self.ui
        toggles = ui.time_slider._ts_legend_toggles
        return {
            'theme': ui.time_slider._current_theme,
            'labels': {str(lbl): bool(v.get()) for lbl, v in toggles.items()},
        }
