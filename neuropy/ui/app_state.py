"""Central navigation state for CCG Review UI.

AppState owns all cross-panel shared state and is the single
write path for any value that multiple panels read.  Panels subscribe
to its signals; they never write each other's state directly.

DisplayConfig is intentionally absent — per-panel display preferences
live inside each Qt widget.

Hierarchy
---------
CCGDataset (cd)       <- raw arrays, pointers, neurons  [data owner]
    |
AppState       <- UI cursor + selection state    [state owner]
    |
Qt Widgets            <- subscribe to signals, own display prefs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import re
import collections
import numpy as np
import glob as _glob
from pathlib import Path as _Path

from pyqtgraph.Qt.QtCore import QObject, Signal, QTimer

import json

from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.utils import _compact_json_str
from neuropy.ui.pair_selection_panel import SelectionData, SelectionDataset
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.analyses.ms_connectivity import CCGDataset

_ALL_SEGS = "All"

# Sentinel for the virtual "All sessions" entry in the session list
_ALL_SESSION_MARKER = object()


class NavField:
    """Storage + read access for one AppState scalar."""

    def __init__(self, name: str, *, coerce=None, compare: bool = True):
        self._name = name
        self._private = f"_{name}"
        self._sanitize = coerce
        self._compare_value = compare

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._private)

    def __set__(self, obj, value):
        raise AttributeError(
            f"use set_{self._name}() on AppState, not assignment")

    def set(self, obj, value, signal):
        """Called from AppState.set_* methods."""
        if self._sanitize is not None:
            value = self._sanitize(value)
        if self._compare_value:
            if value == getattr(obj, self._private):
                return
        setattr(obj, self._private, value)
        signal.emit(value)



class AppState(QObject):
    """Cross-panel shared state.  Single write path via set_* methods.

    All fields that used to be scattered across CCGReviewUI, MiscManager,
    and MultiSessionManager now live here.

    Signals fire only when the value actually changes.
    """

    key_changed              = Signal(object)
    pair_changed             = Signal(int)
    segment_changed          = Signal(str)
    resolution_changed       = Signal(str)   # "lo" | "hi" | "lo_hi"
    session_mode_changed     = Signal(bool)
    cross_session_handles_changed      = Signal(object)
    selection_changed        = Signal()
    norms_changed            = Signal(object)
    sig_threshold_changed            = Signal(float)
    scale_mode_changed       = Signal(object)
    stacked_segments_changed = Signal(object)
    cs_params_changed        = Signal(str, str)
    cs_overlay_changed       = Signal(bool)
    custom_segs_changed      = Signal()
    themes_changed           = Signal(dict)   # {attr: Epoch} for current session
    closing                  = Signal()

    key                  = NavField("key")
    current_pair_idx     = NavField("current_pair_idx", coerce=int)
    current_segment      = NavField("current_segment")
    resolution           = NavField("resolution")   # "lo" | "hi" | "lo_hi"
    session_any_mode     = NavField("session_any_mode", coerce=bool)
    cross_session_handles = NavField("cross_session_handles", compare=False)
    active_sig_threshold         = NavField("active_sig_threshold", coerce=float)
    active_norms         = NavField("active_norms")
    same_scale_mode      = NavField("same_scale_mode")
    stacked_segments     = NavField("stacked_segments")
    baseline_method      = NavField("baseline_method")
    cs_metric            = NavField("cs_metric")
    cs_overlay_active    = NavField("cs_overlay_active", coerce=bool)
    custom_seg_index     = NavField("custom_seg_index")

    def __init__(self, cd: 'CCGDataset', key: Key):
        super().__init__()
        self.cd = cd
        self._key = key
        self._current_pair_idx = 0
        self._current_segment = _ALL_SEGS
        self._resolution = "lo"
        self._session_any_mode = False
        self._cross_session_handles = []
        self.sd = SelectionDataset()
        self._active_sig_threshold = cd.conf.alpha
        self._active_norms = set()
        self._same_scale_mode = None
        self._stacked_segments = []
        self._baseline_method = 'conv'
        self._cs_metric = 'STG'
        self._cs_overlay_active = False
        self._custom_seg_index = {}
        self.root = None  # set by CCGReviewUI after construction
        self.together_pairs: list = []
        self.max_together_pairs: int = 5
        self.bookmarked_pairs: set = set()
        self.any_expanded_group_tags: set = set()

    @property
    def ccg_ptr(self):
        return self.cd.ptr.get(self.key)

    @property
    def ccg_data(self):
        nd_key = self.key.nd()
        res = 'highres' if self.resolution in ("hi", "lo_hi") else 'lowres'
        return self.cd.ccg_for(nd_key, res)

    @property
    def neurons(self):
        return self.cd.nd.data[self.key.nd()]

    @property
    def n_segments(self) -> int:
        return self.cd.n_segments_for(self.key)

    @property
    def all_inds(self) -> np.ndarray:
        """All (ref, tgt) pairs currently visible, as Nx2 int array."""
        if self.session_any_mode:
            hl = self.cross_session_handles
            if not hl:
                return np.empty((0, 2), dtype=int)
            return np.array([[int(r), int(t)] for _, r, t in hl], dtype=int)

        b = self.active_selections
        combined = b.unselected | b.selected | b.deleted
        base = sorted(p for p in combined if p[0] != p[1])
        return np.array(base, dtype=int) if base else np.empty((0, 2), dtype=int)

    @property
    def current_pair_inds(self) -> np.ndarray | None:
        inds = self.all_inds
        if self.current_pair_idx < len(inds):
            return inds[self.current_pair_idx]
        return None

    @property
    def current_pair(self):
        """Current pair in selection representation:
        (ref, tgt) normally, or the (ckey, ref, tgt) handle in all-session mode."""
        idx = self.current_pair_idx
        if self.session_any_mode:
            hl = self.cross_session_handles
            return hl[idx] if 0 <= idx < len(hl) else None
        inds = self.all_inds
        return tuple(int(x) for x in inds[idx]) if 0 <= idx < len(inds) else None

    @property
    def current_session_str(self) -> str:
        return str(self.key.session)

    @property
    def sel_data(self):
        return self.sd.get_selection_by_session(self.key)

    @property
    def active_selections(self):
        return self.sel_data.selections[self.key]

    @property
    def groups(self):
        return self.sd.groups

    def refresh_lists(self):
        self.root.pairs_view.pair_selection.refresh_lists()

    def set_cd(self, cd: 'CCGDataset'):
        """Replace the CCGDataset (on project switch). No signal."""
        self.cd = cd

    def _build_themes(self, key: Key) -> dict:
        nd = getattr(self.cd, 'nd', None)
        if nd is None:
            print("[themes] nd is None")
            return {}
        sessions = getattr(nd, '_sessions', None) or []
        if not isinstance(sessions, (list, tuple)):
            sessions = [sessions]
        sess_name = str(getattr(key, 'session', ''))
        session = next(
            (s for s in sessions
             if (nd._short_session_name(s) if hasattr(nd, '_short_session_name') else str(s)) == sess_name),
            None)
        if session is None:
            known = [nd._short_session_name(s) for s in sessions]
            print(f"[themes] session {sess_name!r} not found; known={known}")
            return {}
        from neuropy.core.epoch import Epoch as _Epoch
        all_epoch_attrs = {attr: obj for attr in vars(session)
                           if isinstance((obj := getattr(session, attr, None)), _Epoch)}
        print(f"[themes] session={sess_name!r}  all Epoch attrs: "
              + ", ".join(f"{a}(n={o.n_epochs}, labels={list(o.labels)[:5]})"
                          for a, o in all_epoch_attrs.items()))
        result = {a: o for a, o in all_epoch_attrs.items() if o.n_epochs > 0}
        print(f"[themes] → emitting {list(result)}")
        return result

    def set_key(self, key: Key):
        type(self).key.set(self, key, self.key_changed)
        self.themes_changed.emit(self._build_themes(key))

    def set_current_pair(self, idx: int, *, source=None):
        type(self).current_pair_idx.set(self, idx, self.pair_changed)

    def set_current_segment(self, name: str, *, source=None):
        type(self).current_segment.set(self, name, self.segment_changed)

    def set_resolution(self, value: str):
        assert value in ("lo", "hi", "lo_hi"), f"invalid resolution {value!r}"
        type(self).resolution.set(self, value, self.resolution_changed)

    def set_session_any_mode(self, value: bool):
        type(self).session_any_mode.set(self, value, self.session_mode_changed)

    def set_cross_session_handles(self, handles: list):
        type(self).cross_session_handles.set(self, handles, self.cross_session_handles_changed)

    def notify_selection_changed(self):
        self.selection_changed.emit()

    def reset_selection_for_project(self, cd, save_dir: str = '') -> None:
        """Replace selection state with a fresh dataset when switching projects."""
        self.sd = SelectionDataset(save_dir=save_dir)
        self.notify_selection_changed()

    def set_active_sig_threshold(self, alpha: float):
        type(self).active_sig_threshold.set(self, alpha, self.sig_threshold_changed)

    def set_active_norms(self, norms: set):
        type(self).active_norms.set(self, set(norms), self.norms_changed)

    def set_same_scale_mode(self, mode: str | None):
        type(self).same_scale_mode.set(self, mode, self.scale_mode_changed)

    def set_cs_overlay(self, active: bool):
        type(self).cs_overlay_active.set(self, bool(active), self.cs_overlay_changed)

    def set_cs_params(self, baseline_method: str, cs_metric: str):
        if (baseline_method == self.baseline_method
                and cs_metric == self.cs_metric):
            return
        self._baseline_method = baseline_method
        self._cs_metric = cs_metric
        self.cs_params_changed.emit(baseline_method, cs_metric)

    def refresh_custom_seg_index(self):
        """Scan cd's save path for .npz custom CCG files; update registry in-place."""
        base = self.cd.conf.save_path(suffix='customseg')
        new = {}
        for p in sorted(_glob.glob(base + '_*.npz')):
            stem = _Path(p).stem
            prefix = _Path(base).stem + '_'
            if stem.startswith(prefix):
                new[stem[len(prefix):]] = p
        if new != self.custom_seg_index:
            self._custom_seg_index = new
            self.custom_segs_changed.emit()

    def toggle_stacked_segment(self, seg_idx: int):
        segs = list(self.stacked_segments)
        if seg_idx in segs:
            segs.remove(seg_idx)
        else:
            segs.append(seg_idx)
        self._stacked_segments = segs
        self.stacked_segments_changed.emit(segs)

    def clear_stacked_segments(self):
        if not self.stacked_segments:
            return
        self._stacked_segments = []
        self.stacked_segments_changed.emit([])

    def is_significant(self, ref: int, tgt: int, seg: int) -> bool:
        n = self.n_segments
        if seg > n:
            return False
        if seg == n:
            return any(self.is_significant(ref, tgt, s) for s in range(n))

        j = self.cd._jitter.get(self.key) if self.cd._jitter else None
        if j is not None:
            inds = j.ccg_ptr.inds
            if j.ccg_ptr.stored_by_segment:
                mask = (inds[:, 0] == seg) & (inds[:, -2] == ref) & (inds[:, -1] == tgt)
            else:
                mask = (inds[:, -2] == ref) & (inds[:, -1] == tgt)
            if mask.any():
                return bool(j.j_sig[mask].any())
            return False

        data = self.cd.ccg.get(self.key.nd())
        if data is None:
            return False
        lb = data.conf.min_lag_bin
        ub = data.conf.max_lag_bin
        pc = data.pval_corrected
        if pc is not None and seg < pc.shape[0] and ref < pc.shape[1] and tgt < pc.shape[2]:
            return bool(pc[seg, ref, tgt, lb:ub].min() <= self.active_sig_threshold)
        sig = data.significant
        if sig is not None:
            if sig.ndim == 4 and seg < sig.shape[0] and ref < sig.shape[1] and tgt < sig.shape[2]:
                return bool(sig[seg, ref, tgt].any())
            if sig.ndim == 3 and ref < sig.shape[0] and tgt < sig.shape[1]:
                return bool(sig[ref, tgt].any())
        return False

    def _custom_names(self) -> list:
        return list(self.custom_seg_index.keys())

    def seg_idx(self, name: str) -> int:
        if name is None or name == _ALL_SEGS:
            return self.n_segments
        names = self.cd.segment_names_for(self.key)
        if name in names:
            return names.index(name)
        custom = self._custom_names()
        if name in custom:
            return self.n_segments + 1 + custom.index(name)
        return self.n_segments

    def seg_name(self, idx: int) -> str:
        if idx is None or idx == self.n_segments:
            return _ALL_SEGS
        n = self.n_segments
        if 0 <= idx < n:
            names = self.cd.segment_names_for(self.key)
            if idx < len(names):
                return names[idx]
            return _ALL_SEGS
        custom = self._custom_names()
        ci = idx - n - 1
        return custom[ci] if 0 <= ci < len(custom) else _ALL_SEGS

    def all_segment_names(self) -> list[str]:
        names = list(self.cd.segment_names_for(self.key))
        names.append(_ALL_SEGS)
        names.extend(self._custom_names())
        return names

    def clamp_segment(self):
        if self.current_segment not in self.all_segment_names():
            self.set_current_segment(_ALL_SEGS)

    def ensure_groups_loaded_for(self, sessions: list[str]) -> None:
        """Load SelectionData from disk for unvisited sessions; sync group tags into groups.

        Source of truth for group membership is SelectionData.tags, not groups._fwd.
        sd.sessions tracks which sessions have been loaded.
        """
        save_dir = self.sd.save_dir
        if not save_dir:
            return
        loaded = {str(k.session) for k in self.sd.sessions}
        for sess in sessions:
            if sess in loaded:
                continue
            path = os.path.join(save_dir, sess)
            if not os.path.exists(path + '.json'):
                continue
            sel = SelectionData()
            sel.load(path)
            if sel.selections:
                nd_key = next(iter(sel.selections)).nd()
                self.sd.sessions[nd_key] = sel
            for bucket in sel.selections.values():
                for (ref, tgt), entry in bucket.tags.items():
                    for gname in (entry.get('groups') or []):
                        if isinstance(gname, str) and gname:
                            self.groups.add_to_group(gname, sess, (ref, tgt))

    def available_type_keys(self, nd_key) -> list:
        if nd_key is _ALL_SESSION_MARKER:
            return self.available_type_keys_any()
        nd_session = nd_key.session
        return [k for k in self.cd.ptr.keys() if k.nd().session == nd_session]

    def available_type_keys_any(self) -> list:
        by_lbl: dict = {}
        for k in sorted(self.cd.ptr.keys(),
                        key=lambda k: (str(k.session), k.type_label())):
            by_lbl.setdefault(k.type_label(), k)
        return sorted(by_lbl.values(), key=self._conn_type_sort_key)

    def type_key_for_nd(self, nd_key) -> 'Key | None':
        cur_lbl = self.key.type_label()
        matches = [k for k in self.cd.ptr.keys()
                   if k.nd() == nd_key and k.type_label() == cur_lbl]
        return matches[0] if matches else None

    @staticmethod
    def _conn_type_sort_key(key: Key) -> tuple:
        ep = 0 if key.excitability == 'E' else 1 if key.excitability == 'I' else 2

        def _rank(cell):
            s = str(cell).lower()
            if s in ('pyr', 'pyramidal'): return (0, s)
            if s in ('inter', 'int'):     return (1, s)
            return (2, s)

        ct = key.conn_type
        ct_key = (_rank(ct[0]), _rank(ct[1])) if ct else ((99, ''), (99, ''))
        return (ep, ct_key, str(key.epoch or ''))

    def real_nd_keys(self) -> list:
        """Unique session nd-keys from cd, excluding _ALL_SESSION_MARKER."""
        seen, keys = set(), []
        for k in list(self.cd.ccg.keys()) + list(self.cd.ptr.keys()):
            nd = k.nd()
            sess = str(nd.session)
            if sess not in seen:
                seen.add(sess)
                keys.append(nd)
        return [k for k in keys if k is not _ALL_SESSION_MARKER]

    def nd_key_for_session(self, sess_str: str) -> 'Key | None':
        for nk in self.real_nd_keys():
            if str(nk.session) == sess_str:
                return nk
        return None

    def iter_type_keys(self) -> list:
        """All ptr keys matching current type label, one per session."""
        lbl = self.key.type_label()
        return [tk for nk in self.real_nd_keys()
                for tk in (self.type_key_for_nd(nk),)
                if tk is not None and tk.type_label() == lbl]

    def all_nd_keys(self) -> list:
        """Unique nd-keys including _ALL_SESSION_MARKER if present."""
        seen, keys = set(), []
        for k in list(self.cd.ccg.keys()) + list(self.cd.ptr.keys()):
            nd = k.nd()
            sess = str(nd.session)
            if sess not in seen:
                seen.add(sess)
                keys.append(nd)
        return keys

    @staticmethod
    def session_label(nd_key) -> str:
        if nd_key is _ALL_SESSION_MARKER:
            return 'All'
        return str(nd_key.session)

    @staticmethod
    def sanitize_sess_slug(sess: str) -> str:
        s = re.sub(r'[^\w.\-]+', '_', str(sess))[:48]
        return s or 'sess'

    def _pairs_by_conn_type(self, session_str: str, pairs) -> dict:
        pair_ct = {}
        for key in self.cd.ptr:
            if str(key.session) != session_str and str(key.nd().session) != session_str:
                continue
            ct_label = Key.format_conn_type(key.conn_type)
            pt = self.cd.ptr[key]
            for ref, tgt in map(tuple, pt.inds2):
                pair_ct[(ref, tgt)] = ct_label
        result = collections.OrderedDict()
        for pair in sorted(pairs):
            result.setdefault(pair_ct.get(tuple(pair), 'unknown'), []).append(pair)
        return result

    @property
    def _sess_mgr(self):
        return self

    def after(self, ms: int, fn):
        QTimer.singleShot(int(ms), fn)

    def after_cancel(self, _id):
        pass

    def get_pair_index(self, inds) -> int:
        if self.session_any_mode:
            hl = self.cross_session_handles
            if len(inds) >= 3:
                sess = str(inds[0].session)
                r, t = int(inds[1]), int(inds[2])
                for i, (k2, r2, t2) in enumerate(hl):
                    if str(k2.session) == sess and r2 == r and t2 == t:
                        return i
                return 0
            r, t = int(inds[0]), int(inds[1])
            for i, (k2, r2, t2) in enumerate(hl):
                if r2 == r and t2 == t:
                    return i
            return 0
        for i, pair in enumerate(self.all_inds):
            if tuple(pair) == tuple(inds):
                return i
        return 0

    def pair_sess_rt(self, inds) -> tuple:
        """(session_str, (ref, tgt)) for group/tag lookups."""
        if self.session_any_mode:
            return str(inds[0].session), (int(inds[1]), int(inds[2]))
        return str(self.key.session), (int(inds[0]), int(inds[1]))

    def toggle_together(self, pairs):
        for p in pairs:
            pt = tuple(p)
            if pt in [tuple(x) for x in self.together_pairs]:
                self.together_pairs = [x for x in self.together_pairs if tuple(x) != pt]
            else:
                self.together_pairs.append(pt)
        self.root.request_redraw()

    def clear_together(self):
        self.together_pairs.clear()
        self.root.request_redraw()

    @staticmethod
    def _all_inds_set_for_ptr(ptr) -> set:
        base = ptr.inds2
        return set(map(tuple, base[base[:, 0] != base[:, 1]]))

    def _pairs_for_ptr_key(self, key) -> set:
        ptr = self.cd.ptr.get(key)
        if ptr is None:
            return set()
        raw = ptr.inds2
        return set(map(tuple, raw[raw[:, 0] != raw[:, 1]]))

    def _pairs_from_tags_for_key(self, key) -> set:
        sess = str(key.session)
        pairs: set = {tuple(p) for p in self.sel_data.selections[key].tags if p[0] != p[1]}
        for gname in self.groups.defined_groups:
            pairs |= self.groups.pairs_in_group(gname, sess)
        if not pairs:
            return pairs
        ct_lbl = Key.format_conn_type(key.conn_type)
        return set(map(tuple, self._pairs_by_conn_type(sess, pairs).get(ct_lbl, [])))

    def apply_sel_for_key(self, key=None):
        """Initialize or reconcile SelectionData bucket for one conn-type Key."""
        key = key or self.key
        all_pairs = self._pairs_for_ptr_key(key)
        full_universe = all_pairs
        b = self.sd.get_selection_by_session(key).selections[key]
        is_new = not (b.selected or b.unselected or b.deleted)
        if not is_new and full_universe:
            sel = b.selected & full_universe
            deleted = b.deleted & full_universe
            new_pairs = full_universe - sel - deleted - b.unselected
            if new_pairs and key.is_excitatory():
                sel |= new_pairs
            b.reset(full_universe, selected=sel, deleted=deleted)
            return
        if all_pairs:
            sel = set(all_pairs) if key.is_excitatory() else set()
            b.reset(all_pairs, selected=sel, deleted=b.deleted & all_pairs)
            return
        tagged = self._pairs_from_tags_for_key(key)
        if tagged:
            b.reset(tagged, selected=tagged if key.is_excitatory() else set(),
                   deleted=b.deleted & tagged)
            return
        b.reset(set())
