"""Central navigation state for CCG Review UI.

AppState owns all cross-panel shared state and is the single
write path for any value that multiple panels read.  Panels subscribe
to its signals; they never write each other's state directly.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import os
import re
import collections
from collections import defaultdict as _defaultdict
import numpy as np
from pyqtgraph.Qt.QtCore import QObject, Signal, QTimer
import json
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.utils import _compact_json_str
from neuropy.ui.ui_common import is_special_group
from neuropy.ui.pair_selection_panel import SelectionDataset
from neuropy.ui.utils import Tunable
from neuropy.utils.data_storage_util import atomic_write_json

if TYPE_CHECKING:
    from neuropy.analyses.ms_connectivity import CCGDataset

# Whole-session view == the permanent dim0[0]='all' segment (no virtual sum view any more).
_ALL_SEGS = "all"
ALL_PAIRS = '(all pairs)'   # stats-panel "every valid pair" group choice

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
    groups_rewired           = Signal()   # sd.groups replaced → reconnect groups.changed
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
    stacked_transposed   = NavField("stacked_transposed", coerce=bool)
    baseline_method      = NavField("baseline_method")
    cs_metric            = NavField("cs_metric")
    cs_overlay_active    = NavField("cs_overlay_active", coerce=bool)

    max_together_pairs = Tunable(5)
    max_ccg_queue = Tunable(50, on_change=lambda nav, v: setattr(
        nav.root.custom_mgr.worker._runner, '_max_queue', v))
    max_jitter_queue = Tunable(50, on_change=lambda nav, v: setattr(
        nav.root.jitter_mgr.jitter_worker._runner, '_max_queue', v))
    max_jitter_cache = Tunable(500, on_change=lambda nav, v:
                               nav.root.jitter_mgr.jitter_worker._cache.resize(v))

    def __init__(self, cd: 'CCGDataset', key: Key):
        super().__init__()
        self.cd = cd
        self._key = key
        self._current_pair_idx = 0
        self._current_segment = _ALL_SEGS
        self._resolution = "lo"
        self._session_any_mode = False
        self._cross_session_handles = []
        self.set_sd(SelectionDataset(cd))
        self._active_sig_threshold = cd.conf.alpha
        self._active_norms = set()
        self._same_scale_mode = None
        self._stacked_segments = []
        self._stacked_transposed = False
        self._baseline_method = 'conv'
        self._cs_metric = 'STG'
        self._cs_overlay_active = False
        self.root = None  # set by CCGReviewUI after construction
        self.together_pairs: list = []
        self.bookmarked_pairs: set = set()
        self.any_expanded_group_tags: set = set()

    @property
    def ccg_ptr(self):
        return self.cd.ptr.get(self.key.ptr())

    @property
    def ccg_data(self):
        return self.cd.ccg_for(self.get_key_with_resolution())

    @property
    def neurons(self):
        """Neurons of the selected pair's session — the pair's own in all-session mode."""
        return self.cd.nd.neurons_for(self.get_complete_key())

    @property
    def n_segments(self) -> int:
        return self.cd.n_segments(self.get_key_with_resolution())

    @property
    def all_pairs_np(self) -> np.ndarray:
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
    def all_pairs_set(self) -> set:
        """All visible (ref, tgt) pairs as a set of int tuples (membership tests)."""
        return {(int(r), int(t)) for r, t in self.all_pairs_np}

    @property
    def current_pair_inds(self) -> np.ndarray | None:
        inds = self.all_pairs_np
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
        inds = self.all_pairs_np
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

    def set_sd(self, sd) -> None:
        """Install a SelectionDataset, bound and rewired to this nav.

        Binding belongs here rather than at the call site: its groups reach back
        through ``groups.ui`` (hotkey tagging, save paths), so a dataset that is
        installed but unbound is broken, and a project switch builds a new one.
        ``groups_rewired`` lets panels re-subscribe: ``groups.changed`` belongs to
        the Groups instance, so connections made to the old one are dropped with it.
        """
        self.sd = sd
        sd.groups.bind(self)
        self.groups_rewired.emit()

    def set_cd(self, cd: 'CCGDataset'):
        """Replace the CCGDataset (on project switch). No signal."""
        self.cd = cd

    def set_key(self, key: Key):
        if key.conn_type is None:   # nd keys carry no type; every reader of nav.key needs one
            excitability, conn_type = self.cd.conf.conn_types_labeled[0]
            key = key.change(excitability=excitability, conn_type=conn_type)
        type(self).key.set(self, key, self.key_changed)
        self.themes_changed.emit(self.cd.nd.get_themes(key))

    def switch_key(self, new_key: Key, load_selection=None):
        """Single-session transition to *new_key*: set key, reconcile the bucket, keep the
        segment if still valid, reset pair/norms on a real switch.

        load_selection: optional callable run right after the key is set (before reconciling)
        so a session-change can populate the new bucket from disk first. Ordering matters —
        set_key must precede the load, apply_sel_for_key must follow it.
        """
        prev_key, prev_seg = self.key, self.current_segment
        self.set_key(new_key)
        if load_selection is not None:
            load_selection()
        self.apply_sel_for_key(self.key)   # set_key types an nd-key; only that one finds a ptr
        if prev_seg in self.available_segments():
            self.set_current_segment(prev_seg)
        self.clamp_segment()
        if new_key != prev_key:                 # fast path preserves current pair + norms
            self.set_current_pair(0)
            self.set_active_norms(set())

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

    def reset_selection_for_project(self, cd) -> None:
        """Replace selection state with a fresh dataset when switching projects."""
        self.set_sd(SelectionDataset(cd))
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

    def toggle_stacked_transposed(self) -> None:
        self._stacked_transposed = not self._stacked_transposed
        self.stacked_segments_changed.emit(self.stacked_segments)

    def toggle_stacked_segments(self, labels) -> None:
        """Add *labels* to the stack, or remove them when all are already stacked."""
        segs = list(self.stacked_segments)
        if all(l in segs for l in labels):
            segs = [s for s in segs if s not in labels]
        else:
            segs += [l for l in labels if l not in segs]
        self._stacked_segments = segs
        self.stacked_segments_changed.emit(segs)

    def clear_stacked_segments(self):
        if not self.stacked_segments:
            return
        self._stacked_segments = []
        self.stacked_segments_changed.emit([])

    def is_significant(self, ref: int, tgt: int, seg: int) -> bool:
        data = self.ccg_data
        if data is None:
            return False
        lb = data.conf.min_lag_bin
        ub = data.conf.max_lag_bin
        pc = data.pval_corrected
        if pc is not None and seg < pc.shape[0] and ref < pc.shape[1] and tgt < pc.shape[2]:
            return bool(pc[seg, ref, tgt, lb:ub].min() <= self.active_sig_threshold)
        return False

    @property
    def data_resolution(self) -> str:
        """CCGData resolution the view mode reads ('lo_hi' renders both, data is highres)."""
        return 'highres' if self.resolution in ("hi", "lo_hi") else 'lowres'

    def get_key_with_resolution(self) -> Key:
        """CCG store key: session + the resolution the view mode requests."""
        return Key(session=self.key.session, resolution=self.data_resolution)

    def get_complete_key(self) -> Key:
        """Complete Key for the current view: the selected pair's session + resolution,
        plus the current segment label and (ref, tgt) as coordinates on that array."""
        pair = self.current_pair
        inds = self.current_pair_inds
        ref, tgt = (int(inds[0]), int(inds[1])) if inds is not None else (None, None)
        sess = pair[0] if self.session_any_mode and pair is not None else self.key
        return Key(session=sess.session, resolution=self.data_resolution,
                   segment=self.current_segment, ref=ref, tgt=tgt)

    def segment_index(self, label: str) -> int:
        return self.cd.segment_index(self.get_key_with_resolution(), label)

    def segment_name(self, idx: int) -> str:
        return self.cd.segment_name(self.get_key_with_resolution(), idx)

    def segment_names(self) -> list[str]:
        return self.cd.segment_names(self.get_key_with_resolution())

    def available_segments(self) -> list[str]:
        """Disk segment labels for the current session."""
        return self.cd.available_segments(self.key)

    def all_available_segments(self) -> list[str]:
        """Project-wide disk segment labels (union across sessions) — cross-session pickers."""
        return self.cd.available_segments()

    def attach_segment(self, src, seg_data) -> None:
        """Stack a pre-computed window (single-segment CCGData) onto the current session's
        array as a new dim0 segment, then notify listeners. Main-thread only."""
        self.cd.attach_segment(self.key.nd(), src, seg_data)
        self.custom_segs_changed.emit()

    def drop_segment(self, label: str) -> None:
        """Remove an appended window segment from the current session (never 'full')."""
        if label == _ALL_SEGS:
            return
        self.cd.drop_segment([self.key.nd().change(segment=label)])
        if self.current_segment == label:
            self.set_current_segment(_ALL_SEGS)
        self.custom_segs_changed.emit()

    def available_sessions(self) -> list[str]:
        """Session id strings for every real nd-key (ALL-session marker excluded)."""
        return [str(k.session) for k in self.real_nd_keys()]

    def available_resolutions(self) -> list[str]:
        """Resolutions available (live or saved) — robust to lazy loading."""
        return self.cd.available_resolutions()

    def available_conn_types(self) -> list[str]:
        return self.cd.conf.conn_type_labels

    def available_groups(self) -> list[str]:
        gr = self.groups
        return [ALL_PAIRS] + gr.groups + gr.special_groups()

    def pairs_for_group(self, group_name: str, ptr_key) -> set:
        """Valid (significant) pairs of a group for a ptr key; ALL_PAIRS = every valid pair."""
        valid = self.cd.ptr[ptr_key].pair_set
        if group_name == ALL_PAIRS:
            return valid
        return self.groups.pairs_in_group(group_name, ptr_key.session) & valid

    def clamp_segment(self):
        if self.current_segment not in self.available_segments():
            self.set_current_segment(_ALL_SEGS)

    def ensure_groups_loaded_for(self, sessions: list[str]) -> None:
        self.sd.ensure_groups_loaded_for(sessions)

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
        return (ep, ct_key)

    def real_nd_keys(self) -> list:
        """Unique session nd-keys for the dataset, excluding _ALL_SESSION_MARKER.

        cd.nd owns session identity: a session exists iff it has neurons. Pointers left on disk
        under an old naming convention are stale data, not sessions — they have no Neurons.
        """
        seen, keys = set(), []
        for k in self.cd.nd.session_keys():
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
        for k in list(self.cd.ptr.keys()):
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
            for ref, tgt in pt.pair_set:
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
        for i, pair in enumerate(self.all_pairs_np):
            if tuple(pair) == tuple(inds):
                return i
        return 0

    _COMBO_SORT_KEY = staticmethod(
        lambda combo: (1, []) if not combo else (0, list(combo)))

    def _pair_group_combo(self, inds) -> tuple:
        """Sorted tuple of non-special group names this pair belongs to."""
        k = self.key_for_pair(inds)
        return tuple(sorted(
            g for g in self.groups
            if not is_special_group(g)
            and (k.ref, k.tgt) in self.groups.pairs_in_group(g, k.session)))

    def selected_sections(self, sort_mode: str) -> list:
        """Ordered (header|None, [inds]) sections of the selected pairs for a sort mode.

        sort_mode in {'mean','minp','group','tag','plain'}. Pure data ordering; the
        panel only renders the returned sections.
        """
        selected = self.active_selections.selected
        if sort_mode in ('mean', 'minp'):
            ccg_d = self.ccg_data
            if ccg_d is not None:
                seg = self.segment_index(self.current_segment)
                metric = ccg_d.mean_ccg if sort_mode == 'mean' else ccg_d.min_pval
                order = sorted(selected,
                               key=lambda p: metric(int(p[0]), int(p[1]), seg),
                               reverse=(sort_mode == 'mean'))
            else:
                order = sorted(selected)
            return [(None, order)]

        if sort_mode == 'group':
            buckets = _defaultdict(list)
            for inds in sorted(selected):
                buckets[self._pair_group_combo(inds)].append(inds)
            sections = []
            for combo in sorted(buckets, key=self._COMBO_SORT_KEY):
                hdr = ', '.join(combo) if combo else '(untagged)'
                sections.append((hdr, buckets[combo]))
            return sections

        if sort_mode == 'tag':
            non_internal = [g for g in self.groups.defined_groups
                            if not is_special_group(g)]
            tag_buckets = _defaultdict(list)
            untagged = []
            for inds in sorted(selected):
                k = self.key_for_pair(inds)
                tags = [g for g in non_internal
                        if (k.ref, k.tgt) in self.groups.pairs_in_group(g, k.session)]
                if tags:
                    for t in tags:
                        tag_buckets[t].append(inds)
                else:
                    untagged.append(inds)
            sections = [(t, tag_buckets[t]) for t in sorted(tag_buckets)]
            if untagged:
                sections.append(('(untagged)', untagged))
            return sections

        return [(None, sorted(selected))]

    def key_for_pair(self, inds) -> 'Key':
        # TODO band-aid: inds[0] is either a keyed session object or an already-resolved str; unify shapes later.
        if self.session_any_mode:
            sess = inds[0] if isinstance(inds[0], str) else str(inds[0].session)
            return Key.pair(sess, inds[1], inds[2])
        return Key.pair(str(self.key.session), inds[0], inds[1])

    def toggle_together(self, pairs):
        for p in pairs:
            pt = tuple(p)
            if pt in [tuple(x) for x in self.together_pairs]:
                self.together_pairs = [x for x in self.together_pairs if tuple(x) != pt]
            else:
                self.together_pairs.append(pt)
        self.root.mainview.request_render()

    def clear_together(self):
        self.together_pairs.clear()
        self.root.mainview.request_render()

    def _pairs_for_ptr_key(self, key) -> set:
        ptr = self.cd.ptr.get(key.ptr())
        return ptr.pair_set if ptr is not None else set()

    def apply_sel_for_key(self, key=None):
        key = key or self.key
        self.sd.reconcile(key, self._pairs_for_ptr_key(key))
