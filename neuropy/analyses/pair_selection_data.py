"""UI-independent pair-selection data classes.

These hold per-session pair selections and group tags and are pure
data/serialization logic (no Qt/pyqtgraph). They live here so that
notebooks and scripts can load selection JSON without importing the UI
(which pulls in pyqtgraph.Qt and can crash headless kernels).

The Qt UI re-exports these from ``neuropy.ui.pair_selection_panel`` for
backward compatibility.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections import defaultdict as _defaultdict

from neuropy.analyses.utils import (
    JsonSavable, Autosave, BiIndex, _to_json, is_special_group, _SPECIAL_PREFIX,
    ADMITTED_PREFIX, is_shape_label, is_admitted_group, NOTHING,
)
from neuropy.utils.data_storage_util import atomic_write_json
from neuropy.analyses.neurons_dataset import Key


class _SelectionData(JsonSavable):
    """Selections for one conn-type within a session."""

    def __init__(self):
        JsonSavable.__init__(self)
        self.selected:   set = set()
        self.unselected: set = set()
        self.deleted:    set = set()
        self.tags:       dict = {}   # {(ref,tgt): {groups,notes,tags}}
        self.complete:   bool = False  # reviewed exhaustively → untagged pairs are negatives
        self._dirty:     bool = False

    def __setattr__(self, name: str, value) -> None:
        # assigning state must dirty the bucket, or a regular save skips it
        if name in ('selected', 'unselected', 'deleted', 'tags', 'complete') \
                and getattr(self, name, object()) != value:
            self._dirty = True
        object.__setattr__(self, name, value)

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
        self.complete   = bool(state.get('complete', False))
        self._dirty     = False   # just read from disk: nothing to write back

    def set_pair_state(self, pair: tuple, state: str):
        pair = tuple(pair)
        self._dirty = True
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

    @property
    def all_pairs(self) -> set:
        return self.selected | self.unselected | self.deleted


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

    def __setstate__(self, state: dict) -> None:
        sel = state.get('selections')
        if isinstance(sel, list):
            # Legacy broken saves wrote Key-keyed dicts as [[key_str, bucket], ...].
            state = dict(state)
            state['selections'] = {
                str(item[0]): item[1]
                for item in sel
                if isinstance(item, (list, tuple)) and len(item) == 2
            }
        JsonSavable.__setstate__(self, state)

    def serialize(self) -> dict:
        """Write selections with string conn-type keys (load-compatible dict form)."""
        out = {}
        for k, v in self._public_state().items():
            if k == 'selections':
                out[k] = {str(key): _to_json(bucket) for key, bucket in v.items()}
            else:
                out[k] = _to_json(v)
        return out

    @property
    def dirty(self) -> bool:
        return any(b._dirty for b in self.selections.values())

    def save(self, path: str = None, **_):
        if self._nd_key is not None:
            self.session = str(self._nd_key.session)
        self.saved_at = datetime.datetime.now().isoformat()
        JsonSavable.save(self, path=path, **_)
        for b in self.selections.values():
            b._dirty = False

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
        if pairs is None or vals is None:
            return {}
        return {SelectionData.as_pair_key(p): float(v) for p, v in zip(pairs, vals)}


class Group(JsonSavable):
    """One group's metadata (name + optional hotkey + notes)."""

    def __init__(self, name: str = '', hotkey: str = '', notes: str = '',
                 ui_color: str = ''):
        JsonSavable.__init__(self)
        self.name     = name
        self.hotkey   = hotkey
        self.notes    = notes
        self.ui_color = ui_color   # '' = auto colour from name; special groups stay ''

    @property
    def display_name(self) -> str:
        """Name without the internal ``__special_`` prefix, for display."""
        return (self.name[len(_SPECIAL_PREFIX):] if is_special_group(self.name)
                else self.name)

    @property
    def display_color(self) -> str:
        """'#rrggbb' tint for tag chips: set colour, else one seeded by the name.

        Special groups get '' (transparent) — they are not colour-tagged.
        """
        if is_special_group(self.name):
            return ''
        if self.ui_color:
            return self.ui_color
        h = int(hashlib.md5(self.name.encode()).hexdigest(), 16)
        return f'#{0x808080 | (h & 0x7f7f7f):06x}'   # light, name-stable


class GroupDataset(JsonSavable, BiIndex):
    """Group tags (pair ↔ group multimap) + per-group metadata registry.

    Pure data/serialization. The Qt UI subclass ``Groups`` adds the ``changed``
    signal and the queries that need a live UI (pair validity, nd-keys, hotkeys).
    """

    _custom_types = {'registry': Group}

    def __init__(self, save_dir: str = ''):
        JsonSavable.__init__(self, ignored_attrs=['ui', 'dirty'])
        BiIndex.__init__(self)
        self.registry: dict[str, Group] = {}
        self.dirty = False
        self.ui = None
        self._save_dir: str = save_dir

    def bind(self, ui) -> None:
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
        """Tags a user can apply — machine markers are not offered as choices."""
        return sorted(g for g in self.defined_groups
                      if g != NOTHING
                      and not is_special_group(g) and not is_admitted_group(g))

    def special_groups(self) -> list[str]:
        return sorted(g for g in self.defined_groups if is_special_group(g))

    def group_for_hotkey(self, key_str: str) -> str | None:
        """Group a hotkey tags, or ``None`` when the key is unassigned."""
        return next((g.name for g in self.registry.values()
                     if g.hotkey and g.hotkey == key_str
                     and not is_special_group(g.name)), None)

    def get_group_metadata(self, name: str) -> Group:
        if name not in self.registry:
            self.registry[name] = Group(name=name)
        return self.registry[name]

    def save_path(self, **_) -> str | None:
        d = self._save_dir or (groups_dir(self.ui.cd) if self.ui is not None else '')
        return os.path.join(d, 'groups') if d else None

    def serialize(self) -> dict:
        state = {'registry': _to_json(self.registry)}
        if getattr(self, 'saved_at', None):
            state['saved_at'] = self.saved_at
        return state

    def save(self, path: str = None, **_):
        self.saved_at = datetime.datetime.now().isoformat()
        JsonSavable.save(self, path=path, **_)

    @staticmethod
    def member(key: Key) -> tuple:
        """A Key as its stored member tuple; tgt is dropped when the Key has none."""
        ids = (key.ref,) if key.tgt is None else (key.ref, key.tgt)
        return (str(key.session), *(int(i) for i in ids))

    def add_member(self, gname: str, key: Key) -> None:
        self.add(gname, self.member(key))
        self.get_group_metadata(gname)
        self.dirty = True

    def discard_member(self, gname: str, key: Key) -> None:
        self.discard(gname, self.member(key))
        self.dirty = True

    def members_in_group(self, gname: str, sess: str) -> set:
        """Id tuples tagged by *gname* in one session, without the session itself."""
        return {tuple(rest) for s, *rest in self.forward(gname) if s == sess}

    def members_in_groups(self, gnames, sess: str) -> set:
        """Union over several groups; a member tagged twice counts once."""
        return set().union(*(self.members_in_group(g, sess) for g in gnames)) \
            if gnames else set()

    def keys_in_group(self, gname: str, sess: str) -> set:
        """Keys tagged by *gname* in one session — what a view indexes by."""
        return {Key(session=sess, ref=m[0], tgt=m[1] if len(m) > 1 else None)
                for m in self.members_in_group(gname, sess)}

    def groups_for_member(self, key: Key) -> set:
        return self.inverse(self.member(key))

    def chips_for_member(self, key: Key) -> list:
        """(display_name, display_color) per group tagging this member, sorted by name.

        Admitted markers are bookkeeping — which model proposed the pair — so they
        are kept on the member but never shown as a tag.
        """
        metas = [self.get_group_metadata(g)
                 for g in sorted(self.groups_for_member(key))
                 if not is_admitted_group(g)]
        return [(m.display_name, m.display_color) for m in metas]

    def add_to_group(self, gname: str, sess: str, pair: tuple) -> None:
        self.add_member(gname, Key.pair(sess, pair[0], pair[1]))

    def discard_from_group(self, gname: str, sess: str, pair: tuple) -> None:
        self.discard_member(gname, Key.pair(sess, pair[0], pair[1]))

    def pairs_in_group(self, gname: str, sess: str) -> set:
        return self.members_in_group(gname, sess)

    def pairs_in_groups(self, gnames, sess: str) -> set:
        return self.members_in_groups(gnames, sess)

    def groups_for_pair(self, sess: str, ref: int, tgt: int) -> set:
        return self.groups_for_member(Key.pair(sess, ref, tgt))

    def chips_for_pair(self, sess: str, ref: int, tgt: int) -> list:
        return self.chips_for_member(Key.pair(sess, ref, tgt))

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

    @staticmethod
    def valid_hotkey(key_str: str) -> str:
        """A hotkey normalised to lowercase; empty clears it.

        Raises ValueError on anything but a digit 1-9/0 or a single letter.
        """
        key_str = key_str.strip().lower()
        digits = [str(i) for i in range(1, 10)] + ['0']
        if key_str and key_str not in digits and not (
                len(key_str) == 1 and key_str.isalpha()):
            raise ValueError("Enter a digit 1–9/0 or a single letter a–z.")
        return key_str

    def set_group_hotkey(self, name: str, key_str: str) -> None:
        for grp in self.registry.values():
            if grp.hotkey == key_str and grp.name != name:
                grp.hotkey = ''
        self.get_group_metadata(name).hotkey = key_str

    def header_names(self) -> list[str]:
        def _gname_sort_key(n):
            try:
                return (0, int(n), '')
            except (ValueError, TypeError):
                return (1, 0, n)
        return sorted(self.registry, key=_gname_sort_key)


def groups_dir(cd) -> str:
    """Where groups.json lives: beside the projects, not inside one.

    Tags name CCG shapes, which mean the same thing in every project, so one
    registry serves all of them — the same reasoning that puts the trained
    classifiers in ``data_root``.
    """
    return str(cd.data_root)


class NeuronGroups(GroupDataset):
    """Neuron groups, stored per project: a neuron id only means something in its session.

    Membership is written here, unlike pair groups, whose index is rebuilt from the
    per-session selection files.
    """

    def __init__(self, cd=None):
        GroupDataset.__init__(self)
        if cd is not None:
            self._save_dir = str(cd.nd.neuron_dir(cd.save_path))

    def save_path(self, **_) -> str | None:
        return os.path.join(self._save_dir, 'neuron_groups') if self._save_dir else None

    def serialize(self) -> dict:
        state = GroupDataset.serialize(self)
        state['members'] = {g: sorted(list(m) for m in self.forward(g))
                            for g in self.defined_groups if self.forward(g)}
        return state

    def __setstate__(self, state: dict) -> None:
        members = state.pop('members', {})
        GroupDataset.__setstate__(self, state)
        for gname, entries in members.items():
            for member in entries:
                self.add(gname, (str(member[0]), *(int(i) for i in member[1:])))


class SelectionDataset(JsonSavable, Autosave):
    """Project-level owner of groups + per-session SelectionData.

    The Qt UI passes a ``Groups`` instance (subclass of ``GroupDataset``) via
    ``groups_factory``; scripts/notebooks default to the pure ``GroupDataset``.
    """

    def __init__(self, cd, groups_factory=GroupDataset):
        JsonSavable.__init__(self, ignored_attrs=['cd', 'sessions'])
        self.cd = cd
        save_dir = cd.selections_dir
        self.groups = groups_factory()
        self.groups._save_dir = groups_dir(cd)
        # The registry is shared and project-independent, so it is read here
        # rather than waiting on a project's own saved selections — a project
        # with none of its own still knows every tag.
        if os.path.isfile(self.groups.save_path() + '.json'):
            self.groups.load(self.groups.save_path())
        self.sessions: dict[Key, SelectionData] = {}
        self.save_dir = save_dir
        self._indexed: set = set()

    @property
    def dirty(self) -> bool:
        return self.groups.dirty or any(sd.dirty for sd in self.sessions.values())

    def save(self, path: str = None, **kw):
        """Each session owns its file; the roster indexes whichever ones exist."""
        for sd in self.sessions.values():
            if sd.dirty:
                sd.save()
        self.groups.save()
        self.groups.dirty = False
        self.save_roster()

    def saved_sessions(self) -> list:
        """Plan sessions that have a selection file on disk — the roster, scanned."""
        return [s for s in self.cd.sessions
                if os.path.isfile(os.path.join(self.save_dir, s) + '.json')]

    def roster_path(self) -> str:
        return os.path.join(self.save_dir, 'selection_dataset.json')

    def save_roster(self) -> None:
        """Rewrite the roster index only when the set of session files changed."""
        roster = {'groups': self.groups.save_path() + '.json',
                  'save_dir': self.save_dir,
                  'sessions': self.saved_sessions()}
        path = self.roster_path()
        if os.path.isfile(path):
            with open(path) as fh:
                if json.load(fh).get('sessions') == roster['sessions']:
                    return
        atomic_write_json(path, roster)

    def load_sessions(self) -> None:
        """Read every session file the plan claims; the roster is never the authority."""
        self.sessions = {}
        for sess in self.saved_sessions():
            sd = SelectionData(save_dir=self.save_dir)
            sd.load(os.path.join(self.save_dir, sess))
            if not sd.selections:
                continue
            sd._nd_key = next(iter(sd.selections)).nd()
            self.sessions[sd._nd_key] = sd

    def get_selection_by_session(self, key: Key) -> SelectionData:
        nd = key.nd()
        sd = self.sessions.get(nd)
        if sd is None:
            sd = SelectionData(save_dir=self.save_dir, nd_key=nd)
            self.sessions[nd] = sd
        return sd

    def has_shape_tag(self, sess: str, pair: tuple) -> bool:
        """True while the pair carries a tag naming a shape, not a marker or a note."""
        return any(is_shape_label(g)
                   for g in self.groups.groups_for_pair(sess, pair[0], pair[1]))

    def reconcile(self, key: Key, pair_set: set) -> None:
        """Restrict saved selection to universe."""
        b = self.get_selection_by_session(key).selections[key]
        if pair_set:
            b.reset(pair_set, selected=b.selected & pair_set,
                    deleted=b.deleted & pair_set)
            return
        universe = b.selected | b.unselected | b.deleted
        if universe or b.tags:
            b.reset(universe, selected=b.selected & universe,
                    deleted=b.deleted & universe)
            return
        b.reset(set())

    def ensure_groups_loaded_for(self, sessions: list[str]) -> None:
        """Load SelectionData from disk for unvisited sessions; sync group tags into groups.

        Source of truth for group membership is SelectionData.tags, not groups._fwd.
        self.sessions tracks which sessions have been loaded.
        """
        if not self.save_dir:
            return
        loaded = {str(k.session): v for k, v in self.sessions.items()}
        for sess in sessions:
            if sess in self._indexed:
                continue
            sel = loaded.get(sess)
            if sel is None:
                path = os.path.join(self.save_dir, sess)
                if not os.path.exists(path + '.json'):
                    continue
                sel = SelectionData(save_dir=self.save_dir)
                sel.load(path)
                if not sel.selections:
                    continue
                # Loading has to leave the bucket writable: without its nd-key a
                # SelectionData has no save_path and silently never persists.
                sel._nd_key = next(iter(sel.selections)).nd()
                self.sessions[sel._nd_key] = sel
            for bucket in sel.selections.values():
                for (ref, tgt), entry in bucket.tags.items():
                    for gname in (entry.get('groups') or []):
                        if isinstance(gname, str) and gname:
                            self.groups.add_to_group(gname, sess, (ref, tgt))
            self._indexed.add(sess)   # only once synced: a partial pass must retry
