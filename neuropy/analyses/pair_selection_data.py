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
import re
import numpy as np
from collections import defaultdict as _defaultdict
from dataclasses import dataclass, replace

from neuropy.analyses.utils import (
    JsonSavable, Autosave, BiIndex, _to_json, is_special_group, _SPECIAL_PREFIX,
    is_derived_group, group_display_name,
    ADMITTED_PREFIX, is_shape_label, is_admitted_group, NOTHING, is_transient_group, moved,
)
from neuropy.utils.data_storage_util import atomic_write_json
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.neuron_tags import STABLE_TAG
from neuropy.analyses.ccg_transforms import Rules


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
        self.promote_tagged()

    def promote_tagged(self) -> None:
        """A pair carrying a group belongs in Selected; a deleted one stays deleted."""
        for p, t in self.tags.items():
            if p not in self.selected and p not in self.deleted \
                    and any(not is_special_group(g) for g in t.get('groups') or ()):
                self.set_pair_state(p, 'sel')

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
        for b in self.selections.values():
            b.promote_tagged()
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


def _dir_token(v) -> str:
    """Path-safe token: conn-type labels carry arrows and spaces."""
    return re.sub(r'[^A-Za-z0-9_-]', '-', str(v))


class RuleResults:
    """Swept significance-rule arrays for one (session, conn type, segment), as ``[ref, tgt]`` npys.

    One unit dir per segment: a new segment is a new dir and a dropped one is an rmtree,
    so nothing reshapes when the segment set changes."""

    ON = 'on'

    def __init__(self, save_dir: str, nd_key: Key, conn_type: str, segment: str):
        self._dir = os.path.join(
            save_dir, 'rule_results',
            f"{_dir_token(nd_key.session)}.{_dir_token(conn_type)}.{_dir_token(segment)}")
        self.arrays: dict = {}

    @property
    def stored(self) -> bool:
        return os.path.isdir(self._dir)

    def load(self) -> 'RuleResults':
        self.arrays = {f[:-4]: np.load(os.path.join(self._dir, f), mmap_mode='r')
                       for f in os.listdir(self._dir)} if self.stored else {}
        return self

    def save(self) -> None:
        """Write the arrays just set and drop files no array names; kept arrays are memmaps of their own files."""
        os.makedirs(self._dir, exist_ok=True)
        for name, arr in self.arrays.items():
            if not isinstance(arr, np.memmap):
                np.save(os.path.join(self._dir, name + '.npy'), arr)
        for f in os.listdir(self._dir):
            if f[:-4] not in self.arrays:
                os.remove(os.path.join(self._dir, f))

    @staticmethod
    def _set_of(array_name: str) -> str:
        return array_name.rsplit('.', 1)[0].split(':', 1)[0]

    def set_results(self, results: dict, keep: list) -> None:
        """Overwrite with *results*, keeping only the sets in *keep* from before; ``recombine`` sets on."""
        self.arrays = {k: v for k, v in self.arrays.items() if self._set_of(k) in keep}
        for name, (value, passed) in results.items():
            self.arrays[name + '.value'] = np.asarray(value, dtype=float)
            self.arrays[name + '.passed'] = np.asarray(passed, dtype=bool)

    @staticmethod
    def combine(arrays: dict, sets: tuple, mode: str, shape: tuple) -> np.ndarray:
        """``[ref, tgt]`` on from the stored verdicts of *sets* (all when empty): AND, or OR among the sets judging a pair; unjudged pairs stay on."""
        names = [k[:-7] for k in arrays if k.endswith('.passed') and ':' not in k and (not sets or k[:-7] in sets)]
        if not names:
            return np.ones(shape, dtype=bool)
        judged = np.array([~np.isnan(np.asarray(arrays[n + '.value'])) for n in names])
        passed = np.array([np.asarray(arrays[n + '.passed']) for n in names])
        if mode == 'or':
            return (judged & passed).any(axis=0) | ~judged.any(axis=0)
        return (passed | ~judged).all(axis=0)

    def recombine(self, sets: tuple, mode: str, shape: tuple, overrides: dict = None) -> None:
        """on = *sets* combined by *mode*, hand-set entries winning."""
        on = self.combine({k: v for k, v in self.arrays.items() if k != self.ON}, sets, mode, shape)
        for (ref, tgt), state in (overrides or {}).items():
            if ref < on.shape[0] and tgt < on.shape[1]:
                on[ref, tgt] = state
        self.arrays[self.ON] = on

    def pair(self, ref: int, tgt: int) -> dict:
        """``{rule: [value, passed], 'on': bool}`` for one pair, the shape the table reads."""
        if not self.arrays:
            return {}
        out = {name[:-6]: [None if np.isnan(arr[ref, tgt]) else float(arr[ref, tgt]),   # as the live sweep reports it
                           bool(self.arrays[name[:-6] + '.passed'][ref, tgt])]
               for name, arr in self.arrays.items() if name.endswith('.value')}
        out[self.ON] = bool(self.arrays[self.ON][ref, tgt])
        return out


class DerivedRule(JsonSavable):
    """One condition on a pair: any of *values* matches; a neuron rule tests ref, tgt, either or both."""

    PAIR_TAG, PAIR_LOCATION, PAIR_CONN, PAIR_SIGNIFICANCE, NEURON_TAG, NEURON_GROUP = (
        'pair tag', 'pair location', 'pair conn type', 'pair significance', 'neuron tag', 'neuron group')
    KINDS = (PAIR_TAG, PAIR_LOCATION, PAIR_CONN, PAIR_SIGNIFICANCE, NEURON_TAG, NEURON_GROUP)
    VERDICTS = ('on', 'off')   # a significance rule's values; its name is the rule set
    NAMED = {PAIR_LOCATION: ('same channel', 'same shank')}   # kinds whose name picks what is compared
    SIDES = ('ref', 'tgt', 'either', 'both')
    EXCITABILITIES = ('E', 'I')

    def __init__(self, kind: str = PAIR_TAG, name: str = '', values: list = None,
                 side: str = 'either'):
        JsonSavable.__init__(self)
        self.kind = kind
        self.name = name   # the neuron tag, or the location compared; pair tags name theirs in values
        self.values = list(values or [])
        self.side = side

    @staticmethod
    def split(kind: str) -> list:
        """['pair', 'tag']: the subject, then what of it is tested."""
        return kind.split(' ', 1)

    def __str__(self) -> str:
        values = ' or '.join(group_display_name(v) for v in self.values)
        subject = 'pair' if self.kind.startswith('pair') else f"{self.side} neuron"
        what = self.name or self.split(self.kind)[1]
        return f"{subject} {what} is {values}"
        return state

    def set_renamed(self, old: str, new: str) -> 'DerivedRule':
        """This rule pointed at rule set *new* when it named *old*."""
        if self.kind != self.PAIR_SIGNIFICANCE or self.name != old:
            return self
        return DerivedRule(self.kind, new, self.values, self.side)


@dataclass(frozen=True)
class RuleSet:
    """Significance rules for the pairs in its scope; an empty scope field does not narrow."""
    name: str = 'default'
    tests: Rules = Rules()
    sessions: tuple = ()
    conn_types: tuple = ()
    rules: tuple = ()   # DerivedRule, every one must hold
    initial: bool = False   # run by the initial screen that builds the pointers, not by Assign tags
    color: str = ''   # the pass tag's pill; '' = uncoloured
    shown: bool = True   # whether the pass tag shows on pair rows

    def covers(self, key: Key) -> bool:
        return ((not self.sessions or str(key.session) in self.sessions)
                and (not self.conn_types or key.type_label() in self.conn_types))

    def serialize(self) -> dict:
        return {'name': self.name, 'initial': self.initial, 'tag': {'color': self.color, 'shown': self.shown},
                'scope': {'sessions': list(self.sessions), 'conn_types': list(self.conn_types),
                          'pair_rules': [r.serialize() for r in self.rules]},
                'rules': self.tests.serialize()}

    @classmethod
    def from_dict(cls, d: dict) -> 'RuleSet':
        if 'scope' not in d:   # the interim layout: scope fields flat, tests under 'tests'
            return cls(d['name'], Rules.from_dict(d['tests']), tuple(d['sessions']),
                       tuple(d['conn_types']), tuple(DerivedRule(**r) for r in d['rules']))
        scope = d['scope']
        return cls(d['name'], Rules.from_dict(d['rules']), tuple(scope['sessions']),
                   tuple(scope['conn_types']), tuple(DerivedRule(**r) for r in scope['pair_rules']),
                   d.get('initial', False), **d.get('tag', {}))


@dataclass(frozen=True)
class RuleBook:
    """Rule sets in run order: a pair stays on only while it passes every set covering it."""
    sets: tuple = ()
    gray_boxes: tuple = ('rules', 'network focus', 'neuron selection')   # a pair stays clear only when every checked box keeps it
    gray_rules: tuple = ()   # DerivedRule: the 'rules' box keeps the pairs meeting every one
    on_sets: tuple = ()   # later sets whose verdicts make a pair on; empty = every stored set
    on_mode: str = 'and'   # 'and' | 'or' across on_sets

    GRAY_BOXES = ('rules', 'network focus', 'neuron selection')

    def __post_init__(self):
        object.__setattr__(self, 'sets', self.initial_sets() + self.later_sets())   # initial sets always run first

    def initial_sets(self) -> tuple:
        return tuple(r for r in self.sets if r.initial)

    def later_sets(self) -> tuple:
        return tuple(r for r in self.sets if not r.initial)

    def set(self, name: str) -> RuleSet:
        return next(r for r in self.sets if r.name == name)

    def with_set(self, rule_set: RuleSet) -> 'RuleBook':
        """*rule_set* in place of its namesake, or appended last."""
        names = [r.name for r in self.sets]
        if rule_set.name not in names:
            return replace(self, sets=self.sets + (rule_set,))
        return replace(self, sets=tuple(rule_set if r.name == rule_set.name else r for r in self.sets))

    def moved(self, name: str, step: int) -> 'RuleBook':
        """*name* moved by *step*; re-sectioning keeps it within initial or later sets."""
        return replace(self, sets=tuple(moved(self.sets, [r.name for r in self.sets].index(name), step)))

    def renamed(self, old: str, new: str) -> 'RuleBook':
        """Set *old* called *new*, along with every scope or gray-out rule naming it."""
        return replace(self, gray_rules=tuple(r.set_renamed(old, new) for r in self.gray_rules),
                       on_sets=tuple(new if n == old else n for n in self.on_sets),
                       sets=tuple(replace(s, name=new if s.name == old else s.name,
                                          rules=tuple(r.set_renamed(old, new) for r in s.rules))
                                  for s in self.sets))

    def without(self, name: str) -> 'RuleBook':
        return replace(self, sets=tuple(r for r in self.sets if r.name != name),
                       on_sets=tuple(n for n in self.on_sets if n != name))

    def conf_fields(self) -> dict:
        """The CCGConfig fields this book is stored in."""
        return {'rule_sets': [r.serialize() for r in self.sets],
                'gray_out': {'boxes': list(self.gray_boxes),
                             'pair_rules': [r.serialize() for r in self.gray_rules]},
                'pair_on': {'sets': list(self.on_sets), 'mode': self.on_mode}}

    @classmethod
    def from_conf(cls, conf) -> 'RuleBook':
        gray, on = conf.gray_out or {}, conf.pair_on
        return cls(tuple(RuleSet.from_dict(r) for r in conf.rule_sets),
                   tuple(gray.get('boxes', cls.gray_boxes)),
                   tuple(DerivedRule(**r) for r in gray.get('pair_rules', ())),
                   tuple(on.get('sets', ())), on.get('mode', cls.on_mode))

    @classmethod
    def from_legacy(cls, state: dict) -> 'RuleBook':
        """A pre-rule-set config: alphabetical scoped groups then the default, or one ``aux_tests`` dict."""
        if 'aux_tests' in state:
            d = state['aux_tests'] or {}
            if 'groups' not in d:
                return cls((RuleSet(tests=Rules.from_dict(d)),))
            return cls(tuple(RuleSet.from_dict(g) for g in d['groups']), tuple(d.get('gray_boxes', cls.gray_boxes)),
                       tuple(DerivedRule(**r) for r in d.get('gray_rules', ())))
        gray = state.get('gray_out') or {}
        scoped = sorted((RuleSet.from_dict(g) for g in state.get('rule_groups') or ()), key=lambda r: r.name)
        return cls((*scoped, RuleSet(tests=Rules.from_dict(state.get('default_significance_rules') or []))),
                   tuple(gray.get('boxes', cls.gray_boxes)), tuple(DerivedRule(**r) for r in gray.get('pair_rules', ())))


class Group(JsonSavable):
    """One group's metadata (name + optional hotkey + notes); a derived group also holds its rules."""

    def __init__(self, name: str = '', hotkey: str = '', notes: str = '',
                 ui_color: str = '', rules: list = None):
        JsonSavable.__init__(self)
        self.name     = name
        self.hotkey   = hotkey
        self.notes    = notes
        self.ui_color = ui_color   # '' = auto colour from name; special groups stay ''
        self.rules: list[DerivedRule] = list(rules or [])

    def __setstate__(self, state: dict) -> None:
        JsonSavable.__setstate__(self, state)
        self.rules = [DerivedRule(**r) for r in state.get('rules', [])]

    def serialize(self) -> dict:
        state = JsonSavable.serialize(self)
        if not self.rules:   # only derived groups carry rules; others keep their saved shape
            state.pop('rules')
        return state

    @property
    def display_name(self) -> str:
        """Name without its internal prefix, for display."""
        return group_display_name(self.name)

    @property
    def display_color(self) -> str:
        """'#rrggbb' tint for tag chips: set colour, else one seeded by the name.

        Special groups get '' (transparent) — they are not colour-tagged.
        """
        if is_special_group(self.name) or is_derived_group(self.name):
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
        JsonSavable.__init__(self, ignored_attrs=['ui', 'dirty', 'derived'])
        BiIndex.__init__(self)
        self.registry: dict[str, Group] = {}
        self.dirty = False
        self.ui = None
        self.derived: DerivedGroups = None   # set by the SelectionDataset that owns this
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
                      and not is_special_group(g) and not is_admitted_group(g)
                      and not is_derived_group(g))

    def special_groups(self) -> list[str]:
        return sorted(g for g in self.defined_groups if is_special_group(g))

    def derived_groups(self) -> list[str]:
        return sorted(g for g in self.defined_groups if is_derived_group(g))

    def pickable_groups(self) -> list[str]:
        """Every group a scope can draw pairs from, in picker section order."""
        return self.groups + self.special_groups() + self.derived_groups()

    def forward(self, a) -> set:
        if is_derived_group(a):
            return {(sess, *p) for sess in self.derived.sessions()
                    for p in self.derived.members(a, sess)}
        return BiIndex.forward(self, a)

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
        state = {'registry': _to_json({n: g for n, g in self.registry.items() if not is_transient_group(n)})}
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
        if is_derived_group(gname):
            return self.derived.members(gname, sess)
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


class SegmentGroups(GroupDataset):
    """Group tags carrying a segment: a pair can be tagged in ``maze`` but not in ``post``.

    Its own registry and file, so session-wide tags keep their segment-free member keys."""

    ON, OFF = 'on', 'off'

    @staticmethod
    def member(key: Key) -> tuple:
        ids = (key.ref,) if key.tgt is None else (key.ref, key.tgt)
        return (str(key.session), str(key.segment), *(int(i) for i in ids))

    def tag_pair(self, gname: str, sess: str, segment: str, pair: tuple, on: bool) -> None:
        key = Key(session=str(sess), segment=str(segment),
                  ref=int(pair[0]), tgt=int(pair[1]))
        (self.add_member if on else self.discard_member)(gname, key)

    def toggle_pair(self, gname: str, sess: str, segment: str, pair: tuple) -> bool:
        """Flip *pair*'s membership of *gname* in *segment*; returns whether it is now tagged."""
        want = gname not in self.tags_for_pair(sess, segment, pair)
        self.tag_pair(gname, sess, segment, pair, want)
        return want

    def tags_for_pair(self, sess: str, segment: str, pair: tuple) -> set:
        return self.groups_for_member(Key(session=str(sess), segment=str(segment),
                                          ref=int(pair[0]), tgt=int(pair[1])))

    def pairs_tagged(self, gname: str, sess: str, segment: str) -> set:
        """Pairs tagged by *gname* in one (session, segment)."""
        return {(r, t) for s, seg, r, t in self.forward(gname)
                if s == str(sess) and seg == str(segment)}

    def pair_off(self, sess: str, segment: str, pair: tuple) -> bool | None:
        """Manual override: True on, False off, None when the tests decide."""
        tags = self.tags_for_pair(sess, segment, pair)
        if self.OFF in tags:
            return False
        return True if self.ON in tags else None

    def segment_tags(self) -> list:
        """Tag names offered here; on/off always exist, user tags follow."""
        rest = sorted(g for g in self.registry if g not in (self.ON, self.OFF))
        return [self.ON, self.OFF] + rest

    def segments_tagged(self, sess: str) -> list:
        return sorted({seg for _g, members in self._fwd.items()
                       for s, seg, *_ in members if s == str(sess)})

    def save_path(self, **_) -> str | None:
        d = self._save_dir or (groups_dir(self.ui.cd) if self.ui is not None else '')
        return os.path.join(d, 'segment_groups') if d else None

    def serialize(self) -> dict:
        """Membership travels with the registry: there is no per-session file hosting it."""
        state = super().serialize()
        state['members'] = {g: sorted([str(s), str(seg), int(r), int(t)]
                                      for s, seg, r, t in members)
                            for g, members in self._fwd.items()}
        return state

    def __setstate__(self, state: dict) -> None:
        members = (state or {}).pop('members', None) or {}
        super().__setstate__(state)
        self._fwd, self._inv = {}, {}
        for gname, rows in members.items():
            for s, seg, r, t in rows:
                self.add(gname, (str(s), str(seg), int(r), int(t)))


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


class NeuronSelections:
    """Per-session neuron selections in the pair file format; a member is a 1-tuple (neuron_id,)."""

    def __init__(self, cd):
        self.cd = cd
        self.save_dir = str(cd.nd.neuron_dir(cd.save_path) / 'selections')
        self.sessions: dict[Key, SelectionData] = {}

    def bucket(self, key: Key) -> _SelectionData:
        """The session's selection, read from disk or seeded from the stable tag."""
        nd_key = key.nd()
        if nd_key not in self.sessions:
            sd = self.sessions[nd_key] = SelectionData(save_dir=self.save_dir, nd_key=nd_key)
            neurons = self.cd.nd.neurons_for(nd_key)
            ids = [(int(i),) for i in neurons.neuron_ids]
            if os.path.isfile(sd.save_path() + '.json'):
                sd.load(sd.save_path())
                selected, deleted = sd.selections[nd_key].selected, sd.selections[nd_key].deleted
            else:
                stable = neurons.tag_values.get(STABLE_TAG, [1.0] * len(ids))   # values survive relabelling
                selected, deleted = {m for m, v in zip(ids, stable) if v == 1.0}, set()
            sd.selections[nd_key].reset(ids, selected=selected, deleted=deleted)
        return self.sessions[nd_key].selections[nd_key]

    def is_selected(self, key: Key) -> bool:
        return (key.ref,) in self.bucket(key).selected

    def save(self) -> None:
        for sd in self.sessions.values():
            if sd.dirty:
                sd.save()


class DerivedGroups:
    """Members of each derived group: the session's screened pairs meeting every rule, recomputed per ask."""

    def __init__(self, cd, sd: 'SelectionDataset'):
        self.cd = cd
        self.sd = sd
        self.groups = sd.groups
        self._memo: dict = {}

    def sessions(self) -> list:
        return [str(k.session) for k in self.cd.nd.session_keys]

    def members(self, gname: str, sess: str) -> set:
        return self.rule_based_filter(self.groups.get_group_metadata(gname).rules, sess)

    def rule_based_filter(self, rules: list, sess: str) -> set:
        """The session's significant pairs meeting every rule: the one place rules are applied."""
        tests = [self._test(rule, sess) for rule in rules]
        universe = set().union(*(self.cd.ptr[k].pair_set for k in self.cd.ptr
                                 if str(k.session) == sess))
        return {p for p in universe if all(test(p) for test in tests)}

    def cached_members(self, gname: str, sess: str) -> set:
        """members, kept until clear(): a list refresh asks once per row."""
        if (gname, sess) not in self._memo:
            self._memo[(gname, sess)] = self.members(gname, sess)
        return self._memo[(gname, sess)]

    def clear(self) -> None:
        self._memo.clear()

    def groups_for_pair(self, sess: str, ref: int, tgt: int) -> list:
        return [g for g in self.groups.derived_groups()
                if (ref, tgt) in self.cached_members(g, sess)]

    def significance(self, name: str, sess: str) -> dict:
        """``{'on': pairs, 'off': pairs}`` rule set *name* passed or failed in *sess*'s swept 'all' segment."""
        ck = (DerivedRule.PAIR_SIGNIFICANCE, name, sess)
        if ck in self._memo:
            return self._memo[ck]
        out = {v: set() for v in DerivedRule.VERDICTS}
        for k, ptr in self.cd.ptr.items():
            arrays = self.sd.rule_results(k, 'all').arrays if str(k.session) == sess else {}
            if name + '.passed' not in arrays:   # another session, or never swept with this set
                continue
            reached, passed = arrays[name + '.value'], arrays[name + '.passed']
            for p in ptr.pair_set:
                if not np.isnan(reached[p]):   # NaN: outside the set's scope
                    out['on' if passed[p] else 'off'].add(p)
        self._memo[ck] = out
        return out

    def passed_rule_sets(self, sess: str, pair: tuple) -> list:
        """Rule sets the pair passed in the swept 'all' segment."""
        return [r.name for r in self.cd.rule_book.later_sets() if pair in self.significance(r.name, sess)['on']]

    def neuron_conn_counts(self, sess: str, conn_types: list, groups: list = ()) -> np.ndarray:
        """Per position: significant *conn_types* pairs the neuron sends, within *groups* when given."""
        counts = np.zeros(self.cd.nd.neurons_for(Key(session=sess)).n_neurons, dtype=int)
        tagged = self._pair_tagged(groups, sess) if groups else None
        for k, ptr in self.cd.ptr.items():
            if str(k.session) == sess and k.excitability in conn_types:
                pairs = ptr.pair_set if tagged is None else ptr.pair_set & tagged
                np.add.at(counts, [ref for ref, _ in pairs], 1)
        return counts

    def _pair_tagged(self, tags: list, sess: str) -> set:
        self.sd.ensure_groups_loaded_for([sess])   # pair tags are indexed per session on first use
        return self.groups.members_in_groups(tags, sess)

    def _test(self, rule: DerivedRule, sess: str):
        """The rule as a predicate on a (ref, tgt) of *sess*; pairs index neurons by position."""
        neurons = self.cd.nd.neurons_for(Key(session=sess))
        if rule.kind == DerivedRule.PAIR_TAG:
            return self._pair_tagged(rule.values, sess).__contains__
        if rule.kind == DerivedRule.PAIR_CONN:
            return set().union(*(p.pair_set for k, p in self.cd.ptr.items()
                                 if str(k.session) == sess and k.type_label() in rule.values)).__contains__
        if rule.kind == DerivedRule.PAIR_SIGNIFICANCE:
            sig = self.significance(rule.name, sess)
            return set().union(*(sig[v] for v in rule.values)).__contains__
        if rule.kind == DerivedRule.PAIR_LOCATION:
            where = neurons.peak_channels if rule.name == 'same channel' else neurons.shank_ids
            return lambda pair: ('yes' if where[pair[0]] == where[pair[1]] else 'no') in rule.values
        if rule.kind == DerivedRule.NEURON_GROUP:   # members are ids, pairs positions
            hit = np.isin(neurons.neuron_ids,
                          [m[0] for m in self.sd.neuron_groups.members_in_groups(rule.values, sess)])
        else:
            labels = neurons.tags.get(rule.name, [None] * neurons.n_neurons)
            hit = np.array([str(lb) in rule.values for lb in labels], dtype=bool)

        def test(pair) -> bool:
            r, t = hit[pair[0]], hit[pair[1]]
            return {'ref': r, 'tgt': t, 'either': r or t, 'both': r and t}[rule.side]
        return test


class SelectionDataset(JsonSavable, Autosave):
    """Project-level owner of groups + per-session SelectionData.

    The Qt UI passes a ``Groups`` instance (subclass of ``GroupDataset``) via
    ``groups_factory``; scripts/notebooks default to the pure ``GroupDataset``.
    """

    def __init__(self, cd, groups_factory=GroupDataset):
        JsonSavable.__init__(self, ignored_attrs=['cd', 'sessions', 'segtags', 'seg_groups',
                                                  'neuron_groups'])
        self.cd = cd
        save_dir = cd.selections_dir
        self.groups = groups_factory()
        self.groups._save_dir = groups_dir(cd)
        # The registry is shared and project-independent, so it is read here
        # rather than waiting on a project's own saved selections — a project
        # with none of its own still knows every tag.
        if os.path.isfile(self.groups.save_path() + '.json'):
            self.groups.load(self.groups.save_path())
        # segment names are a project's own, so this registry lives with the project
        self.seg_groups = SegmentGroups(save_dir=save_dir)
        if os.path.isfile(self.seg_groups.save_path() + '.json'):
            self.seg_groups.load(self.seg_groups.save_path())
        self.neuron_groups = NeuronGroups(cd)
        if os.path.isfile(self.neuron_groups.save_path() + '.json'):
            self.neuron_groups.load()
        self.groups.derived = DerivedGroups(cd, self)
        self.sessions: dict[Key, SelectionData] = {}
        self.segtags: dict[tuple, RuleResults] = {}
        self.save_dir = save_dir
        self._indexed: set = set()

    @property
    def dirty(self) -> bool:
        return (self.groups.dirty or self.seg_groups.dirty
                or any(sd.dirty for sd in self.sessions.values()))

    def save(self, path: str = None, **kw):
        """Each session owns its file; the roster indexes whichever ones exist."""
        for sd in self.sessions.values():
            if sd.dirty:
                sd.save()
        self.groups.save()
        self.groups.dirty = False
        self.seg_groups.save()
        self.seg_groups.dirty = False
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

    def rule_set_pairs(self, name: str) -> dict:
        """``{ptr key: pairs}`` that passed rule set *name* in the swept 'all' segment."""
        out = {}
        for k, ptr in self.cd.ptr.items():
            arrays = self.rule_results(k, 'all').arrays
            if name + '.passed' not in arrays:
                continue
            reached, passed = arrays[name + '.value'], arrays[name + '.passed']
            out[k] = {p for p in ptr.pair_set if not np.isnan(reached[p]) and passed[p]}
        return out

    def clear_rule_set(self, name: str) -> None:
        """Drop rule set *name*'s stored results in every segment, so no pair carries its tag."""
        for k in self.cd.ptr:
            for seg in self.rule_segments(k):
                store = self.rule_results(k, seg)
                sets = {RuleResults._set_of(a) for a in store.arrays if a != RuleResults.ON}
                if name in sets:
                    store.set_results({}, list(sets - {name}))
                    self.store_results(k, seg, store)

    def store_results(self, key: Key, segment: str, store: RuleResults) -> None:
        """Recombine *store*'s on under the rule book's applied on-sets and mode, then write it."""
        book = self.cd.rule_book
        n = self.cd.nd.neurons_for(key.nd()).n_neurons
        store.recombine(book.on_sets, book.on_mode, (n, n), self.seg_overrides(str(key.session), segment))
        store.save()

    def apply_on(self) -> None:
        """Every stored segment's on rebuilt from its set verdicts: what Apply on the on page runs."""
        for k in self.cd.ptr:
            for seg in self.rule_segments(k):
                self.store_results(k, seg, self.rule_results(k, seg))

    def on_verdict(self, key: Key, segment: str) -> np.ndarray:
        """``[ref, tgt]`` stored on for *segment*; a segment no run has reached is swept with every later set first."""
        store = self.rule_results(key, segment)
        if RuleResults.ON not in store.arrays:
            store.set_results(self.cd.sweep_rule_sets(key.change(segment=segment), self.rule_plan(key))[0], [])
            self.store_results(key, segment, store)
        return store.arrays[RuleResults.ON]

    def on_grid(self, key: Key, segment: str) -> np.ndarray:
        """``[ref, tgt]`` on for *segment*: the applied sets and the lowres p-value screen, hand-set entries winning."""
        data = self.cd.ccg_for(key.change(resolution='lowres', segment=segment))
        on = self.on_verdict(key, segment) & data.screened(data.segment_index(segment), key.excitability)
        n = on.shape[0]
        for (ref, tgt), state in self.seg_overrides(str(key.session), segment).items():
            if ref < n and tgt < n:
                on[ref, tgt] = state
        return on

    def kept_pairs(self) -> dict:
        """``{ptr key: pairs}`` selected or deleted by hand: a rebuilt pointer must still list them."""
        return {k.ptr(): {p for p in b.selected | b.deleted if len(p) == 2}   # all-session triples live elsewhere
                for sd in self.sessions.values() for k, b in sd.selections.items()}

    def get_selection_by_session(self, key: Key) -> SelectionData:
        nd = key.nd()
        sd = self.sessions.get(nd)
        if sd is None:
            sd = SelectionData(save_dir=self.save_dir, nd_key=nd)
            self.sessions[nd] = sd
        return sd

    def pairs_for_group_key(self, group: str | None, ptr_key: Key) -> set:
        """Screened pairs of *ptr_key*, narrowed to *group*; None = every screened pair."""
        valid = self.cd.ptr[ptr_key].pair_set
        if not group:
            return valid
        sess = str(ptr_key.session)
        self.ensure_groups_loaded_for([sess])
        return self.groups.pairs_in_group(group, sess) & valid

    def seg_overrides(self, sess: str, segment: str) -> dict:
        """``{(ref,tgt): bool}`` manual on/off for *segment* — what pair_on_for takes."""
        sg = self.seg_groups
        return {**{p: True for p in sg.pairs_tagged(sg.ON, sess, segment)},
                **{p: False for p in sg.pairs_tagged(sg.OFF, sess, segment)}}

    def rule_plan(self, key: Key, names: list = None) -> list:
        """``[(name, tests, mask)]`` in run order: each rule set covering *key* (of *names*, when given) with the pairs in its scope."""
        n, plan = self.cd.nd.neurons_for(key.nd()).n_neurons, []
        for r in self.cd.rule_book.later_sets():
            if r.covers(key) and (names is None or r.name in names):
                mask = np.full((n, n), not r.rules)   # no pair rules: every pair
                for p in self.groups.derived.rule_based_filter(list(r.rules), str(key.session)) if r.rules else ():
                    mask[p] = True
                plan.append((r.name, r.tests.specs(), mask))
        return plan

    def rule_results(self, key: Key, segment: str) -> RuleResults:
        """Swept rule arrays for one (session, conn type, segment), memmapped on first ask."""
        ck = (key.nd(), key.type_label(), segment)
        res = self.segtags.get(ck)
        if res is None:
            res = RuleResults(self.save_dir, key.nd(), key.type_label(), segment).load()
            self.segtags[ck] = res
        return res

    def rule_segments(self, key: Key) -> list:
        """Segments swept for *key*'s session and conn type, read off the store's dir names."""
        root = os.path.join(self.save_dir, 'rule_results')
        prefix = f"{_dir_token(key.session)}.{_dir_token(key.type_label())}."
        return sorted(n[len(prefix):] for n in os.listdir(root)
                      if n.startswith(prefix)) if os.path.isdir(root) else []

    def rule_members(self, sess: str, keys_pairs: list, on: bool) -> dict:
        """``{segment: [(ref, tgt)]}`` whose verdict is *on*: the swept store, hand-set tags winning."""
        out: dict = {}
        for key, pair_set in keys_pairs:
            pairs = sorted(pair_set)
            if not pairs:
                continue
            idx = tuple(np.array(pairs).T)
            for seg in self.rule_segments(key):
                verdict = np.asarray(self.rule_results(key, seg).arrays[RuleResults.ON])[idx]
                out.setdefault(seg, set()).update(
                    p for p, v in zip(pairs, verdict.tolist()) if v == on)
        for seg in self.seg_groups.segments_tagged(sess):
            bucket = out.setdefault(seg, set())
            for p, v in self.seg_overrides(sess, seg).items():
                (bucket.add if v == on else bucket.discard)(p)
        return {seg: sorted(ps) for seg, ps in sorted(out.items()) if ps}

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
