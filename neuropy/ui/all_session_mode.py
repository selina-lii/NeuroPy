"""AllSessionMode — data coordinator for cross-session review mode."""
from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState
    from neuropy.analyses.ms_connectivity import CCGDataset


class AllSessionMode:
    """Owns all pure-data operations for the all-session view.

    No UI imports. CCGReviewUI delegates enter/exit/rebuild calls here.
    """

    def __init__(self, nav: 'AppState', cd: 'CCGDataset'):
        self.nav = nav
        self.cd = cd

    def load_groups(self):
        sessions = [str(nk.session) for nk in self.nav.real_nd_keys()]
        self.nav.ensure_groups_loaded_for(sessions)

    def rebuild_pair_handles(self):
        lbl = self.nav.key.type_label()
        dead = set()
        for trip in self.nav.active_selections.deleted:
            if len(trip) >= 3:
                dead.add((str(trip[0]), int(trip[1]), int(trip[2])))
        handles: list = []
        seen: set = set()
        for k in sorted(self.cd.ptr.keys(), key=lambda k: str(k.session)):
            if k.type_label() != lbl:
                continue
            sess = str(k.session)
            valid = self.cd.ptr.get(k).pair_set
            for r, t in sorted(valid):
                if (sess, r, t) in dead:
                    continue
                h = (k, int(r), int(t))
                if h not in seen:
                    seen.add(h)
                    handles.append(h)
        self.nav.set_cross_session_handles(handles)

    def rebuild_universe(self):
        """Rebuild the cross-session pair universe for the current conn type."""
        self.load_deleted_aggregate()
        self.rebuild_pair_handles()
        self.sync_selection_from_universe()

    def load_deleted_aggregate(self):
        lbl = self.nav.key.type_label()
        deleted: set = set()
        for k in self.cd.ptr.keys():
            if k.type_label() != lbl:
                continue
            ptr = self.cd.ptr.get(k)
            valid = ptr.pair_set
            raw = set(self.nav.sd.get_selection_by_session(k).selections[k].deleted) & valid
            sess = str(k.session)
            for r, c in raw:
                deleted.add((sess, int(r), int(c)))
        b = self.nav.active_selections
        for p in deleted:
            b.set_pair_state(p, 'del')

    def flush_deleted_to_stores(self):
        lbl = self.nav.key.type_label()
        by_key: dict = defaultdict(set)
        keyobj: dict = {}
        for trip in self.nav.active_selections.deleted:
            s, r, t = trip[0], int(trip[1]), int(trip[2])
            for k in self.cd.ptr.keys():
                if k.type_label() != lbl or str(k.session) != s:
                    continue
                ks = str(k)
                by_key[ks].add((r, t))
                keyobj[ks] = k
        for ks, pairs in by_key.items():
            store = self.nav.sd.get_selection_by_session(keyobj[ks]).selections[keyobj[ks]].deleted
            store.clear()
            store.update(pairs)

    def sync_selection_from_universe(self):
        """Selected = the grouped/tagged pairs (this app's model is tag ⟺ select), aggregated
        across sessions for the current conn-type — not the whole significant universe.
        Deleted is taken from the prior load_deleted_aggregate pass."""
        hl = self.nav.cross_session_handles
        universe = {(str(ckey.session), int(r), int(t)) for ckey, r, t in hl}
        deleted = set(self.nav.active_selections.deleted) & universe
        grp = self.nav.groups
        selected: set = set()
        for gname in grp:
            if gname.startswith('__'):
                continue
            selected |= grp.pairs_in_group_by_session(gname)
        selected = (selected & universe) - deleted
        self.nav.active_selections.reset(universe, selected=selected, deleted=deleted)
