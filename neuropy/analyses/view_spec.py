"""What each view selects: the items it lists, their Keys, and the neurons behind them."""
from __future__ import annotations

import numpy as np

from neuropy.analyses.neurons_dataset import Key

PAIR_VIEW = 'pair'
NEURON_VIEW = 'neuron'


def epoch_bin_mask(times: np.ndarray, bounds: list) -> np.ndarray:
    """True where a bin's time falls inside one of *bounds*, which the slider already filtered."""
    mask = np.zeros(len(times), dtype=bool)
    for start, stop, _label in bounds:
        mask |= (times >= start) & (times < stop)
    return mask


class ViewSpec:
    """One view's idea of an item — subclasses differ only in what an item is.

    A pair is a two-neuron item and a neuron a one-neuron item, so every consumer
    that works per neuron reads `neurons_of` and never asks which view is active.
    """

    name = ''

    def __init__(self, nav):
        self.nav = nav

    def items(self, key: Key) -> list:
        """Every item this view lists for *key*'s session."""
        raise NotImplementedError

    def key_of(self, item) -> Key:
        """The item's identity, as a Key."""
        raise NotImplementedError

    def neurons_of(self, item) -> list:
        """Neuron ids the item covers."""
        raise NotImplementedError

    def row_label(self, item) -> str:
        """What the list panel prints for the item."""
        raise NotImplementedError

    def row_of(self, key: Key) -> int:
        """Row the item with this Key sits on; 0 when it is not listed."""
        for row, item in enumerate(self.items(self.nav.key)):
            if self.key_of(item) == key:
                return row
        return 0

    def groups_of(self, ui):
        """The group registry this view tags into."""
        return ui.nav.groups

    def render(self, ui) -> None:
        """Redraw whatever this view owns after the cursor moved."""
        raise NotImplementedError

    def redraw_network(self, ui) -> None:
        """The probe network follows the pair view only; other views leave it alone."""

    def list_panel(self, ui):
        """The panel listing this view's items — owner of its undo stack."""
        raise NotImplementedError


class PairView(ViewSpec):
    """Items are (ref, tgt) pairs — the original view."""

    name = PAIR_VIEW

    def items(self, key: Key) -> list:
        return [tuple(int(x) for x in p) for p in self.nav.all_pairs_np]

    def key_of(self, item) -> Key:
        return Key.pair(self.nav.current_session_str, item[0], item[1])

    def neurons_of(self, item) -> list:
        return [int(item[0]), int(item[1])]

    def row_label(self, item) -> str:
        return f"{int(item[0])} → {int(item[1])}"

    def row_of(self, key: Key) -> int:
        """Defers to nav: all-session mode indexes cross-session handles, not pairs."""
        if self.nav.session_any_mode:
            return self.nav.get_pair_index((key, key.ref, key.tgt))
        return self.nav.get_pair_index((key.ref, key.tgt))

    def render(self, ui) -> None:
        ui.mainview.request_render()

    def redraw_network(self, ui) -> None:
        ui.neuron_network.draw()

    def list_panel(self, ui):
        return ui.pairs_view.pair_selection


class NeuronView(ViewSpec):
    """Items are single neurons, labelled by type."""

    name = NEURON_VIEW

    def items(self, key: Key) -> list:
        """Neuron ids, or (session, id) handles when every session is in view."""
        nd = self.nav.cd.nd
        if not self.nav.session_any_mode:
            return [int(i) for i in nd.neurons_for(key.nd()).neuron_ids]
        return [(str(k.session), int(i))
                for k in nd.session_keys
                for i in nd.neurons_for(k).neuron_ids]

    def key_of(self, item) -> Key:
        if isinstance(item, tuple):
            return Key(session=item[0], ref=int(item[1]))
        return Key(session=self.nav.current_session_str, ref=int(item))

    def neurons_of(self, item) -> list:
        return [int(item[1]) if isinstance(item, tuple) else int(item)]

    def groups_of(self, ui):
        return ui.neuron_groups

    def render(self, ui) -> None:
        ui.neuron_view.render()

    def list_panel(self, ui):
        return ui.neuron_view.list_panel

    def row_label(self, item) -> str:
        key = self.key_of(item)
        neurons = self.nav.cd.nd.neurons_for(key.nd())
        types = neurons.neuron_type
        index = int(np.flatnonzero(np.asarray(neurons.neuron_ids) == key.ref)[0])
        kind = '' if types is None else f"  {types[index]}"
        prefix = f"{key.session}  " if isinstance(item, tuple) else ''
        return f"{prefix}{key.ref}{kind}"
