"""Wire gradient neuron tags into a running UI.

Everything the feature needs from the host is reached through ``nav``; the host
files stay untouched apart from one ``install(ui)`` call. The pair list and
Manage Groups pick the tags up by wrapping what they already do, so neither has
to know what a gradient tag is.
"""
from __future__ import annotations

import os

from neuropy.analyses.neuron_tags import NeuronTagSet
from neuropy.ui.neuron_tag_ui import NeuronTagPage
from neuropy.ui.pair_selection_panel import _ROLE_AREAS
from neuropy.ui.utils import SideNavPanel


def tags_dir(cd) -> str:
    """Cache lives beside the selections, not among them: that dir is per-session files."""
    return os.path.join(cd.selections_dir, 'neuron_tags')


def load_tags(cd) -> NeuronTagSet:
    """The project's tag set, restored when it has been saved before."""
    tags = NeuronTagSet(save_dir=tags_dir(cd))
    tags.bind(cd)
    if os.path.isfile(tags.save_path() + '.json'):
        tags.load()
        tags.bind(cd)
    return tags


def install(ui) -> NeuronTagSet:
    """Attach the tag set to *ui* and hook the list dots and the manage page."""
    tags = load_tags(ui.nav.cd)
    ui.neuron_tags = tags
    _patch_pair_dots(ui, tags)
    _patch_manage_dialog(ui, tags)
    return tags


def _patch_pair_dots(ui, tags: NeuronTagSet) -> None:
    """Append gradient dots after the region dots the panel already draws."""
    panel = ui.pairs_view.pair_selection
    original = panel._pair_area_rgb

    def with_tags(k):
        dots = list(original(k))
        for neuron in (k.ref, k.tgt):
            if neuron is not None:
                dots.extend(tags.neuron_rgb(k, int(neuron)))
        return dots

    panel._pair_area_rgb = with_tags

    # the delegate reserves room for two dots; make it measure what it is handed
    delegate = panel._tag_delegate
    radius = delegate._DOT_R

    def areas_width(index, _r=radius):
        dots = index.data(_ROLE_AREAS)
        return 0 if not dots else len(dots) * (2 * _r + 2) + 4

    delegate._areas_width = areas_width


def _patch_manage_dialog(ui, tags: NeuronTagSet) -> None:
    """Add a 'Neuron Tags' page to Manage Groups the next time it is built."""
    from neuropy.ui.dialogs import ManageGroupsDialog

    if getattr(ManageGroupsDialog, '_neuron_tags_patched', False):
        return
    original_build = ManageGroupsDialog._build

    def build_with_tags(self):
        original_build(self)
        page = NeuronTagPage(tags, ui.nav, parent=self)
        side_nav = self.findChild(SideNavPanel)
        if side_nav is not None:
            side_nav.add_page("Neuron Tags", page)

    ManageGroupsDialog._build = build_with_tags
    ManageGroupsDialog._neuron_tags_patched = True
