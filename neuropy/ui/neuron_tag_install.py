"""The project's neuron tag set and the panel that edits it."""
from __future__ import annotations

import os

from neuropy.analyses.neuron_tags import NeuronTagSet
from neuropy.analyses.pair_selection_data import NeuronGroups


def load_tags(cd) -> NeuronTagSet:
    """The project's gradient specs, restored when they have been saved before."""
    tags = NeuronTagSet(save_dir=str(cd.nd.neuron_dir(cd.save_path)))
    tags.bind(cd)
    if os.path.isfile(tags.save_path() + '.json'):
        tags.load()
        tags.bind(cd)
    return tags


def load_groups(cd) -> NeuronGroups:
    """The project's neuron groups, restored when they have been saved before."""
    groups = NeuronGroups(cd)
    if os.path.isfile(groups.save_path() + '.json'):
        groups.load()
    return groups


def install(ui) -> NeuronTagSet:
    """Give *ui* its tag set and its neuron groups; Manage Groups holds the editors."""
    ui.neuron_tags = load_tags(ui.nav.cd)
    ui.neuron_groups = load_groups(ui.nav.cd)
    return ui.neuron_tags
