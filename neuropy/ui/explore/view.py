"""The explore view: lists pairs like the pair view, renders into the explore panel."""
from __future__ import annotations

from neuropy.analyses.view_spec import PairView

EXPLORE_VIEW = 'explore'


class ExploreView(PairView):
    name = EXPLORE_VIEW

    def render(self, ui) -> None:
        ui.explore.load_pair()
