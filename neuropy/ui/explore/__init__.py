"""Exploratory views built on the review UI's public objects; nothing outside this package calls in."""
from __future__ import annotations

from neuropy.ui.explore.panel import ExplorePanel
from neuropy.ui.explore.view import EXPLORE_VIEW, ExploreView
from neuropy.ui.utils import AddableDropdown


def install(ui) -> ExplorePanel:
    """Register the explore view on *ui* and add it to the View-by list."""
    nav = ui.nav
    nav._views[EXPLORE_VIEW] = ExploreView(nav)
    ui.explore = ExplorePanel(ui)
    ui.center_stack.addWidget(ui.explore)
    combo = next(c for c in ui.findChildren(AddableDropdown) if c.name == 'view')
    combo.blockSignals(True)
    combo.addItem(EXPLORE_VIEW)
    combo.blockSignals(False)

    def on_view(name: str) -> None:
        # runs after ui.set_view, which shows the pair list and main view for any non-neuron view
        if name == EXPLORE_VIEW:
            nav.set_current_pair(nav._view_rows.get('pair', 0))
            ui.center_stack.setCurrentWidget(ui.explore)
            ui.left_title.setText("Explore")
            ui.explore.activate()
        else:
            ui.explore.deactivate()
            if name == 'pair':
                nav.set_current_pair(nav._view_rows.get(EXPLORE_VIEW, 0))
                ui.mainview.request_render()

    def on_pair(*_):
        if nav.view.name == EXPLORE_VIEW:
            ui.explore.load_pair()

    combo.currentTextChanged.connect(on_view)
    nav.pair_changed.connect(on_pair)
    nav.key_changed.connect(on_pair)
    return ui.explore
