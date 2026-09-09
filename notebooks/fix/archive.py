"""One-shot migrations, retired from the live code once every project on disk
had already been through them. Kept verbatim so an old project can be brought
forward by pasting the relevant function back into its module.

Each entry records where it lived, what it fixed, and what was on disk when it
was retired.
"""
from __future__ import annotations

import os
import shutil

import numpy as np


# ── retired 2026-08-25 from neuropy/analyses/pair_selection_data.py ──────────
# Called by open_project() and ccg_ui._switch_project(). Retired because every
# project already had the shared data/groups.json, so it returned False every
# time. Restore if a project predating the shared registry turns up.

def adopt_project_groups(cd) -> bool:
    """Move a pre-sharing project groups.json up to the shared location, once."""
    from neuropy.analyses.pair_selection_data import groups_dir
    shared = os.path.join(groups_dir(cd), 'groups.json')
    local = os.path.join(cd.selections_dir, 'groups.json')
    if os.path.isfile(shared) or not os.path.isfile(local):
        return False
    shutil.copyfile(local, shared)
    print(f"[Groups] adopted {local} as the shared registry → {shared}")
    return True


# ── retired 2026-08-25 from CCGPointer.inds, neuropy/analyses/ms_connectivity.py ──
# Pointers used to store (ref, tgt); a segment column was prepended when dim0
# became segments. Retired after a scan found zero 2-column pointers left.
#
# NOTE: the *other* filter on that property — dropping ACG self-pairs — is still
# live and must stay: 28 pointers on disk still carry them.

def widen_inds_to_segments(inds: np.ndarray) -> np.ndarray:
    """Prepend a zero segment column to a pre-segment (ref, tgt) pointer array."""
    if inds.ndim == 2 and inds.shape[1] == 2:
        return np.column_stack([np.zeros(len(inds), dtype=int), inds])
    return inds
