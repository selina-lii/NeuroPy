# Simplifying project loading and switching

## Why

Four bugs in two days all came from the same place: **nobody owns the answer to "what is a
project?"** Session identity lives in `cd.nd`, project identity lives in `cd.conf.name`, and the two
are wired together by hand at each call site. Every bug was a place where they disagreed.

- Stale key survived a project switch → panels rendered the old project's data (`find()` searched
  `cd.ptr`, empty for an uncomputed project).
- `_switch_project` reused `self.cd.nd` → selecting 001695 would load it with test2's Bapun rats.
- `add_project` set `_project_dir` *after* `_adopt_project` → panels refreshed against the old value.
- Add-row text reached `_switch_project` as a directory name → created `data/project_oject…/`.

A fifth was worse because it surfaced four stages from its cause: `NWB_DEFAULT` bound `cell_type`
with no value map, so every session loaded 0 neurons and the failure appeared as
`need at least one array to concatenate` inside CCG compute.

None were deep. All were ordering or source-of-truth mistakes that the pipeline below makes
impossible to express.

## The pipeline

```
1. field + value mapping    what the source's names and values mean
2. prepare conf             the project header — complete, inert, on disk
3. load into nd             sessions arrive fully prepared; nd only indexes and filters
4. create cd                compute, keyed off nd
```

**Once data reaches `nd`, everything is in place and internal to NeuroPy.** No parsing, no
filesystem access, no source-format knowledge past stage 1. That is the invariant this plan enforces;
each hurdle below is a place the current code violates it.

## Hurdles

**H1 — naming runs at stage 3, not stage 1.**
`NeuronsDataset.__init__` takes `naming=` and stamps `session.session_name` onto sessions it was
handed (`neurons_dataset.py:264-266`), but `NWBDataset.sessions()` already named them
(`nwb_session.py:175`). Two namers; `nd`'s wins by running last. Naming is "how do I read this
source" — stage 1.

**H2 — `nd` still parses raw sessions.**
`session.neurons`, `session.recinfo`, and `_load_themes` discovering Epochs via `dir(session)`
(`neurons_dataset.py:413`). Theme discovery is mapping logic living inside the dataset.

**H3 — `_load_probe_info` reads the filesystem at stage 3.**
It globs `*.probegroup.npy` beside the session (`neurons_dataset.py:388`). NWB sessions never have
one, so probe info is silently `None` — which is why `ch_per_shank` is hardcoded to `16` downstream
(`ccg_panel.py:1493`, `plotting/ccg.py:486`). The data exists: it's `x_position_probe` /
`y_position_probe`, already mapped at stage 1, arriving by a route that doesn't work.

**H4 — no stage refuses bad input.**
The 0-neuron build passed all four stages. Each boundary must reject what the next stage cannot use.

**H5 — `cd` and `nd` are re-paired after construction.**
`CCGDataset.nd` is mutable and `AppState.set_cd` swaps the pair. Both are per-project; they should be
created together and never re-paired.

**H6 — stage 2's inputs are split between source and GUI.**
`themes` is scan-discovered, `neuron_types` comes from `ConfigOptionsWidget`. A headless caller
cannot reproduce stage 2 without the dialog.

## What exists now

**Three entry points, overlapping:**

| | signature | what it does | callers |
|---|---|---|---|
| `load_project` | `(name, sessions, duration, alpha, use_acceleration, naming)` | builds configs from loose kwargs | `notebooks/run_ccg_gui.py` |
| `build_project` | `(sessions, nd_conf, ccg_conf, naming)` | builds from ready configs | `AddProjectDialog._on_accept` |
| `open_project` | `(name, sessions=None, naming)` | loads from disk; reads header when sessions omitted | `notebooks/CCG_gui.ipynb` |

`load_project` is `build_project` with configs inlined; `open_project` is `build_project` plus
`cd.load()`/`sd.load()`. They differ in *how configs are supplied*, not in what a project is.

**Adoption is 12 steps** (`_adopt_project`, `ccg_ui.py:357`), order-dependent and unenforced. Two of
the four bugs lived here.

## Target

### Stage 2 is the whole interface

`ProjectConfig` becomes the complete description — everything stages 3-4 need, nothing they don't:

```python
name, source, format          # where the data is
fields                        # stage-1 mapping, value maps included
naming                        # dataset module supplying session_name (replaces the callable)
nd_conf                       # NeuronsDatasetConfig kwargs
resolution                    # dangling, as agreed
```

`nd_conf` closes a live hole: reopening currently rebuilds `NeuronsDataset` with `self.cd.nd.conf` —
the *previous* project's neuron filter. Same class as reusing `cd.nd`; latent only because both
projects use defaults.

### One loader

```python
def open_project(name: str) -> tuple[NeuronsDataset, CCGDataset, SelectionDataset]:
    """Everything a project is, reconstructed from its header."""
```

The header is the only input, so no caller can supply the wrong sessions. `build_project` narrows to
one job — **create** a project: write the header, then delegate to `open_project`. Build and reopen
share one path, so a freshly built project is identical to a reopened one. `load_project` is deleted.

### Sessions arrive prepared

Stage 1 outputs sessions that are named, mapped, and probe-annotated. `nd` indexes them by `Key` and
applies its config filters — no globbing, no `dir()` discovery, no filesystem access. `naming=` leaves
`NeuronsDataset.__init__`.

### Refusal at the mapping boundary

`FieldMap` rejects a `value_map` field whose values reach none of the field's targets — the exact
0-neuron case, caught at stage 1 where the mistake is. The GUI already shows it (amber border); this
makes it enforceable, and gates the Build button.

### Adoption becomes three steps

```python
def _adopt_project(self, new_cd):
    self.nav.set_project(new_cd)        # cd + jitter_mgr + time_slider + selection reset
    self._goto(self.nav.first_key())    # one key rule, nd-first
    self._post_load_refresh()           # panels redraw from nav
```

One key rule replaces the three-way fallback: **current session if the new project has it, else its
first — sourced from `nd`**, never `ptr`, which is empty until CCGs exist. That is the stale-key bug.

### Typed project identity

`_project_dir: str` sliced with `[len('project_'):]` is what let `"＋ Add project…"` become a
directory. Replace with `self.project: ProjectConfig`; the combo carries it as `itemData`, so a
non-project row has no data. The `startswith('project_')` guard disappears — the case becomes
unrepresentable rather than guarded.

## Plan

**Phase 1 — header completeness** (`ms_connectivity.py`) — closes H6
Add `naming`, `nd_conf`. `open_project(name)` builds from the header alone; `build_project` writes
then delegates; delete `load_project`.
*Verify:* build headlessly, reopen by name only, assert identical session keys, neuron counts, conf.

**Phase 2 — stage-1 refusal** (`fieldmap.py`, `dialogs.py`) — closes H4
`FieldMap` rejects unusable value maps; Build disabled while any required field is unusable.
*Verify:* `NWB_DEFAULT` on 001695 raises at stage 1; `dandi_001695.FIELDS` passes.

**Phase 3 — prepared sessions** (`nwb_session.py`, `neurons_dataset.py`) — closes H1, H2, H3
Naming moves to stage 1 and leaves `nd`. Themes come from the scan, not `dir()`. Probe info comes
from mapped position columns, not a glob — which retires the hardcoded `16` (todo #12).
*Verify:* `nd` touches no `session.basepath` / `dir(session)`; probe info non-None for NWB.

**Phase 4 — one key rule** (`app_state.py`, `ccg_ui.py`) — closes H5
`nav.first_key()`; `_adopt_project` collapses to three steps; `cd`/`nd` paired at construction.
*Verify:* adopt an uncomputed project (empty `ptr`), assert `nav.key.session` is one of its own —
the regression test for the original bug.

**Phase 5 — typed identity** (`ccg_ui.py`, `menubar.py`)
`self.project: ProjectConfig`; combo carries it in `itemData`; `_switch_project` takes a config.
*Verify:* the Add row cannot produce a project; switching loads that project's own sessions.

**Phase 6 — migration**
Headerless projects (`test2`) get one written on first open, inferred from the loaded `nd`. Until
then `open_project` keeps a "no header → use current sessions" branch — the only such branch in the
design, deletable once every project on disk has a header.

## Not in scope

- Panels owning `cd` references (`jitter_mgr.cd`, `time_slider.cd`) — real, but separate. Once they
  read `nav.cd`, `set_project`'s fan-out becomes one assignment.
- Multi-dataset loading (todo #20). The header makes it possible — several `ProjectConfig`s, one
  `cd` — but nothing here assumes it.
- `AnalysisDataset.example()` still references the phantom `self.data` and has zero callers
  (`__len__`, same bug, already removed). Delete when touched.

## Risk

Phase 1 changes `open_project`'s signature, which the notebook calls — one line, already updated once
this week. Phase 3 is the largest: it moves probe handling, so it touches `ch_per_shank` consumers.
Phases 1-2 and 4-5 are internal.

Each phase is independently shippable and leaves the app working.
