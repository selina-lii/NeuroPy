# Add Project dialog — folder-of-NWB loading

## State: what already landed (working, verified headless)

| Piece | Where |
|---|---|
| `_options` tables | `analyses/neurons_dataset.py:194`, `analyses/ms_connectivity.py:55` |
| `ConfigOptionsWidget` | `ui/utils.py:481` |
| `ListPickerButton.select_all_when_empty` flag | `ui/utils.py` |
| `_TargetBox` + `FieldMapWidget` | `ui/dialogs.py:1246` |
| `AddProjectDialog` | `ui/dialogs.py:1373` |
| `build_project(sessions, nd_conf, ccg_conf)` | `analyses/project.py:23` |
| `_adopt_project(new_cd)` | `ui/ccg_ui.py:348` |
| `IndexBar.add_project` wired | `ui/menubar.py:183` |
| `NWBFile.input_fields` | `io/nwbio.py:78` |
| `FieldMap(..., partial=True)`, `check_required()` public | `io/fieldmap.py:76` |

### Conventions pinned during the build (do not re-litigate)
- `_options` entry = `"init param": ("kind", choices)`. **No default in the table** — defaults
  live only in `__init__`; `ConfigOptionsWidget` reads them via `inspect.signature`.
- kinds: `multi` / `choice` / `metric` / `int` / `float` / `bool`. Adding a kind = one entry in
  `_OPTION_KINDS` (`ui/utils.py`), which maps kind → `(make_fn, read_fn)`.
- `choice` stores the real value as `itemData` (so `use_acceleration=None` survives as `None`,
  not `"None"`); reader is `currentData()`.
- A blank number box returns the `_BLANK` sentinel → the kwarg is omitted → `__init__`'s default
  wins. That is how `bin_size=None` keeps its resolution-preset behavior.
- Configs never import Qt. `_options` is data; `ui/` interprets it. (Rejected: per-config
  `build_ui()` with lazy Qt imports — inverts the analyses→ui dependency.)
- Required fields are validated **on submit**, not by graying the Build button. No `changed`
  signal, no `is_complete()`.
- `load_project` was left untouched; `build_project` is the new entry taking ready-made configs.
- The compute checkbox runs `cd.get_ccg()` **inside** `_on_accept`, so `_out` stays a clean
  3-tuple `(neurons, cd, sd)`.

### Verified
Against a real file (`001695/sub-M05/sub-M05_ses-20240727T100000_ecephys.nwb`, 16 columns):
both configs round-trip through `values()`; assignment / unassignment / arity-1 replace /
`NWB_DEFAULT` prefill / missing-required raise all behave. `report()` shows all five fields ok.
`py_compile` clean on every touched file; no import cycle.

**Not verified:** live GUI interaction (clicks, layout, Browse) — headless only.

Headless Qt now works, and this is the incantation:
```
QT_QPA_PLATFORM=offscreen \
QT_QPA_PLATFORM_PLUGIN_PATH=/Users/selinl/miniforge3/envs/NeuroPy2/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms \
/Users/selinl/miniforge3/envs/NeuroPy2/bin/python <script>
```
(Never the default `python` — it lacks pyqtgraph. Modal `QMessageBox` paths hang headlessly.)

---

## Next: the dataset is a folder, not a file

`/Users/selinl/Documents/ms_synchrony/001695/` — 22 `.nwb` files across 6 `sub-*/` dirs.
**Measured: all 22 share one identical column signature**, so one field map covers the folder.
Scanning all 22 (open + read `colnames`) takes ~4 s.

Columns present in every file:
```
cell_type, cell_area, firing_rate, ab_ratio, burstIndex_Mizuseki2012, cv2,
maxWaveformCh, troughToPeak, acg_tau_decay, acg_tau_rise, thetaModulationIndex,
x_position_probe, y_position_probe, waveforms, acg, spike_times
```

### Where the loop goes
`build_project(sessions, ...)` already takes a list — it does not change. The loop belongs in a
new **`NWBDataset`** in `io/nwbio.py`, *not* in the dialog (else only the GUI can open a folder)
and *not* in `build_project` (which must stay format-agnostic).

`NWBDataset` is a scanned artifact both terminal and GUI read — same role `FieldMap` plays.
A bare `list[NWBSession]` cannot hold the union / per-file coverage / partial summary.

### Decisions pinned
- **Browse = one button, auto-detects** file vs. folder. (Qt's dual `ExistingFile`+`Directory`
  mode is inconsistent on macOS — needs care.)
- **Scan on Browse**, showing progress in the Source field (`scanning 14/22…` →
  `/…/001695  22 sessions, 16 fields`). Use `QApplication.processEvents` to keep it painting.
- **Chips = union** of all files' columns. Missing fields are skipped per session.
- **Missing a required field** (`spike_times`, `neuron_type`) → that session is **skipped**, not
  loaded broken. Missing an optional field → session loads, marked **partial**.
- Summary at the end lists Skipped and Partial groups separately.

### `NWBDataset` outline (`io/nwbio.py`)
Keep it minimal — the user cut the first draft for over-building. Only what the dialog and a
terminal user actually call:

```python
class NWBDataset:
    """A folder of .nwb files scanned as one unit: shared columns, per-file coverage."""

    def __init__(self, path, pattern='**/*.nwb', on_progress=None)
        # self.files, self.fields_by_file = {Path: [columns]}

    @property
    def input_fields(self) -> list      # union, first-seen order

    def split(self, field_map) -> tuple # (usable, skipped); dict Path -> missing target names
    def report(self, field_map) -> str  # the Skipped / Partial summary
    def sessions(self, field_map) -> list[NWBSession]   # usable files only
```

Open question deferred: whether `split`/`report`/`sessions` is the right cut, or whether one
`coverage()` returning the per-file missing-map suffices with the rest computed at the call site.

### Dialog changes
- `_on_browse_btn`: one picker → `NWBDataset(path, on_progress=...)` →
  `set_available(ds.input_fields)` → prefill from `NWB_DEFAULT`.
- `_on_accept`: `ds.sessions(field_map)` instead of `[NWBSession(path, ...)]`; show
  `ds.report(field_map)` before building (or after, as a results dialog).
- `NWBSession` currently takes `fields: dict`; per-session field maps must drop the targets that
  session lacks.

### Schema scope — why only 6 target fields
`UNITS_SCHEMA` mirrors exactly what `NWBSession.neurons_stable` (`core/nwb_session.py:84`) passes to
`Neurons`. The other 10 columns of 001695 (`firing_rate`, `cell_area`, `troughToPeak`, `acg*`, …) have
**zero readers anywhere in the repo** — target fields for them would feed nothing. They still show up
as source chips, so nothing is hidden.

`neuron_id` added as OPTIONAL: `NWBFile.neuron_ids` falls back to `units.id` (the row index) when
unmapped, so a dataset carrying its own cluster IDs can now bind them.

**FUTURE — x/y position override.** `_get_neuron_positions` (`ui/neuron_network.py:1832`) derives
network coordinates *indirectly*: `peak_channels` → probegroup dataframe → x,y. A dataset with
explicit `x_position_probe` / `y_position_probe` already has the answer and should override that.
Two failure modes it would fix: no probegroup means no panel at all (`if pg_info is None: return
None`), and an unmatched peak_channel silently collapses to the origin via `.fillna(0.0)`. Catch:
001695 maps `x_position_probe` → `shank_id` as a stand-in grouping (`nwbio.py:42`), so that column
would serve two purposes — resolve before rewiring.

### Also requested, not yet built
**Chip move-vs-copy flag on `FieldMapWidget`.** Today assignment *copies* — the source chip stays
in the left list, so one column can feed several targets. Requested: **move is the default**
(chip leaves the source list once assigned, returns on unassign), with a flag to allow one input
to land in multiple targets. Touches `set_available` / `_on_target_box_click` /
`_on_unassign_chip` in `ui/dialogs.py`.

---

## Backlog (unrelated to this feature)
- Fold `ch_per_shank` into a probe config object — it is a `NeuronsDatasetConfig` scalar
  (`neurons_dataset.py:206`) but is really probe geometry, hardcoded `16` at `ccg_panel.py:1493`
  and `plotting/ccg.py:486` (both with a `# TODO`).
- Wire `peak_channel` / `shank_id` / `waveforms` in `nwbio.py` to read from `fields` — their
  `DANDI_001695` entries are recorded but unread (only `neuron_type` is wired).
- Value-map editing in `FieldMapWidget` (`Field.value_map`) — `pyr`/`inter` renaming is still the
  hardcoded `DANDI_001695` map.
- `menubar.add_session` and `time_slider.add_theme` are still `pass`.
- Autosave is dead: `schedule_autosave()` never called; `autosave_*` settings unread (hardcoded
  30-min const); groups never autosaved. Note `autosave_sel_interval` is now a `(number, unit)`
  tuple — the reader must unpack.
- `get_pair_index` cross-session bug in `app_state.py:491-509` — the `len(inds) < 3` fallback
  matches `(ref,tgt)` against `cross_session_handles` ignoring session.
- Segment-unify plan (Phase 1) — full design in `~/.claude/plans/how-is-settings-recursive-cherny.md`.
