# CCG Review UI — Codebase Context

> **Purpose:** Persistent context for future conversations about migrating or modifying the CCG manual-review interface.  
> **Authoritative spec for migration:** User-described features (piece by piece), **not** the current broken implementation.  
> **Last updated:** 2026-05-30

---

## 1. Project intent

The CCG Review UI (`CCGReviewUI`) is an interactive desktop tool for manually reviewing cross-correlograms (CCGs) between neuron pairs, marking significant connections, organizing pairs into groups, and exporting selections. It sits at the end of a spike-analysis pipeline built on `NeuronsDataset` → `CCGDataset`.

**Migration goals (user-stated):**
- Drastically reduce code volume (~20k LOC in `neuropy/ui/` today)
- Lower regression risk when adding features
- Faster rendering / interaction (phy-like snappy feel)
- Desktop-first; pip/conda install; macOS + Linux + Windows
- No web hosting required for v1
- Full rewrite (not incremental Tkinter preservation)
- Stack leaning toward **PySide6 + pyqtgraph** (live plots) + **matplotlib** (static export)

**Design principles** (from `design.md` in repo root):
- Computation lives in `analyses/`; UI logic in `ui/`; shared non-project UI helpers in `ui/utils.py`
- Prefer adding to component files over `ccg_ui.py`; prefer managers over the main class
- Maintain modularity so the interface can migrate to another UI package

---

## 2. End-to-end data flow

```
Recording sessions
       │
       ▼
NeuronsDataset          (neuropy/analyses/neurons_dataset.py)
  • Filter neurons by epoch / type
  • Segment edge_times per session
  • Lazy-load ProbeGroup per session
       │
       ▼
CCGDataset              (neuropy/analyses/ms_connectivity.py)
  • spike_correlations → raw CCG arrays
  • EranConv → significance, pointers
  • Disk cache: ccgdata + ccgpointers sidecars (+ optional highres, jitter)
       │
       ├── CCGData       [n_seg, n_ref, n_tgt, n_bins] arrays + pvals
       └── CCGPointer    index of significant pairs + segment structure
       │
       ▼
CCGReviewUI             (neuropy/ui/ccg_ui.py + satellite modules)
  • SelectionData       user review state (sel / unsel / del, groups, tags)
  • Display + interaction (currently Tkinter + matplotlib)
```

### 2.1 `Key` — hierarchical index

Defined in `neurons_dataset.py`. Fields: `session`, `epoch`, `segment`, `ref_ind`, `target_ind`, `excitability`, `conn_type`. Used everywhere to address sessions, connection types, and pairs.

### 2.2 `NeuronsDataset`

**Config:** `NeuronsDatasetConfig` — epochs, neuron types, segment splitting (`n_segments`, `seg_stride`, `seg_len`), recinfo, `ch_per_shank`.

**Per session (`Key.session`):**
| Attribute | Content |
|-----------|---------|
| `data[key]` | Filtered `Neurons` object |
| `edge_times[key]` | DataFrame: start, stop, label per segment |
| `segment_firing_rates[key]` | Firing rates per segment |
| `probegroups[key]` | Probe geometry (lazy) |

### 2.3 `CCGDataset`

**Config:** `CCGConfig` — bin_size, duration (window), alpha, significance criteria, save paths under `data/ccg/`.

**In memory:**
| Store | Type | Role |
|-------|------|------|
| `_ccg[nd_key]` | `CCGData` | Raw + null CCG, pval/qval, significant masks |
| `data[key]` | `CCGPointer` | Which pairs passed significance for this `Key` |
| `spurious[key]` | `CCGPointer` | Pairs that passed rough checks but wrong conn type |
| `_ccg_highres[nd_key]` | `CCGData` | Optional higher-resolution arrays |
| `_jitter_results[nd_key]` | dict | Jitter surrogate results per pair |

**Load/compute strategy (`get_ccg`):**
1. Try load `ccgdata` sidecars (raw arrays) — recompute only if compute fields change
2. Try load `ccgpointers` — recompute EranConv if significance config stale
3. Full compute if missing: `spike_correlations` + `EranConv`

### 2.4 `CCGData`

Shape convention: `ccg[n_seg, n_ref, n_tgt, n_bins]` (same for null, pval, etc.).

### 2.5 `CCGPointer`

Points at significant pairs without duplicating arrays:
- `key` — which session / conn_type / etc.
- `_inds` — array of pair indices; shape `(n_pairs, 2)` or `(n_pairs, 3)` if stored per-segment
- `edge_times`, `conf`, `selected_inds` (initial manual selection from disk)
- Properties: `connectivity`, `n_segments`, `segment_names`, `unselected_inds`

### 2.6 Analysis helpers (UI-agnostic, keep in migration)

| Function / class | File | Role |
|------------------|------|------|
| `compute_ccg_panel_data()` | ms_connectivity | Normalize CCG + compute CS in one authoritative pass → `CCGPanelData` |
| `CCGPanelData` | ms_connectivity | 1D normalized trace, baseline, CS scalar, effective lags |
| `apply_norms_to_ccg()` | ms_connectivity | Firing-rate, time-span, baseline normalizations |
| `compute_pair_conn_strength_1d()` | ms_connectivity | CS methods: conv / tailed / global |
| `deconvolve_ccg()` | ms_connectivity | ACG deconvolution for tailed method |
| `plot_ccg_panel()`, `RenderContext` | plotting/ccg | Matplotlib rendering + PNG export |
| `render_ccg_png()` | plotting/ccg | Headless PNG from `RenderContext` |
| Jitter | analyses/jitter.py | Surrogate spike testing |
| Custom CCG | analyses/custom_ccg.py | User-defined time-window CCG compute |

---

## 3. UI layer — module map

**Total:** ~20,437 LOC across 15 Python files in `neuropy/ui/`.

| File | LOC | Responsibility |
|------|-----|----------------|
| `ccg_ui.py` | ~7,847 | Main window, launch entry, **12 embedded Manager classes** |
| `pair_selection_panel.py` | ~2,339 | Left column: `SelectionData`, pair lists, groups, search, Spike Pairs tab |
| `stats_tests.py` | ~2,233 | Toplevel: t-tests on conn strength, firing rates, baselines |
| `time_slider.py` | ~1,558 | Full-width epoch timeline, zoom, custom-segment UI |
| `dialogs.py` | ~1,365 | Merge groups, export, settings, and other modal dialogs |
| `probe_network.py` | ~1,270 | Right column: probe layout + connection arrows (matplotlib) |
| `ccg_mainview.py` | ~845 | Center column: CCG figure, correlogram toggles, baseline/CS, waveforms |
| `custom_ccg_manager.py` | ~765 | Async custom-segment CCG queue, npz I/O, suggestions |
| `jitter_ui.py` | ~503 | `JitterWorker` + `JitterController` (process pool, cache) |
| `utils.py` | ~464 | `GenericUI`, `BackgroundTaskRunner`, `LRUCache`, `UITheme`, widgets |
| `ccg_renderer.py` | ~438 | `CCGContextBuilder`: UI state → `RenderContext` |
| `nd_builder_ui.py` | ~400 | Separate Tk app to build `NeuronsDataset` + `CCGDataset` |
| `pre_generate_images.py` | ~293 | Headless subprocess for batch PNG pre-generation |
| `selection_migration.py` | ~114 | Selection JSON schema v3→v4 migration |
| `jitter_ui.py` (partial) | — | Also defines `JitterQueueDialog` |
| `__init__.py` | 3 | Exports `CCGReviewUI`, `NDBuildUI` |

---

## 4. `CCGReviewUI` architecture

### 4.1 Construction (`__init__(cd, key)`)

1. Bind `CCGDataset cd`, `Key key`, resolve `ccg_ptr = cd.ptr[key]`
2. Lazy-load `ccg_data = cd._ccg[nd_key]` if not in memory
3. Create `SelectionData()`; seed from `ccg_ptr.selected_inds` / `unselected_inds`
4. Instantiate 12 managers (see below)
5. Setup Tk window via `UISetupManager`, pack panels, bind hotkeys
6. Initial draw via `PlotManager.update_plot()`

### 4.2 Layout (historical — may change per user spec)

```
┌─ Menubar + tool strip ─────────────────────────────────────────┐
│ TimeSliderPanel (optional, full width)                           │
├──────────┬─────────────────────────────┬───────────────────────┤
│ Left     │ Center                      │ Right                 │
│ LeftPanel│ CenterPanelContainer        │ NetworkPanel          │
│ Container│  • CCGPlotPanel (matplotlib)│  (probe + arrows)     │
│          │  • CorrelogramPanel toggles │                       │
│          │  • BaselinePanel / CS       │                       │
│          │  • Waveforms, spike attrib  │                       │
├──────────┴─────────────────────────────┴───────────────────────┤
│ Bottom: pair stats, Save / Cancel                                │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Manager classes (all embedded in `ccg_ui.py`)

Each manager holds `self._ui` back-reference to `CCGReviewUI`.

| Manager | Responsibility |
|---------|----------------|
| `SettingsManager` | UI prefs, dark mode, panel visibility |
| `ExportManager` | PNG/PDF export, group export |
| `GroupManager` | Group CRUD, hotkeys 1–0, merge |
| `SelectionPersistenceManager` | Save/load selections JSON, autosave, `.history/` |
| `MultiSessionManager` | "Any session" mode, cross-session pair lists |
| `PNGCacheManager` | PNG render cache, subprocess pre-generation |
| `ConnectionStrengthManager` | CS method (conv/tailed/global/jitter), adaptive test window |
| `PairAnalysisManager` | `_resolve_segment_data`, extended CCG, spike attribution |
| `SimulationManager` | Simulated spike-train CCG preview |
| `UISetupManager` | Window layout, menubar, hotkeys |
| `PlotManager` | `update_plot()` — main render orchestration |
| `MiscManager` | Bookmarks, segment labels, cleanup on quit |
| `PregenController` | Controls pre-generate subprocess lifecycle |

Also external: `CustomCCGManager` in `custom_ccg_manager.py`, `JitterController` in `jitter_ui.py`.

### 4.4 `SelectionData` (not `SelectionDataset`)

Pure state object in `pair_selection_panel.py` — **no Tk dependency**.

```
selected_inds / unselected_inds / deleted_inds   # pair tuples (ref, tgt) or (sess, ref, tgt)
_groups          # {group_name → {session_str → set(pairs)}}
_pair_tags       # {(ref,tgt) → {notes, tags, groups, admitted}}
_group_registry  # v4 schema: int_id → {name, hotkey, notes}
```

`LeftPanel` renders listboxes bound to this state. User migration docs may refer to a future **`SelectionDataset`** abstraction — not implemented yet.

### 4.5 Hypothetical `AnalysisOutput` (target for rewrite)

**Does not exist today.** Current flow scatters data resolution across managers:

```
PairAnalysisManager._resolve_segment_data(ref, tgt, segment, highres)
    → raw ccg, null, pval, acg arrays
compute_ccg_panel_data(...)                          [ms_connectivity]
    → CCGPanelData
CCGContextBuilder.build_context(...)                 [ccg_renderer.py]
    → RenderContext                                  [plotting/ccg.py]
PNG cache path OR matplotlib FigureCanvasTkAgg       [ccg_mainview / PNGCacheManager]
```

**Proposed role for `AnalysisOutput`:** Single immutable view-model per `(pair, segment, display_config)` containing everything needed for both live display (pyqtgraph) and static export (matplotlib). Would replace dual PNG/live paths and reduce manager back-references.

---

## 5. Rendering pipeline (current)

### 5.1 Segment resolution (`PairAnalysisManager._resolve_segment_data`)

Handles three segment modes:
- **Normal:** `cd.ccg[segment, ref, tgt, :]`
- **All segments:** sum over axis 0
- **Custom:** from `CustomCCGManager._custom_segments` npz-backed dicts

Also supports high-res toggle (`cd._ccg_highres`), extended window recompute, neuron ID → array position mapping via `_nid_to_pos`.

### 5.2 Normalization + CS (`compute_ccg_panel_data`)

Order matters:
1. Apply all normalizations **except** `NormalizeBy.BASELINE`
2. Compute CS + `baseline_1d` on pre-subtraction signal
3. Apply `NormalizeBy.BASELINE` if active

### 5.3 UI → render context (`CCGContextBuilder`)

Reads display toggles from `CCGReviewUI` / center panels:
- Line vs bar style for CCG, baseline, ACGs, jitter overlay
- Significance overlays (conv p, p-corrected, test window, jitter p-corrected)
- ACG deconvolution, y-scale, extend window
- Waveform peak channel overlay
- Dark mode, title config

Returns `RenderContext` for `render_ccg_png()` or inline matplotlib draw.

### 5.4 Dual render path (major complexity / bug source)

| Path | When | How |
|------|------|-----|
| **PNG cache** | Default scrolling through pairs | Render to disk PNG via `PNGCacheManager`, display with `ImageTk` / reload |
| **Live matplotlib** | Some toggles, spike attribution, direct canvas | `FigureCanvasTkAgg` in `CCGPlotPanel` |

Pre-generation: `pre_generate_images.py` subprocess reads JSON job file, renders headless with matplotlib Agg.

**Migration opportunity:** pyqtgraph live rendering at ~10 ms may eliminate PNG cache subsystem entirely.

---

## 6. Satellite features (by module)

### 6.1 Probe network (`probe_network.py`)
- `ProbeNetworkData` dataclass — cached layout inputs (positions, pairs, colors)
- `NetworkPanel` — matplotlib scatter + `FancyArrowPatch` connections
- Debounced redraw (`_net_draw_after`); focus pair / zoom / filter toggles
- Uses `neuropy.plotting.probe` helpers

### 6.2 Time slider (`time_slider.py`)
- Behavioral epoch display with themes (segments, sleep, ripple, etc.)
- Zoom handles, custom-segment creation from selected time range
- Integrates with `CustomCCGManager` for on-the-fly CCG recompute

### 6.3 Custom CCG (`custom_ccg_manager.py` + `analyses/custom_ccg.py`)
- Background thread queue (`BackgroundTaskRunner`, max 50)
- Per-session segment list, npz disk cache under `data/custom_ccg/`
- Stacked segment overlay on main CCG plot

### 6.4 Jitter (`jitter_ui.py` + `analyses/jitter.py`)
- `JitterWorker`: multiprocessing pool, LRU cache (500 entries)
- `JitterController`: Tk polling, persistence to `cd._jitter_results`
- UI: run jitter per pair, overlay on CCG, significance checkbox

### 6.5 Stats tests (`stats_tests.py`)
- Separate Toplevel window
- Compare conn strength, CS norms, firing rates, baselines across groups
- Matplotlib bar charts + t-test; export to text file
- Some data types not yet implemented (Peak Width, Peak Center)

### 6.6 Spike pairs / attribution (`ccg_mainview.py`)
- `SpikePairsPanel`, spike attribution raster
- Uses `neuropy.analyses.spike_attribution.find_spike_pairs`

### 6.7 ND Builder (`nd_builder_ui.py`)
- Upstream GUI: configure epochs/segments → build `NeuronsDataset` → `CCGDataset` → launch `CCGReviewUI`

---

## 7. Entry points & persistence

| Entry | Location |
|-------|----------|
| `launch_ccg_review(cd, key=None)` | `ccg_ui.py` — restores last key from `ui_state.json` |
| `CCGReviewUI(cd, key)` | Direct construction |
| Notebook | `notebooks/CCG_gui.ipynb`, `nwb_session.py` examples |
| ND Builder | `nd_builder_ui.py` → `NDBuildUI` |

**Persistence locations:**
| Data | Path |
|------|------|
| Selections | `data/selections/` (JSON, v4 schema) |
| Custom CCG npz | `data/custom_ccg/` |
| UI state | `ui_state.json` (last session, panel toggles, loaded custom CCGs) |
| PNG cache | Per-session cache dir managed by `PNGCacheManager` |
| CCG compute cache | `data/ccg/{config_name}/` sidecars |

---

## 8. Known architectural problems

1. **God object:** `CCGReviewUI` + 12 managers with circular `self._ui` back-refs
2. **Dual render paths:** PNG cache vs live matplotlib — easy to desync on toggle changes
3. **Blurred boundaries:** segment resolution and normalization split across `PairAnalysisManager`, `ConnectionStrengthManager`, `CCGContextBuilder`, and inline UI methods
4. **Tkinter fragility:** ttk Combobox scroll bugs (patched in `ccg_ui.py`), macOS-specific issues
5. **Size:** Feature additions tend to grow managers or `ccg_ui.py` despite design.md guidance
6. **Broken features:** Current implementation has regressions; **user feature descriptions override existing behavior** for migration spec

---

## 9. What to keep vs rewrite

### Keep (analysis + plotting core)
- `neuropy/analyses/neurons_dataset.py` — `Key`, `NeuronsDataset`, config
- `neuropy/analyses/ms_connectivity.py` — `CCGDataset`, `CCGData`, `CCGPointer`, `compute_ccg_panel_data`
- `neuropy/plotting/ccg.py` — `RenderContext`, `plot_ccg_panel`, static export
- `neuropy/plotting/probe.py` — probe layout math
- `neuropy/analyses/jitter.py`, `custom_ccg.py`, `spike_attribution.py`

### Rewrite (UI shell)
- Entire `neuropy/ui/` package → thin Qt views + `AnalysisOutput` view-model
- Target: drop PNG-cache subsystem if pyqtgraph live render is fast enough
- Collapse 12 managers into clearer separation: **state** / **view-model** / **view**

---

## 10. Migration preferences (user-confirmed)

| Category | Choice |
|----------|--------|
| Platform | macOS, Linux, Windows |
| Deployment | pip/conda; no exe v1; no web v1 |
| Strategy | Full rewrite |
| Spec source | User describes features piece by piece |
| Stack (open) | PySide6 + pyqtgraph (leading) + matplotlib export |
| Code goal | Much smaller, easier to maintain, phy-like speed |

---

## 11. Open items for future conversation

- [ ] User feature list (authoritative) — in progress, piece by piece
- [ ] Formalize `AnalysisOutput` / `SelectionDataset` API
- [ ] MVP scope vs full feature parity
- [ ] Which satellite features are v1: jitter, stats, time slider, custom CCG, multi-session, simulation, spike attribution, ND builder
- [ ] Performance priority: CCG scroll vs probe network vs time slider
- [ ] Static export formats: PNG/PDF only, or also Plotly HTML

---

## 12. Quick reference — key classes

```python
# Launch
from neuropy.ui import CCGReviewUI
from neuropy.ui.ccg_ui import launch_ccg_review

ui = launch_ccg_review(cd, key)

# Analysis chain
from neuropy.analyses.neurons_dataset import NeuronsDataset, NeuronsDatasetConfig, Key
from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig, compute_ccg_panel_data

# UI state (no Tk)
from neuropy.ui.pair_selection_panel import SelectionData

# Render bridge
from neuropy.ui.ccg_renderer import CCGContextBuilder  # → RenderContext
from neuropy.plotting.ccg import RenderContext, render_ccg_png
```

---

*When continuing migration planning, read this file first, then ask the user for the next feature piece. Do not treat current UI behavior as spec unless the user confirms it.*
