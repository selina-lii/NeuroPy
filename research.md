# NeuroPy CCG System — Deep Code Review

*Prepared 2026-03-14, ahead of planned feature work.*

---

## 1. Architecture Overview

### Class Hierarchy

```
Key  (frozen dataclass — session, epoch, ref_ind, target_ind, segment, excitability, conn_type)
 │
CCGConfig  (resolution, bin_size, duration, conv_window, alpha, conn_types, …)
 │
 ├── NeuronsDataset  (sessions → Neurons + edge_times + segment_firing_rates + ProbeGroups)
 │       └── prep()  — filter neurons by type/epoch, segment by time/stride/spikecount
 │
 ├── CCGData  (dataclass — ccg[seg,ref,tgt,bin], ccg_null, pval, qval, significant, conn_strength)
 │       └── get_conn_strength()  — PEAKSIZE or TAILED (FFT deconvolution)
 │
 ├── CCGPointer  (sparse index — _inds 2D/3D, edge_times, connectivity dict)
 │       └── inds2 (unique pairs), inds3 (segment-expanded), split(), get_segment()
 │
 └── CCGDataset  (orchestrator)
         ├── _ccg[Key]         — low-res CCGData (1 ms bins)
         ├── _ccg_highres[Key] — high-res CCGData (0.1 ms bins, on-demand)
         ├── data[Key]         — CCGPointer (significant pairs)
         ├── spurious[Key]     — CCGPointer (wrong-type significant)
         ├── _jitter[Key]      — Jitter (transient, non-persistent)
         └── nd                — NeuronsDataset reference

EranConv  (stateless significance engine — Stark & Abeles 2009)
    ├── _conv()            — Poisson convolution test
    ├── multiple_correction() — Bonferroni / FDR-BH
    ├── significance_mask()   — E (peak) vs I (trough+neighbor)
    └── eranconv()            — full pipeline → CCGPointers + pvals
```

### UI Architecture (ccg_ui.py, ~2300 lines)

```
CCGReviewUI(cd: CCGDataset, key: Key)
 │
 ├── Menubar: Panels │ Groups │ Selections │ Help
 ├── Tool-strip: Session/Type dropdowns
 ├── Group hotkeys bar (dynamic Ctrl+1..0)
 │
 ├── 3-column PanedWindow
 │   ├── LEFT (350px): Pair Selection
 │   │   ├── Available Listbox (EXTENDED selectmode)
 │   │   ├── Selected Listbox (EXTENDED selectmode)
 │   │   ├── Select All / Resolution toggle
 │   │   └── Right-click context menu (move, group tag, pair tags)
 │   │
 │   ├── CENTER (flex): CCG Display
 │   │   ├── Significance controls (foldable: baseline, pval, test window, jitter)
 │   │   ├── Correlogram controls (foldable: CCG/ACG tri-state, Y-scales)
 │   │   ├── Normalization panel (firing rate, time span, same-scale modes)
 │   │   ├── Jitter panel (njitter spinbox, Run/Clear buttons)
 │   │   ├── Waveforms panel (hidden by default)
 │   │   ├── Segment chips row (real + "All" + custom)
 │   │   └── matplotlib FigureCanvas (PNG-cached CCG rendering)
 │   │
 │   └── RIGHT (340px): Probe Network
 │       ├── Neuron/Pair focus inputs
 │       ├── Connection type toggles (P→P, P→I, I→I, I→P)
 │       ├── Display toggles (arrows, hide unconnected/same-ch/same-shank, ch IDs)
 │       ├── Group filter dropdown
 │       ├── Zoom sliders (H/V)
 │       └── matplotlib scatter + FancyArrowPatch arrows
 │
 └── Bottom bar: Statistics + Save/Quit
```

### Data Flow

```
                    ┌─────────────────┐
                    │ NeuronsDataset   │
                    │ (spike trains,   │
                    │  edge_times)     │
                    └────────┬────────┘
                             │ prep()
                             ▼
                    ┌─────────────────┐
                    │spike_correlations│ ← expensive O(n² × n_spikes)
                    └────────┬────────┘
                             │
                     ┌───────┴───────┐
                     ▼               ▼
              ┌────────────┐  ┌────────────┐
              │ CCGData    │  │ CCGData    │
              │ (low-res)  │  │ (high-res) │ ← on-demand
              │  1ms bins  │  │  0.1ms bins│
              └──────┬─────┘  └────────────┘
                     │
                     ▼
              ┌────────────┐
              │  EranConv   │ ← Poisson convolution + multiple correction
              └──────┬─────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
       ┌───────────┐ ┌───────────┐
       │CCGPointer │ │CCGPointer │
       │  (sig)    │ │ (spurious)│
       └─────┬─────┘ └───────────┘
             │
             ▼
       ┌───────────┐     ┌────────┐
       │CCGReviewUI│ ──► │ Jitter │ ← on-demand per pair
       └───────────┘     └────────┘
```

### Cache Strategy

Two independent caches with JSON metadata for invalidation:

| Cache | File | Invalidated by | Metadata file |
|-------|------|----------------|---------------|
| **Compute** (raw CCG arrays) | `*_ccgdata.hkl` | COMPUTE_FIELDS (duration, bin_size, conv_window, resolution, symmetrize, conn_types) | `.compute.meta.json` |
| **Significance** (pair indices) | `*_ccgpointers.hkl` | SIGNIF_FIELDS (alpha, min/max_lag, min_spkcount, multiple_correction) | `.signif.meta.json` |
| **High-res** | `*_highres.hkl` | None (relies on parent config) | — |
| **PNG display** | `tmp/pair_*.png` | 18-bit filename key encoding all toggle states | — |

---

## 2. Confirmed Bugs

### BUG-1: Jitter cache not keyed by resolution

**Location:** `ccg_ui.py:131, 1574, 2643`

**Problem:** `_jitter_cache` is keyed by `(ref, tgt)` only. The `_toggle_resolution()` method clears the entire cache (line 1129), which means:
- Run jitter at lo-res → cache populated
- Ctrl+R to hi-res → cache cleared (jitter results lost)
- Ctrl+R back to lo-res → cache cleared again (still lost)

The user loses jitter results on every resolution toggle. The fix is to key by `(ref, tgt, resolution)` so both resolutions can coexist.

**Impact:** Data loss (user must re-run jitter after any resolution toggle). Will become worse when jitter runs as background tasks (planned feature).

**Fix:** Change cache key to `(ref, tgt, 'hi'/'lo')`. Update `_toggle_resolution` to not clear the cache. Update all cache reads to include current resolution.

---

### BUG-2: `routine_eranconv_connection_info()` crashes on call

**Location:** `ms_connectivity.py:2415-2416`

```python
total_by_EI['E'] += neurons['E']['total']   # line 2415
total_by_EI['I'] += neurons['I']['total']   # line 2416
```

**Problem:** `neurons` is a `Neurons` object (from `nd.data.items()`), not a nested dict. `Neurons` doesn't support `neurons['E']` indexing. This function will raise `TypeError` if ever called.

**Impact:** Dead code — standalone function at module bottom, not imported/called anywhere. But misleading if someone tries to use it.

**Fix:** Replace with `neurons.get_neuron_type('E').n_neurons` (matching the pattern at line 2413), or delete the function entirely.

---

### BUG-3: `all_inds` property recomputes on every access — O(n) per call, ~40 call sites

**Location:** `ccg_ui.py:202-213`

```python
@property
def all_inds(self):
    base = self.ccg_pointer.inds2
    admitted = self._group_pairs(_ADMITTED_GROUP)
    if not admitted:
        return base
    base_set = set(map(tuple, base))
    extra = sorted(admitted - base_set)
    if not extra:
        return base
    return np.vstack([base, np.array(extra, dtype=base.dtype)])
```

**Problem:** Every access triggers `set(map(tuple, base))` (O(n_pairs)) + sorted set difference + `np.vstack`. Called in tight loops at lines 3502, 3593-3594, 3664-3670 (each doing `set(map(tuple, self.all_inds))` — double conversion).

**Impact:** Performance degrades quadratically with pair count when iterating. Currently acceptable at <5k pairs but will compound with future features (shape classification iterates all pairs, time slider creates many custom segments).

**Fix:** Cache the result; invalidate on admitted-group changes. Add `_all_inds_cache` attribute, clear on group mutation.

---

### BUG-4: PNG cache name collision for custom segments

**Location:** `ccg_ui.py:2510`

```python
seg_name = f"custom_{cs['name']}_{cs['t0']:.0f}_{cs['t1']:.0f}"
```

**Problem:** Two custom segments with the same name and times that round to the same integer → identical PNG path → stale image displayed for wrong segment.

**Impact:** Low probability currently. Will increase with time slider expansion (many custom windows, possibly auto-generated with similar bounds).

**Fix:** Include the custom segment index (`ci`) in the filename, or use higher time precision (`.2f`).

---

## 3. Redundancies

### REDUNDANCY-1: Unused code in ms_connectivity.py

| Item | Line | Status |
|------|------|--------|
| `from copy import deepcopy` | 24 | Imported, never used |
| `CCGConfig.cache_key()` | 341-379 | SHA256 method, never called anywhere in codebase |

Both can be safely removed.

### REDUNDANCY-2: Triple metadata validation

Config validation happens independently in three methods:
- `_check_metadata()` (line 1297) — generic checker
- `load_ccgdata()` (line 1360) — validates compute fields inline
- `load_ccgpointers()` (line 1421) — validates signif fields inline

Each reads the JSON, deserializes, and compares field-by-field. Could be unified into a single `_check_metadata(suffix)` call (which already exists but is underused).

### REDUNDANCY-3: Defensive `getattr` guards in ccg_ui.py

~20 instances of this pattern:
```python
getattr(self, '_acg_ref_var', None) and self._acg_ref_var.get()
```

All these Tk variables are initialized in `__init__` before any code path that reads them. The `getattr` guards are unnecessary and add visual noise. They likely exist from incremental development where features were added before `__init__` was updated, but are now stale.

---

## 4. Feature Readiness Assessment

### CCG Shape Classification (Priority 1)

**Data availability:** Excellent.
- `CCGData.ccg[seg, ref, tgt, :]` gives raw shape vectors for any pair/segment
- `CCGPointer.inds2` iterates all significant pairs efficiently
- `conn_strength` (PEAKSIZE, TAILED methods) already extracts shape features
- `ccg_null` provides baseline for normalized shape comparison

**Gaps to fill:**
- No storage slot for classification labels in CCGPointer or CCGData — need to add a field (e.g., `cluster_id[n_seg, n_ref, n_tgt]` or per-pair dict)
- No UI for classification results — need new panel or overlay
- Clustering should be per-session (different recording quality → different distributions)
- Consider whether classification runs on raw CCG or normalized (active_norms changes shape)

**Key existing code to reuse:**
- `CCGData.get_conn_strength()` (ms_connectivity.py:1018-1113) — feature extraction
- `CCGData.save_plots()` (ms_connectivity.py:938-1190) — batch PNG generation for visual inspection
- `EranConv._conv()` static method for baseline-subtracted shapes

---

### Spike Attribution (Priority 2)

**Data availability:** Partial.
- Neurons objects hold spike trains (partially in memory)
- Bin → time offset: `bin_index * bin_size_eff - conf.duration / 2`
- For a given ref-tgt pair and bin, can identify which ref spikes had a tgt spike at that lag

**Stage 1 (spike timing retrieval):**
- Straightforward: given ref spike times and tgt spike times, find pairs where `tgt_time - ref_time` falls in clicked bin's time window
- This is an O(n_spikes) scan per click — acceptable
- Could precompute a spike-pair index per pair for instant lookup

**Stage 2 (.dat waveform retrieval):**
- No current .dat file infrastructure
- Need: session → .dat file path mapping, channel count, sampling rate
- ~500 GB per file → **must** use partial reads (`np.memmap` or `mmap`)
- Per user direction: will explore options later, memory-mapped may be tricky

**Key architectural decisions needed:**
- Where to store .dat file metadata (extend CCGConfig? new config class?)
- Waveform snippet size (e.g., ±1 ms around spike = 60 samples at 30 kHz)
- How to handle missing .dat files gracefully in the UI

---

### Time Slider Module (Priority 3)

**Current state:**
- Canvas-based slider with draggable start/end handles
- `_compute_custom_segment()` runs full `spike_correlations()` per window
- Custom segments store both lo-res + hi-res CCG arrays
- Segment chips provide navigation

**What needs to change for multi-track display:**
- Current slider is single-track (epoch bounds only)
- Behavioral data (speed, licking) needs new data loaders — no current infrastructure
- Ephys events (ripples, SWR) need event detection pipeline integration
- Macro-scale states (REM/NREM) need state-segmentation output

**Per user direction:** Don't touch CCG computation path. The expensive `spike_correlations()` per window is correct and working. Future optimization is deferred.

**Key constraint:** The time slider module should be self-contained (importable independently of CCGReviewUI) to allow reuse in other tools.

---

### Jitter as Background Tasks (Priority 4)

**Current state:**
- `_on_run_jitter()` blocks UI thread (line 1568-1569)
- `update_idletasks()` provides partial refresh but no cancellation
- Jitter is transient (not saved to disk from UI, though JitterDataset has save support)
- Results stored in `_jitter_cache` dict (per-pair, not per-session)

**What needs to change:**
- Background thread/process for jitter computation
- Progress reporting to UI (progress bar, ETA)
- Cancellation support
- Result persistence (disk cache keyed by pair + resolution + njitter)
- Memory management: njitter=5000 × n_bins × n_pairs can overflow RAM
- Notification system when task completes

**Existing infrastructure to leverage:**
- `JitterDataset.run_jitter()` (jitter.py) — batch runner, could be adapted for async
- Jitter saves to `data/jitter/<key>/` as `.npy` files — persistence format exists

**Design consideration:** Python's GIL limits threading for CPU-bound work. Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`. Tkinter's `root.after()` for polling results from the main thread.

---

### NeuronDataset Module (Low Priority)

**Current state:** NeuronsDataset (~300 lines) lives in ms_connectivity.py, tightly coupled to CCGDataset through shared Key objects and config references.

**Extraction path (when needed):**
1. Move NeuronsDataset + NeuronsDatasetConfig to `neuropy/core/neurons_dataset.py`
2. Keep Key as shared type (already generic)
3. CCGDataset holds reference via `self.nd` — interface stays the same
4. GUI wrapper would call `NeuronsDataset.prep()` with user-selected parameters

**Not blocking other features** — can be done independently.

---

## 5. State Management Assessment

### Strengths
- **Toggle pattern:** All UI toggles live in Tk BooleanVar/StringVar — no redundant Python attributes. Callbacks just redraw.
- **`active_norms` @property:** Always reads live from `norm_vars` — no stale state.
- **Significance source of truth:** Always uses lo-res `_ccg` for significance (green chips persist in hi-res mode).
- **Undo/redo:** 30-item cap prevents unbounded growth.
- **Custom segments:** Store both lo-res and hi-res so Ctrl+R works seamlessly.

### Weaknesses
- **No atomic selection save:** `_save_selection_version()` writes JSON without file locking. Two UI instances on the same session → last-write-wins.
- **Undo stack per type:** Doesn't survive type switch. Users may expect global undo.
- **Legacy group formats:** v1.0 (flat set) → v2.0+ (per-session dict). Loading v1.0 groups into a different session assigns all pairs to file's session.

---

## 6. Performance Profile

| Operation | Cost | Frequency | Concern Level |
|-----------|------|-----------|---------------|
| `all_inds` property | O(n_pairs) | ~40 call sites, some in loops | **Medium** — compounds in tight loops |
| `_render_png` | O(1) per pair-seg (cached) | On navigation | **Low** — PNG cache effective |
| `_draw_network_impl` | O(n_types × n_pairs) | On pair select | **Medium** for large networks |
| `refresh_lists` | O(n_pairs) rebuild both listboxes | On any list change | **Medium** at 10k+ pairs |
| `_compute_custom_segment` | O(spike_correlations) | On time slider "Set" | **High** — full recompute, but per user direction: don't optimize |
| Network arrow picker | Always enabled on all arrows | Per matplotlib event | **Low** waste — should be focus-mode only |
| PNG cache growth | Unbounded during session | Toggle changes accumulate | **Low** — `_clear_all_png_cache()` on major toggles |
| `set(map(tuple, self.all_inds))` | O(n_pairs) conversion | ~8 call sites | **Medium** — redundant with all_inds already being an array |

---

## 7. Recommendations Before New Features

### Must fix
1. **BUG-1:** Key jitter cache by resolution to prevent data loss on Ctrl+R
2. **BUG-3:** Cache `all_inds` to prevent O(n²) performance in loops

### Should fix
3. **BUG-4:** Add custom segment index to PNG filename
4. **REDUNDANCY-1:** Remove unused `deepcopy` import and `cache_key()` method
5. **REDUNDANCY-3:** Remove unnecessary `getattr` guards (clean up ~20 sites)

### Can defer
6. **BUG-2:** Fix or delete `routine_eranconv_connection_info()` (dead code)
7. **REDUNDANCY-2:** Unify metadata validation (works correctly, just duplicated)
8. Atomic selection saves (only matters if running multiple UI instances)

---

## 8. Key Patterns to Preserve in Future Work

These patterns are intentional and should be maintained:

1. **Toggle state in Tk vars only** — no redundant Python attributes
2. **Significance from lo-res only** — `self.cd._ccg.get(self.key.nd())`, never `self.ccg_data`
3. **bin_size_eff inferred from array shape** — `conf.duration / (n_bins - 1)`, robust to conf mutation
4. **Custom segments store both resolutions** — enables seamless Ctrl+R
5. **PNG cache encodes full display state** — reproducible renders
6. **`_on_norm_toggle` and `_on_sig_toggle` both call `_clear_all_png_cache()`** — consistent invalidation
7. **180ms debounce on pair select** — prevents double-click artifacts
8. **Two-level cache split (compute vs signif)** — allows re-running EranConv without recomputing spikes
