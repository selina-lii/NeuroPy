# Initial rule sets + per-key display windows

Goal: the initial screen (what lands in pointers / Available) is defined only by rule sets tagged
**initial**; the built-in screen and its config-only knobs go. Display windows (test window, tail,
alpha) follow the governing rule set per key, with Defaults as fallback.

## Pass 1 — initial sets drive the screen

**Data model (`pair_selection_data.py`)**
- `RuleSet.initial: bool = False`, `RuleSet.display: bool = False` (display used in pass 2);
  serialized under the set; `from_dict` defaults both False.
- `RuleBook.initial_sets()` / `later_sets()` — in sidebar order.
- `rule_plan(key)` (Assign tags) iterates `later_sets()` only.

**Screen (`ms_connectivity.py`)**
- `EranConv.screen_mask(ccg, p, excitability)` → `CCGDataset.screen_inds(key, ccg_data)`:
  ordered AND over covering initial sets, reusing `sweep_rule_sets` on the 'all' segment
  (sets as plan, mask = covered); uncovered pairs and "no initial set" → all type pairs pass.
- `eranconv()` keeps the convolution; its index building calls the new screen per conn-type key.
- Delete `Rules.screen`, `EranConv.screen_mask`, `CCGConfig.alpha2` (+ `_options`, migration line).
- `_build_ptrs` / rerun carry `ptr.picked` across; after the rerun, selected pairs not screened →
  `picked` (shows the existing **admitted** pill).

**Constraints**
- Initial sets: scope = session + conn type only (pair-rule scope hidden: tags derive from pointers).
- Initial rules: lowres only (resolution combo locked to the screen's resolution).

**UI (`dialogs.py` SignificanceRulesPanel)**
- Right-click set → "Mark as initial" / "Unmark initial" (and "Use for display" in pass 2).
- Sidebar: Defaults, initial sets (row text carries an "initial" tag), sash, later sets,
  "+ New rule set", sash, Assign tags. `moved` stays within each section.
- Initial set page hides the Pairs scope field and the resolution combo.

**Tests (no writes; config/pointers backed up + cmp'd)**
1. Initial E set (p_value above, ROI 1–3 ms, Bonferroni, + spike_count) and I set (p_value below,
   whole CCG, neighbor_alpha = alpha2, + spike_count) reproduce today's 8065 screened pairs on test2.
2. No initial sets → pointer = all type pairs.
3. Assign tags skips initial sets; rule_results labels contain no initial set.

## Pass 2 — per-key display windows

**Resolver (`ms_connectivity.py`)**
- `Windows(roi, tail, tail_source, alpha, correction)` frozen dataclass.
- `cd.windows_for(key) -> Windows`: governing set = covering `display` set, else first covering
  initial set, else Defaults. From it: first enabled p_value rule (else first rule with roi) → roi,
  alpha, correction; first rule with tail → tail, tail_source; unset/"follow default" → Defaults.
  Cached on (rule_book identity, key.ptr()).

**Reads switched to `windows_for(key)`** (mechanical; per key, so batch math is unchanged)
- `app_state.is_significant` (lb/ub, threshold), `chip_color` fallback.
- `ccg_panel`: test-window shading (each ROI interval), `cs_window` (union span), tail shading,
  `_pval_row`, extend cache key, `_refit_conv` correction.
- `ccg_transforms`: `_bins`, `conn_strength` window, `baseline_tail` default intervals — take
  windows as args instead of reading conf (callers pass `windows_for(key)`).
- `jitter.py` (lb/ub, alpha), `ccg_classifier.py` (lag window, spike count) — take a `Windows`.
- `CCGData.pval_corrected` (conf.alpha/multiple_correction) — correction comes from the key's
  windows when the corrected array is built.

**Kept on CCGConfig as Defaults only**: alpha, min_lag, max_lag, tail_intervals, min_spkcount,
spkcnt_scope, multiple_correction (the "follow default" source). `set_significance_window` /
`set_tail_window` (CCG panel controls) edit Defaults.

**Tests**: E and I keys of one session show their own sets' windows; Defaults fallback with no
tagged sets; CS grid and sig chips equal pass-1 values when windows equal Defaults.

## Critical rerun (after pass 1, user go-ahead, GUI closed)
1. Fresh timestamped backup: pointers, config, selections (minus rule_results), segment_groups,
   groups.json, project_plan.json; ccgdata lowres+highres dirs being overwritten.
2. Recompute lowres + highres CCG/p for all 16 sessions; screen via initial sets; save pointers.
3. Union selected-but-unscreened pairs into `picked`; diff selections before/after (must be equal).
4. Clear `rule_results`; user reruns Assign tags.
Custom/extend CCGs untouched (independent of screen; only stale if key params change).
