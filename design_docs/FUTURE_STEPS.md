# Future Steps — NeuroPy CCG UI

Unbuilt features. Current code runs complete without these (stubs have no callers).
Created 2026-07-27.

## 1. Jitter persistence → JitterDataset
- New `JitterDataset` owning `_jitter_results` + `save`/`load`.
- `jitter_ui`: drop `self.cd._jitter_results` / `hasattr` guards → use `self.jd`.
- Current: jitter results not saved, silently dropped.

## 2. Batch-request store (cd-owned)
- Implement `cd.save_batch_request(req, kind)` / `list_batch_requests(kind)` / `delete_batch_request` (currently `raise NotImplementedError`).
- suggest: curated ≤20. history: append-only jsonl. Store conf relpath for recovery.

## 3. Dissolve CustomCCGState
- `load/save/update_suggestions` → delegate to cd batch-request store.
- Remove local `suggested_custom_ccgs.json` ownership.
- `_emit_inventory_event`: compare `cd.saved_customs` (segment inventory), not JSON.
- `SuggestedCCGDialog` reads `cd.list_batch_requests('suggest')`.

## 4. Ticket owner = CustomCCGManager
- Ticket creation/tracking + queue-custom logic moves into the manager.

## 5. Integrity report (deferrable)
- Implement `cd.check_segment_integrity(key, label)` — verify each resolution's npy set on disk.
- `cd.ccg_config_relpath(key)` — srcconf → conf link.

## 6. meta data/conf link fields (B remainder)
- meta json stores `ccg_config_relpath` + `data.{res}.dir_relpath` for integrity checks.

## 7. Stats collect_group ignores multi conn_type
- `collect_group` (stats_tests.py): `conn_type = cfg.conn_types[0] if cfg.conn_types else ''` — only first conn_type used, `cfg.conn_types[1:]` dropped.
- `cfg.conn_types` is a list (picker allows multi-select) but collection loops sessions/segments/groups, not conn_types.
- Fix: add `for conn_type in (cfg.conn_types or [''])` loop alongside the seg/grp loops so each selected conn_type gets its own ptr key + collection.

## 8. Port Simulation to Qt (dropped in tkinter→Qt cutover)
- Never migrated: `219370e1` dropped it wholesale; no Qt equivalent exists. Last intact at `ee34d49f`.
- Recover `SimulationDialog` (`ee34d49f:neuropy/ui/dialogs.py:195`) + `SimulationManager` (`ee34d49f:neuropy/ui/ccg_ui.py:2626`, `_sim_mgr` at :3989).
- Generation math lived inside the UI file — move it to `analyses/` on port, leave the dialog thin.
- Re-wire under `Modules > Simulation > New simulation…` (was top-level `Simulation` in the spec).
- Spec: `neuropy/ui/CCGUI Display.md:258` — simulated-CCG area, Run / Export / Import simulation settings.

## UI todos (next)
- ~~Stats test segment not refreshing~~ DONE — picker now reads cd.available_segments (disk ccgdata).
- ~~Time slider "select none" button~~ DONE.
- Stats tests plot does not fill width/height.
- Stale overlay: re-running with a different ratio does not clear the previous plot.
- Disk-only segment lazy-load on access (segment_index falls back to 0 when a disk segment isn't in memory).
- Move CS-grid to backend: new `cd.conn_strength_grid(key, active_norms, baseline_method, cs_metric)` builds `ConnStrengthConfig` internally (only construction site is stats_tests.get_cs_values_for_sess). stats + ccg_panel (2 sites) call it; UI keeps baseline_method/cs_metric as nav toggle state.

## UI cleanup (separate branch — 10-item plan)
- items 1–8, 10 approved. item 9 (drop `legend_toggles` from save_state) pending: verify per-theme state fully reconstructs toggles before removing.
