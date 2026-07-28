# Bug queue — ranked by importance (2026-07-28)

Per bug: data class / how modified / interface endpoints. Bullets ≤5 words.

---

## B1 — seg_idx resolved on one CCGData, indexed on another (CRASH ×2)
`IndexError: index N out of bounds for axis 0 with size 1` —
`stats_tests.get_cs_values_for_sess` and `ccg_panel._cs_value` → `cd.pair_slices`.

- Data class: `CCGData` (dim0 = segments)
- Modified by: `attach_segment` / `load_segment` / `drop_segment`
- `segment_index` lazy-loads into lowres `CCGData`
- Caller then indexes highres `CCGData`
- `_load_segment` loads one resolution only
- Endpoints: `cd.segment_index`, `cd.pair_slices`, `cd.get_conn_strength_for`, `nav.segment_index`
- Fix: `_load_segment` on the requested key's resolution; index and array from same `CCGData` fetch.

## B2 — All-session mode uses anchor session key
IndexError on `pair_slices` when clicked pair is from another session.

- Data class: `Key` (session field)
- Modified by: `ccg_panel._cs_value` builds Key
- Hardcodes `nav.key.session`, ignores pair's
- Pair handle already carries its Key
- Endpoints: `_cs_value`, `_together_handle`, `build_context`
- Fix: branch on all-session mode; use pair's own Key.

## B3 — All-session custom CCG runs current session then stalls
Other sessions get no config, no data; queue blocked.

- Data class: `CCGBatchRequest`, `CCGSourceConfig`
- Modified by: `parse_ccg_batch_request`, worker
- Requested session lacks main CCG
- Lazy-load skips it → stall
- Endpoints: `time_slider._on_set`, `cd.ccg_for`, `cd.compute_segment`, `cd.attach_segment`
- Fix: on custom-CCG request, load that session's main CCG first; never lazy-skip.

## B4 — No runtime task-queue viewer for custom CCG
Cannot see pending/running custom-CCG tasks, cannot cancel one.

- Not `CustomCCGManageDialog` (that is saved-config save/load, 💾/📂)
- Data class: custom-CCG worker queue
- Modified by: `time_slider` Set enqueues
- Queue never exposed to UI
- No per-task cancel/delete
- Endpoints: custom-CCG runner, jitter queue manager (pattern to copy)
- Fix: queue view like jitter's, with delete-task.

## B5 — Bottom bar n-significant ≠ n available
- Data class: `CCGPointer` (pairs, significant)
- Modified by: selection panel, stats bar
- Two counts computed independently
- significant should = available ∩ selected
- Bottom-bar CS not via `ConnectionStrength`
- Endpoints: `nav.all_pairs_np`, `ptr.pair_set`, `ccg_panel` stats bar
- Fix: single count source on `AppState`.

---

## Deferred (fix at end)
- B6 load-stats-result dialog needs delete button
- B7 p-corrected line to front
