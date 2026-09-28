# Persisted extend CCGs — storage layout

## Why

Extend CCGs are recomputed from spike times on every cache miss
(`_extend_cache = LRUCache(32)`, `ccg_panel.py`), per pair, in memory only. Only the row
settings survive a restart, via `extend_state`. Computing extend once for all pairs × all
sessions and keeping it means the expensive part is paid once.

## The axis question

Extend **cannot** live on the custom-CCG dim0 segment axis:

- `attach_segment` appends with `np.concatenate(..., axis=0)`, which requires dims 1–3 to
  match.
- `compute_segment` reuses the *parent's* conf, so a custom segment shares the main CCG's
  bin count by construction.
- Extend varies exactly `duration` and `bin_size` (`for_extend`).

So custom CCG = a new dim0 index at fixed (window, bin); extend = a new (window, bin) unit
at fixed dim0. **Perpendicular axes.**

## Layout

`(window, bin)` is already a *directory* key rather than an array axis, so extend reuses
that precedent exactly:

- `CCGSourceConfig.data_dir(resolution, bin_size)` →
  `custom_ccg/<segment>.<session>.<bin_token>`
- `_CCGData.save` already records `n_bins` per file because it varies.

A persisted extend CCG is therefore its own unit directory holding a full
`[seg, ref, tgt, bin]` array, parallel to `lowres`/`highres`, and free to carry its own
dim0 stack later:

```
<project>/extend_ccg/<name>/<session>.<window>ms.<bin_token>/
    ccg.npy          # [seg, ref, tgt, bin] for every pair in the session
    meta.json        # {name, session, segments, window_ms, bin_ms, n_bins,
                     #  n_pairs, n_segments, computed_at}
```

`<name>` is what the per-row save button prompts for, so one named extend set spans every
session; the window/bin tokens keep two different extends from colliding inside it.

Verified on real data: `RatU_Day4SD` at 200 ms / 2 ms stores `(1, 140, 140, 101)` —
all 140 neurons, bins exactly `window/bin + 1` — and reads back identical.

## Segment coverage

**All segments that exist** for the session: `compute_extend_session` reads
`data.segment_names` and computes one dim0 entry each, windowing by the segment's own
`t0`/`t1` when it has them. A set therefore goes stale if a new custom segment is added
later; recomputing under the same name overwrites it.

## API (`CCGDataset`, `ms_connectivity.py`)

- `extend_dir` / `extend_unit_dir(name, session, window_ms, bin_ms)` — path policy
- `compute_extend_session(name, key, window_ms, bin_ms, segments=None)` — computes and
  stores; **headless**, so the queue worker calls it directly
- `extend_ccg_for(name, key, window_ms, bin_ms)` — mmapped array, or None
- `saved_extends()` — one row per stored unit, for the load dialog

## Queue

Extend joins the existing custom-CCG queue rather than starting a second one: `CCGTask`
gained an `extend` field, and `CustomCCGManager.queue_extend` enqueues one task per
session under a shared batch id, so progress reports one x/total across both kinds of work.

The queue cap (`max_ccg_queue`) is now **unrestricted by default** (0), configurable in
Settings → Cache; `BackgroundTaskRunner.enqueue` treats 0 as no limit.

## UI

- Per extend row, beside `+`/`−`: a save button prompting via `prompt_name` (`utils.py`).
- A load button next to the extend rows, modelled on `CustomCCGManageDialog`
  (`dialogs.py`), which already does by-session / by-name tabs and the disk scan.
- The time slider's 💾 is **removed** — verified to save nothing (it opened
  `CustomCCGManageDialog` without `select_mode`, i.e. the same dialog minus Load; saving
  happens implicitly inside `attach_segment`). The 📂 load button stays.
