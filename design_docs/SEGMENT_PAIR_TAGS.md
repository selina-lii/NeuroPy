# Per-segment pair tags

## Two kinds of segment-specific pair data

Everything a segment says about a pair falls into one of two shapes, and the
shape decides where it lives.

| | shape | examples | storage |
|---|---|---|---|
| **Per-bin arrays** | `[bin]` per pair | `ccg`, `ccg_null`, `pval`, `qval` | `custom_ccg/<seg>.<sess>.bin_size-<bs>/*.npy` |
| **Per-pair grids** | `[ref, tgt]` per segment | auxiliary test values and verdicts | `selections/aux_tests/<sess>.<ct>.<seg>/*.npy` |
| **Scalars / flags** | one value per pair | manual on/off | `selections/segment_groups.json` |

The EranConv baseline and p-values are per-segment pair data in the same sense
as an auxiliary test result — the same pair scores differently in `maze` than in
`post`. They are not stored with the tags only because an array per pair per bin
does not belong in a json keyed by pair. `pval_corrected` is not stored at all:
it is a property recomputed from `pval` on read, and Bonferroni multiplies by
the full bin count, so the same pair can be significant at lowres and not at
highres.

## The primary rule: `PvalScreeningConfig`

`ccg_transforms.PvalScreeningConfig` declares what the main peak rule is
configured by:

- `roi_start`, `roi_end` — the single ROI, in seconds
- `alpha` — p-value threshold
- `multiple_correction` — `bonferroni` | `fdr_bh`

**Declaration only.** `EranConv` is not routed through it and no computation path
reads it; initial screening still runs where it always ran, out of `CCGConfig`.
`from_conf()` derives the config a `CCGConfig` currently implies, so the two can
be compared rather than drifting apart silently. Wiring it in is a separate
decision, because doing so would change every stored `pval`.

## Auxiliary tests

Advisory flags, never validity tests: a pair that fails one is still available
everywhere. `ConnectionStrength.AUX_TESTS` maps each name to its function,
threshold argument, default, comparison direction, and tunable kwargs;
`aux_test()` returns `(value, passed)` per pair.

| test | threshold | other args |
|---|---|---|
| `mean_over_max_tail` | `min_ratio` 1.0 | `factor` |
| `mean_over_avg_tail` | `min_ratio` 1.0 | `factor` |
| `peak_percentile` | `top_pct` 5.0 | — |
| `peak_to_baseline_ratio` | `min_ratio` 1.0 | — |
| `spike_count` | `min_count` 2.5 | `scope_ms`, `mode` |

`peak_percentile` is a *top* percent — smaller is stronger — so it passes below
its cut while the others pass above it. The tail tests read `conf.tail_intervals`
live, so an incoming value overrides the default.

A ratio with an empty denominator is `NaN`, not `inf`: no baseline means the
result is unknown, not that the pair dominates. NaN fails every comparison,
which is the safe direction — without this, a pair holding a single spike in the
entire CCG scores as a strong connection.

`spike_count` is extracted from `EranConv.spkcount_mask`, which is left exactly
as it was. At its defaults it reproduces that mask pair-for-pair. `mode` selects
`avg` (what screening does), `every` (every bin in the window clears the cut), or
`one` (any single bin does).

`peak_to_baseline_ratio` is new rather than moved. `acg_features` and
`window_features` in `classifier/features.py` compute something similar, but they
are live classifier inputs and their baseline is the outer thirds (`_flanks`),
not `conf.tail_intervals` — merging them would change classifier numbers.

## Tag storage

Swept results go to `AuxResults`, one unit dir per (session, conn type, segment)
under `selections/aux_tests/`, holding `[ref, tgt]` arrays:

```
<rule>.value.npy    float    the measured statistic
<rule>.passed.npy   bool     that rule's verdict
on.npy              bool     AND of every enabled rule, manual overrides applied
```

A complete grid, not a member list: every pair has an entry whether it passed or
failed, so a failure is distinguishable from an unswept pair, and a re-sweep
replaces the whole segment instead of deleting stale entries. Values sit beside
the booleans so a threshold can be re-read without re-sweeping, and each rule is
stored separately so it is visible which one rejected a pair. A segment is a
directory, so adding or dropping one never reshapes anything.

Hand-assigned tags go to `SegmentGroups`, a second registry keyed by segment.
Nothing automated writes there — a sweep writes `AuxResults` only, so the
override tier stays exactly what a human set.

## SegmentGroups

`GroupDataset` keys membership `(session, ref, tgt)` — no segment, which is
correct for what it holds: `best` and `rift` describe what a pair *is*, and that
does not change between segments. On/off describes whether a pair holds *in a
given stretch of time*, so it needs the segment.

`SegmentGroups` subclasses `GroupDataset` and overrides one hook, `member()`, to
key `(session, segment, ref, tgt)`. `BiIndex` never inspects the tuple, so
everything else — the registry, metadata, hotkeys, `Group` itself — is inherited
unchanged. There is one registry, not one per segment: `best` is a single entry
whose members carry their own segments.

`on` and `off` are ordinary entries in it, so a hand tag and an automated sweep
result are the same kind of thing, differing only in who wrote them. Any other
tag can be segment-scoped too.

It saves to `<project>/selections/segment_groups.json` — inside the project,
unlike `groups.json`, because segment names are a project's own. Membership is
serialized with the registry, since no per-session file hosts it.

**`all` is not special.** It is dim0[0] and an ordinary label everywhere:
`segment_names()` lists it, `segment_index()` maps it to 0, and `SegmentGroups`
stores a tag against it exactly as against `maze`. Nothing branches on it.

`_SelectionData.seg_off` was removed; `SegmentGroups` is the single source.
Nothing was migrated because no session file on disk had ever written one.

## Which tests decide on/off

`AuxTestConfig` (enabled names + per-test args) is stored on
`CCGConfig.aux_tests`, in the `derived` group, so it survives a restart.
`cd.aux_test_config` is the only accessor. The Manage Groups "on" page drives
it with one checkbox per test; unchecked tests render greyed and are not
computed.

A pair is on when every checked test passes. A test that cannot be measured
counts as failed, never backfilled. With no test checked, a pair is on until a
manual tag says otherwise.

## Hotkeys

The segment registry is consulted before the session-wide one, so a
segment-scoped tag takes the key. Assigning a key already held by a session-wide
group is refused rather than allowed to shadow it silently. The keypress path
returns before reaching `apply_group_toggle`, so session-wide tagging,
list ordering and cursor movement are untouched.

## Staleness

These results are derived from the CCG and the config. Nothing currently detects
that a stored tag predates a recompute of the segment it describes. Until a
stamp is added, a re-sweep after any CCG or config change is the only guarantee.
