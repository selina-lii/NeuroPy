# Result 2 — design choices made without asking

Decisions taken while running the sweep and the binarized stats, with the
reasoning, so they can be overruled cheaply.

## 1. `tests={}` means *no* tests, `tests=None` means *all*

`aux_tests_for` took a dict and treated a missing one as "run everything". An
empty dict then also ran everything, so an `AuxTestConfig` with nothing enabled
silently behaved like one with all five enabled. `None` now means all (the
convenience default) and `{}` means none.

With no test enabled a pair is **on**. The alternative — off — would make an
unconfigured project report every pair as disconnected, which is worse than
reporting the status quo. `_combine_on` returns an all-true grid in that case
rather than failing on `np.logical_and.reduce([])`.

## 2. Unmeasurable counts as failed, never backfilled

A ratio whose denominator is empty is `NaN`, not `inf`. NaN fails every
comparison, so such a pair does not pass. Without this a pair holding a *single
spike* in the whole CCG scored as a strong connection: 121 such pairs in
`maze`/RatU_Day4SD passed `mean_over_max_tail` before the fix, 0 after, and the
maximum value fell from 6.7e11 to 2.03. NaN is written to JSON as `null`.

## 3. The enabled-test set is saved to `CCGConfig.aux_tests`

'on' is meaningless without knowing which rules produced it, so the choice is
stored in the `derived` group of the project config and travels with it.
`cd.aux_test_config` is the only accessor. The sweep script saves `conf` after
running, otherwise a later process would re-derive 'on' from a different rule set
than the one that wrote the tags.

## 4. `peak_percentile` is a *top* percent

5.0 means "in the top 5%", so smaller is stronger, and it passes when the value
is **below** its cut while the other four pass above. `AUX_TESTS` carries a
per-test direction flag rather than forcing every test to share one comparison.

## 5. `spike_count` mirrors screening but never touches it

Extracted from `EranConv.spkcount_mask` as a read-only parallel path. At its
defaults it reproduces that mask pair-for-pair (6789 vs 6789, masks identical on
RatU_Day4SD/maze). `spkcount_mask`, `significance_mask` and `build_inds` have an
empty diff. `mode` adds `every` (min over the window) and `one` (max) beside the
screening `avg`.

## 6. `peak_to_baseline_ratio` was written new, not moved

`acg_features` and `window_features` in `classifier/features.py` compute
something similar, but they are live classifier inputs and their baseline is the
outer thirds (`_flanks`), not `conf.tail_intervals`. Moving either would change
classifier numbers. The new one uses `baseline_tail`, so all five auxiliary tests
share one baseline definition.

## 7. Segment-scoped tags are a second registry, not a widened first one

`GroupDataset` keys membership `(session, ref, tgt)`. Adding a segment to that
key would touch 57 call sites across 11 files — most of which (classifier,
network, stats rows) have no segment to supply — and would require migrating 4322
existing tagged pairs on a guess. `SegmentGroups` subclasses it and overrides one
hook, `member()`, to key `(session, segment, ref, tgt)`; `BiIndex` never inspects
the tuple, so registry, metadata and hotkeys are inherited unchanged.

There is **one** registry, not one per segment: `best` is a single entry whose
members carry their own segments. `'all'` is an ordinary label in it, exactly like
`maze`.

## 8. `seg_off` was deleted rather than kept beside the new tags

It stored the same claim (manual on/off for a pair in a segment) that
`SegmentGroups` now stores. Keeping both is the parallel-backend case CLAUDE.md
forbids. Nothing was migrated because no session file on disk had ever written a
`seg_off` entry — verified by grep across every `data/*/selections/*.json`.

## 9. Binarized CS is a metric, not a mode switch

`CS binarized (on/off)` is an entry in `METRICS` carrying `off_mode='binarize'`,
so it appears in the ordinary data-type picker and no caller needs a new flag.
`Metric` gained one field. `collect_row` passes `m.off_mode` through to
`get_cs_values_for_sess`, which previously dropped it — the metric would have
silently returned plain CS without that fix.

## 10. Manual overrides are applied inside the stats path

`get_cs_values_for_sess` passes `sd.seg_overrides(session, segment)` to
`get_conn_strength_for`, so a hand-set on/off wins over the tests in stats
exactly as it does on the chip. A stats run and the GUI therefore cannot disagree
about whether a pair is on.

## 11. Jitter percentiles adapt to njitter

Only 5/95 were kept before. The ladder is now 25/50/75 always, plus 5/95 at
njitter >= 100, 2.5/97.5 at >= 200, and 1/99 at >= 1000 — a tail needs enough
draws to sit on, and 1/99 from 100 trials is one draw. Raw jitter values are
still not stored; the bands are what survives.

## 12. Chunking, not pooling, and threads over chunks

`jitter_chunk=5` and `n_threads=0` (one per core) are defaults on `JitterConfig`.
Cost was superlinear in the pooled spike count, so the old "one big call" was the
worst case; see `JITTER_NOTES.md`. Verified bit-identical to the old path.

## 13. Batch jitter keys on (session, conn-type, segment), group is only a filter

You can run jitter for a group, but the result is stored by segment, not by
group. Two consequences, both intended: running `good` after `best` **merges**
into the same store rather than replacing it, and a pair jittered under one group
is not recomputed when a later group includes it. `run_and_store` does a
`dict.update`, so the newer run wins for pairs present in both — the right way
round when the second run used more jitters.

## 14. The batch dialog runs synchronously behind a progress dialog

`Modules > Jitter > Run batch…` blocks with a modal progress dialog rather than
going through the existing single-pair process queue. That queue is built around
one pair per process and would throw away the batching win — `_group_by_target`
jitters each target once for all its refs. A 327-pair batch is a deliberate
action, not a background nicety, so blocking is honest about what it costs. The
per-batch line is printed to the terminal so progress is visible.

## Open, not decided here

- **The test window does not contain the peak** for 7 of 8 pairs in
  `RatJ_Day1 pyr-pyr`, so jitter p-values there are ~1.0. Per instruction this is
  treated as a window/screening question, not a jitter one: the jitter null was
  instead validated against the convolution baseline (r = 0.85-0.96, mean ratio
  0.994-1.001), which it matches.
- **`post_tenth_sleep-only` still exists on disk** (10 segments x 16 sessions)
  despite the D1 deletion, and its configs now carry the `brainstates` filter.
  They were evidently regenerated. The sweep includes them as instructed.
