# Why classifier F1 sits at 0.4 while AUC reaches 0.9

Measured 2026-08-20/21 on `project_test2`: 4223 pairs, 17 labels. Starts from
model `kernel`; the current state is model `rule`.

## The measurements

| split | bias | mean F1 | mean AUC |
|---|---|---|---|
| leave-one-rat-out | discover | 0.357 | 0.820 |
| pooled random 5-fold | discover | **0.436** | **0.892** |
| pooled random 5-fold | balanced | 0.402 | 0.892 |

Two things follow.

**The rat split was costing ~0.08 F1.** Label rates vary up to 20x between
animals — `rhythm` is 442/1138 in RatN but 17/2075 in RatU; `0rhythm` is 278 in
RatN and 0 in RatR. Leave-one-rat-out therefore trains on one prior and tests on
another. Since the labels describe the CCG alone and are independent of which
animal produced it, pooling is the correct evaluation.

**Thresholds are not the bottleneck.** `discover` (beta=2) and `balanced`
(beta=1) give *identical* AUC (0.892). The bias only trades precision against
recall along one fixed ranking. No threshold recovers what the ranking lacks.

## The actual bottleneck: features cannot express the labels

The per-label pattern is unambiguous — labels score well exactly when the
feature set contains the thing that defines them.

| label | F1 | AUC | defining feature | present? |
|---|---|---|---|---|
| rhythm | 0.72 | 0.92 | FFT periodicity | yes (`rhythm_index`) |
| best | 0.67 | 0.97 | see below | 1 of 5 criteria |
| refractory | 0.62 | 0.96 | central dip | yes (`trough_snr`) |
| 2peakms | 0.55 | 0.93 | two peaks + dip between | **no** |
| burst | 0.23 | 0.73 | 1-3ms peak + second peak >5ms | **no** |
| triple | **0.00** | 0.85 | three peaks | **no** |

`shape_features` computed a single global `argmax`. **The concept "two peaks"
did not exist anywhere in the feature set**, so `2peakms`, `burst` and `triple`
were being asked to be learned from evidence that had been discarded before the
model ever saw it. `triple` scores exactly 0.00.

Note `triple` still has AUC 0.85 — the traces do differ, the ranking is weakly
right, but nothing crosses a threshold.

### The `best` rule, criterion by criterion

The user's stated definition, against what was measured:

| criterion | measured before? |
|---|---|
| peak overlapping 1-3 ms | partly — `ms_peak` used ±1 ms, the wrong window |
| very high spike count (~100+/bin lowres) | **no** — `residual` divides rate out |
| Poisson-shaped peak | **no** |
| continuous in high-res | **no** |
| no other artifacts (refractory, sorting) | **no** — an ACG property |

AUC 0.97 on roughly one of five criteria suggests the rule is highly learnable
once the other four are measured.

## What was added

1. **`peak_features`** (17 cols) — ported from the hand-written rules in
   `analyses/ccg_classifier.py`, which already did prominence-based peak finding.
   Peak count, top-3 lag/height/width, inter-peak spacing and its regularity,
   and `dip_between_peaks`. Smoothing is sigma 0.4, not 1.0: at 1.0 two peaks
   2 bins apart merge, which is precisely the `2peakms` shape.

   Synthetic check — the feature separates the rules as stated:

   | shape | n_peaks | separation | dip |
   |---|---|---|---|
   | 2peakms | 2 | 2.0 ms | **41.1** |
   | burst | 2 | 6.0 ms | 4.0 |
   | triple | 3 | 3.0 ms | 3.7 |

2. **`window_features`** (19 cols) — named millisecond windows (1-3, 0-1, 3-5,
   5-10 ms), each signed, each in both normalized and raw counts. The rules are
   stated in milliseconds, so the windows are named rather than left for a
   kernel bank to approximate. Raw counts carry the spike-count floor.

3. **`acg_features`** (7 cols/neuron) — the autocorrelograms are the diagonal of
   the CCG matrix, dropped by the pointer as self-pairs but present in the loaded
   array at no cost. Refractory depth, 0 ms fill, burst shoulder, firing level.
   "No obvious sorting artifacts" is an ACG property and was invisible.

4. **`deconvolved`** — CCG with each neuron's autocorrelation divided out in the
   frequency domain, regularized. Separates "broad coupling" from "bursty
   reference neuron".

## Where it ended up

Mean F1 over the ten labels with n >= 200, pooled random 5-fold, `discover`:

| configuration | mean F1 | mean AUC |
|---|---|---|
| kernel, leave-one-rat-out | 0.417 | 0.849 |
| kernel, pooled | 0.514 | 0.897 |
| rule features, pooled | 0.554 | 0.909 |
| ~~rule + invented mask~~ | ~~0.623~~ | 0.886 |
| rule, untagged-in-complete-slices as negatives | 0.566 | 0.910 |
| **+ flank dips, two sigmas (0.4, 1.2)** | **0.572** | **0.912** |

The last two rows are the only ones scored on the data as the reviewer marked
it, and are **not** comparable to the masked rows above, which were scored on a
filtered subset. Their mean AUC is the highest of the set: the F1 drop against
0.623 is a harder task (1494 more negatives to reject), not a regression.

### Flank dips and the two-sigma sweep, measured

+0.006 mean F1, +0.002 AUC — at the edge of fold noise. The per-label direction
is the informative part:

| label | dF1 | dAUC | |
|---|---|---|---|
| msconn | **+0.04** | +0.010 | the ±0.5 ms rule; moved on both, so real |
| 2peakms | +0.03 | 0.000 | |
| rift | +0.02 | 0.000 | |
| **2side_dips** | **−0.02** | +0.010 | *the label the flank features were written for* |

`msconn` gained most, and gained on AUC too, so it is not a threshold artifact —
the 0-0.5/0.5-1 ms window split and the sharp sigma are doing real work.

**`2side_dips` got worse.** The feature written specifically for it made its
label harder to predict. The implemented geometry — deepest point either side of
the global argmax, half-depth width, symmetry ratio — does not match what the
reviewer means by "soft inhibitory dip flanking a peak at ~1 ms scale". This is
the concrete instance of the general risk in this whole section: rules
transcribed from verbal descriptions cannot be validated without the reviewer.

Change kept (net positive, `msconn` gain real), but hand-written feature work
stops here pending the reviewer's input.

Per label at n >= 200: rhythm 0.73, best 0.69 (AUC **0.97**), Disinhib 0.64,
good 0.62, 2peakms 0.55, 0rhythm 0.55, rift 0.54, ok 0.47, 2side_dips 0.45,
msconn 0.42.

The same model on leave-one-rat-out scores 0.399 — the gap is the per-animal
label-rate imbalance, not something features can close.

### Where the gains actually came from

| change | delta mean F1 |
|---|---|
| pooling instead of leave-one-rat-out | +0.09 |
| tagged-only / negative handling | +0.05 |
| hand-written rule features | +0.04 |

**The data decisions outweigh the features roughly 3:1.** Hand-written geometry
is also the least verifiable part of the pipeline — the rules are transcribed
from verbal descriptions with no way to check the implementation matches the
reviewer's eye. Recommendation: pause further hand-written rules pending the
reviewer's judgement, and spend effort on label coverage instead.

## Result: `rule` vs `kernel`, same pooled split, same bias

| label | n | kernel F1 | rule F1 | delta |
|---|---|---|---|---|
| triple | 46 | 0.00 | 0.12 | **+0.12** |
| 5-13 | 33 | 0.03 | 0.10 | +0.07 |
| pos_skew | 27 | 0.19 | 0.26 | +0.07 |
| burst | 26 | 0.23 | 0.27 | +0.04 |
| leak | 129 | 0.52 | 0.55 | +0.03 |
| best | 220 | 0.67 | 0.69 | +0.02 |
| Disinhib | 502 | 0.61 | 0.63 | +0.02 |
| msconn | 379 | 0.41 | 0.43 | +0.02 |
| **mean (17)** | | 0.436 | **0.459** | +0.023 |
| **mean (n>=200)** | | 0.514 | **0.554** | +0.040 |
| mean AUC | | 0.892 | 0.900 | +0.008 |

The labels that gained most are exactly the peak-count labels the new features
were built for, which confirms the diagnosis. The gains on the rare ones are
real but small in absolute terms — at n=26..46, roughly 5-9 positives per test
fold, there is not enough to calibrate a threshold against. **Per the user's
direction, these are not worth chasing further**; the effort belongs on labels
with enough support to train on.

## The second data problem, and a wrong answer to it

Counted over the 4223 samples: 2729 carry at least one tag, **1494 carry none**.

### What the labels actually are

All 17 labels are **shape tags**. `best`/`good`/`ok` grade the monosynaptic
shape — they are not a separate quality axis running orthogonal to the others.
There is one axis, and every tag on it describes what the CCG looks like.

Per the reviewer, the three shape rules most recently specified:

| rule | definition |
|---|---|
| `msconn` | sharp defined peak within ±0.5 ms |
| `wideMs` | may overlap `msconn`; cross-session inconsistency is expected |
| `2side_dips` | soft inhibitory dip flanking a peak at ~1 ms scale; auxiliary |

### Untagged pairs inside a complete slice are real negatives

The `complete` flag on `_SelectionData` marks a (session, conn-type) slice as
exhaustively reviewed. **27 of 64 slices are marked complete.** Inside one, an
untagged pointer pair means "reviewed, no meaningful pattern" — a genuine
negative for every label. `build_labeled_set` already converted those into
1494 explicit negatives.

### The mask that should not have existed

An earlier pass inferred, from clustering concentration and tag co-occurrence
counts, that quality and shape were two independent axes with different blank
semantics, and built `LabeledSet.trainable()` — a per-(sample, label) mask —
threaded through `fit`, `_calibrate`, and both benchmarks.

Two things were wrong with it:

1. **The premise was invented.** The reviewer had already said "monosynaptic";
   that was read as a *gate on when grading applies* rather than as a
   description of what was being graded.
2. **It discarded the reviewer's own work.** The mask dropped exactly the 1494
   untagged pairs the `complete` flags had just promoted to negatives, nullifying
   the marking of 27 slices.

It was also worth nothing when measured: 0.623 masked vs 0.608 with a plain
tagged-only filter — inside noise. The whole apparatus was deleted.

### A diagnostic worth keeping

The first version of that mask reported mean F1 0.527, a large apparent gain.
Two signals contradicted it:

- mean AUC **fell** 0.900 → 0.877 while F1 rose;
- `good` went F1 0.61 → 0.68 while its AUC collapsed 0.92 → **0.79**.

**Rising F1 with falling AUC means the test set got easier, not that the model
improved.** F1 moves when the scored population changes; AUC is a property of the
ranking. Any change to what counts as a scorable example should be checked
against AUC before its F1 is believed.

## Clustering (the second strategy)

`neuropy/classifier/cluster.py` groups CCGs in the **rule-feature space** rather
than over raw traces. This matters: raw-trace distance is dominated by
amplitude, while the hand-written rules turn on peak structure. On synthetic
data with three planted shapes it recovers all three at 100% purity.

On real data the clusters are visually coherent — cluster 2 (146 pairs, 79%
`best`) is uniformly sharp-peak-with-flanking-dips; cluster 0 (748 pairs, 61%
`rhythm`) is uniformly oscillatory, including two members with no tags at all.

**The label-spread table is the more useful output:**

| label | concentration | across clusters |
|---|---|---|
| best | 0.52 | 23 |
| refractory | 0.49 | 13 |
| rhythm | 0.41 | 19 |
| ... | | |
| ok | 0.16 | 24 |
| good | 0.13 | 29 |

`good` and `ok` smear across nearly every cluster while `best` concentrates.
All three are shape tags grading the monosynaptic shape, so the spread is not
an axis difference — it says `best` has a tight, specific visual definition
while `good` and `ok` cover a wider range of appearances. **This spread was
what the invented quality/shape axis was wrongly inferred from.**

Bulk-labeling value is limited at present purity: only `rhythm` (+128) and
`Disinhib` (+19) gain, and both are already well-supported. An earlier run
suggested `leak` +61, but that cluster's members were mostly quality-graded —
`cluster_suggest.py` now requires the shape tag itself to have >= 10 votes
before proposing it.

## Open

- **37 of 64 slices are not marked complete.** Their untagged pairs contribute
  nothing — neither positive nor negative. Marking more slices is the single
  highest-value manual action available, and costs one decision per slice.
- `msconn` (F1 0.42, AUC 0.87) and `2side_dips` (0.45 / 0.89) are the weakest
  well-supported labels — a feature gap rather than a threshold problem.
  `flank_dip_features` and the two-sigma sweep target exactly these two;
  their effect is measured separately.
- `ok` has AUC **0.81**, the only well-supported label below 0.86 and well
  below `best` at 0.97. As the bottom grade of the b/g/o scale it is where the
  reviewer's own boundary is likely softest — worth asking about before
  modelling it harder.
- `best` at AUC 0.97 with F1 0.69 is almost entirely threshold loss. The
  criterion is learnable; the operating point is the open question.
- `I-I-flip` appears as a 48-pair cluster at 100% shape purity but is below the
  min_count/min_rats bar, so it never enters training.

## Fixed along the way

`predict_project` never passed `arrays` to `predict_proba`, so a model trained
with the ACG block hit `'NoneType' object has no attribute 'acg_ref'` on every
prediction. The prediction path now builds a `PairArrays` carrying the ACGs off
the matrix diagonal (`ccg[r, r]`, `ccg[t, t]`), the same source training uses;
`RuleNet._encode` additionally zero-fills the block for any caller that cannot
supply them, so the feature width a fitted model expects is never narrowed.
