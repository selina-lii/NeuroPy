# CCG classifier — first pass results (2026-08-16)

Built per `CCG_CLASSIFIER_ML.md`. Code in `neuropy/classifier/`, UI at
Modules ▸ Classify ▸ Run classifier.

## Data
3129 tagged pairs / 16 sessions / 7 animals, joined to their CCG traces with
**zero** missing pointers and zero out-of-bounds pairs. 13 labels have enough
support (n ≥ 60, ≥ 4 animals) to train on.

**The problem is multi-label, not multi-class.** 1560 of 3129 pairs (50%) carry
two or more labels. So every model predicts each label with its own sigmoid, in
parallel — a softmax would force 'rhythm' and 'good' to compete when they in fact
describe different axes.

Quality tiers behave differently from shape labels: 1427 pairs carry exactly one
of best/good/ok and only 14 carry two, i.e. they are near mutually exclusive
while shape labels co-occur freely. `twohead` exploits this; it did **not** help
(see below), so `hybrid` remains the default.

## Models
| name | input |
|---|---|
| `dualres` | **both resolutions, separate embedding heads — default** |
| `hybrid` | lowres trace + derivatives + descriptors |
| `trace` | lowres residual + 1st and 2nd derivative |
| `shape` | 14 interpretable descriptors (peak SNR, lag, width, slope, 0 ms bin, rhythm index…) |
| `twohead` | hybrid, quality tiers forced exclusive |

All operate on the *residual* `(ccg − null) / √null`, never raw counts.

## Results — leave-one-animal-out
| model | mean F1 | mean AUC |
|---|---|---|
| **dualres** | **0.429** | **0.795** |
| trace | 0.358 | 0.703 |
| hybrid | 0.354 | 0.727 |
| twohead | 0.349 | 0.709 |
| shape | 0.322 | 0.699 |

Best labels (dualres): rhythm 0.70, Disinhib 0.53, best 0.52, leak 0.51,
refractory 0.50, rift 0.49. Worst: wideMs 0.17, 2side_dips 0.26.

## Why both resolutions (the biggest single win)
Lowres is 21 bins at 1 ms; highres is 601 bins at 1/30 ms. **Both span the same
±10 ms** — they are different views of one interval, not different extents. So
padding lowres out to 601 would misalign lag 0 and destroy exactly what the fine
bins exist to carry.

Each resolution instead gets **its own scaler + PCA embedding head** (24
components), and the two embeddings fuse with the scalar descriptors from both.
Separate heads are necessary, not cosmetic: highres bins hold ~1/30 the counts of
lowres bins, so one shared scaler over the concatenation would let 601 highres
columns swamp 21 lowres ones by sheer count rather than by information.

`n_components` = 24 by sweep: F1 0.408 / **0.429** / 0.385 at 16 / 24 / 40.

Gains land precisely on the sub-millisecond labels, which is the evidence that
highres is contributing signal and not just capacity:

| label | F1 lowres → dualres | why |
|---|---|---|
| `leak` | 0.15 → **0.51** (+0.36) | "ultra-sharp peak at 0ms" — one bin at 1 ms, a shape at 1/30 ms |
| `rift` | 0.20 → **0.49** (+0.30) | narrow central gap, invisible at 1 ms |
| `rhythm` | 0.59 → **0.70** (+0.11) | |
| `2peakms` | 0.27 → **0.37** (+0.10) | "two sharp peaks (<.5ms wide)" — unresolvable in lowres |
| `refractory` | 0.42 → **0.50** (+0.08) | 0 ms dip; AUC 0.89 |
| `msconn` | 0.34 → **0.38** (+0.05) | |

Three labels lose, all broad/quality ones where fine bins only add noise:
`ok` −0.04, `good` −0.07, `2side_dips` −0.08. Worth revisiting by letting those
labels read the lowres head preferentially.

All 16 test2 sessions have highres on disk, so no missing-data masking is needed.
A set missing highres for any sample falls back to lowres for **every** sample, so
the nets can never learn "missing" as a feature.

### Per-label thresholds matter more than the architecture
Naive 0.5 cut → mean F1 **0.25**. Calibrated per-label thresholds → **0.35**.
Every optimal threshold lands well below 0.5 because the labels are rare.
Thresholds are fit on training folds only, so the number above is honest.

## How good is this really?
`hybrid` on a random split instead of by-animal: **F1 0.46 / AUC 0.81** — which
`dualres` now nearly matches (0.429 / 0.795) while still generalizing across
animals. The remaining gap is *labeling drift between animals*, not shape difficulty —
`rhythm` prevalence runs from 1.7% (RatU) to 72% (RatS), a 40× swing. RatU is
the single largest fold (1198 pairs) and barely uses the label.

**The ceiling here is label consistency, not model capacity.** Two concrete
things would raise it more than any architecture change:
1. Re-label a held-out ~100 pairs to measure your own test–retest agreement —
   still never measured, and it bounds everything.
2. `0rhythm`/`rhythm` overlap 45%, `wideMs`/`rhythm` 43%. These are not cleanly
   separable classes as currently tagged; merging or sharpening them would help.

## Visual verification (`data/project_test2/classifier/`)
Figures are the real check, not the F1 table.
- `examples_<label>.png` — top-scoring pairs per label. **'best' verified: every
  top-scoring pair shows one prominent sharp peak just after 0 ms with a clean
  pre-peak dip**, which is what the label is supposed to mean.
- `mean_shapes.png` — labeled vs. predicted mean residual per label. All 13
  track closely, so the nets learned the shape rather than a confound.
- `feature_separation.png` — Cohen's d per feature per label. Confirms the
  features encode the group notes' own definitions: `leak` and `msconn` peak on
  the 0 ms bin, `rhythm` on the rhythm index, `refractory` is the strongest
  separation in the figure (d = −2.0 on the 0 ms bin, i.e. a central dip),
  `best` = high peak SNR + high slope + *narrow* width.
- `wideMs` diagnostic: its top-scoring panels look homogeneous but are mostly
  `true=n`, and all come from one animal → sparse/inconsistent tagging, not a
  model failure.
- `leak` under `dualres`: top scores are now p=0.96–1.00 (were capped ~0.35 with
  lowres alone), 10/12 true, every panel showing the ultra-sharp peak centered on
  0 ms. Direct visual confirmation that the fine bins, not extra capacity, are
  what earned the +0.36.

## How the UI handles tentative labels
Predictions never write into group tags on their own — otherwise the next
training run learns from its own output. They go into a `PredictionStore`
(`predictions.json`), and the review dialog:
- lists pairs **least-confident first** (confident ones need no attention),
- filters by label or to the current session,
- shows the selected pair in the main CCG view,
- promotes to real group tags only on **Accept**.

## Notes
- Runs on sklearn MLPs, not torch. At n=3129 a small MLP is right-sized, and the
  dual-resolution result shows the returns are in *input representation*, not
  model capacity. The interface is model-agnostic if a CNN is wanted later —
  `fit(ccg, null, Y, ccg_hi, null_hi)` / `predict_proba(...)`.
- Full pipeline: ~1–2 min for CV + figures, 8283 pairs scored.
- Both models train on the identical 3047 samples (3129 tagged pairs minus those
  whose only labels fell below the min_count/min_rats floor), and **zero** samples
  lack highres — so the F1 comparison above isolates the input representation.
