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
| `shape` | 14 interpretable descriptors (peak SNR, lag, width, slope, asymmetry, 0 ms bin, rhythm index…) |
| `trace` | residual + 1st and 2nd derivative, full trace |
| `hybrid` | both — **default** |
| `twohead` | hybrid, quality tiers forced exclusive |

All operate on the *residual* `(ccg − null) / √null`, never raw counts.

## Results — leave-one-animal-out
| model | mean F1 | mean AUC |
|---|---|---|
| shape | 0.322 | 0.699 |
| trace | 0.358 | 0.703 |
| **hybrid** | **0.354** | **0.727** |
| twohead | 0.349 | 0.709 |

Best labels (hybrid): rhythm 0.59, best 0.50, Disinhib 0.49, ok 0.46,
refractory 0.42. Worst: wideMs 0.09, leak 0.15, rift 0.20.

### Per-label thresholds matter more than the architecture
Naive 0.5 cut → mean F1 **0.25**. Calibrated per-label thresholds → **0.35**.
Every optimal threshold lands well below 0.5 because the labels are rare.
Thresholds are fit on training folds only, so the number above is honest.

## How good is this really?
Same model, random split instead of by-animal: **F1 0.46 / AUC 0.81**.
The 0.11 F1 gap is *labeling drift between animals*, not shape difficulty —
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

## How the UI handles tentative labels
Predictions never write into group tags on their own — otherwise the next
training run learns from its own output. They go into a `PredictionStore`
(`predictions.json`), and the review dialog:
- lists pairs **least-confident first** (confident ones need no attention),
- filters by label or to the current session,
- shows the selected pair in the main CCG view,
- promotes to real group tags only on **Accept**.

## Notes
- Runs on sklearn MLPs, not torch — torch install hit a full disk, and at n=3129
  a small MLP is the right size of model anyway. The interface is model-agnostic
  if a CNN is wanted later.
- Full pipeline: ~6 s to fit, ~1 min for CV + figures, 8283 pairs scored.
