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
| name | representation | head |
|---|---|---|
| `conv` | **learned local filters, both resolutions — default** | GB |
| `kernel` | fixed Gaussian-derivative bank at (lag, width, order) | GB |
| `dualres` | whole-trace PCA per resolution | MLP |
| `hybrid` | lowres trace + derivatives + descriptors | MLP |
| `trace` | lowres residual + 1st and 2nd derivative | MLP |
| `shape` | 14 interpretable descriptors | MLP |
| `twohead` | hybrid, quality tiers forced exclusive | MLP |

All operate on the *residual* `(ccg − null) / √null`, never raw counts.
Every model takes `head='mlp'|'gb'`, so representation and classifier vary
independently.

## Results — leave-one-animal-out

Representation × head, mean F1:

| representation | + MLP | + GB |
|---|---|---|
| whole-trace PCA (`dualres`) | 0.429 | 0.431 |
| fixed kernels (`kernel`) | 0.401 | **0.462** |
| **learned filters (`conv`)** | 0.437 | **0.457** |

Lowres-only baselines: `trace` 0.358, `hybrid` 0.354, `twohead` 0.349, `shape` 0.322.

**Shipping default `conv`+GB: F1 0.457, AUC 0.832.** Per label:
rhythm 0.67, best 0.61, leak 0.53, 2peakms 0.51, rift 0.50, 0rhythm 0.50,
Disinhib 0.47, refractory 0.47, good 0.45, ok 0.45, msconn 0.39,
2side_dips 0.28, wideMs 0.11. AUC is strong where it matters —
refractory 0.94, best 0.92, leak 0.92, 2peakms 0.90, rift 0.90.

### The head mattered more than the representation
GB beats MLP on every representation, and by **+0.061** on a fixed kernel bank.
A bank is 1012 correlated columns: an MLP drowns in that, boosted trees select
from it. This is the most likely explanation for why the older hand-tuned
`PeakRule` template classifier disappointed — the representation was reasonable,
the decision rule was not (hard pass/fail thresholds on lag and width, so a pair
0.1 ms outside a window scored zero).

### Learned filters ≈ fixed kernels at this sample size
Over 3 seeds: `kernel` 0.462 ± 0.008, `conv` 0.459 ± 0.003, identical AUC 0.833.
The gap is inside the seed spread. This matches the literature — CoNNECT
([Sci Rep 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8187444/)) learns its
first-layer filters but trains on **~80,000 simulated pairs**; at n=3047 real
ones there is no measurable advantage to learning them.

`conv` ships anyway on secondary criteria: **97 features vs 1012**, ~35% faster,
and more stable across seeds. It learns filters from sliding *patches* rather
than whole traces, so one small filter is reused at every lag and trains on
n_pos× more examples — the CNN's inductive bias at a parameter count this
dataset supports.

### What the literature says
- **CoNNECT** — 1D CNN over CCGs, ~50k params, beat conventional cross-correlation
  and jitter methods; traded wins with GLMCC. Trained purely on simulation.
- **[PLOS Comp Biol 2026 diagnostic study](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1014615)**
  — three CNN failure modes: *amplitude over-reliance* (narrow training data →
  model keys on absolute magnitude, not relational shape), *unimodal
  representation collapse*, and *activity-distribution shift*. Finding:
  **diversity beats quantity** (6 simulated networks → >95% on unseen networks;
  1 network → 71–93%). Our 7 rats with rhythm prevalence swinging 1.7%→72% is
  that narrow-distribution regime.
- Same study: tail-reference normalization helps within-domain but **degrades
  cross-domain generalization**. Our `residual()` divides by `√null`, which is
  that kind of normalization — an ablation worth running.

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
- `leak`: top scores reach p=0.96–1.00 (were capped ~0.35 with lowres alone),
  10/12 true, every panel showing the ultra-sharp peak centered on 0 ms. Direct
  visual confirmation that the fine bins, not extra capacity, earned the +0.36.
- Example panels draw the **highres trace in grey under the lowres blue**,
  rescaled to the lowres peak — plotted raw it is a flat line, since each fine
  bin holds ~1/30 the counts. Visual inspection happens at highres, so the
  verification figures have to show it.

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
