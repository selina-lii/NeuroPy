# CCG auto-classification — findings (2026-08-03)

## Question
What model could automatically classify CCG pairs the way the user does manually
(via images or arrays), and is there enough labeled data to try it?

## Data audit (`data/project_test2/selections/*.json`)
- 3140 selected (positive) pairs, 379 explicit unselected, 3577 total tagged, across
  16 sessions / 6 rats.
- Usable per-tag counts (n≥100): rhythm, Disinhib, msconn, 2side_dips, 2peakms, leak,
  rift, wideMs, refractory, ok/good/best quality tiers. Tail tags (bimodal, ppinhib,
  inhib2sides, lateMono, pruning, acgPattern, etc.) have <15 examples — too few to
  learn, drop or merge into "other".
- 6 rats means generalization must be checked with leave-one-rat-out CV, not a random
  split — random split leaks via shared neurons within a session/rat.
- No repeat-labeling exists to measure the user's own test–retest consistency; that
  consistency is the real ceiling on any classifier's achievable F1.

## Recommended architecture
1D CNN over the CCG **array** (not a rendered image — rendering throws away
precision the network would have to re-learn). Multi-channel input:
- `[ccg, null]` at minimum — every existing hand-built classifier in
  `ccg_classifier.py` first computes `residual = ccg - null` before extracting
  features/embeddings, i.e. raw CCG alone is insufficient signal.
- Optionally + ACG ref/tgt channels (refractory/burst-type tags may depend on ACG
  shape).
- Optionally + **both resolutions as separate channels** (`ccg_lo, null_lo, ccg_hi,
  null_hi, ...`), since low-res captures overall shape/rhythm and high-res captures
  narrow features like the ±1ms msconn peak — different information, not redundant.
  Caveats: interpolate on the **time axis** (align lag=0), upsample low-res rather
  than downsample high-res (avoids destroying narrow peaks), norm each resolution
  independently before stacking, and mask/flag pairs missing highres data (not all
  labeled pairs have `cd.load_highres()` run) so the network doesn't learn "missing"
  as a feature.

## Existing classifier stubs (`neuropy/analyses/ccg_classifier.py`) — all implemented, none wired to UI
- `CCGClassifier` — 12 hand-crafted shape features → one LogisticRegression per group
  + a trash detector.
- `CCGClusterClassifier` — L2-normalized waveform → PCA → nearest centroid, with
  `plot_embedding()` for visualizing separability. **Zero training cost — use this
  first** as a cheap ceiling check: if tags don't separate in 2D PCA, that's a
  labeling-consistency problem, not an architecture problem, and should be resolved
  before investing in a CNN.
- `CCGTemplateClassifier` — rule-based (`PeakRule`/`GroupTemplate`), no training data
  needed; encodes the user's own peak/lag/width judgment criteria explicitly, useful
  as a feature-engineering checklist for the CNN.
- UI hook exists but is a stub: `neuropy/ui/menubar.py:236` → `ccg_ui.py:410`
  (`_run_classifier` just shows "Classifier not implemented yet.").
- `fit()`/`predict()` contract (`labeled_pairs: dict[str,set[(ref,tgt)]]`,
  `deleted_pairs: set`, `predict(pairs) -> list[ClassifyResult]`) is model-agnostic —
  a CNN can be dropped in behind the same interface, but needs: (a) decoupling from a
  single `CCGData`/session (current classes bind one `ccg_data` at construction — a
  CNN needs to train across all 16 sessions), (b) save/load for trained weights (only
  `CCGTemplateClassifier` has this today), (c) `(session, ref, tgt)` keys instead of
  session-local `(ref, tgt)`.

## Next step (not yet started)
Wire `_run_classifier` to `CCGClusterClassifier` + `plot_embedding()` as the cheap
first pass / ceiling check before any CNN work.

## Prerequisite: multi-dataset loading
Cross-rat training (6 rats, leave-one-rat-out CV) and cross-session PCA/CNN both need
the UI/backend to load and hold multiple sessions' `CCGData` simultaneously — today's
`cd`/`Key` model is built around one active session. This is a separate backlog item,
tracked below.
