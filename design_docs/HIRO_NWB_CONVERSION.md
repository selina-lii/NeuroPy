# Hiro MATLAB export → NWB: field coverage checklist

Source `~/Documents/ms_synchrony/hiro` (4.9 GB, 33 `.mat` files) →
`~/Documents/ms_synchrony/hiro_nwb` (2.6 GB, 27 `.nwb`, one per session).

Reader `neuropy/io/datasets/hiro/reader.py`, writer `neuropy/io/nwbwriter.py`,
driver `notebooks/convert_hiro.py`.

**The dataset is NOT completely translated.** Spikes, epochs, brain states,
events, behaviour and spectrograms are converted and verified. 14 files of
precomputed analysis results are deliberately not converted — they are
derived quantities, recomputable from what was converted. Details below.

## Verified conversions

Checked numerically on RoyMaze1, and unit/spike totals checked on all 27
sessions (`mismatches: none`).

| source field | → NWB | check |
|---|---|---|
| `spikes.<ses>.time` (µs) | `units.spike_times` (s) | ratio exactly 1e6; 27/27 sessions match spike-for-spike |
| `spikes.<ses>.quality` | `units.quality` + `neuron_type` | `q<4→pyr`, `q==8→inter`, else `unclassified` — all three verified true |
| `spikes.<ses>.isStable` | `units.is_stable` | `all(a,b,c)` per the PDF's example |
| `spikes.<ses>.id` `[a,b]` | `shank_id`, `cluster_id` | both arrays equal element-wise |
| `behavior.<ses>.list` rows 1–3 | `intervals['brainstates']` | 668 rows = 668 raw; `1→nrem 2→rem 3→quiet 4→active`; start matches to 1e-9 s |
| `behavior.<ses>.time` | `intervals['paradigm']` | 3 cols → pre/maze/post; sleep's 1 col → the condition name |
| `ripple.<ses>.time`/`peakTime` | `intervals['ripple']` (+`peak_time`) | row counts match |
| `spindle.<ses>.HPC.time`/`peakTime` | `intervals['spindle']` | row counts match |
| `basics.<ses>.SpkGrps.Channels` | electrode table, one group per shank | 8 shanks × 8 ch |
| `basics.<ses>.Ch.CA1Shanks` | electrode `location` = `CA1` | 1-based → 0-based |
| `basics.<ses>.SampleRate` | `Neurons.sampling_rate` | 30 kHz |
| `position.<ses>.t/x/y` | `acquisition['position']` | 7 wake sessions |
| `speed.<ses>.t/v` | `acquisition['speed']` | separate clock from position (3.2 M vs 0.97 M samples) |
| `pfcEeg` / `spectrum` `Pxx/freq/time` | `DecompositionSeries` | 83 bands; `DecompositionSeries` not `ElectricalSeries` because the export ships no raw trace |

## Not converted — precomputed analysis results

Recomputable from the spikes/epochs above; none feed the CCG pipeline.

| file | holds |
|---|---|
| `wake-firing`, `wake-trisecFiring`, `wake-eventFiring` | binned population rates, split pyr/inh |
| `wake-eventRate`, `wake-trisecEvent` | per-event counts/rates/durations |
| `wake-participationEachCell/EachEvent/Rate` | ripple & spindle participation |
| `wake-modulatedCell` | firing-rate modulation p-values |
| `wake-onOff` | UP/DOWN state detection |
| `wake-HL`, `wake-HLfine` | high/low LFP-state segmentation |
| `wake-stateChange` | state-transition tables (nrem2rem etc.) |
| `*-pfcSWA`, `wake-hpcSWA` | slow-wave amplitude/slope/troughs |
| `sleep-deltaBand`, `sleep-rippleBand` | band energy/power |
| `sleep-spikes_check` | duplicate of `sleep-spikes` (v7 copy) |

Also dropped, per-unit: `isoDist`, `isoDistSep`, `stability`, `ampChange`,
`meanF`, `stdF`, `meanFREpoch`, `FRfractionPrePost`, `StablePrePost` — cluster
quality metrics beyond the `quality`/`isStable` pair the PDF says to gate on.
`basics.Ch.{CA1theta, CA1thetaCand, Gamma, bestRipple, ripple, spindle, emg}`
(per-band best channels) and `SpkGrps.{PeakSample, nFeatures, nSamples}`
(clustering params) are likewise not carried.

## Discrepancies found in the source

1. **`RamboSleep1`** — a 5th animal present in `sleep-spikes`/`sleep-behavior`
   but absent from `basics`, `ripple`, `spindle`. Converted with 22 units and
   brain states, no electrodes or events. `_Absent` in the reader exists for this.
2. **`KevinMaze1` has 4 paradigm blocks**, not the documented 3. The 4th is
   labelled `block3` rather than guessed at.
3. **`saved_at` vs mtime** — unrelated to this export, but noted while working:
   metadata timestamps run 1 h behind file mtimes (timezone).
4. **No per-unit `peak_channel`** — the export carries shank only, so
   `NWB_DEFAULT`'s `peak_channel` maps to nothing. `FIELDS` in
   `datasets/hiro/__init__.py` omits it.
5. **Two MATLAB vintages** — `wake-spikes.mat` and `sleep-spikes_check.mat` are
   v7 (whole-file `scipy.io` read, 1.9 s); the other 31 are v7.3 (h5py, 0.6 s).
   Not a blocker; it is why conversion is one-time rather than on demand.

## Session inventory

27 sessions, 5 animals. Sleep/rest 20 (Kevin 4, Rambo 1, Roy 7, Steve 1, Ted 7),
wake/maze 7 (Kevin 1, Roy 3, Ted 3). Totals: 2,860 units, 150.1 M spikes.
