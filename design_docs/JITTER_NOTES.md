# Jitter: fixes, performance, and one open question

## Bugs fixed

**`os` imported inside a function.** `jitter.py` imported `os` at line 527 inside
`plot_jitter_verification`, but `JitterResults.save`/`load` use it at module
scope — a guaranteed `NameError` the moment either was called. Neither ever was,
which is how it survived. Moved to the top of the file.

**`CCGPointer.stored_by_segment` did not exist.** `Jitter.run` reads it on its
first line, so any run against a real pointer raised `AttributeError`; only the
worker's `SimpleNamespace` supplied it. `inds` is always `[seg, ref, tgt]`, so it
is now a property returning True rather than a stored duplicate. This is why
`JitterDataset.run_jitter` had never executed.

**`Neurons.merge` crashed on 1-D waveforms.** `_safe_merge3d` indexed
`list1[0].shape[1]`, assuming every waveform is 2-D. Sessions carrying a single
mean trace per unit (RatJ_Day1) raised `IndexError` as soon as jitter built its
combined `Neurons`. It now follows the present side's own shape.

## Performance: 28x

Cost was **superlinear in the pooled spike count**, not linear in njitter:

| njitter | time | per jitter |
|---|---|---|
| 10 | 0.11 s | 11 ms |
| 25 | 0.59 s | 23 ms |
| 50 | 2.29 s | 46 ms |
| 100 | 8.91 s | 89 ms |

`_compute_jitter_ccg_batch` merged all njitter jittered trains into one `Neurons`
and made a single `spike_correlations` call. That call's inner loop walks
`shift` outward until no spike pair is within the window; pooling 100 trains
(2.6 M spikes) forces far more iterations, each over a far larger array. The old
batch-size formula counted output bytes only and so always chose the worst case.

Chunking five trials per call and threading the chunks:

| | 1 thread | 10 threads |
|---|---|---|
| pooled (old) | 8.91 s | — |
| chunk 5 | 0.88 s | **0.36 s** |

Both knobs are on `JitterConfig` (`jitter_chunk=5`, `n_threads=0` → one per
core). Threads help because the CCG loop is NumPy and releases the GIL. CuPy is
used when present; this machine (arm64 macOS) has none, so threads are the path.

Verified bit-identical: same jitter trains through the chunked-threaded path and
the old pooled path give `np.array_equal` True. A full 8-pair session run went
**291.48 s → 10.35 s**.

## Persistence and the batch path

`cd._jitter_results` and `cd.save_jitter` did not exist; every UI reference was a
`hasattr` guard against nothing, so saving silently no-opped and `load_from_cd`
returned immediately. Both now exist on `CCGDataset`, writing
`<project>/jitter/jitter_results.hkl`.

Only **percentile bands** are stored, never the raw `njitter x n_bins` draw.

Per the spec, a single-pair run from the panel stays in the live cache only — it
is exploratory — and batches persist. `jitter_batch.run_and_store` merges into
the existing store keyed `(session, conn-type, segment)`, so running a second
group over the same segment adds to the first rather than replacing it.

One convention had to be reconciled: `JitterManager.seg()` maps dim0 index 0 to
`None` (the whole-session view), so the batch stores `None` for `'all'` rather
than `0`, or the GUI would never find its own results. The hickle round-trip also
had to cast the segment key back to `int`, since dict keys serialize as strings.

## A latent NameError in the render path

`plot_ccg_panel` read `wf_peak_ms` and `wf_peak_amp` (line 216) without declaring
them as parameters, and `render_ccg_png` never passed them. The block only runs
when an ACG overlay is active, which is why it had never fired. Both are now
parameters, passed from `RenderContext`.

## Open question: the test window does not contain the peak

With jitter working, every pair in `RatJ_Day1 pyr-pyr` returns p ~= 1.0. The CCG
path is provably correct — feeding the *unjittered* train through the jitter code
reproduces the stored CCG exactly (total 9169, window 1185). The null really does
exceed the real value in the test window.

The reason is the window itself. `min_lag`/`max_lag` are +1..+3 ms, but:

```
  1->  3 peak@bin 14 (lag  +4ms)   OUT
  2->  1 peak@bin  6 (lag  -4ms)   OUT
  3->  2 peak@bin 10 (lag  +0ms)   OUT
  4->  5 peak@bin  5 (lag  -5ms)   OUT
  5->  4 peak@bin 15 (lag  +5ms)   OUT
 18-> 11 peak@bin 10 (lag  +0ms)   OUT
 22-> 14 peak@bin 11 (lag  +1ms)   IN
 23->  3 peak@bin 13 (lag  +3ms)   IN
```

Seven of eight peaks fall outside the window, several at negative lags where a
monosynaptic `pyr-pyr` connection cannot be. For pair 1->3 the window sits on the
rising flank (mean 395) *below* the CCG's own mean (437); jitter smooths the peak
and spreads mass into those bins, so the null wins and p -> 1.

Jitter is answering the question it was asked. The question is wrong: these pairs
were screened by hollow convolution, which finds a *local* peak at any lag (the
C1 issue), so being in the pointer does not mean the peak is monosynaptic.

**Not fixed here**, because narrowing or moving the window changes screening for
every stored result. Two candidates when it is settled: test at the pair's own
peak lag rather than a fixed window, or require the peak to be inside the window
at screening time so the pointer only holds pairs the window describes.
