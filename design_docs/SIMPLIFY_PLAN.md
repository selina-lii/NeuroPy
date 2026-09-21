# Simplify pass — standing plan

Read this before every move. It is the brief; the conversation is not.

## Task

Structural simplification of the 8 edited files over 1000 lines.

| order | file | lines |
|---|---|---|
| 1 | `neuropy/ui/ccg_panel.py` | 2021 |
| 2 | `neuropy/ui/neuron_network.py` | 2055 |
| 3 | `neuropy/ui/dialogs.py` | 1806 |
| 4 | `neuropy/ui/pair_selection_panel.py` | 1755 |
| 5 | `neuropy/analyses/ms_connectivity.py` | 1526 |
| 6 | `neuropy/ui/utils.py` | 1357 |
| 7 | `neuropy/ui/time_slider.py` | 1277 |
| 8 | `neuropy/ui/stats_tests.py` | 1225 |

**Scope is the whole file, not recent edits.** For each function ask: what does
it do, who calls it, what does it call. Then simplify the structure globally.
The prior agent patched over simplifying; that is what this pass undoes.

Retaining all functionality is common sense, not a question to raise.

## Acceptance test

The user opens a large file and edits the backend without asking what a
function does. Line count (~2700 out, ~21%) is a proxy, not the goal.

## Style reference

`neuropy/core/neurons.py`. Verb-noun names. One-line docstring stating the
return. Short bodies with named intermediates. `@property` for derived values.
Explicit `if/elif/else` with `raise ValueError`. No swallowed errors.
Comments default to none; only non-obvious *why*, ~1 line.

Older code may be touched to streamline caller chains. `ms_connectivity.py` is
core — keep it especially terse.

## Safety net

```
/Users/selinl/miniforge3/envs/NeuroPy2/bin/python tests/test_ccg_context_golden.py
/Users/selinl/miniforge3/envs/NeuroPy2/bin/python tests/test_ccg_render_smoke.py
```

The golden test pins the **data**: 660 arrays over stored+extend paths, the
ACG/baseline/pval/deconv grid, 8 window x bin combos, 4 baseline methods.
The smoke test pins the **drawing**: item counts per subplot, ACG view box and
p-value view box across a 16-case toggle sweep (12 distinct signatures, so it
discriminates).

**Run both after every edit. Any diff means behavior changed — revert.**
`--record` re-baselines, only for intended change. When refactoring rendering,
record the baseline from the *pre-edit* code (`git stash`, record, pop) or the
test proves nothing.

Backup: `data/BACKUPS/pre_simplify_20260909_004951/neuropy_source.tar.gz`
(verified, 133 .py files).

## Do not touch, do not re-derive

- `p.titleLabel.setMaximumWidth(1)` in `_rebuild_subplots` — the layout fix.
  Passing the widget width instead fails: Qt raises the minimum to match the
  maximum.
- Each `PlotWidget` owns its scene; z cannot leak across plots. ACG z=1,
  pval z=2, above PlotItem z=0. Separate ViewBoxes are required — different
  y-scales.
- `EranConv._conv(ccg, W, wintype) -> (pvals, pred, qvals)`; `pred` is the
  baseline. W is in bins, widened to ~6 sigma internally, capped by
  `max(1, int(min(conf.conv_window / bs, (len(ccg) - 1) / 3)))`.
- `_bin_size` returns seconds.
- `shift N` prints come from `correlations.py`, pre-existing.

## Never commit

Standing instruction. No commits in this pass.

## ccg_panel.py — findings from the structural read

`CCGContextBuilder` (L800-1283) is a namespace of 20 `@staticmethod`s threading
`(nav, panel)` through every call. It should be an object holding
`nav`/`panel`/`cor`/`cs`.

`build_context` (L987) and `build_extend_context` (L1086) run the same five
steps in the same order, differing only in where the CCG comes from:

1. resolve pair and data
2. pick overlays by toggle
3. deconvolve, then refit the baseline
4. `CCGNorm.apply` x3
5. `_make_context`

Steps 2-5 are ~60 duplicated lines. One gateway, two thin entry points.

Other targets:

- the nested `dc = [...]` comprehension (L1043 and its twin in the extend path)
- `_style_key` (L938) — a 17-field hand-listed mirror of `_make_context`'s
  widget reads. It already drifted once: `.line` was missing, so a baseline
  style change was ignored. It must not be hand-maintained.
- `_dark_mode` (L817) — `try/except: pass` around our own code
- `_make_context` (L1182) — 18 kwargs, `_`-prefixed locals, reads widgets
  directly. Widget reads belong in the panel, not in context assembly.
- `wf_peak_ms` / `wf_peak_amp` are always `None` — dead fields.

## Grouping rule

Flat structs stop being readable at about 10-15 fields — see `CCGConfig`, which
groups its parameters rather than listing thirty. When a struct passes that,
find the unit that repeats and name it. `DisplayToggles` was 17 flat fields; the
repeating unit was (show, line) per trace, so it became `TraceToggle` x4 and
five top-level fields.

## Progress

- [x] backup taken and verified
- [x] 4 `[DBG]` probes removed (`pair_selection_panel.py` x3 including a dead
      `_n` counter, `pair_selection_data.py` x1)
- [x] golden test written, baseline recorded, passing
- [x] render smoke test written, baseline recorded from HEAD, passing
- [x] deleted `neuropy/ui/export_backend.py` — 568 lines, no importers, called
      a `build_context` signature that no longer exists
- [x] 1. ccg_panel.py — 2021 -> 1991 lines, longest function 145 -> 65
      - `DisplayToggles` / `TraceToggle` / `AcgToggle`: widgets read once, and
        the frozen object *is* the extend cache key, so `_style_key` is gone
        and cache/draw can no longer drift
      - `CCGSource`: what the two paths actually disagree about
      - `for_current` / `for_extend` fill a source; `_finish` is the one gateway
      - `_render_one` 145 lines -> 36, split into one `_draw_*` per layer
      - `LagAxis`: x geometry computed once, shared by every draw method
      - removed: `_style_key`, `_make_context`, `_dark_mode`'s `try/except: pass`
        (`_theme_fn` is `lambda: self.theme`, it cannot raise), `_resolve_pair`'s
        unused `sess_label`, `_firing_rates`' unused `seg_idx`
      - `_compute_extend_ccg`: blanket `except Exception` narrowed to
        `(ValueError, IndexError, MemoryError)` on the call that can fail
      - widget sections rebuilt on `widget_row` / `radio_group` /
        `set_checked_quietly`, new in `utils.py`; `CorrelogramPanel._build`
        split into `_build_plot_area` and `_build_toolbox`
      - `NormSection` dispatched scale buttons by matching their **label
        string**; renaming a label would have silently broken the wiring.
        Split into `_NORM_OPTIONS` and `_SCALE_OPTIONS`.
      - `_view_values`: 5-deep chained ternary and an inline `import numpy`
        replaced by one `_VIEWABLE` table that also builds the menu

### Not every `except` is scaffolding

Deleting the `try/except` around `p.titleLabel.item.setAttr` broke rendering:
that item is sometimes a plain `QGraphicsTextItem`, which has no `setAttr`. The
smoke test caught it. The fix is an explicit `hasattr` check — the guard was
real, it was only written as a swallowed exception. Check what a guard actually
catches before removing it.
- [x] 2. neuron_network.py — 2055 -> 1977 lines, longest function 193 -> 73
      - deleted `_FlowLayout`, an 89-line near-copy of `utils.FlowLayout` that
        was never instantiated; the file already imported the real one
      - `NetworkView`: the resolved state of one draw. `_draw_connections` took
        **15 positional parameters**; now it takes `(data, view)`.
      - `_draw_connections` 193 lines -> seven named methods: `_skip_pair`,
        `_entry_style`, `_draw_one_entry`, `_draw_current_pair`,
        `_draw_same_channel_arcs`, `_draw_deleted_pairs`, `_add_arrow`
      - `_render` now reads widgets in `_resolve_view` and draws from the view,
        which is the backend/GUI split for this panel
      - `_draw_neurons` split into `_plain_neuron_spots` / `_highlight_neuron_spots`
      - `_setup_toggles_and_groups` 145 -> ~40 via `widget_row`, `_toggle_chip`,
        `_heading`, `_zoom_grid`
      - two `except Exception` blocks removed: one wrapped a `getattr` chain that
        cannot raise, one hid an out-of-range index that a bounds check states
- [x] 3. dialogs.py — 1806 -> 1814 lines, longest function 154 -> ~60
      Line count rose slightly: six nested closures became named methods, which
      costs lines and buys the readability this pass is for.
      - `_make_group_tab` 154 lines -> `_name_row`, `_hotkey_row`,
        `_notes_and_pairs`, `_session_pair_tabs`, `_group_button_row`,
        `_convert_to_group`, `_delete_group`
      - the four-step tail (autosave, mutate, emit, refresh, rebuild) was
        written out four times; now `_apply_group_change`
      - `ExportOptionsDialog._build` 150 -> ~30 via `_field_box`,
        `_checkbox_box`, `_preview_column`, `_scope_box`
      - `_refresh_preview`: a blanket `except Exception` wrapping 30 lines
        removed, along with a dead `alpha` local it was hiding
      - `_collect`'s one-line `def`s became `_text_or_none` / `_float_or` /
        `_parse_xticks`, catching `ValueError` instead of everything
      - a `try/except` around `set_hotkey_ui` caught nothing: neither it nor
        `set_group_hotkey` raises
      - `_save_defaults` now reports a failed write instead of silently
        swallowing it
- [ ] 4. pair_selection_panel.py
- [ ] 5. ms_connectivity.py
- [ ] 6. utils.py
- [ ] 7. time_slider.py
- [ ] 8. stats_tests.py
