# NeuroPy — working rules

## Comments

Every comment and docstring: **~1 line, concise, reusable.** Default to none.

A comment earns its place only by carrying non-obvious *why* — a constraint the
reader cannot see in the code. Everything else is noise that drifts out of date.

**Never write:**

- Measurements, scores, or dates — "scored exactly 0.00", "~0.09 mean F1 worse",
  "(2026-08-21)", "was the best strategy measured". These are wrong after the
  next run.
- Refactor or session history — "this used to...", "adding X would have touched
  every model", "previously...".
- The specific fix you just made, or the bug it replaced.
- Restatements of the code, the identifier, or the error message on the next line.
- The same rationale in two places. One invariant, one home.
- Multi-line docstrings. Collapse the summary and the paragraph into one
  sentence, or drop the paragraph.

**Put the reasoning where it belongs:** explain it in your reply to the user, or
in `design_docs/` if it must persist. Not in the source file.

Good examples from this codebase:

```python
# rows, not cells: every action here acts on a pair
# on the ViewBox: p.clear() would drop it from the PlotItem
# a stated rate beats one recovered from spike gaps
# registry, not _fwd: untagged groups count too
```

`neuropy/analyses/ms_connectivity.py` is the core file — keep it especially
terse. No explanatory comments, no prints.

## Code

- No defensive guards around our own code; write against the interface you want.
- One source of truth — derived structures are rebuilt, not stored twice.
- No parallel backends: data/load/save belongs to the data class. Uncalled
  duplicates get deleted, not polished.
- Imports at the top of the file, never inside a function.
- Root cause over band-aid. Probe first; ask before ~10+ new lines.
- Never delete working behavior to make a bug disappear.

## Architecture

- SD/CD own all data and IO; panels read and write through their API only.
- Core runs headless — `plan.json` alone must reproduce GUI behavior.
- UI handlers are named after their widget, with the widget type as a suffix
  (`_btn`, `_check`, `_combo`).

## Testing

Mirror the real data, then run against the actual dataset. Interpreter:
`/Users/selinl/miniforge3/envs/NeuroPy2/bin/python`. For headless Qt:

```bash
QT_QPA_PLATFORM=offscreen \
QT_QPA_PLATFORM_PLUGIN_PATH=/Users/selinl/miniforge3/envs/NeuroPy2/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms
```

The GUI is launched from a separate process: `notebooks/run_ccg_gui.py`
(`app.exec()` blocks a Jupyter kernel's heartbeat).
