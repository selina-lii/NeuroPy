"""
Style cleanup for neuropy/ui/:
  1. Collapse 3+ consecutive blank lines between methods to 2 (PEP8 between top-level)
     and 2+ consecutive blank lines inside class bodies to 1.
  2. Remove `ui = self._ui` alias when `ui` is used <= 3 times in the method body,
     replacing each `ui.` / bare `ui` with `self._ui`.
"""

import ast
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pass 1: blank lines
# ---------------------------------------------------------------------------

def fix_blank_lines(src: str) -> str:
    """Inside class bodies collapse 2+ consecutive blank lines to 1."""
    # Between top-level defs (class/def at col 0) allow exactly 2 blank lines.
    # Inside class bodies (indented defs) allow exactly 1.
    # Strategy: collapse any run of 3+ blank lines to 2, then
    # collapse runs of 2 blank lines that appear inside an indented block to 1.

    # First: collapse 3+ blank lines → 2
    src = re.sub(r'\n{4,}', '\n\n\n', src)

    # Collapse 2 blank lines to 1 when surrounded by indented code
    # (i.e., the non-blank lines before/after start with whitespace)
    lines = src.split('\n')
    out = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        # Look for a run of blank lines
        if lines[i] == '':
            j = i
            while j < len(lines) and lines[j] == '':
                j += 1
            blank_count = j - i
            if blank_count >= 2:
                # Check context: are surrounding non-blank lines indented?
                prev_nonblank = next(
                    (lines[k] for k in range(i - 1, -1, -1) if lines[k].strip()), None)
                next_nonblank = next(
                    (lines[k] for k in range(j, len(lines)) if lines[k].strip()), None)
                prev_indented = prev_nonblank and prev_nonblank[0] in (' ', '\t')
                next_indented = next_nonblank and next_nonblank[0] in (' ', '\t')
                if prev_indented or next_indented:
                    # Inside a class/function body — keep only 1 blank line
                    # (we already appended lines[i] which is blank, skip rest)
                    i = j
                    continue
                else:
                    # Top-level — allow 2 blank lines; skip extras
                    if blank_count > 2:
                        # append one more blank then skip to j
                        out.append('')
                        i = j
                        continue
        i += 1
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Pass 2: remove cheap ui = self._ui aliases
# ---------------------------------------------------------------------------

_ALIAS_RE = re.compile(r'^(\s*)ui\s*=\s*self\._ui\s*$')


def _count_ui_uses(lines: list[str], start: int, end: int) -> int:
    """Count occurrences of `ui` (as identifier) in lines[start:end], excl. the alias line."""
    count = 0
    for ln in lines[start:end]:
        stripped = ln.strip()
        if _ALIAS_RE.match(ln):
            continue
        # Count word-boundary `ui` references (ui. or bare ui)
        count += len(re.findall(r'\bui\b', stripped))
    return count


def fix_ui_alias(src: str) -> str:
    lines = src.split('\n')
    # Find all alias lines
    alias_lines = [i for i, ln in enumerate(lines) if _ALIAS_RE.match(ln)]
    if not alias_lines:
        return src

    # Parse AST to get method boundaries
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src  # don't touch broken files

    # Build map: line_no (1-based) → (start_line, end_line) for FunctionDef
    method_bounds: dict[int, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # end_lineno is 1-based inclusive
            method_bounds[node.lineno] = (node.lineno, node.end_lineno)

    # For each alias line, find its enclosing method
    removals = set()
    for ali in alias_lines:
        ali_1based = ali + 1
        enclosing = None
        for mstart, (ms, me) in method_bounds.items():
            if ms <= ali_1based <= me:
                if enclosing is None or ms > enclosing[0]:
                    enclosing = (ms, me)
        if enclosing is None:
            continue
        ms, me = enclosing
        body_start = ali  # 0-based
        body_end   = me   # 0-based exclusive (me is 1-based inclusive → me itself)
        uses = _count_ui_uses(lines, ms - 1, me)
        if uses <= 3:
            removals.add(ali)

    if not removals:
        return src

    # Apply: remove alias lines and replace `ui.` / `\bui\b` in their methods
    # Build per-method replacement ranges
    method_for_alias: dict[int, tuple[int, int]] = {}
    for ali in removals:
        ali_1based = ali + 1
        enclosing = None
        for mstart, (ms, me) in method_bounds.items():
            if ms <= ali_1based <= me:
                if enclosing is None or ms > enclosing[0]:
                    enclosing = (ms, me)
        if enclosing:
            method_for_alias[ali] = enclosing

    # Collect line ranges that need `ui` → `self._ui` substitution
    # (union of all method ranges for removed aliases)
    replace_ranges: list[tuple[int, int]] = list(set(method_for_alias.values()))

    new_lines = list(lines)
    for i, ln in enumerate(new_lines):
        i_1based = i + 1
        in_range = any(ms <= i_1based <= me for ms, me in replace_ranges)
        if i in removals:
            new_lines[i] = None  # mark for removal
        elif in_range and i not in removals:
            # Replace `ui.` and bare `ui` (not inside string literals — simple heuristic)
            if _ALIAS_RE.match(ln):
                continue
            # Replace `ui.` first (more specific), then bare `ui`
            new_ln = re.sub(r'\bui\.', 'self._ui.', ln)
            new_ln = re.sub(r'\bui\b', 'self._ui', new_ln)
            new_lines[i] = new_ln

    return '\n'.join(ln for ln in new_lines if ln is not None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(path: Path) -> bool:
    src = path.read_text()
    out = fix_blank_lines(src)
    out = fix_ui_alias(out)
    # Verify it still parses
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR after transform: {e} — skipping {path.name}")
        return False
    if out != src:
        path.write_text(out)
        return True
    return False


if __name__ == '__main__':
    ui_dir = Path(__file__).parent.parent / 'neuropy' / 'ui'
    targets = sorted(ui_dir.glob('*.py'))
    for p in targets:
        changed = process_file(p)
        print(f"{'CHANGED' if changed else 'ok     '} {p.name}")
