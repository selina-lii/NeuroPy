"""Apply PEP 8 blank-line style to Python source files.

Rules enforced:
- 2 blank lines before top-level class definitions
- 1 blank line after the class docstring (if present)
- 1 blank line between methods within a class

Usage:
    python scripts/fix_style.py neuropy/ui/ccg_panel.py [file2 ...]
    python scripts/fix_style.py --all    # runs on all files listed in TARGETS
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGETS = [
    "neuropy/ui/ccg_panel.py",
    "neuropy/ui/time_slider.py",
    "neuropy/ui/utils.py",
    "neuropy/ui/stats_tests.py",
    "neuropy/ui/ccg_ui.py",
    "neuropy/ui/neuron_network.py",
    "neuropy/ui/menubar.py",
]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def fix_file(path: str) -> bool:
    src = Path(path).read_text(encoding="utf-8")
    lines = src.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 2 blank lines before top-level class
        if re.match(r'^class\s', line):
            # trim trailing blanks already in out
            while out and out[-1].strip() == "":
                out.pop()
            out.append("")
            out.append("")
            out.append(line)
            i += 1
            # 1 blank line after class docstring
            # find first non-blank, non-decorator after class:
            j = i
            while j < n and lines[j].strip() == "":
                j += 1
            # check for def __init__ or docstring
            if j < n and '"""' in lines[j]:
                # consume docstring
                out.append(lines[i] if i < j else "")
                i = j
                out.append(lines[i]); i += 1
                if lines[i - 1].count('"""') < 2:  # multi-line docstring
                    while i < n and '"""' not in lines[i]:
                        out.append(lines[i]); i += 1
                    if i < n:
                        out.append(lines[i]); i += 1
                # ensure 1 blank after docstring
                while i < n and lines[i].strip() == "":
                    i += 1
                out.append("")
            continue

        # 1 blank line between methods (def at indent > 0)
        if re.match(r'^(\s+)def\s', line):
            ind = _indent(line)
            if ind > 0:
                while out and out[-1].strip() == "":
                    out.pop()
                out.append("")
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1

    result = "\n".join(out) + "\n"
    if result == src:
        print(f"  {path}: no changes")
        return False
    Path(path).write_text(result, encoding="utf-8")
    print(f"  {path}: updated")
    return True


def main():
    args = sys.argv[1:]
    if "--all" in args:
        files = TARGETS
    elif args:
        files = args
    else:
        print(__doc__)
        sys.exit(0)

    root = Path(__file__).resolve().parents[1]
    changed = 0
    for f in files:
        p = root / f
        if not p.exists():
            print(f"  {f}: not found — skipped")
            continue
        if fix_file(str(p)):
            changed += 1
    print(f"\n{changed}/{len(files)} files updated.")


if __name__ == "__main__":
    main()
