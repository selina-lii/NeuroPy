"""Rename post_tenth* / post_tenth_sleep-only* segments 1..10 to zero-padded 01..10.

Four reference sites move together: custom_ccg files, selections/aux_tests unit dirs,
the per-session customtags + segment_groups JSONs, and stats_results rows.
"""

import json
import os
import re
import shutil
import sys
import time

ROOT = 'data/project_test2'
FAMILIES = ('post_tenth_sleep-only', 'post_tenth')   # longest prefix first
WIDTH = 2

# RatN_Day1 holds both an old unpadded set and a newer equal-effective 01..10 recompute;
# the recompute is authoritative, so the old set is dropped rather than renamed onto it.
DROP_COLLIDING = {('post_tenth_sleep-only', 'RatN_Day1')}


def pad(name: str) -> str | None:
    """`post_tenth9` -> `post_tenth09`; None when already padded or not a target."""
    for fam in FAMILIES:
        m = re.fullmatch(re.escape(fam) + r'(\d+)', name)
        if m:
            digits = m.group(1)
            if len(digits) >= WIDTH:
                return None
            return f"{fam}{int(digits):0{WIDTH}d}"
    return None


def family_of(seg: str) -> str | None:
    for fam in FAMILIES:
        if re.fullmatch(re.escape(fam) + r'\d+', seg):
            return fam
    return None


def plan_dir(d: str, seg_at: int, sess_at: int) -> tuple:
    """(renames, drops); the two layouts put the segment field at opposite ends."""
    renames, drops = [], []
    for entry in sorted(os.listdir(d)):
        parts = entry.split('.')
        if len(parts) <= max(seg_at, sess_at):
            continue
        seg, sess = parts[seg_at], parts[sess_at]
        new_seg = pad(seg)
        if new_seg is None:
            continue
        if (family_of(seg), sess) in DROP_COLLIDING:
            drops.append(os.path.join(d, entry))
            continue
        parts[seg_at] = new_seg
        renames.append((os.path.join(d, entry), os.path.join(d, '.'.join(parts))))
    return renames, drops


def plan_inner_meta(d: str) -> list:
    """A unit dir holds a meta json named after the dir; it must follow the dir's rename."""
    renames = []
    for entry in sorted(os.listdir(d)):
        unit = os.path.join(d, entry)
        if not os.path.isdir(unit):
            continue
        want = os.path.basename(unit).split('.bin_size')[0] + '.json'
        for f in os.listdir(unit):
            if f.endswith('.json') and f != want and pad(f.split('.')[0]) is not None:
                renames.append((os.path.join(unit, f), os.path.join(unit, want)))
    return renames


def rewrite_json(path: str, apply: bool) -> int:
    """Pad every post_tenth* token in a JSON file; returns the number of substitutions."""
    text = open(path).read()
    hits = [0]

    def sub(m):
        new = pad(m.group(0))
        if new is None:
            return m.group(0)
        hits[0] += 1
        return new

    out = re.sub(r'post_tenth(?:_sleep-only)?\d+', sub, text)
    if hits[0] and apply:
        json.loads(out)                      # never write a file we just corrupted
        open(path, 'w').write(out)
    return hits[0]


def main() -> int:
    apply = '--apply' in sys.argv
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    # custom_ccg is <seg>.<session>...; aux_tests is <session>.<conn_type>.<seg>
    dirs = [(os.path.join(ROOT, 'custom_ccg'), 0, 1),
            (os.path.join(ROOT, 'selections', 'aux_tests'), 2, 0)]
    jsons = ([os.path.join(ROOT, 'selections', f)
              for f in sorted(os.listdir(os.path.join(ROOT, 'selections')))
              if f.endswith('.json')]
             + [os.path.join(ROOT, 'stats_results', f)
                for f in sorted(os.listdir(os.path.join(ROOT, 'stats_results')))
                if f.endswith('.json')])

    all_renames, all_drops = [], []
    for d, seg_at, sess_at in dirs:
        r, dr = plan_dir(d, seg_at, sess_at)
        all_renames += r
        all_drops += dr
        print(f"{d}: {len(r)} rename, {len(dr)} drop")
    inner = plan_inner_meta(os.path.join(ROOT, 'custom_ccg'))
    all_renames += inner
    print(f"inner unit meta: {len(inner)} rename")

    if apply:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        bak = os.path.join(ROOT, f'.backup_pad_{stamp}')
        os.makedirs(bak, exist_ok=True)
        for p in jsons:
            shutil.copy2(p, os.path.join(bak, os.path.basename(p)))
        with open(os.path.join(bak, 'renames.txt'), 'w') as fh:
            for a, b in all_renames:
                fh.write(f"{a}\t{b}\n")
            for p in all_drops:
                fh.write(f"DROP\t{p}\n")
        print(f"backup -> {bak} ({len(jsons)} json + manifest)")

    # 10 before 1: an unpadded shorter name must not land on a longer one mid-run
    all_renames.sort(key=lambda ab: -len(os.path.basename(ab[0])))
    clashes = [b for _, b in all_renames if os.path.exists(b)]
    if clashes:
        print(f"ABORT: {len(clashes)} targets already exist, e.g. {clashes[:3]}")
        return 1

    for src in all_drops:
        print(f"  drop {os.path.basename(src)}")
        if apply:
            shutil.rmtree(src) if os.path.isdir(src) else os.remove(src)
    for src, dst in all_renames:
        if apply:
            os.rename(src, dst)

    total = 0
    for p in jsons:
        n = rewrite_json(p, apply)
        if n:
            total += n
            print(f"  {p}: {n} tokens")

    print(f"\n{'APPLIED' if apply else 'DRY RUN'}: "
          f"{len(all_renames)} renamed, {len(all_drops)} dropped, {total} json tokens")
    return 0


if __name__ == '__main__':
    sys.exit(main())
