"""Migrate CCGConfig from .hkl to .json.

Usage:
    python scripts/migrate_config_hkl_to_json.py data/project_test2/ccg/config/test2_lowres_config.hkl
    python scripts/migrate_config_hkl_to_json.py --all   # finds all *_config.hkl under data/
"""
import sys
import json
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def migrate_one(hkl_path: str) -> str:
    import hickle as hkl
    obj = hkl.load(hkl_path)
    if hasattr(obj, 'pack'):
        d = obj.pack()
    elif isinstance(obj, dict):
        d = {k: v for k, v in obj.items()
             if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    else:
        d = {k: v for k, v in obj.__dict__.items()
             if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    out = str(hkl_path).replace('.hkl', '.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2)
    print(f"  {hkl_path} → {out}")
    return out


def main():
    args = sys.argv[1:]
    if '--all' in args:
        paths = glob.glob(str(ROOT / 'data' / '**' / '*_config.hkl'), recursive=True)
    else:
        paths = args

    if not paths:
        print(__doc__); return

    for p in paths:
        if not os.path.isfile(p):
            print(f"  not found: {p}"); continue
        try:
            migrate_one(p)
        except Exception as e:
            print(f"  FAILED {p}: {e}")

    print(f"\nDone: {len(paths)} file(s).")


if __name__ == '__main__':
    main()
