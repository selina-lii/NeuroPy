#!/usr/bin/env python
"""One-shot: repoint saved project plans at their dataset's current shank mapping.

A plan freezes the field map it was built with, so a correction in
``io/datasets/<name>`` does not reach projects already on disk. This rewrites the
``shank_id`` entry of matching plans to whatever that module now says, leaving
every other field alone.

    python notebooks/fix_shank_map.py            # report only
    python notebooks/fix_shank_map.py --apply    # write, backing up first
"""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime
from importlib import import_module
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'


def dataset_fields(name: str) -> dict | None:
    try:
        return getattr(import_module(f'neuropy.io.datasets.{name}'), 'FIELDS', None)
    except ModuleNotFoundError:
        return None


def plans(root: Path) -> list[Path]:
    return sorted(root.glob('project_*/project_plan.json'))


def main(apply: bool):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in plans(DATA):
        plan = json.loads(path.read_text())
        name = plan.get('dataset') or plan.get('source') or ''
        want = dataset_fields(name)
        if not want:
            print(f'{path.parent.name:<22} dataset {name!r}: no module, skipped')
            continue
        have, target = plan.get('fields', {}), want.get('shank_id')
        current = have.get('shank_id')
        if current == target:
            print(f'{path.parent.name:<22} shank_id already current')
            continue
        print(f'{path.parent.name:<22} shank_id {current!r}')
        print(f'{"":<22}       -> {str(target)[:90]}')
        if not apply:
            continue
        backup = path.with_suffix(f'.json.bak-{stamp}')
        shutil.copy2(path, backup)
        have['shank_id'] = target
        plan['fields'] = have
        path.write_text(json.dumps(plan, indent=1))
        print(f'{"":<22} written; backup {backup.name}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write (default: report only)')
    main(ap.parse_args().apply)
