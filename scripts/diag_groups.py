#!/usr/bin/env python3
"""Print groups.json location, format, and session pair_tags structure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'project_test2'
SEL_DIR = ROOT / 'data' / PROJECT / 'selections'
GROUPS_JSON = SEL_DIR / 'groups.json'
GROUPS_DIR = ROOT / 'data' / PROJECT / 'groups' / 'groups_default.json'


def pair_tags_shape(pt: dict) -> str:
    if not pt:
        return 'empty'
    top = next(iter(pt))
    if '-' in str(top) or str(top) == 'unknown':
        return f'ct-first (sample ct={top!r})'
    if str(top).lstrip('-').isdigit():
        return f'ref-first (sample ref={top!r}) — needs normalize at load'
    return f'unknown (sample={top!r})'


def count_memberships(pt: dict) -> tuple[int, int]:
    """Return (pairs_with_groups, total_assignments) for ref-first or ct-first."""
    pairs, assigns = 0, 0

    def _scan(entry):
        nonlocal pairs, assigns
        if isinstance(entry, dict) and entry.get('groups'):
            pairs += 1
            assigns += len(entry['groups'])

    top = next(iter(pt)) if pt else None
    if top and ('-' in str(top) or str(top) == 'unknown'):
        for refs in pt.values():
            if not isinstance(refs, dict):
                continue
            for tgts in refs.values():
                if isinstance(tgts, dict):
                    _scan(tgts)
    elif top and str(top).lstrip('-').isdigit():
        for tgts in pt.values():
            if not isinstance(tgts, dict):
                continue
            for entry in tgts.values():
                _scan(entry)
    return pairs, assigns


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else 'RatK_Day2'
    print(f'=== Groups diagnostic ({PROJECT}) ===\n')

    print('Load path 1 (primary):', GROUPS_JSON)
    print('  exists:', GROUPS_JSON.is_file())
    print('Load path 2 (fallback):', GROUPS_DIR)
    print('  exists:', GROUPS_DIR.is_file())

    if GROUPS_JSON.is_file():
        data = json.loads(GROUPS_JSON.read_text(encoding='utf-8'))
        print('\ngroups.json top keys:', list(data.keys()))
        if 'groups' in data:
            print('  format: NEW (groups dict)')
            print('  n defs:', len(data['groups']))
            print('  sample:', list(data['groups'].keys())[:6])
        elif 'group_registry' in data:
            print('  format: OLD (group_registry) — loader now converts')
        else:
            print('  format: UNRECOGNIZED')
    else:
        print('\ngroups.json MISSING at primary path')

    sess_path = SEL_DIR / f'{session}.json'
    print(f'\nSession file: {sess_path}')
    print('  exists:', sess_path.is_file())
    if sess_path.is_file():
        sd = json.loads(sess_path.read_text(encoding='utf-8'))
        pt = sd.get('pair_tags', {})
        print('  pair_tags shape:', pair_tags_shape(pt))
        p, a = count_memberships(pt)
        print(f'  pairs with groups: {p}  assignments: {a}')
        print('  has selections by type:', bool(sd.get('selections')))


if __name__ == '__main__':
    main()
