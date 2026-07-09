"""Migrate selections from previous/ format to current project_test2/ format.

Old session format:
  selections[full_key] = [[ref,tgt], ...]          (selected pairs, flat list)
  deleted_by_type[full_key] = [[ref,tgt], ...]
  pair_tags["ref,tgt"] = {groups:[...], notes:...} (flat, no conn-type)

New session format (JsonSavable / _SelectionData):
  selections[full_key] = {
    selected:   {"__set__": [[r,t], ...]},
    unselected: {"__set__": []},
    deleted:    {"__set__": [[r,t], ...]},
    tags:       {"__dict__": [[[r,t], {groups:[...]}], ...]}
  }

Old groups source ('groups copy.json' in DEST_DIR):
  {"groups": {name: {hotkey, notes}}}

New groups.json:
  {"registry": {name: {name, hotkey, notes}}}
"""
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _compact_json_str(obj) -> str:
    s = json.dumps(obj, indent=2, ensure_ascii=False)
    s = re.sub(r'\[\s*(-?\d+),\s*(-?\d+)\s*\]', r'[\1, \2]', s)
    def _join_str_array(m):
        inner = re.sub(r'\s+', ' ', m.group(1)).strip().rstrip(',')
        return f'[{inner}]'
    s = re.sub(r'\[\s*((?:"[^"]*",?\s*)+)\s*\]', _join_str_array, s)
    def _compact_obj(m):
        flat = re.sub(r'\s+', ' ', m.group(0)).strip()
        return flat if len(flat) <= 100 else m.group(0)
    s = re.sub(r'\{[^{}]*\}', _compact_obj, s, flags=re.DOTALL)
    return s

SRC_DIR  = 'data/previous/selection_data_to_migrate'
DEST_DIR = 'data/project_test2/selections'


def type_label_from_key(full_key: str) -> str:
    """'sess_RatK_Day2.ex_E.type_pyr-pyr' → 'pyr-pyr'"""
    return full_key.split('.type_', 1)[1] if '.type_' in full_key else 'unknown'


def session_name_from_filename(fname: str) -> str:
    return fname.replace('__latest.json', '').replace('.json', '')


def to_set(pairs: list) -> list:
    return sorted([list(p) for p in pairs])


def to_tags_dict(tags: dict[tuple, dict]) -> dict:
    """Convert {(ref,tgt): entry} → {"r,t": entry}"""
    return {f"{r},{t}": entry for (r, t), entry in tags.items()}


def migrate_session(old_path: str, dest_path: str) -> None:
    with open(old_path, encoding='utf-8') as f:
        old = json.load(f)

    old_selections = old.get('selections', {})    # {full_key: [[r,t], ...]}
    old_deleted    = old.get('deleted_by_type', {})  # {full_key: [[r,t], ...]}
    old_tags_flat  = old.get('pair_tags', {})     # {"r,t": {groups:[...]}}

    # Build pair → conn_type_label from old selections
    pair_to_type: dict[tuple, str] = {}
    for full_key, pairs in old_selections.items():
        lbl = type_label_from_key(full_key)
        for r, t in pairs:
            pair_to_type[(int(r), int(t))] = lbl

    # Parse flat pair_tags into per-key tags dicts
    tags_by_key: dict[str, dict[tuple, dict]] = {}
    for key_str, entry in old_tags_flat.items():
        try:
            ref_s, tgt_s = key_str.split(',', 1)
            pair = (int(ref_s.strip()), int(tgt_s.strip()))
        except (ValueError, AttributeError):
            print(f"  SKIP bad pair_tags key: {key_str!r}")
            continue
        lbl = pair_to_type.get(pair, 'unknown')
        # Find matching full_key
        matched = next((k for k in old_selections if type_label_from_key(k) == lbl), None)
        if matched is None:
            matched = next(iter(old_selections), None)
        if matched:
            tags_by_key.setdefault(matched, {})[pair] = entry

    # Build new selections dict in JsonSavable format
    all_keys = set(old_selections) | set(old_deleted)
    new_selections: dict = {}
    for full_key in all_keys:
        selected = [list(p) for p in old_selections.get(full_key, [])]
        deleted  = list(old_deleted.get(full_key, []))

        # Pairs tagged groups:["deleted"] → move to deleted set, remove from selected
        tags = dict(tags_by_key.get(full_key, {}))
        explicitly_deleted: set = set()
        for pair, entry in list(tags.items()):
            groups = entry.get('groups', [])
            if 'deleted' in groups:
                explicitly_deleted.add(pair)
                deleted.append(list(pair))
                remaining_groups = [g for g in groups if g != 'deleted']
                if remaining_groups or entry.get('notes', '').strip():
                    tags[pair] = {**entry, 'groups': remaining_groups}
                else:
                    del tags[pair]

        # Also: selected with no tag = deleted (reviewed but not kept)
        selected_set = {tuple(p) for p in selected}
        tagged_set   = set(tags.keys())
        for pair in list(selected_set):
            if pair not in tagged_set and pair not in explicitly_deleted:
                explicitly_deleted.add(pair)
                deleted.append(list(pair))
        selected = [list(p) for p in selected_set if p not in explicitly_deleted]

        new_selections[full_key] = {
            'selected':   to_set(selected),
            'unselected': to_set([]),
            'deleted':    to_set(deleted),
            'tags':       to_tags_dict(tags),
        }

    out = {
        'session':   old.get('session', ''),
        'saved_at':  datetime.now().isoformat(),
        'selections': new_selections,
    }

    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(_compact_json_str(out))

    n_tags = sum(len(t) for t in tags_by_key.values())
    n_sel  = sum(len(old_selections.get(k, [])) for k in all_keys)
    print(f"  {os.path.basename(dest_path)}: {n_sel} selected pairs, {n_tags} pair-tag entries")


def migrate_groups(src_groups: str, dest_groups: str) -> None:
    with open(src_groups, encoding='utf-8') as f:
        old = json.load(f)

    old_groups = old.get('groups', {})
    registry = {
        name: {'name': name, 'hotkey': meta.get('hotkey', ''), 'notes': meta.get('notes', '')}
        for name, meta in old_groups.items()
    }

    if os.path.exists(dest_groups):
        with open(dest_groups, encoding='utf-8') as f:
            existing = json.load(f)
        existing_reg = existing.get('registry', existing.get('groups', {}))
        for name, meta in registry.items():
            if name not in existing_reg:
                existing_reg[name] = meta
            else:
                for field in ('hotkey', 'notes'):
                    if not existing_reg[name].get(field) and meta.get(field):
                        existing_reg[name][field] = meta[field]
        existing['registry'] = existing_reg
        existing.pop('groups', None)
        out = existing
    else:
        out = {'registry': registry, 'saved_at': datetime.now().isoformat()}

    with open(dest_groups, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"  groups.json: {len(registry)} groups → registry")


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DEST_DIR}")
    os.makedirs(DEST_DIR, exist_ok=True)

    src_groups = os.path.join(DEST_DIR, 'groups copy.json')
    dest_groups = os.path.join(DEST_DIR, 'groups.json')
    if os.path.exists(src_groups):
        print(f"\nMigrating groups:")
        migrate_groups(src_groups, dest_groups)
    else:
        print(f"\nNo 'groups copy.json' found, skipping groups.")

    print(f"\nMigrating session files:")
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith('.json'):
            continue
        sess = session_name_from_filename(fname)
        migrate_session(os.path.join(SRC_DIR, fname),
                        os.path.join(DEST_DIR, sess + '.json'))

    print("\nDone.")


if __name__ == '__main__':
    main()
