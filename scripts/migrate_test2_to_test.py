"""Migrate data/project_test2/selections → data/project_test/selections.

Standalone — no neuropy imports.  Writes new JsonSavable JSON format directly.

Old format (per-session JSON):
  selections:     {type_key_str: [[ref, tgt], ...]}
  deleted_by_type:{type_key_str: [[ref, tgt], ...]}
  pair_tags:      {ct_str: {ref_str: {tgt_str: entry_dict}}}

Old groups.json:
  {groups: {name: {hotkey, notes}}, saved_at: ...}

New format (JsonSavable):
  selection_dataset.json  →  top-level with __ref__ to session files + groups
  {session}.json          →  SelectionData: selections as __dict__ of _SelectionData
  groups.json             →  Groups: {groups: {name: {name,hotkey,notes}}}
"""

import json
import os
import tempfile
import shutil
import datetime

SRC = 'data/project_test2/selections'
DST = 'data/project_test/selections'
REF = 'data/previous/selection_data_to_migrate'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _atomic_write(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(path) or '.',
                                    suffix='.tmp', delete=False, encoding='utf-8') as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)


def _set_json(pairs):
    """Encode a set/list of [ref,tgt] pairs as __set__ sentinel."""
    return {'__set__': sorted([list(p) for p in pairs])}


def _dict_json(d: dict):
    """Encode a dict with non-string keys as __dict__ sentinel."""
    return {'__dict__': [[list(k), v] for k, v in sorted(d.items())]}


# ---------------------------------------------------------------------------
# build new-format dicts in memory
# ---------------------------------------------------------------------------

def _build_selection_data_dict(old: dict) -> dict:
    """Build the JSON dict that SelectionData.serialize() would produce."""
    sel_by_type  = old.get('selections', {})
    del_by_type  = old.get('deleted_by_type', {})
    pair_tags_ct = old.get('pair_tags', {})

    # flatten pair_tags: {ct: {ref: {tgt: entry}}} → {(ref,tgt): entry}
    flat_tags = {}
    for ct, by_ref in pair_tags_ct.items():
        for ref_str, by_tgt in by_ref.items():
            for tgt_str, entry in by_tgt.items():
                flat_tags[(int(ref_str), int(tgt_str))] = dict(entry)

    all_type_keys = set(sel_by_type) | set(del_by_type)

    # build per-bucket _SelectionData serializations
    buckets = {}  # type_key_str → _SelectionData dict
    covered_pairs = set()

    for tk_str in sorted(all_type_keys):
        selected  = [tuple(p) for p in sel_by_type.get(tk_str, [])]
        deleted   = [tuple(p) for p in del_by_type.get(tk_str, [])]
        covered_pairs |= set(selected) | set(deleted)

        # tags for pairs in this bucket
        bucket_pairs = set(selected) | set(deleted)
        tags = {p: flat_tags[p] for p in bucket_pairs if p in flat_tags}

        buckets[tk_str] = {
            'selected':   _set_json(selected),
            'unselected': _set_json([]),   # unknown; UI repopulates
            'deleted':    _set_json(deleted),
            'tags':       _dict_json(tags) if tags else {'__dict__': []},
        }

    # any tags for pairs not covered by a bucket → put in first bucket
    uncovered = {p: v for p, v in flat_tags.items() if p not in covered_pairs}
    if uncovered and buckets:
        first_key = next(iter(buckets))
        existing_tags = buckets[first_key]['tags']['__dict__']
        existing_pairs = {tuple(entry[0]) for entry in existing_tags}
        for p, v in uncovered.items():
            if p not in existing_pairs:
                existing_tags.append([list(p), v])

    # SelectionData.serialize() produces {"selections": {"__dict__": [[key_str, bucket], ...]}}
    return {
        'selections': {
            '__dict__': [[tk_str, bucket] for tk_str, bucket in sorted(buckets.items())]
        }
    }


def _build_groups_dict(old_groups: dict) -> dict:
    """Build the JSON dict that Groups.serialize() would produce."""
    groups_out = {}
    for name, meta in old_groups.items():
        groups_out[name] = {
            'name':   name,
            'hotkey': meta.get('hotkey', ''),
            'notes':  meta.get('notes', ''),
        }
    return {'groups': groups_out}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def migrate():
    os.makedirs(DST, exist_ok=True)

    session_refs = {}   # nd_key_str → ref path

    for fname in sorted(os.listdir(SRC)):
        if fname == 'groups.json' or not fname.endswith('.json'):
            continue
        old = _read(os.path.join(SRC, fname))
        sel_keys = list(old.get('selections', {}).keys())
        if not sel_keys:
            print(f'  SKIP {fname}: no selections')
            continue

        # derive nd_key_str: strip the conn-type suffix to get the nd portion
        # e.g. 'sess_RatK_Day2.ex_E.type_pyr-pyr' → 'sess_RatK_Day2.ex_E'
        # But nd() drops type, so key is 'sess_{session}.ex_{exc}'
        # Actually Key.nd() keeps session+excitability, drops conn_type.
        # We need the session filename (without __latest suffix) as nd key.
        session_field = old.get('session', fname[:-5])

        # parse nd_key_str from the first type_key
        # format: 'sess_{session}.ex_{exc}.type_{ct}'
        tk0 = sel_keys[0]
        # nd portion = everything before '.type_'
        nd_str = tk0.split('.type_')[0] if '.type_' in tk0 else tk0

        sel_data_dict = _build_selection_data_dict(old)
        out_path = os.path.join(DST, f'{session_field}.json')
        _atomic_write(out_path, sel_data_dict)
        session_refs[nd_str] = out_path
        n_sel = sum(len(b.get('selected', {}).get('__set__', [])) for b in
                    [entry[1] for entry in sel_data_dict['selections']['__dict__']])
        print(f'  {session_field}: {n_sel} selected pairs → {out_path}')

    # groups
    grp_path = os.path.join(SRC, 'groups.json')
    grp_out_path = os.path.join(DST, 'groups.json')
    if os.path.exists(grp_path):
        old_groups = _read(grp_path).get('groups', {})
        _atomic_write(grp_out_path, _build_groups_dict(old_groups))
        print(f'  groups: {len(old_groups)} groups → {grp_out_path}')

    # top-level selection_dataset.json
    dataset_path = os.path.join(DST, 'selection_dataset.json')
    dataset = {
        'save_dir': DST,
        'groups': {'__ref__': grp_out_path},
        'sessions': {
            '__dict__': [[nd_str, {'__ref__': ref}]
                         for nd_str, ref in sorted(session_refs.items())]
        },
    }
    _atomic_write(dataset_path, dataset)
    print(f'\nWrote {dataset_path}')

    _verify(session_refs, old_groups if os.path.exists(grp_path) else {})


def _verify(session_refs: dict, old_groups: dict):
    print('\n--- Verification against data/previous ---')
    if not os.path.isdir(REF):
        print('  reference dir not found, skipping')
        return

    # check sessions
    ref_files = sorted(f for f in os.listdir(REF) if f.endswith('.json'))
    for fname in ref_files:
        old = _read(os.path.join(REF, fname))
        session_field = old.get('session', fname.replace('__latest', '')[:-5])
        # find matching written file
        out_path = os.path.join(DST, f'{session_field}.json')
        if not os.path.exists(out_path):
            print(f'  MISSING: {session_field}')
            continue
        new = _read(out_path)
        buckets = new['selections']['__dict__']

        ref_sel = sum(len(v) for v in old.get('selections', {}).values())
        ref_del = sum(len(v) for v in old.get('deleted_by_type', {}).values())
        new_sel = sum(len(entry[1]['selected']['__set__']) for entry in buckets)
        new_del = sum(len(entry[1]['deleted']['__set__']) for entry in buckets)

        ok_sel = '✓' if new_sel == ref_sel else f'✗ sel: new={new_sel} ref={ref_sel}'
        ok_del = '✓' if new_del == ref_del else f'✗ del: new={new_del} ref={ref_del}'
        print(f'  {session_field}: selected {ok_sel}  deleted {ok_del}')

    # check groups
    ref_grp_path = os.path.join(SRC, 'groups.json')
    if os.path.exists(ref_grp_path):
        ref_groups = _read(ref_grp_path).get('groups', {})
        new_groups = _read(os.path.join(DST, 'groups.json')).get('groups', {})
        missing = set(ref_groups) - set(new_groups)
        extra   = set(new_groups) - set(ref_groups)
        if missing:
            print(f'  MISSING groups: {missing}')
        if extra:
            print(f'  Extra groups: {extra}')
        if not missing and not extra:
            print(f'  Groups ✓ ({len(new_groups)} groups)')


if __name__ == '__main__':
    migrate()
