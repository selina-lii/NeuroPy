#!/usr/bin/env python3
"""Migrate pre-Qt CCG + selections into data/project_test2 for exact parity testing.

Sources:
  data/previous/ccg/3-12-26-allepoch_sd+nsd_alpha0.05_fdr/
  data/selection_data_to_migrate/*__latest.json

Destination:
  data/project_test2/ccg/
    {project}_lowres_ccgdata_{session}__lowres.npz   — one array file per session
    {project}_highres_ccgdata_{session}__highres.npz — optional high-res per session
    {project}_lowres_ccgpointers_{session}.hkl       — all conn types for session
  data/project_test2/selections/

Run (needs hickle):
  python scripts/migrate_test2.py
"""
from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_CCG = ROOT / "data" / "previous" / "ccg" / "3-12-26-allepoch_sd+nsd_alpha0.05_fdr"
SRC_SEL = ROOT / "data" / "selection_data_to_migrate"
if not SRC_SEL.is_dir():
    SRC_SEL = ROOT / "data" / "previous" / "selection_data_to_migrate"
SRC_GROUPS = ROOT / "data" / "test" / "selections" / "groups.json"
DST_ROOT = ROOT / "data" / "project_test2"
DST_CCG = DST_ROOT / "ccg"
DST_CCGDATA = DST_CCG / "ccgdata"
DST_POINTERS = DST_CCG / "pointers"
DST_CONFIG = DST_CCG / "config"
DST_SEL = DST_ROOT / "selections"

PROJECT = "test2"
OLD_PREFIX = "3-12-26-allepoch_sd+nsd_alpha0.05_fdr"


import dataclasses


@dataclasses.dataclass(eq=False)
class _MigrKey:
    session: str | None = None
    epoch: str | None = None
    ref_ind: int | None = None
    target_ind: int | None = None
    segment: int | None = None
    excitability: str | None = None
    conn_type: tuple | None = None
    resolution: str = 'lowres'

    def add(self, **kwargs):
        d = dataclasses.asdict(self)
        d.update({k: v for k, v in kwargs.items() if v is not None})
        return _MigrKey(**d)


class _Ptr:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _Stub:
    def __init__(self, *a, **kw):
        self.__dict__.update(kw)

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __getitem__(self, key):
        raise KeyError(key)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _stub_unpickle(data: bytes):
    class StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'Key' and 'neurons_dataset' in module:
                return _MigrKey
            if name == 'CCGPointer' and 'ms_connectivity' in module:
                return _Ptr
            if module == 'pandas.core.indexes.numeric' and name == 'Int64Index':
                import pandas as pd
                return pd.Index
            if module in ('conf', 'conn_strength_method'):
                return _Stub
            if (module.startswith('pandas')
                    or module.startswith('numpy')
                    or module.startswith('pytz')
                    or module.startswith('datetime')):
                try:
                    return super().find_class(module, name)
                except (AttributeError, ImportError, ModuleNotFoundError):
                    import pandas as pd
                    if hasattr(pd, name):
                        return getattr(pd, name)
                    return _Stub
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError, ModuleNotFoundError):
                return _Stub
    return StubUnpickler(io.BytesIO(data)).load()


def _load_hkl_entries(path: Path) -> list:
    """Load all objects from a monolithic or single-blob hickle file."""
    with h5py.File(str(path), 'r') as f:
        data_node = f.get('data')
        if data_node is None:
            return []
        if hasattr(data_node, 'shape') and data_node.shape:
            obj = _stub_unpickle(bytes(data_node[()]))
            if obj is None:
                return []
            if isinstance(obj, list):
                return obj
            return [obj]
        results = []
        for entry_key in sorted(data_node.keys()):
            grp = data_node[entry_key]
            if not hasattr(grp, 'keys'):
                continue
            payload_key = 'data1' if 'data1' in grp else ('data0' if 'data0' in grp else None)
            if payload_key is None:
                continue
            try:
                obj = _stub_unpickle(bytes(grp[payload_key][()]))
                if obj is not None:
                    results.append(obj)
            except Exception as exc:
                print(f"  [warn] {path.name} entry {entry_key}: {exc}")
        return results


def _session_from_key(obj) -> str:
    key = getattr(obj, 'key', None) or getattr(obj, '_key', None)
    if key is None:
        return 'unknown'
    return str(getattr(key, 'session', None) or 'unknown')


def _session_pointer_path(session: str) -> Path:
    DST_POINTERS.mkdir(parents=True, exist_ok=True)
    return DST_POINTERS / f'{PROJECT}_lowres_ccgpointers_{session}.hkl'


def _session_from_pointer_fname(name: str) -> str | None:
    prefix = f'{PROJECT}_lowres_ccgpointers_'
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    if rest.endswith('.hkl'):
        rest = rest[:-4]
    if '__E__' in rest:
        return rest.split('__E__', 1)[0]
    if '__I__' in rest:
        return rest.split('__I__', 1)[0]
    return rest or None


def _pointer_key_str(obj) -> str:
    key = obj.get('key') if isinstance(obj, dict) else getattr(obj, 'key', None)
    if key is None:
        return ''
    if hasattr(key, 'session'):
        parts = [
            f"sess_{key.session}" if key.session else "",
            f"ex_{key.excitability}" if key.excitability else "",
            (f"type_{key.conn_type[0]}-{key.conn_type[1]}"
             if key.conn_type else ""),
        ]
        return '.'.join(filter(None, parts))
    if isinstance(key, dict):
        sess = str(key.get('session', ''))
        ex = key.get('excitability', None) or 'E'
        ct = key.get('conn_type', None)
        if isinstance(ct, (list, tuple)) and len(ct) == 2:
            ct_s = f'{ct[0]}-{ct[1]}'
        else:
            ct_s = str(ct) if ct else ''
        return f'sess_{sess}.ex_{ex}.type_{ct_s}'
    return str(key)


def _iter_pointer_entries(obj):
    """Yield (key_str, obj) from a pointer file blob."""
    if isinstance(obj, dict) and 'key' in obj and ('_inds' in obj or 'inds' in obj):
        yield _pointer_key_str(obj), obj
        return
    if isinstance(obj, dict):
        for val in obj.values():
            if isinstance(val, dict) and 'key' in val:
                yield _pointer_key_str(val), val
            elif hasattr(val, 'key'):
                yield _pointer_key_str(val), val
        return
    if hasattr(obj, 'key'):
        yield _pointer_key_str(obj), obj


def _inds2_from_pointer(obj):
    if isinstance(obj, dict):
        for name in ('inds2', '_inds', 'inds'):
            if name in obj and obj[name] is not None:
                return obj[name]
        return None
    inds = getattr(obj, 'inds2', None)
    if inds is None:
        inds = getattr(obj, '_inds', None)
    if inds is None:
        inds = getattr(obj, 'inds', None)
    return inds


def migrate_pointers(conf=None) -> int:
    """Write one lowres pointer bundle per session (all conn types inside)."""
    import hickle as hkl
    from collections import defaultdict

    mono = SRC_CCG / f"{OLD_PREFIX}_lowres_ccgpointers.hkl"
    if not mono.exists():
        print(f"[pointers] missing {mono}")
        return 0
    print(f"[pointers] loading {mono.name} ...")
    by_sess: dict[str, dict] = defaultdict(dict)
    for obj in _load_monolithic_pointers(mono):
        if conf is not None:
            ptr = _rehydrate_pointer(obj, conf)
            if ptr is None:
                print(f"  [skip] unexpected type {type(obj).__name__}")
                continue
            entry = ptr
            sess = str(getattr(ptr.key, 'session', 'unknown'))
            kstr = str(ptr.key)
        else:
            entry = _pointer_blob(obj)
            if entry is None:
                print(f"  [skip] unexpected type {type(obj).__name__}")
                continue
            key = entry.get('key')
            sess = str(getattr(key, 'session', None) or (key.get('session') if isinstance(key, dict) else 'unknown'))
            kstr = _pointer_key_str(entry)
        by_sess[sess][kstr] = entry
    n_types = 0
    for sess, bundle in sorted(by_sess.items()):
        out = _session_pointer_path(sess)
        hkl.dump(bundle, str(out))
        print(f"  → {out.name}  {len(bundle)} conn-type pointers")
        n_types += len(bundle)
    return n_types


def _key_to_dict(key) -> dict:
    if isinstance(key, dict):
        return key
    if dataclasses.is_dataclass(key):
        return dataclasses.asdict(key)
    return {
        'session': getattr(key, 'session', None),
        'excitability': getattr(key, 'excitability', None),
        'conn_type': getattr(key, 'conn_type', None),
        'resolution': getattr(key, 'resolution', 'lowres'),
    }


def _pointer_blob(obj) -> dict | None:
    if isinstance(obj, dict) and 'key' in obj:
        obj = dict(obj)
        obj['key'] = _key_to_dict(obj['key'])
        return obj
    if not hasattr(obj, 'key'):
        return None
    inds = getattr(obj, '_inds', None)
    if inds is None:
        inds = getattr(obj, 'inds', None)
    return {
        'key': _key_to_dict(obj.key),
        '_inds': inds,
        'selected_inds': getattr(obj, 'selected_inds', None),
        'significant': getattr(obj, 'significant', None),
    }


def repack_per_type_pointers() -> int:
    """Merge existing per-conn-type pointer files into per-session bundles."""
    import hickle as hkl
    from collections import defaultdict

    by_sess: dict[str, dict] = defaultdict(dict)
    n_src = 0
    for path in sorted(DST_CCG.glob(f'{PROJECT}_lowres_ccgpointers_*__*.hkl')):
        sess = _session_from_pointer_fname(path.name)
        if not sess:
            continue
        try:
            obj = hkl.load(str(path))
        except Exception as exc:
            print(f"  [warn] {path.name}: {exc}")
            continue
        for kstr, entry in _iter_pointer_entries(obj):
            if not kstr:
                continue
            by_sess[sess][kstr] = entry
            n_src += 1
    for sess, bundle in sorted(by_sess.items()):
        hkl.dump(bundle, str(_session_pointer_path(sess)))
        print(f"  [repack] {sess}.hkl ← {len(bundle)} pointers")
    return n_src


def _extract_ccg_arrays(obj) -> dict | None:
    if isinstance(obj, _Stub) or not hasattr(obj, '__dict__'):
        return None
    v = vars(obj)
    ccg = v.get('ccg')
    if ccg is None or not isinstance(ccg, np.ndarray):
        return None
    qval = v.get('qval') if v.get('qval') is not None else v.get('qval_corrected')
    out = {'ccg': np.asarray(ccg, dtype=np.float32)}
    if v.get('ccg_null') is not None:
        out['ccg_null'] = np.asarray(v['ccg_null'], dtype=np.float32)
    if v.get('pval') is not None:
        out['pval'] = np.asarray(v['pval'], dtype=np.float64)
    if qval is not None:
        out['qval'] = np.asarray(qval, dtype=np.float64)
    return out


def _build_config():
    from neuropy.analyses.ms_connectivity import CCGConfig

    compute, signif = {}, {}
    cp = SRC_CCG / f"{OLD_PREFIX}_lowres.compute.meta.json"
    sp = SRC_CCG / f"{OLD_PREFIX}_lowres.signif.meta.json"
    if cp.exists():
        compute = json.loads(cp.read_text()).get('conf', {})
    if sp.exists():
        signif = json.loads(sp.read_text()).get('conf', {})

    conn_types = []
    for ei, label in [('E', 'conn_types_E'), ('I', 'conn_types_I')]:
        for ct in compute.get(label, []):
            conn_types.append((ei, tuple(ct)))

    return CCGConfig(
        name=PROJECT,
        duration=compute.get('duration', 20e-3),
        bin_size=compute.get('bin_size'),
        conv_window=compute.get('conv_window', 5e-3),
        conn_types=conn_types or None,
        alpha=signif.get('alpha', 0.05),
        alpha2=signif.get('alpha2', 0.1),
        min_lag=signif.get('min_lag', 1e-3),
        max_lag=signif.get('max_lag', 3e-3),
        min_spkcount=signif.get('min_spkcount', 2.5),
        spkcount_scope=signif.get('spkcount_scope', 12e-3),
        multiple_correction=signif.get('multiple_correction', 'fdr_bh'),
        use_acceleration=compute.get('use_acceleration', False),
        symmetrize_ccg=compute.get('symmetrize_ccg', True),
        resolution='lowres',
    )


def migrate_ccgdata() -> int:
    DST_CCGDATA.mkdir(parents=True, exist_ok=True)
    n = 0
    mono = SRC_CCG / f"{OLD_PREFIX}_lowres_ccgdata.hkl"
    if not mono.exists():
        print(f"[ccgdata] missing {mono}")
        return 0
    print(f"[ccgdata] loading {mono.name} ...")
    for obj in _load_hkl_entries(mono):
        arrays = _extract_ccg_arrays(obj)
        if not arrays:
            continue
        sess = _session_from_key(obj)
        fname = f"{PROJECT}_lowres_ccgdata_{sess}__lowres.npz"
        np.savez_compressed(str(DST_CCGDATA / fname), **arrays)
        print(f"  → {fname}  ccg={arrays['ccg'].shape}")
        n += 1

    hi = SRC_CCG / f"{OLD_PREFIX}_lowres_highres.hkl"
    if hi.exists():
        print(f"[highres] loading {hi.name} ...")
        for obj in _load_hkl_entries(hi):
            arrays = _extract_ccg_arrays(obj)
            if not arrays:
                continue
            sess = _session_from_key(obj)
            fname = f"{PROJECT}_highres_ccgdata_{sess}__highres.npz"
            np.savez_compressed(str(DST_CCGDATA / fname), **arrays)
            print(f"  → {fname}  ccg={arrays['ccg'].shape}")
            n += 1
    return n


def _load_monolithic_pointers(path: Path) -> list:
    """Load CCGPointer objects from nested dict-style ccgpointers hkl."""
    results = []
    with h5py.File(str(path), 'r') as f:
        data_node = f.get('data')
        if data_node is None:
            return results
        bucket_keys = [k for k in data_node.keys() if k.strip('"') == 'data']
        if not bucket_keys:
            bucket_keys = list(data_node.keys())
        for bucket in bucket_keys:
            bucket_grp = data_node[bucket]
            if not hasattr(bucket_grp, 'keys'):
                continue
            for entry_key in sorted(bucket_grp.keys()):
                entry = bucket_grp[entry_key]
                if not hasattr(entry, 'keys'):
                    continue
                if 'data1' not in entry:
                    continue
                try:
                    if 'data0' in entry:
                        key_obj = _stub_unpickle(bytes(entry['data0'][()]))
                    else:
                        key_obj = None
                    ptr = _stub_unpickle(bytes(entry['data1'][()]))
                    if key_obj is not None and not hasattr(ptr, 'key'):
                        ptr.key = key_obj
                    if ptr is not None:
                        results.append(ptr)
                except Exception as exc:
                    print(f"  [warn] {path.name} {bucket}/{entry_key}: {exc}")
    return results


def _rehydrate_pointer(obj, conf):
    from neuropy.analyses.ms_connectivity import CCGPointer

    if isinstance(obj, CCGPointer):
        obj.conf = conf
        return obj
    if isinstance(obj, dict):
        inds = obj.get('_inds', obj.get('inds'))
        return CCGPointer(
            key=obj['key'],
            inds=inds,
            conf=conf,
            selected_inds=obj.get('selected_inds'),
            significant=obj.get('significant'),
        )
    if hasattr(obj, 'key') and (hasattr(obj, '_inds') or hasattr(obj, 'inds')):
        inds = getattr(obj, '_inds', None) or getattr(obj, 'inds', None)
        return CCGPointer(
            key=obj.key,
            inds=inds,
            conf=conf,
            selected_inds=getattr(obj, 'selected_inds', None),
            significant=getattr(obj, 'significant', None),
        )
    return None


def reorganize_flat_ccg() -> tuple[int, int, int]:
    """Move legacy flat ccg/ files into ccgdata/, pointers/, config/ subdirs."""
    DST_CCGDATA.mkdir(parents=True, exist_ok=True)
    DST_POINTERS.mkdir(parents=True, exist_ok=True)
    DST_CONFIG.mkdir(parents=True, exist_ok=True)
    n_data = n_ptr = n_cfg = 0
    for path in list(DST_CCG.glob(f'{PROJECT}_*_ccgdata_*.npz')):
        path.rename(DST_CCGDATA / path.name)
        n_data += 1
    for path in list(DST_CCG.glob(f'{PROJECT}_*_ccgpointers_*.hkl')):
        path.rename(DST_POINTERS / path.name)
        n_ptr += 1
    for path in list(DST_CCG.glob(f'{PROJECT}_*config.hkl')):
        path.rename(DST_CONFIG / path.name)
        n_cfg += 1
    for path in list(DST_CCG.glob(f'{PROJECT}_*.meta.json')):
        path.rename(DST_CONFIG / path.name)
        n_cfg += 1
    if n_data or n_ptr or n_cfg:
        print(f"[layout] moved {n_data} ccgdata, {n_ptr} pointers, {n_cfg} config/meta")
    return n_data, n_ptr, n_cfg


    DST_CONFIG.mkdir(parents=True, exist_ok=True)
    conf.save()
    print(f"[config] → {PROJECT}_lowres_config.hkl")
    for suffix in ('compute', 'signif', 'ccgdata', 'ccgpointers'):
        src = SRC_CCG / f"{OLD_PREFIX}_lowres.{suffix}.meta.json"
        if not src.exists():
            continue
        data = json.loads(src.read_text())
        if 'conf' in data and isinstance(data['conf'], dict):
            data['conf']['name'] = PROJECT
        dst = DST_CONFIG / f"{PROJECT}_lowres.{suffix}.meta.json"
        dst.write_text(json.dumps(data, indent=2))
        print(f"[meta]   → {dst.name}")


def _ct_from_sel_key(sel_key: str) -> str | None:
    if '.type_' not in sel_key:
        return None
    return sel_key.split('.type_', 1)[1]


def _ct_label_from_key(key) -> str:
    if isinstance(key, dict):
        ct = key.get('conn_type', None)
    else:
        ct = getattr(key, 'conn_type', None)
    if isinstance(ct, (list, tuple)) and len(ct) == 2:
        return f'{ct[0]}-{ct[1]}'
    return str(ct) if ct else ''


def build_pair_to_ct(old: dict) -> dict[tuple[int, int], str]:
    pair_to_ct: dict[tuple[int, int], str] = {}
    for bucket in (old.get('selections', {}), old.get('deleted_by_type', {})):
        for sel_key, pairs in bucket.items():
            ct = _ct_from_sel_key(sel_key)
            if not ct:
                continue
            for p in pairs:
                try:
                    pair_to_ct[(int(p[0]), int(p[1]))] = ct
                except (TypeError, ValueError, IndexError):
                    pass
    return pair_to_ct


def pair_to_ct_from_pointers(session: str) -> dict[tuple[int, int], str]:
    """Fill pair→ct from per-session pointer bundle."""
    path = _session_pointer_path(session)
    if not path.exists():
        return {}
    import hickle as hkl

    try:
        obj = hkl.load(str(path))
    except Exception:
        return {}
    out: dict[tuple[int, int], str] = {}
    for _kstr, ptr in _iter_pointer_entries(obj):
        key_obj = ptr.get('key') if isinstance(ptr, dict) else getattr(ptr, 'key', None)
        ct = _ct_label_from_key(key_obj)
        inds2 = _inds2_from_pointer(ptr)
        if inds2 is None:
            continue
        for row in inds2:
            try:
                out[(int(row[0]), int(row[1]))] = ct
            except (TypeError, ValueError, IndexError):
                pass
    return out


def flat_tags_to_nested_by_ct(flat: dict, pair_to_ct: dict) -> dict:
    nested: dict = {}
    for key_str, tdata in flat.items():
        parts = str(key_str).split(',')
        if len(parts) != 2:
            continue
        ref_s, tgt_s = parts[0].strip(), parts[1].strip()
        try:
            pair = (int(ref_s), int(tgt_s))
        except ValueError:
            continue
        entry = dict(tdata) if isinstance(tdata, dict) else {'notes': str(tdata)}
        entry = {k: v for k, v in entry.items() if v}
        if not entry:
            continue
        ct = pair_to_ct.get(pair, 'unknown')
        nested.setdefault(ct, {}).setdefault(ref_s, {})[tgt_s] = entry
    return nested


def _ex_from_ct(ct: str) -> str:
    return 'I' if ct.startswith('inter-') else 'E'


def _legacy_sel_key(session: str, ex: str, ct: str) -> str:
    return f'sess_{session}.ex_{ex}.type_{ct}'


def _count_nested_tag_pairs(nested: dict) -> int:
    n = 0
    for ref_dict in nested.values():
        if not isinstance(ref_dict, dict):
            continue
        for tgt_dict in ref_dict.values():
            if isinstance(tgt_dict, dict):
                n += len(tgt_dict)
    return n


def enrich_selections_from_tags(
    session: str,
    selections_by_type: dict,
    nested_tags: dict,
    pair_to_ct: dict,
) -> dict:
    """Union tagged pairs into per-type selections buckets.

    Legacy saves often left selections empty while pair_tags held reviewed pairs.
    """
    out: dict[str, list] = {}
    seen: dict[str, set] = {}
    for key, pairs in selections_by_type.items():
        bucket, s = [], set()
        for p in pairs:
            try:
                t = (int(p[0]), int(p[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if t not in s:
                s.add(t)
                bucket.append([t[0], t[1]])
        if bucket:
            out[key] = bucket
            seen[key] = s
    for ct, ref_dict in nested_tags.items():
        if not isinstance(ref_dict, dict):
            continue
        for ref_str, tgt_dict in ref_dict.items():
            if not isinstance(tgt_dict, dict):
                continue
            for tgt_str in tgt_dict:
                try:
                    pair = (int(ref_str), int(tgt_str))
                except (TypeError, ValueError):
                    continue
                ct_eff = pair_to_ct.get(pair)
                if not ct_eff or ct_eff == 'unknown':
                    ct_eff = ct if ct != 'unknown' else None
                if not ct_eff:
                    continue
                ex = _ex_from_ct(ct_eff)
                key = _legacy_sel_key(session, ex, ct_eff)
                seen.setdefault(key, set())
                out.setdefault(key, [])
                if pair not in seen[key]:
                    seen[key].add(pair)
                    out[key].append([pair[0], pair[1]])
    for key in out:
        out[key].sort(key=lambda p: (p[0], p[1]))
    return out


def _selected_union(selections_by_type: dict) -> list[list[int]]:
    selected, seen = [], set()
    for pairs in selections_by_type.values():
        for p in pairs:
            t = (int(p[0]), int(p[1]))
            if t not in seen:
                seen.add(t)
                selected.append([t[0], t[1]])
    selected.sort()
    return selected


def sync_pointer_selected_inds(session: str, selections_by_type: dict) -> int:
    """Write per-type selected_inds into the session pointer bundle."""
    import hickle as hkl
    import numpy as np

    path = _session_pointer_path(session)
    if not path.exists():
        return 0
    try:
        bundle = hkl.load(str(path))
    except Exception as exc:
        print(f'  [warn] pointer sync {path.name}: {exc}')
        return 0
    if not isinstance(bundle, dict):
        return 0
    n = 0
    for kstr, ptr in list(bundle.items()):
        inds2 = _inds2_from_pointer(ptr)
        if inds2 is None:
            continue
        ptr_pairs = {tuple(int(x) for x in row) for row in inds2}
        ptr_sel = {tuple(int(x) for x in p)
                   for p in selections_by_type.get(kstr, [])} & ptr_pairs
        selected_inds = (np.array(sorted(ptr_sel), dtype=int)
                         if ptr_sel else None)
        if isinstance(ptr, dict):
            ptr['selected_inds'] = selected_inds
        else:
            ptr.selected_inds = selected_inds
        bundle[kstr] = ptr
        n += 1
    hkl.dump(bundle, str(path))
    return n


def _remove_per_type_pointers() -> int:
    """Drop per-conn-type pointer files (superseded by per-session bundles)."""
    removed = 0
    for path in DST_CCG.glob(f'{PROJECT}_lowres_ccgpointers_*__*.hkl'):
        path.unlink()
        removed += 1
        print(f'  [clean] removed {path.name}')
    return removed


def migrate_selections() -> int:
    import datetime

    DST_SEL.mkdir(parents=True, exist_ok=True)
    n = 0
    for fpath in sorted(SRC_SEL.glob("*__latest.json")):
        old = json.loads(fpath.read_text(encoding='utf-8'))
        session = old.get('session', fpath.stem.split('__')[0])

        deleted, seen_del = [], set()
        for pairs in old.get('deleted_by_type', {}).values():
            for p in pairs:
                t = (int(p[0]), int(p[1]))
                if t not in seen_del:
                    seen_del.add(t)
                    deleted.append([t[0], t[1]])
        for p in old.get('deleted', []):
            t = (int(p[0]), int(p[1]))
            if t not in seen_del:
                seen_del.add(t)
                deleted.append([t[0], t[1]])
        deleted.sort()

        deleted_by_type = old.get('deleted_by_type', {})
        pair_to_ct = build_pair_to_ct(old)
        pair_to_ct.update(pair_to_ct_from_pointers(session))
        pair_tags = flat_tags_to_nested_by_ct(old.get('pair_tags', {}), pair_to_ct)
        selections_by_type = enrich_selections_from_tags(
            session, old.get('selections', {}), pair_tags, pair_to_ct)
        selected = _selected_union(selections_by_type)
        new = {
            'session':         session,
            'saved_at':        old.get('saved_at', datetime.datetime.now().isoformat()),
            'selections':      selections_by_type,
            'deleted_by_type': deleted_by_type,
            'selected':        selected,
            'deleted':         deleted,
            'pair_tags':       pair_tags,
        }
        dst = DST_SEL / f"{session}.json"
        tmp = str(dst) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(new, f, indent=2, ensure_ascii=False)
        os.replace(tmp, dst)
        n_ptr = sync_pointer_selected_inds(session, selections_by_type)
        n_tags = _count_nested_tag_pairs(pair_tags)
        print(f"[sel] {session}.json — {len(selected)} selected, "
              f"{len(deleted)} deleted, {n_tags} tagged pairs, "
              f"{n_ptr} pointers synced")
        n += 1

    if SRC_GROUPS.exists():
        shutil.copy2(SRC_GROUPS, DST_SEL / 'groups.json')
        print("[sel] groups.json copied from data/test/selections/")
    return n


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--selections-only',
        action='store_true',
        help='Re-migrate selection JSON + pointer selected_inds only (skip CCG).',
    )
    ap.add_argument(
        '--pointers-only',
        action='store_true',
        help='Re-migrate pointer bundles from monolithic source (skip CCG + selections).',
    )
    ap.add_argument(
        '--repack-pointers',
        action='store_true',
        help='Merge per-conn-type pointer files into per-session bundles.',
    )
    ap.add_argument(
        '--reorganize-only',
        action='store_true',
        help='Move flat ccg/ files into ccgdata/, pointers/, config/ subdirs.',
    )
    args = ap.parse_args()

    if args.reorganize_only:
        reorganize_flat_ccg()
        return

    if not SRC_SEL.is_dir() and not args.repack_pointers:
        print(f"Selections source missing: {SRC_SEL}")
        sys.exit(1)

    print(f"Migrating → {DST_ROOT}\n")

    if args.repack_pointers:
        n = repack_per_type_pointers()
        if n:
            n_clean = _remove_per_type_pointers()
            print(f"[pointers] removed {n_clean} per-type files")
        else:
            print("[pointers] nothing repacked — per-type files kept")
        print(f"\nDone: repacked {n} pointers")
        return

    if args.pointers_only:
        if not SRC_CCG.is_dir():
            print(f"CCG source missing: {SRC_CCG}")
            sys.exit(1)
        n_ptr = migrate_pointers()
        print(f"\nDone: {n_ptr} pointers in session bundles (pointers-only)")
        return

    if args.selections_only:
        n_clean = _remove_per_type_pointers()
        if n_clean:
            print(f"[pointers] removed {n_clean} per-type files\n")
        n_sel = migrate_selections()
        print(f"\nDone: {n_sel} selection files (selections-only)")
        return

    if not SRC_CCG.is_dir():
        print(f"CCG source missing: {SRC_CCG}")
        sys.exit(1)

    conf = _build_config()
    print(f"Config: name={conf.name}  correction={conf.multiple_correction}  "
          f"duration={conf.duration}s  bin={conf.bin_size}\n")

    n_ccg = migrate_ccgdata()
    n_ptr = migrate_pointers(conf)
    n_clean = _remove_per_type_pointers()
    if n_clean:
        print(f"[pointers] removed {n_clean} per-type files")
    save_config_and_meta(conf)
    n_sel = migrate_selections()
    reorganize_flat_ccg()

    print(f"\nDone: {n_ccg} ccgdata files, {n_ptr} pointers, {n_sel} selection files")


if __name__ == '__main__':
    main()
