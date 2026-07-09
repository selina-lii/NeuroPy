"""migrate_ccg.py — convert first-pass CCG data from hkl → npz + copy suggestions.

Run with the NeuroPy2 conda env (needs hickle + h5py):
  /Users/selinl/miniforge3/envs/NeuroPy2/bin/python scripts/migrate_ccg.py

Output:
  data_migration/ccg/{dataset_name}/  — npz files, one per session × resolution
  data_migration/custom_ccg/          — suggested_custom_ccgs.json (copied as-is)
"""

from __future__ import annotations
import io
import json
import os
import pickle
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT         = Path(__file__).resolve().parents[1]
SRC_CCG      = ROOT / "data" / "ccg"
DST_CCG      = ROOT / "data_migration" / "ccg"
SRC_CUSTOM   = ROOT / "data" / "custom_ccg"
DST_CUSTOM   = ROOT / "data_migration" / "custom_ccg"


# ── Stub unpickler — tolerates missing/renamed classes ────────────────────────

class _Stub:
    """Stand-in for any class that no longer exists."""
    def __init__(self, *a, **kw): pass

def _stub_unpickle(data: bytes):
    class StubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError, ModuleNotFoundError):
                return _Stub
    return StubUnpickler(io.BytesIO(data)).load()


def _load_hkl(path: Path):
    """Load a hickle file, returning either a single CCGData or list of them."""
    with h5py.File(str(path), 'r') as f:
        data_node = f.get('data')
        if data_node is None:
            return None

        # Format A: /data is a dataset (single pickled blob) — per-session file
        if hasattr(data_node, 'shape') and data_node.shape:
            return _stub_unpickle(bytes(data_node[()]))

        # Format B: /data is a group of dataN sub-groups — monolithic dict file
        # Each sub-group has data0=Key bytes, data1=CCGData bytes
        results = []
        for entry_key in sorted(data_node.keys()):
            grp = data_node[entry_key]
            if not hasattr(grp, 'keys'):
                continue
            # We only need data1 (the CCGData value, not the Key)
            payload_key = 'data1' if 'data1' in grp else None
            if payload_key is None:
                continue
            try:
                obj = _stub_unpickle(bytes(grp[payload_key][()]))
                results.append(obj)
            except Exception as e:
                print(f"  [warn] entry {entry_key}: {e}")
        return results if results else None


# ── Per-session npz writer ────────────────────────────────────────────────────

def _extract_arrays(obj) -> dict | None:
    """Pull the four core arrays out of a CCGData (or stub) object."""
    if isinstance(obj, _Stub) or not hasattr(obj, '__dict__'):
        return None
    v = vars(obj)
    ccg = v.get('ccg')
    if ccg is None or not isinstance(ccg, np.ndarray):
        return None
    qval = v.get('qval') if v.get('qval') is not None else v.get('qval_corrected')
    return {
        'ccg':      np.asarray(ccg,          dtype=np.float32),
        'ccg_null': np.asarray(v['ccg_null'], dtype=np.float32) if v.get('ccg_null') is not None else None,
        'pval':     np.asarray(v['pval'],     dtype=np.float64) if v.get('pval')     is not None else None,
        'qval':     np.asarray(qval,          dtype=np.float64) if qval               is not None else None,
    }


def _session_from_key(obj) -> str:
    """Extract session string from a CCGData key attribute."""
    key = getattr(obj, 'key', None) or getattr(obj, '_key', None)
    if key is None:
        return 'unknown'
    return str(getattr(key, 'session', None) or getattr(key, 'nd_key', None) or 'unknown')


def _write_npz(dst_dir: Path, dataset_name: str, session: str, resolution: str, arrays: dict):
    dst_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset_name}_ccgdata_{session}__{resolution}.npz"
    out   = {k: v for k, v in arrays.items() if v is not None}
    np.savez_compressed(str(dst_dir / fname), **out)
    shapes = {k: v.shape for k, v in out.items()}
    print(f"    → {fname}  {shapes}")


# ── Dataset migration ─────────────────────────────────────────────────────────

def migrate_dataset(src_dir: Path, dst_dir: Path):
    name = src_dir.name
    print(f"\n[{name}]")

    # Find per-session ccgdata hkl files (preferred over monolithic)
    per_sess = sorted(src_dir.glob(f"*_ccgdata_sess_*.hkl"))
    mono     = sorted(src_dir.glob(f"*_ccgdata.hkl"))
    hi_sess  = sorted(src_dir.glob(f"*_highres_sess_*.hkl"))
    hi_mono  = sorted(src_dir.glob(f"*_highres.hkl"))

    def _migrate_files(files, resolution):
        if not files:
            return
        for fpath in files:
            print(f"  loading {fpath.name} ...")
            obj = _load_hkl(fpath)
            if obj is None:
                print(f"  [skip] could not load {fpath.name}")
                continue
            # Single CCGData object
            if hasattr(obj, 'ccg') and isinstance(getattr(obj, 'ccg', None), np.ndarray):
                arrays = _extract_arrays(obj)
                if arrays:
                    sess = _session_from_key(obj)
                    _write_npz(dst_dir, name, sess, resolution, arrays)
            # List of CCGData objects (monolithic file)
            elif isinstance(obj, list):
                for item in obj:
                    if hasattr(item, 'ccg') and isinstance(getattr(item, 'ccg', None), np.ndarray):
                        arrays = _extract_arrays(item)
                        if arrays:
                            sess = _session_from_key(item)
                            _write_npz(dst_dir, name, sess, resolution, arrays)
            else:
                print(f"  [skip] unexpected type {type(obj).__name__} in {fpath.name}")

    if per_sess:
        _migrate_files(per_sess, 'lowres')
    elif mono:
        _migrate_files(mono, 'lowres')
    else:
        print("  [skip] no lowres ccgdata hkl found")

    if hi_sess:
        _migrate_files(hi_sess, 'highres')
    elif hi_mono:
        _migrate_files(hi_mono, 'highres')

    # Copy meta.json files as-is
    dst_dir.mkdir(parents=True, exist_ok=True)
    for meta in src_dir.glob("*.meta.json"):
        shutil.copy2(meta, dst_dir / meta.name)
        print(f"  copied {meta.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # CCG datasets
    dataset_dirs = [d for d in SRC_CCG.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print("No CCG dataset directories found.")
    for src_dir in sorted(dataset_dirs):
        dst_dir = DST_CCG / src_dir.name
        migrate_dataset(src_dir, dst_dir)

    # suggested_custom_ccgs.json — copy as-is
    DST_CUSTOM.mkdir(parents=True, exist_ok=True)
    src_sugg = SRC_CUSTOM / "suggested_custom_ccgs.json"
    if src_sugg.exists():
        shutil.copy2(src_sugg, DST_CUSTOM / "suggested_custom_ccgs.json")
        with open(src_sugg) as f:
            n = len(json.load(f).get('items', []))
        print(f"\n[custom_ccg] suggested_custom_ccgs.json copied ({n} items)")
    else:
        print("\n[custom_ccg] suggested_custom_ccgs.json not found, skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
