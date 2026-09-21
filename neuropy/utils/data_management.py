"""Naming and integrity for on-disk CCG units; data objects supply paths, never check them."""
import json
import os
import re

SINGLE_SAMPLE = 'single_sample'
NPY_HEADER_MAX = 128
_TOKEN_RE = re.compile(r'bin_size-(single_sample|[0-9]*\.?[0-9]+ms)')


def bin_token(bin_size: float, sampling_rate: float = None) -> str:
    """Directory token naming a unit by the bin size it holds."""
    if sampling_rate and abs(bin_size - 1.0 / sampling_rate) < 1e-9:
        return f'bin_size-{SINGLE_SAMPLE}'
    return f'bin_size-{bin_size * 1e3:g}ms'


def parse_token(token: str, sampling_rate: float = None) -> float | None:
    """Bin size in seconds from a token, or None when it names no size."""
    m = _TOKEN_RE.search(token)
    if m is None:
        return None
    if m.group(1) == SINGLE_SAMPLE:
        return None if not sampling_rate else 1.0 / sampling_rate
    return float(m.group(1)[:-2]) / 1e3


def split_unit_name(name: str) -> tuple[str, str]:
    """(stem, token) for a unit dir; token is '' when the name carries none."""
    m = _TOKEN_RE.search(name)
    return (name[:m.start()].rstrip('.'), m.group(0)) if m else (name, '')


def meta_name(unit_dir: str) -> str:
    """A unit's meta json is named for the unit, not a fixed filename."""
    return split_unit_name(os.path.basename(str(unit_dir).rstrip('/')))[0] + '.json'


def read_meta(path: str) -> dict | None:
    f = os.path.join(path, meta_name(path))
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def _has_data(path: str) -> bool:
    """A torn write leaves the npy header with no array behind it."""
    return os.path.isfile(path) and os.path.getsize(path) > NPY_HEADER_MAX


def is_complete(path: str) -> bool:
    """A unit needs its four arrays and the meta that declares what they are."""
    return (read_meta(path) is not None
            and all(_has_data(os.path.join(path, f'{a}.npy'))
                    for a in ('ccg', 'ccg_null', 'pval', 'qval')))
