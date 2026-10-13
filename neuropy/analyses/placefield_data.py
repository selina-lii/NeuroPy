"""Per-neuron 1D place-field peak stats, stored per (session, segment) Key as one array per field."""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.utils import JsonSavable


def _num_token(x: float) -> str:
    return f"{x:g}".replace('.', '_')


class PlacefieldConfig(JsonSavable):
    """Place-field run parameters; its stem names every file the run writes."""

    STEM = re.compile(r'placefield_(?P<name>.+)\.bin(?P<grid_bin>[\d_]+)\.sigmabin(?P<sigma_bin>[\d_]+)$')

    def __init__(self, name: str = 'grp_stats', grid_bin: float = 5.0, sigma_bin: float = 1.5,
                 frmax_threshold: float = 3.0, segment: str = 'maze', root: str = ''):
        JsonSavable.__init__(self)
        self.name = name
        self.grid_bin = grid_bin
        self.sigma_bin = sigma_bin
        self.frmax_threshold = frmax_threshold   # Hz
        self.segment = segment
        self._root = root

    @classmethod
    def from_stem(cls, stem: str, root: str) -> 'PlacefieldConfig':
        m = cls.STEM.match(stem)
        return cls(m['name'], float(m['grid_bin'].replace('_', '.')),
                   float(m['sigma_bin'].replace('_', '.')), root=root)

    @property
    def stem(self) -> str:
        return f"placefield_{self.name}.bin{_num_token(self.grid_bin)}.sigmabin{_num_token(self.sigma_bin)}"

    def path(self, ext: str) -> str:
        return os.path.join(self._root, self.stem + ext)

    def save_path(self, **_) -> str:
        return self.path('.meta')


class PlacefieldData:
    """Peak rows per Key(session, segment); cell_id is the neuron id."""

    FIELDS = ('cell_id', 'peak_no', 'height', 'prominence', 'center_bin', 'width_bin',
              'left_edge', 'right_edge', 'r_odd_even', 'r_maze_remaze', 'FRmax',
              'FRmax_other_session', 'direction', 'sid')
    NEURON_STATS = ('FRmax', 'r_odd_even', 'r_maze_remaze')

    def __init__(self, conf: PlacefieldConfig, fields: dict = None):
        self.conf = conf
        self.fields: dict = fields or {}

    @classmethod
    def from_csv(cls, path: str) -> 'PlacefieldData':
        """Source table: one row per peak, session named by animal/day/grp, segment by session."""
        conf = PlacefieldConfig.from_stem(os.path.splitext(os.path.basename(path))[0],
                                          os.path.dirname(path))
        df = pd.read_csv(path, index_col=0)
        return cls(conf, {Key(session=f"{a}_{d}{g}", segment=s.lower()):
                          {f: sub[f].to_numpy() for f in cls.FIELDS}
                          for (a, d, g, s), sub in df.groupby(['animal', 'day', 'grp', 'session'])})

    @classmethod
    def load(cls, conf: PlacefieldConfig) -> 'PlacefieldData':
        conf.load()
        raw = np.load(conf.path('.npy'), allow_pickle=True).item()
        return cls(conf, {Key.from_str(k): v for k, v in raw.items()})

    @classmethod
    def load_dir(cls, root: str) -> 'PlacefieldData':
        """The run saved in *root*, ingesting its source CSV on first use; empty when there is none."""
        for meta in sorted(glob.glob(os.path.join(root, 'placefield_*.meta.json'))):
            return cls.load(PlacefieldConfig.from_stem(os.path.basename(meta)[:-len('.meta.json')], root))
        for csv in sorted(glob.glob(os.path.join(root, 'placefield_*.csv'))):
            data = cls.from_csv(csv)
            data.save()
            return data
        return cls(PlacefieldConfig(root=root))

    def save(self) -> None:
        os.makedirs(self.conf._root, exist_ok=True)
        self.conf.save()
        np.save(self.conf.path('.npy'), {k.json_str(): v for k, v in self.fields.items()},
                allow_pickle=True)

    @property
    def n_bins(self) -> float:
        """Track length in bins: the farthest field edge seen anywhere."""
        return max((float(np.nanmax(t[f])) for t in self.fields.values()
                    for f in ('center_bin', 'right_edge')), default=1.0)

    def segments(self) -> list:
        return sorted({str(k.segment) for k in self.fields})

    def has_session(self, session: str) -> bool:
        return any(str(k.session) == session for k in self.fields)

    def neuron_rows(self, session: str, nid: int, segment: str = None) -> dict:
        """*nid*'s peak rows in one segment, field → array; empty arrays when it has none."""
        table = self.fields.get(Key(session=session, segment=segment or self.conf.segment))
        if table is None:
            return {f: np.array([]) for f in self.FIELDS}
        rows = table['cell_id'] == nid
        return {f: v[rows] for f, v in table.items()}

    def neuron_stat(self, session: str, nid: int, stat: str, segment: str = None) -> float:
        """A per-neuron stat (max over directions); NaN when the neuron has no fields."""
        values = self.neuron_rows(session, nid, segment)[stat]
        return float(np.nanmax(values)) if len(values) else np.nan

    def place_cell_mask(self, session: str, segments: list, neuron_ids) -> np.ndarray:
        """Per neuron: FRmax reaches the threshold in any of *segments*."""
        hit = np.zeros(len(neuron_ids), dtype=bool)
        for seg in segments:
            table = self.fields.get(Key(session=session, segment=seg))
            if table is not None:
                strong = table['cell_id'][table['FRmax'] >= self.conf.frmax_threshold]
                hit |= np.isin(neuron_ids, strong)
        return hit


class PlacefieldCurveData:
    """Saved unthresholded 1D tuning curves per (session, block, direction), one ``<session>.npz`` each."""

    FIELDS = ('neuron_ids', 'coords', 'tuning_curves', 'split_curves')
    DIRECTIONS = ('up', 'down')
    UNOCCUPIED_HZ = 1e9   # Pf1D pads occupancy by 1e-16 s, so a spike in a bin never run through reads ~1e16 Hz

    def __init__(self, curves: dict = None):
        self.curves: dict = curves or {}

    @classmethod
    def load_dir(cls, root: str) -> 'PlacefieldCurveData':
        curves = {}
        for path in sorted(glob.glob(os.path.join(root, '*.npz'))):
            session = os.path.basename(path)[:-len('.npz')]
            with np.load(path) as z:
                for block, direction in {tuple(k.split('.')[:2]) for k in z.files}:
                    curves[(session, block, direction)] = {f: z[f"{block}.{direction}.{f}"] for f in cls.FIELDS}
        return cls(curves)

    def blocks(self) -> list:
        return sorted({b for _, b, _ in self.curves})

    def has_session(self, session: str) -> bool:
        return any(s == session for s, _, _ in self.curves)

    def n_bins(self, session: str) -> int:
        return max((len(c['coords']) for (s, _, _), c in self.curves.items() if s == session), default=1)

    def curve(self, session: str, block: str, direction: str, nid: int, sigma_bin: float):
        """``(curve, (even, odd))`` of one neuron, smoothed by *sigma_bin* bins; None when it was not computed."""
        c = self.curves.get((session, block, direction))
        if c is None or nid not in c['neuron_ids']:
            return None
        i = int(np.flatnonzero(c['neuron_ids'] == nid)[0])
        return self._smooth(c['tuning_curves'][i], sigma_bin), tuple(self._smooth(h[i], sigma_bin) for h in c['split_curves'])

    @classmethod
    def _smooth(cls, rate: np.ndarray, sigma_bin: float) -> np.ndarray:
        """Gaussian-smoothed over occupied bins only; unoccupied bins stay NaN."""
        seen = rate < cls.UNOCCUPIED_HZ
        if sigma_bin <= 0:
            return np.where(seen, rate, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = gaussian_filter1d(np.where(seen, rate, 0.0), sigma_bin) / gaussian_filter1d(seen.astype(float), sigma_bin)
        return np.where(seen, out, np.nan)

    def neuron_stats(self, session: str, block: str, nid: int, sigma_bin: float) -> dict:
        """FRmax, r_odd_even and r_maze_remaze, each the max over directions; NaN where nothing was computed."""
        other = {'maze': 'remaze', 'remaze': 'maze'}.get(block)
        out = {'FRmax': [], 'r_odd_even': [], 'r_maze_remaze': []}
        for d in self.DIRECTIONS:
            here = self.curve(session, block, d, nid, sigma_bin)
            if here is None:
                continue
            tc, (even, odd) = here
            out['FRmax'].append(np.nanmax(tc))
            out['r_odd_even'].append(self._corr(even, odd))
            there = self.curve(session, other, d, nid, sigma_bin) if other else None
            if there is not None:   # blocks bin their own track span: resample onto this one's bins
                out['r_maze_remaze'].append(self._corr(
                    tc, np.interp(np.linspace(0, 1, len(tc)), np.linspace(0, 1, len(there[0])), there[0])))
        with np.errstate(invalid='ignore'):
            return {k: float(np.nanmax(v)) if v and not np.all(np.isnan(v)) else np.nan for k, v in out.items()}

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson r over bins both curves occupy; NaN when either is flat there."""
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
            return np.nan
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
