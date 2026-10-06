"""Per-neuron 1D place-field peak stats, stored per (session, segment) Key as one array per field."""
import glob
import os
import re

import numpy as np
import pandas as pd

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
