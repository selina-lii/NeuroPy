"""NWBSession — duck-type shim over NWBFile that satisfies the ProcessData interface.

Usage::

    from neuropy.core.nwb_session import NWBSession
    from neuropy.analyses.neurons_dataset import NeuronsDataset, NeuronsDatasetConfig
    from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig
    from neuropy.ui.ccg_review import CCGReviewUI as _CCGReviewUI
def launch_ccg_review(cd, key=None): return _CCGReviewUI.launch(cd, key)

    sess = [NWBSession("recording.nwb")]
    nd = NeuronsDataset(sess, NeuronsDatasetConfig())
    cd = CCGDataset(CCGConfig(duration=20e-3, alpha=0.05, name='test'), nd)
    launch_ccg_review(cd)
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from neuropy.core.neurons import Neurons
from neuropy.core.epoch import Epoch
from neuropy.io.fieldmap import FieldMap
from neuropy.io.nwbio import NWB_DEFAULT, NWBFile

# Fallback clock for quantizing second-valued spike times, used only when the file's own
# rate can't be recovered. 30 kHz is a typical acquisition rate and the lowest clock that
# keeps the 33 us highres bin from collapsing to zero (`int(rate * bin_size)` must be >= 1).
SPIKE_CLOCK_HZ = 30_000


class NWBSession:
    """Wraps a single .nwb file to look like a ProcessData session object.

    NeuronsDataset accesses: basepath, filePrefix, neurons, themes, recinfo (optional).
    All other ProcessData attributes are absent by design.
    """

    def __init__(
        self,
        nwb_path: str | Path,
        session_name: str | None = None,
        fields: dict = None,
        sampling_rate: float = None,
        drop_absent: bool = False,
    ):
        """
        Parameters
        ----------
        nwb_path : path to the .nwb file
        session_name : name used as session identifier; defaults to file stem
        fields : per-dataset field map, e.g. io.dandi_001695.FIELDS; None = NWB_DEFAULT
        sampling_rate : clock the spike times quantize onto; derived from them when omitted
        drop_absent : unbind optional fields whose columns this file lacks
        """
        self._path = Path(nwb_path)
        self.session_name = session_name or self._path.stem
        self._nwb = NWBFile(self._path, fields=fields, drop_absent=drop_absent)
        # a stated rate beats one recovered from spike gaps
        self.sampling_rate = (sampling_rate or self._nwb.declared_sampling_rate
                              or self._nwb.sampling_rate or SPIKE_CLOCK_HZ)
        self.themes    # binds each intervals table as an attribute, before anything discovers them

    # ── ProcessData path attributes ────────────────────────────────────

    @property
    def basepath(self) -> Path:
        return self._path.parent

    @property
    def filePrefix(self) -> Path:
        return self._path.parent / self.session_name

    # ── recinfo (optional) ────────────────────────────────────────────

    @cached_property
    def recinfo(self):
        class _MinimalRecinfo:
            skipped_channels = np.array([], dtype=int)
        return _MinimalRecinfo()

    # ── Core data ─────────────────────────────────────────────────────

    @cached_property
    def neurons(self) -> Neurons:
        nwb = self._nwb
        spiketrains = nwb.spiketrains
        if not spiketrains:
            raise ValueError(f"NWB file has no units: {self._path}")
        return Neurons(
            spiketrains=spiketrains,
            t_stop=nwb.t_stop,
            t_start=0.0,
            sampling_rate=self.sampling_rate,
            neuron_ids=nwb.neuron_ids,
            neuron_type=nwb.neuron_type,
            waveforms=nwb.waveforms,
            peak_channels=nwb.peak_channels,
            shank_ids=nwb.shank_ids,
            metadata={**{k: v for k, v in (('positions', nwb.positions),
                                           ('cell_area', nwb.cell_area))
                         if v is not None},
                      **nwb.extra_columns} or None,
        )

    @cached_property
    def themes(self) -> dict:
        """Every intervals table in the file as an Epoch, keyed by table name."""
        found = {}
        for name in self._nwb.interval_tables:
            df = self._nwb.intervals_df(name)
            if df is not None and not df.empty:
                found[name] = Epoch(df)
                setattr(self, name, found[name])   # NeuronsDataset discovers Epoch attributes
        return found

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def columns(self) -> list:
        """The units-table columns this file offers."""
        return self._nwb.input_fields

    def close(self):
        self._nwb.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"NWBSession('{self.session_name}', path='{self._path}')"


class NWBDataset:
    """A folder of .nwb files scanned as one unit — one file is one session."""

    def __init__(self, path: str | Path, pattern: str = '**/*.nwb', on_progress=None,
                 naming=None):
        self.path = Path(path)
        self.naming = naming or (lambda p: p.stem)
        self._paths = sorted(self.path.glob(pattern)) if self.path.is_dir() else [self.path]
        self.fields_by_file: dict[Path, list] = {}
        self.themes_by_file: dict[Path, dict] = {}
        self.values_by_file: dict[Path, dict] = {}
        if on_progress is not None:
            self.scan(on_progress)

    def scan(self, on_progress=None) -> None:
        """Read every file's columns, themes and label sets — what the mapping widget offers."""
        for i, p in enumerate(self._paths):
            if on_progress is not None:
                on_progress(i, len(self._paths), p)
            self._scan_file(p)

    def _scan_file(self, path: Path) -> list:
        """One file's columns, caching its themes and label sets alongside."""
        if path not in self.fields_by_file:
            with NWBFile(path, fields=NWB_DEFAULT) as f:
                self.fields_by_file[path] = f.input_fields
                self.themes_by_file[path] = f.interval_labels
                self.values_by_file[path] = {c: f.column_values(c)
                                             for c in f.input_fields if f.is_categorical(c)}
        return self.fields_by_file[path]

    def column_values(self, column: str) -> list:
        """Every distinct value *column* takes across the scanned files."""
        self.scan()
        return self._union(v.get(column, []) for v in self.values_by_file.values())

    @property
    def files(self) -> list:
        return list(self._paths)

    @staticmethod
    def _union(per_file) -> list:
        """Every entry any file offers, in first-seen order."""
        seen = {}
        for entries in per_file:
            seen.update(dict.fromkeys(entries))
        return list(seen)

    @property
    def input_fields(self) -> list:
        """Every column any file offers, in first-seen order."""
        self.scan()
        return self._union(self.fields_by_file.values())

    @property
    def themes(self) -> list:
        """Every intervals-table name any file offers — the theme choices."""
        self.scan()
        return self._union(self.themes_by_file.values())

    @staticmethod
    def _missing(field_map: FieldMap, columns) -> list:
        """Target fields the given columns cannot supply."""
        return [name for name, b in field_map.bindings.items()
                if any(c not in columns for c in b.columns)]

    def coverage(self, field_map: FieldMap) -> dict:
        """Per file, the target fields it cannot supply — empty list means fully mapped."""
        return {p: self._missing(field_map, self._scan_file(p)) for p in self._paths}

    def _usable(self, field_map: FieldMap) -> dict:
        """Files whose *required* fields are all present, mapped to their missing optionals."""
        required = {f.name for f in field_map.schema if f.required}
        return {p: missing for p, missing in self.coverage(field_map).items()
                if not required & set(missing)}

    def sessions(self, field_map: FieldMap, sampling_rate: float = None,
                 overrides: dict = None) -> list:
        """One NWBSession per usable file; *overrides* names the sessions on a different clock."""
        overrides = overrides or {}
        required = {f.name for f in field_map.schema if f.required}
        out = []
        for path in self._paths:
            name = self.naming(path)
            # each session narrows the map against its own open handle: scanning here would reopen every file
            session = NWBSession(path, session_name=name,
                                 sampling_rate=overrides.get(name, sampling_rate),
                                 fields=field_map, drop_absent=True)
            if required & set(self._missing(field_map, session.columns)):
                session.close()
                continue
            out.append(session)
        return out

    def report(self, field_map: FieldMap) -> str:
        """Which files load whole, which load partial, and which are skipped."""
        usable = self._usable(field_map)
        skipped = {p: m for p, m in self.coverage(field_map).items() if p not in usable}
        partial = {p: m for p, m in usable.items() if m}
        lines = [f"{len(usable)} of {len(self.fields_by_file)} sessions usable"]
        for title, group in [("Partial", partial), ("Skipped", skipped)]:
            if group:
                lines.append(f"\n{title}:")
                lines += [f"  {p.name}  missing {sorted(m)}" for p, m in group.items()]
        return '\n'.join(lines)
