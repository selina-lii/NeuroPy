"""Dataset field mapping — declare what a loader needs, bind a dataset's columns to it."""
from __future__ import annotations


ONE = '1'            # exactly one source column
OPTIONAL = '0/1'     # zero or one
MANY = '1+'          # one or more
ANY = '0+'           # zero or more

_ARITIES = (ONE, OPTIONAL, MANY, ANY)
_MULTI = (MANY, ANY)
_REQUIRED = (ONE, MANY)


class Field:
    """One target field a loader needs: how many source columns, and whether values remap."""

    def __init__(self, name: str, arity: str = ONE, value_map: bool = False,
                 note: str = '', values: list = None):
        if arity not in _ARITIES:
            raise ValueError(f"{name}: arity must be one of {_ARITIES}, got {arity!r}")
        self.name = name
        self.arity = arity
        self.value_map = value_map
        self.note = note
        self.values = list(values or [])   # what a value map may translate into

    @property
    def required(self) -> bool:
        return self.arity in _REQUIRED

    @property
    def multi(self) -> bool:
        return self.arity in _MULTI

    def __repr__(self) -> str:
        return f"Field({self.name!r}, arity={self.arity!r}, value_map={self.value_map})"


class Binding:
    """What a dataset bound to one Field: its source columns and any value renaming."""

    def __init__(self, field: Field, columns: list, value_map: dict = None):
        self.field = field
        self.columns = columns
        self.value_map = value_map or {}

    @property
    def column(self) -> str:
        """The single source column; for MANY fields use .columns instead."""
        return self.columns[0]

    @property
    def renames(self) -> str:
        """The value transform in readable form, for provenance in reports and headers."""
        shown = [f"{src}->{dst}" for src, dst in list(self.value_map.items())[:3]]
        rest = len(self.value_map) - len(shown)
        return ', '.join(shown) + (f", +{rest} more" if rest > 0 else '')

    def apply(self, values):
        """Rename values through the value map, leaving unlisted ones untouched."""
        return [self.value_map.get(v, v) for v in values]

    def __repr__(self) -> str:
        vm = f", value_map={len(self.value_map)} entries" if self.value_map else ''
        return f"Binding({self.field.name!r} <- {self.columns!r}{vm})"


class FieldSchema:
    """The set of target fields one loader needs."""

    def __init__(self, fields: list):
        self.fields = list(fields)
        self.by_name = {f.name: f for f in self.fields}

    def __iter__(self):
        return iter(self.fields)

    @property
    def names(self) -> list:
        return [f.name for f in self.fields]


class FieldMap:
    """A dataset's mapping dict parsed and checked against a schema."""

    EXTRA = 'extra'   # reserved key: columns carried as metadata, under names of the user's choosing

    def __init__(self, schema: FieldSchema, mapping: dict, partial: bool = False):
        self.schema = schema
        self.mapping = dict(mapping)
        self.extra = {name: self._parse(Field(name, OPTIONAL, value_map=True), value)
                      for name, value in (self.mapping.get(self.EXTRA) or {}).items()}
        self.bindings = {name: self._parse(schema.by_name[name], value)
                         for name, value in self.mapping.items()
                         if name != self.EXTRA and self._known(name)}
        if not partial:   # an editor holds an unfinished map; call check_required() when done
            self.check_required()

    def _known(self, name: str) -> bool:
        if name not in self.schema.by_name:
            raise ValueError(
                f"{name!r} is not a field of this schema; expected one of {self.schema.names}")
        return True

    @staticmethod
    def _parse(field: Field, value) -> Binding:
        """str -> one column; list -> several; dict -> {'col'/'cols', 'map'}."""
        value_map = None
        if isinstance(value, dict):
            columns = value.get('cols', value.get('col'))
            value_map = value.get('map')
        else:
            columns = value
        if isinstance(columns, str):
            columns = [columns]
        columns = list(columns)

        if not columns:
            raise ValueError(f"{field.name!r}: no source column given")
        if len(columns) > 1 and not field.multi:
            raise ValueError(
                f"{field.name!r} takes one source column (arity {field.arity}), got {columns!r}")
        if value_map and not field.value_map:
            raise ValueError(f"{field.name!r} does not support value mapping")
        return Binding(field, columns, value_map)

    def check_required(self) -> None:
        missing = [f.name for f in self.schema
                   if f.required and f.name not in self.bindings]
        if missing:
            raise ValueError(f"required field(s) not mapped: {missing}")

    def check(self, available: list) -> None:
        """Raise if any bound column is absent from *available* (the dataset's real columns)."""
        for name, binding in {**self.bindings, **self.extra}.items():
            for col in binding.columns:
                if col not in available:
                    raise ValueError(
                        f"{name!r} maps to column {col!r}, which the dataset does not have; "
                        f"available: {sorted(available)}")

    def get(self, name: str) -> Binding:
        """The binding for *name*, or None when an optional field went unmapped."""
        return self.bindings.get(name)

    def report(self, available: list = None) -> str:
        """Target/source/status table — the terminal view of this mapping."""
        rows = []
        for field in self.schema:
            binding = self.bindings.get(field.name)
            if binding is None:
                status = '—' if not field.required else 'MISSING'
                source = ''
            else:
                source = ', '.join(binding.columns)
                if binding.value_map:
                    source += f"  (renamed: {binding.renames})"
                elif field.value_map:
                    source += "  (values used as-is)"
                status = 'ok'
                if available is not None:
                    absent = [c for c in binding.columns if c not in available]
                    status = 'ok' if not absent else f"NOT IN DATA: {absent}"
            rows.append((field.name, field.arity, source, status))
        w0 = max(len(r[0]) for r in rows)
        w2 = max(len(r[2]) for r in rows)
        return '\n'.join(f"{n:<{w0}}  {a:<3}  {s:<{w2}}  {st}" for n, a, s, st in rows)
