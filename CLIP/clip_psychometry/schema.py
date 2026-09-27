"""Portable, explicit item identities; chapter 2 and chapter 13."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n')


@dataclass(frozen=True)
class Column:
    id: str
    text: str
    cue: str
    channel: str
    roles: tuple[str, ...] = ()
    polarity: int = 1  # For an explicit reflective audit, not regression signing.
    kind: str = 'unspecified'
    criterion: str | None = None
    sources: tuple[str, ...] = ()


@dataclass
class Registry:
    items: tuple[Column, ...]
    # If a role is removed, also remove these declared conjunction/proxy roles.
    dependents: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        if len({c.id for c in self.items}) != len(self.items):
            raise ValueError('Column IDs must be unique across channels.')
        if any(not c.id or not c.text or not c.cue or not c.channel or c.polarity not in (-1, 1) for c in self.items):
            raise ValueError('Every column needs identity, text, cue, channel, and polarity ±1.')

    @property
    def roles(self):
        return sorted({r for c in self.items for r in (c.cue, *c.roles)})

    def columns(self, roles, *, closure=False):
        selected = {roles} if isinstance(roles, str) else set(roles)
        unknown = selected-set(self.roles)
        if unknown:
            raise ValueError(f'Unknown roles: {sorted(unknown)}')
        if closure:
            while True:
                expanded = selected | {r for s in selected for r in self.dependents.get(s, [])}
                if expanded == selected:
                    break
                selected = expanded
        return tuple(i for i, c in enumerate(self.items) if selected.intersection((c.cue, *c.roles)))


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray  # Missing labels are NaN, never numeric zero placeholders.
    targets: tuple[str, ...]
    paths: tuple[str, ...]
    groups: np.ndarray
    registry: Registry
    ids: tuple[str, ...]
    metadata: dict = field(default_factory=dict)
    se: np.ndarray | None = None

    def __post_init__(self):
        self.x, self.y = np.asarray(self.x, float), np.asarray(self.y, float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        self.groups = np.asarray(self.groups, dtype=str)
        n = len(self.x)
        if self.x.ndim != 2 or self.x.shape[1] != len(self.registry.items) or not np.isfinite(self.x).all():
            raise ValueError('Finite X[n, columns] must match the registry exactly.')
        if self.y.shape != (n, len(self.targets)) or np.isinf(self.y).any():
            raise ValueError('Y[n, targets] must be finite or NaN.')
        if any(len(v) != n for v in (self.paths, self.groups, self.ids)) or len(set(self.ids)) != n:
            raise ValueError('Paths, groups and unique IDs must align with rows.')
        if self.groups.shape != (n,) or len(set(self.targets)) != len(self.targets):
            raise ValueError('Groups must be one-dimensional; target names must be unique.')
        if self.se is not None:
            self.se = np.asarray(self.se, float)
            if self.se.shape != self.y.shape or np.any(self.se[np.isfinite(self.se)] < 0):
                raise ValueError('SEM must be nonnegative and align with Y.')

    def subset(self, rows, *, exposure=None):
        rows = np.asarray(rows, dtype=int)
        meta = dict(self.metadata)
        if exposure:
            meta['exposure'] = exposure
        return Dataset(self.x[rows], self.y[rows], self.targets,
                       tuple(self.paths[i] for i in rows), self.groups[rows], self.registry,
                       tuple(self.ids[i] for i in rows), meta,
                       None if self.se is None else self.se[rows])

    def fingerprint(self):
        h = hashlib.sha256()
        for a in (self.x, self.y, self.groups, np.asarray(self.ids)):
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(json.dumps([asdict(c) for c in self.registry.items], sort_keys=True).encode())
        h.update(json.dumps(self.registry.dependents, sort_keys=True).encode())
        h.update(json.dumps(self.targets).encode())
        return h.hexdigest()

    def save(self, path):
        manifest = dict(format='clip_psychometry_v1', columns=[asdict(c) for c in self.registry.items],
                        dependents=self.registry.dependents, targets=self.targets, metadata=self.metadata)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=self.x, y=self.y, paths=np.asarray(self.paths),
                            groups=self.groups, ids=np.asarray(self.ids),
                            se=np.full_like(self.y, np.nan) if self.se is None else self.se,
                            manifest=np.asarray(json.dumps(json_value(manifest))))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            m = json.loads(str(d['manifest']))
            if m['format'] != 'clip_psychometry_v1':
                raise ValueError('Unknown dataset format.')
            cols = tuple(Column(**{**c, 'roles': tuple(c.get('roles', [])),
                                   'sources': tuple(c.get('sources', []))}) for c in m['columns'])
            return cls(d['x'], d['y'], tuple(m['targets']), tuple(d['paths'].tolist()), d['groups'],
                       Registry(cols, m.get('dependents', {})), tuple(d['ids'].tolist()),
                       m.get('metadata', {}), d['se'])
