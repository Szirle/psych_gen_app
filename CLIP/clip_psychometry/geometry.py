"""Separate proximity coordinates from varying-coefficient coordinates.

Research hypothesis/protocol: research/clip_psychometry/iteration_01/PROTOCOL.md.
Products of PSD kernels remain PSD; these are kernel varying-coefficient models,
not ordinary locally weighted least squares or an optimal-transport algorithm.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from .evaluation import group_splits, gram
from .metrics import regression, paired_contrast
from .schema import write_json


@dataclass(frozen=True)
class GeometryRecipe:
    family: str
    gamma: float = .1
    rank: int = 2
    whiten: bool = False
    alpha: float = .1


class RoleGeometry:
    def fit(self, x, roles, rank=2, whiten=False):
        self.mean = x.mean(0)
        self.scale = np.where(x.std(0) > 1e-12, x.std(0), 1.)
        z = (x-self.mean)/self.scale
        self.parts = []
        for role in sorted(set(roles)):
            columns = np.flatnonzero(np.asarray(roles) == role)
            values, vectors = eigh(z[:, columns].T@z[:, columns]/len(z), check_finite=False)
            take = np.argsort(values)[::-1][:min(rank, len(columns))]
            variance = values[take]
            good = variance > 1e-10
            take, variance = take[good], variance[good]
            if not len(take):
                continue
            basis = vectors[:, take]
            divisor = np.sqrt(variance) if whiten else np.full(len(take), np.sqrt(variance.mean()))
            # Each role contributes total expected squared norm one, independent of rank.
            self.parts.append((columns, basis/(divisor*np.sqrt(len(take)))))
        if not self.parts:
            raise ValueError('All role coordinates are constant.')
        return self

    def transform(self, x):
        z = (x-self.mean)/self.scale
        factor = np.concatenate([z[:, cols]@basis for cols, basis in self.parts], axis=1)
        # gram divides by feature width: convert to mean squared role distance.
        factor *= np.sqrt(factor.shape[1]/len(self.parts))
        return z, factor


def predictions(train, y, query, roles, recipes):
    result = np.empty((len(query), len(recipes)))
    means, scales = train.mean(0), train.std(0)
    scales = np.where(scales > 1e-12, scales, 1.)
    z, q = (train-means)/scales, (query-means)/scales
    geometry, grouped = {}, {}
    for i, r in enumerate(recipes):
        grouped.setdefault((r.family, r.gamma, r.rank, r.whiten), []).append(i)
    ym = y.mean()
    for (family, gamma, rank, whiten), indices in grouped.items():
        if family.startswith('factor'):
            if (rank, whiten) not in geometry:
                mapping = RoleGeometry().fit(train, roles, rank, whiten)
                geometry[rank, whiten] = (mapping.transform(train)[1], mapping.transform(query)[1])
            h, hq = geometry[rank, whiten]
        else:
            h, hq = z, q
        base = family if family in ('linear', 'poly2') else 'rbf'
        k, cross = gram(h, h, base, gamma), gram(hq, h, base, gamma)
        if family.endswith('_product'):
            k *= 1+gram(z, z, 'linear')
            cross *= 1+gram(q, z, 'linear')
        km, kg = k.mean(0), k.mean()
        k = k-km[:, None]-km[None, :]+kg
        cross = cross-cross.mean(1, keepdims=True)-km[None, :]+kg
        values, vectors = eigh(k, check_finite=False, driver='evr')
        left, right = cross@vectors, vectors.T@(y-ym)
        for i in indices:
            result[:, i] = left@(right/(np.maximum(values, 0)+recipes[i].alpha))+ym
    return result


def compare_geometry(data, families, *, folds=5, inner_folds=3, seed=20260924, output=None):
    """Each family is a list of recipes, selected entirely inside outer train."""
    if data.y.shape[1] != 1 or not np.isfinite(data.y).all():
        raise ValueError('This focused geometry comparison takes one fully observed target.')
    y, roles = data.y[:, 0], [c.cue for c in data.registry.items]
    all_recipes = list(dict.fromkeys(r for v in families.values() for r in v))
    family_ids = {name: [all_recipes.index(r) for r in recipes] for name, recipes in families.items()}
    outer = group_splits(data.groups, folds, seed)
    result = {name: np.empty(len(y)) for name in families}
    selections = {name: [] for name in families}
    with threadpool_limits(limits=1):
        for train, test in outer:
            inner = group_splits(data.groups[train], inner_folds, seed+31)
            oof = np.empty((len(train), len(all_recipes)))
            for a, b in inner:
                oof[b] = predictions(data.x[train[a]], y[train[a]], data.x[train[b]], roles, all_recipes)
            losses = ((oof-y[train, None])**2).mean(0)
            selected = {name: ids[int(losses[ids].argmin())] for name, ids in family_ids.items()}
            needed = sorted(set(selected.values()))
            values = predictions(data.x[train], y[train], data.x[test], roles, [all_recipes[j] for j in needed])
            for name, j in selected.items():
                result[name][test] = values[:, needed.index(j)]
                selections[name].append(asdict(all_recipes[j]))
    report = dict(data_sha256=data.fingerprint(), status='Development comparison; family comparisons not independent selection validation.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  families={name: [asdict(r) for r in rs] for name, rs in families.items()}, selected=selections,
                  folds=[dict(train=a.tolist(), test=b.tolist()) for a, b in outer],
                  metrics={name: regression(y, pred) for name, pred in result.items()}, contrasts={})
    for a, b in (('raw_rbf', 'factor_rbf'), ('raw_product', 'factor_product'), ('factor_rbf', 'factor_product'), ('baseline', 'factor_product')):
        if a in result and b in result:
            report['contrasts'][f'{a}_to_{b}'] = paired_contrast(y, result[a], result[b], groups=data.groups, seed=seed)
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'results.json', report)
        np.savez_compressed(output/'predictions.npz', predictions=np.column_stack(list(result.values())),
                            families=np.asarray(list(result)), y=y, ids=np.asarray(data.ids))
    return report, result


def initial_families():
    alphas, gammas = (.001, .01, .1, 1., 10.), (.03, .1, .3, 1.)
    families = {}
    for family in ('raw_rbf', 'factor_rbf', 'raw_product', 'factor_product'):
        families[family] = [GeometryRecipe(family, g, r, w, a)
            for r in ((2, 4) if family.startswith('factor') else (2,))
            for w in ((False, True) if family.startswith('factor') else (False,))
            for g in gammas for a in alphas]
    families['baseline'] = [*families['raw_rbf'],
                            *[GeometryRecipe(f, alpha=a) for f in ('linear', 'poly2') for a in alphas]]
    return families
