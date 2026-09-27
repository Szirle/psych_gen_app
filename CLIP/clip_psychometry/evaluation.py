"""Matched, group-disjoint nested prediction of prespecified banks; chapter 6.

The generic kernel implements the existing centered spectral solve for arbitrary
columns/transforms. adapters.legacy_predict reuses the exact existing recipe and
ensemble machinery when its two-channel schema applies.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from .features import FeatureMap
from .metrics import regression, paired_contrast
from .schema import write_json


@dataclass(frozen=True)
class ComparisonConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    seed: int = 20260924
    alphas: tuple[float, ...] = (.01, .1, 1., 10.)
    # (family, gamma), with gamma unused outside RBF.
    kernels: tuple[tuple[str, float], ...] = (('linear', 1.), ('poly2', 1.), ('rbf', .1), ('rbf', 1.))
    bootstrap: int = 1000
    backend: str = 'generic'
    selector: str = 'single'  # existing_ensemble reuses the production selector.


def group_splits(groups, folds, seed):
    """Randomized ties, size-balanced allocation; no identity crosses folds."""
    groups = np.asarray(groups)
    unique, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if folds < 2 or len(unique) < folds:
        raise ValueError(f'Need at least {folds} groups and at least two folds.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind='stable')]
    loads, allocation = np.zeros(folds, int), np.empty(len(unique), int)
    for j in order:
        slot = int(np.argmin(loads))
        allocation[j] = slot
        loads[slot] += counts[j]
    return [(np.flatnonzero(allocation[inv] != k), np.flatnonzero(allocation[inv] == k)) for k in range(folds)]


def gram(a, b, family, gamma=1.):
    if family == 'split_product':
        if a.shape[1] % 2:
            raise ValueError('Separate metric/function kernels require equal halves.')
        half = a.shape[1]//2
        return gram(a[:,:half], b[:,:half], 'rbf', gamma)*(1+gram(a[:,half:], b[:,half:], 'linear'))
    if family == 'rbf_product':
        return gram(a,b,'rbf',gamma)*(1+gram(a,b,'linear'))
    dot = a@b.T/a.shape[1]
    if family == 'linear':
        return dot
    if family == 'poly2':
        return (1+dot)**2
    if family == 'rbf':
        distance = (a*a).mean(1)[:, None]+(b*b).mean(1)[None, :]-2*dot
        return np.exp(-gamma*np.maximum(distance, 0.))
    raise ValueError(f'Unknown kernel {family}')


def candidates(train, y, query, kernels, alphas):
    """One eigendecomposition per kernel for all alphas and aligned targets."""
    out = np.empty((len(query), len(kernels)*len(alphas), y.shape[1]))
    target_mean = y.mean(0)
    if train.shape[1] == 0:
        out[:] = target_mean
        return out
    for k, (family, gamma) in enumerate(kernels):
        g = gram(train, train, family, gamma)
        m, grand = g.mean(0), g.mean()
        g = g-m[:, None]-m[None, :]+grand
        values, vectors = eigh(g, check_finite=False, driver='evr')
        cross = gram(query, train, family, gamma)
        cross = cross-cross.mean(1, keepdims=True)-m[None, :]+grand
        left, right = cross@vectors, vectors.T@(y-target_mean)
        for a, alpha in enumerate(alphas):
            out[:, k*len(alphas)+a] = left@(right/(np.maximum(values, 0)[:, None]+alpha))+target_mean
    return out


def nested_predict(data, train, test, spec, config):
    prediction = np.full((len(test), len(data.targets)), np.nan)
    choices = {}
    mask_groups = {}
    for t in range(len(data.targets)):
        mask_groups.setdefault(np.isfinite(data.y[train, t]).tobytes(), []).append(t)
    for targets in mask_groups.values():
        labeled = train[np.isfinite(data.y[train, targets[0]])]
        y = data.y[np.ix_(labeled, targets)]
        if config.backend == 'legacy':
            from .adapters import legacy_predict
            pred, selected = legacy_predict(data, labeled, test, targets, spec, config)
            prediction[:, targets] = pred
            choices.update({data.targets[t]: selected[i] for i, t in enumerate(targets)})
            continue
        if config.selector == 'existing_ensemble':
            from .adapters import _repo
            native = _repo('race_perception')
            inner = native.cv_splits(len(labeled), config.inner_folds, config.seed, data.groups[labeled])
        else:
            inner = group_splits(data.groups[labeled], config.inner_folds, config.seed+31)
        oof = np.empty((len(labeled), len(config.kernels)*len(config.alphas), len(targets)))
        for tr, va in inner:
            mapping = FeatureMap(spec, data.registry).fit(data.x[labeled[tr]])
            oof[va] = candidates(mapping.transform(data.x[labeled[tr]]), y[tr],
                                 mapping.transform(data.x[labeled[va]]), config.kernels, config.alphas)
        if config.selector == 'existing_ensemble':
            recipes = [dict(family=f, gamma=g, alpha=a, bank='both', centered=False)
                       for f,g in config.kernels for a in config.alphas]
            strategies, _, _ = native.select_strategies(oof, y, recipes)
            mapping = FeatureMap(spec, data.registry).fit(data.x[labeled])
            values = candidates(mapping.transform(data.x[labeled]), y,
                                mapping.transform(data.x[test]), config.kernels, config.alphas)
            prediction[:, targets] = native.apply_strategies(values, strategies)
            choices.update({data.targets[t]: dict(strategy=s, recipes=[recipes[j] for j in s['indices']])
                            for t,s in zip(targets,strategies)})
            continue
        best = ((oof-y[:, None, :])**2).mean(0).argmin(0)
        mapping = FeatureMap(spec, data.registry).fit(data.x[labeled])
        # Evaluate only selected families/alphas at final fit.
        for j in np.unique(best):
            k, a = divmod(int(j), len(config.alphas))
            outputs = np.flatnonzero(best == j)
            values = candidates(mapping.transform(data.x[labeled]), y[:, outputs],
                                mapping.transform(data.x[test]), (config.kernels[k],), (config.alphas[a],))[:, 0]
            prediction[:, np.asarray(targets)[outputs]] = values
            for index in outputs:
                choices[data.targets[targets[index]]] = dict(kernel=config.kernels[k], alpha=config.alphas[a])
    return prediction, choices


def compare(data, variants, *, config=None, splits=None, output=None, contrasts=None):
    """Compare fixed proposals, not a search selecting the best outer score.

contrasts is {name: (baseline_variant, candidate_variant)}. With None, use
the first variant as baseline for each remaining variant. Persist all OOF values.
"""
    config = config or ComparisonConfig()
    if not variants or config.backend not in ('generic', 'legacy'):
        raise ValueError('Need variants and a known backend.')
    if config.selector not in ('single', 'existing_ensemble'):
        raise ValueError('Unknown recipe selector.')
    if not config.alphas or any(not np.isfinite(a) or a <= 0 for a in config.alphas):
        raise ValueError('Alpha values must be finite and positive.')
    if not config.kernels or any(f not in ('linear', 'poly2', 'rbf', 'rbf_product', 'split_product') or not np.isfinite(g) or g <= 0 for f, g in config.kernels):
        raise ValueError('Invalid kernel grid.')
    if any(spec.separate_metric for spec in variants.values()) != all(spec.separate_metric for spec in variants.values()):
        raise ValueError('Compare separate-metric variants with a matched separate-metric control.')
    if any(spec.separate_metric for spec in variants.values()) != all(f == 'split_product' for f,g in config.kernels):
        if any(spec.separate_metric for spec in variants.values()) or any(f == 'split_product' for f,g in config.kernels):
            raise ValueError('Separate-metric representations require split_product kernels exclusively.')
    splits = list(splits if splits is not None else group_splits(data.groups, config.outer_folds, config.seed))
    coverage, fold_ids = np.zeros(len(data.x), int), np.full(len(data.x), -1)
    for fold, (tr, te) in enumerate(splits):
        tr, te = np.asarray(tr, int), np.asarray(te, int)
        if (not len(tr) or not len(te) or len(np.unique(tr)) != len(tr) or len(np.unique(te)) != len(te)
                or min(tr.min(), te.min()) < 0 or max(tr.max(), te.max()) >= len(data.x)):
            raise ValueError('Invalid fold indices.')
        if np.intersect1d(data.groups[tr], data.groups[te]).size:
            raise ValueError('Train/test identity groups overlap.')
        coverage[te] += 1
        fold_ids[te] = fold
    if not np.all(coverage == 1):
        raise ValueError('Each image must have exactly one outer prediction.')
    predictions, selection, memo = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, spec in variants.items():
            if spec in memo:
                predictions[name], selection[name] = memo[spec]
                continue
            p = np.full_like(data.y, np.nan)
            selected = []
            for train, test in splits:
                p[test], chosen = nested_predict(data, np.asarray(train), np.asarray(test), spec, config)
                selected.append(chosen)
            predictions[name], selection[name] = p, selected
            memo[spec] = p, selected
    baseline = next(iter(variants))
    contrasts = contrasts if contrasts is not None else {name: (baseline, name) for name in variants if name != baseline}
    report = dict(format='clip_psychometry_comparison_v1', data_sha256=data.fingerprint(),
                  config=asdict(config), metadata=data.metadata,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob('*.py'))},
                  variants={k: asdict(v) for k, v in variants.items()}, selection=selection,
                  folds=[dict(train=tr.tolist(), test=te.tolist()) for tr, te in splits],
                  metrics={name: {t: regression(data.y[:, j], p[:, j]) for j, t in enumerate(data.targets)}
                           for name, p in predictions.items()}, contrasts={})
    for name, (a, b) in contrasts.items():
        report['contrasts'][name] = dict(baseline=a, candidate=b,
            targets={t: paired_contrast(data.y[:, j], predictions[a][:, j], predictions[b][:, j],
                                        groups=data.groups, samples=config.bootstrap, seed=config.seed)
                     for j, t in enumerate(data.targets)})
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'comparison.json', report)
        np.savez_compressed(out/'predictions.npz', predictions=np.stack(list(predictions.values()), axis=1),
                            variants=np.asarray(list(predictions)), y=data.y, ids=np.asarray(data.ids),
                            groups=data.groups, folds=fold_ids, targets=np.asarray(data.targets))
    return report, predictions
