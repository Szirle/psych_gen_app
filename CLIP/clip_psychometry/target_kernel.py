"""Deployable per-target AGOP pruning with fully nested retention selection.

Inputs use global-short then local-box-max columns. The readout, kernel grid,
and ensemble selector are shared with the original frozen-kernel pipeline.
"""
from dataclasses import asdict
import numpy as np
from .adapters import _repo
from .pruning import AGOPConfig, rank_paths, select_units

DEFAULT_KEEPS = (1., .95, .9, .75, .5, .25, .1)


def fit_target_kernel(x, y, phrases, groups, *, inner_folds=4, seed=20260921,
                      keeps=DEFAULT_KEEPS, progress=print):
    """Fit complete-label target blocks; caller separates differing label masks.

    Every inner validation row is excluded from scaling, pilot readout selection,
    AGOP estimation and mask construction. 100% retention is always a candidate.
    No outer-test data enter this function. Output includes deployable readouts.
    """
    native = _repo('race_perception')
    x, y = native.validate_xy(x, y)
    phrases = tuple(phrases)
    if x.shape[1] != 2*len(phrases) or len(set(phrases)) != len(phrases):
        raise ValueError('Expected unique phrases with two aligned feature channels.')
    if len(groups) != len(x) or any(not np.isfinite(k) or not 0 < k <= 1 for k in keeps):
        raise ValueError('Invalid groups or retention fractions.')
    keeps = tuple(sorted({1., *keeps}, reverse=True))
    units = tuple(np.array([j, j+len(phrases)]) for j in range(len(phrases)))
    recipes = [r for r in native.candidate_grid() if r['bank']=='both' and not r.get('centered')]
    config = AGOPConfig(kernel='frozen', rounds=1, seed=seed)
    oof = np.empty((len(x), len(keeps), len(recipes), y.shape[1]))
    splits = native.cv_splits(len(x), inner_folds, seed, groups)
    for fold, (tr, va) in enumerate(splits):
        progress(f'AGOP inner {fold+1}/{inner_folds}: {len(tr)} training images, {y.shape[1]} targets', flush=True)
        rank = rank_paths(x[tr], y[tr], config, groups=np.asarray(groups)[tr])[0]
        oof[va, 0] = native.predict_candidates(x[tr], y[tr], x[va], recipes)
        for t in range(y.shape[1]):
            memo = {}
            for j, keep in enumerate(keeps[1:], 1):
                cols, _ = select_units(rank[:, t], units, keep)
                key = tuple(cols)
                if key not in memo:
                    memo[key] = native.predict_candidates(x[np.ix_(tr, cols)], y[tr, t:t+1],
                                                         x[np.ix_(va, cols)], recipes)[:, :, 0]
                oof[va, j, :, t] = memo[key]
    strategies, losses = [], []
    for j in range(len(keeps)):
        s, _, _ = native.select_strategies(oof[:, j], y, recipes)
        strategies.append(s)
        losses.append(((native.apply_strategies(oof[:, j], s)-y)**2).mean(0))
    losses = np.asarray(losses)
    best = losses.argmin(0)  # ties prefer more phrases
    rank = rank_paths(x, y, config, groups=groups)[0]
    parts, details = [], []
    for t, j in enumerate(best):
        cols, importance = select_units(rank[:, t], units, keeps[j])
        strategy = strategies[j][t]
        readout = native.fit_bundle_readout(x[:, cols], y[:, t:t+1], recipes, [strategy])
        parts.append(dict(targets=[t], columns=cols, readout=readout))
        details.append(dict(keep=keeps[j], phrases=len(cols)//2, columns=cols.tolist(),
                            importance=importance.tolist(), inner_mse=losses[j, t],
                            inner_curve=losses[:, t].tolist(), strategy=strategy))
    return dict(format='clip_psychometry_target_kernel_v1', width=x.shape[1], outputs=y.shape[1],
                readouts=parts, phrases=phrases, selection=details, keep_fractions=keeps,
                agop=asdict(config), inner_folds=inner_folds, seed=seed)


def predict_target_kernel(model, x):
    native = _repo('race_perception')
    x = native.validate_xy(x)
    if x.shape[1] != model['width']:
        raise ValueError('Feature width does not match saved phrase bank.')
    result = np.empty((len(x), model['outputs']))
    for part in model['readouts']:
        result[:, part['targets']] = native.predict_readout(part['readout'], x[:, part['columns']])
    return result
