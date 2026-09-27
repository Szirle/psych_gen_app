"""Deployable B2: expanded-bank, per-target continuous one-step AGOP geometry.

Selection is identical to the factorial benchmark's six continuous candidates.
All scaling, pilot fitting and metric learning happen within each training split.
"""
import numpy as np
from scipy.linalg import eigh
from .adapters import _repo
from .learned_geometry import pilot, standardize, metric_predictions, recipes, metric_dots

OPTIONS = ('unweighted', 'diag_50', 'diag_100', 'full_25', 'full_50', 'full_100')
METRICS = (('diag', 0.), ('diag', .5), ('diag', 1.), ('full', .25), ('full', .5), ('full', 1.))


def continuous_candidates(x, y, q, groups, seed):
    native = _repo('race_perception')
    x, y, q = np.asarray(x, float), np.asarray(y, float), np.asarray(q, float)
    z, query = standardize(x, q)
    gradients = pilot(x, y, groups, seed)
    out = np.empty((len(q), len(OPTIONS), len(recipes()), y.shape[1]))
    out[:, 0] = native.predict_candidates(x, y, q, recipes())
    for t, g in enumerate(gradients):
        for j, (kind, beta) in enumerate(METRICS[1:], 1):
            out[:, j, :, t] = metric_predictions(z, y[:, t:t+1], query, g, kind, beta)[:, :, 0]
    return out


def kernel_from_dots(dot, a_diag, b_diag, recipe):
    if recipe['family'] == 'linear':
        return dot
    if recipe['family'] == 'poly2':
        return (1+dot)**2
    return np.exp(-recipe['gamma']*np.maximum(a_diag[:, None]+b_diag[None, :]-2*dot, 0))


def fit_metric_readout(z, y, g, option, strategy):
    """Refit a selected target metric and kernel strategy on all supplied rows."""
    rs = recipes()
    kind, beta = METRICS[option]
    energy = float(np.square(g).sum())
    if energy <= 1e-30:
        beta = 0.
    # Store a normalized gradient factor rather than a dense p x p matrix.
    # Diagonal metrics need only p numbers; supports are shared across targets.
    metric = dict(kind=kind, beta=beta)
    if kind == 'full' and beta:
        metric['factor'] = g / np.sqrt(energy)
    else:
        metric['kind'] = 'diag'
        metric['weights'] = ((1-beta)+beta*z.shape[1]*(g*g).sum(0)/energy
                             if beta else np.ones(z.shape[1]))
    dots, _, diag, _ = metric_dots(z, z[:0], g, kind, beta)
    components = []
    target_mean = float(y.mean())
    for index, weight in zip(strategy['indices'], strategy['weights']):
        r = rs[index]
        gram = kernel_from_dots(dots, diag, diag, r)
        km, grand = gram.mean(0), float(gram.mean())
        centered = gram-km[:, None]-km[None, :]+grand
        values, vectors = eigh(centered, check_finite=False, driver='evr')
        coef = vectors@((vectors.T@(y-target_mean))/(np.maximum(values, 0)+r['alpha']))
        components.append(dict(recipe=r, weight=weight, kernel_mean=km,
                               kernel_grand_mean=grand, coef=coef))
    return dict(metric=metric, components=components, target_mean=target_mean)


def fit_weighted_kernel(x, y, phrases, groups, *, inner_folds=4, seed=20260921):
    native = _repo('race_perception')
    x, y = native.validate_xy(x, y)
    groups = np.asarray(groups)
    if x.shape[1] != 2*len(phrases) or len(set(phrases)) != len(phrases):
        raise ValueError('Expected two aligned channels of unique phrases.')
    rs = recipes()
    inner = np.empty((len(x), len(OPTIONS), len(rs), y.shape[1]))
    for k, (tr, va) in enumerate(native.cv_splits(len(x), inner_folds, seed, groups)):
        print(f'AGOP weighting inner {k+1}/{inner_folds}: {len(tr)} rows, {y.shape[1]} targets', flush=True)
        inner[va] = continuous_candidates(x[tr], y[tr], x[va], groups[tr], seed)
    strategies, losses = [], []
    for j in range(len(OPTIONS)):
        s, _, _ = native.select_strategies(inner[:, j], y, rs)
        strategies.append(s)
        losses.append(((native.apply_strategies(inner[:, j], s)-y)**2).mean(0))
    losses = np.asarray(losses)
    choices = losses.argmin(0)
    del inner
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x-mean)/scale
    gradients = pilot(x, y, groups, seed)
    parts, details = [], []
    for t, j in enumerate(choices):
        g = gradients[t]
        strategy = strategies[j][t]
        parts.append(fit_metric_readout(z, y[:, t], g, j, strategy))
        importance = np.square(g).sum(0)
        importance = importance/max(float(importance.sum()), 1e-30)
        details.append(dict(chosen=OPTIONS[j], inner_mse=float(losses[j, t]),
            inner_curve=losses[:, t].tolist(), strategy=strategy,
            phrase_sensitivity=(importance[:len(phrases)]+importance[len(phrases):]).tolist()))
    return dict(format='clip_psychometry_weighted_kernel_v1', width=x.shape[1], outputs=y.shape[1],
                phrases=list(phrases), mean=mean, scale=scale, support=z, readouts=parts,
                selection=details, options=OPTIONS, inner_folds=inner_folds, seed=seed)


def predict_weighted_kernel(model, x):
    x = _repo('race_perception').validate_xy(x)
    if x.shape[1] != model['width']:
        raise ValueError('Feature width differs from the saved bank.')
    z, q = model['support'], (x-model['mean'])/model['scale']
    p = z.shape[1]
    result = np.empty((len(q), model['outputs']))
    for t, part in enumerate(model['readouts']):
        m = part['metric']
        if m['kind'] == 'diag':
            w = m['weights']
            cross = (q*w)@z.T/p
            diag, qdiag = (z*z*w).mean(1), (q*q*w).mean(1)
        else:
            a, b = z@m['factor'].T, q@m['factor'].T
            beta = m['beta']
            cross = (1-beta)*(q@z.T/p)+beta*(b@a.T)
            diag = (1-beta)*(z*z).mean(1)+beta*(a*a).sum(1)
            qdiag = (1-beta)*(q*q).mean(1)+beta*(b*b).sum(1)
        pred = np.zeros(len(q))
        for c in part['components']:
            k = kernel_from_dots(cross, qdiag, diag, c['recipe'])
            k = k-k.mean(1, keepdims=True)-c['kernel_mean'][None, :]+c['kernel_grand_mean']
            pred += c['weight']*(k@c['coef']+part['target_mean'])
        result[:, t] = pred
    return result
