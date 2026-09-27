"""Shared one-step AGOP geometry for B2; no local partitions or recursive fitting.

Geometry uses standardized coordinates and is never standardized away afterward.
Historical xRFM probe findings remain in research/clip_psychometry/xrfm_geometry.
"""
import numpy as np
from scipy.linalg import eigh
from .adapters import _repo



def recipes():
    return [r for r in _repo('race_perception').candidate_grid()
            if r['bank']=='both' and not r.get('centered')]


def readout_gradients(fitted):
    """Gradients in standardized input coordinates; ensemble cross terms retained."""
    native = _repo('race_perception')
    result = []
    for t, strategy in enumerate(fitted['strategies']):
        total = None
        for index, weight in zip(strategy['indices'], strategy['weights']):
            c = next(c for c in fitted['components'].values() if index in c['coefficients'])
            z = c['state']['z']
            r = c['state']['recipe']
            a = c['coefficients'][index][:, t]
            a = a-a.mean()
            p = z.shape[1]
            if r['family']=='linear':
                g = np.broadcast_to(a@z/p, z.shape)
            elif r['family']=='poly2':
                g = (2/p)*(((1+z@z.T/p)*a[None, :])@z)
            else:
                h = native.kernel(z, z, r)*a[None, :]
                g = -(2*r['gamma']/p)*(z*h.sum(1)[:, None]-h@z)
            total = weight*g if total is None else total+weight*g
        result.append(total)
    return result


def pilot(x, y, groups, seed):
    native = _repo('race_perception')
    rs = recipes()
    inner, _ = native.inner_predictions(x, y, rs, 2, seed, groups)
    strategies, _, _ = native.select_strategies(inner, y, rs)
    return readout_gradients(native.fit_bundle_readout(x, y, rs, strategies))


def standardize(x, query):
    mean, sd = x.mean(0), x.std(0)
    sd = np.where(sd>1e-12, sd, 1.)
    return (x-mean)/sd, (query-mean)/sd


def metric_dots(z, q, gradient, kind, beta):
    p = z.shape[1]
    energy = float(np.square(gradient).sum())
    if energy <= 1e-30:
        beta = 0.
    if not beta:
        return z@z.T/p, q@z.T/p, (z*z).mean(1), (q*q).mean(1)
    if kind=='diag':
        weight = (1-beta)+beta*p*(gradient*gradient).sum(0)/energy
        return ((z*weight)@z.T/p, (q*weight)@z.T/p,
                (z*z*weight).mean(1), (q*q*weight).mean(1))
    if kind!='full':
        raise ValueError('Expected diagonal or full metric.')
    a, b = z@gradient.T, q@gradient.T
    return ((1-beta)*(z@z.T/p)+beta*(a@a.T/energy),
            (1-beta)*(q@z.T/p)+beta*(b@a.T/energy),
            (1-beta)*(z*z).mean(1)+beta*(a*a).sum(1)/energy,
            (1-beta)*(q*q).mean(1)+beta*(b*b).sum(1)/energy)


def metric_predictions(z, y, q, gradient, kind='full', beta=.5):
    dots, cross, diag, qdiag = metric_dots(z, q, gradient, kind, beta)
    rs = recipes()
    out = np.empty((len(q), len(rs), y.shape[1]))
    means = y.mean(0)
    native = _repo('race_perception')
    groups = {}
    for i, r in enumerate(rs):
        groups.setdefault(native.base_key(r), []).append(i)
    for indices in groups.values():
        r = rs[indices[0]]
        if r['family']=='linear':
            k, c = dots, cross
        elif r['family']=='poly2':
            k, c = (1+dots)**2, (1+cross)**2
        else:
            k = np.exp(-r['gamma']*np.maximum(diag[:, None]+diag[None, :]-2*dots, 0))
            c = np.exp(-r['gamma']*np.maximum(qdiag[:, None]+diag[None, :]-2*cross, 0))
        km, grand = k.mean(0), k.mean()
        k = k-km[:, None]-km[None, :]+grand
        c = c-c.mean(1, keepdims=True)-km[None, :]+grand
        values, vectors = eigh(k, check_finite=False, driver='evr')
        left, right = c@vectors, vectors.T@(y-means)
        for i in indices:
            out[:, i] = left@(right/(np.maximum(values, 0)[:, None]+rs[i]['alpha']))+means
    return out

