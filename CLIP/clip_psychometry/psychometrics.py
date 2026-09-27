"""Assumption-aware reflective audits, not automatic latent validation; chapter 5."""
import numpy as np
from .metrics import correlation, paired_mean


def factor_audit(training, heldout, *, orientation=None):
    from sklearn.decomposition import FactorAnalysis
    train, test = np.asarray(training, float), np.asarray(heldout, float)
    if train.ndim != 2 or train.shape[1] < 3 or test.ndim != 2 or test.shape[1] != train.shape[1]:
        raise ValueError('A reflective hypothesis requires at least three aligned indicators.')
    if len(train) <= train.shape[1] or len(test) < 3 or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('Need finite train/heldout matrices with adequate rows.')
    signs = np.ones(train.shape[1]) if orientation is None else np.asarray(orientation)
    if signs.shape != (train.shape[1],) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('Orientation must be explicit signs ±1 for each item.')
    mean, scale = train.mean(0), train.std(0)
    if np.any(scale <= 1e-12):
        raise ValueError('Constant items do not identify a factor.')
    a, b = (train-mean)/scale*signs, (test-mean)/scale*signs
    model = FactorAnalysis(n_components=1, random_state=0, svd_method='lapack').fit(a)
    loading = model.components_[0]
    if loading.sum() < 0:
        loading = -loading
    uniqueness = model.noise_variance_
    implied = np.outer(loading, loading)+np.diag(uniqueness)
    covariance = np.cov(b, rowvar=False, ddof=0)
    residual = covariance-implied
    off = residual[np.triu_indices(len(loading), 1)]
    numerator = loading.sum()**2
    return dict(n_training=len(train), n_heldout=len(test), loadings=loading.tolist(),
                uniqueness=uniqueness.tolist(), omega_diagonal_model=float(numerator/(numerator+uniqueness.sum())),
                heldout_covariance_residual=residual.tolist(), heldout_off_diagonal_rms=float(np.sqrt(np.mean(off**2))),
                negative_loadings=int((loading < 0).sum()), near_zero_uniqueness=int((uniqueness < .01).sum()),
                assumptions='One common factor; diagonal errors; explicit item orientation. Omega is not construct validity; covariance residuals include domain variance shifts.')


def multitrait_method(scores, cues, methods):
    """Descriptive MTMM contrasts on caller-declared audit rows."""
    x = np.asarray(scores)
    if x.shape[1] != len(cues) or len(cues) != len(methods):
        raise ValueError('One cue and method label per column required.')
    pools = {k: [] for k in ('same_cue_same_method', 'same_cue_cross_method', 'cross_cue_same_method', 'cross_cue_cross_method')}
    for i in range(x.shape[1]):
        for j in range(i):
            key = ('same_cue' if cues[i] == cues[j] else 'cross_cue')+('_same_method' if methods[i] == methods[j] else '_cross_method')
            value = correlation(x[:, i], x[:, j])
            if np.isfinite(value):
                pools[key].append(value)
    return {key: dict(pairs=len(v), median_signed_r=float(np.median(v)) if v else np.nan,
                      median_absolute_r=float(np.median(np.abs(v))) if v else np.nan) for key, v in pools.items()}


def conditional_drift(score, anchor, context, splits, *, alpha=1., groups=None):
    """Cross-fitted context gain over polynomial external-anchor calibration.

Continuous/categorical context must be explicitly encoded by the caller. This
is an anchor-model diagnostic, not an identified or formally tested DIF model.
"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    y, a, c = np.asarray(score, float), np.asarray(anchor, float), np.asarray(context, float)
    if a.ndim != 1 or c.ndim != 2 or len(c) != len(a) or len(y) != len(a):
        raise ValueError('Need score[n], external anchor[n], encoded context[n,q].')
    base = np.column_stack((a, a*a, a*a*a))
    expanded = np.column_stack((base, c, c*a[:, None]))
    p, q, count = np.full(len(y), np.nan), np.full(len(y), np.nan), np.zeros(len(y), int)
    for tr, te in splits:
        if np.intersect1d(tr, te).size or (groups is not None and np.intersect1d(np.asarray(groups)[tr], np.asarray(groups)[te]).size):
            raise ValueError('Drift audit folds must be disjoint, including identity groups.')
        for x, destination in ((base, p), (expanded, q)):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[tr], y[tr])
            destination[te] = model.predict(x[te])
        count[te] += 1
    if not np.all(count == 1):
        raise ValueError('Every audit row needs one held-out prediction.')
    result = paired_mean((y-p)**2-(y-q)**2, groups)
    result['interpretation'] = 'Positive: context predicts measurement beyond the assumed cubic anchor model; not proof of DIF or bias.'
    return result
