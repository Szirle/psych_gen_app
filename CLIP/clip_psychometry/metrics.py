"""Explicit estimands and paired descriptive uncertainty; chapters 4, 6 and 11."""
import numpy as np
from scipy.stats import rankdata


def correlation(a, b, *, ranks=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return float('nan')
    if ranks:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def regression(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    good = np.isfinite(y) & np.isfinite(prediction)
    y, p = y[good], prediction[good]
    if not len(y):
        return dict(n=0)
    mse, variance = np.mean((y-p)**2), np.var(y)
    slope = np.cov(p, y, ddof=0)[0, 1]/np.var(p) if len(p) > 1 and np.var(p) > 1e-12 else np.nan
    return dict(n=len(y), mse=float(mse), rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(y-p))),
                r2=float(1-mse/variance) if variance > 0 else np.nan,
                pearson=correlation(y, p), spearman=correlation(y, p, ranks=True),
                calibration_slope=float(slope), calibration_intercept=float(y.mean()-slope*p.mean()))


def paired_mean(values, groups=None, *, samples=1000, seed=0):
    values = np.asarray(values, float)
    labels = np.arange(len(values)).astype(str) if groups is None else np.asarray(groups)
    if labels.shape != values.shape or samples < 0:
        raise ValueError('Aligned vector of groups and nonnegative bootstrap count required.')
    good = np.isfinite(values)
    values, labels = values[good], labels[good]
    if not len(values):
        return dict(n=0, estimate=np.nan, ci95=None)
    unique, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    interval = None
    if samples and len(unique) > 1:
        rng = np.random.default_rng(seed)
        boot = np.empty(samples)
        for start in range(0, samples, 256):
            draw = rng.integers(len(unique), size=(min(256, samples-start), len(unique)))
            boot[start:start+len(draw)] = totals[draw].sum(1)/counts[draw].sum(1)
        interval = np.quantile(boot, [.025, .975]).tolist()
    return dict(n=len(values), groups=len(unique), estimate=float(values.mean()), ci95=interval,
                uncertainty='Paired cluster bootstrap conditional on fitted predictions; excludes retraining/search uncertainty.')


def paired_contrast(y, baseline, candidate, *, groups=None, samples=1000, seed=0):
    y, a, b = (np.asarray(v, float) for v in (y, baseline, candidate))
    if y.shape != a.shape or a.shape != b.shape:
        raise ValueError('Predictions must align.')
    delta = (y-a)**2-(y-b)**2
    result = paired_mean(delta, groups, samples=samples, seed=seed)
    good = np.isfinite(delta)
    result.update(estimand='baseline MSE minus candidate MSE; positive favors candidate',
                  baseline_mse=float(np.mean((y[good]-a[good])**2)) if good.any() else np.nan,
                  candidate_mse=float(np.mean((y[good]-b[good])**2)) if good.any() else np.nan)
    return result


def complementarity(y, base, a, b, joint, *, groups=None, samples=1000, seed=0):
    y = np.asarray(y, float)
    losses = [(y-np.asarray(p, float))**2 for p in (base, a, b, joint)]
    delta = losses[1]+losses[2]-losses[0]-losses[3]
    result = paired_mean(delta, groups, samples=samples, seed=seed)
    good = np.isfinite(delta)
    result.update(estimand='R(B+A)+R(B+D)-R(B)-R(B+A+D); positive is predictive complementarity',
                  risks=dict(zip(('base', 'a', 'b', 'joint'), [float(v[good].mean()) for v in losses])))
    return result


def noise_accounting(y, prediction, sem):
    y, p, se = (np.asarray(a, float) for a in (y, prediction, sem))
    good = np.isfinite(y) & np.isfinite(p) & np.isfinite(se) & (se >= 0)
    observed, noise = np.mean((y[good]-p[good])**2), np.mean(se[good]**2)
    return dict(n=int(good.sum()), observed_mse=float(observed), mean_sem_squared=float(noise),
                estimated_latent_mean_mse=float(observed-noise),
                assumptions='Unbiased means; independent mean errors and held-out predictor; SEM correctly estimated. No clipping or claimed ceiling.')


def lower_tail(training_scores, scores, criterion, *, quantile=.5):
    threshold = float(np.quantile(training_scores, quantile))
    s, y = np.asarray(scores), np.asarray(criterion)
    below = (s <= threshold) & np.isfinite(s) & np.isfinite(y)
    return dict(threshold=threshold, n=int(below.sum()),
                criterion_mean=float(y[below].mean()) if below.any() else np.nan,
                criterion_sd=float(y[below].std()) if below.any() else np.nan,
                score_criterion_r=correlation(s[below], y[below]),
                note='Descriptive lower-tail audit; threshold came from training, not a semantic probability boundary.')


def contrast_audit(training_pair, pair, target):
    """Common and differential response evidence for two phrases; chapter 4.3."""
    train, x = np.asarray(training_pair, float), np.asarray(pair, float)
    if train.ndim != 2 or train.shape[1] != 2 or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('Two aligned phrase columns are required.')
    scale = train.std(0)
    if np.any(scale <= 1e-12):
        raise ValueError('Cannot interpret a contrast with a constant training item.')
    z = (x-train.mean(0))/scale
    common, contrast = (z[:, 0]+z[:, 1])/2, z[:, 0]-z[:, 1]
    result = dict(pair_r=correlation(z[:, 0], z[:, 1]),
                  item_target_r=[correlation(z[:, j], target) for j in (0, 1)],
                  common_target_r=correlation(common, target), contrast_target_r=correlation(contrast, target),
                  contrast_variance=float(contrast.var()), common_variance=float(common.var()),
                  interpretation='A differential response is not necessarily the intended semantic modifier. An invertible pair re-expression adds no information.')
    return result, common, contrast


def conjunction_overlap(component_score, lexical_score, *, fraction=.1):
    a, b = np.asarray(component_score), np.asarray(lexical_score)
    if a.shape != b.shape or a.ndim != 1 or not 0 < fraction <= 1:
        raise ValueError('Aligned scores and a fraction in (0,1] required.')
    k = max(1, int(np.ceil(len(a)*fraction)))
    left = set(np.argsort(a, kind='stable')[-k:])
    right = set(np.argsort(b, kind='stable')[-k:])
    return dict(k=k, jaccard=len(left & right)/len(left | right), retention=len(left & right)/k,
                shared=sorted(left & right), component_only=sorted(left-right), lexical_only=sorted(right-left),
                warning='Top-set ties use stable row order. Agreement does not establish cue fidelity.')


def support_distance(train, query, *, batch=128):
    train, query = np.asarray(train, float), np.asarray(query, float)
    if train.ndim != 2 or query.shape[1] != train.shape[1] or not train.shape[1]:
        raise ValueError('Use aligned nonempty matrices in the fitted feature geometry.')
    result, nearest = [], []
    for start in range(0, len(query), batch):
        q = query[start:start+batch]
        d = np.maximum((q*q).sum(1)[:, None]+(train*train).sum(1)[None, :]-2*q@train.T, 0.)
        j = d.argmin(1)
        result.extend(np.sqrt(d[np.arange(len(q)), j]/train.shape[1]))
        nearest.extend(j)
    return dict(rms_distance=np.asarray(result), training_neighbor=np.asarray(nearest))


def support_cells(training, values, *, quantile=.5):
    """Counts for all four cells of two measured cues; no interaction inference."""
    if np.shape(training)[1] != 2 or np.shape(values)[1] != 2:
        raise ValueError('Two cue columns required.')
    threshold = np.quantile(training, quantile, axis=0)
    high = np.asarray(values) > threshold
    return dict(threshold=threshold.tolist(), counts={f'{a}{b}': int(((high[:, 0] == a) & (high[:, 1] == b)).sum())
                                                       for a in (0, 1) for b in (0, 1)})


def annotation_metrics(present, score, *, threshold, probability=None, inclusion_probability=None, bins=5):
    """Known-probability sampling weights; cutoff/calibration must be externally fit."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s = np.asarray(present, float), np.asarray(score, float)
    inclusion = np.ones(len(y)) if inclusion_probability is None else np.asarray(inclusion_probability, float)
    if y.shape != s.shape or y.shape != inclusion.shape or np.any((inclusion <= 0) | (inclusion > 1)):
        raise ValueError('Aligned labels, scores, and valid inclusion probabilities required.')
    good = np.isfinite(y) & np.isfinite(s)
    if not good.any() or not np.isin(y[good], [0, 1]).all():
        raise ValueError('Binary annotations required; use NaN for unclear.')
    y, s, w = y[good], s[good], 1/inclusion[good]
    positive = s >= threshold
    tp, pp, actual = np.sum(w*y*positive), np.sum(w*positive), np.sum(w*y)
    result = dict(n=len(y), effective_n=float(w.sum()**2/(w@w)),
                  prevalence=float(np.average(y, weights=w)),
                  precision=float(tp/pp) if pp else np.nan, recall=float(tp/actual) if actual else np.nan,
                  auc=float(roc_auc_score(y, s, sample_weight=w)) if len(np.unique(y)) == 2 else np.nan,
                  average_precision=float(average_precision_score(y, s, sample_weight=w)) if actual else np.nan)
    if probability is not None:
        p = np.asarray(probability, float)[good]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Probabilities must come from a held-out calibration procedure and lie in [0,1].')
        result['brier'] = float(np.average((p-y)**2, weights=w))
        result['calibration'] = []
        for lo, hi in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
            use = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
            if use.any():
                result['calibration'].append(dict(lo=float(lo), hi=float(hi), n=int(use.sum()),
                    predicted=float(np.average(p[use], weights=w[use])), observed=float(np.average(y[use], weights=w[use]))))
    return result
