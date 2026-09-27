"""Criterion precision with explicit annotation support; methodology selected-set audit."""
import numpy as np


def retrieval_audit(ids, rows, labels):
    """labels maps image ID to binary criterion; no imputation of missing labels.

    Bounds describe this finite retrieved set, not population or annotator uncertainty.
    """
    rows = list(rows)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError('Retrieval must contain distinct rows and be nonempty.')
    if any(i < 0 or i >= len(ids) for i in rows):
        raise ValueError('Retrieval row outside dataset.')
    values = [labels[ids[i]] for i in rows if ids[i] in labels]
    if any(v not in (0, 1, False, True) for v in values):
        raise ValueError('Criterion annotations must be binary.')
    k, a, h = len(rows), len(values), int(sum(values))
    return dict(k=k, annotated=a, hits=h, coverage=a/k,
                precision=h/k if a == k else None,
                precision_bounds=[h/k, (h+k-a)/k],
                unannotated_ids=[ids[i] for i in rows if ids[i] not in labels])


def top_rows(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 < k <= len(scores):
        raise ValueError('Finite score vector and valid k required.')
    return np.argsort(-scores, kind='stable')[:k]


def risk_partition(y, baseline, candidate, strata):
    """Exact additive decomposition of paired finite-sample squared-error gain."""
    y, baseline, candidate = [np.asarray(v, float) for v in (y, baseline, candidate)]
    strata = np.asarray(strata)
    if y.ndim != 1 or any(v.shape != y.shape for v in (baseline, candidate, strata)):
        raise ValueError('Aligned one-dimensional arrays required.')
    valid = np.isfinite(y) & np.isfinite(baseline) & np.isfinite(candidate)
    if not valid.any():
        raise ValueError('No paired observations.')
    gain = (y[valid]-baseline[valid])**2-(y[valid]-candidate[valid])**2
    strata = strata[valid]
    return dict(n=len(gain), mean_gain=float(gain.mean()), strata={str(s):dict(
        n=int((strata==s).sum()), conditional_gain=float(gain[strata==s].mean()),
        contribution=float(gain[strata==s].sum()/len(gain))) for s in np.unique(strata)})
