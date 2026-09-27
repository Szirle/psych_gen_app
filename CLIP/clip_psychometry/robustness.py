"""Paired view audits with explicit label-preservation assumptions; chapter 8."""
import numpy as np
from .metrics import correlation, paired_mean


def augmentation_audit(training_clean, clean, views, *, top_fraction=.1):
    """Views[n, view, column], clean[n,column]; scale from training only."""
    train, clean, views = (np.asarray(x, float) for x in (training_clean, clean, views))
    if clean.ndim != 2 or views.ndim != 3 or views.shape[0] != len(clean) or views.shape[2] != clean.shape[1]:
        raise ValueError('Expected clean[n,p], views[n,v,p].')
    if train.ndim != 2 or train.shape[1] != clean.shape[1] or views.shape[1] < 2:
        raise ValueError('Aligned training reference and at least two views required.')
    if not all(np.isfinite(x).all() for x in (train, clean, views)) or not 0 < top_fraction <= 1:
        raise ValueError('Finite scores and valid top fraction required.')
    variance = train.var(0)
    delta = views-clean[:, None, :]
    # Random-view one-way variance-components proxy; collapse is undefined.
    v = views.shape[1]
    within = views.var(1, ddof=1).mean(0)
    between = np.maximum(views.mean(1).var(0, ddof=1)-within/v, 0.)
    k = max(1, int(np.ceil(len(clean)*top_fraction)))
    output = []
    for j in range(clean.shape[1]):
        stable = variance[j] > 1e-12
        denominator = between[j]+within[j]
        top = set(np.argsort(clean[:, j], kind='stable')[-k:])
        output.append(dict(column=j, training_variance=float(variance[j]), collapsed=not stable,
            signed_drift=float(delta[:, :, j].mean()), mean_squared_drift=float(np.mean(delta[:, :, j]**2)),
            standardized_bias=float(delta[:, :, j].mean()/np.sqrt(variance[j])) if stable else np.nan,
            normalized_squared_drift=float(np.mean(delta[:, :, j]**2)/variance[j]) if stable else np.nan,
            normalized_within_variance=float(within[j]/variance[j]) if stable else np.nan,
            repeatability_proxy=float(between[j]/denominator) if stable and denominator > 1e-12 else np.nan,
            spearman_by_view=[correlation(clean[:, j], views[:, k_, j], ranks=True) for k_ in range(v)],
            top_retention_by_view=[len(top & set(np.argsort(views[:, k_, j], kind='stable')[-k:]))/k for k_ in range(v)]))
    return dict(columns=output, note='Repeatability does not establish validity. Top-set ties depend on row order; normalized statistics undefined for collapsed training scores.')


def prediction_audit(clean, views, *, y=None, label_preserving=False, groups=None):
    clean, views = np.asarray(clean, float), np.asarray(views, float)
    if clean.ndim != 1 or views.ndim != 2 or len(clean) != len(views):
        raise ValueError('Use one target at a time: clean[n], views[n,v].')
    result = dict(instability=paired_mean(((views-clean[:, None])**2).mean(1), groups),
                  label_preserving=bool(label_preserving))
    if y is not None:
        if not label_preserving:
            raise ValueError('View error against original ratings requires an explicit label-preservation assumption.')
        y = np.asarray(y, float)
        result['view_minus_clean_mse'] = paired_mean(((views-y[:, None])**2).mean(1)-(clean-y)**2, groups)
    return result
