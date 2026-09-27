"""Descriptive item evidence and explicit, noncausal routing; chapters 4 and 10."""
import numpy as np
from .metrics import correlation


def detector_summary(data):
    rows = []
    for j, item in enumerate(data.registry.items):
        values = data.x[:, j]
        rows.append(dict(column=j, id=item.id, text=item.text, cue=item.cue, channel=item.channel,
                         sd=float(values.std()), quantiles=np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist(),
                         associations={name: dict(pearson=correlation(values, data.y[:, t]),
                                                  spearman=correlation(values, data.y[:, t], ranks=True),
                                                  n=int(np.isfinite(data.y[:, t]).sum()),
                                                  direct_criterion=item.criterion == name)
                                       for t, name in enumerate(data.targets)}))
    return dict(items=rows, interpretation='Descriptive target alignment; associations do not establish detector validity or conditional utility.')


def evidence_routes(evidence, thresholds):
    """Route observed diagnostics to next actions; callers specify thresholds.

Recognized fields: annotated_precision, lower_tail_r, factor_residual_rms,
conditional_context_gain, group_gain, single_gain, complementarity,
normalized_view_drift, min_support_cell. No probabilities or causal verdicts.
"""
    routes = []
    def add(code, observation, action, alternatives):
        routes.append(dict(code=code, observation=observation, next_action=action, alternatives=alternatives))
    def above(key):
        return key in evidence and key in thresholds and evidence[key] > thresholds[key]
    def below(key):
        return key in evidence and key in thresholds and evidence[key] < thresholds[key]
    if below('annotated_precision'):
        add('M1', 'Annotation precision below declared threshold.', 'Inspect disagreement and rewrite the visual cue.', 'Annotation ambiguity; domain mismatch; cutoff calibration.')
    if above('factor_residual_rms'):
        add('M4', 'Reflective model has large held-out covariance residuals.', 'Inspect method factors and construct heterogeneity.', 'Population variance shift; small sample; misspecified factor model.')
    if above('conditional_context_gain'):
        add('M5', 'Context improves conditional measurement prediction.', 'Collect independent anchors and check common support.', 'Anchor error or misspecified response curve, not necessarily DIF.')
    if above('group_gain') and below('single_gain'):
        add('P1', 'Role deletion matters more than individual deletion.', 'Audit substitute closure and preserve group comparisons.', 'Finite-sample tuning variation or metric changes.')
    if above('complementarity'):
        add('P2', 'Joint predictive gain exceeds separate gains.', 'Inspect joint support and validate both detectors.', 'Correlation, readout approximation, or search noise.')
    if above('normalized_view_drift'):
        add('R1', 'View drift exceeds declared tolerance.', 'Determine whether the transformation changes the human target.', 'Valid sensitivity versus measurement instability.')
    if below('min_support_cell'):
        add('P2/R2', 'At least one joint cue cell lacks support.', 'Prioritize new images/annotations in that cell.', 'No reliable conclusion about absent interactions.')
    return dict(routes=routes, status='Diagnostic hypotheses only; absence of a flag does not validate a detector.',
                supplied_thresholds=thresholds)
