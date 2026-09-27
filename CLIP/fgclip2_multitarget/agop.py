"""Bridge the psychometry selector to all-target CV and checkpoint inference."""
import copy
import numpy as np
from ..clip_psychometry.phrase_candidates import CANDIDATES
from ..clip_psychometry.weighted_kernel import fit_weighted_kernel, predict_weighted_kernel
from ..fgclip2_face_impressions import write_json
from .assets import atomic_save


def expanded_bank(base):
    bank = copy.deepcopy(base)
    bank['phrases'] = list(dict.fromkeys(base['phrases']+[p for v in CANDIDATES.values() for p in v]))
    bank['profile'] = 'agop-expanded-v1'
    return bank


def bank_view(assets, bank):
    """Share visual tensors; expose precisely the requested text/score columns."""
    view = copy.copy(assets)
    index = {p: i for i, p in enumerate(assets.bank['phrases'])}
    ids = np.array([index[p] for p in bank['phrases']])
    view.bank = bank
    view.text = tuple(t[ids] for t in assets.text)
    view.scores = assets.scores[:, np.r_[ids, ids+len(index)]]
    return view


def fit_agop(assets, data, train, test, directory, args):
    groups = {}
    for t in range(len(data.targets)):
        groups.setdefault(data.mask[train, t].tobytes(), []).append(t)
    prediction = np.empty((len(test), len(data.targets))) if test is not None else None
    parts, details = [], {}
    for targets in groups.values():
        rows = train[data.mask[train, targets[0]]]
        model = fit_weighted_kernel(assets.scores[rows], data.y[np.ix_(rows, targets)],
            assets.bank['phrases'], data.groups[rows], inner_folds=args.inner_folds,
            seed=args.seed)
        if test is not None:
            prediction[:, targets] = predict_weighted_kernel(model, assets.scores[test])
        for local, target in enumerate(targets):
            details[data.targets[target]] = model['selection'][local]
        parts.append(dict(targets=targets, model=model))
    write_json(directory/'selection.json', dict(targets=details, phrases=assets.bank['phrases'],
        note='B2 continuous AGOP metric; full expanded bank for every target. Sensitivity is not causal importance.'))
    if not args.smoke:
        atomic_save(dict(format='fgclip2_multitarget_agop_weighted_v1', targets=data.targets, blocks=parts,
            phrases=assets.bank['phrases'], feature_order='global-short then local-box-max',
            config=args.serializable_config, model=assets.engine.model_id,
            revision=assets.engine.revision), directory/'model.pt')
    return prediction, {}


def predict_scores(packet, scores):
    result = np.empty((len(scores), len(packet['targets'])))
    for block in packet['blocks']:
        if packet['format'] == 'fgclip2_multitarget_agop_v1':
            from ..clip_psychometry.target_kernel import predict_target_kernel
            result[:, block['targets']] = predict_target_kernel(block['model'], scores)
        else:
            result[:, block['targets']] = predict_weighted_kernel(block['model'], scores)
    return result
