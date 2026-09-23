"""Masked all-target ratings and fold-local phrase/ordering selection."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge

from ..fgclip2_face_impressions import (
    FaceDatasetConfig, HumanRatingsStore, RegressionResults, discover_face_images,
    array_hash, merge_groups, make_cv_splits, fit_score_model, predict_score_model,
)
from .phrases import hierarchy


@dataclass
class Ratings:
    paths: tuple
    targets: tuple
    y: np.ndarray
    mask: np.ndarray
    se: np.ndarray
    hist: np.ndarray
    groups: np.ndarray
    image_hashes: tuple


def load_ratings(images, ratings, *, targets=None, bins=9, limit=None, seed=0, groups=None):
    store = HumanRatingsStore(ratings)
    targets = tuple(targets or store.variables)
    if len(set(targets)) != len(targets):
        raise ValueError('Targets must be unique.')
    paths = discover_face_images(FaceDatasetConfig(images, limit=None))
    vectors = [store.means(t, paths, min_ratings=2, reject_nonfinite=True) for t in targets]
    y = np.column_stack([v.means for v in vectors])
    se = np.column_stack([v.mean_se for v in vectors])
    mask = np.isfinite(y) & np.isfinite(se)
    rows = np.flatnonzero(mask.any(1))
    if limit:
        rows = np.sort(np.random.default_rng(seed).permutation(rows)[:limit])
    paths, y, se, mask = tuple(paths[i] for i in rows), y[rows], se[rows], mask[rows]
    if len(paths) < 16 or (mask.sum(0) < 8).any():
        raise ValueError('Need at least 16 images and eight labeled images per requested target.')
    if np.any((y[mask] < 0) | (y[mask] > 1)):
        raise ValueError('Expected existing 0–1 rating units. Convert raw OMI CSV with the data setup helper.')
    raw, hist = store._load(), np.zeros((*y.shape, bins), np.float32)
    for i, path in enumerate(paths):
        for j, target in enumerate(targets):
            if not mask[i, j]:
                continue
            values = np.asarray(raw[target][path.name], dtype=float).reshape(-1)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f'Invalid individual ratings: {path.name}/{target}')
            pos = values * (bins-1)
            low = pos.astype(int)
            np.add.at(hist[i, j], low, (1-pos+low)/len(values))
            np.add.at(hist[i, j], np.minimum(low+1, bins-1), (pos-low)/len(values))
    hashes = []
    for path in paths:
        with Image.open(path) as image:
            hashes.append(array_hash(np.asarray(ImageOps.exif_transpose(image).convert('RGB'))))
    labels = np.asarray(hashes)
    if groups:
        mapping = json.loads(Path(groups).read_text())
        labels = merge_groups(labels, [str(mapping[p.name]) for p in paths])
    return Ratings(paths, targets, np.nan_to_num(y).astype(np.float32), mask,
                   np.nan_to_num(se).astype(np.float32), hist, labels, tuple(hashes))


def split_rows(data, rows, folds, seed):
    pairs = make_cv_splits(len(rows), folds, seed, data.groups[rows])
    return [(rows[a], rows[b]) for a, b in pairs]


def tail_split(data, rows, target, fraction):
    t = data.targets.index(target)
    observed = data.mask[rows, t]
    low, high = np.quantile(data.y[rows[observed], t], [fraction, 1-fraction])
    test = observed & ((data.y[rows, t] <= low) | (data.y[rows, t] >= high))
    test = np.isin(data.groups[rows], data.groups[rows[test]])
    if min(test.sum(), (~test).sum()) < 8:
        raise ValueError('Tail split leaves too few groups/images.')
    return rows[~test], rows[test]


def masked_report(data, rows, predictions, name, metadata=None):
    """Reuse the finite RegressionResults contract separately for each target."""
    metrics = {}
    for t, target in enumerate(data.targets):
        valid = data.mask[rows, t]
        if not valid.any():
            metrics[target] = {'n': 0}
            continue
        result = RegressionResults(data.y[rows[valid], t], {name: predictions[valid, t]}, (target,))
        metrics[target] = dict(n=int(valid.sum()), **result.summary()['metrics'][name][target])
    mse = float(np.mean([m['mse'] for m in metrics.values() if m['n']]))
    return dict(method=name, n=len(rows), target_macro_mse=mse, target_macro_rmse=mse**.5,
                metrics=metrics, metadata=metadata or {})


def select_phrases(scores, data, rows, bank, args):
    """No validation/test labels enter MI, OOF difficulty, or residual sampling."""
    rng = np.random.default_rng(args.seed)
    count, width = len(data.targets), len(bank['phrases'])
    information = np.zeros((count, width))
    for t in range(count):
        labeled = rows[data.mask[rows, t]]
        if len(labeled) < 8:
            raise ValueError(f'Insufficient training ratings for {data.targets[t]}')
        for repeat in range(args.mi_repeats):
            selected = rng.choice(labeled, max(8, int(.8*len(labeled))), replace=False)
            mi = mutual_info_regression(scores[selected], data.y[selected, t],
                                        random_state=args.seed+repeat, n_neighbors=3)
            information[t] += mi.reshape(2, width).max(0)  # global OR local evidence
    information /= args.mi_repeats
    # A redundancy penalty discourages selecting only near-identical paraphrases.
    def diverse(values, eligible, number):
        eligible = list(eligible)
        chosen = []
        while eligible and len(chosen) < number:
            value = values[eligible].copy()
            if chosen:
                a, b = scores[rows][:, eligible], scores[rows][:, chosen]
                a, b = a-a.mean(0), b-b.mean(0)
                correlation = np.abs(a.T@b) / np.maximum(np.linalg.norm(a, axis=0)[:, None]*np.linalg.norm(b, axis=0), 1e-8)
                value -= args.redundancy * correlation.max(1)
            j = eligible.pop(int(value.argmax()))
            chosen.append(j)
        return chosen
    lookup = {p: i for i, p in enumerate(bank['phrases'])}
    shared_candidates = [lookup[p] for p in bank['shared']]
    shared = diverse(information.mean(0), shared_candidates, min(args.shared_phrases, len(shared_candidates)))
    entries = [(bank['phrases'][i], -1, i) for i in shared]
    target_indices = []
    for t, target in enumerate(data.targets):
        own = [lookup[p] for p in bank['own'][target]]
        selected = diverse(information[t], [i for i in range(width) if i not in own+shared], args.target_phrases)
        ids = list(range(len(shared)))
        for i in dict.fromkeys(own+selected):
            ids.append(len(entries))
            entries.append((bank['phrases'][i], t, i))
        target_indices.append(ids)
    # Fixed ridge recipe: selection of MI features is deliberately not reused
    # here, so each difficulty estimate's fitted predictor excludes its label.
    oof = np.full((len(rows), count), np.nan)
    position = {int(i): j for j, i in enumerate(rows)}
    for train, valid in split_rows(data, rows, args.inner_folds, args.seed+19):
        for t in range(count):
            labeled = train[data.mask[train, t]]
            if len(labeled) < 3:
                raise ValueError(f'Insufficient inner-training labels for {data.targets[t]}')
            readout = fit_score_model(scores[labeled], data.y[labeled, t], estimator=Ridge(alpha=100., solver='lsqr'))
            oof[[position[int(i)] for i in valid], t] = np.asarray(predict_score_model(readout, scores[valid])).reshape(-1)
    error = (oof-data.y[rows])**2
    skill = []
    for t in range(count):
        valid = data.mask[rows, t]
        skill.append(float(1-error[valid, t].mean()/max(data.y[rows[valid], t].var(), 1e-6)))
    weighted = error / (data.se[rows]**2+.01)
    difficulty = (weighted*data.mask[rows]).sum(1)/np.maximum(data.mask[rows].sum(1), 1)
    difficulty = np.minimum(difficulty, np.quantile(difficulty, .9)) + 1e-6
    probability = (1-args.hard_fraction)/len(rows) + args.hard_fraction*difficulty/difficulty.sum()
    return dict(phrases=[e[0] for e in entries], owners=[e[1] for e in entries],
                bank_indices=[e[2] for e in entries], target_indices=target_indices,
                mi=information[:, [e[2] for e in entries]].tolist(), skill=skill,
                levels=hierarchy(data.targets, skill, args.hierarchy_spec, args.order),
                sampling=probability.tolist(), training_rows=rows.tolist(),
                oof_mse=[float(error[data.mask[rows, t], t].mean()) for t in range(count)])
