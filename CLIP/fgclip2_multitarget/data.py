"""Masked all-target ratings and fold-local phrase/ordering selection."""
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge

from ..fgclip2_face_impressions import (
    FaceDatasetConfig, HumanRatingsStore, RegressionResults, discover_face_images,
    array_hash, merge_groups, make_cv_splits, fit_score_model, predict_score_model,
)
from .phrases import hierarchy


def batched_mutual_information(x, y, *, seed, device, memory_gib=4.):
    """Continuous KSG-1 (k=3), using batched exhaustive Chebyshev neighbors.

    Same scaling, jitter/RNG order, strict radius counts and nonnegative clipping
    as sklearn mutual_info_regression. Float64 preserves its 1e-10 tie jitter;
    float32 distances would silently change the estimator on repeated scores.
    CPU tensors are supported for parity checks; production uses CUDA here.
    """
    import torch
    from scipy.special import digamma
    from sklearn.preprocessing import scale

    n, features = x.shape
    if n < 4 or memory_gib <= 0:
        raise ValueError('MI requires at least four rows and a positive memory budget')
    rng = np.random.RandomState(seed)
    # Preserve sklearn's dtype of y (including float32 jitter rounding).
    x = scale(np.asarray(x, dtype=np.float64).copy(), with_mean=False, copy=False)
    x += 1e-10*np.maximum(1, np.abs(x).mean(0))*rng.standard_normal(x.shape)
    y = scale(np.asarray(y), with_mean=False)
    y += 1e-10*max(1, np.abs(y).mean())*rng.standard_normal(n)
    device = torch.device(device)
    budget = memory_gib*1024**3
    if device.type == 'cuda':
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info()
        budget = min(budget, free*.4)
    # Bound two distance tensors, masks, top-k workspace and temporary buffers.
    chunk = min(features, max(1, int(budget//(40*n*n))))
    result = []
    with torch.no_grad():
        yt = torch.as_tensor(np.asarray(y, dtype=np.float64), device=device)
        dy = (yt[:, None]-yt[None, :]).abs()
        psi = torch.as_tensor(digamma(np.arange(1, n+1)), device=device)
        for start in range(0, features, chunk):
            values = torch.as_tensor(x[:, start:start+chunk].T.copy(), device=device)
            dx = (values[:, :, None]-values[:, None, :]).abs_()
            joint = torch.maximum(dx, dy)
            joint.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
            radius = joint.topk(3, dim=-1, largest=False).values[:, :, -1]
            del joint
            radius = torch.nextafter(radius, torch.zeros_like(radius))[:, :, None]
            # Includes self: count == nx+1 / ny+1 in sklearn's formula.
            nx = (dx <= radius).sum(-1).clamp_min(1)
            ny = (dy <= radius).sum(-1).clamp_min(1)
            result.append((digamma(n)+digamma(3)-psi[nx-1].mean(-1)
                           -psi[ny-1].mean(-1)).clamp_min(0).cpu().numpy())
            del dx, radius, nx, ny
    return np.concatenate(result)


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
    backend = getattr(args, 'selection_backend', 'auto')
    device = str(args.device)
    if backend == 'auto':
        backend = 'cuda' if device.startswith('cuda') else 'sklearn'
    if backend == 'cuda' and not device.startswith('cuda'):
        raise ValueError('--selection-backend cuda requires a CUDA training device')
    started = perf_counter()
    print(f'  MI: {backend}, {count} targets × {args.mi_repeats} repeats × {2*width} features', flush=True)
    information = np.zeros((count, width))
    for t in range(count):
        labeled = rows[data.mask[rows, t]]
        if len(labeled) < 8:
            raise ValueError(f'Insufficient training ratings for {data.targets[t]}')
        for repeat in range(args.mi_repeats):
            selected = rng.choice(labeled, max(8, int(.8*len(labeled))), replace=False)
            if backend == 'cuda':
                mi = batched_mutual_information(scores[selected], data.y[selected, t], seed=args.seed+repeat,
                    device=device, memory_gib=getattr(args, 'selection_memory_gib', 4.))
            else:
                mi = mutual_info_regression(scores[selected], data.y[selected, t],
                    random_state=args.seed+repeat, n_neighbors=3, n_jobs=getattr(args, 'cpu_threads', 1))
            information[t] += mi.reshape(2, width).max(0)  # global OR local evidence
        # print(f'  MI {t+1}/{count}: {data.targets[t]} ({perf_counter()-started:.1f}s elapsed)', flush=True)
    information /= args.mi_repeats
    mi_seconds = perf_counter()-started
    # All greedy selections use the same training-only GLOBAL correlations.
    # Compute once, instead of centering/multiplying again at every greedy step.
    centered = scores[rows, :width].astype(np.float64)
    centered -= centered.mean(0)
    norms = np.linalg.norm(centered, axis=0)
    correlation = np.abs(centered.T@centered)/np.maximum(norms[:, None]*norms[None, :], 1e-8)
    # A redundancy penalty discourages selecting only near-identical paraphrases.
    def diverse(values, eligible, number):
        eligible = list(eligible)
        chosen = []
        while eligible and len(chosen) < number:
            value = values[eligible].copy()
            if chosen:
                value -= args.redundancy * correlation[np.ix_(eligible, chosen)].max(1)
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
    selection_seconds = perf_counter()-started-mi_seconds
    solves = 0
    for fold, (train, valid) in enumerate(split_rows(data, rows, args.inner_folds, args.seed+19)):
        # Targets sharing observed rows share scaling and a single factorization.
        # Cholesky solves the same alpha=100 ridge objective as former LSQR,
        # without separate iterative solves or their stopping tolerance.
        groups = {}
        for t in range(count):
            groups.setdefault(data.mask[train, t].tobytes(), []).append(t)
        for targets in groups.values():
            labeled = train[data.mask[train, targets[0]]]
            if len(labeled) < 3:
                raise ValueError(f'Insufficient inner-training labels for {data.targets[targets[0]]}')
            readout = fit_score_model(scores[labeled], data.y[np.ix_(labeled, targets)],
                                     estimator=Ridge(alpha=100., solver='cholesky'))
            values = np.asarray(predict_score_model(readout, scores[valid])).reshape(len(valid), len(targets))
            oof[np.ix_([position[int(i)] for i in valid], targets)] = values
            solves += 1
        print(f'  Difficulty fold {fold+1}/{args.inner_folds}: {len(groups)} shared ridge solve(s)', flush=True)
    error = (oof-data.y[rows])**2
    skill = []
    for t in range(count):
        valid = data.mask[rows, t]
        skill.append(float(1-error[valid, t].mean()/max(data.y[rows[valid], t].var(), 1e-6)))
    weighted = error / (data.se[rows]**2+.01)
    difficulty = (weighted*data.mask[rows]).sum(1)/np.maximum(data.mask[rows].sum(1), 1)
    difficulty = np.minimum(difficulty, np.quantile(difficulty, .9)) + 1e-6
    probability = (1-args.hard_fraction)/len(rows) + args.hard_fraction*difficulty/difficulty.sum()
    timing = dict(mi=mi_seconds, selection=selection_seconds,
                  difficulty=perf_counter()-started-mi_seconds-selection_seconds, total=perf_counter()-started)
    print(f'  Selection complete in {timing["total"]:.1f}s '
          f'(MI {mi_seconds:.1f}s, diversity {selection_seconds:.1f}s, ridge {timing["difficulty"]:.1f}s)', flush=True)
    return dict(phrases=[e[0] for e in entries], owners=[e[1] for e in entries],
                bank_indices=[e[2] for e in entries], target_indices=target_indices,
                mi=information[:, [e[2] for e in entries]].tolist(), skill=skill,
                levels=hierarchy(data.targets, skill, args.hierarchy_spec, args.order),
                sampling=probability.tolist(), training_rows=rows.tolist(),
                oof_mse=[float(error[data.mask[rows, t], t].mean()) for t in range(count)],
                computation=dict(mi_backend=backend, ridge_solver='shared-cholesky', ridge_solves=solves,
                                 seconds=timing))
