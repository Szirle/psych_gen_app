"""Compare overlapping W coordinates and the application's ridge directions."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils

OUT = ROOT / 'tests' / 'old_vs_new_coords_analysis'
OUT.mkdir(exist_ok=True)
old, ratings = utils.load_psychGAN_data(str(ROOT / 'data'))
new = torch.load(ROOT / 'tests/image_to_coords_3k.pt', map_location='cpu', weights_only=True)
old_total, new_total = len(old), len(new)
images = sorted(set(old) & set(new), key=lambda name: (int(Path(name).stem), name))
old = {name: old[name].detach().float().cpu().reshape(-1) for name in images}
new = {name: new[name].detach().float().cpu().reshape(-1) for name in images}
a = torch.stack(list(old.values())).numpy().astype(np.float64)
b = torch.stack(list(new.values())).numpy().astype(np.float64)
assert a.shape == b.shape == (len(images), 512)
assert np.isfinite(a).all() and np.isfinite(b).all()

def cosine_rows(x, y):
    return np.sum(x*y, axis=-1) / (np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1))

ac, bc = a-a.mean(0), b-b.mean(0)
delta = np.linalg.norm(b-a, axis=1)
old_radius, new_radius = np.linalg.norm(ac, axis=1), np.linalg.norm(bc, axis=1)
per_image = pd.DataFrame(dict(image=images, distance=delta,
    coordinate_rmse=delta/np.sqrt(512), raw_cosine=cosine_rows(a, b),
    own_centered_cosine=cosine_rows(ac, bc), old_radius=old_radius, new_radius=new_radius))
per_image.to_csv(OUT/'per_image_distances.csv', index=False)
summary = dict(overlap=len(images), old_total=old_total, new_total=new_total,
    distance_mean=float(delta.mean()), distance_median=float(np.median(delta)),
    distance_std=float(delta.std()), distance_min=float(delta.min()),
    distance_p05=float(np.quantile(delta,.05)), distance_p95=float(np.quantile(delta,.95)),
    distance_max=float(delta.max()), distance_rms=float(np.sqrt(np.mean(delta**2))),
    raw_cosine_mean=float(per_image.raw_cosine.mean()),
    own_centered_cosine_mean=float(per_image.own_centered_cosine.mean()),
    center_shift=float(np.linalg.norm(a.mean(0)-b.mean(0))),
    old_rms_radius=float(np.sqrt(np.mean(old_radius**2))),
    new_rms_radius=float(np.sqrt(np.mean(new_radius**2))))
summary['new_over_old_rms_radius'] = summary['new_rms_radius']/summary['old_rms_radius']
summary['rms_displacement_over_old_radius'] = summary['distance_rms']/summary['old_rms_radius']
# A deterministic wrong-identity pairing provides a distance scale, not a statistical test.
wrong_b = np.roll(b, 1, axis=0)
summary['mismatched_pair_mean_distance'] = float(np.linalg.norm(wrong_b-a, axis=1).mean())

backends = {label: SimpleNamespace(photo_to_coords=coords, dim_to_photo_to_ratings=ratings,
                                  device=torch.device('cpu'), dtype=torch.float32)
            for label, coords in [('old', old), ('new', new)]}
original_get_data = utils.get_data
observed_samples = {}
def aligned_get_data(dim, **kwargs):
    """Keep production preprocessing; explicitly align its set-based image output."""
    X, y, weights, names = original_get_data(dim, **kwargs)
    names = list(names)
    order = sorted(range(len(names)), key=lambda i: names[i])
    ids = torch.tensor(order, device=X.device)
    ordered_names = [names[i] for i in order]
    if not torch.isfinite(X).all() or not torch.isfinite(y).all():
        raise ValueError(f'{dim}: non-finite coordinates/logit mean ratings')
    if set(ordered_names) != set(images):
        raise ValueError(f'{dim}: missing ratings on the shared training set')
    observed_samples[dim] = len(names)
    return X[ids], y[ids], weights[ids], ordered_names

rows, coefficients = [], {}
with warnings.catch_warnings(record=True) as fit_warnings, threadpool_limits(limits=1), patch.object(utils, 'get_data', aligned_get_data), patch.object(utils, 'device', torch.device('cpu')):
    warnings.simplefilter('always')
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        for dim in ratings:
            vectors = {label: utils.ridge_coefs(dim, alpha=alpha, backend=backend).cpu()
                       for label, backend in backends.items()}
            va, vb = [vectors[label].numpy().astype(np.float64) for label in ['old', 'new']]
            cos = float(np.clip(cosine_rows(va, vb), -1, 1))
            rows.append(dict(target=dim, alpha=alpha, n=observed_samples[dim],
                             control='gender' if dim == 'age' else 'age', cosine=cos,
                             angle_degrees=float(np.degrees(np.arccos(cos))),
                             old_coef_norm=float(np.linalg.norm(va)),
                             new_coef_norm=float(np.linalg.norm(vb))))
            if alpha == 100:
                coefficients[dim] = vectors
results = pd.DataFrame(rows)
results.to_csv(OUT/'ridge_cosines_alpha_sweep.csv', index=False)
main = results[results.alpha == 100].sort_values('cosine')
main.to_csv(OUT/'ridge_cosines_alpha100.csv', index=False)
(OUT/'solver_warnings.txt').write_text('\n'.join(sorted({str(w.message) for w in fit_warnings}))+'\n')

# Same ridge objective in float64/SVD, grouped by shared control for efficiency.
# Retain float32 rating preprocessing exactly as supplied by get_data.
prepared = {dim: aligned_get_data(dim, backend=backends['old'], imgs=images)
            for dim in ratings}
ordered_names = prepared[next(iter(ratings))][3]
stable_rows, stable_coefficients = [], {}
outlier_name = images[int(np.argmax(delta))]
with threadpool_limits(limits=1):
    for scenario in ['all_overlap', 'without_extreme_outlier']:
        keep = np.array([scenario == 'all_overlap' or name != outlier_name for name in ordered_names])
        fitted = {'old': {}, 'new': {}}
        for label, coords in [('old', old), ('new', new)]:
            X = np.stack([coords[name].numpy() for name in ordered_names]).astype(np.float64)
            for control, dims in [('age', [d for d in ratings if d != 'age']), ('gender', ['age'])]:
                control_y = prepared[control][1].numpy().astype(np.float64)
                Y = np.stack([prepared[d][1].numpy() for d in dims], axis=1).astype(np.float64)
                design = np.column_stack([X, control_y])
                model = Ridge(alpha=100.0, fit_intercept=True, solver='svd').fit(design[keep], Y[keep])
                for dim, coef in zip(dims, np.atleast_2d(model.coef_)[:, :512]):
                    fitted[label][dim] = coef
        stable_coefficients[scenario] = fitted
        for dim in ratings:
            va, vb = fitted['old'][dim], fitted['new'][dim]
            cos = float(np.clip(cosine_rows(va, vb), -1, 1))
            prod = coefficients[dim]
            stable_rows.append(dict(scenario=scenario, target=dim, n=int(keep.sum()), cosine=cos,
                angle_degrees=float(np.degrees(np.arccos(cos))),
                old_cosine_to_production=float(cosine_rows(va, prod['old'].numpy())),
                new_cosine_to_production=float(cosine_rows(vb, prod['new'].numpy()))))
stable = pd.DataFrame(stable_rows)
stable.to_csv(OUT/'ridge_cosines_stable_alpha100.csv', index=False)
mask = np.array([name != outlier_name for name in images])
aa, bb = a[mask], b[mask]
rr_old = np.linalg.norm(aa-aa.mean(0), axis=1)
rr_new = np.linalg.norm(bb-bb.mean(0), axis=1)
robust_summary = dict(excluded_image=outlier_name, remaining=int(mask.sum()),
    distance_mean=float(delta[mask].mean()), distance_median=float(np.median(delta[mask])),
    distance_p95=float(np.quantile(delta[mask], .95)),
    center_shift=float(np.linalg.norm(aa.mean(0)-bb.mean(0))),
    old_rms_radius=float(np.sqrt(np.mean(rr_old**2))),
    new_rms_radius=float(np.sqrt(np.mean(rr_new**2))),
    own_centered_cosine_mean=float(cosine_rows(aa-aa.mean(0), bb-bb.mean(0)).mean()),
    mismatched_pair_mean_distance=float(np.linalg.norm(np.roll(bb,1,axis=0)-aa, axis=1).mean()))
robust_summary['new_over_old_rms_radius'] = robust_summary['new_rms_radius']/robust_summary['old_rms_radius']
(OUT/'summary_without_outlier.json').write_text(json.dumps(robust_summary, indent=2)+'\n')
torch.save({'alpha':100.0, 'images':images, 'directions':coefficients}, OUT/'ridge_coefficients.pt')
summary['ridge_alpha100_cosine_mean'] = float(main.cosine.mean())
summary['ridge_alpha100_cosine_median'] = float(main.cosine.median())
(OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

fig, axes = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={'width_ratios':[1, 1.3]})
axes[0].hist(delta[mask], bins=40, color='#3785a5')
axes[0].axvline(np.median(delta), color='#b74c36', label=f'Median {np.median(delta):.2f}')
axes[0].set(xlabel='Paired old–new W distance', ylabel='Images', title=f'{len(images):,} overlaps; {outlier_name} omitted from histogram')
axes[0].legend()
axes[1].barh(main.target, main.cosine, color='#3785a5')
axes[1].set(xlabel='Cosine of old vs new ridge W coefficients', xlim=(0,1), title='Same ratings and images; α = 100')
fig.tight_layout()
fig.savefig(OUT/'comparison.png', dpi=160)
plt.close(fig)

lines = ['# Old versus new OMI coordinates', '',
    'Old: `data/photo_to_coords.pkl`. New: `tests/image_to_coords_3k.pt`.',
    'Ratings: `data/dim_to_photo_to_ratings.pkl`.', '',
    f'All {len(images):,} shared filenames are 1.jpg–1004.jpg. All 34 targets and their controls have finite logit mean ratings for these images.', '',
    '## Paired latent distances', '', '| Statistic | Value |', '|---|---:|']
for key, value in summary.items():
    if key not in ['old_total', 'new_total']:
        lines.append(f'| {key} | {value:.6f} |')
lines += ['', 'Raw cosine is measured from the origin; centered cosine subtracts each cloud’s own mean. Radius uses the overlap cloud’s own center of mass. The wrong-identity baseline cyclically pairs each old image with the next new image; it is descriptive.', '',
    '## Ridge method', '',
    'Calls `utils.ridge_coefs` directly with overlap-only backends. Uses float32 W, logit of the mean available ratings, an intercept, no feature standardization, no rating-count sample weights, and an appended logit age rating (gender for age). The final control coefficient is excluded from direction cosine. The `get_data` wrapper only sorts returned rows by filename to ensure target/control alignment; it rejects missing or nonfinite observations.', '',
    'Primary α = 100 is the `_get_direction` default. The application’s cached path can use a different α via 2**((steps−30)/4); the sensitivity CSV includes α=1,10,100,1000. Cosines describe fitted direction agreement, not held-out predictive accuracy. No sign alignment or coordinate rescaling is applied.', '',
    '## Direction similarities (α = 100)', '', '| Target | N | Cosine | Angle (°) |', '|---|---:|---:|---:|']
for row in main.itertuples():
    lines.append(f'| {row.target} | {row.n} | {row.cosine:.6f} | {row.angle_degrees:.2f} |')
lines += ['', '## Regularization sensitivity', '', '| Alpha | Mean cosine | Min | Max |', '|---|---:|---:|---:|']
for alpha, group in results.groupby('alpha'):
    lines.append(f'| {alpha:g} | {group.cosine.mean():.6f} | {group.cosine.min():.6f} | {group.cosine.max():.6f} |')
lines += ['', '## Extreme outlier and numerical stability', '',
    f'{outlier_name} has displacement {delta.max():,.3f}; the next largest is {np.sort(delta)[-2]:.3f}. '
    'Full-overlap mean/radius are dominated by this finite but extreme coordinate vector. '
    'The production float32 solver emitted ill-conditioning warnings, saved in solver_warnings.txt. '
    'The table below solves the same alpha=100 ridge objective using float64 and SVD, '
    'retaining the existing float32 logit-rating preprocessing and no feature scaling. '
    'It separately repeats the fit with the outlier excluded from BOTH coordinate datasets; '
    'no source data were changed.', '',
    '| Target | Full overlap, stable | Without outlier, stable |', '|---|---:|---:|']
stable_pivot = stable.pivot(index='target', columns='scenario', values='cosine')
for dim, row in stable_pivot.iterrows():
    lines.append(f"| {dim} | {row['all_overlap']:.6f} | {row['without_extreme_outlier']:.6f} |")
lines += ['', '### Distance statistics excluding the outlier', '', '| Statistic | Value |', '|---|---:|']
for key, value in robust_summary.items():
    lines.append(f'| {key} | {value} |')
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary, indent=2))
print(main[['target','cosine','angle_degrees']].to_string(index=False))
print(results.groupby('alpha').cosine.agg(['mean','min','max']).to_string())
print('WITHOUT OUTLIER:', json.dumps(robust_summary))
print('STABLE COSINES:')
print(stable_pivot.to_string())
print(stable.groupby('scenario').cosine.agg(['mean','min','max']).to_string())
print('Max production vs stable full-overlap cosine difference:',
      (stable_pivot.all_overlap-main.set_index('target').cosine).abs().max())
print('Artifacts:', OUT)
