"""Paired outer CV of old/new W representations for predicting human ratings."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from scipy.special import expit
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils

OUT = ROOT / 'tests/old_vs_new_coords_cv'
OUT.mkdir(exist_ok=True)
SEED = 20260918
ALPHAS = np.logspace(-2, 5, 8)
old, ratings = utils.load_psychGAN_data(str(ROOT/'data'))
new = torch.load(ROOT/'tests/image_to_coords_3k.pt', map_location='cpu', weights_only=True)
images = sorted(set(old) & set(new))
targets = list(ratings)
backend = SimpleNamespace(photo_to_coords=old, dim_to_photo_to_ratings=ratings,
                          device=torch.device('cpu'), dtype=torch.float32)
Y = []
for target in targets:
    _, y, _, names = utils.get_data(target, backend=backend, imgs=images)
    lookup = dict(zip(names, y.numpy()))
    Y.append([lookup[name] for name in images])
Y = np.asarray(Y, dtype=np.float64).T
X = {label: np.stack([coords[name].detach().cpu().numpy().reshape(-1) for name in images]).astype(np.float64)
     for label, coords in [('old',old), ('new',new)]}
assert np.isfinite(Y).all() and all(np.isfinite(x).all() for x in X.values())
fold_ids = np.empty(len(images), dtype=int)
for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(images)):
    fold_ids[test] = fold

# Exact centered ridge/OLS solutions; reuse one float64 SVD across targets/alphas.
# Equivalent to Ridge(solver='svd', fit_intercept=True); alpha=0 is minimum-norm OLS.
def predict_grid(x_train, y_train, x_test, alphas):
    xm, ym = x_train.mean(0), y_train.mean(0)
    xc, yc = x_train-xm, y_train-ym
    u, s, vt = svd(xc, full_matrices=False, check_finite=False)
    uy = u.T @ yc
    projected = (x_test-xm) @ vt.T
    predictions = []
    for alpha in alphas:
        if alpha == 0:
            cutoff = np.finfo(float).eps*max(xc.shape)*s[0]
            factor = np.divide(1., s, out=np.zeros_like(s), where=s>cutoff)
        else:
            factor = s/(s*s+alpha)
        predictions.append((projected*factor) @ uy + ym)
    return np.stack(predictions)

def metrics(y, p):
    residual = p-y
    total = np.sum((y-y.mean())**2)
    yr, pr = expit(y), expit(p)
    return dict(r2_logit=1-np.sum(residual**2)/total,
                rmse_logit=np.sqrt(np.mean(residual**2)),
                pearson_logit=np.corrcoef(y,p)[0,1],
                r2_rating=1-np.sum((yr-pr)**2)/np.sum((yr-yr.mean())**2),
                mae_rating=np.mean(np.abs(yr-pr)))

rows, fold_rows, tuning_rows, predictions_archive = [], [], [], {}
with threadpool_limits(limits=1):
    for scenario in ['all_overlap', 'without_678']:
        keep = np.array([scenario == 'all_overlap' or name != '678.jpg' for name in images])
        yy, ff = Y[keep], fold_ids[keep]
        names = np.asarray(images)[keep]
        predictions_archive[scenario+'__images'] = names
        predictions_archive[scenario+'__fold'] = ff
        predictions_archive[scenario+'__y_logit'] = yy
        for feature_set in ['coords_only', 'coords_plus_control']:
            for label in ['old','new']:
                xx = X[label][keep]
                preds = {model: np.full_like(yy, np.nan) for model in ['OLS','ridge100','ridge_tuned','train_mean']}
                if feature_set == 'coords_only':
                    groups = [(np.arange(len(targets)), xx)]
                else:
                    age, gender = targets.index('age'), targets.index('gender')
                    groups = [(np.array([i for i in range(len(targets)) if i != age]), np.column_stack([xx,yy[:,age]])),
                              (np.array([age]), np.column_stack([xx,yy[:,gender]]))]
                for fold in range(5):
                    train, test = np.flatnonzero(ff != fold), np.flatnonzero(ff == fold)
                    preds['train_mean'][test] = yy[train].mean(0)
                    for target_indices, design in groups:
                        yt = yy[:, target_indices]
                        inner_sse = np.zeros((len(ALPHAS), len(target_indices)))
                        for inner_train, inner_valid in KFold(3, shuffle=True, random_state=SEED+fold).split(train):
                            it, iv = train[inner_train], train[inner_valid]
                            pp = predict_grid(design[it], yt[it], design[iv], ALPHAS)
                            inner_sse += np.square(pp-yt[iv][None]).sum(axis=1)
                        best = np.argmin(inner_sse, axis=0)
                        grid = predict_grid(design[train], yt[train], design[test], [0.,100.,*ALPHAS])
                        preds['OLS'][np.ix_(test,target_indices)] = grid[0]
                        preds['ridge100'][np.ix_(test,target_indices)] = grid[1]
                        preds['ridge_tuned'][np.ix_(test,target_indices)] = grid[2+best,:,np.arange(len(best))].T
                        for target_index, choice in zip(target_indices, best):
                            tuning_rows.append(dict(scenario=scenario, feature_set=feature_set, coords=label,
                                fold=fold, target=targets[target_index], alpha=ALPHAS[choice]))
                    print(f'{scenario} | {feature_set} | {label} | fold {fold+1}/5', flush=True)
                for model, p in preds.items():
                    assert np.isfinite(p).all()
                    prefix = dict(scenario=scenario, feature_set=feature_set, coords=label, model=model)
                    predictions_archive['__'.join(prefix.values())] = p
                    for j, target in enumerate(targets):
                        rows.append(dict(**prefix, target=target, n=len(yy), **metrics(yy[:,j],p[:,j])))
                        for fold in range(5):
                            test = ff == fold
                            fold_rows.append(dict(**prefix,target=target,fold=fold,n=int(test.sum()),
                                                  **metrics(yy[test,j],p[test,j])))

scores = pd.DataFrame(rows)
scores.to_csv(OUT/'per_target_oof_metrics.csv', index=False)
pd.DataFrame(fold_rows).to_csv(OUT/'per_fold_metrics.csv', index=False)
tuning = pd.DataFrame(tuning_rows)
tuning.to_csv(OUT/'selected_alphas.csv', index=False)
np.savez_compressed(OUT/'out_of_fold_predictions.npz', targets=np.array(targets), **predictions_archive)
metric_names = ['r2_logit','rmse_logit','pearson_logit','r2_rating','mae_rating']
summary = scores.groupby(['scenario','feature_set','model','coords'])[metric_names].mean()
summary.to_csv(OUT/'summary.csv')
wide = scores.pivot(index=['scenario','feature_set','model','target'], columns='coords', values=metric_names)
paired = pd.DataFrame(index=wide.index)
for metric in metric_names:
    paired[metric+'_old'] = wide[(metric,'old')]
    paired[metric+'_new'] = wide[(metric,'new')]
    paired[metric+'_delta_new_minus_old'] = wide[(metric,'new')]-wide[(metric,'old')]
paired.to_csv(OUT/'paired_target_comparison.csv')
wins = paired.groupby(level=['scenario','feature_set','model']).agg(
    new_wins_logit_r2=('r2_logit_delta_new_minus_old',lambda s:int((s>0).sum())),
    new_wins_rating_r2=('r2_rating_delta_new_minus_old',lambda s:int((s>0).sum())),
    mean_delta_logit_r2=('r2_logit_delta_new_minus_old','mean'),
    median_delta_logit_r2=('r2_logit_delta_new_minus_old','median'))
wins.to_csv(OUT/'win_counts.csv')
method = dict(seed=SEED, outer_folds=5,inner_folds=3,alpha_grid=ALPHAS.tolist(),
    overlap=len(images),target_count=len(targets),outlier='678.jpg')
(OUT/'method.json').write_text(json.dumps(method,indent=2)+'\n')
lines = ['# Paired cross-validation of old and new coordinates', '',
    'Predict human ratings from W coordinates, not coordinates from ratings. Both representations use the same overlapping images, targets, and shuffled 5-fold outer splits. Every saved prediction is made by a model that did not train on that image. The outlier-excluded scenario removes 678.jpg from both datasets while preserving the remaining outer fold assignments.', '',
    'Primary evaluation is coordinates only. The separate coords_plus_control condition matches the application by appending logit mean age (gender for predicting age); it assumes that control rating is known for the held-out image. It therefore is not prediction from W alone.', '',
    'Targets reuse utils.get_data: logit of each image’s mean nonmissing ratings, without sample weights. All model fits use train-only centering, an intercept, raw unstandardized coordinates, and float64 SVD. OLS uses the minimum-norm solution. Ridge100 uses alpha=100. Ridge_tuned selects alpha independently for each target and representation using 3-fold CV inside each outer training fold, minimizing logit MSE over '+str(ALPHAS.tolist())+'. No outer test labels select alpha.', '',
    'Scores below pool all out-of-fold predictions per target, then average equally across the 34 targets. R² may be negative and has no lower bound. Rating-scale metrics sigmoid-transform the predicted logits and compare with original 0–1 mean ratings. Train_mean predicts the outer training fold’s mean logit and is a baseline. Fold-level scores, predictions, and per-target comparisons are saved alongside this report.', '',
    '## Mean scores across targets', '', '| Scenario | Inputs | Model | Coordinates | Logit R² | Rating R² | Rating MAE |', '|---|---|---|---|---:|---:|---:|']
for idx, row in summary.iterrows():
    lines.append('| '+' | '.join(idx)+f' | {row.r2_logit:.6g} | {row.r2_rating:.4f} | {row.mae_rating:.4f} |')
lines += ['', '## Per-target tuned ridge: coordinates only, excluding 678.jpg', '',
          '| Target | Old logit R² | New logit R² | Difference |', '|---|---:|---:|---:|']
selected = paired.loc[('without_678','coords_only','ridge_tuned')].sort_values('r2_logit_delta_new_minus_old',ascending=False)
for target,row in selected.iterrows():
    lines.append(f'| {target} | {row.r2_logit_old:.4f} | {row.r2_logit_new:.4f} | {row.r2_logit_delta_new_minus_old:+.4f} |')
lines += ['', 'Predictive performance measures linear accessibility of ratings in these coordinates. It does not establish image reconstruction quality or generative-distribution plausibility. This is one randomized 5-fold partition, not an independent external validation cohort. The exclusion was based on the previously identified coordinate outlier, not a CV-score optimization. No source coordinates or application code were modified.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print('\nSUMMARY\n',summary.to_string())
print('\nWIN COUNTS\n',wins.to_string())
print('\nCLEAN TUNED COORDS ONLY\n',selected[['r2_logit_old','r2_logit_new','r2_logit_delta_new_minus_old']].to_string())
print('Saved:',OUT)
