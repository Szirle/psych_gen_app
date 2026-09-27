"""OOF accuracy and heteroskedastic mean-rating noise ceilings; see PRODUCTION.md."""
import csv
import json
import pickle
from pathlib import Path
import numpy as np

PRIMARY_OMIT = {'looks-like-you', 'memorable'}


def rating_summaries(path, names, paths):
    # Existing trusted-local project format contains lists, not rater IDs.
    with Path(path).open('rb') as f:
        raw = pickle.load(f)
    y = np.full((len(paths), len(names)), np.nan)
    se, counts = y.copy(), np.zeros(y.shape, dtype=int)
    for i, image in enumerate(paths):
        for t, name in enumerate(names):
            r = np.asarray(raw[name].get(Path(image).name, []), dtype=float).reshape(-1)
            if len(r) < 2 or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
                raise ValueError(f'Missing/invalid individual ratings: {image}/{name}')
            y[i, t], counts[i, t] = r.mean(), len(r)
            se[i, t] = r.std(ddof=1)/np.sqrt(len(r))
    return y, se, counts


def estimates(y, prediction, se):
    vy = y.var(axis=0, ddof=1)
    noise = np.mean(se**2, axis=0)
    reliability = np.divide(vy-noise, vy, out=np.full_like(vy, np.nan), where=vy>0)
    ceiling = np.sqrt(np.clip(reliability, 0, 1))
    a, b = y-y.mean(0), prediction-prediction.mean(0)
    denominator = np.sqrt((a*a).sum(0)*(b*b).sum(0))
    r = np.divide((a*b).sum(0), denominator, out=np.full_like(vy, np.nan), where=denominator>0)
    mse = ((prediction-y)**2).mean(0)
    return dict(pearson=r, mse=mse, rmse=np.sqrt(mse), ceiling=ceiling,
                observed_variance=vy, noise_variance=noise, signal_variance=vy-noise,
                raw_mean_reliability=reliability)


def _jsonable(x):
    if isinstance(x, dict): return {k: _jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return _jsonable(x.tolist())
    if isinstance(x, np.generic): return _jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write_report(out, paths, names, y, prediction, se, counts, fold_ids, groups, *, bootstrap=1000, seed=20260927):
    out = Path(out)
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Require complete aligned OOF predictions.')
    pooled = estimates(y, prediction, se)
    primary = [t for t,n in enumerate(names) if n not in PRIMARY_OMIT]
    def target_rows(values):
        result = {name: {k: float(v[t]) for k,v in values.items()} for t,name in enumerate(names)}
        for name, row in result.items():
            row['ceiling_status'] = ('undefined_constant_means' if not np.isfinite(row['ceiling'])
                else 'nonpositive_signal_estimate' if row['signal_variance'] <= 0 else 'estimated')
        return result
    report = dict(targets=target_rows(pooled), primary_targets=[names[t] for t in primary],
        pooled_primary=dict(mean_mse=float(pooled['mse'][primary].mean()),
                            root_mean_mse=float(np.sqrt(pooled['mse'][primary].mean())),
                            mean_pearson=float(np.nanmean(pooled['pearson'][primary]))), folds={})
    for fold in np.unique(fold_ids):
        rows = fold_ids == fold
        v = estimates(y[rows], prediction[rows], se[rows])
        report['folds'][str(fold)] = dict(n=int(rows.sum()), targets=target_rows(v),
            primary_mean_mse=float(v['mse'][primary].mean()), primary_mean_pearson=float(np.nanmean(v['pearson'][primary])))
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = {k: [] for k in ('pearson', 'ceiling')}
    for _ in range(bootstrap):
        rows = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        v = estimates(y[rows], prediction[rows], se[rows])
        for k in draws: draws[k].append(v[k])
    if bootstrap:
        for k, v in draws.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5], axis=0)
            for t,n in enumerate(names): report['targets'][n][k+'_ci95'] = [float(lo[t]), float(hi[t])]
    report['notes'] = ['OOF evaluates held-out fold models, not the ten-member deployment ensemble.',
        'Ceiling assumes independent sampled raters and independent mean errors across images; rater IDs unavailable.',
        'Bootstrap resamples image groups conditional on OOF predictions, without retraining.',
        'All prior exploration used this cohort: internal evaluation only.']
    (out/'metrics.json').write_text(json.dumps(_jsonable(report), indent=2)+'\n')
    with (out/'oof_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row','path','fold','target','human_mean','prediction','raters','mean_se'])
        writer.writerows((i,path,int(fold_ids[i]),name,y[i,t],prediction[i,t],int(counts[i,t]),se[i,t])
                         for i,path in enumerate(paths) for t,name in enumerate(names))
    np.savez_compressed(out/'oof_predictions.npz', paths=np.asarray(paths), targets=np.asarray(names),
        rows=np.arange(len(y)), predictions=prediction, y=y, mean_se=se, counts=counts,
        fold_ids=fold_ids, groups=groups)
    lines = ['# B2 ten-fold production evaluation', '',
        f'{len(y)} images; {len(names)} targets; {len(np.unique(fold_ids))} outer folds.', '',
        'Primary averages omit looks-like-you and memorable. All target results remain below.', '',
        f"Primary mean MSE: {report['pooled_primary']['mean_mse']:.8f}; mean Pearson r: {report['pooled_primary']['mean_pearson']:.4f}.", '',
        '## Held-out results by fold', '', '| Fold | Test images | Primary mean MSE | Primary mean r |', '|---|---:|---:|---:|']
    for fold,v in report['folds'].items():
        lines.append(f"| {int(fold)+1} | {v['n']} | {v['primary_mean_mse']:.8f} | {v['primary_mean_pearson']:.4f} |")
    lines += ['', '## Pooled out-of-fold results', '',
              '| Target | Pearson r | Estimated Pearson ceiling | MSE | RMSE |', '|---|---:|---:|---:|---:|']
    order = np.argsort(-np.nan_to_num(pooled['ceiling'], nan=-1))
    for t in order:
        v=report['targets'][names[t]]
        lines.append(f"| {names[t]} | {v['pearson']:.4f} | {v['ceiling']:.4f} | {v['mse']:.8f} | {v['rmse']:.5f} |")
    lines += ['', *report['notes'], '', 'Per-fold per-target metrics and bootstrap intervals are in metrics.json.',
              'Ceiling = sqrt(max(0, 1 − mean(s_i²/n_i)/Var(image means))). Raw unclipped reliability is retained.',
              '![OOF correlation and estimated prediction ceiling](prediction_ceiling.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(11, max(8, .42*len(names)+2)))
    yy = np.arange(len(names))
    c, r = pooled['ceiling'][order], pooled['pearson'][order]
    ax.barh(yy, c, color='#f08080', height=.62, zorder=1)
    ax.barh(yy, r, color='black', height=.36, zorder=2)
    for pos, ci, ri in zip(yy,c,r):
        if np.isfinite(ci): ax.text(ci+.009,pos-.20,f'{ci:.3f}',color='#d33',fontsize=8,va='center')
        if np.isfinite(ri): ax.text(ri+.009,pos+.18,f'{ri:.3f}',color='black',fontsize=8,va='center')
    ax.set_yticks(yy, [names[t].replace('-', ' ') for t in order])
    ax.invert_yaxis()
    finite = r[np.isfinite(r)]
    ax.set_xlim(min(0., float(finite.min())-.03) if len(finite) else 0.,1.09)
    ax.set_xticks(np.arange(0,1.01,.1))
    ax.set_xlabel('Pearson correlation (r)')
    ax.legend(handles=[Patch(color='black',label='Out-of-fold prediction'),
        Patch(color='#f08080',label='Estimated prediction ceiling of mean ratings')],
        loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'prediction_ceiling.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    return report
