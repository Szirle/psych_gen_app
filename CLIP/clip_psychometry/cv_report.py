"""Matched all-target reports from fgclip2_multitarget heldout artifacts.

Usage: python -m CLIP.clip_psychometry.cv_report RUN_DIRECTORY
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .metrics import paired_contrast, regression
from .pruning import all_target_summary
from .schema import write_json


def compare_run(path, baseline='frozen-kernel', candidate='agop-kernel'):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    targets = manifest['targets']
    arrays = []
    for name in (baseline, candidate):
        with np.load(path/name/'heldout.npz') as f:
            order = np.argsort(f['rows'])
            arrays.append({k: f[k][order] for k in ('rows', 'predictions', 'y', 'mask', 'fold_ids')})
    a, b = arrays
    for key in ('rows', 'y', 'mask', 'fold_ids'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'Unmatched {key}; cannot produce a paired report.')
    if not np.array_equal(a['rows'], np.arange(len(manifest['paths']))):
        raise ValueError('This report requires complete outer-CV coverage of every image.')
    y = np.where(a['mask'], a['y'], np.nan)
    group = np.asarray(manifest['groups'])[a['rows']]
    summary = all_target_summary(y, a['predictions'], b['predictions'], targets)
    details = {}
    for t, target in enumerate(targets):
        fold_changes = []
        for fold in np.unique(a['fold_ids']):
            take = (a['fold_ids']==fold) & a['mask'][:, t]
            ref = np.mean((y[take, t]-a['predictions'][take, t])**2)
            cur = np.mean((y[take, t]-b['predictions'][take, t])**2)
            fold_changes.append(float(100*(cur/ref-1)))
        details[target] = dict(baseline=regression(y[:, t], a['predictions'][:, t]),
            candidate=regression(y[:, t], b['predictions'][:, t]),
            paired=paired_contrast(y[:, t], a['predictions'][:, t], b['predictions'][:, t],
                                   groups=group, samples=2000, seed=manifest['config']['seed']),
            mse_change_pct=100*(summary['mse_ratios'][target]-1), fold_mse_changes_pct=fold_changes)
    selection = []
    for f in sorted((path/candidate).glob('fold-*/refit/selection.json')):
        selection.append(json.loads(f.read_text()))
    selected_counts, union = [], set()
    for fold in selection:
        for d in fold['targets'].values():
            selected_counts.append(d['phrases'])
            p = len(fold['phrases'])
            union.update(fold['phrases'][c % p] for c in d['columns'])
    result = dict(summary=summary, targets=details, baseline=baseline, candidate=candidate,
        mean_selected_phrases=float(np.mean(selected_counts)) if selected_counts else None,
        union_selected_phrases=len(union),
        caveat='Internal CV includes previously explored images. Bootstrap is conditional on fitted OOF predictions, not retraining or search. No outer-label fallback was applied.')
    write_json(path/'paired_comparison.json', result)
    rmse_a = np.sqrt(np.mean([d['baseline']['mse'] for d in details.values()]))
    rmse_b = np.sqrt(np.mean([d['candidate']['mse'] for d in details.values()]))
    lines = ['# Full all-target default-model comparison', '',
        f"{len(y)} images, {len(targets)} targets, {len(np.unique(a['fold_ids']))} matched outer folds; "
        f"{manifest['config']['inner_folds']} inner folds; seed {manifest['config']['seed']}.", '',
        f"Old/new target-macro RMSE: **{rmse_a:.6f} → {rmse_b:.6f}**. "
        f"Mean relative MSE change: **{100*(summary['macro_mse_ratio']-1):+.2f}%**. "
        f"Improved {summary['improved']}/{len(targets)}; worsened {summary['worsened']}/{len(targets)}; "
        f"worst target change {100*(summary['worst_mse_ratio']-1):+.2f}%.", '',
        result['caveat'], '',
        'Negative ΔMSE favors the new model. Rating units are 0–1. Fold wins count lower MSE, not significance.', '',
        '| Target | Old RMSE | New RMSE | ΔMSE % | ΔR² | Fold wins |',
        '|---|---:|---:|---:|---:|---:|']
    for name, d in details.items():
        lines.append(f"| {name} | {d['baseline']['rmse']:.5f} | {d['candidate']['rmse']:.5f} | "
            f"{d['mse_change_pct']:+.2f} | {d['candidate']['r2']-d['baseline']['r2']:+.4f} | "
            f"{sum(v<0 for v in d['fold_mse_changes_pct'])}/{len(d['fold_mse_changes_pct'])} |")
    lines += ['', '## Interpretation', '',
        'The treatment combines the 80 added phrases with target-specific pruning. This comparison '
        'estimates their combined utility; it cannot attribute gains separately to expansion and pruning. '
        'Each target independently selects its retention and kernel strategy using training-only inner folds. '
        'There is no fallback selected using the outer-fold outcomes.', '',
        f"Average retained phrases per target/fold: {result['mean_selected_phrases']}. "
        f"Union across targets and folds: {len(union)} phrases. A smaller individual mask does not imply a smaller encoder bank.", '',
        'Per-target paired bootstrap intervals and fold changes are in `paired_comparison.json`. '
        'They quantify image-sampling variability conditional on the fitted predictions; they are neither '
        'simultaneous all-target noninferiority guarantees nor independent confirmation after model exploration.', '',
        '## Deployment', '',
        'The user-authorized default is `agop-kernel`. The old baseline remains selectable. '
        'Final all-image fits are deployment artifacts, not additional validation. See '
        '[CURRENT_MODEL.md](../../../fgclip2_multitarget/CURRENT_MODEL.md) for the exact algorithm, '
        'checkpoint inference and future directions.']
    (path/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(compare_run(args.run)['summary'])
