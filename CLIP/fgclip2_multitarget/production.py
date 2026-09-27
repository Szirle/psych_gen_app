"""Resumable 10x10 nested B2 fitting, equal-weight deployment ensemble, OOF ceilings."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..race_perception import cv_splits
from ..clip_psychometry.weighted_kernel import fit_weighted_kernel, predict_weighted_kernel, OPTIONS
from ..fgclip2_face_impressions import _atomic_save_npz
from .assets import atomic_save
from .agop import expanded_bank
from .ceiling_report import rating_summaries, write_report, estimates


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def run(args):
    if min(args.folds,args.inner_folds) < 2 or args.cpu_threads < 1 or args.bootstrap < 0:
        raise ValueError('Require at least two folds, positive threads and nonnegative bootstrap count.')
    source, out = args.source.resolve(), args.output.resolve()
    m = json.loads((source/'manifest.json').read_text())
    cache = args.score_cache or Path(json.loads((source/'checkpoint_roundtrip.json').read_text())['score_cache'])
    ratings = args.ratings or Path(m['config']['ratings'])
    names, paths, phrases = m['targets'], m['paths'], m['bank']['phrases']
    if expanded_bank(m['baseline_bank'])['phrases'] != phrases:
        raise ValueError('Source bank must be the complete expanded B2 phrase bank.')
    with np.load(cache) as f:
        if f['paths'].tolist() != paths or f['phrases'].tolist() != phrases:
            raise ValueError('Cache paths/phrases do not exactly match the source manifest.')
        x = np.asarray(f['scores'], dtype=float)
    y, se, counts = rating_summaries(ratings, names, paths)
    with np.load(source/'agop-kernel/heldout.npz') as f:
        order = np.argsort(f['rows'])
        if (not np.array_equal(f['rows'][order], np.arange(len(paths))) or not f['mask'].all()
                or not np.allclose(f['y'][order], y, atol=1e-8, rtol=1e-7)):
            raise ValueError('Ratings do not match source OOF labels and complete row coverage.')
    if len(x) != len(y) or x.shape[1] != 2*len(phrases) or not np.isfinite(x).all():
        raise ValueError('Invalid source scores.')
    groups = np.asarray(m['groups'])
    seed = m['config']['seed'] if args.seed is None else args.seed
    package = Path(__file__).resolve().parent
    code_paths = [package/n for n in ('production.py','agop.py','ceiling_report.py','assets.py')]
    code_paths += [package.parent/'clip_psychometry'/n for n in ('weighted_kernel.py','learned_geometry.py','adapters.py')]
    code_paths += [package.parent/'race_perception/__init__.py',package.parent/'fgclip2_face_impressions.py']
    code_hash = hashlib.sha256(''.join(sha256(p) for p in code_paths).encode()).hexdigest()
    design = dict(format='b2_production_run_v1', source_identity=m['identity'],
        source=str(source), cache=str(Path(cache).resolve()), cache_sha256=sha256(cache),
        ratings=str(Path(ratings).resolve()), ratings_sha256=sha256(ratings),
        folds=args.folds, inner_folds=args.inner_folds, seed=seed, options=list(OPTIONS),
        bootstrap=args.bootstrap, code_sha256=code_hash,
        targets=names, paths=paths, groups=groups.tolist(), phrases=phrases,
        model=m['model'], revision=m['revision'], precision=m['config']['precision'])
    design['identity'] = hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != design:
            raise ValueError('Resume identity changed. Use a new output directory.')
    elif any(out.iterdir()):
        raise ValueError('A new run requires an empty output directory.')
    save_json(out/'manifest.json', design)
    config = dict(m['config'], variants=['agop-kernel'], folds=args.folds,
        inner_folds=args.inner_folds, seed=seed, output=str(out), agop_algorithm='continuous-B2-v1')
    config.pop('agop_keeps', None)
    metadata = dict(targets=names, phrases=phrases, config=config, model=m['model'], revision=m['revision'])
    prediction = np.full_like(y,np.nan)
    fold_ids = np.full(len(y),-1,dtype=int)
    members = []
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        for fold,(train,test) in enumerate(cv_splits(len(y),args.folds,seed,groups)):
            if np.intersect1d(groups[train],groups[test]).size or np.any(fold_ids[test] != -1):
                raise ValueError('Outer split overlap.')
            d = out/f'fold-{fold}'
            checkpoint, saved, done = d/'model.pt',d/'predictions.npz',d/'complete.json'
            if done.exists() and checkpoint.exists() and saved.exists():
                prior = json.loads(done.read_text())
                if prior['identity'] != design['identity'] or prior['checkpoint_sha256'] != sha256(checkpoint):
                    raise ValueError('Completed fold identity/checkpoint mismatch.')
                with np.load(saved) as f:
                    if not np.array_equal(f['train'],train) or not np.array_equal(f['test'],test):
                        raise ValueError('Stored fold rows differ from current splits.')
                    pred = f['predictions']
                print(f'Fold {fold+1}/{args.folds}: reusing completed fit',flush=True)
            else:
                print(f'Fold {fold+1}/{args.folds}: train {len(train)}, test {len(test)}',flush=True)
                start = time.monotonic()
                model = fit_weighted_kernel(x[train],y[train],phrases,groups[train],
                                            inner_folds=args.inner_folds,seed=seed)
                pred = predict_weighted_kernel(model,x[test])
                packet = dict(format='fgclip2_multitarget_agop_weighted_v1', **metadata,
                    feature_order='global-short then local-box-max',
                    training_rows=train.tolist(), run_identity=design['identity'],
                    blocks=[dict(targets=list(range(len(names))),model=model)])
                atomic_save(packet,checkpoint)
                save_json(d/'selection.json',dict(targets=dict(zip(names,model['selection'])),options=list(OPTIONS)))
                del model,packet
                _atomic_save_npz(saved,train=train,test=test,predictions=pred,y=y[test])
                values=estimates(y[test],pred,se[test])
                save_json(d/'metrics.json',{name:{k:float(v[t]) if np.isfinite(v[t]) else None for k,v in values.items()}
                                           for t,name in enumerate(names)})
                prior=dict(identity=design['identity'],checkpoint_sha256=sha256(checkpoint),
                           seconds=time.monotonic()-start,train_rows=train.tolist(),test_rows=test.tolist())
                save_json(done,prior)
            if pred.shape != y[test].shape or not np.isfinite(pred).all():
                raise ValueError('Invalid saved fold predictions.')
            prediction[test],fold_ids[test] = pred,fold
            members.append(dict(path=str(checkpoint.relative_to(out)),weight=1/args.folds,
                                sha256=prior['checkpoint_sha256']))
    if np.any(fold_ids < 0): raise ValueError('Incomplete OOF coverage.')
    ensemble=dict(format='fgclip2_multitarget_agop_ensemble_v1',**metadata,members=members,
                  run_identity=design['identity'],aggregation='equal arithmetic mean of predictions')
    save_json(out/'ensemble.json',ensemble)
    write_report(out,paths,names,y,prediction,se,counts,fold_ids,groups,bootstrap=args.bootstrap,seed=seed)
    save_json(out/'complete.json',dict(identity=design['identity'],images=len(y),folds=args.folds,
        ensemble='ensemble.json',oof='oof_predictions.npz',report='REPORT.md'))
    print(f'Complete: {out}/ensemble.json and REPORT.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--score-cache',type=Path)
    p.add_argument('--ratings',type=Path)
    p.add_argument('--folds',type=int,default=10)
    p.add_argument('--inner-folds',type=int,default=10)
    p.add_argument('--seed',type=int)
    p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--bootstrap',type=int,default=1000)
    run(p.parse_args())

if __name__ == '__main__': main()
