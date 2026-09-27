"""Nested all-target AGOP pruning, original readout and baseline-protected policies."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from .adapters import _repo
from .evaluation import group_splits,candidates
from .pruning import AGOPConfig,rank_paths,units,select_units,all_target_summary
from .schema import write_json

KEEPS=(1.,.95,.9,.75,.5,.25,.1)
KERNELS=(('linear',1.),('poly2',1.),('rbf',.1),('rbf',1.))
ALPHAS=tuple(np.logspace(-4,1,11))
RECIPES=[dict(family=f,gamma=g,alpha=a,bank='both',centered=False) for f,g in KERNELS for a in ALPHAS]
POLICIES=('baseline','expanded','prune_base','prune_expanded','guard_0','guard_1','guard_3','guard_10','guard_consistent')


def scaled_predictions(train,y,query,columns):
    x,q=train[:,columns],query[:,columns];m=x.mean(0);s=x.std(0);s=np.where(s>1e-12,s,1.)
    return candidates((x-m)/s,y,(q-m)/s,KERNELS,ALPHAS)


def fit_bank_fold(d,train,test,columns,config,seed,unit='phrase',keeps=KEEPS):
    native=_repo('race_perception')
    x,y=d.x[np.ix_(train,columns)],d.y[train];query=d.x[np.ix_(test,columns)]
    keys,groups=units(d.registry,columns,unit)
    options=[(0,1.)]+[(step,k) for step in range(config.rounds) for k in keeps if k < 1.]
    inner=native.cv_splits(len(train),3,seed,d.groups[train]);inner_ids=np.empty(len(train),int)
    oof=np.empty((len(train),len(options),len(RECIPES),y.shape[1]))
    for fold,(tr,va) in enumerate(inner):
        inner_ids[va]=fold
        ranks=rank_paths(x[tr],y[tr],config,groups=d.groups[train[tr]])
        oof[va,0]=scaled_predictions(x[tr],y[tr],x[va],np.arange(x.shape[1]))
        for t in range(y.shape[1]):
            memo={}
            for j,(step,keep) in enumerate(options[1:],1):
                mask,_=select_units(ranks[step,:,t],groups,keep);key=tuple(mask)
                if key not in memo:memo[key]=scaled_predictions(x[tr],y[tr,t:t+1],x[va],mask)[:,:,0]
                oof[va,j,:,t]=memo[key]
    strategies=[];losses=[];inner_predictions=[]
    for j in range(len(options)):
        s,_,_=native.select_strategies(oof[:,j],y,RECIPES)
        p=native.apply_strategies(oof[:,j],s)
        strategies.append(s);losses.append(((p-y)**2).mean(0));inner_predictions.append(p)
    losses=np.asarray(losses);best=losses.argmin(0)
    rank=rank_paths(x,y,config,groups=d.groups[train])
    full=scaled_predictions(x,y,query,np.arange(x.shape[1]))
    selected=np.empty((len(test),y.shape[1]));unpruned=native.apply_strategies(full,strategies[0]);chosen=[]
    selected_inner=np.empty_like(y);masks=np.zeros((y.shape[1],d.x.shape[1]),bool)
    for t,j in enumerate(best):
        step,keep=options[j];mask,importance=select_units(rank[step,:,t],groups,keep)
        if j:
            p=scaled_predictions(x,y[:,t:t+1],query,mask)[:,:,0]
            s=strategies[j][t];selected[:,t]=sum(w*p[:,i] for i,w in zip(s['indices'],s['weights']))
        else:selected[:,t]=unpruned[:,t]
        selected_inner[:,t]=inner_predictions[j][:,t]
        masks[t,np.asarray(columns)[mask]]=True
        chosen.append(dict(round=step+1,keep=keep,phrases=int(len(mask)//2),inner_mse=float(losses[j,t]),strategy=strategies[j][t]))
    # Per-fold artifacts let all-target policies be changed without rerunning fits.
    return dict(full=unpruned,pruned=selected,inner_full=inner_predictions[0],inner_pruned=selected_inner,
                full_loss=losses[0],best_loss=losses[best,np.arange(y.shape[1])],
                chosen=chosen,masks=masks,rank=rank,unit_names=keys,groups=groups,
                inner_folds=inner_ids,curves=losses,options=options,strategies=strategies,
                all_inner=np.stack(inner_predictions,1))


def run(d,output,seed=20260926,config=AGOPConfig(),keeps=KEEPS):
    if not np.isfinite(d.y).all():raise ValueError('This study requires full target coverage; use target-mask subsets for missing labels.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    fingerprint=d.fingerprint();base=tuple(d.registry.columns('baseline'));expanded=tuple(range(d.x.shape[1]))
    result={k:np.empty_like(d.y) for k in POLICIES};selections=[];outer=group_splits(d.groups,5,seed)
    module_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('pruning.py').read_bytes()).hexdigest()
    with threadpool_limits(limits=1):
        for fold,(train,test) in enumerate(outer):
            banks={}
            for label,cols in [('base',base),('expanded',expanded)]:
                print(f'Fold {fold+1}/5: {label}, {len(d.targets)} targets',flush=True)
                banks[label]=fit_bank_fold(d,train,test,cols,config,seed,keeps=keeps)
            a,b=banks['base'],banks['expanded']
            local={'baseline':a['full'],'expanded':b['full'],'prune_base':a['pruned'],'prune_expanded':b['pruned']}
            choose_b=b['best_loss']<a['best_loss'];best_loss=np.minimum(a['best_loss'],b['best_loss'])
            best_pred=np.where(choose_b[None,:],b['pruned'],a['pruned'])
            best_inner=np.where(choose_b[None,:],b['inner_pruned'],a['inner_pruned'])
            improvement=1-best_loss/a['full_loss']
            accepted={}
            for margin in (0,1,3,10):
                name=f'guard_{margin}';take=improvement>margin/100+1e-12;accepted[name]=take
                local[name]=np.where(take[None,:],best_pred,a['full'])
            fold_gains=np.array([1-((best_inner[a['inner_folds']==f]-d.y[train][a['inner_folds']==f])**2).mean(0)/
                ((a['inner_full'][a['inner_folds']==f]-d.y[train][a['inner_folds']==f])**2).mean(0) for f in range(3)])
            take=(fold_gains>.01).all(0);accepted['guard_consistent']=take
            local['guard_consistent']=np.where(take[None,:],best_pred,a['full'])
            for k,v in local.items():result[k][test]=v
            details=dict(fold=fold,train=train.tolist(),test=test.tolist(),base=a['chosen'],expanded=b['chosen'],
                         choose_expanded=choose_b,inner_relative_gain=improvement,accepted=accepted,fold_gains=fold_gains)
            selections.append(details);write_json(out/f'fold-{fold}.json',details)
            write_json(out/f'fold-{fold}-strategies.json',{k:v['strategies'] for k,v in banks.items()})
            np.savez_compressed(out/f'fold-{fold}.npz',test=test,train=train,**local,
                base_masks=a['masks'],expanded_masks=b['masks'],base_rank=a['rank'],expanded_rank=b['rank'],
                base_curves=a['curves'],expanded_curves=b['curves'],options=np.asarray(a['options']),
                base_inner=a['all_inner'],expanded_inner=b['all_inner'],inner_folds=a['inner_folds'])
            print(f'Completed fold {fold+1}/5',flush=True)
    report=dict(data_sha256=fingerprint,source_sha256=module_hash,seed=seed,config=asdict(config),keep_fractions=keeps,
                targets=d.targets,selection=selections,metadata=d.metadata,
                summary={k:all_target_summary(d.y,result['baseline'],v,d.targets) for k,v in result.items()})
    write_json(out/'comparison.json',report)
    write_json(out/'expansion_under_pruning.json',all_target_summary(d.y,result['prune_base'],result['prune_expanded'],d.targets))
    np.savez_compressed(out/'predictions.npz',predictions=np.stack(list(result.values()),1),variants=np.asarray(list(result)),y=d.y,ids=np.asarray(d.ids),targets=np.asarray(d.targets))
    print({k:{kk:vv for kk,vv in s.items() if kk!='mse_ratios'} for k,s in report['summary'].items()},flush=True)
    return report,result
