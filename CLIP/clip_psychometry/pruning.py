"""Per-target diagonal AGOP ranking and phrase-bundled pruning.

Method and departure ledger: docs/METHODOLOGY.md, target-specific AGOP pruning.
No model attribution is interpreted as causal importance or unique information.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.linalg import solve


@dataclass(frozen=True)
class AGOPConfig:
    kernel: str = 'laplace'
    gamma: float = 1.
    alpha: float = .1
    rounds: int = 2
    center_gradients: bool = False
    pilot_folds: int = 2
    seed: int = 20260926


def radial(x,z,weights,gamma,kind):
    p=x.shape[1]
    distance=np.maximum((x*x*weights).sum(1)[:,None]/p+(z*z*weights).sum(1)[None,:]/p-2*(x*weights)@z.T/p,0.)
    if x is z: np.fill_diagonal(distance,0.)
    if kind=='rbf':return np.exp(-gamma*distance),distance
    if kind=='laplace':return np.exp(-gamma*np.sqrt(distance)),distance
    raise ValueError('Expected laplace or rbf.')


def fit_dual(z,y,weights,config):
    k,_=radial(z,z,weights,config.gamma,config.kernel)
    centered=k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
    a=solve(centered+config.alpha*np.eye(len(z)),y-y.mean(0),assume_a='pos',check_finite=False)
    # Query centering subtracts the mean kernel derivative times sum(a).
    return a-a.mean(0)


def gradients(z,query,dual,weights,config):
    k,d2=radial(query,z,weights,config.gamma,config.kernel)
    if config.kernel=='laplace':
        h=np.divide(k,np.sqrt(d2),out=np.zeros_like(k),where=d2>1e-12)
        factor=config.gamma/z.shape[1]
    else:
        h=k;factor=2*config.gamma/z.shape[1]
    h=h*dual[None,:]
    return -factor*(query*h.sum(1)[:,None]-h@z)*weights


def rank_paths(x,y,config=AGOPConfig(),groups=None):
    """Return [round, coordinate, target] sensitivity scores, fitted on x/y only.

    First fit shares its solve across targets. Later fits have target-specific
    diagonal metrics. No validation/test inputs enter ranking or standardization.
    """
    x,y=np.asarray(x,float),np.asarray(y,float)
    if y.ndim==1:y=y[:,None]
    if x.ndim!=2 or y.shape[0]!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned arrays required; split missing-target masks before ranking.')
    if config.rounds<1 or config.alpha<=0 or config.gamma<=0:raise ValueError('Positive ranking parameters required.')
    if config.kernel == 'frozen':
        if config.rounds != 1:raise ValueError('Frozen-readout attribution is a one-step comparator, not diagonal RFM recursion.')
        return frozen_readout_agop(x,y,groups=groups,center_gradients=config.center_gradients, pilot_folds=config.pilot_folds, seed=config.seed)[None,:,:]
    std=x.std(0);z=(x-x.mean(0))/np.where(std>1e-12,std,1.)
    weights=np.ones(x.shape[1]);dual=fit_dual(z,y,weights,config)
    result=np.empty((config.rounds,x.shape[1],y.shape[1]))
    for t in range(y.shape[1]):
        w=weights.copy();a=dual[:,t]
        for step in range(config.rounds):
            if step:a=fit_dual(z,y[:,t],w,config)
            g=gradients(z,z,a,w,config)
            if config.center_gradients:g-=g.mean(0)
            s=np.mean(g*g,axis=0)
            result[step,:,t]=s
            w=s/max(float(s.mean()),1e-30) if s.sum()>1e-30 else np.ones_like(s)
    return result


def frozen_readout_agop(x,y,groups=None,center_gradients=False,pilot_folds=2,seed=20260926):
    """AGOP of the actual kernel ensemble, tuned strictly within supplied rows.

    Two pilot folds choose its recipe; caller's validation rows are never supplied.
    Uses standardized-coordinate gradients and includes ensemble cross terms by
    adding component gradients before taking their squared average.
    """
    from .adapters import _repo
    native=_repo('race_perception')
    recipes=[r for r in native.candidate_grid() if r['bank']=='both' and not r.get('centered')]
    groups=np.arange(len(x)).astype(str) if groups is None else groups
    inner,_=native.inner_predictions(x,y,recipes,pilot_folds,seed,groups)
    strategies,_,_=native.select_strategies(inner,y,recipes)
    fitted=native.fit_bundle_readout(x,y,recipes,strategies)
    scores=np.empty((x.shape[1],y.shape[1]))
    for t,strategy in enumerate(strategies):
        gradient=np.zeros_like(x)
        for index,weight in zip(strategy['indices'],strategy['weights']):
            c=fitted['components'][native.base_key(recipes[index])]
            z=c['state']['z'];a=c['coefficients'][index][:,t];a=a-a.mean();p=z.shape[1]
            family=recipes[index]['family']
            if family=='linear':g=np.broadcast_to(a@z/p,z.shape)
            elif family=='poly2':g=(2/p)*(((1+z@z.T/p)*a[None,:])@z)
            else:
                k=native.kernel(z,z,recipes[index]);h=k*a[None,:]
                g=-(2*recipes[index]['gamma']/p)*(z*h.sum(1)[:,None]-h@z)
            gradient+=weight*g
        if center_gradients:gradient-=gradient.mean(0)
        scores[:,t]=np.mean(gradient*gradient,axis=0)
    return scores


def units(registry,columns,by='phrase'):
    """Both channels are indivisible. Role grouping is an optional coarser unit."""
    if by not in ('phrase','role'):raise ValueError('Unknown pruning unit.')
    groups={}
    for local,c in enumerate(columns):
        item=registry.items[c];key=item.text if by=='phrase' else item.cue
        groups.setdefault(key,[]).append(local)
    return tuple(groups),tuple(np.asarray(v,int) for v in groups.values())


def select_units(scores,groups,keep):
    if not 0<keep<=1:raise ValueError('Keep fraction must lie in (0,1].')
    importance=np.array([np.sum(scores[g]) for g in groups])
    order=np.argsort(-importance,kind='stable')
    chosen=order[:max(1,int(np.ceil(keep*len(groups))))]
    return np.sort(np.concatenate([groups[i] for i in chosen])),importance


def all_target_summary(y,baseline,candidate,names):
    ratios={}
    for t,name in enumerate(names):
        good=np.isfinite(y[:,t]) & np.isfinite(baseline[:,t]) & np.isfinite(candidate[:,t])
        if not good.any():continue
        ref=np.mean((y[good,t]-baseline[good,t])**2)
        ratios[name]=float(np.mean((y[good,t]-candidate[good,t])**2)/ref)
    a=np.array(list(ratios.values()))
    return dict(targets=len(a),macro_mse_ratio=float(a.mean()),worst_mse_ratio=float(a.max()),
                improved=int((a<1-1e-10).sum()),worsened=int((a>1+1e-10).sum()),
                worsened_over_1pct=int((a>1.01).sum()),observed_all_targets_nondegraded=bool((a<=1+1e-10).all()),
                mse_ratios=ratios)


def fit_phrase_pruner(data,target,*,columns=None,keep=.75,config=AGOPConfig(),unit='phrase'):
    """Fit a portable mask for one target. `keep` must be selected externally in CV.

    Missing labels are removed only for this target. Return original column IDs
    and text identities so a caller can remap another cache without positional guesses.
    This function does not claim validation accuracy or choose keep on training error.
    """
    t=data.targets.index(target)
    columns=tuple(range(data.x.shape[1])) if columns is None else tuple(columns)
    rows=np.flatnonzero(np.isfinite(data.y[:,t]))
    if len(rows)<4:raise ValueError('Insufficient labeled images for target.')
    paths=rank_paths(data.x[np.ix_(rows,columns)],data.y[rows,t],config,groups=data.groups[rows])
    names,groups=units(data.registry,columns,unit)
    selected,importance=select_units(paths[-1,:,0],groups,keep)
    return dict(target=target,keep=keep,unit=unit,config=asdict(config),training_rows=rows.tolist(),
                columns=[columns[i] for i in selected],
                selected_items=[asdict(data.registry.items[columns[i]]) for i in selected],
                unit_importance=dict(zip(names,importance.tolist())),
                status='Training-fit sensitivity and mask; retention requires separate validation.')
