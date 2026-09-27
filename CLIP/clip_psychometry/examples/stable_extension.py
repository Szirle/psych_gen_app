"""Keep incumbent target masks fixed while testing ranked additions."""
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
from ..adapters import _repo
from ..schema import Dataset,write_json
from ..pruning import rank_paths,units,select_units,AGOPConfig,all_target_summary
from ..pruning_study import scaled_predictions,RECIPES


def main():
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_03';d=Dataset.load(root/'data.npz')
 source=root/'seed-20260926';report=json.loads((source/'comparison.json').read_text());out=root/'stable-extension';out.mkdir(exist_ok=True)
 base=tuple(d.registry.columns('baseline'));new=tuple(i for i in range(d.x.shape[1]) if i not in set(base))
 _,basegroups=units(d.registry,base);_,newgroups=units(d.registry,new)
 counts=(0,4,16,40,80);native=_repo('race_perception');config=AGOPConfig()
 result={k:np.empty_like(d.y) for k in ('incumbent','extension','extension_guard1','extension_consistent')};details=[]
 with threadpool_limits(limits=1):
  for fold,old in enumerate(report['selection']):
   train,test=np.asarray(old['train']),np.asarray(old['test']);y=d.y[train];nt=y.shape[1]
   inner=native.cv_splits(len(train),3,report['seed'],d.groups[train]);ids=np.empty(len(train),int)
   oof=np.empty((len(train),len(counts),len(RECIPES),nt))
   for f,(tr,va) in enumerate(inner):
    ids[va]=f
    ranks=rank_paths(d.x[np.ix_(train[tr],base)],y[tr],config)
    expanded=rank_paths(d.x[train[tr]],y[tr],config)
    for t in range(nt):
     selected,_=select_units(ranks[old['base'][t]['round']-1,:,t],basegroups,old['base'][t]['keep'])
     incumbent=np.asarray(base)[selected]
     importance=np.array([expanded[-1,np.asarray(new)[g],t].sum() for g in newgroups]);order=np.argsort(-importance,kind='stable')
     for j,n in enumerate(counts):
      extra=np.asarray(new)[np.concatenate([newgroups[g] for g in order[:n]])] if n else np.array([],int)
      cols=np.sort(np.r_[incumbent,extra])
      oof[va,j,:,t]=scaled_predictions(d.x[train[tr]],y[tr,t:t+1],d.x[train[va]],cols)[:,:,0]
   strategies=[];loss=[];ip=[]
   for j in range(len(counts)):
    s,_,_=native.select_strategies(oof[:,j],y,RECIPES);pred=native.apply_strategies(oof[:,j],s)
    strategies.append(s);ip.append(pred);loss.append(((pred-y)**2).mean(0))
   loss=np.asarray(loss);best=loss.argmin(0);chosen=np.empty((len(test),nt));unchanged=np.empty_like(chosen)
   with np.load(source/f'fold-{fold}.npz') as old_npz:
    masks=old_npz['base_masks'];rank=old_npz['expanded_rank']
   masks_out=np.zeros((nt,d.x.shape[1]),bool);fold_detail=[]
   for t,j in enumerate(best):
    incumbent=np.flatnonzero(masks[t]);importance=np.array([rank[-1,np.asarray(new)[g],t].sum() for g in newgroups]);order=np.argsort(-importance,kind='stable')
    for option in set((0,int(j))):
     n=counts[option];extra=np.asarray(new)[np.concatenate([newgroups[g] for g in order[:n]])] if n else np.array([],int)
     cols=np.sort(np.r_[incumbent,extra]);p=scaled_predictions(d.x[train],y[:,t:t+1],d.x[test],cols)[:,:,0]
     s=strategies[option][t];v=sum(w*p[:,i] for i,w in zip(s['indices'],s['weights']))
     if option==0:unchanged[:,t]=v
     if option==j:chosen[:,t]=v;masks_out[t,cols]=True
    fold_detail.append(dict(additions=counts[j],inner_gain=float(1-loss[j,t]/loss[0,t])))
   gain=1-loss[best,np.arange(nt)]/loss[0]
   picked=np.stack([ip[j][:,t] for t,j in enumerate(best)],axis=1)
   consistent=np.array([((picked[ids==f]-y[ids==f])**2).mean(0)<.99*((ip[0][ids==f]-y[ids==f])**2).mean(0) for f in range(3)]).all(0)
   local=dict(incumbent=unchanged,extension=chosen,extension_guard1=np.where((gain>.01)[None,:],chosen,unchanged),extension_consistent=np.where(consistent[None,:],chosen,unchanged))
   for k,v in local.items():result[k][test]=v
   np.savez_compressed(out/f'fold-{fold}.npz',test=test,masks=masks_out,**local)
   details.append(fold_detail);print(f'Stable extension fold {fold+1}/5',flush=True)
 with np.load(source/'predictions.npz') as a:original=a['predictions'][:,list(a['variants']).index('baseline')];oldpruned=a['predictions'][:,list(a['variants']).index('prune_base')]
 summary=dict(versus_incumbent={k:all_target_summary(d.y,result['incumbent'],v,d.targets) for k,v in result.items()},
   versus_original={k:all_target_summary(d.y,original,v,d.targets) for k,v in result.items()},
   incumbent_max_prediction_difference=float(np.max(np.abs(oldpruned-result['incumbent']))),selection=details,
   protocol='Exploratory stable extension. Incumbent budget selected on outer-training inner CV; outer test labels never select additions.')
 write_json(out/'comparison.json',summary)
 np.savez_compressed(out/'predictions.npz',predictions=np.stack(list(result.values()),1),variants=np.asarray(list(result)),targets=np.asarray(d.targets),y=d.y)
 print({k:{name:{kk:vv for kk,vv in s.items() if kk!='mse_ratios'} for name,s in v.items()} for k,v in summary.items() if k.startswith('versus')},flush=True)

if __name__=='__main__':main()
