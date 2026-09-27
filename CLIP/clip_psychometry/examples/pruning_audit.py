"""Export all-target phrase sensitivity and cross-fold mask stability."""
from pathlib import Path
import json,sys
import numpy as np
from ..schema import Dataset,write_json
from ..pruning import units,all_target_summary


def main():
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_03';d=Dataset.load(root/'data.npz')
 seed=int(sys.argv[1]) if len(sys.argv)>1 else 20260926;prefix=(sys.argv[2]+'-') if len(sys.argv)>2 and sys.argv[2] in ('mild','frozen') else '';out=root/f'{prefix}seed-{seed}'
 with np.load(out/'predictions.npz') as prediction_file:
  variants=list(prediction_file['variants']);pred=prediction_file['predictions']
  write_json(out/'expansion_under_pruning.json',all_target_summary(d.y,pred[:,variants.index('prune_base')],pred[:,variants.index('prune_expanded')],d.targets))
 report=json.loads((out/'comparison.json').read_text());cols=tuple(range(d.x.shape[1]));names,groups=units(d.registry,cols)
 nt=len(d.targets);nb=len(names);scores=[];counts={k:np.zeros((nt,d.x.shape[1])) for k in report['summary']}
 base=np.zeros((nt,d.x.shape[1]),bool);base[:,d.registry.columns('baseline')]=True
 fold_widths={k:[] for k in counts}
 for f,detail in enumerate(report['selection']):
  with np.load(out/f'fold-{f}.npz') as a:
   rank=a['expanded_rank'][-1];importance=np.stack([rank[g].sum(0) for g in groups]);importance/=np.maximum(importance.sum(0),1e-30)
   scores.append(importance)
   masks={'baseline':base,'expanded':np.ones_like(base),'prune_base':a['base_masks'],'prune_expanded':a['expanded_masks']}
   selected=np.where(np.asarray(detail['choose_expanded'])[:,None],a['expanded_masks'],a['base_masks'])
   for k,take in detail['accepted'].items():masks[k]=np.where(np.asarray(take)[:,None],selected,base)
   for k,v in masks.items():counts[k]+=v;fold_widths[k].append(v.sum(1)//2)
 mean=np.mean(scores,0);std=np.std(scores,0)
 compression={k:dict(mean_retained_phrases=float(np.mean(fold_widths[k])),
    median_retained_phrases=float(np.median(fold_widths[k])),minimum_retained=int(np.min(fold_widths[k])),
    maximum_retained=int(np.max(fold_widths[k])),union_across_targets_and_folds=sum(bool(v[:,g].any()) for g in groups),
    globally_unused_in_all_folds=[names[j] for j,g in enumerate(groups) if not v[:,g].any()]) for k,v in counts.items()}
 rows=[]
 for j,(name,g) in enumerate(zip(names,groups)):
  rows.append(dict(text=name,cue=d.registry.items[g[0]].cue,
     agop_share_mean={t:float(mean[j,i]) for i,t in enumerate(d.targets)},
     agop_share_sd={t:float(std[j,i]) for i,t in enumerate(d.targets)},
     selected_fold_fraction={k:{t:float(v[i,g[0]]/5) for i,t in enumerate(d.targets)} for k,v in counts.items() if k not in ('baseline','expanded')}))
 write_json(out/'importance.json',dict(targets=d.targets,unit='phrase, both channels together',
    note='Training-fold normalized AGOP sensitivity, not causal or unique importance. Selection frequencies are stability diagnostics.',phrases=rows))
 write_json(out/'compression.json',compression)
 print({k:{kk:vv for kk,vv in v.items() if kk!='globally_unused_in_all_folds'} for k,v in compression.items()})

if __name__=='__main__':main()
