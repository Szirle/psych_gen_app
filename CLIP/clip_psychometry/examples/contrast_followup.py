"""Control overall kernel energy and compare old/new contrast directions."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from ..schema import Dataset,write_json
from ..features import Representation
from ..evaluation import ComparisonConfig,compare
from .geometry_screen import specs


def main():
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_02';d=Dataset.load(root/'data.npz')
 base=set(d.registry.columns('baseline'));wording=set().union(*(set(d.registry.columns('simple_'+t)) for t in ('smart','trustworthy','alert','well-groomed')))
 cols=tuple(sorted(base|wording));old=specs(d)[1]['contrast_1.0'].contrast_pairs
 new=[]
 for target in ('smart','trustworthy','alert','well-groomed'):
  for channel in ('global-short','local-box-max'):
   ix=[i for i in d.registry.columns('simple_'+target) if d.registry.items[i].channel==channel]
   new.extend((ix[a],ix[b]) for a,b in ((0,1),(2,3)))
 banks={'baseline':Representation(tuple(sorted(base))),'wording':Representation(cols)}
 for w in (4.,16.,64.):
  banks[f'old_normalized_{w}']=Representation(cols,contrast_pairs=old,contrast_weight=w,normalize_contrast_energy=True)
 banks['new_normalized_16']=Representation(cols,contrast_pairs=tuple(new),contrast_weight=16.,normalize_contrast_energy=True)
 banks['both_normalized_4']=Representation(cols,contrast_pairs=old+tuple(new),contrast_weight=4.,normalize_contrast_energy=True)
 config=ComparisonConfig(selector='existing_ensemble',alphas=tuple(np.logspace(-4,1,11)),bootstrap=500)
 for seed in (20260924,20260925):
  r,_=compare(d,banks,config=replace(config,seed=seed),output=root/f'contrast_followup/seed-{seed}')
  b=r['metrics']['baseline'];summary={k:{t:round(100*(m['mse']/b[t]['mse']-1),2) for t,m in v.items()} for k,v in r['metrics'].items()}
  write_json(root/f'contrast_followup/seed-{seed}/summary.json',summary);print(seed,json.dumps(summary),flush=True)

if __name__=='__main__':main()
