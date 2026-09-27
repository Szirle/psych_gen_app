"""Follow promising mechanisms; match the original selector/grid and perturb folds."""
from pathlib import Path
import json
import numpy as np
from dataclasses import replace
from ..schema import Dataset,write_json
from ..features import Representation
from ..evaluation import ComparisonConfig,compare
from .geometry_screen import specs


def main():
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_02';d=Dataset.load(root/'data.npz')
 cols=tuple(d.registry.columns('baseline'));base=set(cols)
 _,contrasts=specs(d);pairs=contrasts['contrast_1.0'].contrast_pairs
 role=lambda r:set(d.registry.columns(r))
 wording=set().union(*(role('simple_'+t) for t in ('smart','trustworthy','alert','well-groomed')))
 attention=role('attention_cues')
 banks={'baseline':Representation(cols),
        'wording':Representation(tuple(sorted(base|wording))),
        'wording_attention':Representation(tuple(sorted(base|wording|attention)))}
 for w in (4.,16.,64.):
  banks[f'contrasts_{w}']=Representation(cols,contrast_pairs=pairs,contrast_weight=w)
  banks[f'wording_contrasts_{w}']=Representation(tuple(sorted(base|wording)),contrast_pairs=pairs,contrast_weight=w)
 # The broad contrast gain may be carried by a small role rather than every pair.
 alertpairs=tuple((a,b) for a,b in pairs if d.registry.items[a].cue=='own_alert')
 banks['alert_contrasts_64']=Representation(cols,contrast_pairs=alertpairs,contrast_weight=64.)
 banks['wording_alert_contrasts_64']=Representation(tuple(sorted(base|wording)),contrast_pairs=alertpairs,contrast_weight=64.)
 config=ComparisonConfig(selector='existing_ensemble',alphas=tuple(np.logspace(-4,1,11)),bootstrap=500)
 for seed in (20260924,20260925):
  r,_=compare(d,banks,config=replace(config,seed=seed),output=root/f'combined/seed-{seed}')
  b=r['metrics']['baseline'];summary={k:{t:round(100*(m['mse']/b[t]['mse']-1),2) for t,m in v.items()} for k,v in r['metrics'].items()}
  write_json(root/f'combined/seed-{seed}/summary.json',summary);print(seed,json.dumps(summary),flush=True)

if __name__=='__main__':main()
