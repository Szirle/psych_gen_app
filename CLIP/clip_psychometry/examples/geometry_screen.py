"""Specific metric and contrast hypotheses, preserving the complete current bank."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from ..schema import Dataset,write_json
from ..features import Representation
from ..evaluation import ComparisonConfig,compare


def specs(d):
 cols=tuple(d.registry.columns('baseline'));items=[d.registry.items[i] for i in cols]
 categoric=lambda c:c.cue.startswith('race_perceived_') or c.cue in ('race_regional_wordings','race_ambiguous_appearance','race_skin_tone','race_skin_lightness_variants')
 abstract=lambda c:c.cue.startswith('own_') and c.cue[4:] not in ('age','gender','weight','skin-color','hair-color','long-haired','alert','happy','well-groomed')
 variants={'uniform':Representation(cols,separate_metric=True)}
 for strength in (.05,.25):
  variants[f'category_weight_{strength}']=Representation(cols,metric_weights=tuple(strength if categoric(c) else 1. for c in items),separate_metric=True)
  variants[f'abstract_local_weight_{strength}']=Representation(cols,metric_weights=tuple(strength if c.channel=='local-box-max' and (categoric(c) or abstract(c)) else 1. for c in items),separate_metric=True)
 for channel in ('global-short','local-box-max'):
  variants['downweight_'+channel]=Representation(cols,metric_weights=tuple(.25 if c.channel==channel else 1. for c in items),separate_metric=True)
 pairs=[]
 for target in ('smart','trustworthy','alert','well-groomed','happy'):
  for channel in ('global-short','local-box-max'):
   chosen=[i for i in d.registry.columns('own_'+target) if d.registry.items[i].channel==channel]
   # Four descriptors, each with two templates: contrast descriptor 0/1 and 2/3.
   pairs.extend((chosen[a],chosen[b]) for a,b in ((0,2),(1,3),(4,6),(5,7)))
 contrasts={'uniform':Representation(cols)}
 for weight in (1.,4.,16.):contrasts[f'contrast_{weight}']=Representation(cols,contrast_pairs=tuple(pairs),contrast_weight=weight)
 return variants,contrasts


def main():
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_02';d=Dataset.load(root/'data.npz')
 variants,contrasts=specs(d)
 for label,banks,kernels in [('metric',variants,tuple(('split_product',g) for g in (.03,.1,.3,1.))),
                              ('contrasts',contrasts,(('linear',1.),('poly2',1.),('rbf',.03),('rbf',.1),('rbf',1.)))]:
  r,_=compare(d,banks,config=ComparisonConfig(kernels=kernels,alphas=(.001,.01,.1,1.,10.),bootstrap=500),output=root/label)
  b=r['metrics']['uniform'];summary={k:{t:round(100*(m['mse']/b[t]['mse']-1),2) for t,m in v.items()} for k,v in r['metrics'].items()}
  write_json(root/label/'summary.json',summary);print(label,json.dumps(summary),flush=True)

if __name__=='__main__':main()
