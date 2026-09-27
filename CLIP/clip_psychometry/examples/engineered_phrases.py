"""Current-bank interventions, with the existing global/local Assets pipeline."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import torch
from ...fgclip2_multitarget.phrases import phrase_bank, DESCRIPTORS
from ...race_perception.phrases import build_phrase_bank
from ...fgclip2_multitarget.assets import Assets
from ...fgclip2_core import FGCLIP2
from ...fgclip2_face_impressions import HumanRatingsStore
from ..schema import Column, Registry, Dataset, write_json

from ..phrase_candidates import CANDIDATES


def main():
 repo=Path(__file__).resolve().parents[2];out=repo/'research/clip_psychometry/iteration_02'
 old=Dataset.load(repo/'research/clip_psychometry/iteration_01/expanded.npz')
 bank=phrase_bank(tuple(DESCRIPTORS));race=build_phrase_bank()
 roles={p:k for k,v in bank['own'].items() for p in v}
 race_roles={race['phrases'][i]:k for k,v in race['groups'].items() for i in v} if isinstance(race.get('groups'),dict) else {}
 base=bank['phrases'];phrases=list(dict.fromkeys(base+[p for v in CANDIDATES.values() for p in v]))
 write_json(out/'design.json',dict(baseline=bank,candidates=CANDIDATES,phrases=phrases,
   targets=['trustworthy','smart','alert','well-groomed','happy'],
   purpose='Addition-first current-bank interventions; matched rewording and cue expansion separated.'))
 data=SimpleNamespace(paths=tuple(Path(p) for p in old.paths),image_hashes=tuple(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old.paths))
 args=SimpleNamespace(cache=repo/'fgclip2_multitarget/.cache',patches=576,encode_batch=1,text_chunk=16,batch_size=32)
 engine=FGCLIP2(model_id='qihoo360/fg-clip2-so400m',device='mps',revision='d57d30fe94a107dd6a2610eb4e9a135004d823a4',local_files_only=True)
 # Stage frozen encoders to fit MPS without altering numerical paths or limits.
 for name in ('text_model','longtext_head','boxtext_head'):
  getattr(engine.model,name).to('cpu')
 torch.mps.empty_cache()
 original_encode=engine.encode_text
 staged=False
 def staged_text(*a,**kw):
  nonlocal staged
  if not staged:
   engine.model.vision_model.to('cpu');engine.model.dense_feature_head.to('cpu')
   torch.mps.empty_cache()
   for name in ('text_model','longtext_head','boxtext_head'):
    getattr(engine.model,name).to(engine.device)
   staged=True
  return original_encode(*a,**kw)
 engine.encode_text=staged_text
 assets=Assets(engine,data,dict(bank,phrases=phrases),args)
 cols=[]
 for channel in ('global-short','local-box-max'):
  for i,p in enumerate(phrases):
   role='own_'+roles[p] if p in roles else 'race_'+race_roles.get(p,'shared') if p in race['phrases'] else 'shared'
   rs=tuple(k for k,v in CANDIDATES.items() if p in v)
   cols.append(Column(f'iteration02:{channel}:{i}',p,role if p in base else rs[0],channel,rs+(("baseline",) if p in base else ())))
 ratings=HumanRatingsStore(repo.parent/'data/dim_to_photo_to_ratings.pkl');targets=('trustworthy','smart','alert','well-groomed','happy')
 ys=[ratings.means(t,data.paths,min_ratings=2,reject_nonfinite=True) for t in targets]
 d=Dataset(assets.scores,np.column_stack([y.means for y in ys]),targets,old.paths,old.groups,Registry(tuple(cols)),old.ids,
   dict(exposure='development',baseline='fgclip2_multitarget.phrases.phrase_bank(all DESCRIPTORS)',channels='global-short/local-box-max',patches=576,visual_cache_key=assets.key,evaluation_excluded=604),np.column_stack([y.mean_se for y in ys]))
 d.save(out/'data.npz')
 write_json(out/'encoding.json',dict(baseline_phrases=len(base),total_phrases=len(phrases),model=engine.model_id,revision=engine.revision,cache_key=assets.key))
 print(f'{len(base)} baseline phrases, {len(phrases)-len(base)} additions; 400 images, five targets.',flush=True)

if __name__=='__main__':main()
