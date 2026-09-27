"""Addition-first phrase screens with the exact existing frozen-kernel selector."""
from pathlib import Path
import json
from ..schema import Dataset,write_json
from ..features import Representation
from ..evaluation import ComparisonConfig,compare


def main():
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_02'
 d=Dataset.load(root/'data.npz');design=json.loads((root/'design.json').read_text())
 base=set(d.registry.columns('baseline'));banks={'baseline':base};detectors={}
 for target in ('smart','trustworthy','alert','well-groomed'):
  role='simple_'+target;old=set(d.registry.columns('own_'+target));allnew=set(d.registry.columns(role))
  texts=set(design['candidates'][role][:4]);matched={i for i in allnew if d.registry.items[i].text in texts}
  banks['add_matched_'+target]=base|matched
  banks['replace_matched_'+target]=base-old|matched
  banks['add_expanded_'+target]=base|allnew
  detectors['old_'+target]=old;detectors['matched_'+target]=matched;detectors['expanded_'+target]=allnew
 for target in ('smart','trustworthy','alert'):
  new=set(d.registry.columns('conditional_'+target));old=set(d.registry.columns('own_'+target))
  banks['add_conditional_'+target]=base|new;banks['replace_conditional_'+target]=base-old|new
  detectors['conditional_'+target]=new
 for role in ('attention_cues','surface_presentation'):
  banks['add_'+role]=base|set(d.registry.columns(role))
 config=ComparisonConfig(backend='legacy',bootstrap=500)
 for label,sets in [('phrases',banks),('alignment',detectors)]:
  r,_=compare(d,{k:Representation(tuple(sorted(v))) for k,v in sets.items()},config=config,output=root/label)
  if label=='phrases':
   b=r['metrics']['baseline']
   summary={k:{t:round(100*(m['mse']/b[t]['mse']-1),2) for t,m in v.items()} for k,v in r['metrics'].items()}
  else:
   summary={k:{t:round(m['pearson'],3) for t,m in v.items() if k.endswith(t)} for k,v in r['metrics'].items()}
  write_json(root/label/'summary.json',summary);print(label,json.dumps(summary),flush=True)

if __name__=='__main__':main()
