"""Separate addition from removal, then assess sensitivity to fold assignment."""
from pathlib import Path
from ..schema import Dataset, write_json
from ..features import Representation
from ..evaluation import ComparisonConfig, compare


def main():
    root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_01'
    d=Dataset.load(root/'expanded.npz')
    original=set(range(512)); direct=set(d.registry.columns('direct_impressions', closure=True))
    seen={(c.text,c.channel) for c in d.registry.items[:512]}
    extra={i for i,c in enumerate(d.registry.items) if i>=512 and (c.text,c.channel) not in seen}
    smile=set(d.registry.columns('smile_mechanics'))
    banks={'original':original,'without_direct':original-direct,'add_all':original|extra,
           'replace_direct':original-direct|extra,'add_smile':original|smile}
    variants={k:Representation(tuple(sorted(v))) for k,v in banks.items()}
    contrasts={'addition':('original','add_all'),'removal':('original','without_direct'),
               'removal_after_addition':('add_all','replace_direct'),
               'replacement':('original','replace_direct'),'smile':('original','add_smile')}
    for seed in (20260924,20260925):
        r,_=compare(d,variants,config=ComparisonConfig(seed=seed),contrasts=contrasts,output=root/f'study4/seed-{seed}')
        print(seed, {k:{t:round(m['mse'],8) for t,m in v.items()} for k,v in r['metrics'].items()}, flush=True)
    write_json(root/'study4/design.json',{'status':'Exploratory follow-up selected after study3',
        'question':'Does replacement benefit require removal, and does the sign survive new folds?',
        'seeds':[20260924,20260925], 'warning':'Same 400 development images; repeat is split sensitivity, not replication.'})

if __name__=='__main__':main()
