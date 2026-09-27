"""Join every available rating target to the existing development feature cache."""
from pathlib import Path
import numpy as np
from ...fgclip2_face_impressions import HumanRatingsStore
from ..schema import Dataset,write_json


def main():
 root=Path(__file__).resolve().parents[2];d=Dataset.load(root/'research/clip_psychometry/iteration_02/data.npz')
 store=HumanRatingsStore(root.parent/'data/dim_to_photo_to_ratings.pkl')
 targets=store.variables;values=[store.means(t,[Path(p) for p in d.paths],min_ratings=2,reject_nonfinite=True) for t in targets]
 d.targets=targets;d.y=np.column_stack([v.means for v in values]);d.se=np.column_stack([v.mean_se for v in values])
 d.metadata.update(all_targets=True,ratings_path=str(store.path))
 output=root/'research/clip_psychometry/iteration_03';d.save(output/'data.npz')
 write_json(output/'target_coverage.json',{t:int(np.isfinite(d.y[:,i]).sum()) for i,t in enumerate(targets)})
 print(dict(rows=len(d.x),targets=len(targets),coverage={t:int(np.isfinite(d.y[:,i]).sum()) for i,t in enumerate(targets)}))

if __name__=='__main__':main()
