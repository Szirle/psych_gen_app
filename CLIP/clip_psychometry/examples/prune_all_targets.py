"""python -m CLIP.clip_psychometry.examples.prune_all_targets [seed]."""
from pathlib import Path
import sys
from ..schema import Dataset
from ..pruning_study import run

if __name__=='__main__':
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_03'
 seed=int(sys.argv[1]) if len(sys.argv)>1 else 20260926
 run(Dataset.load(root/'data.npz'),root/f'seed-{seed}',seed=seed)
