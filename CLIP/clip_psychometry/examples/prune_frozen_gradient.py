"""Check whether using the actual selected readout improves phrase attribution."""
from pathlib import Path
from ..schema import Dataset
from ..pruning import AGOPConfig
from ..pruning_study import run

if __name__=='__main__':
 root=Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_03'
 run(Dataset.load(root/'data.npz'),root/'frozen-seed-20260926',config=AGOPConfig(kernel='frozen',rounds=1))
