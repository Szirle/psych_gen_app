"""Exact prediction-level fusion; one trusted-local fold checkpoint in RAM at a time."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .agop import predict_scores


def predict_ensemble(packet, scores, directory):
    members = packet['members']
    if not members or not np.isclose(sum(m['weight'] for m in members),1.):
        raise ValueError('Ensemble weights must sum to one.')
    result=np.zeros((len(scores),len(packet['targets'])))
    for member in members:
        if not np.isfinite(member['weight']) or member['weight'] < 0:
            raise ValueError('Invalid ensemble weight.')
        path=Path(directory)/member['path']
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        if h.hexdigest()!=member['sha256']: raise ValueError(f'Changed checkpoint: {path}')
        fitted=torch.load(path,map_location='cpu',weights_only=False)
        if fitted['format']!='fgclip2_multitarget_agop_weighted_v1':
            raise ValueError('Expected a B2 fold model.')
        for key in ('targets','phrases','model','revision','run_identity'):
            if fitted[key]!=packet[key]: raise ValueError(f'Ensemble member differs in {key}.')
        for key in ('precision','patches','text_chunk'):
            if fitted['config'][key]!=packet['config'][key]: raise ValueError(f'Encoder config mismatch: {key}')
        # Bound query-by-support matrices on large downstream image sets.
        for start in range(0,len(scores),256):
            result[start:start+256]+=member['weight']*predict_scores(fitted,scores[start:start+256])
        del fitted
    return result
