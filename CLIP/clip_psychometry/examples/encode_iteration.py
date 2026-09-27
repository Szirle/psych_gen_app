"""Encode only new text; reuse manifest-checked image feature archives."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch

from ...fgclip2_core import DEFAULT_CACHE_DIR
from ..text_encoder import FrozenTextEncoder
from ...fgclip2_face_impressions import HumanRatingsStore
from ..schema import Column, Dataset, write_json
from ..adapters import extend_dataset


PHRASES = {
    'eyewear_explicit': [
        'a face wearing clear eyeglasses', 'a face wearing dark sunglasses',
        'a person wearing prescription eyeglasses', 'a person wearing sunglasses',
        'a person without any eyewear', 'a face with no glasses',
        'a smiling person wearing clear eyeglasses', 'a smiling person wearing sunglasses',
        'a person with a neutral expression wearing eyeglasses',
        'a person with a broad smile and no glasses'],
    'smile_mechanics': [
        'a smiling person with raised cheeks and wrinkles around the eyes',
        'a smiling person with relaxed eyes',
        'a person with a broad smile and visible teeth',
        'a person smiling with closed lips',
        'a person with a smiling mouth and tense eyes',
        'a person with pressed lips and lowered eyebrows',
        'a person with a one-sided smile and a raised upper lip',
        'a person with a relaxed mouth and no smile'],
    'photo_conditions': [
        'a face lit by warm orange light', 'a face lit by cool blue light',
        'a face with uneven lighting and deep shadows around the eyes',
        'a face photographed in soft even light',
        'a face with unnatural distorted teeth', 'a face with asymmetrical distorted eyes',
        'a blurry low contrast face photograph', 'a sharp detailed face photograph'],
    'presentation_geometry': [
        'a woman with a strong square jaw', 'a woman with a narrow rounded jaw',
        'a man with a strong square jaw', 'a man with a narrow rounded jaw',
        'a face with a strong square jaw', 'a face with a narrow rounded jaw',
        'a woman', 'a man'],
}


def main():
    repo = Path(__file__).resolve().parents[2]
    root = repo/'research/clip_psychometry/iteration_01'
    data = Dataset.load(repo/'research/clip_psychometry/development/data.npz')
    write_json(root/'candidate_design.json', dict(phrases=PHRASES,
        purpose='Small concrete-cue alternatives; no target-driven wording search during encoding.'))
    phrases = [p for group in PHRASES.values() for p in group]
    engine = FrozenTextEncoder(model_id='qihoo360/fg-clip2-so400m', device='mps',
                     revision='d57d30fe94a107dd6a2610eb4e9a135004d823a4', local_files_only=True)
    text = torch.cat([engine.encode_text(phrases[i:i+16], mode='short').float().cpu()
                      for i in range(0, len(phrases), 16)])
    scale, bias = engine.model.logit_scale.float().exp().item(), engine.model.logit_bias.float().item()
    blocks, columns, provenance, control_errors = [], [], [], {}
    for channel, key in [('p128', '5428a4e3820ff3b54fb7317c'), ('p256', 'a52bac38180b0e0ef16d3b5c')]:
        path = DEFAULT_CACHE_DIR/'dataset-rankings'/f'features-{key}.npz'
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            if meta['model'] != engine.model_id or meta['revision'] != engine.revision:
                raise ValueError('Encoder and image archive revisions differ.')
            names = archive['paths'].tolist()
            indices = [names.index(i) for i in data.ids]
            z = torch.from_numpy(archive['features'][indices].copy()).float()
        with torch.inference_mode():
            blocks.append((z@text.T*scale+bias).numpy())
        for j, phrase in enumerate(phrases):
            old = [k for k,c in enumerate(data.registry.items) if c.text == phrase and c.channel == channel]
            if old:
                control_errors[channel+':'+phrase] = float(np.max(np.abs(blocks[-1][:, j]-data.x[:, old[0]])))
        for cue, group in PHRASES.items():
            for p in group:
                columns.append(Column(f'iteration01:{channel}:{phrases.index(p)}', p, cue, channel, (cue,)))
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               model=meta['model'], revision=meta['revision'], dtype=meta['dtype'], channel=channel))
    expanded = extend_dataset(data, np.column_stack(blocks), columns,
                              provenance=dict(image_archives=provenance, new_texts=len(phrases), newly_encoded_images=0))
    ratings = HumanRatingsStore(repo.parent/'data/dim_to_photo_to_ratings.pkl')
    targets = ('trustworthy', 'happy', 'smart')
    vectors = [ratings.means(t, [Path(p) for p in data.paths], min_ratings=2, reject_nonfinite=True) for t in targets]
    expanded.y = np.column_stack([v.means for v in vectors])
    expanded.se = np.column_stack([v.mean_se for v in vectors])
    expanded.targets = targets
    expanded.save(root/'expanded.npz')
    write_json(root/'encoding.json', dict(new_texts=len(phrases), newly_encoded_images=0,
         model=engine.model_id, revision=engine.revision, dtype=str(engine.dtype),
         image_archives=provenance, logit_scale=scale, logit_bias=bias, targets=targets,
         text_weight_gib=engine.loaded_weight_bytes/1024**3, cache_overlap_max_errors=control_errors))
    print(f'Encoded {len(phrases)} new phrase entries against cached development images; wrote {root}/expanded.npz', flush=True)


if __name__ == '__main__':
    main()
