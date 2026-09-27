"""Streaming frozen score cache: identical half-cache arithmetic, bounded RAM."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..fgclip2_face_impressions import array_hash, _atomic_save_npz
from .assets import atomic_save, image_inputs, stage_encoder
from .models import visual_features


class FrozenScoreAssets:
    @torch.no_grad()
    def __init__(self, engine, data, bank, args):
        self.engine, self.data, self.bank, self.args = engine, data, bank, args
        cache = Path(args.cache)
        cache.mkdir(parents=True, exist_ok=True)
        identity = np.asarray([engine.model_id, engine.revision, str(engine.dtype), str(args.patches),
            Path(__file__).read_text(), Path(__file__).with_name('models.py').read_text(),
            Path(__file__).with_name('assets.py').read_text()])
        self.key = array_hash(np.asarray(data.image_hashes), np.asarray([p.name for p in data.paths]), identity)
        text_key = array_hash(identity, np.asarray(bank['phrases']))
        text_path = cache/f'frozen-text-{text_key}.pt'
        if text_path.exists():
            self.text = torch.load(text_path, map_location='cpu', weights_only=True)
        else:
            stage_encoder(engine, 'text')
            self.text = tuple(torch.cat([engine.encode_text(bank['phrases'][i:i+args.text_chunk], mode=mode).float().cpu()
                for i in range(0, len(bank['phrases']), args.text_chunk)]) for mode in ('short', 'box'))
            atomic_save(self.text, text_path)
        path = cache/f'frozen-scores-{self.key}-{text_key}.npz'
        self.score_path = path
        if path.exists():
            with np.load(path) as f:
                self.scores = f['scores']
            return
        partial = path.with_suffix('.partial.npz')
        self.scores = np.empty((len(data.paths), 2*len(bank['phrases'])), np.float32)
        start = 0
        if partial.exists():
            with np.load(partial) as f:
                start = int(f['completed'])
                self.scores[:start] = f['scores']
        stage_encoder(engine, 'vision')
        box = self.text[1].to(engine.device)
        print(f'Streaming p{args.patches} scores: {start}/{len(data.paths)} images', flush=True)
        for begin in range(start, len(data.paths), args.encode_batch):
            end = min(begin+args.encode_batch, len(data.paths))
            z, dense, mask, _ = visual_features(engine, image_inputs(engine, data.paths[begin:end], args.patches))
            # Match Assets: CPU global dot from half z; normalized half dense on device.
            glob = z.half().float().cpu().numpy() @ self.text[0].numpy().T
            local = (F.normalize(dense.half().float(), dim=-1) @ box.T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values
            self.scores[begin:end] = np.concatenate((glob, local.cpu().numpy()), axis=1)
            if end//50 != begin//50 or end == len(data.paths):
                _atomic_save_npz(partial, scores=self.scores[:end], completed=np.asarray(end))
                print(f'Scored {end}/{len(data.paths)} images', flush=True)
        _atomic_save_npz(path, scores=self.scores, paths=np.asarray([str(p) for p in data.paths]),
                         phrases=np.asarray(bank['phrases']))
        partial.unlink(missing_ok=True)
