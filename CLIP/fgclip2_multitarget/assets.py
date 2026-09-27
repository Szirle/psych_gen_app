"""Label-independent caches and conservative online image augmentation."""
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from ..fgclip2_face_impressions import array_hash, _atomic_save_npz
from .models import visual_features


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def image_inputs(engine, paths, patches, augment=False):
    images = []
    for path in paths:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            if augment and random.random() < .5:
                image = ImageOps.mirror(image)
            # No color jitter, grayscale, erasing, MixUp, or context-removing crop.
            images.append(image.copy())
    inputs = engine.prepare_images(images, max_num_patches=patches)
    return inputs


def stage_encoder(engine, mode):
    """Swap frozen MPS encoder residency without altering precision or outputs."""
    if engine.device.type != 'mps':
        return
    text = ('text_model', 'longtext_head', 'boxtext_head')
    vision = ('vision_model', 'dense_feature_head')
    active, inactive = (text, vision) if mode == 'text' else (vision, text)
    for name in inactive:
        getattr(engine.model, name).to('cpu')
    torch.mps.empty_cache()
    for name in active:
        getattr(engine.model, name).to(engine.device)


class Assets:
    def __init__(self, engine, data, bank, args):
        self.engine, self.data, self.args, self.bank = engine, data, args, bank
        cache = Path(args.cache)
        cache.mkdir(parents=True, exist_ok=True)
        identity = np.asarray([engine.model_id, engine.revision, str(engine.dtype), str(args.patches),
                               Path(__file__).read_text(), Path(__file__).with_name('models.py').read_text()])
        self.key = array_hash(np.asarray(data.image_hashes), np.asarray([p.name for p in data.paths]), identity)
        path = cache/f'visual-{self.key}.pt'
        if path.exists():
            self.features = torch.load(path, map_location='cpu', weights_only=True)
        else:
            if getattr(args, 'stage_frozen_encoders', False): stage_encoder(engine, 'vision')
            print(f'Encoding {len(data.paths)} images at p{args.patches} (global + dense)…', flush=True)
            batches = []
            with torch.no_grad():
                for start in range(0, len(data.paths), args.encode_batch):
                    inputs = image_inputs(engine, data.paths[start:start+args.encode_batch], args.patches)
                    output = visual_features(engine, inputs)
                    if start % 100 == 0: print(f'Encoded {start+len(output[0])}/{len(data.paths)} images', flush=True)
                    batches.append(tuple(t.detach().cpu() if t.dtype == torch.bool else t.detach().half().cpu() for t in output))
            self.features = tuple(torch.cat([batch[k] for batch in batches]) for k in range(4))
            atomic_save(self.features, path)
        text_path = cache/f'text-{array_hash(identity, np.asarray(bank["phrases"]))}.pt'
        if text_path.exists():
            self.text = torch.load(text_path, map_location='cpu', weights_only=True)
        else:
            if getattr(args, 'stage_frozen_encoders', False): stage_encoder(engine, 'text')
            with torch.no_grad():
                self.text = tuple(torch.cat([engine.encode_text(bank['phrases'][i:i+args.text_chunk], mode=mode).float().cpu()
                                              for i in range(0, len(bank['phrases']), args.text_chunk)])
                                  for mode in ('short', 'box'))
            atomic_save(self.text, text_path)
        scores_path = cache/f'scores-{self.key}-{array_hash(np.asarray(bank["phrases"]))}.npz'
        if scores_path.exists():
            with np.load(scores_path) as saved:
                self.scores = saved['scores']
        else:
            local, box_text = [], self.text[1].to(engine.device)
            with torch.no_grad():
                for start in range(0, len(data.paths), args.encode_batch):
                    z, dense, mask, coordinates = self.batch(np.arange(start, min(start+args.encode_batch, len(data.paths))))
                    local.append((dense@box_text.T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values.cpu().numpy())
            global_scores = self.features[0].float().numpy() @ self.text[0].numpy().T
            self.scores = np.concatenate((global_scores, np.concatenate(local)), axis=1)
            _atomic_save_npz(scores_path, scores=self.scores)

    def batch(self, rows, live=False, augment=False):
        if live:
            inputs = image_inputs(self.engine, [self.data.paths[i] for i in rows], self.args.patches, augment)
            return visual_features(self.engine, inputs)
        z, dense, mask, coordinates = (value[rows].to(self.engine.device) for value in self.features)
        return F.normalize(z.float(), dim=-1), F.normalize(dense.float(), dim=-1), mask, coordinates.float()

    def initial(self, selection):
        ids = selection['bank_indices']
        return tuple(value[ids].to(self.engine.device) for value in self.text)

    @torch.no_grad()
    def statistics(self, rows, text):
        chunks = []
        for start in range(0, len(rows), self.args.batch_size):
            z, dense, mask, coordinates = self.batch(rows[start:start+self.args.batch_size])
            local = (dense @ text[1].T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values
            chunks.append(torch.stack((z @ text[0].T, local), -1).cpu())
        scores = torch.cat(chunks)
        return scores.mean(0).to(self.engine.device), scores.std(0).to(self.engine.device)
