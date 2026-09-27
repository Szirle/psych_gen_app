"""Inference from trusted-local research checkpoints, with recorded phrase order."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..fgclip2_core import FGCLIP2, DEFAULT_CACHE_DIR
from ..fgclip2_adaptation_legacy.models import BackboneSession
from ..fgclip2_face_impressions import FaceDatasetConfig, discover_face_images
from ..race_perception import predict_readout
from .assets import image_inputs, stage_encoder
from .models import AnchoredTokens, RatingDAG, visual_features
from .train import restore


@torch.no_grad()
def predict_checkpoint(checkpoint, paths, *, device='cuda', model_cache=DEFAULT_CACHE_DIR, batch_size=16):
    if not paths or batch_size < 1:
        raise ValueError('Provide images and a positive batch size')
    packet = (json.loads(Path(checkpoint).read_text()) if Path(checkpoint).suffix == '.json'
              else torch.load(checkpoint, map_location='cpu', weights_only=False))
    if packet.get('format') not in ('fgclip2_multitarget_v2', 'fgclip2_multitarget_kernel_v1', 'fgclip2_multitarget_agop_v1', 'fgclip2_multitarget_agop_weighted_v1', 'fgclip2_multitarget_agop_ensemble_v1'):
        raise ValueError('Unsupported checkpoint format; neural inference requires a joint-v2 model.')
    config, targets = packet['config'], packet['targets']
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[config['precision']]
    if device == 'cpu':
        dtype = torch.float32
    frozen = packet['format'] in ('fgclip2_multitarget_kernel_v1', 'fgclip2_multitarget_agop_v1', 'fgclip2_multitarget_agop_weighted_v1', 'fgclip2_multitarget_agop_ensemble_v1')
    engine = FGCLIP2(packet['model'], revision=packet['revision'], device=device, dtype=dtype,
                     cache_dir=model_cache, local_files_only=True,
                     memory_reserve_gb=.5 if frozen and str(device).startswith('mps') else 1.25)
    frozen = packet['format'] in ('fgclip2_multitarget_kernel_v1', 'fgclip2_multitarget_agop_v1', 'fgclip2_multitarget_agop_weighted_v1', 'fgclip2_multitarget_agop_ensemble_v1')
    if frozen: stage_encoder(engine, 'text')
    phrases = packet.get('phrases', packet.get('selection', {}).get('phrases'))
    text = tuple(torch.cat([engine.encode_text(phrases[i:i+config['text_chunk']], mode=mode).float()
                            for i in range(0, len(phrases), config['text_chunk'])]) for mode in ('short', 'box'))
    if frozen: stage_encoder(engine, 'vision')
    if packet.get('variant', '').startswith('kernel-'):
        from .kernel_residual import predict_kernel_residual
        return predict_kernel_residual(packet, engine, text, paths, batch_size)
    if packet.get('variant', '').startswith('shared-'):
        from .shared_neural import predict_shared_neural
        return predict_shared_neural(packet, engine, text, paths, batch_size)
    def features(rows):
        return visual_features(engine, image_inputs(engine, rows, config['patches']))
    if frozen:
        result = []
        for i in range(0, len(paths), batch_size):
            z, dense, mask, coordinates = features(paths[i:i+batch_size])
            # Match the half-precision frozen cache used in baseline training.
            z, dense = z.half().float(), torch.nn.functional.normalize(dense.half().float(), dim=-1)
            scores = torch.cat((z@text[0].T, (dense@text[1].T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values), -1).cpu().numpy()
            if packet['format'] == 'fgclip2_multitarget_agop_ensemble_v1':
                result.append(scores)
                continue
            if packet['format'] in ('fgclip2_multitarget_agop_v1', 'fgclip2_multitarget_agop_weighted_v1'):
                from .agop import predict_scores
                result.append(predict_scores(packet, scores))
                continue
            values = np.zeros((len(z), len(targets)))
            for part in packet['readouts']:
                values[:, part['targets']] = predict_readout(part['readout'], scores)
            result.append(values)
        if packet['format'] == 'fgclip2_multitarget_agop_ensemble_v1':
            from .ensemble import predict_ensemble
            import gc
            del engine.model
            gc.collect()
            if engine.device.type == 'mps': torch.mps.empty_cache()
            elif engine.device.type == 'cuda': torch.cuda.empty_cache()
            return tuple(targets), predict_ensemble(packet, np.concatenate(result), Path(checkpoint).parent)
        return tuple(targets), np.concatenate(result)
    variant, selection, state = packet['variant'], packet['selection'], packet['state']
    kind = 'partial' if variant == 'chain-partial' else 'lora' if variant == 'chain-joint' else 'frozen'
    blocks = config['partial_blocks'] if kind == 'partial' else config['blocks']
    with BackboneSession(engine, kind, blocks, config['rank']):
        backbone = [(n, p) for n, p in engine.model.named_parameters() if p.requires_grad]
        prompt = AnchoredTokens(engine, phrases, selection['owners'], len(targets), text,
                                config['context_tokens'], config['prompt_rank'], config['text_chunk'])
        saved = state['network']
        network = RatingDAG(text[0].shape[-1], selection, len(targets), saved['means'].to(engine.device),
                            (saved['score_mean'].to(engine.device), saved['score_std'].to(engine.device)), prompt,
                            hidden=config['hidden'], bins=config['bins'], rank=config['pool_rank'],
                            independent=variant == 'independent', dropout=config['dropout'],
                            detach_context=not config['chain_gradients']).to(engine.device)
        restore(state, network, backbone)
        network.eval().requires_grad_(False)
        engine.model.eval().requires_grad_(False)
        text = prompt()
        result = []
        for i in range(0, len(paths), batch_size):
            values = features(paths[i:i+batch_size])
            if kind == 'frozen':
                z, dense, mask, coordinates = values
                values = (torch.nn.functional.normalize(z.half().float(), dim=-1),
                          torch.nn.functional.normalize(dense.half().float(), dim=-1), mask, coordinates.half().float())
            result.append(network(values, text=text, learned_pool='pool' in packet['components'])['prediction'].cpu().numpy())
        return tuple(targets), np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model-cache', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    paths = discover_face_images(FaceDatasetConfig(args.images, limit=None))
    targets, values = predict_checkpoint(args.checkpoint, paths, device=args.device,
                                         model_cache=args.model_cache, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', *targets])
        writer.writerows([path.name, *row.tolist()] for path, row in zip(paths, values))


if __name__ == '__main__':
    main()
