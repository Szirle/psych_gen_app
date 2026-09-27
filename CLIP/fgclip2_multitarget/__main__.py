"""All-target, phrase-grounded FG-CLIP2 research for a large CUDA GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..fgclip2_core import FGCLIP2, DEFAULT_MODEL_ID, DEFAULT_CACHE_DIR, BASE_MODEL_ID
from ..fgclip2_face_impressions import array_hash, write_json, _atomic_save_npz
from .assets import Assets, image_inputs
from .data import load_ratings, masked_report, split_rows, tail_split
from .models import visual_features
from .phrases import phrase_bank
from .train import VARIANTS, DEFAULT_VARIANTS, fit
from .shared_neural import add_arguments, resolve_profile, SHARED_VARIANTS
from .optimization import POLICY
from . import augmentation
from .agop import expanded_bank, bank_view


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    augmentation.add_arguments(p)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--ratings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path(__file__).parent/'.cache')
    p.add_argument('--model-cache', type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    p.add_argument('--targets', nargs='+', help='Default: every dimension in the ratings store.')
    p.add_argument('--phrase-overrides', type=Path, help='JSON target -> hand-written phrase list.')
    p.add_argument('--hierarchy', type=Path, help='JSON array of target groups, first group predicted first.')
    p.add_argument('--groups', type=Path, help='JSON filename -> related-identity/latent-family group.')
    p.add_argument('--order', choices=('learned', 'fixed'), default='learned')
    p.add_argument('--protocol', choices=('holdout', 'cv'), default='cv')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=4)
    p.add_argument('--tail-target', help='Hold out both rating tails for this predeclared target; holdout protocol only.')
    p.add_argument('--tail-fraction', type=float, default=.1)
    p.add_argument('--final-fit', action='store_true', help='Also fit all images using median inner-selected joint duration.')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='32 faces, four joint minibatches; never saves model/optimizer checkpoints.')
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--model', choices=('so400m', 'base'), default='so400m')
    p.add_argument('--precision', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    p.add_argument('--patches', type=int, choices=(128, 256, 576, 784, 1024), default=576)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--encode-batch', type=int, default=256)
    p.add_argument('--text-chunk', type=int, default=16)
    p.add_argument('--accumulate', type=int, default=1)
    p.add_argument('--joint-epochs', type=int, help='Override joint epoch maximum for every neural variant; otherwise use family-specific budgets.')
    p.add_argument('--epochs', type=int, default=200, help='Joint epoch maximum for legacy/kernel neural models.')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--pool-lr', type=float, default=1e-4, help='Joint pooling learning rate, with no weight decay.')
    p.add_argument('--prompt-lr', type=float, default=2e-5)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--partial-lr', type=float, default=1e-6)
    p.add_argument('--warmup-fraction', type=float, default=.1, help='Fraction of planned optimizer updates for linear warmup.')
    p.add_argument('--min-lr-ratio', type=float, default=.1, help='Cosine LR floor as a fraction of each group base LR.')
    p.add_argument('--grad-clip', type=float, default=1., help='Global L2 gradient norm limit, applied once per optimizer update.')
    p.add_argument('--min-train-fraction', type=float, default=1., help='Earliest stopping fraction; default runs the full cosine horizon while selecting the best inner-validation checkpoint.')
    p.add_argument('--vision-weight-decay', type=float, default=.01, help='Decay for partially unfrozen vision matrices; LoRA factors have no decay.')
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--kernel-hidden', type=int, default=256, help='Shared full-bank residual encoder width; kernel-* variants only.')
    p.add_argument('--kernel-auxiliary-losses', action='store_true', help='Enable distribution/ranking losses for kernel-* corrections; default mean MSE only.')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--partial-blocks', type=int, default=2)
    p.add_argument('--pool-rank', type=int, default=32)
    p.add_argument('--prompt-rank', type=int, default=8)
    p.add_argument('--context-tokens', type=int, default=4)
    p.add_argument('--bins', type=int, default=9)
    p.add_argument('--mi-repeats', type=int, default=3)
    p.add_argument('--selection-backend', choices=('auto', 'cuda', 'sklearn'), default='auto',
                   help='auto uses batched exact KSG MI on CUDA, sklearn elsewhere.')
    p.add_argument('--selection-memory-gib', type=float, default=32.,
                   help='Temporary CUDA MI workspace budget; also capped by free VRAM.')
    p.add_argument('--shared-phrases', type=int, default=16)
    p.add_argument('--target-phrases', type=int, default=8, help='MI-selected extras, in addition to each target\'s hand-designed phrases.')
    p.add_argument('--redundancy', type=float, default=.03)
    p.add_argument('--hard-fraction', type=float, default=.25)
    p.add_argument('--distribution-weight', type=float, default=.2)
    p.add_argument('--rank-weight', type=float, default=.03)
    p.add_argument('--token-anchor', type=float, default=.1)
    p.add_argument('--feature-anchor', type=float, default=.1)
    p.add_argument('--prompt-logit-weight', type=float, default=.05,
                   help='Preserve original global/local phrase agreement on frozen image features; 0 disables.')
    p.add_argument('--prompt-logit-temperature', type=float, default=.07,
                   help='Cosine-score temperature for phrase agreement distillation (not rating probabilities).')
    p.add_argument('--preserve-weight', type=float, default=.1)
    p.add_argument('--consistency-weight', type=float, default=.05)
    p.add_argument('--chain-gradients', action='store_true', help='Allow later-target losses to change earlier predictors (ablation).')
    p.add_argument('--diagnostic-images', type=int, default=8)
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--cpu-threads', type=int, default=8)
    return p


@torch.no_grad()
def native_parity(engine, data, patches):
    inputs = image_inputs(engine, data.paths[:2], patches)
    joint = visual_features(engine, inputs)
    native_global = engine.encode_image_preprocessed(inputs)
    native_dense = engine.encode_dense_preprocessed(inputs)
    torch.testing.assert_close(joint[0], native_global.float(), atol=.006, rtol=.02)
    for i, native in enumerate(native_dense):
        torch.testing.assert_close(joint[1][i, joint[2][i]], native.reshape(-1, native.shape[-1]).float(), atol=.006, rtol=.02)
    print('Native global/dense parity passed.', flush=True)


def main():
    args = parser().parse_args()
    if args.smoke:
        args.limit = args.limit or 32
        args.joint_epochs = 1
        args.augmentation_views = min(args.augmentation_views, 2)
        args.mi_repeats, args.inner_folds, args.folds = 1, 2, 4
        args.batch_size, args.encode_batch, args.text_chunk, args.accumulate = 4, 4, 8, 1
        args.shared_phrases, args.target_phrases, args.diagnostic_images = 4, 2, min(args.diagnostic_images, 2)
        args.blocks, args.partial_blocks, args.protocol = 1, 1, 'holdout'
        if args.resume or args.final_fit:
            raise ValueError('Smoke mode forbids resume/final-fit and writes no checkpoints.')
    for name in ('folds', 'inner_folds', 'bins'):
        if getattr(args, name) < 2:
            raise ValueError(f'{name} must be at least two')
    augmentation.validate(args)
    augmentation.cleanup_cache(args.cache)
    for name in ('batch_size', 'encode_batch', 'text_chunk', 'accumulate', 'epochs',
                 'patience', 'hidden', 'rank', 'blocks', 'partial_blocks',
                 'pool_rank', 'prompt_rank', 'context_tokens', 'mi_repeats', 'shared_phrases', 'target_phrases',
                 'cpu_threads', 'log_every', 'kernel_hidden'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if not 0 <= args.hard_fraction <= 1 or not 0 <= args.dropout < 1 or not 0 < args.tail_fraction < .5:
        raise ValueError('Invalid sampling/dropout/tail fraction')
    if args.tail_target and args.protocol != 'holdout':
        raise ValueError('--tail-target requires --protocol holdout')
    if args.context_tokens > 16 or args.diagnostic_images < 0:
        raise ValueError('At most 16 context tokens; diagnostic-images must be nonnegative')
    if not np.isfinite(args.selection_memory_gib) or args.selection_memory_gib <= 0:
        raise ValueError('selection-memory-gib must be finite and positive')
    if args.joint_epochs is not None and args.joint_epochs < 1:
        raise ValueError('joint-epochs must be positive')
    for name in ('head_lr', 'pool_lr', 'prompt_lr', 'encoder_lr', 'partial_lr', 'prompt_logit_temperature', 'grad_clip'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('prompt_logit_weight', 'token_anchor', 'feature_anchor', 'preserve_weight', 'consistency_weight', 'weight_decay', 'vision_weight_decay'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 <= args.warmup_fraction < 1 or not 0 <= args.min_lr_ratio <= 1 or not args.warmup_fraction <= args.min_train_fraction <= 1:
        raise ValueError('Require 0 <= warmup < 1, 0 <= min-lr-ratio <= 1, and warmup <= min-train-fraction <= 1')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run in the PyTorch-CUDA VM; no silent CPU fallback.')
    if args.precision == 'bf16' and str(args.device).startswith('cuda') and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU lacks bf16 support. Use --precision fp32.')
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('high')
    args.hierarchy_spec = json.loads(args.hierarchy.read_text()) if args.hierarchy else None
    overrides = json.loads(args.phrase_overrides.read_text()) if args.phrase_overrides else None
    data = load_ratings(args.images, args.ratings, targets=args.targets, bins=args.bins, limit=args.limit,
                        seed=args.seed, groups=args.groups)
    base_bank = phrase_bank(data.targets, overrides)
    bank = expanded_bank(base_bank) if 'agop-kernel' in args.variants else base_bank
    args.stage_frozen_encoders = str(args.device).startswith('mps') and all(v in ('frozen-kernel', 'agop-kernel') for v in args.variants)
    args.serializable_config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    for variant in args.variants:
        if variant in SHARED_VARIANTS:
            resolve_profile(args, variant)  # validate profiles before loading the large model
    source_paths = [*Path(__file__).parent.glob('*.py'), Path(__file__).parent.parent/'fgclip2_core.py',
                    Path(__file__).parent.parent/'fgclip2_face_impressions.py',
                    Path(__file__).parent.parent/'fgclip2_adaptation_legacy/models.py']
    source_paths.extend((Path(__file__).parent.parent/'clip_psychometry').glob('*.py'))
    source_paths.extend((Path(__file__).parent.parent/'race_perception').glob('*.py'))
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source_paths))).hexdigest()
    manifest = dict(training_policy=POLICY, config=args.serializable_config, targets=list(data.targets),
                    paths=[str(p.resolve()) for p in data.paths], groups=data.groups.tolist(),
                    label_counts=data.mask.sum(0).tolist(), bank=bank, baseline_bank=base_bank,
                    data_hash=array_hash(data.y, data.mask, data.se, data.hist, data.groups, np.asarray(data.image_hashes)), source_hash=source_hash,
                    environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda))
    # CUDA host name/version can change on resume; experiment inputs must not.
    identity_config = {k: v for k, v in args.serializable_config.items() if k not in ('resume', 'log_every', 'cpu_threads')}
    identity = array_hash(np.asarray([json.dumps(identity_config, sort_keys=True), manifest['data_hash'], source_hash,
                                     json.dumps(bank, sort_keys=True)]))
    manifest['identity'] = identity
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.resume:
            raise ValueError('Use a new output directory, or --resume.')
        elif old.get('training_policy') != POLICY:
            raise ValueError('Training policy mismatch. Use a new --output / RUN_DIR.')
        elif old['identity'] != identity:
            raise ValueError('Resume identity mismatch: code, data or configuration changed. Use a new RUN_DIR to avoid mixing experiments.')
    elif any(args.output.iterdir()):
        raise ValueError('Output directory must be empty for a new run.')
    write_json(manifest_path, manifest)
    print(f'{len(data.paths)} unique-image rows; {len(data.targets)} dimensions; {len(bank["phrases"])} candidate phrases.', flush=True)
    print(f'Variants: {", ".join(args.variants)}; protocol={args.protocol}; patches={args.patches}', flush=True)
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.precision]
    kernel_only = all(v in ('frozen-kernel', 'agop-kernel') for v in args.variants)
    engine = FGCLIP2(BASE_MODEL_ID if args.model == 'base' else DEFAULT_MODEL_ID, device=args.device, dtype=dtype,
                     cache_dir=args.model_cache, local_files_only=True,
                     memory_reserve_gb=.5 if args.stage_frozen_encoders else 1.25)
    manifest.update(model=engine.model_id, revision=engine.revision, dtype=str(dtype))
    if engine.device.type == 'cuda':
        manifest['environment'].update(gpu=torch.cuda.get_device_name(engine.device),
            vram_gib=torch.cuda.get_device_properties(engine.device).total_memory/1024**3)
    write_json(manifest_path, manifest)
    if args.smoke:
        native_parity(engine, data, args.patches)
    if kernel_only:
        from .frozen_assets import FrozenScoreAssets
        assets = FrozenScoreAssets(engine, data, bank, args)
    else:
        assets = Assets(engine, data, bank, args)
    full_assets = assets
    if all(v in ('frozen-kernel', 'agop-kernel') for v in args.variants):
        # Kernel training uses cached scores only, so release the large encoder.
        import gc
        del engine.model
        gc.collect()
        if engine.device.type == 'mps': torch.mps.empty_cache()
        elif engine.device.type == 'cuda': torch.cuda.empty_cache()
    all_rows = np.arange(len(data.paths))
    splits = [tail_split(data, all_rows, args.tail_target, args.tail_fraction)] if args.tail_target else split_rows(data, all_rows, args.folds, args.seed)
    if args.protocol == 'holdout':
        splits = splits[:1]
    comparison = {}
    with threadpool_limits(limits=args.cpu_threads):
        for variant in args.variants:
            variant_bank = bank if variant == 'agop-kernel' else base_bank
            assets = bank_view(full_assets, variant_bank)
            reports, durations, rows_all, predictions_all, fold_all = [], [], [], [], []
            for fold, (train, test) in enumerate(splits):
                directory = args.output/variant/f'fold-{fold}'
                completed, prediction_path = directory/'results.json', directory/'predictions.npz'
                if args.resume and completed.exists() and prediction_path.exists():
                    report = json.loads(completed.read_text())
                    with np.load(prediction_path) as stored:
                        prediction = stored['predictions']
                        if not np.array_equal(stored['rows'], test):
                            raise ValueError('Stored fold rows do not match current split')
                    chosen = report['metadata']['durations']
                    print(f'{variant}/fold-{fold}: already complete', flush=True)
                else:
                    inner_train, inner_valid = (tail_split(data, train, args.tail_target, args.tail_fraction)
                        if args.tail_target else split_rows(data, train, args.inner_folds, args.seed+fold+1)[0])
                    started = time.monotonic()
                    chosen = {}
                    if variant not in ('frozen-kernel', 'agop-kernel'):
                        _, chosen = fit(variant, assets, data, variant_bank, inner_train, inner_valid, directory/'selection-fit', args)
                    prediction, _ = fit(variant, assets, data, variant_bank, train, None, directory/'refit', args, durations=chosen, test=test)
                    report = masked_report(data, test, prediction, variant, dict(fold=fold,
                        train=train.tolist(), test=test.tolist(), inner_train=inner_train.tolist(),
                        inner_validation=inner_valid.tolist(), durations=chosen, seconds=time.monotonic()-started))
                    _atomic_save_npz(prediction_path, predictions=prediction, rows=test, y=data.y[test], mask=data.mask[test])
                    write_json(completed, report)
                    print(f'{variant}/fold-{fold}: held-out target-macro RMSE={report["target_macro_rmse"]:.5f}', flush=True)
                reports.append(report)
                durations.append(chosen)
                rows_all.extend(test.tolist())
                predictions_all.append(prediction)
                fold_all.extend([fold]*len(test))
            rows = np.asarray(rows_all)
            prediction = np.concatenate(predictions_all)
            pooled = masked_report(data, rows, prediction, variant, dict(protocol=args.protocol, smoke=args.smoke,
                                    folds=reports, note='Internal validation; not an unseen external cohort.'))
            write_json(args.output/variant/'summary.json', pooled)
            comparison[variant] = dict(target_macro_rmse=pooled['target_macro_rmse'],
                targets={t: m.get('rmse') for t, m in pooled['metrics'].items()})
            write_json(args.output/'comparison.json', comparison)
            _atomic_save_npz(args.output/variant/'heldout.npz', rows=rows, predictions=prediction, y=data.y[rows],
                             mask=data.mask[rows], fold_ids=np.asarray(fold_all))
            if args.final_fit:
                chosen = {phase: int(np.median([d[phase] for d in durations])) for phase in durations[0]}
                fit(variant, assets, data, variant_bank, all_rows, None, args.output/variant/'final', args, durations=chosen)
    write_json(args.output/'complete.json', dict(variants=args.variants, smoke=args.smoke,
                                                identity=identity, model_checkpoints_saved=not args.smoke))
    print('Complete. ' + ('No model checkpoints saved.' if args.smoke else 'Research checkpoints and held-out predictions saved.'), flush=True)


if __name__ == '__main__':
    main()
