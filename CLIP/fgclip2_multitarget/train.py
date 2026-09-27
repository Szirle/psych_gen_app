"""Fold-isolated joint fits, resumable at epoch boundaries."""
import gc
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from ..fgclip2_adaptation_legacy.models import BackboneSession
from ..fgclip2_face_impressions import array_hash, write_json
from ..race_perception import candidate_grid, inner_predictions, select_strategies, fit_bundle_readout, predict_readout
from .assets import atomic_save, cpu_tree
from .data import masked_report, select_phrases
from .models import AnchoredTokens, RatingDAG, loss_function
from .shared_neural import SHARED_VARIANTS, SharedAssets, SharedNeural, full_selection, resolve_profile, shared_loss
from .optimization import POLICY, StepSchedule, optimizer_groups, initialize_pool, joint_epochs, phrase_preservation
from .augmentation import TrainingViews


VARIANTS = {
    'frozen-kernel': (),
    'agop-kernel': (),
    'independent': ('head',),
    'chain-fixed': ('head',),
    'chain-pool': ('head', 'pool'),
    'chain-prompt': ('head', 'pool', 'prompt'),
    'chain-joint': ('head', 'pool', 'prompt', 'vision'),
    'chain-partial': ('head', 'pool', 'prompt', 'vision'),
}
VARIANTS.update({
    'kernel-residual': ('head',),
    'kernel-chain': ('head',),
    'kernel-chain-pool': ('head', 'pool'),
    'kernel-chain-prompt': ('head', 'pool', 'prompt'),
    'kernel-chain-joint': ('head', 'pool', 'prompt', 'vision'),
})
VARIANTS.update(SHARED_VARIANTS)
DEFAULT_VARIANTS = ('agop-kernel',)
# Values describe simultaneously enabled components, never a training schedule.


def kernel_readouts(assets, data, train, test, args):
    """The same full-bank recipe search for baseline and residual models."""
    recipes = [r for r in candidate_grid() if r['bank'] == 'both' and not r.get('centered')]
    if args.smoke:
        recipes = [dict(family=f, bank='both', centered=False, alpha=.1,
                        **({'gamma': .1} if f == 'rbf' else {})) for f in ('linear', 'poly2', 'rbf')]
    groups = {}
    for t in range(len(data.targets)):
        groups.setdefault(data.mask[train, t].tobytes(), []).append(t)
    prediction = np.zeros((len(test), len(data.targets))) if test is not None else None
    fitted = []
    for targets in groups.values():
        rows = train[data.mask[train, targets[0]]]
        x, y = assets.scores[rows], data.y[np.ix_(rows, targets)]
        inner, _ = inner_predictions(x, y, recipes, args.inner_folds, args.seed, data.groups[rows])
        strategies, _, _ = select_strategies(inner, y, recipes)
        readout = fit_bundle_readout(x, y, recipes, strategies)
        if test is not None:
            prediction[:, targets] = predict_readout(readout, assets.scores[test])
        fitted.append(dict(targets=targets, readout=readout))
    return prediction, fitted


def fit_kernel(assets, data, train, test, directory, args):
    """Matched frozen full-phrase baseline; reuse kernel eigendecompositions."""
    directory.mkdir(parents=True, exist_ok=True)
    prediction, fitted = kernel_readouts(assets, data, train, test, args)
    if not args.smoke:
        atomic_save(dict(format='fgclip2_multitarget_kernel_v1', targets=data.targets, readouts=fitted,
                         phrases=assets.bank['phrases'], feature_order='global-short then local-box-max',
                         config=args.serializable_config, model=assets.engine.model_id,
                         revision=assets.engine.revision), directory/'model.pt')
    return prediction, {}


def selection_for(assets, data, rows, bank, args):
    fields = ('seed', 'mi_repeats', 'target_phrases', 'shared_phrases', 'redundancy',
              'inner_folds', 'order', 'hard_fraction', 'hierarchy_spec')
    execution = dict(backend=getattr(args, 'selection_backend', 'auto'), device=str(args.device))
    key = array_hash(np.asarray([assets.key, Path(__file__).with_name('data.py').read_text(),
                                Path(__file__).with_name('phrases.py').read_text(),
                                json.dumps({k: getattr(args, k) for k in fields}, sort_keys=True),
                                json.dumps(execution, sort_keys=True)]),
                     rows, data.y[rows], data.mask[rows], data.se[rows], data.groups[rows],
                     np.asarray(bank['phrases']))
    path = Path(args.cache)/f'selection-{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    print(f'Selecting phrases and estimating target difficulty on {len(rows)} training images…', flush=True)
    selection = select_phrases(assets.scores, data, rows, bank, args)
    write_json(path, selection)
    return selection


def capture(network, backbone):
    return {'network': cpu_tree(network.state_dict()),
            'backbone': {n: cpu_tree(p) for n, p in backbone}}


def restore(state, network, backbone):
    network.load_state_dict(state['network'])
    parameters = dict(backbone)
    if set(parameters) != set(state['backbone']):
        raise ValueError('Checkpoint backbone topology mismatch')
    with torch.no_grad():
        for name, value in state['backbone'].items():
            parameters[name].copy_(value.to(parameters[name]))


@torch.no_grad()
def predict(network, assets, rows, *, live, learned_pool, batch):
    network.eval()
    assets.engine.model.eval()
    text = network.prediction_text() if hasattr(network, 'prediction_text') else network.prompt()
    return np.concatenate([network(assets.batch(rows[i:i+batch], live), text=text,
                                    learned_pool=learned_pool)['prediction'].cpu().numpy()
                           for i in range(0, len(rows), batch)])


def diagnostics(network, assets, data, rows, path, *, live, learned_pool, selection, limit):
    """Local sensitivity with predecessor ratings held fixed; not causality."""
    if not limit:
        return
    rows = rows[:limit]
    network.eval()
    assets.engine.model.eval()
    with torch.no_grad():
        features = assets.batch(rows, live)
        text = network.prediction_text() if hasattr(network, 'prediction_text') else network.prompt()
        output = network(features, text=text, learned_pool=learned_pool)
        coordinates = features[3].cpu().numpy()
        attention = output['attention'].cpu().float().numpy()
        token_drift = network.prompt.deltas().square().mean((1, 2)).sqrt().cpu().numpy()
    scores = output['scores'].detach().requires_grad_(True)
    extra = {k: output[k] for k in ('baseline', 'frozen_scores') if k in output}
    mean, distribution = network.decode(scores, **extra)
    records = [dict(filename=data.paths[i].name, row=int(i), targets={}) for i in rows]
    for t, target in enumerate(data.targets):
        derivative = torch.autograd.grad(mean[:, t].sum(), scores, retain_graph=t < len(data.targets)-1)[0]
        sensitivity = (derivative*(scores-network.score_mean)).sum(-1).detach().cpu().numpy()
        for i, row in enumerate(rows):
            selected = sorted(selection['target_indices'][t], key=lambda j: -abs(sensitivity[i, j]))[:3]
            phrases = []
            for j in selected:
                locations = np.argsort(-attention[i, :, j])[:3]
                phrases.append(dict(text=selection['phrases'][j], owner=selection['owners'][j],
                    training_mi=selection['mi'][t][j], token_delta_rms=float(token_drift[j]),
                    conditional_sensitivity=float(sensitivity[i, j]),
                    global_cosine=float(scores[i, j, 0].detach()), local_cosine=float(scores[i, j, 1].detach()),
                    patches=[dict(x=float(coordinates[i, k, 0]), y=float(coordinates[i, k, 1]),
                                  weight=float(attention[i, k, j])) for k in locations]))
            records[i]['targets'][target] = dict(prediction=float(mean[i, t].detach()),
                observed=float(data.y[row, t]) if data.mask[row, t] else None,
                mean_se=float(data.se[row, t]) if data.mask[row, t] else None,
                rating_distribution=(distribution[i, t].softmax(-1).detach().cpu().tolist()
                    if getattr(network, 'distribution_supervised', True) else None), phrases=phrases)
    note = 'Conditional score sensitivity, holding earlier predictions fixed when detach_context is enabled; attention is not a causal explanation.'
    if extra:
        note += ' Kernel variants show adaptive-residual sensitivity only: frozen scores and kernel predictions are held fixed; effective pooling weights can be signed.'
    write_json(path, dict(note=note, images=records))


def fit(variant, assets, data, bank, train, validation, directory, args, *, durations=None, test=None):
    shared_family = variant in SHARED_VARIANTS
    if shared_family:
        args = resolve_profile(args, variant)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    components = VARIANTS[variant]
    if variant == 'agop-kernel':
        from .agop import fit_agop
        return fit_agop(assets, data, train, test, directory, args)
    if variant == 'frozen-kernel':
        return fit_kernel(assets, data, train, test, directory, args)
    training_views = TrainingViews(assets, train, args, scores_only=shared_family and components == ('head',))
    try:
        assets = training_views
        if durations is not None and set(durations) != {'joint'}:
            raise ValueError('Expected one joint fit duration')
        profile = {k: getattr(args, k) for k in ('head_lr', 'pool_lr', 'prompt_lr', 'encoder_lr',
            'partial_lr', 'weight_decay', 'batch_size', 'accumulate', 'dropout', 'patience',
            'token_anchor', 'feature_anchor', 'prompt_logit_weight', 'prompt_logit_temperature',
            'preserve_weight', 'consistency_weight', 'warmup_fraction', 'min_lr_ratio',
            'grad_clip', 'min_train_fraction', 'vision_weight_decay')}
        profile.update(policy=POLICY, components=components, joint_epochs=joint_epochs(args))
        profile['augmentation'] = {k: v for k, v in vars(args).items() if k == 'augmentation' or k.startswith('augmentation_')}
        if shared_family and 'vision' in components:
            profile['batch_size'] = args.shared_vision_batch_size
        write_json(directory/'training-profile.json', profile)
        print(f'{variant} training profile: {profile}', flush=True)
        kernel_family = variant.startswith('kernel-')
        if shared_family:
            assets, selection = SharedAssets(assets), full_selection(data, train, bank, args)
        elif kernel_family:
            from .kernel_residual import KernelResidualDAG, prepare_reference
            assets, selection = prepare_reference(assets, data, train, bank, args)
        else:
            selection = selection_for(assets, data, train, bank, args)
        write_json(directory/'selection.json', selection)
        kind = 'partial' if variant == 'chain-partial' else 'lora' if variant in ('chain-joint', 'kernel-chain-joint') or (shared_family and variant.endswith('-joint')) else 'frozen'
        blocks = args.partial_blocks if kind == 'partial' else args.blocks
        engine, device = assets.engine, assets.engine.device
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        initial = assets.initial(selection)
        stats = assets.statistics(train, initial)
        means = (data.y[train]*data.mask[train]).sum(0)/data.mask[train].sum(0)
        with BackboneSession(engine, kind, blocks, args.rank):
            backbone = [(n, p) for n, p in engine.model.named_parameters() if p.requires_grad]
            torch.manual_seed(args.seed)  # identical head initialization across backbone variants
            prompt = AnchoredTokens(engine, selection['phrases'], selection['owners'], len(data.targets), initial,
                                    args.context_tokens, args.prompt_rank, args.text_chunk)
            network_args = (initial[0].shape[-1], selection, len(data.targets),
                            torch.tensor(means, device=device, dtype=torch.float32), stats, prompt)
            if shared_family:
                network = SharedNeural(*network_args, args=args, variant=variant).to(device)
            else:
                network_class = KernelResidualDAG if kernel_family else RatingDAG
                network = network_class(*network_args, hidden=args.kernel_hidden if kernel_family else args.hidden,
                    bins=args.bins, rank=args.pool_rank, independent=variant in ('independent', 'kernel-residual'),
                    dropout=args.dropout, detach_context=not args.chain_gradients).to(device)
            initialize_pool(network.pool)
            loss_args = args
            if kernel_family and not args.kernel_auxiliary_losses:
                from copy import copy
                loss_args = copy(args)
                loss_args.distribution_weight = loss_args.rank_weight = 0.
            if kernel_family:
                network.distribution_supervised = bool(loss_args.distribution_weight)
            if args.smoke:
                # Pinning-specific text port validation, before its deltas are trained.
                with torch.no_grad():
                    actual = prompt()
                    for mode, expected, value in zip(('short', 'box'), initial, actual):
                        if not torch.allclose(value, expected, atol=.006, rtol=.01):
                            error = (value-expected).abs().max().item()
                            raise RuntimeError(f'Zero-delta {mode} soft tokens do not match native FG-CLIP2 text outputs '
                                               f'(maximum absolute error {error:.6g}).')
            tensors = [torch.tensor(v, device=device) for v in (data.y, data.mask, data.se, data.hist)]
            live = 'vision' in components
            learned_pool = 'pool' in components
            text_live = 'prompt' in components
            if shared_family:
                network.cached_head = assets.cached_head = components == ('head',)
                args.batch_size = args.shared_vision_batch_size if live else args.serializable_config['batch_size']
            network.requires_grad_(True)
            network.prompt.requires_grad_(text_live)
            network.pool.requires_grad_(learned_pool)
            for _, parameter in backbone:
                parameter.requires_grad_(live)
            groups, component_parameters = optimizer_groups(network, backbone, args, kind)
            head = component_parameters['head']
            optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
            parameters = [p for g in groups for p in g['params']]
            maximum = joint_epochs(args)
            epochs = durations['joint'] if durations is not None else maximum
            batches_per_epoch = 4 if args.smoke else math.ceil(len(train)/args.batch_size)
            updates_per_epoch = math.ceil(batches_per_epoch/args.accumulate)
            # A duration-selected refit follows the selected prefix of the original
            # schedule, rather than compressing an unevaluated cosine into it.
            schedule = StepSchedule(maximum*updates_per_epoch, args.warmup_fraction, args.min_lr_ratio)
            schedule_spec = dict(total_steps=schedule.total_steps, warmup_steps=schedule.warmup_steps,
                                 minimum_ratio=schedule.minimum_ratio, updates_per_epoch=updates_per_epoch)
            optimizer_step = 0
            profile.update(schedule=schedule_spec, effective_batch_size=args.batch_size*args.accumulate,
                optimizer=dict(name='AdamW', betas=[.9, .999], eps=1e-8),
                parameter_groups=[dict(component=g['component'], base_lr=g['base_lr'], weight_decay=g['weight_decay'],
                                       parameters=sum(p.numel() for p in g['params'])) for g in groups])
            write_json(directory/'training-profile.json', profile)
            best_path, last_path = directory/'joint.best.pt', directory/'joint.last.pt'
            best, best_epoch, stale, start, history, done = float('inf'), 0, 0, 0, [], False
            if args.resume and last_path.exists():
                state = torch.load(last_path, map_location='cpu', weights_only=False)
                if state.get('training_policy') != POLICY or tuple(state.get('components', ())) != components:
                    raise ValueError('Joint optimizer checkpoint policy/components mismatch')
                if state['schedule'] != schedule_spec:
                    raise ValueError('Optimizer schedule mismatch on resume')
                optimizer_step = state['optimizer_step']
                restore(state, network, backbone)
                optimizer.load_state_dict(state['optimizer'])
                start, best, best_epoch, stale, history, done = (state[k] for k in ('epoch', 'best', 'best_epoch', 'stale', 'history', 'done'))
                if optimizer_step != start*updates_per_epoch:
                    raise ValueError('Optimizer step counter does not match the epoch boundary')
            elif kernel_family and not args.smoke:
                # Epoch zero is a real candidate, so an unhelpful correction
                # need not displace the kernel.
                if validation is not None:
                    initial_prediction = predict(network, assets, validation, live=live, learned_pool=learned_pool, batch=args.batch_size)
                    initial_metrics = masked_report(data, validation, initial_prediction, variant)
                    best = initial_metrics['target_macro_mse']
                    history.append(dict(epoch=0, validation=initial_metrics))
                atomic_save(capture(network, backbone), best_path)
            frozen_text = None
            if not text_live:
                frozen_text = initial
            if shared_family:
                network.fixed_text = frozen_text
            before = {n: p.detach().cpu().clone() for n, p in network.named_parameters()} if args.smoke else None
            before_backbone = {n: p.detach().cpu().clone() for n, p in backbone} if args.smoke and live else None
            gradient_checks = dict(head=False, pool=False, prompt=False, vision=False)
            print(f'{directory.name}/{variant}/joint: {len(train)} train, {epochs} epochs, '
                  f'{sum(p.numel() for p in parameters):,} trainable parameters', flush=True)
            for epoch in range(start, epochs) if not done else ():
                torch.manual_seed(args.seed+epoch)
                random.seed(args.seed+epoch)
                rng = np.random.default_rng(args.seed+epoch)
                training_views.set_epoch(epoch, live=live)
                network.train()
                engine.model.vision_model.train(live)
                order = rng.choice(train, len(train), replace=True, p=selection['sampling']) if live else rng.permutation(train)
                batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
                if args.smoke:
                    # Zero-initialized kernel corrections/gates open gradients
                    # over successive updates; no separate warm-up phase.
                    batches = (batches*4)[:4]
                optimizer.zero_grad(set_to_none=True)
                total, seen = 0., 0
                loss_totals = dict(objective=0., token_anchor=0., text_anchor=0., phrase_preservation=0.)
                norm_totals = {name: torch.zeros((), device=device) for name in component_parameters}
                norm_totals['global'] = torch.zeros((), device=device)
                clipped_steps, epoch_updates = torch.zeros((), device=device), 0
                for step, rows in enumerate(batches):
                    features = assets.batch(rows, live, augment=training_views.enabled)
                    text = prompt() if text_live else frozen_text
                    output = network(features, text=text, learned_pool=learned_pool)
                    ix = torch.as_tensor(rows, device=device)
                    teacher = assets.features[0][rows].float().to(device)
                    teacher = torch.nn.functional.normalize(teacher, dim=-1)
                    objective = shared_loss if shared_family else loss_function
                    loss, mean_loss = objective(output, *(t[ix] for t in tensors), teacher=teacher, args=loss_args)
                    agreement = loss.new_zeros(())
                    if text_live and args.prompt_logit_weight:
                        # Keep the reference view identical for teacher/student.
                        # Even with live augmented vision, this prompt-only branch
                        # reuses frozen clean-image features; no second encoder.
                        reference = assets.batch(rows, False) if live else features
                        agreement = phrase_preservation(reference, text, initial, args.prompt_logit_temperature)
                        loss = loss + args.prompt_logit_weight*agreement
                    if live and args.consistency_weight:
                        with torch.no_grad():
                            network.eval()
                            engine.model.eval()
                            clean = network(assets.batch(rows, True), text=tuple(t.detach() for t in text),
                                            learned_pool=learned_pool)['prediction']
                        network.train()
                        engine.model.vision_model.train(True)
                        loss = loss+args.consistency_weight*(output['prediction']-clean).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss in {variant}/joint')
                    window_start = (step//args.accumulate)*args.accumulate
                    window_rows = sum(len(b) for b in batches[window_start:window_start+args.accumulate])
                    if args.smoke:
                        # Prove that regression supervision reaches components,
                        # rather than merely observing anchor/AdamW updates.
                        task_grads = torch.autograd.grad(mean_loss, parameters, retain_graph=True, allow_unused=True)
                        active = {id(p) for p, g in zip(parameters, task_grads)
                                  if g is not None and bool(g.abs().max() > 0)}
                        for name, params in (('head', head), ('pool', network.pool.parameters()),
                                             ('prompt', prompt.parameters()), ('vision', (p for _, p in backbone))):
                            gradient_checks[name] |= any(id(p) in active for p in params)
                        del task_grads
                    (loss*(len(rows)/window_rows)).backward()
                    if (step+1) % args.accumulate == 0 or step+1 == len(batches):
                        # Schedule advances only when AdamW actually updates, not
                        # for microbatches, cache refreshes or validation passes.
                        schedule.apply(optimizer, optimizer_step)
                        for name, params in component_parameters.items():
                            norms = [p.grad.detach().norm() for p in params if p.grad is not None]
                            if norms:
                                norm_totals[name] += torch.stack(norms).norm()
                        norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
                        norm_totals['global'] += norm
                        clipped_steps += (norm > args.grad_clip)
                        optimizer.step()
                        optimizer_step += 1
                        epoch_updates += 1
                        optimizer.zero_grad(set_to_none=True)
                    terms = dict(objective=loss.detach(), token_anchor=args.token_anchor*output['token'].detach(),
                                 text_anchor=args.feature_anchor*output['anchor'].detach(),
                                 phrase_preservation=args.prompt_logit_weight*agreement.detach())
                    for name, value in terms.items():
                        loss_totals[name] += float(value)*len(rows)
                    total += mean_loss.item()*len(rows)
                    seen += len(rows)
                metrics = None
                if validation is not None:
                    prediction = predict(network, assets, validation, live=live, learned_pool=learned_pool, batch=args.batch_size)
                    metrics = masked_report(data, validation, prediction, variant)
                value = metrics['target_macro_mse'] if metrics else total/seen
                history.append(dict(epoch=epoch+1, training_mse=total/seen, validation=metrics,
                                    augmentation_generation=training_views.generation,
                                    losses={name: value/seen for name, value in loss_totals.items()},
                                    optimizer_step=optimizer_step, learning_rates={g['component']: g['lr'] for g in optimizer.param_groups},
                                    gradient_norms={name: float(value)/epoch_updates for name, value in norm_totals.items()},
                                    clipped_step_fraction=float(clipped_steps)/epoch_updates))
                improved = validation is None or value < best
                if improved:
                    best, best_epoch, stale = value, epoch+1, 0
                    if not args.smoke:
                        atomic_save(capture(network, backbone), best_path)
                else:
                    stale += 1
                minimum = math.ceil(maximum*args.min_train_fraction)
                done = epoch+1 == epochs or (validation is not None and stale >= args.patience and epoch+1 >= minimum)
                if not args.smoke:
                    state = dict(**capture(network, backbone), optimizer=cpu_tree(optimizer.state_dict()),
                                 training_policy=POLICY, components=components,
                                 optimizer_step=optimizer_step, schedule=schedule_spec,
                                 epoch=epoch+1, best=best, best_epoch=best_epoch, stale=stale, history=history, done=done)
                    atomic_save(state, last_path)
                    del state
                if epoch == 0 or (epoch+1) % args.log_every == 0 or done:
                    print(f'  epoch {epoch+1}: train MSE {total/seen:.6f}; '
                          + (f'inner-validation RMSE {value**.5:.5f}' if metrics else 'refit'), flush=True)
                if done:
                    break
            if args.smoke:
                for name in components:
                    if not gradient_checks[name]:
                        raise RuntimeError(f'No nonzero regression gradient reached enabled component: {name}')
                changed = [n for n, p in network.named_parameters() if not torch.equal(p.detach().cpu(), before[n])]
                if not any(n.startswith(('heads.', 'encoders.')) for n in changed):
                    raise RuntimeError('Head did not update during smoke training')
                if text_live and not any(n.startswith('prompt.') for n in changed):
                    raise RuntimeError('Phrase tokens did not update')
                if learned_pool and not any(n.startswith('pool.') for n in changed):
                    raise RuntimeError('Spatial pooling did not update')
                backbone_changed = live and any(not torch.equal(p.detach().cpu(), before_backbone[n]) for n, p in backbone)
                if live and not backbone_changed:
                    raise RuntimeError('Visual backbone did not update')
                history[-1]['gradient_checks'] = dict(changed=changed, backbone_changed=bool(backbone_changed),
                                                      nonzero_gradients=gradient_checks)
            else:
                restore(torch.load(best_path, map_location='cpu', weights_only=False), network, backbone)
            chosen = {'joint': best_epoch}
            write_json(directory/'history.json', dict(training_policy=POLICY, components=components,
                                                      durations=chosen, history=history))
            del optimizer, parameters, head, groups
            prediction = None
            if test is not None:
                live = 'vision' in components
                prediction = predict(network, assets, test, live=live, learned_pool=learned_pool, batch=args.batch_size)
                diagnostics(network, assets, data, test, directory/'heldout-diagnostics.json', live=live,
                            learned_pool=learned_pool, selection=selection, limit=args.diagnostic_images)
            if not args.smoke:
                extra = dict(kernel_readouts=assets.readouts) if kernel_family else {}
                atomic_save(dict(format='fgclip2_multitarget_v2', variant=variant, targets=data.targets,
                                 training_policy=POLICY, components=components,
                                 selection=selection, durations=chosen, state=capture(network, backbone),
                                 config=args.serializable_config, model=engine.model_id, revision=engine.revision, **extra), directory/'model.pt')
            del network, backbone, prompt
        training_views.cache = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return prediction, chosen
    finally:
        training_views.close()
