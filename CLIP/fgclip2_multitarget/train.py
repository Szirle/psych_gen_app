"""Fold-isolated staged fits, resumable at epoch boundaries."""
import gc
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from ..fgclip2_adaptation.models import BackboneSession
from ..fgclip2_face_impressions import array_hash, write_json
from ..race_perception import candidate_grid, inner_predictions, select_strategies, fit_bundle_readout, predict_readout
from .assets import atomic_save, cpu_tree
from .data import masked_report, select_phrases
from .models import AnchoredTokens, RatingDAG, loss_function


VARIANTS = {
    'frozen-kernel': (),
    'independent': ('head',),
    'chain-fixed': ('head',),
    'chain-pool': ('head', 'pool'),
    'chain-prompt': ('head', 'pool', 'prompt'),
    'chain-lora': ('head', 'pool', 'prompt', 'vision'),
    'chain-joint': ('head', 'pool', 'prompt', 'vision'),
    'chain-partial': ('head', 'pool', 'prompt', 'vision'),
}


def fit_kernel(assets, data, train, test, directory, args):
    """Matched frozen full-phrase baseline; reuse kernel eigendecompositions."""
    directory.mkdir(parents=True, exist_ok=True)
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
    if not args.smoke:
        atomic_save(dict(format='fgclip2_multitarget_kernel_v1', targets=data.targets, readouts=fitted,
                         phrases=assets.bank['phrases'], feature_order='global-short then local-box-max',
                         config=args.serializable_config, model=assets.engine.model_id,
                         revision=assets.engine.revision), directory/'model.pt')
    return prediction, {}


def selection_for(assets, data, rows, bank, args):
    fields = ('seed', 'mi_repeats', 'target_phrases', 'shared_phrases', 'redundancy',
              'inner_folds', 'order', 'hard_fraction', 'hierarchy_spec')
    key = array_hash(np.asarray([assets.key, Path(__file__).with_name('data.py').read_text(),
                                Path(__file__).with_name('phrases.py').read_text(),
                                json.dumps({k: getattr(args, k) for k in fields}, sort_keys=True)]),
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
    text = network.prompt()
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
        output = network(features, text=network.prompt(), learned_pool=learned_pool)
        coordinates = features[3].cpu().numpy()
        attention = output['attention'].cpu().float().numpy()
        token_drift = network.prompt.deltas().square().mean((1, 2)).sqrt().cpu().numpy()
    scores = output['scores'].detach().requires_grad_(True)
    mean, distribution = network.decode(scores)
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
                rating_distribution=distribution[i, t].softmax(-1).detach().cpu().tolist(), phrases=phrases)
    write_json(path, dict(note='Conditional score sensitivity, holding earlier predictions fixed when detach_context is enabled; attention is not a causal explanation.', images=records))


def fit(variant, assets, data, bank, train, validation, directory, args, *, durations=None, test=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if variant == 'frozen-kernel':
        return fit_kernel(assets, data, train, test, directory, args)
    selection = selection_for(assets, data, train, bank, args)
    write_json(directory/'selection.json', selection)
    kind = 'partial' if variant == 'chain-partial' else 'lora' if variant in ('chain-lora', 'chain-joint') else 'frozen'
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
        network = RatingDAG(initial[0].shape[-1], selection, len(data.targets), torch.tensor(means, device=device, dtype=torch.float32),
                            stats, prompt, hidden=args.hidden, bins=args.bins, rank=args.pool_rank,
                            independent=variant == 'independent', dropout=args.dropout,
                            detach_context=not args.chain_gradients).to(device)
        if args.smoke:
            # Pinning-specific text port validation, before its deltas are trained.
            with torch.no_grad():
                actual = prompt()
                for expected, value in zip(initial, actual):
                    if not torch.allclose(value, expected, atol=.006, rtol=.01):
                        raise RuntimeError('Zero-delta soft tokens do not match native FG-CLIP2 text outputs.')
        tensors = [torch.tensor(v, device=device) for v in (data.y, data.mask, data.se, data.hist)]
        chosen, histories, learned_pool = {}, {}, False
        for stage in VARIANTS[variant]:
            live = stage == 'vision'
            learned_pool = learned_pool or stage == 'pool'
            text_live = stage == 'prompt' or (live and variant == 'chain-joint')
            network.requires_grad_(True)
            network.prompt.requires_grad_(text_live)
            network.pool.requires_grad_(learned_pool)
            for _, parameter in backbone:
                parameter.requires_grad_(live)
            prompt_ids = {id(p) for p in prompt.parameters()}
            head = [p for p in network.parameters() if p.requires_grad and id(p) not in prompt_ids]
            groups = [{'params': head, 'lr': args.head_lr}]
            if text_live:
                groups.append({'params': list(prompt.parameters()), 'lr': args.prompt_lr, 'weight_decay': 0.})
            if live:
                groups.append({'params': [p for _, p in backbone], 'lr': args.partial_lr if kind == 'partial' else args.encoder_lr})
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            rates = [g['lr'] for g in groups]
            parameters = [p for g in groups for p in g['params']]
            maximum = getattr(args, stage+'_epochs')
            epochs = durations[stage] if durations is not None else maximum
            best_path, last_path = directory/f'{stage}.best.pt', directory/f'{stage}.last.pt'
            best, best_epoch, stale, start, history, done = float('inf'), 0, 0, 0, [], False
            if args.resume and last_path.exists():
                state = torch.load(last_path, map_location='cpu', weights_only=False)
                restore(state, network, backbone)
                optimizer.load_state_dict(state['optimizer'])
                start, best, best_epoch, stale, history, done = (state[k] for k in ('epoch', 'best', 'best_epoch', 'stale', 'history', 'done'))
            frozen_text = None
            if not text_live:
                with torch.no_grad():
                    frozen_text = tuple(t.detach() for t in prompt()) if stage != 'head' else initial
            before = {n: p.detach().cpu().clone() for n, p in network.named_parameters()} if args.smoke else None
            before_backbone = {n: p.detach().cpu().clone() for n, p in backbone} if args.smoke and live else None
            print(f'{directory.name}/{variant}/{stage}: {len(train)} train, {epochs} epochs, '
                  f'{sum(p.numel() for p in parameters):,} trainable parameters', flush=True)
            for epoch in range(start, epochs) if not done else ():
                torch.manual_seed(args.seed+epoch+100*list(VARIANTS[variant]).index(stage))
                random.seed(args.seed+epoch+100*list(VARIANTS[variant]).index(stage))
                rng = np.random.default_rng(args.seed+epoch)
                network.train()
                engine.model.vision_model.train(live)
                order = rng.choice(train, len(train), replace=True, p=selection['sampling']) if live else rng.permutation(train)
                batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
                if args.smoke:
                    batches = batches[:1]
                scale = min(1., (epoch+1)/2)*(.1+.9*(1+math.cos(math.pi*epoch/max(maximum, 1)))/2)
                for group, rate in zip(optimizer.param_groups, rates):
                    group['lr'] = rate*scale
                optimizer.zero_grad(set_to_none=True)
                total, seen = 0., 0
                for step, rows in enumerate(batches):
                    features = assets.batch(rows, live, augment=live and args.augmentation != 'none')
                    output = network(features, text=frozen_text, learned_pool=learned_pool)
                    ix = torch.as_tensor(rows, device=device)
                    teacher = assets.features[0][rows].float().to(device)
                    teacher = torch.nn.functional.normalize(teacher, dim=-1)
                    loss, mean_loss = loss_function(output, *(t[ix] for t in tensors), teacher=teacher, args=args)
                    if live and args.consistency_weight:
                        with torch.no_grad():
                            network.eval()
                            engine.model.eval()
                            clean = network(assets.batch(rows, True), text=tuple(t.detach() for t in prompt()) if frozen_text is None else frozen_text,
                                            learned_pool=learned_pool)['prediction']
                        network.train()
                        engine.model.vision_model.train(True)
                        loss = loss+args.consistency_weight*(output['prediction']-clean).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss in {variant}/{stage}')
                    window = min(args.accumulate, len(batches)-(step//args.accumulate)*args.accumulate)
                    (loss/window).backward()
                    if (step+1) % args.accumulate == 0 or step+1 == len(batches):
                        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    total += mean_loss.item()*len(rows)
                    seen += len(rows)
                metrics = None
                if validation is not None:
                    prediction = predict(network, assets, validation, live=live, learned_pool=learned_pool, batch=args.batch_size)
                    metrics = masked_report(data, validation, prediction, variant)
                value = metrics['target_macro_mse'] if metrics else total/seen
                history.append(dict(epoch=epoch+1, training_mse=total/seen, validation=metrics))
                improved = validation is None or value < best
                if improved:
                    best, best_epoch, stale = value, epoch+1, 0
                    if not args.smoke:
                        atomic_save(capture(network, backbone), best_path)
                else:
                    stale += 1
                done = epoch+1 == epochs or (validation is not None and stale >= args.patience)
                if not args.smoke:
                    state = dict(**capture(network, backbone), optimizer=cpu_tree(optimizer.state_dict()),
                                 epoch=epoch+1, best=best, best_epoch=best_epoch, stale=stale, history=history, done=done)
                    atomic_save(state, last_path)
                    del state
                if epoch == 0 or (epoch+1) % args.log_every == 0 or done:
                    print(f'  epoch {epoch+1}: train MSE {total/seen:.6f}; '
                          + (f'inner-validation RMSE {value**.5:.5f}' if metrics else 'refit'), flush=True)
                if done:
                    break
            if args.smoke:
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
                history[-1]['gradient_checks'] = dict(changed=changed, backbone_changed=bool(backbone_changed))
            else:
                restore(torch.load(best_path, map_location='cpu', weights_only=False), network, backbone)
            chosen[stage], histories[stage] = best_epoch, history
            write_json(directory/'history.json', dict(durations=chosen, stages=histories))
            if validation is not None and stage == 'prompt':
                diagnostics(network, assets, data, validation, directory/'before-vision-diagnostics.json', live=False,
                            learned_pool=learned_pool, selection=selection, limit=args.diagnostic_images)
            del optimizer, parameters, head, groups
        prediction = None
        if test is not None:
            live = 'vision' in VARIANTS[variant]
            prediction = predict(network, assets, test, live=live, learned_pool=learned_pool, batch=args.batch_size)
            diagnostics(network, assets, data, test, directory/'heldout-diagnostics.json', live=live,
                        learned_pool=learned_pool, selection=selection, limit=args.diagnostic_images)
        if not args.smoke:
            atomic_save(dict(format='fgclip2_multitarget_v1', variant=variant, targets=data.targets,
                             selection=selection, durations=chosen, state=capture(network, backbone),
                             config=args.serializable_config, model=engine.model_id, revision=engine.revision), directory/'model.pt')
        del network, backbone, prompt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return prediction, chosen
