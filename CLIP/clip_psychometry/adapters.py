"""Optional integrations. Existing training code and caches remain untouched."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from .schema import Column, Dataset, Registry


def _repo(name):
    # Repository-specific adapters run as CLIP.clip_psychometry from its parent.
    if not __package__.startswith('CLIP.'):
        raise ImportError('Run repository adapters from the project parent as python -m CLIP.clip_psychometry; pure analysis also works as clip_psychometry.')
    return importlib.import_module('CLIP.'+name)


def from_legacy_features(paths, channels, *, phrase_groups=None, identity_groups=None):
    """Read the existing extract_impression_features NPZ contract without torch.

    Preserve original row/text order; never silently join by position. For a new
    multi-target export use from_assets, preserving its observed-label mask.
    """
    if len(paths) != len(channels) or len(set(channels)) != len(channels):
        raise ValueError('One unique channel name per archive is required.')
    banks, first, sources, columns = [], None, [], []
    groups = json.loads(Path(phrase_groups).read_text()) if phrase_groups else []
    for source, channel in zip(paths, channels):
        with np.load(source, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata']))
            names, texts, y, x = tuple(d['paths'].tolist()), tuple(d['texts'].tolist()), d['y'].copy(), d['x'].copy()
            if first is None:
                first = names, texts, y, meta['variable']
            elif names != first[0] or texts != first[1] or not np.array_equal(y, first[2], equal_nan=True) or meta['variable'] != first[3]:
                raise ValueError('Legacy archives differ in image, phrase or target order/values.')
            banks.append(x)
            sources.append(dict(path=str(Path(source).resolve()), sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                                channel=channel, encoding={k: meta.get('encoding', {}).get(k) for k in ('model', 'revision', 'dtype')},
                                feature_identity=meta.get('feature_identity')))
            for j, text in enumerate(texts):
                matched = [g for g in groups if text in g['phrases']]
                roles = tuple(g['name'] for g in matched)
                columns.append(Column(f'{channel}:{j}', text, roles[0] if roles else f'item:{j}', channel, roles,
                                      sources=tuple(str(s) for g in matched for s in g.get('sources', []))))
    if first is None:
        raise ValueError('At least one archive required.')
    names, texts, y, target = first
    ids = tuple(Path(p).name for p in names)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate image basenames; supply a portable dataset with explicit IDs.')
    groups = np.asarray(ids if identity_groups is None else [identity_groups[i] for i in ids])
    return Dataset(np.concatenate(banks, axis=1), y, (target,), names, groups, Registry(tuple(columns)), ids,
                   dict(sources=sources, exposure='undeclared', group_note='Image IDs only unless identity groups supplied; does not rule out latent relatives.'))


def from_assets(assets, data, registry):
    """Reuse existing Assets scores and Ratings, with mandatory item provenance."""
    columns = registry.items
    expected = [(p, c) for c in ('global-short', 'local-box-max') for p in assets.bank['phrases']]
    if [(c.text, c.channel) for c in columns] != expected:
        raise ValueError('Registry must match global-short then local-box-max and the exact original bank.')
    return Dataset(assets.scores, np.where(data.mask, data.y, np.nan), tuple(data.targets),
                   tuple(str(p) for p in data.paths), data.groups, registry, tuple(p.name for p in data.paths),
                   dict(model=assets.engine.model_id, revision=assets.engine.revision, cache_key=assets.key,
                        exposure='undeclared', image_hashes=list(data.image_hashes)),
                   np.where(data.mask, data.se, np.nan))


def encode_candidate_phrases(assets, phrases, *, text_chunk=64, score_batch=512):
    """Score alternative wording with existing original Assets image features.

    Exact already-present phrases reuse score columns. Only unseen text is
    encoded; images are never re-encoded. Returns global-short then local-box-max
    scores and provenance. Does not mutate the original bank or its caches.
    """
    import torch
    phrases = tuple(phrases)
    if not phrases or len(set(phrases)) != len(phrases) or any(not p.strip() for p in phrases):
        raise ValueError('Provide unique nonempty candidate phrases.')
    if min(text_chunk, score_batch) < 1:
        raise ValueError('Positive chunk sizes required.')
    lookup = {p: i for i, p in enumerate(assets.bank['phrases'])}
    width, original_width = len(phrases), len(lookup)
    result = np.empty((len(assets.scores), 2*width), dtype=assets.scores.dtype)
    fresh = [i for i, p in enumerate(phrases) if p not in lookup]
    for i, phrase in enumerate(phrases):
        if phrase in lookup:
            result[:, [i, width+i]] = assets.scores[:, [lookup[phrase], original_width+lookup[phrase]]]
    if fresh:
        engine = assets.engine
        with torch.no_grad():
            new = [phrases[i] for i in fresh]
            texts = [torch.cat([engine.encode_text(new[start:start+text_chunk], mode=mode).float()
                                for start in range(0, len(new), text_chunk)]).to(engine.device)
                     for mode in ('short', 'box')]
            for start in range(0, len(result), score_batch):
                rows = np.arange(start, min(start+score_batch, len(result)))
                _, dense, mask, _ = assets.batch(rows)
                # Match original Assets global cache precision exactly.
                z = assets.features[0][rows].to(engine.device).float()
                result[np.ix_(rows, fresh)] = (z@texts[0].T).cpu().numpy()
                result[np.ix_(rows, np.asarray(fresh)+width)] = (dense@texts[1].T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values.cpu().numpy()
    return result, dict(phrases=phrases, channels=['global-short', 'local-box-max'],
                        model=assets.engine.model_id, revision=assets.engine.revision,
                        image_cache_key=assets.key, new_texts=len(fresh), newly_encoded_images=0)


def extend_dataset(data, scores, items, *, provenance=None, dependents=None):
    """Build a new union bank for matched alternative subsets; keep old intact."""
    scores = np.asarray(scores, float)
    if scores.shape != (len(data.x), len(items)):
        raise ValueError('Candidate scores/registry must align with original images.')
    metadata = dict(data.metadata)
    metadata['candidate_provenance'] = provenance or {}
    dependencies = dict(data.registry.dependents)
    for role, children in (dependents or {}).items():
        dependencies[role] = list(dict.fromkeys([*dependencies.get(role, []), *children]))
    return Dataset(np.column_stack((data.x, scores)), data.y.copy(), data.targets, data.paths,
                   data.groups.copy(), Registry((*data.registry.items, *items), dependencies), data.ids,
                   metadata, None if data.se is None else data.se.copy())


def legacy_predict(data, train, test, targets, spec, config):
    """Exact existing candidate_grid/inner_predictions/ensemble selection reuse."""
    legacy = _repo('race_perception')
    if (spec.transform != 'raw' or spec.balance_blocks or spec.blocks is not None
            or spec.metric_weights is not None or spec.contrast_pairs or spec.separate_metric):
        raise ValueError('Exact legacy comparisons accept raw score banks; use generic backend for transforms.')
    chosen = [data.registry.items[j] for j in spec.columns]
    channels = list(dict.fromkeys(c.channel for c in chosen))
    if not chosen:
        y = data.y[np.ix_(train, targets)]
        return np.tile(y.mean(0), (len(test), 1)), [dict(mean_only=True)]*len(targets)
    if len(channels) != 2:
        raise ValueError('Legacy readout needs two aligned channel blocks.')
    halves = [[j for j in spec.columns if data.registry.items[j].channel == c] for c in channels]
    if [data.registry.items[j].text for j in halves[0]] != [data.registry.items[j].text for j in halves[1]]:
        raise ValueError('Drop/replace all channel copies together for legacy evaluation.')
    x = data.x[:, halves[0]+halves[1]]  # Removal precedes any legacy row centering.
    recipes = [r for r in legacy.candidate_grid() if r['bank'] == 'both' and not r.get('centered')]
    y = data.y[np.ix_(train, targets)]
    inner, _ = legacy.inner_predictions(x[train], y, recipes, config.inner_folds, config.seed, data.groups[train])
    strategies, _, _ = legacy.select_strategies(inner, y, recipes)
    fitted = legacy.fit_bundle_readout(x[train], y, recipes, strategies)
    return legacy.predict_readout(fitted, x[test]), [dict(strategy=s, recipes=[recipes[j] for j in s['indices']]) for s in strategies]


def augmentation_scores(assets, rows, args, *, views=4, seed=0):
    """Reuse PreparedImages transforms; no TrainingViews disk banks or cleanup.

    Returned order: original row, view, global-short/local-box-max column. Keep
    the returned rows with their original identity in all downstream splits.
    """
    import torch
    module = _repo('fgclip2_multitarget.tensor_views')
    models = _repo('fgclip2_multitarget.models')
    rows = np.asarray(rows, int)
    if views < 2 or not len(rows):
        raise ValueError('At least two views and one image required.')
    prepared = module.PreparedImages(assets)
    result = np.empty((len(rows), views, assets.scores.shape[1]), np.float32)
    device = assets.engine.device
    global_text, box_text = (t.to(device).float() for t in assets.text)
    with torch.no_grad():
        for start in range(0, len(rows), args.encode_batch):
            chunk = rows[start:start+args.encode_batch]
            for view in range(views):
                # Each image/view seed is stable under encoding batch changes.
                segments = [(1, int(np.random.SeedSequence([seed, int(row), view]).generate_state(1)[0])) for row in chunk]
                inputs = prepared.batch(chunk, args, segments, enabled=torch.ones(len(chunk), device=device, dtype=torch.bool))
                z, dense, mask, _ = models.visual_features(assets.engine, inputs)
                global_scores = z.float()@global_text.T
                local_scores = (dense.float()@box_text.T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values
                result[start:start+len(chunk), view] = torch.cat((global_scores, local_scores), dim=1).cpu().numpy()
    return result, dict(rows=rows.tolist(), views=views, seed=seed, model=assets.engine.model_id,
                        revision=assets.engine.revision, settings={k: v for k, v in vars(args).items() if k.startswith('augmentation')},
                        label_preservation='Not assumed; declare cue/target-specific policies before scoring error.')
