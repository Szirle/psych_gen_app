"""Full-bank kernel reference with shared, additive neural corrections.

Every target keeps every frozen global/local score. Learned spatial/language/
visual evidence and earlier predictions add inputs; they never replace them.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..fgclip2_face_impressions import array_hash
from ..race_perception import predict_readout
from .assets import atomic_save
from .data import split_rows
from .models import PhrasePool
from .phrases import hierarchy


def reference_predictions(readouts, scores, targets):
    prediction = np.zeros((len(scores), targets))
    for part in readouts:
        prediction[:, part['targets']] = predict_readout(part['readout'], scores)
    return prediction


class ReferenceAssets:
    """Decorate batch inputs; the frozen cache remains owned by Assets."""
    def __init__(self, assets, baseline, readouts):
        self.original, self.baseline, self.readouts = assets, baseline, readouts

    def __getattr__(self, name):
        return getattr(self.original, name)

    def batch(self, rows, live=False, augment=False):
        features = self.original.batch(rows, live, augment)
        frozen = torch.as_tensor(self.scores[rows], device=self.engine.device, dtype=torch.float32)
        frozen = frozen.reshape(len(rows), 2, -1).transpose(1, 2)
        baseline = torch.as_tensor(self.baseline[rows], device=self.engine.device, dtype=torch.float32)
        return (*features, baseline, frozen)

    def statistics(self, rows, text):
        scores = self.scores[rows].reshape(len(rows), 2, -1).transpose(0, 2, 1)
        return tuple(torch.as_tensor(v, device=self.engine.device, dtype=torch.float32)
                     for v in (scores.mean(0), scores.std(0, ddof=1)))


def prepare_reference(assets, data, rows, bank, args):
    """Strict cross-fitting: each reference's recipe search excludes its fold.

    The full training fit predicts validation/test images. During residual
    training, use nested OOF kernel predictions, never in-sample fitted values.
    Reference caches are shared by all new variants for an identical split.
    """
    from .train import kernel_readouts
    code = Path(__file__).read_text()+Path(__file__).with_name('train.py').read_text()
    key = array_hash(np.asarray([assets.key, code, str(args.seed), str(args.inner_folds), str(args.smoke)]),
                     rows, data.y[rows], data.mask[rows], data.groups[rows],
                     np.asarray(bank['phrases']), np.asarray(data.targets))
    path = Path(args.cache)/f'kernel-reference-{key}.pt'
    cached = getattr(assets, '_kernel_reference_cache', None)
    if cached is not None and cached[0] == key:
        reference = cached[1]
    elif path.exists() and not args.smoke:
        reference = torch.load(path, map_location='cpu', weights_only=False)
    else:
        print(f'Full-bank kernel reference: {len(rows)} training images; nested cross-fitting…', flush=True)
        predictions, readouts = kernel_readouts(assets, data, rows, np.arange(len(data.y)), args)
        for fold, (train, valid) in enumerate(split_rows(data, rows, args.inner_folds, args.seed+19)):
            predictions[valid], _ = kernel_readouts(assets, data, train, valid, args)
            print(f'  Kernel reference OOF fold {fold+1}/{args.inner_folds}', flush=True)
        reference = dict(baseline=predictions, readouts=readouts)
        if not args.smoke:
            atomic_save(reference, path)
    assets._kernel_reference_cache = (key, reference)
    errors = (reference['baseline'][rows]-data.y[rows])**2
    skill = [float(1-errors[data.mask[rows, t], t].mean()/max(data.y[rows[data.mask[rows, t]], t].var(), 1e-6))
             for t in range(len(data.targets))]
    difficulty = ((errors/(data.se[rows]**2+.01))*data.mask[rows]).sum(1)/np.maximum(data.mask[rows].sum(1), 1)
    difficulty = np.minimum(difficulty, np.quantile(difficulty, .9))+1e-6
    sampling = (1-args.hard_fraction)/len(rows)+args.hard_fraction*difficulty/difficulty.sum()
    width, targets = len(bank['phrases']), len(data.targets)
    selection = dict(phrases=list(bank['phrases']), owners=[-1]*width, bank_indices=list(range(width)),
        target_indices=[list(range(width)) for _ in range(targets)], mi=[[None]*width for _ in range(targets)],
        skill=skill, levels=hierarchy(data.targets, skill, args.hierarchy_spec, args.order),
        sampling=sampling.tolist(), training_rows=rows.tolist(),
        oof_mse=[float(errors[data.mask[rows, t], t].mean()) for t in range(targets)],
        policy='full bank for every target; nested OOF kernel reference; no MI pruning')
    return ReferenceAssets(assets, reference['baseline'], reference['readouts']), selection


class KernelResidualDAG(nn.Module):
    """Kernel + zero-initialized direct linear/shared nonlinear corrections.

    No sigmoid: zero correction reproduces the kernel's original rating units.
    Chaining concatenates previous predicted ratings to the shared evidence.
    Frozen scores have a direct linear path to every target as well as the MLP.
    """
    def __init__(self, width, selection, targets, means, stats, prompt, *, hidden=256,
                 bins=9, rank=32, independent=False, dropout=.15, detach_context=True):
        super().__init__()
        self.prompt = prompt
        phrases = len(selection['phrases'])
        self.pool = AdditivePhrasePool(width, phrases, rank)
        self.levels = [list(range(targets))] if independent else selection['levels']
        self.detach_context, self.context_dropout = detach_context, dropout
        self.register_buffer('means', means)
        self.register_buffer('score_mean', stats[0])
        self.register_buffer('score_std', stats[1].clamp_min(.005))
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(2*phrases, hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden), nn.GELU()),
            nn.Linear(2*phrases, hidden, bias=False),
        ])
        self.direct = nn.Linear(2*phrases, targets, bias=False)
        self.heads = nn.ModuleList([nn.Linear(hidden+targets, bins+1) for _ in range(targets)])
        nn.init.zeros_(self.direct.weight)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def decode(self, scores, *, baseline, frozen_scores):
        frozen = ((frozen_scores-self.score_mean)/self.score_std).flatten(1)
        change = ((scores-frozen_scores)/self.score_std).flatten(1)
        shared = self.encoders[0](frozen)+self.encoders[1](change)
        direct = self.direct(frozen)
        predictions, distributions = [None]*len(self.heads), [None]*len(self.heads)
        for level in self.levels:
            earlier = torch.stack([p if p is not None else self.means[t].expand(len(scores))
                                   for t, p in enumerate(predictions)], -1)-self.means
            if self.detach_context:
                earlier = earlier.detach()
            earlier = F.dropout(earlier, self.context_dropout, training=self.training)
            inputs = torch.cat((shared, earlier), -1)
            for t in level:
                output = self.heads[t](inputs)
                predictions[t] = baseline[:, t]+direct[:, t]+output[:, 0]
                distributions[t] = output[:, 1:]
        return torch.stack(predictions, -1), torch.stack(distributions, 1)

    def forward(self, features, *, text=None, learned_pool=False):
        z, dense, mask, coordinates, baseline, frozen = features
        embeddings = self.prompt() if text is None else text
        local, attention = self.pool(dense, mask, coordinates, embeddings[1], learned_pool)
        scores = torch.stack((z.float()@embeddings[0].T, local), -1)
        prediction, distribution = self.decode(scores, baseline=baseline, frozen_scores=frozen)
        token, anchor = self.prompt.penalties(embeddings)
        return dict(prediction=prediction, distribution=distribution, scores=scores, attention=attention,
                    token=token, anchor=anchor, z=z, baseline=baseline, frozen_scores=frozen)


class AdditivePhrasePool(PhrasePool):
    """A zero-gated pooling residual preserves max scores at initialization."""
    def __init__(self, width, phrases, rank):
        super().__init__(width, phrases, rank)
        self.residual_gain = nn.Parameter(torch.zeros(phrases))

    def forward(self, dense, mask, coordinates, text, learned):
        maximum, max_weights = super().forward(dense, mask, coordinates, text, False)
        if not learned:
            return maximum, max_weights
        pooled, weights = super().forward(dense, mask, coordinates, text, True)
        gain = self.residual_gain.tanh()
        # Effective weights can be signed: these are residual contributions,
        # not a probability distribution or causal attribution.
        return maximum+gain*(pooled-maximum), max_weights+gain*(weights-max_weights)


@torch.no_grad()
def predict_kernel_residual(packet, engine, text, paths, batch_size):
    """Compute the original kernel inputs before attaching visual adaptation."""
    from ..fgclip2_adaptation_legacy.models import BackboneSession
    from .assets import image_inputs
    from .models import AnchoredTokens, visual_features
    from .train import restore

    config, selection, state = packet['config'], packet['selection'], packet['state']
    targets, variant = packet['targets'], packet['variant']
    cached, scores = [], []
    for i in range(0, len(paths), batch_size):
        values = visual_features(engine, image_inputs(engine, paths[i:i+batch_size], config['patches']))
        z, dense, mask, coordinates = (v if v.dtype == torch.bool else v.half().float() for v in values)
        local = (F.normalize(dense, dim=-1)@text[1].T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values
        scores.append(torch.cat((z@text[0].T, local), -1).cpu().numpy())
        cached.append(tuple(v.cpu() if v.dtype == torch.bool else v.half().cpu() for v in (z, dense, mask, coordinates)))
    scores = np.concatenate(scores)
    baseline = reference_predictions(packet['kernel_readouts'], scores, len(targets))
    kind = 'lora' if variant == 'kernel-chain-joint' else 'frozen'
    with BackboneSession(engine, kind, config['blocks'], config['rank']):
        backbone = [(n, p) for n, p in engine.model.named_parameters() if p.requires_grad]
        prompt = AnchoredTokens(engine, selection['phrases'], selection['owners'], len(targets), text,
                                config['context_tokens'], config['prompt_rank'], config['text_chunk'])
        saved = state['network']
        network = KernelResidualDAG(text[0].shape[-1], selection, len(targets), saved['means'].to(engine.device),
            (saved['score_mean'].to(engine.device), saved['score_std'].to(engine.device)), prompt,
            hidden=config['kernel_hidden'], bins=config['bins'], rank=config['pool_rank'],
            independent=variant == 'kernel-residual', dropout=config['dropout'],
            detach_context=not config['chain_gradients']).to(engine.device)
        restore(state, network, backbone)
        network.eval().requires_grad_(False)
        engine.model.eval().requires_grad_(False)
        learned_text, result = prompt(), []
        for b, i in enumerate(range(0, len(paths), batch_size)):
            if kind == 'lora':
                values = visual_features(engine, image_inputs(engine, paths[i:i+batch_size], config['patches']))
            else:
                z, dense, mask, coordinates = (v.to(engine.device) for v in cached[b])
                values = F.normalize(z.float(), dim=-1), F.normalize(dense.float(), dim=-1), mask, coordinates.float()
            raw = torch.as_tensor(scores[i:i+batch_size], device=engine.device, dtype=torch.float32)
            raw = raw.reshape(len(raw), 2, -1).transpose(1, 2)
            base = torch.as_tensor(baseline[i:i+batch_size], device=engine.device, dtype=torch.float32)
            output = network((*values, base, raw), text=learned_text, learned_pool='pool' in packet['components'])
            result.append(output['prediction'].cpu().numpy())
    return tuple(targets), np.concatenate(result)
