"""Joint full-bank neural models. No fitted predictor supplies model inputs."""
from copy import copy
from argparse import BooleanOptionalAction
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import PhrasePool, loss_function, target_mean
from .phrases import hierarchy


SHARED_VARIANTS = {
    f'shared-{family}{suffix}': components
    for family in ('tabm', 'transformer')
    for suffix, components in (
        ('', ('head',)), ('-chain', ('head',)),
        ('-chain-pool', ('head', 'pool')),
        ('-chain-prompt', ('head', 'pool', 'prompt')),
        ('-chain-joint', ('head', 'pool', 'prompt', 'vision')),
    )
}


def add_arguments(parser):
    for family, lr, decay in (('tabm', .001, .0003), ('transformer', .0002, .01)):
        for name, default, typ in (('epochs', 800, int), ('batch-size', 64, int),
                                   ('lr', lr, float), ('adapt-lr', .0001, float), ('weight-decay', decay, float)):
            parser.add_argument(f'--{family}-{name}', type=typ, default=default)
    for name, default in (('tabm-width', 64), ('tabm-members', 32), ('tabm-blocks', 2),
                          ('tabm-residual-blocks', 1), ('transformer-width', 128),
                          ('transformer-blocks', 3), ('transformer-heads', 8),
                          ('transformer-decoder-blocks', 2), ('shared-patience', 80),
                          ('shared-vision-batch-size', 64)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--shared-dropout', type=float, default=.1)
    parser.add_argument('--shared-output-link', choices=('sigmoid', 'linear'), default='sigmoid')
    parser.add_argument('--shared-detach-chain', action='store_true', help='Detach earlier predictions; default allows joint gradients.')
    parser.add_argument('--shared-auxiliary-losses', action='store_true', help='Opt into distribution/ranking auxiliaries; default mean MSE only.')
    parser.add_argument('--shared-amp', action=BooleanOptionalAction, default=True,
                        help='BF16 autocast for shared neural heads on supported CUDA devices.')


def resolve_profile(args, variant):
    """Family options intentionally override old small-head/launcher defaults."""
    args = copy(args)
    family = 'tabm' if variant.startswith('shared-tabm') else 'transformer'
    args.epochs = getattr(args, family+'_epochs')
    adaptive = SHARED_VARIANTS[variant] != ('head',)
    args.head_lr = getattr(args, family+('_adapt_lr' if adaptive else '_lr'))
    args.weight_decay = getattr(args, family+'_weight_decay')
    args.batch_size = getattr(args, family+'_batch_size')
    args.accumulate, args.dropout, args.patience = 1, args.shared_dropout, args.shared_patience
    for field in ('epochs', 'patience',
                  'batch_size', 'shared_vision_batch_size', 'tabm_width', 'tabm_members',
                  'tabm_blocks', 'transformer_width', 'transformer_blocks', 'transformer_heads', 'transformer_decoder_blocks'):
        if getattr(args, field) < 1:
            raise ValueError(f'{field} must be positive')
    if args.tabm_residual_blocks < 0 or args.transformer_width % args.transformer_heads:
        raise ValueError('Residual block count must be nonnegative; transformer width must divide into heads')
    if not 0 <= args.dropout < 1:
        raise ValueError('Invalid shared dropout')
    if not math.isfinite(args.head_lr) or args.head_lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Invalid family learning rate or weight decay')
    if args.smoke:
        args.joint_epochs = 1
        args.batch_size = args.shared_vision_batch_size = 4
    if not args.shared_auxiliary_losses:
        args.distribution_weight = args.rank_weight = 0.
    fields = ('epochs', 'joint_epochs', 'head_lr', 'weight_decay',
              'batch_size', 'accumulate', 'dropout', 'patience', 'distribution_weight', 'rank_weight',
              'shared_vision_batch_size')
    args.serializable_config = dict(args.serializable_config, **{k: getattr(args, k) for k in fields})
    return args


class SharedAssets:
    """Use exact cached kernel-input scores during head training, without kernels."""
    def __init__(self, original):
        self.original, self.cached_head = original, True

    def __getattr__(self, name):
        return getattr(self.original, name)

    def batch(self, rows, live=False, augment=False):
        if self.cached_head:
            if augment:
                z, scores = self.original.score_batch(rows)
            else:
                z = F.normalize(self.features[0][rows].float().to(self.engine.device), dim=-1)
                scores = torch.as_tensor(self.scores[rows], device=self.engine.device, dtype=torch.float32)
            scores = scores.reshape(len(rows), 2, -1).transpose(1, 2)
            scores = torch.cat((scores, torch.zeros_like(scores[:, :, :1])), -1)
            return z, None, None, z.new_empty(len(rows), 0, 6), scores
        return (*self.original.batch(rows, live, augment), None)

    def statistics(self, rows, text):
        scores = self.scores[rows].reshape(len(rows), 2, -1).transpose(0, 2, 1)
        mean, std = scores.mean(0), scores.std(0, ddof=1)
        return tuple(torch.as_tensor(v, device=self.engine.device, dtype=torch.float32) for v in
                     (np.column_stack((mean, np.zeros(len(mean)))), np.column_stack((std, std[:, 1]))))


def full_selection(data, rows, bank, args):
    # No MI pruning, kernel/ridge reference, or label-derived model input.
    # Explicit groups override the predeclared visual -> basic -> race -> complex order.
    p, t = len(bank['phrases']), len(data.targets)
    return dict(phrases=list(bank['phrases']), owners=[-1]*p, bank_indices=list(range(p)),
        target_indices=[list(range(p)) for _ in range(t)], mi=[[None]*p for _ in range(t)],
        levels=hierarchy(data.targets, [0.]*t, args.hierarchy_spec, 'fixed'),
        sampling=[1/len(rows)]*len(rows), training_rows=rows.tolist(),
        policy='joint full-bank neural model; fixed hierarchy; uniform image sampling; no predictor inputs')


class SharedResidualBlock(nn.Module):
    """Pre-normalized, gated residual mixing shared by every task/member."""
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up, self.down = nn.Linear(width, 4*width), nn.Linear(2*width, width)
        self.dropout, self.gain = nn.Dropout(dropout), nn.Parameter(torch.full((width,), .1))

    def forward(self, x):
        a, b = self.up(self.norm(x)).chunk(2, -1)
        return x+self.gain*self.dropout(self.down(F.silu(a)*b))


class JointTabM(nn.Module):
    def __init__(self, features, targets, levels, args):
        super().__init__()
        import tabm
        width, k = args.tabm_width, args.tabm_members
        # Member-specific scaling is BEFORE the first feature-mixing layer.
        self.encoder = nn.Sequential(tabm.EnsembleView(k=k), tabm.make_tabm_backbone(
            d_in=features, n_blocks=args.tabm_blocks, d_block=width, dropout=args.dropout, k=k,
            arch_type='tabm', start_scaling_init='normal', start_scaling_init_chunks=[1]*features),
            *[SharedResidualBlock(width, args.dropout) for _ in range(args.tabm_residual_blocks)])
        self.output = tabm.LinearEnsemble(width, targets, k=k)
        self.context = nn.Sequential(nn.Linear(targets, width, bias=False), nn.SiLU(), nn.Linear(width, width, bias=False))
        self.levels, self.detach = levels, args.shared_detach_chain

    def forward(self, x, means, link):
        hidden = self.encoder(x.flatten(1))
        predictions = [None]*len(means)
        for level in self.levels:
            context = torch.stack([p if p is not None else means[t].expand(hidden.shape[:2])
                                   for t, p in enumerate(predictions)], -1)-means
            if self.detach:
                context = context.detach()
            values = link(self.output(hidden+self.context(context)))
            for t in level:
                predictions[t] = values[:, :, t]
        return torch.stack(predictions, -1)


class JointTransformer(nn.Module):
    """Full phrase-token encoder plus parallel or free-running target decoder."""
    def __init__(self, phrases, targets, levels, args):
        super().__init__()
        d, h = args.transformer_width, args.transformer_heads
        self.feature_weight = nn.Parameter(torch.randn(phrases, 3, d)/3**.5)
        self.feature_bias = nn.Parameter(torch.randn(phrases, d)*.02)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(d, h, 4*d, args.dropout,
            activation='gelu', batch_first=True, norm_first=True) for _ in range(args.transformer_blocks)])
        self.encoder_norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(targets, d)*.02)
        self.values = nn.Linear(1, d, bias=False)
        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(d, h, 4*d, args.dropout,
            activation='gelu', batch_first=True, norm_first=True) for _ in range(args.transformer_decoder_blocks)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.bias = nn.Parameter(torch.zeros(targets))
        self.levels, self.detach = levels, args.shared_detach_chain

    def forward(self, x, means, link):
        memory = (x.unsqueeze(-1)*self.feature_weight).sum(-2)+self.feature_bias
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        predictions, prefix, group_ids = [None]*len(means), [], []
        for group, level in enumerate(self.levels):
            prefix.extend(level)
            group_ids.extend([group]*len(level))
            values = torch.stack([predictions[t]-means[t] if predictions[t] is not None
                                  else means[t].new_zeros(len(x)) for t in prefix], -1)
            if self.detach:
                values = values.detach()
            queries = self.queries[prefix][None]+self.values(values.unsqueeze(-1))
            order = torch.tensor(group_ids, device=x.device)
            # Previous groups cannot see later groups. Current-group targets
            # may share query states but never receive current/future ratings.
            mask = order[None, :] > order[:, None]
            for layer in self.decoder:
                queries = layer(queries, memory, tgt_mask=mask)
            result = link(self.output(queries).squeeze(-1)+self.bias[prefix])
            for j, t in enumerate(level, start=len(prefix)-len(level)):
                predictions[t] = result[:, j]
        return torch.stack(predictions, -1)[:, None]


class SharedNeural(nn.Module):
    def __init__(self, width, selection, targets, means, stats, prompt, *, args, variant):
        super().__init__()
        self.prompt, self.cached_head, self.fixed_text = prompt, False, None
        p = len(selection['phrases'])
        self.pool = PhrasePool(width, p, args.pool_rank)
        self.register_buffer('means', means)
        self.register_buffer('score_mean', stats[0])
        self.register_buffer('score_std', stats[1].clamp_min(.005))
        self.levels = selection['levels'] if '-chain' in variant else [list(range(targets))]
        self.family = 'tabm' if variant.startswith('shared-tabm') else 'transformer'
        model = JointTabM(3*p, targets, self.levels, args) if self.family == 'tabm' else JointTransformer(p, targets, self.levels, args)
        self.encoders = nn.ModuleList([model])
        self.heads = nn.ModuleList([nn.Linear(targets, targets*args.bins)])
        self.bins, self.link_name = args.bins, args.shared_output_link
        self.distribution_supervised = bool(args.distribution_weight)
        self.amp = args.shared_amp and means.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        with torch.no_grad():
            offset = torch.logit(means.clamp(.01, .99)) if self.link_name == 'sigmoid' else means
            if self.family == 'tabm':
                model.output.bias.copy_(offset.expand_as(model.output.bias))
                model.output.weight.mul_(.1)
            else:
                model.bias.copy_(offset)
                model.output[-1].weight.mul_(.1)
                model.output[-1].bias.zero_()

    def prediction_text(self):
        if self.cached_head:
            return self.prompt.initial_short, self.prompt.initial_box
        return self.fixed_text if self.fixed_text is not None else self.prompt()

    def members(self, scores):
        link = torch.sigmoid if self.link_name == 'sigmoid' else lambda x: x
        with torch.autocast(device_type=self.means.device.type, dtype=torch.bfloat16, enabled=self.amp):
            return self.encoders[0]((scores-self.score_mean)/self.score_std, self.means, link).float()

    def decode(self, scores):
        prediction = self.members(scores).mean(1)
        return prediction, self.heads[0](prediction).reshape(len(scores), -1, self.bins)

    def forward(self, features, *, text=None, learned_pool=False):
        z, dense, mask, coordinates, cached = features
        embeddings = self.prediction_text() if text is None else text
        if cached is not None:
            scores, attention = cached, z.new_empty(len(z), 0, cached.shape[1])
        else:
            maximum, attention = self.pool(dense, mask, coordinates, embeddings[1], False)
            local = maximum
            if learned_pool:
                local, attention = self.pool(dense, mask, coordinates, embeddings[1], True)
            # Preserve global and local MAX inputs; learned pooling is additional.
            scores = torch.stack((z.float()@embeddings[0].T, maximum, local-maximum), -1)
        members = self.members(scores)
        prediction = members.mean(1)
        token, anchor = self.prompt.penalties(embeddings)
        return dict(prediction=prediction, members=members, scores=scores, attention=attention,
            distribution=self.heads[0](prediction).reshape(len(z), -1, self.bins), token=token, anchor=anchor, z=z)


def shared_loss(output, y, mask, se, hist, *, teacher, args):
    loss, mean = loss_function(output, y, mask, se, hist, teacher=teacher, args=args)
    # TabM minimizes mean member loss, NOT loss of the ensemble mean.
    member_loss = target_mean((output['members']-y[:, None]).square().mean(1), mask)
    return loss-mean+member_loss, member_loss


@torch.no_grad()
def predict_shared_neural(packet, engine, text, paths, batch_size):
    from types import SimpleNamespace
    from ..fgclip2_adaptation_legacy.models import BackboneSession
    from .assets import image_inputs
    from .models import AnchoredTokens, visual_features
    from .train import restore

    args, variant = SimpleNamespace(**packet['config']), packet['variant']
    selection, saved = packet['selection'], packet['state']['network']
    kind = 'lora' if variant.endswith('-joint') else 'frozen'
    with BackboneSession(engine, kind, args.blocks, args.rank):
        backbone = [(n, p) for n, p in engine.model.named_parameters() if p.requires_grad]
        prompt = AnchoredTokens(engine, selection['phrases'], selection['owners'], len(packet['targets']), text,
                                args.context_tokens, args.prompt_rank, args.text_chunk)
        network = SharedNeural(text[0].shape[-1], selection, len(packet['targets']), saved['means'].to(engine.device),
            (saved['score_mean'].to(engine.device), saved['score_std'].to(engine.device)), prompt,
            args=args, variant=variant).to(engine.device)
        restore(packet['state'], network, backbone)
        network.cached_head = tuple(packet['components']) == ('head',)
        network.eval().requires_grad_(False)
        engine.model.eval().requires_grad_(False)
        learned_text, result = network.prediction_text(), []
        for i in range(0, len(paths), batch_size):
            z, dense, mask, coordinates = visual_features(engine, image_inputs(engine, paths[i:i+batch_size], args.patches))
            cached = None
            if kind == 'frozen':
                z, dense, coordinates = z.half().float(), F.normalize(dense.half().float(), dim=-1), coordinates.half().float()
                if network.cached_head:
                    # Exact fixed-feature ordering/quantization of Assets.scores.
                    global_scores = torch.as_tensor(z.cpu().numpy()@text[0].cpu().numpy().T, device=engine.device)
                    local = (dense@text[1].T).masked_fill(~mask[:, :, None], -torch.inf).max(1).values
                    cached = torch.stack((global_scores, local, torch.zeros_like(local)), -1)
                z = F.normalize(z, dim=-1)
            output = network((z, dense, mask, coordinates, cached), text=learned_text,
                             learned_pool='pool' in packet['components'])
            result.append(output['prediction'].cpu().numpy())
    return tuple(packet['targets']), np.concatenate(result)
