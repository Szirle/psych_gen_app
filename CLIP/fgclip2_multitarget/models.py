"""Anchored language, phrase-conditioned spatial filters, and a rating DAG."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


def visual_features(engine, inputs):
    """One visual forward, equivalent to native global and dense feature paths.

    The pinned FG-CLIP2 global head is attention pooling, not max pooling. The
    max/learned-pooling ablation below operates on local *phrase alignment maps*.
    """
    model = engine.model
    mask = inputs['pixel_attention_mask']
    output = model.vision_model(pixel_values=inputs['pixel_values'], attention_mask=mask,
                                spatial_shapes=inputs['spatial_shapes'])
    hidden, head = output.last_hidden_state, model.dense_feature_head
    attention_mask = _prepare_4d_attention_mask(mask, hidden.dtype, hidden.shape[1])
    attention_mask = attention_mask.repeat(1, head.num_heads, 1, 1).reshape(-1, hidden.shape[1], hidden.shape[1])
    # need_weights=False changes the implementation, not the native pooling rule.
    local = head.attention(hidden, hidden, hidden, attn_mask=attention_mask, need_weights=False)[0]
    local = local + head.mlp(head.layernorm(local))
    coordinates = hidden.new_zeros((*mask.shape, 6), dtype=torch.float32)
    for i, (h, w) in enumerate(inputs['spatial_shapes'].tolist()):
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=hidden.device),
                                torch.linspace(-1, 1, w, device=hidden.device), indexing='ij')
        x, y = xx.flatten(), yy.flatten()
        coordinates[i, :h*w] = torch.stack((x, y, x*x, y*y, x*y, torch.ones_like(x)), -1)
    return (F.normalize(output.pooler_output.float(), dim=-1),
            F.normalize(local.float(), dim=-1), mask.bool(), coordinates)


class AnchoredTokens(nn.Module):
    """Delta=0 reproduces the actual hand-written tokens, without added tokens.

    Shared deltas serve every phrase; target deltas only its private phrases.
    A low-rank phrase residual gives selected paraphrases limited independence.
    Both short/global and box/local embeddings come from the same text forward.
    """
    def __init__(self, engine, phrases, owners, targets, initial, tokens=4, rank=8, chunk=16):
        super().__init__()
        self.engine, self.chunk = engine, chunk
        inputs, _ = engine.prepare_text(phrases, mode='short')
        ids, mask = inputs['input_ids'], inputs.get('attention_mask')
        with torch.no_grad():
            base = engine.model.text_model.embeddings.token_embedding(ids).detach().clone()
        # Native FG-CLIP2 leaves attention unmasked when the tokenizer omits
        # this field. Padding still participates in attention/final pooling;
        # special tokens are excluded only from learned token displacement.
        valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
        for special in engine.tokenizer.all_special_ids:
            valid = valid & (ids != special)
        valid = valid & (valid.cumsum(1) <= tokens)
        slots = (valid.cumsum(1)-1).clamp(min=0, max=tokens-1)
        self.register_buffer('base', base, persistent=False)
        self.register_buffer('mask', mask, persistent=False)
        self.register_buffer('valid', valid, persistent=False)
        self.register_buffer('slots', slots, persistent=False)
        self.register_buffer('owners', torch.tensor(owners, device=base.device), persistent=False)
        self.register_buffer('initial_short', initial[0].clone(), persistent=False)
        self.register_buffer('initial_box', initial[1].clone(), persistent=False)
        d = base.shape[-1]
        self.shared = nn.Parameter(torch.zeros(tokens, d))
        self.target = nn.Parameter(torch.zeros(targets, tokens, d))
        self.phrase = nn.Parameter(torch.zeros(len(phrases), tokens, rank))
        self.basis = nn.Parameter(torch.randn(rank, d)*.01)

    def deltas(self):
        return (self.shared[None] + self.target[self.owners.clamp_min(0)]*(self.owners >= 0)[:, None, None]
                + self.phrase @ self.basis)

    def forward(self):
        tower = self.engine.model.text_model
        delta, short, box = self.deltas(), [], []
        for start in range(0, len(self.base), self.chunk):
            end = start+self.chunk
            shift = delta[start:end].gather(1, self.slots[start:end, :, None].expand(-1, -1, delta.shape[-1]))
            shift = shift*self.valid[start:end, :, None]
            embeds = self.base[start:end] + shift.to(self.base.dtype)
            h = tower.embeddings(inputs_embeds=embeds, use_short_position_ids=True)
            mask = None
            implementation = getattr(getattr(tower, 'config', None), '_attn_implementation', '') or ''
            if self.mask is not None and 'flash' not in implementation:
                mask = _prepare_4d_attention_mask(self.mask[start:end], h.dtype)
            for block in tower.encoder.layers:
                h = checkpoint(block, h, mask, use_reentrant=False) if torch.is_grad_enabled() and delta.requires_grad else block(h, mask)
            pooled = tower.final_layer_norm(h)[:, -1]
            # Match get_text_features, including its distinct local projection.
            # The native short path applies its head one phrase at a time.
            short.append(F.normalize(torch.cat([tower.head(row) for row in pooled.split(1)]).float(), dim=-1))
            box.append(F.normalize(self.engine.model.boxtext_head(pooled).float(), dim=-1))
        return torch.cat(short), torch.cat(box)

    def penalties(self, embeddings):
        # Actual displacement from original word vectors, not decay toward zero
        # of their pretrained embeddings. Native embeddings never get optimized.
        scale = self.base[self.valid].float().square().mean().clamp_min(1e-6)
        token = self.deltas().square().mean()/scale
        feature = ((1-(embeddings[0]*self.initial_short).sum(-1)).mean()
                   + (1-(embeddings[1]*self.initial_box).sum(-1)).mean())/2
        return token, feature


class PhrasePool(nn.Module):
    def __init__(self, width, phrases, rank):
        super().__init__()
        self.key = nn.Linear(width, rank, bias=False)
        self.query = nn.Parameter(torch.randn(phrases, rank)*.01)
        self.spatial = nn.Parameter(torch.zeros(phrases, 6))
        self.temperature = nn.Parameter(torch.full((phrases,), math.log(.07)))

    def forward(self, dense, mask, coordinates, text, learned):
        alignment = dense.float() @ text.T
        if learned:
            logits = (alignment/self.temperature.exp().clamp(.02, .5)
                      + self.key(dense.float()) @ self.query.T / self.query.shape[1]**.5
                      + coordinates @ self.spatial.T)
            weights = logits.masked_fill(~mask[:, :, None], -torch.inf).softmax(1)
            pooled = (weights*alignment).sum(1)
        else:
            values = alignment.masked_fill(~mask[:, :, None], -torch.inf)
            pooled, indices = values.max(1)
            weights = F.one_hot(indices, dense.shape[1]).transpose(1, 2).to(dense.dtype)
        return pooled, weights


class RatingDAG(nn.Module):
    """Sequential continuous predictions; no ground-truth context input exists."""
    def __init__(self, width, selection, targets, means, stats, prompt, *, hidden=64, bins=9,
                 rank=16, independent=False, dropout=.15, detach_context=True):
        super().__init__()
        self.prompt = prompt
        self.pool = PhrasePool(width, len(selection['phrases']), rank)
        self.levels = [list(range(targets))] if independent else selection['levels']
        self.detach_context, self.context_dropout = detach_context, dropout
        self.indices = selection['target_indices']
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(2*len(ids), hidden), nn.GELU(),
                                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
                                       for ids in self.indices])
        self.context = nn.Linear(targets, hidden, bias=False)
        self.heads = nn.ModuleList([nn.Linear(hidden, bins+1) for _ in range(targets)])
        self.register_buffer('means', means)
        self.register_buffer('score_mean', stats[0])
        self.register_buffer('score_std', stats[1].clamp_min(.005))
        self.register_buffer('anchors', torch.linspace(0, 1, bins))
        with torch.no_grad():
            for t, head in enumerate(self.heads):
                head.weight.mul_(.01)
                head.bias.zero_()
                head.bias[0] = torch.logit(means[t].clamp(.01, .99))

    def decode(self, scores):
        batch, targets = len(scores), len(self.heads)
        scaled = (scores-self.score_mean)/self.score_std
        predictions, distributions = [None]*targets, [None]*targets
        for level in self.levels:
            earlier = torch.stack([p if p is not None else self.means[t].expand(batch)
                                   for t, p in enumerate(predictions)], -1)
            if self.detach_context:
                earlier = earlier.detach()
            context = earlier-self.means
            if self.training:
                context = F.dropout(context, self.context_dropout)
            latent_context = self.context(context)
            # All nodes in this level see the same earlier-level predictions.
            for t in level:
                h = self.encoders[t](scaled[:, self.indices[t]].flatten(1)) + latent_context
                output = self.heads[t](h)
                predictions[t], distributions[t] = output[:, 0].sigmoid(), output[:, 1:]
        return torch.stack(predictions, -1), torch.stack(distributions, 1)

    def forward(self, features, *, text=None, learned_pool=False):
        z, dense, mask, coordinates = features
        embeddings = self.prompt() if text is None else text
        local, attention = self.pool(dense, mask, coordinates, embeddings[1], learned_pool)
        scores = torch.stack((z.float() @ embeddings[0].T, local), -1)
        prediction, distribution = self.decode(scores)
        token, anchor = self.prompt.penalties(embeddings)
        return dict(prediction=prediction, distribution=distribution, scores=scores,
                    attention=attention, token=token, anchor=anchor, z=z)


def target_mean(values, mask):
    count = mask.sum(0)
    return ((values*mask).sum(0)/count.clamp_min(1))[count > 0].mean()


def loss_function(output, y, mask, se, hist, *, teacher, args):
    prediction = output['prediction']
    mean = target_mean((prediction-y).square(), mask)
    p = output['distribution'].softmax(-1)
    cdf = target_mean((p.cumsum(-1)-hist.cumsum(-1)).square().mean(-1), mask)
    centers = torch.linspace(0, 1, hist.shape[-1], device=y.device)
    consistency = target_mean(((p*centers).sum(-1)-prediction).square(), mask)
    loss = mean + args.distribution_weight*(cdf+consistency)
    if args.rank_weight and len(y) > 1:
        difference = y[:, None]-y[None, :]
        uncertainty = (se[:, None].square()+se[None, :].square()+1e-6).sqrt()
        q = .5*(1+torch.erf(difference/(uncertainty*2**.5)))
        pairs = torch.triu(torch.ones(len(y), len(y), device=y.device, dtype=torch.bool), diagonal=1)
        valid = (mask[:, None] & mask[None, :])[pairs]
        pair_loss = F.binary_cross_entropy_with_logits((prediction[:, None]-prediction[None, :])[pairs]/.1, q[pairs], reduction='none')
        if valid.any():
            loss = loss + args.rank_weight*target_mean(pair_loss, valid)
    preserve = (1-(F.normalize(output['z'].float(), dim=-1)*teacher).sum(-1)).mean()
    return (loss + args.token_anchor*output['token'] + args.feature_anchor*output['anchor']
            + args.preserve_weight*preserve), mean
