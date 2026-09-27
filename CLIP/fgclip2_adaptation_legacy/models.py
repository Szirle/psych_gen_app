"""Small regression adaptations, not reproductions of the cited papers.

The frozen encoder is owned by the caller. Backbone edits are temporary and
restored by BackboneSession, including after failed fits.
"""
from contextlib import AbstractContextManager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


def level_prompts(targets, levels):
    words = ("not at all", "slightly", "moderately", "strongly", "extremely")
    if levels != 5:
        words = tuple(words[round(i * 4 / (levels - 1))] for i in range(levels))
    labels = {"native":"Native American", "islander":"Pacific Islander", "skin-color":"dark skinned"}
    return [f"a face perceived as {degree} {labels.get(target, target.replace('-', ' '))}"
            for target in targets for degree in words]


class LowRankLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x.float())).to(x.dtype)


class TrainableBlock(nn.Module):
    """Checkpoint late blocks; optionally use fp32 for unfrozen base weights."""
    def __init__(self, block, use_checkpoint, float_inputs=False):
        super().__init__()
        self.block, self.use_checkpoint, self.float_inputs = block, use_checkpoint, float_inputs

    def forward(self, x, attention_mask=None, **kwargs):
        value = x.float() if self.float_inputs else x
        mask = attention_mask.to(value.dtype) if attention_mask is not None else None
        def run(value):
            return self.block(value, mask, **kwargs)
        value = checkpoint(run, value, use_reentrant=False) if self.training and self.use_checkpoint else run(value)
        return value.to(x.dtype)


class BackboneSession(AbstractContextManager):
    def __init__(self, engine, kind, blocks, rank, use_checkpoint=True):
        self.engine, self.replacements, self.originals = engine, [], []
        model = engine.model
        model.eval().requires_grad_(False)
        layers = model.vision_model.encoder.layers
        if not 1 <= blocks <= len(layers):
            raise ValueError(f"blocks must be in [1, {len(layers)}]")
        if kind == "lora":
            for i in range(len(layers) - blocks, len(layers)):
                block = layers[i]
                for name in ("q_proj", "v_proj"):
                    parent, original = block.self_attn, getattr(block.self_attn, name)
                    self.replacements.append((parent, name, original))
                    setattr(parent, name, LowRankLinear(original, rank))
                self.replacements.append((layers, str(i), block))
                layers[i] = TrainableBlock(block, use_checkpoint)
        elif kind == "partial":
            for i in range(len(layers) - blocks, len(layers)):
                original = layers[i]
                for parameter in original.parameters():
                    self.originals.append((parameter, parameter.detach().cpu().clone(), parameter.dtype))
                    parameter.data = parameter.data.float()
                    parameter.requires_grad_(True)
                self.replacements.append((layers, str(i), original))
                layers[i] = TrainableBlock(original, use_checkpoint, float_inputs=True)
        elif kind == "pool":
            parameter = model.vision_model.head.probe
            self.originals.append((parameter, parameter.detach().cpu().clone(), parameter.dtype))
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            # MHA requires probe and patch dtypes to agree. Convert just at its input.
            self.probe_hook = model.vision_model.head.attention.register_forward_pre_hook(
                lambda module, inputs: (inputs[0].to(inputs[1].dtype), *inputs[1:]))

    def __exit__(self, *exc):
        if hasattr(self, "probe_hook"):
            self.probe_hook.remove()
        for parent, name, original in reversed(self.replacements):
            setattr(parent, name, original)
        with torch.no_grad():
            for parameter, value, dtype in self.originals:
                parameter.data = value.to(device=parameter.device, dtype=dtype)
        self.engine.model.eval().requires_grad_(False)


class SoftContext(nn.Module):
    """Shared soft tokens; native short-text positions, mask, and final pooling.

    Do not register the shared text tower here: only context is optimized/saved.
    """
    def __init__(self, engine, prompts, tokens, use_checkpoint=True):
        super().__init__()
        self.engine, self.use_checkpoint = engine, use_checkpoint
        inputs, _ = engine.prepare_text(prompts, mode="short")
        count = inputs["input_ids"].shape[1]
        if tokens + max(engine.text_info(prompts, mode="short")["token_counts"]) > count:
            raise ValueError("Soft context would truncate the prompt or EOS.")
        tower = engine.model.text_model
        with torch.no_grad():
            embeddings = tower.embeddings.token_embedding(inputs["input_ids"])
            # Prepend context, retain the native final padding position.
            tail = embeddings[:, :count-tokens].detach().clone()
            seed, _ = engine.prepare_text("a photo of a face", mode="short")
            initial = tower.embeddings.token_embedding(seed["input_ids"])[0, :tokens].float()
        self.context = nn.Parameter(initial.clone())
        self.register_buffer("tail", tail)
        mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
        self.register_buffer("mask", torch.cat([torch.ones(len(prompts), tokens, device=mask.device, dtype=mask.dtype),
                                                mask[:, :count-tokens]], dim=1))

    def forward(self):
        tower = self.engine.model.text_model
        outputs = []
        # Keep text activations bounded even with many targets.
        for start in range(0, len(self.tail), 4):
            tail = self.tail[start:start+4]
            tokens = self.context.to(tail.dtype).unsqueeze(0).expand(len(tail), -1, -1)
            h = tower.embeddings(inputs_embeds=torch.cat([tokens, tail], dim=1), use_short_position_ids=True)
            mask = _prepare_4d_attention_mask(self.mask[start:start+4], h.dtype)
            for block in tower.encoder.layers:
                h = checkpoint(block, h, mask, use_reentrant=False) if self.training and self.use_checkpoint else block(h, mask)
            outputs.append(tower.head(tower.final_layer_norm(h)[:, -1]).float())
        return F.normalize(torch.cat(outputs), dim=-1)


class RatingNetwork(nn.Module):
    def __init__(self, width, targets, method, *, rank=8, bottleneck=32, levels=5,
                 prototype_init=None, phrase_init=None, soft_context=None):
        super().__init__()
        self.method, self.targets, self.levels = method, targets, levels
        self.head = nn.Linear(width, targets)
        self.distribution = nn.Linear(width, targets * levels)
        self.order = nn.Linear(width, targets * 8, bias=False)
        self.register_buffer("anchors", torch.linspace(0, 1, levels))
        self.soft_context = soft_context
        if method == "adapter":
            self.adapter = nn.Sequential(nn.Linear(width, bottleneck, bias=False), nn.GELU(),
                                         nn.Linear(bottleneck, width, bias=False))
            nn.init.zeros_(self.adapter[-1].weight)
        if method in ("prototypes", "soft-prompt"):
            self.register_buffer("text0", prototype_init.clone())
            if soft_context is None:
                self.text_down = nn.Parameter(torch.randn(targets * levels, rank) * .01)
                self.text_up = nn.Parameter(torch.zeros(rank, width))
            self.log_temperature = nn.Parameter(torch.full((targets,), np.log(.07), dtype=torch.float32))
            self.correction = nn.Linear(width, targets)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)
        if method == "phrase-adapter":
            self.register_buffer("text0", phrase_init.clone())
            self.text_down = nn.Parameter(torch.randn(len(phrase_init), rank) * .01)
            self.text_up = nn.Parameter(torch.zeros(rank, width))
            self.phrase_head = nn.Sequential(nn.Linear(len(phrase_init), bottleneck), nn.GELU(), nn.Linear(bottleneck, targets))
        if method == "local":
            self.keys = nn.Linear(width, bottleneck, bias=False)
            self.values = nn.Linear(width, bottleneck, bias=False)
            self.queries = nn.Parameter(torch.randn(targets, bottleneck) * .02)
            self.local_weights = nn.Parameter(torch.randn(targets, bottleneck) * .02)

    def forward(self, x, dense=None, mask=None, text_override=None):
        z = F.normalize(x.float(), dim=-1)
        prior = z
        anchor_loss = z.new_zeros(())
        if self.method == "adapter":
            z = F.normalize(z + self.adapter(z), dim=-1)
        logits = self.distribution(z).reshape(-1, self.targets, self.levels)
        prediction = self.head(z)
        if self.method in ("prototypes", "soft-prompt", "phrase-adapter"):
            text = (text_override if text_override is not None else self.soft_context()) if self.soft_context else F.normalize(self.text0 + self.text_down @ self.text_up, dim=-1)
            anchor_loss = (text - self.text0).square().sum(-1).mean()
            scores = z @ text.T
            if self.method == "phrase-adapter":
                prediction = self.phrase_head(scores)
            else:
                logits = scores.reshape(-1, self.targets, self.levels) / self.log_temperature.exp().clamp(.01, 1)[None, :, None]
                prediction = (logits.softmax(-1) * self.anchors).sum(-1) + .1 * torch.tanh(self.correction(z))
        if self.method == "local":
            scores = torch.einsum("bpk,tk->btp", self.keys(dense.float()), self.queries) / self.queries.shape[1]**.5
            weights = scores.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
            local = weights @ self.values(dense.float())
            prediction = prediction + (local * self.local_weights).sum(-1)
        return prediction, logits, z, self.order(z).reshape(-1, self.targets, 8), anchor_loss, prior


def objective(outputs, y, se, hist, *, auxiliary, aux_weight, anchor_weight,
              preserve, preserve_weight, teacher=None):
    prediction, logits, z, order, anchor, prior = outputs
    loss = F.mse_loss(prediction, y) + anchor_weight * anchor
    if auxiliary == "distribution":
        # Ordered CDF loss plus consistency between distribution mean and mean head.
        probabilities = logits.softmax(-1)
        cdf = (probabilities.cumsum(-1) - hist.cumsum(-1)).square().mean()
        centers = torch.linspace(0, 1, logits.shape[-1], device=y.device)
        dist_mean = (probabilities * centers).sum(-1)
        loss = loss + aux_weight * (cdf + F.mse_loss(dist_mean, y) + F.mse_loss(dist_mean, prediction))
    elif auxiliary == "pairwise" and len(y) > 1:
        difference = y[:, None] - y[None, :]
        uncertainty = (se[:, None].square() + se[None, :].square() + 1e-6).sqrt()
        q = .5 * (1 + torch.erf(difference / (uncertainty * 2**.5)))
        pair_logits = (prediction[:, None] - prediction[None, :]) / .1
        pairs = torch.triu(torch.ones(len(y), len(y), device=y.device, dtype=torch.bool), diagonal=1)
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(pair_logits[pairs], q[pairs])
    elif auxiliary == "ordinal" and len(y) > 1:
        # Soft neighborhood matching inspired by continuous contrastive learning;
        # deliberately not labeled an exact ConOrd reproduction.
        representation = F.normalize(order, dim=-1).permute(1, 0, 2)
        similarity = representation @ representation.transpose(1, 2) / .1
        distance = (y.T[:, :, None] - y.T[:, None, :]).square()
        diagonal = torch.eye(len(y), device=y.device, dtype=torch.bool)[None]
        target = (-distance / .02).masked_fill(diagonal, -1e4).softmax(-1)
        log_probability = similarity.masked_fill(diagonal, -1e4).log_softmax(-1)
        loss = loss - aux_weight * (target * log_probability).sum(-1).mean()
    if preserve != "none":
        reference = F.normalize(teacher.float(), dim=-1) if teacher is not None else prior
        penalty = (1 - (z * reference).sum(-1)).mean() if preserve == "feature" else (z @ z.T - reference @ reference.T).square().mean()
        loss = loss + preserve_weight * penalty
    return loss
