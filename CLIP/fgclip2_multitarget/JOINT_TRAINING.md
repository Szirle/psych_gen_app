# Joint optimization of ratings, prompts and pooling

All neural variants in this package now use one joint optimization loop. The
variant controls which components exist as trainable inputs to that loop;
there is no heads → pooling → prompts → vision schedule. The frozen kernel
reference and the original architecture/input differences remain unchanged.

## Research basis

- [CoOp, §3.2](https://arxiv.org/html/2109.01134v6): train continuous context
  tokens using the classification loss, backpropagating through a frozen text
  encoder. The paper does not first fit a predictor and then learn prompts.
  It also does not introduce our separate regression head: jointly training
  that head is our task-specific extension.
- [CoCoOp](https://arxiv.org/abs/2203.05557) jointly optimizes context tokens and
  its image-conditioned prompt network through the task objective. The
  [official trainer](https://github.com/KaiyangZhou/CoOp/blob/main/trainers/cocoop.py)
  puts both in the same optimizer. Our prompts remain image-independent;
  this change does not implement CoCoOp's conditioning network.
- [MaPLe's official trainer](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/trainers/maple.py)
  optimizes its coupled multimodal prompt parameters together. Our visual
  adaptation uses LoRA or partial block unfreezing, not MaPLe's visual prompts.
- [PromptSRC](https://arxiv.org/html/2307.06948v2) combines task supervision
  and preservation constraints in the same optimization. We retain token
  displacement, text-feature cosine and visual-feature preservation losses.
  We now also distill original image–phrase agreement. This is an adaptation
  for regression: cosine feature anchors replace its L1 feature losses, and
  phrase agreement replaces class logits. Gaussian prompt averaging and
  its textual-template ensemble are not implemented.
- [Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html) provides
  a precedent for learning aggregation from image-level supervision in an
  end-to-end network. Separate spatial labels or a separate pooling fit are
  not required by our differentiable pooling operator.

These sources support joint task-driven adaptation while keeping pretrained
weights fixed when appropriate. They do not prove that joint fitting always
beats staged fitting, or that these settings improve our held-out ratings.

## Implemented behavior

`*-pool` optimizes heads and pooling together; `*-prompt` also optimizes tokens;
`*-joint` also optimizes visual LoRA. `chain-partial` jointly optimizes tokens,
pooling, heads and the selected visual blocks. The redundant `chain-lora` alias
and all staged-budget options have been removed. Fixed-feature variants train
only their predictors. The launcher and Python CLI default to `frozen-kernel`
plus all ten shared TabM/transformer variants. Legacy heads and kernel-residual
architectures remain explicit opt-ins.

All enabled components participate from the first minibatch in one AdamW
optimizer, with a common warmup/cosine schedule and separate base rates:

| Parameter group | Default base learning rate | Matrix weight decay |
| --- | --- | --- |
| Legacy/kernel neural heads | 0.0001 | 0.01 |
| Shared TabM, fixed inputs | 0.0003 (`--tabm-lr`) | 0.0003 |
| Shared transformer, fixed inputs | 0.0002 (`--transformer-lr`) | 0.01 |
| Shared TabM / transformer, adaptive inputs | 0.0001 (`--tabm-adapt-lr` / `--transformer-adapt-lr`) | Family decay |
| Spatial pooling | 0.0001 (`--pool-lr`) | 0 |
| Prompt token deltas | 0.00002 (`--prompt-lr`) | 0; explicit anchor penalties instead |
| Visual LoRA | 0.00001 (`--encoder-lr`) | 0 |
| Partially unfrozen vision | 0.000001 (`--partial-lr`) | 0.01 (`--vision-weight-decay`) |

Biases, normalization parameters, ensemble scales and embedding identities are
excluded from decay. These are conservative research defaults, not measured
optima. See the [training audit](TRAINING_AUDIT.md) for the reasoning and limits.

Pooling starts with **zero learned logit corrections** to the existing
phrase-alignment softmax prior (temperature 0.07). Queries and coordinate
weights are zero; keys remain random so the low-rank product can learn.
This is not uniform pooling and not exactly max pooling. Shared models keep
the global and max scores explicitly and add pooled-minus-max evidence.
Kernel residual models additionally retain their zero residual gate, so their
initial prediction still matches the frozen kernel. Zero gates/outputs may
delay gradients into upstream parameters by a few optimizer steps; they do
not require separate training phases. Nothing is frozen on a schedule.

Prompt embeddings are recomputed with gradients each training minibatch when
enabled. Frozen visual features remain cacheable while training prompts and
pooling. Cached scalar phrase scores are used only for shared fixed-feature
variants. Visual-adaptation variants process augmented images from the start.
All neural variants use the [shared augmentation pipeline](AUGMENTATION.md),
including periodically refreshed frozen-feature views. The pretrained text
weights stay frozen, with gradients passing through them
to token deltas. Anchors remain active throughout the same joint objective.

## Prompt preservation

The token-displacement (`--token-anchor 0.1`) and text-feature cosine
(`--feature-anchor 0.1`) penalties were already active. The new term adds
**KL(original phrase agreement || learned phrase agreement)** separately for
global-short and local-box-max scores, averaged across channels and images.
Both sides use the same frozen image features; the teacher uses the original
handwritten phrases. The student uses the current soft tokens. This preserves
relative image–phrase evidence, not predicted human ratings or kernel outputs.
Phrase softmax values are a distillation device, not demographic/class probabilities.

`--prompt-logit-weight 0.05` controls the term; `--prompt-logit-temperature 0.07`
controls the cosine-score softmax. No temperature-squared rescaling is applied.
These are research starting values, not transferred optimal classification
hyperparameters. Use `--prompt-logit-weight 0` for the direct ablation, keeping
the existing token/feature anchors active. Excessive anchoring can impede useful
adaptation, so compare matched held-out runs rather than assuming more is better.

Even with augmented live vision, this branch uses cached clean frozen features
for both student and teacher. It needs no extra image-encoder pass or second
model copy. Its gradients reach prompts only; learned pooling is free to improve
over max pooling, and vision receives its existing feature-preservation and
regression losses. Prompt deltas and pooling have zero AdamW weight decay;
all enabled groups use the joint warmup/cosine schedule and gradient clipping.

`training-profile.json` now records all learning rates and preservation settings
for every neural fit. Each epoch's `history.json` records regression MSE, total
objective, and the weighted token, text-feature and phrase-agreement losses.
It also records optimizer steps, component learning rates, pre-clipping gradient
norms and the fraction of updates clipped. The profile records effective
optimizer groups and schedule length. These expose regularization and gradient
imbalances; they do not automatically resolve them.

## Budgets, checkpoints and running

Shared families default to 600 **joint** epochs, real batch 64, accumulation 1.
Legacy/kernel neural models use 200 epochs. Every neural variant uses linear
warmup over 10% of planned optimizer updates, then cosine decay to 5% of each
group's base rate, with global gradient norm clipping at 1.0. The default runs
the full planned horizon while retaining the best inner-validation checkpoint.
For optional early stopping, lower `--min-train-fraction` from 1.0; patience is
80 epochs for shared models and 20 otherwise. Warmup and stopping fractions
are relative to the configured budget, including `--joint-epochs` overrides.

Use `--tabm-epochs` and `--transformer-epochs` for separate budgets or
`--joint-epochs` to override every neural variant. Family batch/rate flags take
precedence over generic flags. Fixed-input and adaptive-input head rates have
separate flags; the latter applies throughout joint fitting, not in stages.

Inner validation selects one joint duration; outer and all-data refits start
afresh for that duration, following the same schedule prefix rather than
compressing its horizon to the selected duration. Kernel residual variants retain the epoch-zero
fallback. New neural packets use format `fgclip2_multitarget_v2` with explicit
components. Older neural checkpoint formats and staged metadata are unsupported.
Use a new `RUN_DIR`; the launcher defaults to `runs/cuda96-joint-v4-gpu-views`. Frozen
kernel bundles remain unchanged. Existing frozen feature caches are reusable.

From the repository root, run a focused comparison:

```bash
IMAGES=/workspace/data/images \
RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-joint-v4 PROTOCOL=holdout FINAL_FIT=0 \
SMOKE_VARIANTS="shared-tabm-chain-prompt shared-transformer-chain-prompt" \
bash CLIP/scripts/setup_fgclip2_multitarget.sh \
  --variants frozen-kernel shared-tabm-chain shared-tabm-chain-prompt \
             shared-transformer-chain shared-transformer-chain-prompt \
  --tabm-epochs 600 --transformer-epochs 600
```

For five-fold training and all-data models, omit `PROTOCOL=holdout FINAL_FIT=0`
and use another output directory. Add `shared-tabm-chain-joint` and
`shared-transformer-chain-joint` for concurrent visual adaptation. Generic
launcher batches are now 64 with accumulation 1; shared family settings
override those values. Smoke runs use four minibatches and save no model or
optimizer checkpoints, allowing zero-initialized gates to open before checking
that **regression-loss gradients** reach each enabled component.

## Validation of this change

Before the v4 optimizer audit, all 21 neural variants completed four synthetic minibatches with finite
predictions and regression gradients in every enabled component. The new
preservation term is zero with native phrases, positive after perturbation,
and differentiates both text modes while leaving teacher/image features fixed.
The check uses a small CPU surrogate, not pretrained FG-CLIP2, CUDA or real
ratings. No model/optimizer checkpoints were saved. See
[joint_training_validation.json](joint_training_validation.json) for scope and
inference checks. These checks do not establish improved held-out accuracy.

No tests or training were run for the v4 optimizer changes, as requested. The
historical validation above does not validate the current schedule or defaults.
