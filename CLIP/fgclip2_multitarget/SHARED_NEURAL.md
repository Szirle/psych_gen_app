# Joint neural learning from the complete phrase bank

These default models predict ratings directly. They consume no kernel/ridge
predictions, fitted reference, or residual labels. All existing architectures,
including `kernel-*`, remain available. All neural variants now use
[joint optimization](JOINT_TRAINING.md) of their enabled components.

## Architecture choices

Every target uses the complete global-short and local-box-max phrase bank.
Scaling is fitted inside the training split. When learned pooling is enabled, the
learned pooled-minus-max score is added as a third channel: global and max
evidence remain explicit inputs. No MI pruning or private target phrase subset
is used. Enabled prompts, pooling and visual adapters train from the start.

**Shared TabM.** The official `tabm==0.0.3` components provide 32 efficient
ensemble members, width 64 and two feature-mixing blocks. All targets supervise
the same backbone; only the final output coordinates distinguish targets.
Member-specific input scaling precedes the first mixing layer, as required by
the [official TabM guidance](https://github.com/yandex-research/tabm).
One additional shared pre-LayerNorm, gated SiLU residual block refines the
representation after member diversity has been introduced. Its small learned
residual gain starts at 0.1. Set `--tabm-residual-blocks 0` for the closer
plain-TabM control. This residual extension is a research hypothesis, not an
established improvement over TabM.

Training minimizes the average **individual member MSE** across all observed
targets; inference averages member predictions. It does not optimize only the
MSE of the ensemble average, which would weaken the ensemble training signal.
The sequential TabM variant keeps exactly the same shared encoder and adds
previous member-specific predicted ratings to the final prediction layers.

The [local report](../research/neural_architecture_benchmark/REPORT.md) and
[search](../research/neural_architecture_benchmark/SEARCH.md) identify
`tabm64_k32` as the strongest tested standalone NN. Those reports describe eight
targets and 1,032 older phrase logits; they are not evidence that a larger model
will be better on the new targets. Their baseline used two blocks, 32 members,
width 64, dropout 0.1 and learning rate 0.002. We retain those architecture choices, while
testing the residual block explicitly. The learning rates have been reconsidered
for joint adaptation in the [training audit](TRAINING_AUDIT.md). This is not an exact reproduction of
that older feature set or evaluation.

**Shared transformer.** Each phrase becomes a feature token with separately
learned numerical weights for its global score, local maximum, and optional
pooling residual, plus a learned phrase identity. Three pre-normalized encoder
blocks mix the entire bank using eight attention heads, width 128 and GELU
feed-forward layers. A two-block transformer decoder uses learned target
queries and cross-attends to every encoded phrase. This adapts the numerical
tokenization/shared-attention approach of
[FT-Transformer](https://arxiv.org/abs/2106.11959); it is not an exact FT-Transformer
reproduction, nor a claim of universal 2026 SOTA performance.

In the parallel variant, all target queries interact and predict in one pass.
Thus shared representations receive supervision from happy, trustworthy and
other targets even without an explicit prediction chain. In the sequential
variant, only the decoder runs by successive target groups. Earlier predicted
ratings are embedded into prefix tokens; later groups cannot affect earlier
groups, while each group retains cross-attention to the **full** image-derived
memory. The expensive phrase encoder runs once per image batch. The decoder
prefix is recomputed per group; KV caching is not implemented.

Both chains use predictions during training and inference, never true
predecessor labels. Gradients flow through earlier predictions by default;
`--shared-detach-chain` is the ablation. Visual/basic groups are parallel, race
and complex groups sequential in the predeclared order. `--hierarchy` can supply
explicit groups. These families intentionally skip label-derived difficulty
ordering and use uniform image sampling; the legacy `--order` and
`--hard-fraction` options do not apply to them.

Shared supervision enables transfer between targets, but can also cause
negative transfer. Only held-out comparisons can establish whether it helps.
Earlier predictions are learned transformations of the same image evidence,
not additional independent measurements.

## Variants and commands

| Architecture | TabM | Transformer |
| --- | --- | --- |
| Joint, parallel predictions | `shared-tabm` | `shared-transformer` |
| Joint encoder, sequential outputs | `shared-tabm-chain` | `shared-transformer-chain` |
| Add spatial pooling | `shared-tabm-chain-pool` | `shared-transformer-chain-pool` |
| Jointly learn anchored phrase tokens and pooling | `shared-tabm-chain-prompt` | `shared-transformer-chain-prompt` |
| Jointly learn pooling, prompts and visual LoRA | `shared-tabm-chain-joint` | `shared-transformer-chain-joint` |

All ten shared variants and the frozen kernel run by default. For a focused comparison,
run this from the repository root:

```bash
IMAGES=/workspace/data/images \
RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-shared-neural \
PROTOCOL=holdout FINAL_FIT=0 \
SMOKE_VARIANTS="shared-tabm-chain shared-transformer-chain" \
bash CLIP/scripts/setup_fgclip2_multitarget.sh \
  --variants frozen-kernel shared-tabm shared-tabm-chain \
             shared-transformer shared-transformer-chain
```

For the full five-fold joint run and all-data refits, omit `PROTOCOL=holdout`
and `FINAL_FIT=0`, use a fresh output directory, and add the six adaptation
variants from the table. This is a substantial experiment. The launcher installs
the two small pinned TabM dependencies. If reusing an environment with
`INSTALL_REQUIREMENTS=0`, install them first:

```bash
python3 -m pip install tabm==0.0.3 rtdl_num_embeddings==0.0.12
```

Use a separate output directory to keep results comparable to existing runs.
Do not overwrite the ongoing experiment's files. Existing frozen feature
caches can be reused. The generic prediction CLI supports the new artifacts;
neither kernel bundles nor kernel reference caches are needed for inference.

## Training profiles: independent of old NN-head defaults

| Setting | Shared TabM | Shared transformer |
| --- | --- | --- |
| Joint epochs, maximum | 600 | 600 |
| Real batch size, frozen or live vision | 64 | 64 |
| Gradient accumulation | 1 | 1 |
| Fixed-input head base LR | 0.0003 | 0.0002 |
| Adaptive-input head base LR | 0.0001 | 0.0001 |
| Matrix weight decay | 0.0003 | 0.01 |
| Dropout | 0.1 | 0.1 |
| Joint warmup | 10% of optimizer steps | 10% of optimizer steps |
| Schedule | Cosine to 5% of base LR | Cosine to 5% of base LR |
| Global gradient norm limit | 1.0 | 1.0 |
| Default stopping | Full budget; best inner checkpoint | Full budget; best inner checkpoint |
| Patience if early stopping enabled | 80 epochs | 80 epochs |

These are starting profiles, not tuned optima. Batch 64 preserves more optimizer
updates than the previous batch 128. Refits preserve the original schedule
horizon in epochs, following its selected prefix. Set `--min-train-fraction`
below 1.0 to allow patience-based early stopping. `--joint-epochs` overrides
either family budget. Biases, norms, ensemble scales and embedding identities
are excluded from decay.

Fixed-input variants use `--tabm-lr` / `--transformer-lr`. Variants with learned
pooling, prompts or vision instead use `--tabm-adapt-lr` /
`--transformer-adapt-lr`, from the first update. Pooling uses `--pool-lr`
(0.0001), prompt tokens `--prompt-lr` (0.00002), and visual LoRA
`--encoder-lr` (0.00001). All groups share the schedule multiplier.

The official TabM package's standalone AdamW rate is 0.002; our rates differ
because this is a small-data joint adaptation problem. AdamW remains the
optimizer. EMA and alternative optimizers are candidates for controlled
experiments, not requirements for calling an implementation modern.

**Family flags take precedence over legacy head-training flags and launcher
`BATCH_SIZE`/`ACCUMULATE`.** This prevents the old shell defaults from accidentally
underpowering the new models. Examples:

```bash
# Append to the launcher command:
--tabm-epochs 600 --tabm-batch-size 64 --tabm-lr 0.0003 --tabm-adapt-lr 0.0001 \
--transformer-epochs 600 --transformer-batch-size 64 --transformer-lr 0.0002 --transformer-adapt-lr 0.0001 \
--warmup-fraction 0.1 --min-lr-ratio 0.05 --grad-clip 1.0
```

Other controls: `--tabm-width`, `--tabm-members`, `--tabm-blocks`,
`--tabm-residual-blocks`, `--transformer-width`, `--transformer-blocks`,
`--transformer-heads`, `--transformer-decoder-blocks`,
`--shared-vision-batch-size`, `--shared-dropout`, `--pool-lr`, `--prompt-lr`,
`--prompt-logit-weight`, `--prompt-logit-temperature`. Staged-budget options
and the adaptation-rate multiplier have been removed. Resolved settings are saved as
`training-profile.json` per fit and in the final model configuration.

Both families default to sigmoid ratings, matching the successful local TabM
experiment; `--shared-output-link linear` is the unbounded-output ablation.
Mean-rating MSE is the default objective. Distribution/ranking losses require
`--shared-auxiliary-losses`; untrained distributions are omitted from diagnostics.

Fixed-feature training uses cached full-bank scores from clean and augmented
views, bypassing image patch transfers, dense alignment and text encoding
between cache refreshes. The [shared augmentation pipeline](AUGMENTATION.md)
refreshes four views per image every ten epochs by default. Adaptive variants use the
existing differentiable FG-CLIP2 paths. CUDA heads use BF16 autocast when
supported, with float32 parameters, objective calculation and optimizer state;
`--no-shared-amp` disables it. Attention uses PyTorch's attention implementations,
including SDPA where supported. Peak VRAM has not been measured here; reduce
the family batch-size flag if necessary. No distributed training is required.

## Validation scope

Historical, pre-v4 joint-training gradient and inference checks are recorded in
[joint training validation](JOINT_TRAINING.md#validation-of-this-change).

The short surrogate fits verify implementation behavior, not real-data accuracy
or 96 GB CUDA memory/runtime. No claim is made that a new model beats the kernel.

The v4 optimizer audit was code review only; no tests or training were run.
