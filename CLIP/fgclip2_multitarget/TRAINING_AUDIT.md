# Training audit: joint-v4

2026-09-23. Scope: experimental `fgclip2_multitarget` neural fits and their VM
launcher. No tests, surrogate fits or real training were run for this audit,
as requested. Earlier validation JSON files describe earlier policies.
These changes are reasoned starting points; improved accuracy is unverified.

## Principles considered before choosing defaults

A strong recipe for this problem should combine:

1. A task-aligned, numerically stable objective with consistent target scaling,
   missing-label masking and shared supervision. For an ensemble, train its
   members individually; averaging only their output before the loss changes
   the training problem.
2. Joint optimization of every enabled component, with deliberate rates for
   random predictor weights, learned pooling, prompt deltas and pretrained
   visual adaptation. Warmup helps a randomly initialized predictor begin
   adapting before large updates; it does not guarantee useful upstream gradients.
3. An optimizer-update budget, linear warmup and gradual decay, correct decay
   exclusions, finite-gradient checks and clipping after accumulation. Different
   groups can share a schedule multiplier without sharing a base learning rate.
4. Small-data regularization consistent with the labels: restrained image
   augmentations, modest capacity/dropout and explicit preservation of useful
   pretrained representations. More regularization is not automatically better.
5. Clean validation, training-only feature statistics/selection, no true-label
   inputs to prediction chains, and reproducible checkpoint selection and resume.
6. Diagnostics that separate task error, regularization, numerical instability
   and component imbalance. Training error falling quickly alone cannot identify
   which mechanism limits validation performance.

There is no single architecture-independent “2026 SOTA recipe.” Training and
regularization depend on data size and model regime; this is also a central
finding of [How to train your ViT?](https://arxiv.org/abs/2106.10270).
Our effective sample size is the number of independent images, not the number
of image–target pairs. Extra targets provide supervision, not extra images.

## Findings and changes

| Item | Finding before this audit | v4 action |
| --- | --- | --- |
| Warmup/cosine | Already present, but epoch-based with different family warmups | One optimizer-step implementation for all neural variants |
| Gradient clipping | Already present at norm 1 | Configurable limit, nonfinite rejection, component norms and clipping-frequency logs |
| Head rates | Standalone TabM rate reused with adaptive inputs | Separate fixed-input and adaptive-input family rates |
| Other rates | Pool/prompt/vision rates insufficiently differentiated from the old recipe | More conservative component defaults below |
| Weight decay | A dimension-only filter misses TabM's multi-dimensional biases/scales | Explicit bias, normalization, ensemble-scale and embedding exclusions; separate visual decay |
| Joint fitting | Enabled components already trained together | Retained; no staged optimization |
| Prompt preservation | Token/feature anchors and original phrase-agreement KL already active | Retained concurrently with regression; no extra stage |
| Update budget | Batch 128 and 400 shared epochs; patience could stop before much cosine decay | Batch 64, 600 epochs; full scheduled horizon by default |
| Accumulation | Equal microbatch weighting, including a shorter final microbatch | Weight by row count within each accumulation window |
| Precision | FP32 trainable parameters already used, but CLI exposed unscaled FP16 training | BF16/FP32 training only; assert FP32 optimizer parameters |
| Resume | Identity mismatch could warn and continue | Reject changed code/data/configuration; save/check optimizer step and schedule |
| Evaluation and augmentation | Grouped evaluation, training-only preprocessing, refreshed training views already present | Retained; clean validation and frozen-kernel reference unchanged |

The initial premise that scheduling and clipping were entirely absent was not
supported by the code. Their implementation and surrounding defaults needed
revision. The new scheduler follows the update-based structure in
[OpenCLIP's scheduler](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py),
with a configurable nonzero floor added here.

## Optimizer and rates

All enabled groups use AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`.
These stable defaults were retained; there is no evidence here justifying a
different optimizer or beta sweep before establishing a corrected baseline.

| Model/component | Base LR | Matrix decay | Control |
| --- | ---: | ---: | --- |
| Legacy independent/chain and kernel-residual heads | 1e-4 | 0.01 | `--head-lr` |
| Shared TabM, parallel or fixed-feature chain | 3e-4 | 0.0003 | `--tabm-lr` |
| Shared transformer, parallel or fixed-feature chain | 2e-4 | 0.01 | `--transformer-lr` |
| Shared TabM with pool/prompt/vision adaptation | 1e-4 | 0.0003 | `--tabm-adapt-lr` |
| Shared transformer with pool/prompt/vision adaptation | 1e-4 | 0.01 | `--transformer-adapt-lr` |
| Pooling | 1e-4 | 0 | `--pool-lr` |
| Prompt deltas | 2e-5 | 0 | `--prompt-lr` |
| Visual LoRA | 1e-5 | 0 | `--encoder-lr` |
| Partially unfrozen vision | 1e-6 | 0.01 | `--partial-lr`, `--vision-weight-decay` |

The [official TabM package](https://github.com/yandex-research/tabm) explicitly
recommends `lr=0.002`, `weight_decay=0.0003` as a standalone AdamW starting point
and recommends tuning. Thus 0.002 is not intrinsically invalid. Reusing it
without reconsideration for roughly 1,000 images and jointly changing input
features was unjustified. Its individual-member loss requirement is already
implemented and remains important independently of LR.

The replacement rates are hypotheses based on this setup, not values established
by that paper. Adaptive heads now start lower, pooling learns at the head rate,
and pretrained semantics change more slowly. LR magnitude alone cannot measure
relative functional change: parameter scales, Adam moments, loss weights and
chain depth all matter. Global clipping can also couple components. The new
logs make this inspectable; they do not prove balanced optimization.

Matrix decay remains family-specific. Biases (including TabM's 2D biases),
normalization parameters, TabM `r`/`s` and elementwise scales, and embedding
identities are excluded. Pool/prompt decay is zero; explicit prompt anchors
control semantic drift. LoRA factor decay is zero as a conservative starting
choice, not a claim that decaying adapters is always wrong. Partial backbone
matrices retain their own decay rather than inheriting TabM's much smaller one.

## Schedule, budgets and checkpoint selection

- `--warmup-fraction 0.1`: linear warmup over 10% of **planned optimizer updates**.
  The first update uses a small positive rate, and warmup reaches the base rate.
- `--min-lr-ratio 0.05`: cosine decay reaches 5% of each group's base rate on the
  final planned update. `0` permits decay to zero. A one-update fit uses the base
  rate because it cannot meaningfully contain both warmup and decay.
- `--grad-clip 1.0`: global L2 norm clipping once per accumulated optimizer update.
- Shared families: 600 epochs, real batch 64, accumulation 1, dropout 0.1.
  Live vision also defaults to batch 64. Legacy/kernel neural heads: 200 epochs,
  generic launcher batch 64, accumulation 1.
- `--min-train-fraction 1.0`: run the complete planned schedule, retaining the
  best inner-validation checkpoint. Lower this explicitly to permit early
  stopping; patience is 80 epochs for shared models and 20 otherwise.

For approximately 602 inner-training images, batch 64 gives 10 updates/epoch:
6,000 planned updates and 600 warmup updates at 600 epochs. With 803 outer-training
images it gives 13 updates/epoch; with 1,004 all-data images it gives 16. This is
why epochs alone and VRAM occupancy are insufficient measures of training budget.
Large VRAM remains useful for encoding/cache throughput without requiring a
near-full-dataset optimization batch. No automatic linear LR scaling is applied
when batch size changes.

Default full-horizon selection costs more than patience-based stopping, but
lets validation evaluate the low-LR portion before rejecting a run. It does
not force the last epoch to win. A duration-selected refit follows the **prefix
of the original epoch horizon**, with step counts adjusted for its training
size. Compressing the entire cosine into a selected early duration would train
under a policy that inner selection never evaluated. Consequently an early
selected refit may finish before the LR floor; this is intentional.

Caching, validation and accumulated microbatches do not advance the scheduler.
Resume restores the optimizer and exact update count at epoch boundaries.
Existing deterministic epoch/view seeding is retained; cross-device bitwise
reproducibility is not promised. Use a new run directory after code changes.

Accumulation now weights a short microbatch by its actual row count. With
missing labels, per-target normalization can still differ from a single large
batch, and ranking losses do not create pairs across microbatches. Accumulation
is therefore not claimed to reproduce every large-batch objective exactly;
shared defaults use accumulation 1.

## Reviewed and deliberately retained

- **Joint task and preservation losses.** PromptSRC uses concurrent preservation
  and task learning in its [official trainer](https://github.com/muzairkhattak/PromptSRC/blob/main/trainers/promptsrc.py).
  Our regression adaptation keeps token anchor 0.1, text-feature anchor 0.1,
  phrase-agreement KL 0.05 at temperature 0.07, visual preservation 0.1 and
  live-view consistency 0.05. Those weights need ablation; reducing LR does not
  fix over-anchoring. We do not implement its Gaussian prompt averaging or
  textual-template ensemble.
- **Loss alignment.** Shared models optimize masked, target-balanced mean-rating
  MSE, with memberwise TabM loss. Distribution/ranking auxiliaries are opt-in
  for shared and kernel-residual models. Legacy heads retain their original
  auxiliary defaults, so their objectives are not identical controls.
- **Capacity/output links.** TabM k32/width64/two mixing blocks plus one optional
  residual block; transformer width128/three encoder/two decoder blocks;
  dropout 0.1; sigmoid outputs. These remain explicit starting architectures.
  Smaller capacity, no extra TabM residual block, and linear outputs are useful
  ablations, especially for extrapolation; none is established as best here.
- **Augmentation.** Four restrained FFHQ training views refreshed every ten
  epochs, 25% clean draws, mild geometry/noise/blur; color and representation
  noise off by default. Live vision receives fresh transforms online. Avoid
  assuming strong color edits are label-preserving for skin/hair-color ratings.
- **Numerics.** Optimized parameters and Adam state are FP32; supported CUDA
  paths can use BF16 compute. FP16 cache storage is unrelated to FP16 training.
  Exposing FP16 training without scaling was inappropriate; PyTorch documents
  the scaling/unscaling requirements in its
  [AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).
- **Evaluation.** Statistics, selections and fitted references stay inside
  training partitions. Prediction chains use predicted predecessors in both
  training and inference. Validation is clean and selects duration on an inner
  split; outer results are for comparison, not optimizer feedback.

## Diagnostics and next experiments

`training-profile.json` records resolved group rates/decays/parameter counts,
effective batch size, optimizer settings and schedule length. `history.json`
records task MSE, weighted preservation terms, optimizer steps, last-used rates,
mean pre-clipping component/global gradient norms, and clipping frequency.
Frequent clipping, preservation dominating MSE, or vanishing prompt gradients
are reasons to investigate rather than blindly extend training. Gradient norms
are not Adam parameter-update norms; use them as diagnostics, not direct measures
of learning speed.

After a corrected baseline, prioritize matched inner-validation experiments:
head LR at 0.5x/1x/2x, prompt/pool rates, anchor weights, and capacity/dropout.
Keep splits, seeds, augmentations and update budgets comparable, then confirm
the chosen recipe over repeated seeds and untouched evaluation data. More
seeds quantify uncertainty; selecting only the luckiest seed does not.

EMA/SWA or PromptSRC-style Gaussian averaging are reasonable subsequent
candidates, but require consistent averaging of heads, prompts, pool and visual
adapters, plus a defined validation/inference policy. SAM, alternative optimizers,
layerwise LR decay for deeper unfreezing, and multi-task gradient conflict
methods are also experiments rather than universal missing requirements.
Adding all of them now would obscure which change helped.

## Running

The launcher now defaults to `RUN_DIR=runs/cuda96-joint-v4` and still runs the
frozen kernel plus all ten shared variants. Existing architecture names remain
available. Frozen feature caches remain reusable, but previous-policy optimizer
checkpoints cannot resume into v4. No inference artifact format change is needed.

Append overrides to the existing launcher invocation if desired:

```bash
--warmup-fraction 0.1 --min-lr-ratio 0.05 --grad-clip 1.0 \
--tabm-lr 0.0003 --tabm-adapt-lr 0.0001 \
--transformer-lr 0.0002 --transformer-adapt-lr 0.0001 \
--pool-lr 0.0001 --prompt-lr 0.00002 --encoder-lr 0.00001
```

These are already the defaults. The launcher's future VM smoke behavior is
unchanged; nothing was executed for this audit. Current real-model accuracy,
CUDA runtime and the revised optimizer path remain unvalidated.

Subsequent augmentation throughput rewrite: see [AUGMENTATION.md](AUGMENTATION.md)
for GPU tensor transforms, persistent/replayed view banks and the current
`runs/cuda96-joint-v4-gpu-views` launcher default. The optimizer policy is unchanged.
