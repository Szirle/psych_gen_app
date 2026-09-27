# All-target FG-CLIP2 models

The default is **`agop-kernel`**, a frozen FGCLIP2 encoder with per-target AGOP
continuous weighting and independently selected kernel readouts. See [CURRENT_MODEL.md](CURRENT_MODEL.md)
for its algorithm, checkpoint interface, full-CV comparison and future directions.
`frozen-kernel` preserves the original full-bank baseline. Neural experiments below
remain available through explicit `--variants` selections.

These models predict human subjective appearance ratings, not verified demographic
or personality facts.

## Fresh CUDA VM: uploaded data

Clone the repository, upload the original images and existing ratings pickle,
then run this from the repository root (the directory containing `CLIP/`):

```bash
IMAGES=/workspace/data/images \
RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-all-targets \
bash CLIP/scripts/setup_fgclip2_multitarget.sh
```

The launcher expects Python >=3.10 and working CUDA PyTorch/torchvision in the
container. It preserves those installed versions, installs the remaining
dependencies, downloads the pinned FG-CLIP2 So400M model through the existing
download utility and starts training. A short real-model CUDA smoke is available
with `RUN_SMOKE=1`.
It requires your uploaded paths by default. Run it in `tmux` or an equivalent
persistent session if the SSH connection may close.

The pickle must have the existing `target -> filename -> individual ratings`
structure, with ratings in [0, 1]. Use trusted files. All available targets are
loaded automatically; missing image/target labels are masked. Repeated ratings
remain supervision for one image row, rather than separate training images.
Pixel-identical images stay in the same split. Supply `--groups groups.json`
(`filename -> group ID`) for related identities or latent families that should
also stay together.

**Default workload:** `agop-kernel`, five outer folds, four inner folds for per-target
metric/readout selection, and a final all-image fit in the launcher. For a matched
comparison set `VARIANTS="frozen-kernel agop-kernel"`. Neural variants are opt-in.
The launcher smoke run is opt-in with `RUN_SMOKE=1`.

For the cached-score 10×10 nested-CV deployment ensemble and OOF noise-ceiling
figure, use [PRODUCTION.md](PRODUCTION.md).

For a smaller first comparison, use a separate output directory:

```bash
IMAGES=/workspace/data/images \
RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-first-comparison \
PROTOCOL=holdout FINAL_FIT=0 \
VARIANTS="frozen-kernel chain-fixed chain-pool chain-prompt chain-joint" \
bash CLIP/scripts/setup_fgclip2_multitarget.sh
```

For setup plus smoke only, set `RUN_TRAINING=0`. Smoke uses 32 images,
128 patches, four joint minibatches, and writes diagnostics but **no model or
optimizer checkpoints**. It checks native global/dense feature parity,
zero-delta text parity, finite gradients, and head/pool/prompt/backbone updates.

## Augmented training views

Neural training now uses conservative FFHQ augmentations, including cached
augmented features for frozen backbones: four views per training image,
refreshed every ten epochs, with 25% clean draws. Fixed-feature shared models
cache phrase scores; prompt/pooling models cache dense features. Live visual
adaptation uses the same transforms online. Validation remains clean.
See [augmentation settings and cache behavior](AUGMENTATION.md).

## Architectures and joint training

For **direct joint neural prediction without a kernel baseline**, see
[shared TabM and transformer families](SHARED_NEURAL.md). These opt-in families
use the complete phrase bank, joint feature transformations, optional sequential
output decoding, and separate training profiles (600 joint epochs, real batch
64, accumulation 1). All neural variants now optimize their enabled components
together. See [joint training and research rationale](JOINT_TRAINING.md) for
learning rates, preservation losses, budgets and pooling initialization.

### Opt-in full-bank kernel residual family

The original neural baselines are **not capacity/input-matched to the kernel**:
each target receives only selected phrases through a small private MLP, uses a
sigmoid output, and trains with distribution/ranking losses. The kernel uses
the complete global/local bank, tunes regularization and family blends, and
predicts unbounded rating means. Separate outputs alone do not establish the
cause of an accuracy gap: the kernel also selects readouts per target.

The new `kernel-*` family keeps the **actual tuned frozen kernel as an additive
reference**, rather than trying to rediscover it through neural optimization:

`prediction = frozen kernel(all original scores) + learned correction`

Every target has access to **all** original global and local-max phrase scores,
both through a direct linear connection and a shared nonlinear encoder (width
256 by default). There is no MI pruning or private phrase subset. The original
scores remain available when prompts, spatial pooling, or vision are adapted;
the adaptive evidence is an additional input. In chain variants, earlier
predicted ratings are concatenated to that shared representation. These are
derived features of the image, not new independent measurements.

| New variant | Additions to the same full-bank kernel reference |
| --- | --- |
| `kernel-residual` | Shared nonlinear + direct linear corrections, no chain |
| `kernel-chain` | Same inputs/capacity, plus earlier predicted ratings |
| `kernel-chain-pool` | Add learned spatial pooling residuals |
| `kernel-chain-prompt` | Jointly learn corrections, pooling and anchored tokens |
| `kernel-chain-joint` | Jointly learn corrections, pooling, prompts and visual LoRA |

Kernel residuals are opt-in and train jointly. From the repository
root, invoke the **same launcher** with an additional `--variants` argument:

```bash
IMAGES=/workspace/data/images \
RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-kernel-backed \
bash CLIP/scripts/setup_fgclip2_multitarget.sh \
  --variants frozen-kernel kernel-residual kernel-chain kernel-chain-pool \
             kernel-chain-prompt kernel-chain-joint
```

For an initial comparison, use `PROTOCOL=holdout FINAL_FIT=0` and just
`--variants frozen-kernel kernel-residual kernel-chain`. Keep the same targets,
patch budget, seed and split protocol as the previous run for a matched
comparison. Existing model/frozen-feature caches are reusable. A new output
directory is required by the experiment fingerprints; do not interrupt or
modify the old run's files to add these variants.

Implementation details that matter for interpretation:

- The kernel reference uses the **same candidate grid, grouped inner CV,
  target-specific blends, scaling and feature ordering** as `frozen-kernel`.
  Corrections initialize to zero, reproducing that reference within float32
  rounding. Outputs have no sigmoid or clipping.
- Residual training uses **nested out-of-fold kernel predictions**. Each image
  is excluded from both fitting and recipe selection for its own reference.
  Validation/test predictions use the kernel fitted on the complete relevant
  training split. This avoids learning corrections to optimistic in-sample
  predictions, though the smaller OOF fits still differ from the final fit.
- Reference creation is more expensive than one kernel fit. It is cached per
  training split and reused across the new variants. MI selection is skipped;
  target ordering and hard sampling use the cross-fitted kernel errors.
- The loss defaults to mean-rating MSE plus existing adaptation anchors.
  Distribution/ranking supervision is opt-in with `--kernel-auxiliary-losses`.
  Without that flag, distribution predictions are untrained and omitted from
  diagnostics. `--kernel-hidden 256` controls the shared encoder width.
- Inner validation includes **epoch zero** for the joint fit. Zero selected
  epochs retain the kernel reference; refits honor zero durations. Pooling
  starts as max pooling through a zero-initialized residual gate. This is a
  validation-selected fallback, not a guarantee about outer-test accuracy.
  As with all variants, the outer held-out labels never select checkpoints.
- All full-bank phrase tokens are shared across targets, with shared and
  phrase-specific low-rank deltas. Original frozen scores remain present after
  token learning. Chaining never uses true predecessor labels.
- Kernel-joint inference computes original frozen scores before attaching the
  learned visual adapters, then evaluates the adapted branch. This requires an
  extra image pass. The existing `predict` CLI recognizes the new checkpoints,
  which embed their fitted kernel readouts.
- Phrase diagnostics describe the adaptive correction path with the fixed
  kernel/frozen-score inputs held constant. Gated spatial residual weights can
  be signed; they are not attention probabilities or causal explanations.

Current validation is described in [joint training](JOINT_TRAINING.md#validation-of-this-change).

### Original variants

| Variant | Simultaneously optimized components | Purpose |
| --- | --- | --- |
| `frozen-kernel` | Frozen full phrase bank + tuned kernel | Strong matched baseline using global and local phrase scores |
| `independent` | Heads | Separate target predictions with the same selected phrase inputs |
| `chain-fixed` | Heads | Add the sequential target dependency structure |
| `chain-pool` | Heads + pooling | Learn where each phrase should read visual evidence |
| `chain-prompt` | Heads + pooling + prompts | Learn anchored phrase tokens through the frozen text transformer |
| `chain-joint` | Heads + pooling + prompts + visual LoRA | Joint adaptation with augmented image inputs |
| `chain-partial` | Heads + pooling + prompts + last visual blocks | Optional, higher-capacity joint comparison |

All neural variants are opt-in; the default run uses `agop-kernel`. The `chain-lora` alias has been
removed. Add `chain-partial` through `VARIANTS` to test
limited block unfreezing. Full-model training is deliberately not a default for
this small image dataset.

1. **Phrase selection uses training labels only.** Average repeated subsample
   mutual information estimates across global and local evidence; penalize
   redundant selections. Keep eight hand-designed phrases per known target,
   add eight target-specific MI selections, and share sixteen visual phrases.
   Counts are configurable. The candidate bank includes the existing race
   phrase bank and dedicated phrases for all 34 known dimensions.
2. **Predict targets in the agreed order:** visual → basic attributions → race
   impressions → complex judgments. Visual and basic groups each predict in
   parallel; race and complex targets are sequential within their group.
   Descending training-only OOF ridge R² orders those latter groups. White is
   first only if its estimated predictability supports that choice. Use
   `--order fixed` for the declared order or `--hierarchy hierarchy.json` for an
   explicit array of groups covering every target exactly once.
3. **Train on predicted predecessors throughout.** A target-specific MLP sees
   its global/local phrase scores and a shared projection of earlier predicted
   ratings. There is no ground-truth context/teacher forcing. Context dropout
   and detached predecessors reduce error propagation and prevent later losses
   from distorting earlier predictions. `--chain-gradients` is an ablation.
4. **Learn spatial filters.** A masked softmax combines phrase alignment,
   low-rank visual keys/phrase queries, and six spatial coordinates. It replaces
   the maximum over a local phrase alignment map. FG-CLIP2's native global
   attention pooling is retained; it is not incorrectly treated as max pooling.
5. **Learn actual phrase tokens.** Zero deltas reproduce handwritten token
   embeddings. Shared, target-specific, and low-rank phrase residuals adjust
   the first four non-special tokens while preserving the semantic suffix.
   Both native short/global and box/local text representations are used.
   Penalties anchor token displacement and output cosine similarity to the
   original phrases. A PromptSRC-inspired KL penalty additionally preserves
   original global/local phrase agreement on frozen image features. See
   [preservation settings](JOINT_TRAINING.md#prompt-preservation).
   Text blocks are activation-checkpointed and processed in
   chunks; original embeddings and text weights are frozen.
6. **Adapt vision jointly when enabled.** LoRA defaults to rank 8 in the last eight visual
   blocks; partial unfreezing defaults to the last two. Training uses random
   the shared FFHQ augmentation pipeline, clean-view prediction consistency, frozen global-feature
   preservation, and a mixture of uniform sampling with clipped, training-only
   OOF difficulty. Colour jitter is opt-in and destructive crops are excluded because
   skin/hair colour and scene context are themselves targets.

The objective combines equally weighted target mean errors, nine-bin empirical
rating-distribution CDF errors, distribution/mean consistency, uncertainty-aware
pairwise ranking, and the anchor terms above. Individual ratings provide richer
supervision without pretending there are more independent images. Standard
errors assume independent ratings; repeated raters can violate that assumption.

Before vision adaptation, inner-validation diagnostics record per-image target
errors, phrase sensitivities, token movement, and spatial weights. These are
inspection aids. The current optimizer does **not** automatically convert those
maps into target-specific regional losses; the implemented adaptive sampling
uses training OOF residuals. Attention and conditional sensitivity are not
causal explanations.

## Evaluation, resources, and resume

### Faster phrase selection (2026-09-23)

`--selection-backend auto` now uses CUDA for mutual information when training
on CUDA. It batches exact pairwise Chebyshev distances and nearest-neighbor
radius counts across phrase features, avoiding thousands of separate CPU tree
builds. It retains the continuous KSG estimator, k=3, sklearn scaling, seeded
tie-breaking noise, repeat subsamples, and global/local maximum rule. Float64
distances are intentional: float32 would erase the tiny tie-breaking noise.
The reference is [sklearn's continuous MI implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_selection/_mutual_info.py).

`--selection-memory-gib 4` budgets temporary GPU workspace, additionally capped
at 40% of currently free VRAM. Increasing it can reduce batching overhead, but
filling 96 GB is unnecessary for ~642 sampled images. The algorithm evaluates
O(features × images²) distances; its benefit is GPU parallelism at this small
sample count, not better asymptotic scaling. `--selection-backend sklearn`
retains the CPU estimator, now using `--cpu-threads` parallel feature workers.

Two independent algorithmic savings apply to both backends: compute the global
phrase correlation matrix once per training split, and group targets with the
same observed rows into one multi-output ridge solve per inner fold. Complete
34-target data therefore needs four shared factorizations instead of 136
independent fits. The ridge objective/alpha/scaling remain unchanged, but exact
Cholesky replaces tolerance-stopped LSQR, so small numerical differences are
expected. Missing-label patterns receive separate solves. All calculations
remain inside training folds. Logs report progress per target and ridge fold;
`selection.json` includes timings and the selected backend.

Local checks: MI agreed with sklearn within 1.4e-15 on CPU tensors with both
float32/float64 inputs, including ties, constants, and different chunk sizes.
A complete small selection check retained phrase choices and target order.
On synthetic 602-training/201-validation rows, 1,500 features and 34 targets,
the ridge step fell from 0.402 s to 0.034 s (~12×); this is **not** a measured
CUDA or whole-stage speedup. No pretrained model was loaded for these checks.

Measure CUDA parity, runtime and allocated memory on the VM without loading a
model or writing checkpoints:

```bash
python3 -m CLIP.fgclip2_multitarget.benchmark_selection \
  --samples 642 --features 1500 --memory-gib 4
```

Normal training picks CUDA automatically; optionally append
`--selection-memory-gib 8` to the launcher command. A running Python process
keeps the implementation it already imported. **Use a new `RUN_DIR` after
updating this code:** the joint policy cannot resume staged optimizers.
Code, data and configuration identity mismatches now also reject resume.
See the [v4 training audit](TRAINING_AUDIT.md); the default run directory is
`runs/cuda96-joint-v4-gpu-views`.
The frozen feature/model caches remain reusable; selections are recomputed
under the new implementation.

Legacy/kernel neural fits default to 200 joint epochs, patience 20; shared
families use 600 and patience 80. By default all run the full cosine horizon
while selecting the best inner checkpoint; patience applies only if
`--min-train-fraction` is lowered below 1.0. One joint
duration is chosen on an inner split, then a new model, phrase selection,
and ordering are fitted on the complete outer-training split. This is cheaper
than averaging neural duration selection across every inner fold. The kernel
baseline uses the existing grouped inner-CV readout selection.

Compare `comparison.json`, per-target metrics, and paired held-out predictions.
Macro RMSE is the square root of equally weighted per-target MSE. A broad
improvement should survive multiple seeds and an independently chosen evaluation
set. Choosing a winning architecture on these results makes that choice part
of model selection; the same scores are not an unbiased final external test.

For extrapolation, predeclare a tail target and use another output directory:

```bash
IMAGES=/workspace/data/images RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-attractive-tails PROTOCOL=holdout FINAL_FIT=0 \
bash CLIP/scripts/setup_fgclip2_multitarget.sh --tail-target attractive
```

Both tails (10% each by default) are held out, with group boundaries preserved.
Inner duration selection uses tails of the remaining training range. This
measures extrapolation in the declared target, not all targets simultaneously.

Useful launcher settings:

| Setting | Default | Effect |
| --- | --- | --- |
| `PATCHES` | `576` | Native image patch budget; try 784/1024 in a separate experiment |
| `BATCH_SIZE`, `ACCUMULATE` | `64`, `1` | Training microbatch and gradient accumulation |
| `AUGMENTATION_ENCODE_BATCH` | `512` | No-grad GPU augmentation encoding batch with OOM backoff; persistent view bank |
| `ENCODE_BATCH`, `TEXT_CHUNK` | `64`, `16` | Frozen image cache and differentiable phrase chunk sizes |
| `PRECISION`, `MODEL` | `bf16`, `so400m` | Pretrained backbone precision and size (`base` also supported) |
| `FOLDS`, `INNER_FOLDS` | `5`, `4` | Outer evaluation and inner grouped splitting |
| `FINAL_FIT` | `1` | Save an additional all-data model per variant |
| `RESUME` | `1` | Resume identical experiments at epoch boundaries |
| `MODEL_CACHE`, `FEATURE_CACHE` | Repository-local directories | Reuse downloaded weights and frozen features |
| `INSTALL_REQUIREMENTS`, `RUN_SMOKE` | `1`, `1` | Set to 0 when reusing a verified environment |
| `SKIP_DOWNLOAD` | `0` | Set to 1 when model/data already exist locally |

Reduce batch/chunk sizes if memory is insufficient; 96 GB does not guarantee
every combination fits. BF16 is the intended CUDA training precision; FP32 is also supported.
FP16 training is not exposed because this loop has no gradient scaler.
Frozen caches can still use FP16 storage, independently of training precision. Host RAM/disk must also hold dense feature caches, dataset, model
weights, and best/last checkpoints. Runs execute variants sequentially on one
GPU; there is no distributed-training requirement.

Additional CLI flags can follow the script path, e.g.
`--joint-epochs 600 --pool-lr 0.0001 --token-anchor 0.2`. See
`python3 -m CLIP.fgclip2_multitarget --help` for all controls.

Resume requires identical code, data, paths and training configuration. Changing
variants, epochs, final-fit choice, or patch budget requires a new `RUN_DIR`.
Use a new directory for this policy. Old neural checkpoint formats and
staged CLI options are no longer supported. Completed joint folds are reused; interrupted fits restore model and optimizer
state. Freeze versions and keep the same checkout while a run is in progress.

Outputs include `manifest.json`, `comparison.json`, per-variant `summary.json`
and `heldout.npz`, per-fold selections/histories/diagnostics, and model snapshots.
Logs and `pip-freeze.txt` are in a timestamped sibling setup directory.

## Predict with the resulting models

With `FINAL_FIT=1`, each all-data artifact is at
`RUN_DIR/<variant>/final/model.pt`. For example, from the repository root:

```bash
python3 -m CLIP.fgclip2_multitarget.predict \
  --checkpoint /workspace/fgclip2-all-targets/chain-joint/final/model.pt \
  --images /workspace/new-images \
  --output /workspace/predictions.csv --device cuda
```

Pass `--model-cache` if setup used a custom model cache. Saved artifacts record
the pinned model revision, phrase ordering, target ordering, network state and
adapted backbone parameters. They require the original pretrained snapshot and
this research code. Fold models are in `<variant>/fold-N/refit/model.pt`.

## Optional data transport

Uploaded paths are the default. To bundle the local files before uploading:

```bash
/opt/anaconda3/envs/manip311/bin/python -m CLIP.fgclip2_multitarget.prepare_data pack \
  --images /path/to/images --ratings /path/to/dim_to_photo_to_ratings.pkl \
  --archive /tmp/fgclip2-data.tar.gz
```

On the VM, `DATA_ARCHIVE=/path/to/fgclip2-data.tar.gz` or `DATA_URL=...` can
replace `IMAGES`/`RATINGS`; optionally set `DATA_SHA256`. Public OMI acquisition
is an explicit alternative: `DATA_SOURCE=omi`. That path downloads a pinned
official archive, preserves its license/provenance, and converts CSV ratings
with `(rating + 0.5) / 101`, matching the existing store. Public OMI is licensed
CC BY-NC-SA 4.0. The local/public audit matched all 34,136 original-image target
pairs; maximum mean difference was 2.53e-8. Use uploaded data for exact continuity
with your existing experiments.

## Research basis and validation status

- [FG-CLIP2](https://arxiv.org/html/2510.10921v3): native global and local
  image–text paths underpin the architecture; preserving both matters for
  phrase grounding.
- [CoOp](https://arxiv.org/abs/2109.01134) and
  [PromptSRC](https://arxiv.org/abs/2307.06948): motivate differentiable context
  tokens with pretrained semantic anchors. This implementation is an anchored
  token-delta experiment, not a reproduction of either training recipe.
- [PLOT](https://arxiv.org/abs/2210.01253): supports exploring local alignment
  of multiple prompts. Our pooling uses learned softmax filters, not PLOT's
  optimal transport objective.
- [Multi-target regression and regressor chains](https://arxiv.org/abs/1211.6581):
  motivates conditional prediction and avoiding training/test context mismatch.
  This is a free-running neural DAG, not a faithful ERC implementation.
- [NIMA](https://arxiv.org/abs/1709.05424): motivates learning subjective rating
  distributions alongside means.
- [OMI study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9169911/) and
  [official dataset](https://github.com/jcpeterson/omi): provide the target/data
  context. Broader fine-tuning literature is in the existing
  [research review](../FGCLIP2_FINETUNING_RESEARCH_2026-09-19.md).

Completed locally: syntax checks, the full label-conversion audit, and tiny
CPU surrogate training through all eight variants (19 neural optimizer updates,
finite predictions, expected parameter updates, restored backbone, zero saved
checkpoints). These checks do **not** validate real FG-CLIP2 CUDA numerics,
96 GB memory use, resume/inference on the real model, or improved accuracy.
The launcher runs the real-model smoke on your VM before the long experiments.
No additional large local model was loaded beside the existing training job.
