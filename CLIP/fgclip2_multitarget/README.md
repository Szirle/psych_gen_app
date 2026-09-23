# All-target FG-CLIP2 research

Isolated experiments for a 96 GB CUDA GPU and approximately 1,004 shared face
images. Existing core and `fgclip2_adaptation` implementations are unchanged.
This package reuses their encoder, temporary backbone adaptation, ratings,
grouped splits, regression metrics, and kernel readouts. Remove this directory
and `../scripts/setup_fgclip2_multitarget.sh` to remove the experiment.

These models predict subjective ratings of the synthetic OMI faces, including
perceived demographic attributes. The outputs are not verified demographic or
personality facts. Accuracy improvements are hypotheses to evaluate, not measured
results from this implementation.

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
download utility, runs a short real-model CUDA smoke, and starts training.
It requires your uploaded paths by default. Run it in `tmux` or an equivalent
persistent session if the SSH connection may close.

The pickle must have the existing `target -> filename -> individual ratings`
structure, with ratings in [0, 1]. Use trusted files. All available targets are
loaded automatically; missing image/target labels are masked. Repeated ratings
remain supervision for one image row, rather than separate training images.
Pixel-identical images stay in the same split. Supply `--groups groups.json`
(`filename -> group ID`) for related identities or latent families that should
also stay together.

**Default workload:** seven variants, five outer folds, one inner validation
split per neural fold, then a fresh outer-training refit. Inner grouped OOF
predictions separately drive phrase ordering and sampling. Finally each variant
is fitted to all images using median inner-selected stage durations. This is a
substantial research run, not a quick benchmark. The frozen image/text cache is
shared; labels never enter it. No wall-time or peak-memory guarantee is implied.

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
128 patches, one batch per stage, and writes diagnostics but **no model or
optimizer checkpoints**. It checks native global/dense feature parity,
zero-delta text parity, finite gradients, and head/pool/prompt/backbone updates.

## Architectures and staged training

| Variant | Stages | Purpose |
| --- | --- | --- |
| `frozen-kernel` | Frozen full phrase bank + tuned kernel | Strong matched baseline using global and local phrase scores |
| `independent` | Heads | Separate target predictions with the same selected phrase inputs |
| `chain-fixed` | Heads | Add the sequential target dependency structure |
| `chain-pool` | Heads → pooling | Learn where each phrase should read visual evidence |
| `chain-prompt` | Heads → pooling → prompts | Learn anchored phrase tokens through the frozen text transformer |
| `chain-lora` | Previous stages → visual LoRA | Freeze learned prompts during augmented visual adaptation |
| `chain-joint` | Previous stages → visual LoRA + prompts | Continue language and vision adaptation jointly |
| `chain-partial` | Previous stages → last visual blocks | Optional, higher-capacity comparison; prompts frozen in the visual stage |

The first seven run by default. Add `chain-partial` through `VARIANTS` to test
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
   original phrases. Text blocks are activation-checkpointed and processed in
   chunks; original embeddings and text weights are frozen.
6. **Adapt vision last.** LoRA defaults to rank 8 in the last eight visual
   blocks; partial unfreezing defaults to the last two. Training uses random
   horizontal flips, clean-view prediction consistency, frozen global-feature
   preservation, and a mixture of uniform sampling with clipped, training-only
   OOF difficulty. Colour jitter and destructive crops are avoided because
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

Neural stages default to at most 40/15/15/15 epochs with patience 5. Stage
durations are chosen on one inner split, then a new model, phrase selection,
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
| `BATCH_SIZE`, `ACCUMULATE` | `32`, `2` | Training microbatch and gradient accumulation |
| `ENCODE_BATCH`, `TEXT_CHUNK` | `16`, `16` | Frozen image cache and differentiable phrase chunk sizes |
| `PRECISION`, `MODEL` | `bf16`, `so400m` | Pretrained backbone precision and size (`base` also supported) |
| `FOLDS`, `INNER_FOLDS` | `5`, `4` | Outer evaluation and inner grouped splitting |
| `FINAL_FIT` | `1` | Save an additional all-data model per variant |
| `RESUME` | `1` | Resume identical experiments at epoch boundaries |
| `MODEL_CACHE`, `FEATURE_CACHE` | Repository-local directories | Reuse downloaded weights and frozen features |
| `INSTALL_REQUIREMENTS`, `RUN_SMOKE` | `1`, `1` | Set to 0 when reusing a verified environment |
| `SKIP_DOWNLOAD` | `0` | Set to 1 when model/data already exist locally |

Reduce batch/chunk sizes if memory is insufficient; 96 GB does not guarantee
every combination fits. BF16 is the intended CUDA training precision. FP16 is
exposed for experiments but has no gradient scaler and is not the recommended
training mode. Host RAM/disk must also hold dense feature caches, dataset, model
weights, and best/last checkpoints. Runs execute variants sequentially on one
GPU; there is no distributed-training requirement.

Additional CLI flags can follow the script path, e.g.
`--head-epochs 60 --vision-epochs 20 --token-anchor 0.2`. See
`python3 -m CLIP.fgclip2_multitarget --help` for all controls.

Resume requires identical code, data, paths and training configuration. Changing
variants, epochs, final-fit choice, or patch budget requires a new `RUN_DIR`.
Completed folds are reused; interrupted stages restore model and optimizer
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
