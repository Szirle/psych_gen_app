# Experimental FG-CLIP2 rating adaptation

Everything new lives in this directory. Removing it removes the experiments;
the encoder, shared utilities, existing phrase bank, kernel implementation, and
saved production bundles are unchanged. These are compact regression adaptations
of the [research proposals](../FGCLIP2_FINETUNING_RESEARCH_2026-09-19.md), **not exact
reproductions of the papers or demonstrated accuracy improvements**.

The outputs are eight independent mean subjective impressions on the original
0–1 sliders. They are neither objective attributes nor a class probability simplex.

## Architectures

| CLI method | Trained components and input |
| --- | --- |
| `ridge` | Training-scaled ridge on complete frozen visual embeddings; inner-CV alpha selection. |
| `kernel` | Existing linear/quadratic/RBF selection and target-specific blending, on complete visual embeddings. |
| `phrase-kernel` | Reference using the existing 516 phrases and kernel machinery. With `--patches 128 256`, uses the complete existing candidate grid. |
| `hybrid` | Phrase kernels plus a ridge residual on visual coordinates orthogonal to the fixed phrase span. Residual training labels are cross-fitted; per-target correction weights are selected inside outer training. |
| `adapter` | Zero-initialized residual bottleneck `normalize(z + U GELU(Vz))`, then an eight-output head. |
| `prototypes` | Five verbal levels per target, anchored low-rank updates of their embeddings, learned temperatures, expected rating plus a small continuous correction. Softmax is within each target. |
| `phrase-adapter` | Anchored low-rank updates to all 516 phrase embeddings plus a small nonlinear readout. Text vectors genuinely change; original phrases no longer exactly explain the learned directions. |
| `soft-prompt` | Shared trainable context tokens through the frozen native text tower, feeding the prototype predictor. Preserves native short positions, noncausal masking and final-position pooling. |
| `local` | Target-specific masked attention over native dense features, with small shared key/value projections and a global regression branch. |
| `lora` | Trainable float32 low-rank updates to Q/V in the last visual blocks plus a direct head. Frozen pretrained weights retain their native dtype. |
| `partial` | Last visual block(s) unfrozen in float32 plus a direct head. Start with one block and a smaller encoder learning rate. |
| `pool` | Only the native visual attention-pooling probe is unfrozen, plus a direct head. The rest of the native pooling head stays frozen. |

`--visual-lora` also combines visual LoRA with `adapter`, `prototypes`,
`phrase-adapter`, `soft-prompt`, or `local`. `--rank`, `--blocks`,
`--bottleneck`, `--context-tokens`, and `--levels {3,5}` control capacity.

All neural methods use mean MSE. Optional additions, tested separately:

- `--auxiliary pairwise`: soft ordering probabilities derived from mean differences
  and rating mean-SEs; assumes independent approximately normal mean estimates.
- `--auxiliary ordinal`: target-specific soft neighborhood contrastive supervision,
  inspired by continuous ordinal learning. Use batches of at least 4; two examples
  supply no nontrivial neighborhood choice. This is not the full ConOrd algorithm.
- `--auxiliary distribution`: empirical individual-rating histograms with
  mean-preserving interpolation onto slider anchors, CDF loss, and mean consistency.
- `--preserve feature` (default), `gram`, or `none`: restrain visual drift relative
  to the original frozen FG-CLIP2 features. `--preserve-weight`, `--aux-weight`,
  and `--anchor-weight` control the separate penalties.

The small distribution and ordinal heads only receive supervision when their
corresponding auxiliary loss is enabled. For a frozen visual backbone with no
visual adapter, a visual preservation penalty is constant and has no training effect.

## Reused infrastructure and evaluation

The CLI uses `FaceImpressionPipeline`, `HumanRatingsStore`, `fit_score_model`,
`make_cv_splits`, `cross_validate_predictor`, `RegressionResults`, and the current
kernel API. The only custom data adapter reads individual ratings from the trusted
local ratings store to construct distributions. No new dependency is needed in
the existing `manip311` environment.

- Every outer fold starts from the pinned pretrained weights. Temporary LoRA,
  pooling-probe and unfrozen-block changes are restored after each fit.
- Frozen, label-independent visual features can be cached across folds. Tuned
  encoder features are recomputed from images on every step.
- Neural early stopping uses **the first inner split**, then starts a fresh fit
  on the complete outer-training set for the selected duration. It does not average
  early stopping over all inner folds. CLI hyperparameters are fixed recipes;
  comparing many recipes on outer scores also constitutes model selection.
- Classical hyperparameters use all inner folds. The hybrid uses another internal
  cross-fit to obtain residual labels without training on its validation faces.
- Exact RGB duplicates are grouped. Add `--groups /absolute/path/groups.json`
  containing `{ "image_filename.png": "identity_or_latent_family", ... }` for
  related identities or generator variants. Every selected filename must be covered.
- All methods use the same seeded outer groups. Only faces with complete means
  and SEs for every requested target are eligible. Missing-target masking and
  additional auxiliary tasks are not implemented.
- Multiple patch budgets are concatenated for classical readouts; neural models
  average their normalized global vectors. Local pooling uses the last budget.
  Dense caches use float16 on CPU and are not written to disk.
- Only the two-view configuration supports the complete original kernel grid;
  a single view uses the applicable full-bank, non-feature-centered candidates.
- `holdout` evaluates one outer split; `cv` writes complete out-of-fold predictions;
  `fit` evaluates a holdout first, then trains a final model on every eligible face.
  Final-fit predictions are never reported as validation accuracy.
- Results include per-target metrics, pooled MSE, fold IDs, row indices, paths,
  groups, seed, data/source hashes, model revision, and training histories.

The historical RMSE 0.0697 used 50/10 CV. Compare baseline and candidate **in the
same new run protocol**, not a new five-fold score against that historical number.
Paired uncertainty analysis and additional seeds belong to the finalist evaluation;
they are not performed automatically by this training CLI.

## Short validation, no model checkpoints

Completed on **19 September 2026**, using pinned **So400m on MPS**:
all 12 methods, all five supported joint LoRA combinations, pairwise/distribution/
ordinal auxiliaries, feature/Gram/no preservation, and four-fold two-view CV.
Across 24 method/configuration evaluations, 38 neural optimizer updates passed
the runtime checks. **Zero model checkpoints were saved.** See
[validation.json](validation.json) for exact configurations and parameter-update
evidence; raw metrics and predictions are under `/tmp/fgclip2-adaptation-validation`.
The long training runs, full hyperparameter search, checkpoint export path, and
production inference were not validated by these smoke runs.

Run from the parent project directory, outside the sandbox so PyTorch sees MPS:

```bash
cd /Users/adamsobieszek/PycharmProjects/psych_gen_app
PY=/opt/anaconda3/envs/manip311/bin/python

"$PY" -m CLIP.fgclip2_adaptation --methods all --smoke --blocks 1 \
  --output /tmp/fgclip2-adaptation-smoke
```

Smoke mode samples 32 complete faces, uses one optimizer update in the inner
selection fit and one in the fresh outer-training fit, disables warm-up, and
limits the classical kernel grid. It asserts finite losses/gradients/predictions
and actual changes in the method-specific adaptation parameters (and backbone
parameters for live adaptation). Default batch size is 4. `--limit` can override
the sample size. It forcibly rejects checkpoint saving. Without `--cache-dir`,
its frozen feature cache is temporary and removed afterward.

## Long evaluation commands

These run all eligible faces, So400m, five outer folds, and four inner folds
(neural duration selection uses the first inner split as explained above).
Each output directory must be new or empty. Metrics and held-out predictions are
saved, **model checkpoints are not**. The defaults are starting recipes, not tuned
hyperparameters or guarantees that 30 epochs suffice.

```bash
cd /Users/adamsobieszek/PycharmProjects/psych_gen_app
PY=/opt/anaconda3/envs/manip311/bin/python
RUN=CLIP/fgclip2_adaptation/runs

# Complete-embedding baselines and phrase-complement residuals; same two views.
"$PY" -m CLIP.fgclip2_adaptation \
  --methods phrase-kernel ridge kernel hybrid --mode cv --patches 128 256 \
  --output "$RUN/two-view-baselines"

# Frozen-backbone adaptations, plus a matched single-view reference.
"$PY" -m CLIP.fgclip2_adaptation \
  --methods phrase-kernel adapter prototypes phrase-adapter local \
  --mode cv --patches 256 --epochs 30 --batch-size 16 \
  --output "$RUN/frozen-adaptations"

# Shared learned text context through the frozen text transformer.
"$PY" -m CLIP.fgclip2_adaptation --methods soft-prompt \
  --mode cv --epochs 30 --batch-size 32 --context-tokens 4 \
  --output "$RUN/soft-prompt"

# Actual visual encoder adaptation, last four blocks, Q/V rank 4.
"$PY" -m CLIP.fgclip2_adaptation --methods lora \
  --mode cv --epochs 30 --batch-size 4 --blocks 4 --rank 4 \
  --encoder-lr 5e-5 --output "$RUN/lora"

# Native pooling probe; then limited unfreezing at a smaller learning rate.
"$PY" -m CLIP.fgclip2_adaptation --methods pool \
  --mode cv --epochs 30 --batch-size 4 --output "$RUN/pool"
"$PY" -m CLIP.fgclip2_adaptation --methods partial \
  --mode cv --epochs 30 --batch-size 2 --blocks 1 --encoder-lr 2e-6 \
  --output "$RUN/partial"

# Joint language and visual adaptation. Run these after the simpler controls.
"$PY" -m CLIP.fgclip2_adaptation --methods prototypes soft-prompt local \
  --visual-lora --mode cv --epochs 30 --batch-size 4 --blocks 4 --rank 4 \
  --output "$RUN/joint-lora"

# Auxiliary-loss ablations with the same visual adapter and splits.
for LOSS in pairwise ordinal distribution; do
  "$PY" -m CLIP.fgclip2_adaptation --methods adapter \
    --mode cv --epochs 30 --batch-size 16 --auxiliary "$LOSS" \
    --output "$RUN/adapter-$LOSS"
done

# Proposed encoder + uncertainty-aware ordering recipe.
"$PY" -m CLIP.fgclip2_adaptation --methods lora \
  --mode cv --epochs 30 --batch-size 4 --blocks 4 --auxiliary pairwise \
  --output "$RUN/lora-pairwise"

# Geometry-preservation ablation against the default pointwise penalty.
"$PY" -m CLIP.fgclip2_adaptation --methods lora \
  --mode cv --epochs 30 --batch-size 4 --blocks 4 --preserve gram \
  --output "$RUN/lora-gram"
```

For finalists, repeat candidate **and reference** with `--seed 20260920` and
`--seed 20260921`, each in separate output directories. Use `--folds 50
--inner-folds 10` only when the extra fits are warranted; this does not recreate
historical saved splits automatically. Use `--mode holdout` for cheaper triage.

So400m is large for a 16 GB Mac. The short validation does not establish that a
full run with four trainable blocks fits. If needed reduce `--batch-size 2
--encode-batch 2 --blocks 1`, keeping settings consistent across comparisons.
For frozen-image soft prompts, a larger image batch amortizes the shared text
forward pass; it does not multiply the number of prompts traversing the text tower.
Late trainable visual blocks and soft text context use activation checkpointing.
`--no-checkpointing` disables that memory optimization; it does **not** control
checkpoint files. `--model base` is an explicit different-backbone experiment,
not a transparent substitute for So400m. Checkpoints must already be cached by
the core loader; these commands do not download models.

## Optional final research weights

After choosing a recipe, explicitly request a final all-data fit and weights:

```bash
"$PY" -m CLIP.fgclip2_adaptation --methods lora --mode fit \
  --epochs 30 --batch-size 4 --blocks 4 --rank 4 --auxiliary pairwise \
  --output "$RUN/final-lora-pairwise" --save-checkpoint
```

This writes a versioned **research state**, including head/adaptation tensors,
configuration, model revision, target order and prompts; frozen pretrained weights
are not duplicated. Classical fits store their readout. These `.pt` files are
trusted-local Python research artifacts, not `RacePerceptionPredictor` bundles.
An inference/export adapter to the production application is deliberately outside
this implementation. Load by reconstructing the matching `RatingNetwork` and
`BackboneSession` with the recorded config; partial and LoRA parameter names refer
to those temporary wrappers. Retain the source hash/version alongside weights.

Fixed-region baselines, mixtures of prompt experts, full text-tower fine-tuning,
additional unlabeled data and end-to-end production inference are deferred. The
implemented `hybrid` is the phrase-complement residual experiment; it is not yet
a generic ensemble of arbitrary trained neural models with the production model.
