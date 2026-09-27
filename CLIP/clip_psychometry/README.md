# CLIP psychometry

A separate research package for diagnosing **visual cue measurement** and **joint prediction of human impression intensity**. Its per-target AGOP selector now powers the `fgclip2_multitarget` default; legacy/neural variants remain available. See [current model](../fgclip2_multitarget/CURRENT_MODEL.md).

Start with [the methodology](docs/METHODOLOGY.md), [the compressed agent brief](docs/AGENT_BRIEF.md), and [the empirical ledger](../research/clip_psychometry/INSIGHTS.md). The textbook derives the estimands before their implementation, labels literature/observations/hypotheses separately, and maps ambiguous failure modes to discriminating measurements.

## What is available

| Operation | API | Methodology |
|---|---|---|
| Deployable nested per-target AGOP selection | `target_kernel.fit_target_kernel`, `predict_target_kernel` | All-target pruning / promotion |
| Matched full-CV report and per-target uncertainty | `cv_report.compare_run` | Promotion into the full-CV pipeline |
| Explicit phrase/channel/construct identity and removal closure | `Column`, `Registry`, `Dataset` | §§2, 13 |
| Import cached historical features, or current `Assets`/`Ratings` | `adapters.from_legacy_features`, `from_assets` | §§7, 13 |
| Score new wording using existing image features, build a union bank | `adapters.encode_candidate_phrases`, `extend_dataset` | §§3–4 |
| Nested group-disjoint kernel comparison of entire banks | `evaluation.compare` | §6 |
| Exact current frozen-kernel grid and ensemble reuse | `ComparisonConfig(backend='legacy')` | §§6.3, 7 |
| Raw/rank/hinge response scales; equal role mass | `features.FeatureMap`, `Representation` | §§4.2, 7 |
| Conditional group utility and four-bank complementarity | `metrics.paired_contrast`, `complementarity` | §6 |
| Extreme, score-range, conjunction and residual image cards | `cards.*` | §4 |
| Blinded stratified annotation sheets and weighted validation | `cards.annotation_sheet`, `metrics.annotation_metrics` | §4.1 |
| Reflective factor, MTMM, conditional anchor/context diagnostics | `psychometrics.*` | §5 |
| Paired view drift, repeatability and prediction sensitivity | `robustness.*` | §8 |
| Reuse existing device-side augmentation pipeline | `adapters.augmentation_scores` | §8 |
| Noise accounting, support cells and training-neighbor distances | `metrics.*` | §§9–11 |
| Evidence routing with caller-specified thresholds | `diagnostics.evidence_routes` | §10 |
| Differentiable centered kernel solve for later prompt training | `differentiable.kernel_predict` | §12.2 |

The generic kernel uses the same centered spectral algorithm as the existing kernel code, extended to arbitrary single-channel/empty/transformed feature sets. It deliberately has a smaller explicit grid, selects one recipe per target, and does **not** reproduce the historical selector's ensembles. For comparisons to the current multi-target frozen-kernel baseline use `backend='legacy'`, which directly calls the existing `candidate_grid`, `inner_predictions`, `select_strategies`, `fit_bundle_readout`, and `predict_readout`. That backend uses the current multi-target **both-channel, uncentered-score** recipe subset, with the existing kernel centering. It requires both aligned channel copies of every selected phrase and raw response scales. Hyperparameters in `config.kernels`/`alphas` apply only to the generic backend; the legacy backend uses its recorded existing grid.

## Local workflow

From the `CLIP` directory, use `/opt/anaconda3/envs/manip311/bin/python`. Analysis dependencies are NumPy, SciPy, scikit-learn, threadpoolctl, and Pillow, already used in this repository. Pure analysis does not import PyTorch. No dependency installation or model download is needed for existing scores.

Import only the historical development partition:

```bash
/opt/anaconda3/envs/manip311/bin/python -m clip_psychometry import-legacy \
  --features research/trustworthiness/p128/trustworthy-features.npz research/trustworthiness/p256/trustworthy-features.npz \
  --channels p128 p256 \
  --phrase-groups research/trustworthiness/phrase_groups.json \
  --protocol research/trustworthiness/protocol.json \
  --output research/clip_psychometry/dev.npz
```

The optional `--identity-groups` takes a filename-to-identity/latent-family JSON mapping. Without it, each image ID is a separate group: identity independence is **not established**. The original protocol's image names must match exactly before its development indices are used. A new dataset import requires an explicit exposure declaration before image cards can be produced.

```bash
/opt/anaconda3/envs/manip311/bin/python -m clip_psychometry compare \
  --data research/clip_psychometry/dev.npz \
  --plan clip_psychometry/examples/trustworthiness_plan.json \
  --output research/clip_psychometry/comparison
```

A plan specifies alternatives using explicit `columns`, `keep_roles`, or `drop_roles`; transforms and balanced role weights are optional. Every variant is evaluated on the same outer folds. All preprocessing and hyperparameter selection occur inside each fold; targets with matching missingness share eigendecompositions. An empty bank is a training-mean baseline. The saved outer scores compare **fixed proposals**; choosing a winner from those scores is development, not an unbiased evaluation of that further selection.

```bash
/opt/anaconda3/envs/manip311/bin/python -m clip_psychometry audit \
  --data research/clip_psychometry/dev.npz --output research/clip_psychometry/detectors.json
/opt/anaconda3/envs/manip311/bin/python -m clip_psychometry cards \
  --data research/clip_psychometry/dev.npz --output research/clip_psychometry/cards
```

`cards` defaults to one image for every column (phrase × channel): four lowest and four highest scoring independent groups, with large phrase text, labels and JSON manifests. Limit generation with `--columns 0 16 20`. Add `--predictions .../predictions.npz --variant full` for residual cards. There is no automatic LLM image upload or validity verdict. Reading error cards consumes those images for development purposes.

```bash
/opt/anaconda3/envs/manip311/bin/python -m clip_psychometry annotations \
  --data research/clip_psychometry/dev.npz --column 225 --per-stratum 10 \
  --output research/clip_psychometry/annotations
```

Choose the column from `detectors.json`; the number above is illustrative. Annotators fill the blinded CSV with visible evidence and unclear cases. Keep the separate score/sampling-design file hidden during annotation. Weighted metrics require the saved inclusion probabilities; score cutoffs and probability calibrators must be fit on separate annotation-training data.

## A focused, reproducible development study

```bash
/opt/anaconda3/envs/manip311/bin/python -m clip_psychometry.examples.development_study
```

This uses the cached trustworthiness data and original 400 development images. It compares direct adjective-family removal, authenticity-role removal including four declared conjunctions, single “insincere smile” removal, and their joint background. It saves all OOF predictions, exact folds, selected recipes, source/data hashes, paired contrasts, modifier associations, 17 targeted cards, and a blinded annotation sheet. Two contrast cards and their numeric diagnostics were added as an explicitly exploratory follow-up. The 604 historical evaluation images are excluded. No new encoding, backbone training, or augmented-view experiment is run. [Results and interpretation](../research/clip_psychometry/REPORT.md).

## Python: functional alternatives and complementarity

```python
from clip_psychometry import Representation, ComparisonConfig, compare
from clip_psychometry.metrics import complementarity

# Indices refer to explicitly registered columns, with ALL channel copies.
B = set(background_columns)
A, D = set(component_a_columns), set(component_b_columns)
assert not (B & A or B & D or A & D)
variants = {name: Representation(tuple(sorted(cols))) for name, cols in {
    'base': B, 'a': B | A, 'b': B | D, 'joint': B | A | D,
}.items()}
report, predictions = compare(data, variants, config=ComparisonConfig())
result = complementarity(data.y[:, 0], *(predictions[k][:, 0]
    for k in ('base', 'a', 'b', 'joint')), groups=data.groups)
```

For lexical-versus-component alternatives, compare `B+{A,B}`, `B+{AB}`, and `B+{A,B,AB}`; use `cards.conjunction_card` to inspect retrieval disagreement separately. A predictive win cannot choose the more valid detector without annotation.

## Existing model and augmentation integration

Run repository-specific adapters from the **parent** directory, importing `CLIP.clip_psychometry`. These paths lazily import existing PyTorch modules; use the specified Python outside the sandbox locally so MPS remains available. The pure analysis commands above do not need that escalation.

```python
from CLIP.clip_psychometry.adapters import from_assets, augmentation_scores
from CLIP.clip_psychometry.robustness import augmentation_audit

# assets/data/args come from existing setup; registry matches its exact text order.
portable = from_assets(assets, ratings, registry)
views, provenance = augmentation_scores(assets, audit_rows, args, views=4, seed=24)
audit = augmentation_audit(portable.x[training_rows], portable.x[audit_rows], views)
```

The adapter uses the current tensor transformations and image/phrase scoring functions. It does not enter `TrainingViews`, allocate its on-disk augmentation bank, change its ownership, or delete caches. For a CLI audit, save `views` and matching `ids` in an NPZ and pass it to `views --view-scores ... --training-reference ...`; cue/target-specific label preservation is still the researcher's responsibility. Feature-space noise is not included by this adapter; it audits image transformations.

For new wording, `encode_candidate_phrases(original_assets, phrases)` reuses cached original-image features and already-present phrase scores, encoding only unseen text. Its output columns are all candidate global-short scores followed by all candidate local-box-max scores. Supply matching `Column` records with unique IDs to `extend_dataset`, then compare functional subsets of the resulting union bank. The caller must preserve exact image order; this function takes the original `Assets`, before any training wrapper. No existing phrase bank, cache, or training configuration is mutated.

`differentiable.kernel_predict` provides a support/query solve for a later optimizer. It is not a trained checkpoint or a complete ManiPT/PromptSRC reproduction. Its stop-gradient options change the objective and are exposed for explicit comparison. No training defaults have been changed.

## Interpretation boundaries

The real-data development study exercises the generic comparison and card-generation path. Optional repository adapters, latent-model diagnostics, augmentation audits and differentiable support are provided but have not been run in this task. There are no new unit tests, synthetic training runs, UI checks, or visual inspection of generated cards. The report distinguishes numerical observations from unverified visual hypotheses.

Conditional bootstrap intervals concern the saved predictions, omitting retraining/search uncertainty. Unlabeled detectors remain unvalidated. Source score caches are reused as supplied; no new image encoding or image-identity audit is implied. Advanced identified MTMM/IRT models, controlled image edits with new human ratings, automatic LLM review and the standalone soft-prompt trainer remain later work, with their requirements specified in the methodology.

Iteration tools: `geometry.compare_geometry` compares role-PCA metrics with matched
product-kernel controls; `retrieval.retrieval_audit` reports criterion precision and
missing-annotation bounds; `retrieval.risk_partition` decomposes paired error gains.
`cards.blind_contact_sheets` supplies randomized ID-only annotation sheets.
`text_encoder.FrozenTextEncoder` reuses the official text path and streamed weights
without allocating the unused vision encoder. This research-only loader cannot
encode images. See [four-study report](../research/clip_psychometry/iteration_01/REPORT.md)
and its candidate manifests for opt-in development leads and limitations.

Current-bank engineering: [iteration 02](../research/clip_psychometry/iteration_02/REPORT.md)
uses the original global/local extraction and frozen-kernel selector. Fold-local
`Representation` options include `metric_weights`, `separate_metric`,
`contrast_pairs`, `contrast_weight`, and `normalize_contrast_energy`.
`ComparisonConfig(selector='existing_ensemble')` reuses the current selector after
these transforms (run as `CLIP.clip_psychometry` from the parent). The exact `legacy`
backend accepts raw banks only and rejects transforms it cannot apply. The saved
candidate overrides add phrases; contrast behavior requires this analysis adapter.

Per-target pruning: `pruning.rank_paths` supplies diagonal AGOP/RFM scores;
`frozen_readout_agop` attributes the actual selected kernel ensemble;
`select_units` retains phrase-channel bundles or declared roles;
`fit_phrase_pruner(data, target, keep=...)` exports a portable trained mask for an
externally validated retention setting. Rankings are model sensitivities, not
causal or unique importance. `pruning_study.run` evaluates mask learning and
retention inside nested folds with the original readout selector and all-target
risk summaries. See [iteration 03](../research/clip_psychometry/iteration_03/REPORT.md)
for all 34 targets, retention tradeoffs, baseline guards and per-phrase audits.
