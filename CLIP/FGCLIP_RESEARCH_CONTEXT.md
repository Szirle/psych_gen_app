# FG-CLIP: method and maintenance context

## Task and data

Predict **mean subjective human judgments of generated faces**, not objective labels. For example, an age rating is perceived age on a slider, not chronological age; race-related ratings describe participants’ impressions, not verified ancestry.

The current dataset contains **1,004 StyleGAN2-generated faces**, with over 30 continuous-slider ratings per face/target. The joint model predicts eight independent rating means: `asian`, `middle-eastern`, `hispanic`, `islander`, `native`, `black`, `white`, and `skin-color`. These are regression outputs, not mutually exclusive classes or probabilities summing to one. Incorporating StyleGAN latents is future work.

- Original images: `/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images`
- Ratings: `data/dim_to_photo_to_ratings.pkl` — trusted local mapping `dimension → filename → individual ratings`.
- Match the intended original images explicitly: the ratings file also contains other study stimuli. Use `limit=None` when discovering all faces; the dataset-config default is 1,000.

## Method to retain

**Frozen FG-CLIP2 encoder → phrase agreement logits → supervised kernel regression.**

Use FG-CLIP2 So400m (`qihoo360/fg-clip2-so400m`); the model bundle pins the revision, dtype, phrase order, and encoding configuration. A phrase score is `exp(logit_scale) × dot(normalized_image, normalized_text) + logit_bias`. It is not a probability.

The race model uses **516 phrases at each of two patch budgets**, 128 and 256: 1,032 features ordered as all p128 phrases, then all p256 phrases. Patch budgets are not pixel dimensions. Keep the phrase bank intact; it defines the training inputs and target order.

The original kernel model is good enough. Keep its linear/quadratic/RBF candidate selection and per-target regularization/blending; do not restart alternative-architecture research unless asked. Its pooled 50-fold held-out RMSE was about **0.0697**. The saved bundle is the authority for inference, not a reconstructed recipe from these notes.

## Stable code contracts

Prefer these boundaries over experiment-specific functions:

| Location / API | Responsibility |
|---|---|
| `CLIP/fgclip2_core.py` | Encoder and preprocessing; universal infrastructure |
| `CLIP/fgclip2_face_impressions.py` | Shared datasets, encoding/cache, ratings, fitting, evaluation, results, inference, and optional plotting |
| `FaceDatasetConfig`, `FaceImpressionPipeline`, `HumanRatingsStore`, `RegressionDataset` | Discover/encode faces and align human ratings with phrase scores |
| `fit_score_model`, `predict_score_model` | Training-only preprocessing and readouts; accept standard sklearn estimators when needed |
| `make_cv_splits`, `cross_validate_predictor`, `evaluate_predictors` | Shared evaluation; fitting callbacks receive training data and held-out features |
| `RegressionResults` | `y_true[rows, targets]`, `predictions={method: array}`, target names, optional fold IDs and metadata; metrics indexed by method then target; `summary()` is JSON-ready, `metric_table()` is concise Markdown |
| `Viz.oof_scatter`, `Viz.export` | Plot/export the results contract; add future result visualizations here |
| `ImpressionPredictor` | Frozen inference bundle: `phrases`, `patches`, `encoding`, `targets`, `readout`; an optional readout callback supports specialized models |
| `CLIP/race_perception/` | Three files: kernel API in `__init__.py`, phrase bank in `phrases.py`, CLI in `__main__.py` |
| `CLIP/fgclip2_legacy_research.py` | Existing historical experiment policies and old-contract adapters; not the template for new shared APIs |

Race API: `from CLIP.race_perception import RacePerceptionPredictor`.
CLI: `python -m CLIP.race_perception {prepare,train,predict,verify}`.
Existing kernel artifacts are under `CLIP/research/race_perception/`; load a specifically requested artifact rather than exploring the archive.

## Scientific pitfalls

- Fit scaling, feature selection, kernel centering, tuning, and blend selection **inside training folds**. Dataset-wide standardization is not CV preprocessing.
- Keep related identities/latent variants together; group exact image duplicates. The original protocol used 50 outer folds and 10 inner folds, then fitted the final model on all eligible faces. Fitted-data accuracy is not validation accuracy.
- Select for MSE/RMSE in original slider units; correlation alone misses calibration errors. Preserve input/target order, encoder identity, split IDs, and held-out predictions.
- Previously inspected faces do not become a fresh test set. Bootstrap intervals over saved predictions are conditional on fitted models, not retraining/search uncertainty.
- Optional binary diagnostics use human/predicted means **>0.5**, exclude skin color, and balance positive/negative recall. Extreme-rating subsets can have very few positive faces; zero errors there is weak evidence about rare failures.

## Keep the code small

For each helper or proposed addition:

1. **Already covered?** Reuse the shared helper and adapt the caller to its contract; delete the duplicate.
2. **Existing contract too narrow?** Extend it concisely for a concrete use, then remove the special case.
3. **Distinct, likely reusable capability?** Put it in the shared module with a standard array/dataclass/sklearn contract. Reuse `RegressionResults` and `Viz` instead of inventing another report format.
4. **Only an old experiment policy, fixed search budget, or ad hoc adapter?** Keep it beside its actual legacy caller if still needed; otherwise delete it. Renaming a `trust_*` function does not make it reusable.
5. **No useful caller or future role?** Delete it. Avoid compatibility layers, new files, and speculative abstractions whose maintenance costs exceed their value.

Check live callers when changing an API. Preserve working model capabilities, not every historical function name. Keep reports to useful metrics/figures rather than generated narrative.

## Working rules

- **Do not browse, search inside, or clean `CLIP/research/`. It is an archive**, not the active codebase. Read an exact artifact only when the task requires it.
- Do not inspect or modify universal encoder/demo infrastructure for model-specific cleanup unless requested. Shared-module edits should serve an actual reusable capability.
- Use `/opt/anaconda3/envs/manip311/bin/python`. Code importing PyTorch must run outside the sandbox because it otherwise hides MPS. Kernel numerics use CPU; image encoding uses MPS.
- Prefer established CLI commands, quiet output, and targeted reads. For ordinary research/documentation work, do not run unit tests, visual verification, or superfluous checks unless requested.
- **Never poll long training jobs once per fold or send routine per-fold updates.** Use completion signals or infrequent, long waits. If a simple action fails and a costly investigation would be slower than the user fixing it, ask the user instead.
