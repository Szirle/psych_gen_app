# Frozen perceived-trustworthiness model

Predicts mean subjective slider ratings using 256 phrases, FG-CLIP2 So400m, and an equal-weight blend of centered RBF kernel ridge and quadratic kernel ridge. The target is perceived trustworthiness, not actual character.

## Predict new images

Run from `/Users/adamsobieszek/PycharmProjects/psych_gen_app` using `/opt/anaconda3/envs/manip311/bin/python`. The environment and cached encoder weights are already available. MPS requires execution outside the Codex filesystem sandbox.

```python
from CLIP.fgclip2_face_impressions import TrustworthinessPredictor

model = TrustworthinessPredictor.load(
    '/Users/adamsobieszek/PycharmProjects/psych_gen_app/CLIP/research/trustworthiness/trustworthiness_model.joblib'
)
means = model.predict_images(['/path/to/face_a.png', '/path/to/face_b.png'])
# Same order as input. No labels or fitting on these images.
difference_b_minus_a = float(means[1] - means[0])
```

For a directory, replace `/path/to/hidden_faces` and choose an output directory:

```bash
/opt/anaconda3/envs/manip311/bin/python -m CLIP.fgclip2_face_impressions trust-predict \
  --bundle CLIP/research/trustworthiness/trustworthiness_model.joblib \
  --images /path/to/hidden_faces \
  --output /private/tmp/trustworthiness_predictions
```

This writes `predictions.csv` with `face,predicted_trustworthy_mean`. Discovery is nonrecursive and uses natural filename order. The Python API accepts an explicit ordered path list and is preferable for pairing variants.

`predict_images(..., return_scores=True)` also returns the 512 score features. `predict_scores(scores)` bypasses encoding but requires the exact stored contract: 256 phrases at 128 patches followed by the same 256 at 256 patches, pinned So400m revision, short text head, normalized embedding logits including learned scale/bias. Do not provide softmax probabilities or reorder phrases. Output is unclipped in the original stored slider units.

The predictor caches its encoder and text embeddings for subsequent calls. Use `batch_size=1` if image memory is constrained. Fresh-image/cache/batch checks on seven original images are recorded in `inference_validation.json`.

## Frozen configuration

- Encoder: `qihoo360/fg-clip2-so400m`, revision `d57d30fe94a107dd6a2610eb4e9a135004d823a4`.
- Encoding: MPS float16, short text mode, 128/256 patch budgets.
- Regression: float64, scikit-learn 1.8.0; trained on all 1,004 original faces.
- Component 1: subtract each image's mean phrase score separately at each resolution, standardize columns using training statistics, RBF KRR with gamma `0.1/512`, alpha `0.1`.
- Component 2: raw p256 scores, training-column standardization, quadratic KRR with gamma `1/256`, coef0 `1`, alpha `0.1`.
- Equal 0.5 weights; each component adds its fitted training target mean.
- `model_manifest.json` records hashes and versions. Joblib contains sklearn estimators; load only trusted bundles.

## What was validated

The complete selection procedure achieved r=.9380, R²=.8795, RMSE=.04333 and MAE=.03386 on 604 development-excluded faces in nested ten-fold CV. The 400 phrase-development faces were always in outer training. These are procedure-level estimates; the final all-data model has no independent test result yet.

High-region RMSE was .03900; close-pair ordering was 68.7% across 438 same-fold pairs with rating gaps .01–.05. These pairs are not verified same-identity variants. Nonlinear prediction improved mean-rating error but did not improve ordering over ridge. Only three close pairs passed an approximate SEM-based separation screen, so that subset provides little evidence. The additional 1,560 study-stimulus rating records were excluded. No hidden test images were evaluated.

## Reusable research pipeline

All functions live in `CLIP/fgclip2_face_impressions.py`:

| Function | Purpose |
|---|---|
| `extract_impression_features`, `load_trust_data` | Cached logits, ratings and aligned feature banks |
| `trust_recipe_grid`, `evolve_trust_groups` | Finite recipe search and evolutionary cue-group subsets |
| `trust_candidate_cv` | Complete inner OOF predictions with training-only preprocessing |
| `choose_trust_strategy` | MSE-based recipe or convex-blend selection |
| `trust_outer_cv` | Evaluation with development faces kept in training |
| `fit_trust_recipe`, `fit_trust_strategy`, `predict_trust_readout` | Reusable readouts |
| `trust_tail_metrics` | High-rating error and close-pair ordering |
| `evaluate_trustworthiness` | Evaluation, final fit and artifact export |
| `TrustworthinessPredictor` | Label-free image or score inference |
| `validate_trust_inference` | Fresh-image/cache/batch numerical checks |
| `report_trustworthiness` | Regenerate report and figures without fitting |

CLI stages are `trust-encode`, `trust-explore`, `trust-evaluate`, `trust-validate`, and `trust-report`, each with `--output CLIP/research/trustworthiness`. Encoding accepts `--images` and `--ratings`; defaults point to the original images and ratings file. The frozen bank is `phrase_groups.json`; provenance is `literature.json`. Exploration uses seed 20260910 and 400 development faces. For a new experiment, copy those bank/provenance files into a new output directory to preserve this frozen experiment.

```bash
# Rebuild presentation only:
MPLCONFIGDIR=/private/tmp/fgclip-mpl /opt/anaconda3/envs/manip311/bin/python \
  -m CLIP.fgclip2_face_impressions trust-report --output CLIP/research/trustworthiness

# Numerical, cache, serialization and held-out-label leakage tests:
/opt/anaconda3/envs/manip311/bin/python -W error -m CLIP.test_fgclip2_research
```

`evaluation_source.py.txt` preserves the evaluation implementation. Later report/deployment additions have separate runtime/report source hashes. The report includes all 256 phrases, 14 literature sources, all tested model families and high-end diagnostics. Figures are exported as PNG and SVG.
