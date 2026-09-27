# B2 deployment and ten-fold evaluation

The default `agop-kernel` now means expanded-bank continuous AGOP weighting (B2).
It uses all 852 phrases for all 34 targets, with no bank exclusions or hard pruning.
See [CURRENT_MODEL.md](CURRENT_MODEL.md) for adoption evidence. Training is deliberately
left to the user; no ten-fold results exist until the command finishes.

## Selection, refitting and combining models

Run from the directory containing `CLIP/`:

```bash
/opt/anaconda3/envs/manip311/bin/python -m CLIP.fgclip2_multitarget.production \
  --source CLIP/research/clip_psychometry/default_full_cv \
  --output CLIP/research/clip_psychometry/production_b2 \
  --folds 10 --inner-folds 10 --cpu-threads 1
```

This uses the validated, frozen full-image score cache and original individual ratings;
it does not reload the encoder or regenerate scores. Exact cache path/phrase order,
label alignment, grouping and source identity are checked. To relocate the source
assets, use `--score-cache` and `--ratings`. Restart the identical command to resume
completed folds; changed data/configuration or training source code are rejected.

Ten group-preserving outer folds each hold out approximately 10% of the images.
Ten inner folds fit candidates on approximately 81% of the full dataset and
validate each on approximately 9%; all outer-training rows contribute inner OOF
predictions. Each target independently selects one of the six B2 metric options
and a kernel strategy using pooled inner MSE. The two-fold pilot inside each fit
is unchanged from B2. The selected metric and readout are then **refitted on all
90% outer-training images**, including former inner validation rows. Group sizes
can cause small differences from these percentages. This is a well-established
nested model-selection procedure, not a claim of new or universally SOTA accuracy.
[Reference: nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

All ten refitted models are retained. The deployment artifact `ensemble.json`
combines their predictions with fixed equal weights. It references checkpoint
files using relative paths and SHA256 hashes, avoiding duplicate copies of large
AGOP factors. Copy the **whole output directory**, not just the JSON, to deploy.
The retained checkpoints can occupy several GiB; the ensemble JSON avoids a second
copy of those weights. The image encoder runs once per input; score predictions are evaluated one model
at a time so all ten checkpoints need not reside in memory. This is exact
prediction-level averaging: metrics and dual coefficients from different kernels
cannot in general be averaged into one equivalent kernel. There is no learned
stacking, outer-score weighting, model selection by outer outcomes, or extra
all-image fit. Each training image participates in nine ensemble members.

```bash
/opt/anaconda3/envs/manip311/bin/python -m CLIP.fgclip2_multitarget.predict \
  --checkpoint CLIP/research/clip_psychometry/production_b2/ensemble.json \
  --images /path/to/new/images --output /path/to/predictions.csv \
  --device mps --batch-size 1
```

`oof_predictions.npz` and `oof_predictions.csv` contain exactly one held-out
prediction per image/target, in original order, with fold, path, observed mean,
rater count and mean standard error. `REPORT.md`/`metrics.json` report every fold
and pooled OOF Pearson correlation, MSE and RMSE. Primary aggregate metrics omit
looks-like-you and memorable; all 34 targets remain in files and figures.
**These evaluate the single held-out fold learner, not the final ten-member
ensemble.** The other nine models trained on that image. An unbiased ensemble
score requires external images or another level of outer evaluation; this pipeline
makes no claim that averaging has a measured accuracy gain. Earlier exploratory
use of these same 1004 images also means this is internal, not external validation.

## Prediction ceiling for noisy image means

For target t and image i let individual ratings be R_ij, n_i their count,
Y_i their mean, and μ_i the expected rating in the sampled perceiver population.
Write Y_i = μ_i + ε_i, with E[ε_i | image] = 0. Estimate the conditional variance
of the mean by v_i = s_i²/n_i, using **unbiased sample variance** (ddof=1).
Under independent within-image sampling and independent errors across images,

    V_observed = sample variance across image means (ddof=1)
    V_noise = mean_i(v_i)             # equal image weights, not rater weights
    V_signal = V_observed - V_noise
    reliability_mean = V_signal / V_observed
    Pearson ceiling = sqrt(reliability_mean).

Why subtract the mean variance despite heteroskedasticity? For centering matrix
H = I − 11'/N, E[ε'Hε]/(N−1) = mean_i(v_i) when Cov(ε) is diagonal.
More generally the correction is tr(HΣ)/(N−1). The available rating lists lack
rater identities, so shared-rater covariance cannot be identified here. Systematic
rater composition or correlated errors would require that richer covariance
model. The ceiling assumes population means are predictable from the image;
unobserved context can impose a lower achievable ceiling. It is an estimate,
not a proven absolute maximum for a finite sample.

With independent zero-mean measurement error, Corr(f(X),Y) =
Corr(f(X),μ) sqrt(V_signal/V_observed). The theoretical maximum is attained
when the image predictor recovers μ up to positive affine transformation. Thus
**reliability is a squared-correlation ceiling, not the Pearson ceiling itself**.
This is neither split-half reliability nor inter-rater correlation. Gaussian
rating errors are unnecessary; bounded sliders and different n_i are allowed.
A negative estimated signal variance is flagged, with zero used only as the
constrained plotted estimate; raw reliability remains in the report. Constant
mean targets have undefined correlation/ceiling. Observed r is never clipped to
the estimated ceiling. See the [noise-ceiling derivation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6426260/).

`prediction_ceiling.png` and `.pdf` follow the supplied figure's horizontal black
performance bars and salmon ceiling markers/background, ordered by descending
estimated ceiling. Their x-axis is **Pearson r**, not R² or percentage accuracy.
No invented SOTA-gain segment is drawn. Both target-level raw estimates and
image-group bootstrap intervals appear in `metrics.json`. These intervals
resample observed image/rating summaries jointly, conditional on the fitted OOF
predictions; they do not include retraining/model-selection uncertainty or recover
unknown repeated-rater covariance. This is the relevant limit on interpretation,
not an invitation to choose a model from the displayed test scores.
