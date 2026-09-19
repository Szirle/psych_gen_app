# Scientific and product intent

## Purpose

Help researchers construct realistic face stimuli for experimental conditions
that vary specified psychological impressions while limiting unrelated changes.
The intended workflow is to define a control population, choose manipulated and
controlled variables, inspect matched variants, and generate reproducible stimuli
for an experiment. Fast previews serve that workflow; producing an attractive
image grid alone is not its scientific endpoint.

Scientific source: *Theory of optimal experimental manipulation with generative
models via reverse correlation of stimuli*, the author's manuscript identified in
[documentation-map.md](documentation-map.md). This is an interpretation of that
manuscript for the app, not a claim that all its methods are already integrated.

## Preserve the paper's distinctions

- **Impressions are perceived attributes.** The modeled quantity is the expected
  human judgment elicited by a stimulus, not an objective property of the person
  depicted. This applies to trustworthiness, intelligence, age, and social categories.
- **Prediction and manipulation are different objectives.** A more accurate
  predictor need not yield a better manipulation direction. The paper explicitly
  motivates this distinction through reverse correlation and shrinkage.
- **Local direction and finite path are different problems.** A differentiable
  impression model supplies a local gradient; following it over a longer path
  can leave the distribution where its predictions remain useful.
- **Plausibility belongs to the method.** The paper motivates support preservation
  to avoid unrealistic feature combinations and unreliable impressions, especially
  for valence-laden traits. Stronger predicted trustworthiness alone is insufficient.
- **Control is a path constraint.** Filtering the starting population or removing
  a linear component does not establish Conditional Support Preservation (CSP).
  The paper requires paths to remain within the feature distribution conditional
  on the controlled values. Do not describe a checked UI box as empirical control.
- **Dose needs interpretation.** The paper seeks a stable relationship between
  manipulation amount and impression change. Current slider strength is a latent
  displacement parameter, not a calibrated change in human ratings.

## Method destination

Retain a clear distinction among linear-direction baselines, nonlinear gradient
paths, and Psychometric Gradient Flow Matching (PGFM). The manuscript's PGFM
recipe fits impression models, samples features and predicted ratings, stratifies
by controlled values, pairs examples using manipulation transport cost
`distance / positive impression difference`, learns a vector field from those
pairings, and integrates it for new stimuli. It describes perceived age/gender
control in its studies; this does not establish arbitrary-variable control in
the application.

The app should make the selected method and its supported controls explicit.
Multi-axis layouts are useful for experimental designs, but do not themselves
prove independent or orthogonal manipulations. Combining nonlinear paths requires
a defined policy; summing linear directions is not automatically transferable.

## Product consequences

Keep stimulus selection distinct from manipulation. Selection defines the control
population; manipulation constructs variants of its sampled identities. Preserve
their relationship in previews and eventual exports. A nine-face selection
preview illustrates a sampling policy and is not already an export dataset.

Low-resolution previews and full-resolution outputs should correspond to the same
latent manipulation. Preserve responsive cancellation and stable coordinates as
methods become more expensive. Record enough method, model, sampling, and latent
information to reconstruct exported conditions and assess their limitations.

Assess target change, unwanted changes, ecological validity, and dose response
separately. Software correctness and empirical validation with human judgments
are complementary; neither substitutes for the other.

The current app implements the linear baseline and interactive preview machinery.
Gradient/PGFM integration, effective user-selected controls, and dataset export
remain work; see [backlog.md](backlog.md). Broader voice, video, text, and stereotype
visualization applications discussed in the paper are not this app's current scope.
