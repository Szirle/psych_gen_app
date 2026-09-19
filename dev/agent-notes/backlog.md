# Open work

Proposed order based on the active `app.py` application and the manuscript,
2026-09-17. These are development directions, not authorization to implement them
all or reproduced bug claims. Current contracts live in [implementation.md](implementation.md).

## 1. Make the interface reflect supported behavior

**Observed:** dataset generation has an empty callback; identity preservation,
requested controls, and dataset size do not affect the preview backend.

**Next:** label/disable unsupported actions, or implement each behind a precise
contract. Clarify that strength is latent displacement and `max_steps` currently
changes ridge regularization. Keep empirical source ratings distinct from output
predictions. **Done when:** every available control has documented behavior and
unimplemented capabilities cannot be mistaken for effective scientific controls.

## 2. Integrate the manuscript's manipulation methods

**Observed:** the active path uses normalized ridge coefficients. Gradient and
PGFM methods described by the paper are not wired into it.

**Next:** identify the author's intended trained impression/flow artifacts and
their preprocessing, latent space, target scales and integration settings. Define
a small method boundary that produces manipulated latents for both preview and
full resolution. Retain the linear baseline for comparison. Decide nonlinear
multi-dimension composition explicitly. **Done when:** selected methods load
identified artifacts, preserve condition coordinates across resolutions, and
state supported dimensions/controls without claiming untested equivalence to
the paper. Do not launch new training or invent checkpoints as a substitute.

## 3. Implement meaningful controlled-variable behavior

**Observed:** `controlled_variables` is accepted but unused. The ridge fit's fixed
covariate does not implement arbitrary controls or CSP.

**Next:** expose only controls supported by the selected method and artifacts;
specify tolerances, starting-population eligibility and infeasible requests.
Keep selection filters separate from constraints along paths. **Done when:**
control values are tracked across conditions and method-specific evidence
supports the stated behavior. Orthogonalization alone must not be labelled CSP.

## 4. Generate reproducible experimental datasets

**Observed:** a manipulation grid uses one base; the nine selection tiles are a
fresh distribution preview. Individual-image downloads are not dataset export.

**Next:** define base count versus total condition count, sample/freeze base
identities, construct matched conditions, and export images plus a manifest.
Include source IDs or latents/seeds, generator and method identity, truncation,
noise settings, levels, controls, coordinate mapping and resolution. Define
identity-preservation semantics before activating that option. Reuse cancellation
and bounded generation rather than building a second independent inference loop.
**Done when:** an exported condition can be traced back to its base and settings
and reconstructed within documented reproducibility limits.

## 5. Establish manipulation-specific validation

**Observed:** existing tests address software contracts; they do not establish
human impression change, confound control or PGFM validity in this application.

**Next:** define a comparison protocol for linear, gradient and PGFM outputs:
target effect, non-target changes, perceptual change/realism and dose response.
Distinguish predictor diagnostics from independent human validation and the
manuscript's study results. Check low/full-resolution correspondence for the
actual integrated artifacts. **Done when:** each scientific claim identifies its
method, evaluated stimuli, evidence and limits. Run expensive studies only as
an explicitly scoped task.

## 6. Document operation and complete the API reference

**Observed:** root README is mostly Flutter scaffolding/proposed architecture;
the API guide describes preview revisions but lacks standalone full payload
specifications for full-resolution creation, status and priority routes.

**Next:** document local model/data requirements, startup side effects, device
configuration and one-worker deployment. Add full-resolution examples from both
handlers and Dart transport. Clarify Python/PyTorch provisioning: requirements
alone do not describe the Docker's separately installed torch stack.
**Done when:** a new developer can locate required artifacts, start the intended
app, and understand all active route contracts without guessing.

## Conditional: multi-user hosting

Revisions, current face and jobs are process-wide. If concurrent independent
users become a requirement, scope ownership, caches and cancellation by session
before adding workers. Preserve one shared accelerator scheduler. This is a
deployment prerequisite for that use case, not a defect in the current single-user
contract or a reason to expand scope now.
