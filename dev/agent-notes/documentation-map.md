# Source map

| Source | Role and caveats |
| --- | --- |
| [Author's manuscript](</Users/adamsobieszek/Deixis/psych_methods25 2/manuscript.tex>) | Scientific intent: *Theory of optimal experimental manipulation with generative models via reverse correlation of stimuli*. External local path supplied by the author; not a portable repo dependency. |
| [API contract](../../docs/API_COMMUNICATION.md) | Detailed Flutter–Flask payloads, sampling and revision protocol. Full-resolution route schemas still need expansion. |
| `app.py`, `api_contract.py`, `stimuli_selection_backend.py` | Active server behavior, parsing and sampling. Prefer code over obsolete comments when they disagree. |
| `utils.py:ridge_coefs`, `app.py:FastStyleGANBackend` | Actual linear manipulation method; do not infer PGFM support from the paper or nearby research code. |
| `lib/features/face_generation/` | Active client transport, state, controls and preview/Quick Look behavior. |
| `config.yaml`, `early_output_backend.py`, `gan_backend.py`, `Dockerfile` | Runtime selection, model loading and deployment. Configuration is not proof an artifact exists. |
| [Root README](../../README.md) | Historical scaffold and proposed architecture; not a current setup guide. |
| `.cursor/rules/` | Existing architecture, localization, API synchronization and build/staging guidance; user instructions take precedence. |

## Manuscript reading by question

Use LaTeX labels rather than unstable line numbers:

- `sec:high_road`, `loc_opt_crit`, `loc_opt`: feature geometry and local manipulation.
- `orth`, `def:path_risk`, `def:csp`, `path_opt`: linear control limits, accumulated
  prediction risk, Conditional Support Preservation and dose-response objectives.
- `sec:flow_ot`, `psych_cost`, `alg:pgfm`: transport cost, stratification, learned
  vector field and manipulation-path integration.
- `sec:impression_models`: differentiable prediction models of 28 impressions.
- `sec:study1` through `sec:study4`: empirical evidence and method variants.
- `sec:data_av`: pointers to the paper's research artifacts. Locate the intended
  artifact before making an integration plan; availability in the paper does not
  establish that it is present or served in this repository.

The vision draws on the manuscript's aims, path criteria and PGFM method. Its
study claims belong to those studies, not to the current app's ridge preview.
Do not upgrade manuscript claims into general guarantees or assign its model
coverage to the application's enum automatically.

Ignore traversal-named files and FG-CLIP/`CLIP/` for this application's context.
Do not scan unrelated research archives or vendor source to reconstruct intent.
