# Current implementation and code map

Source-inspected 2026-09-17. Paths are relative to repository root. Exact request
and response contracts belong in [API_COMMUNICATION.md](../../docs/API_COMMUNICATION.md).

## Runtime and boundaries

- `lib/main.dart` composes Flutter localization, theme, BLoCs and the face-generation
  page. `lib/features/face_generation/` contains data, domain, and presentation
  layers. `lib/core/` holds shared design components, API configuration and utilities.
- `app.py` serves Flask endpoints and `build/web`. Importing it initializes the
  generator and data; missing models can trigger downloads. Do not import it just
  to inspect routes or documentation.
- `config.yaml` selects a 256-pixel early-output checkpoint by default.
  `early_output_backend.py` adapts 128/256 checkpoints; accelerator auto-selection
  prefers CUDA, then MPS, then CPU. If the selected checkpoint is unavailable,
  startup falls back to `gan_backend.py` and the original generator path.
- `models/` and `data/` are runtime inputs, not material to regenerate casually.
  The relevant mappings are `photo_to_coords.pkl` and
  `dim_to_photo_to_ratings.pkl`. Inspect loaders before changing formats.
- API URLs default to same-origin; `API_BASE_URL` is a Flutter build-time override.
  The Docker entry point uses one threaded Gunicorn worker. Current process state
  is not safe to treat as independent browser sessions or independent workers.

## Selection and empirical distributions

`api_contract.py` owns parsing, dimension-name normalization, filter intersections,
histograms and grid reshaping. `stimuli_selection_backend.py` separates sampling
from rendering and reuses the loaded preview generator under the caller's GPU lock.

- `/stimuli/preview` produces nine faces in a 3×3 grid without manipulation.
  No filters means Gaussian Z sampling followed by mapping to W. Filters mean
  sampling stored W vectors whose source faces satisfy inclusive mean-rating ranges.
- Selection uses no replacement when enough eligible faces exist, otherwise
  reports replacement explicitly. An empty eligible population is a 422 error,
  never permission to silently sample unfiltered faces.
- Truncation is applied after sampling. Ratings describe original stored faces;
  they are not new ratings of truncated outputs.
- Selection does not mutate the manipulation preview's current base face.
  `StimuliSelectionCubit` debounces and suppresses stale successes/errors; a failed
  replacement clears the sample. Committed filters survive histogram loading.
- `/distributions` returns empirical histograms. Empty intersections stay empty.
  `/charts` serves mocked prototype charts and is disabled in the active UI.

## Manipulation baseline

`app.py:FastStyleGANBackend` maintains one current base W and caches directions.
`utils.py:ridge_coefs` fits ridge regression to logit-transformed mean ratings,
appending age as a covariate (gender when the target is age), then drops the
covariate coefficient. This is not the requested `controlled_variables` mechanism.
The positive coefficient direction is score ascent; do not negate it.

`_get_direction` normalizes the coefficient vector and repeats it across style
layers. The backend constructs a Cartesian grid by adding weighted directions
to a truncated base W. `max_steps` currently controls ridge regularization through
`alpha = 2 ** ((steps - 30) / 4)`, not numerical integration steps. Strengths are
scaled by 0.1 internally.

- Preview supports 1–3 distinct dimensions, each with 2–5 levels.
- `shape`, `color`, and `both` select style-layer slices `[0:9]`, `[9:end]`,
  and `[0:end]`; these names are not guarantees of perceptual disentanglement.
- 2D API grids are `[dimension1][dimension0]`; 3D grids retain model dimension
  order and the client maps them to display slices. Preserve unequal-grid ordering.
- A retained base remains stable while eligible; active filters select stored
  latents. The manipulation sampler is separate from the nine-face preview.
- `num_faces`, `preserve_identity`, and `controlled_variables` are reserved and
  ignored for preview generation. The dataset button's callback is empty in
  `face_generation_page.dart`. There is no dataset-export pipeline.
- The active route uses ridge directions, not the manuscript's nonlinear
  impression gradients or learned PGFM vector fields.

## Full resolution and responsiveness

`/images/full`, `/images/full/status`, `/images/full/prioritize`, and
`/images/full/cancel` implement manipulation-only upgrades. `app.py` holds grid
jobs and the worker; `adaptive_batching.py` provides memory-aware batch planning.
Quick Look navigation and downloading individual assets exist; these are distinct
from generating a dataset of experimental conditions.

Preview revisions are monotonically increasing and process-wide. New preview
intent cancels obsolete full-resolution work; both preview routes share the GPU
lock. The client rejects stale results and invalidates Quick Look state as intents
change. Cancellation must allow active inference to relinquish the accelerator
before replacement work proceeds. Keep status polling able to progress while
generation runs, preserve coordinate priorities, and retain OOM retry behavior.

Full-resolution latent construction is separate from preview synthesis, including
extension to the original generator's style layers. Changing manipulation math
must update both paths so an upgrade represents the selected preview condition.

## Architecture constraints

Keep domain code independent of data/presentation; keep transport in data and
interaction state in BLoCs/Cubits. Existing composition roots and core design
widgets legitimately import Flutter. The README's proposed architecture is not
an exact inventory or a mandate for a sweeping refactor. Add localized user-facing
strings to `assets/translations/en.json` and use `tr()`.
