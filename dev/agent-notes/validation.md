# Validation guidance

Select checks for the requested task; do not run this whole list by default.
Documentation-only work needs no tests, build, model execution or screenshots.
For ordinary non-production work, follow the user's minimal-check policy. Launch
the app only when requested or when an error cannot be diagnosed from code.

## Environment

Use `/opt/anaconda3/envs/manip311/bin/python` locally, outside the sandbox for
PyTorch imports so MPS is visible. Remote VMs use their configured environment.
Do not import `app.py` as a smoke check: startup loads models/data and can download
weights. Prefer the existing fake-builder or extracted-route fixtures for
contract checks. Do not install dependencies or run real-model workloads merely
to update notes.

## Targeted coverage when needed

| Change | Existing coverage |
| --- | --- |
| Parsing, filter semantics, encoding, unequal grids | `tests/test_api_contract.py` |
| Selection sampling, empty populations, cancellation, route behavior | `tests/test_stimuli_selection.py`, `test/stimuli_selection_test.dart` |
| Ridge direction sign | `tests/test_ridge_direction.py` |
| Full-resolution jobs, polling, cancellation, OOM and memory-aware batches | `tests/test_full_resolution_jobs.py`, `tests/test_adaptive_batching.py`, `tests/test_cancellable_synthesis.py` |
| Early-output style inputs and adapter behavior | `tests/test_early_output_backend.py` |
| Grid decoding, retained previews, navigation and gestures | `test/widget_test.dart`, `test/quick_look_navigation_test.dart`, `test/preview_canvas_gestures_test.dart` |

Example commands from repo root, only when appropriate:

```bash
/opt/anaconda3/envs/manip311/bin/python -m pytest -q tests/test_api_contract.py
flutter test test/stimuli_selection_test.dart
flutter build web --release
```

Existing `.cursor/rules/` request release web builds after code edits and staging
`build/web` after successful builds. They do not require a build for these notes.
Honor more specific user instructions; never stage unrelated changes or use
`git add .`. A successful web build does not validate model inference.

For future production changes, cover stale results, shared-generator exclusion,
empty filters, unequal grids, and exact preview-to-full-resolution condition
mapping where affected. Report what ran, what passed, and what was not exercised.
No checks were run to establish the initial 2026-09-17 documentation snapshot.
