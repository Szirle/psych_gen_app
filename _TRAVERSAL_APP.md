# Traversal explorer

This temporary app reuses the original Flutter widgets and the existing
StyleGAN backend, but replaces rating-model variables with traversal indices
`0` through `511`. The original `app.py` entry point is unchanged.

Build the traversal-specific Flutter entry point:

```bash
flutter build web -t lib/main_traversals.dart --output build/traversals_web
```

Run its Flask backend (port `8001` by default):

```bash
/opt/anaconda3/envs/manip311/bin/python traversal_app.py
```

The model directory and rollout settings can be overridden with
`TRAVERSAL_MODELS_DIR`, `TRAVERSAL_STRENGTH_SCALE`,
`TRAVERSAL_ROLLOUT_STEPS`, and `TRAVERSAL_ROLLOUT_CHUNK_SIZE`.
