# Training views for generated FFHQ faces

All neural architectures share `augmentation.py` and `tensor_views.py`: batched
device-side transforms feed either a persistent frozen-feature bank or the live
visual encoder. CPU PIL transforms are no longer used for augmented views. The frozen
kernel comparator, MI selection, feature scaling and validation/test inference
continue to use clean images. Augmentations inherit their source image's labels
and split; they are never counted as new independent subjects.

## Default FFHQ recipe

The transforms are inspired by the configurable geometry/filter/noise families
in [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/augment.py)
and the image-view augmentation approach in
[OpenCLIP](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py).
This is an original conservative implementation, not NVIDIA's augmentation
code or its adaptive discriminator-based probability controller. Settings are
research starting points, not an established optimum for subjective ratings.

For an augmented view:

| Transform | Default |
| --- | --- |
| Horizontal flip | Probability 0.5 |
| Isotropic scale | Uniform 0.98–1.02 |
| Translation | Up to 1.5% of width/height per axis |
| Gaussian pixel noise | Probability 0.35; SD sampled from 0–0.01 on a 0–1 pixel scale |
| Gaussian blur | Probability 0.2; sigma 0.5 encoder-raster pixels, independent blend strength 0–1 |
| Brightness/contrast | Disabled; opt-in bounded jitter with probability 0.2 |
| Feature-vector noise | Disabled; optional small Gaussian perturbation then L2 normalization |

Geometry uses a single batched affine grid and reflection-padded bilinear sampling.
Transforms act at the native encoder raster resolution, after deterministic
preprocessing and before the encoder. Noise and blur are therefore measured at
that resolution, not at the original 1024-pixel source resolution. This is a
changed augmentation distribution, not a bitwise reproduction of PIL transforms.
There are no rotations, anisotropic stretches, large crops, cutout, grayscale,
hue changes or label mixing. Strong photometric changes are avoided because
skin/hair colour and perceived appearance are targets. Of training draws,
25% use the clean image/features; the remainder sample augmented views.

## GPU execution and persistent caches

The official processor decodes/resizes/normalizes each original once on first
use. Its native patch tensors are retained in a shared CPU bank across fits.
Each augmentation batch gathers those tensors, transfers them to the encoder
device, unpatches them into BCHW images, and applies GPU operations. It restores
the processor's normalization and exact patch layout before the model forward;
attention masks, spatial shapes and padded patches remain intact. The current
batch must have one native raster shape, as aligned FFHQ images do. Other aspect
ratios are not silently stretched into squares.

Geometry is batched PyTorch `affine_grid`/`grid_sample`, blur uses torchvision's
batched Gaussian convolution, and color/noise use tensor operations with
independent per-image parameters. There is no per-image PIL augmentation or
per-image Gaussian-noise allocation. See torchvision's
[tensor and batch transform guidance](https://docs.pytorch.org/vision/stable/transforms.html).
Randomness is generated in fixed 32-image shard segments, so changing how many
shards share a forward, including OOM backoff, does not change sampled transforms.
Encoder numerics can still differ slightly with batch size/device.

- **Encoding batch:** `--augmentation-encode-batch 512`, independent of training
  batch size and `--encode-batch`. It packs shards across images and views, runs
  without gradients, and halves the number of packed shards on CUDA OOM. The
  minimum is one 32-image shard (or fewer at a split boundary). The successful
  batch limit is retained for subsequent refreshes within the fit.
- **Active views:** four per training image, refreshed every ten epochs. Training
  draws still use the original number of image rows, with 25% clean draws.
- **Startup prefill:** before the first training epoch, encode the full bounded
  bank: `bank_views - views` reserve views plus `views` additional views per
  training image (28 + 4 by default). Only four selected views become resident.
  Selection uses the full-sized replay pool starting at epoch one, including
  when a resumed fit reconstructs its bank.
- **Refresh policy:** with `--augmentation-reuse-fraction 0.5`, select two freshly
  generated views plus two sampled from the retained replay pool. Before
  generating replacements, delete the oldest two view IDs and their dense/score
  shards. Bank cardinality stays at or below 32 throughout refresh. The active
  subset changes; the remaining replay views stay on disk.
- **Hard bound:** `--augmentation-bank-views 32` must be at least
  `--augmentation-views`. Unlimited banks (`0`) are no longer supported. Reuse
  fraction 0 refreshes all active slots; fraction 1 samples entirely from the
  prefilled bank without generating replacements.
- **Disk layout:** exactly one disposable bank at
  `--cache/augmentation-v2/active/dense/<view>/<shard>.pt`, plus score sidecars.
  Identity-indexed bank directories are no longer created. Every fit starts
  empty and owns this slot exclusively; cross-fit migration is not implemented.
- **Canonical encodings:** FP16 global/dense features, masks and coordinates are
  saved alongside phrase scores when a shared fixed-input head needs them.
  Dense variants use those encodings directly. Score-only resident
  caches contain global features and global/local-max phrase scores; score
  sidecars avoid recalculation on subsequent loads. Phrase projection is chunked
  to bound temporary alignment memory.
- **Disk I/O:** shards are atomically replaced, loaded via memory mapping, and
  written by a bounded background writer overlapping the next GPU batch.
  The writer finishes before fit cleanup. A fit holds an exclusive cache lock;
  concurrent jobs must use separate `--cache` directories. This prevents
  one job from deleting another current job's augmentation bank.
- **Resident memory:** only the active four views are loaded. Auto placement
  uses CUDA when they fit 16 GiB and 25% of free VRAM, otherwise host RAM. Only
  views within the retained bank window remain on disk. The prepared original tensors also occupy host RAM;
  background writes retain at most a small bounded number of encoding batches.
- **Live vision:** fresh GPU transforms every minibatch, using the same prepared
  originals. Clean consistency/evaluation inputs also reuse native tensors.
  Encoded banks are bypassed because changing vision weights invalidate them.
- **Isolation:** only the fit's training rows receive cached augmented features.
  The prepared-original bank may also serve clean validation inputs, without
  making their augmented views available to training.

The log reports new versus reused encodings, elapsed refresh time, active view
IDs, resident allocation, disk directory and estimated full dense bank size.
Each neural fit removes its augmentation bank on exit, including exceptions.
CLI startup and every neural fit clear abandoned banks in `augmentation-v2/`,
including legacy identity-indexed folders. Startup cleanup also runs for
kernel-only and completed/resumed runs that skip neural fitting. Cleanup logs
the directory and number of entries removed; errors are not silently ignored.
Clean original-image `.pt` files outside that subtree are never removed, nor
are model checkpoints. Stop older training processes before using this cleanup
policy: they predate the ownership lock. A hard kill cannot execute cleanup;
its leftovers are removed at the next CLI startup or fit.

This implementation chooses deletion after every selection/refit/final fit,
rather than migrating shared samples between fits. Consequently every new fit
pays its prefill cost again, but banks do not accumulate across folds, variants
or runs. All disk eviction is restricted to the augmentation cache subtree.

Clean caches are not overwritten. Validation, checkpoint selection and inference
use clean inputs with no test-time augmentation. Prompt agreement preservation
compares learned/original phrases on the same frozen view; with live vision its
reference branch remains the cached clean view. Kernel-residual variants retain
their clean OOF reference and clean original-score inputs while augmenting the
learned branch.

Refresh encoding adds GPU work, especially for otherwise fast fixed-feature
heads. Prefilling moves the reserve encoding cost before epoch one. Later
refreshes only encode replacement views, while replay reads existing shards.
This does not make 1,004 original faces equivalent to a larger independent
dataset, and improved validation accuracy still requires a matched comparison.

## Controls

The existing launcher enables this recipe by default. Its default run directory
is now `runs/cuda96-joint-v4-gpu-views`; use a fresh `RUN_DIR` for the changed training policy.
No additional dependencies or model downloads are introduced.
The default VM smoke covers score caches, dense caches and live augmentation
with `shared-tabm`, `shared-tabm-chain-prompt` and `shared-transformer-chain-joint`.

```bash
# From the repository root; adjust uploaded data paths.
IMAGES=/workspace/data/images \
RATINGS=/workspace/data/dim_to_photo_to_ratings.pkl \
RUN_DIR=/workspace/fgclip2-augmented \
bash CLIP/scripts/setup_fgclip2_multitarget.sh \
  --augmentation ffhq --augmentation-views 4 --augmentation-refresh 10 \
  --augmentation-clean-prob 0.25 --augmentation-cache-gib 16 \
  --augmentation-encode-batch 512 --augmentation-bank-views 32 \
  --augmentation-reuse-fraction 0.5
```

Use `--augmentation none` for the clean ablation or `--augmentation flip` for
flip-only views. These choices apply to both cached and live training.
The launcher also accepts `AUGMENTATION_ENCODE_BATCH=1024` for larger starting
batches; actual capacity depends on patch budget and GPU. CPU decoding batch
`ENCODE_BATCH` and training batch `BATCH_SIZE` retain their separate meanings.
Other controls: `--augmentation-translate`, `--augmentation-scale`,
`--augmentation-noise`, `--augmentation-blur`, `--augmentation-color`,
`--augmentation-cache-device auto|cpu|cuda`. Explicit CUDA storage fails clearly
if its budget is insufficient instead of silently falling back.

For a separate feature-noise ablation, add `--augmentation-feature-noise 0.01`.
This gives approximately 0.01 noise-vector norm before renormalization,
independent of embedding width. Frozen variants encode this perturbation into
each cached view; live variants apply it online to augmented draws. Labels,
patch masks and coordinates are not perturbed.

All settings are recorded in the manifest and each fit's training profile;
histories record the cached generation. No tests, surrogate training or CUDA
benchmark were run for the GPU/cache-lifecycle rewrites. The old
[augmentation_validation.json](augmentation_validation.json) describes the
superseded PIL/in-memory implementation and does not validate this path.
No measured speedup is claimed. Use a fresh run directory: changed source and
augmentation settings deliberately fail the experiment's strict resume check.
Native clean feature caches remain reusable.

Stop/restart already-running Python workers after updating this code; they keep
the previously imported cache implementation. This change does not remotely
delete files on an already-running VM. After normal fit exit, only `.fit.lock`
remains in `augmentation-v2/`; original-image `.pt` caches remain outside it.

Ownership is acquired in `TrainingViews.__init__`, before a usable object is
returned. `fit()` uses an explicit `try/finally: training_views.close()` around
all selection, training, evaluation and checkpoint work. Correctness no longer
depends on `TrainingViews.__enter__` or `__exit__` being invoked. The optional
context-manager API remains idempotent. Update both `augmentation.py` and
`train.py` on the VM; the former acquires ownership and the latter releases it.
