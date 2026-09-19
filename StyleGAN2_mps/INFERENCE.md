# StyleGAN2 inference

The application builders enable shared-convolution/polyphase synthesis by default,
independently of `torch.compile`, on CUDA, MPS and CPU:

```python
import torch
from models.gan_load import build_stylegan2mps, build_stylegan2_early_output

G = build_stylegan2mps(
    '/path/to/stylegan2-ffhq-1024x1024.pkl', resolution=1024,
    mixed_precision='fp16', noise_mode='const',
).to('cuda').eval()  # Use 'mps' on Apple Silicon.
with torch.inference_mode():
    images = G(torch.randn(4, 512, device='cuda'))

# Same runtime and defaults for either early-output checkpoint:
G128 = build_stylegan2_early_output(
    '/path/to/sg128_pointwise_style32.pt', mixed_precision='fp16',
).to('cuda').eval()
```

`build_stylegan2mps` retains its historical name but supports CUDA. It accepts
NVIDIA `.pkl`, NVIDIA state dictionaries, and the repository's converted
Rosinality checkpoints. Missing learned tensors or unexpected keys are errors;
only absent fixed resampling buffers are allowed for converted exports.


## Runtime choices

| Option | Behavior |
| --- | --- |
| `use_optimized=True` (default) | Shared convolution, precomposed polyphase FIR, frozen weights, latent gradients supported |
| `use_optimized=False` | Original synthesis; also respected when compiling early-output models |
| `mixed_precision='no'` (default) | FP32 throughout optimized synthesis |
| `mixed_precision='fp16'` | FP16 in blocks tagged `use_fp16`, FP32 in lower blocks |
| `mixed_precision='bf16'` | BF16 in those same optimized blocks; FP32 elsewhere |
| `op_backend='auto'` (default) | Eager CUDA/Metal custom ops for supported tensors; PyTorch fallback for BF16 and CPU; traceable PyTorch ops with compilation |
| `op_backend='native'` | Pure PyTorch custom-op replacements; useful for isolating synthesis improvements |
| `op_backend='ref'` | Reference custom-op implementations for debugging |
| `compile=True` | Optional compilation of synthesis; mapping stays eager |

The early-output decoder retains its existing CUDA autocast policy when mixed
precision is enabled.

The original synthesis has NVIDIA's precision policy: high blocks use FP16 on
CUDA when `mixed_precision` is enabled, even if its requested name is BF16;
o explicit BF16 original-model implementation exists. On MPS the original
synthesis forces FP32. The benchmark labels these limitations explicitly.

`train.py`, `train_StyleGAN.py`, pair generation, traversal and validation now
use optimized synthesis without requiring `--compile`. Pass
`--no-optimized-synthesis` when initializing an experiment to request the
original path; saved experiment settings are respected by downstream tools.

The application builders are **frozen inference** APIs, also used when training
traversal models through image/latent gradients. For an unwrapped full generator:

```python
from models.StyleGAN2_mps.early_output_model import _load_full_source_generator

G = _load_full_source_generator('/path/to/stylegan2-ffhq-config-f.pt').to(device)
ws = G.mapping(z, None)
image = G.synthesis(ws, noise_mode='const', force_fp32=True)
image = G.synthesis(ws, noise_mode='random', low_precision='fp16')
image = G(z, None, noise_mode='none', low_precision='bf16')
```

This loader accepts both `.pt` and `.pkl` and enables optimized synthesis directly
inside the existing `G.synthesis`. It returns the normal Generator, retaining
mapping, block access and state-dict keys. It does not bind precision or noise
at initialization. Noise defaults to the original `random` policy; explicit
`noise_mode` is evaluated on each call. Unspecified precision follows ambient
autocast, otherwise the original CUDA FP16 / non-CUDA FP32 policy. Explicit
`low_precision='fp32'|'fp16'|'bf16'` selects the tagged high-resolution blocks;
`force_fp32=True` overrides precision and autocast. Call
`_load_full_source_generator(path, use_optimized=False)` for original synthesis.
The original `fused_modconv` hint is accepted by the optimized path for caller
compatibility; shared convolution avoids constructing per-sample fused weights.

Low-level `model.Generator` construction and the early-output training checkpoint
loader retain their original trainable defaults. Weight training requires the
original path; optimized inference rejects re-enabled weight gradients. On an
application wrapper, `G.G` exposes its underlying source model; call the wrapper
to use that wrapper's configured inference policy.

## Prepared state

Only one precision and one FIR representation are prepared per convolution.
Low-resolution blocks keep FP32; high blocks use the selected dtype. An upsample
layer stores its polyphase kernel and demodulation energy, without an extra
expanded or ordinary kernel. FP16 normalization and FP32 demodulation-energy
accumulation are preserved.

Derived buffers are nonpersistent. They do not duplicate parameter paths in
checkpoints. Precision changes, device moves, and state-dictionary reloads
invalidate the prepared runtime; ordinary in-place weight updates are detected
through tensor versions before eager execution or the compiled wrapper. Do not
mutate parameter storage via `.data`, which bypasses PyTorch version tracking;
call `reset_runtime()` after an external update that does so.

`inference_config()` is the production policy. `b64_compile_config()` remains
only as a compatibility alias for existing benchmark scripts; no production
entry point uses it. Historical ablation flags remain available in
`InferenceOptConfig`, separate from the production policy. Low-level callers
that compile an optimized module directly must call
`module.prepare(force_fp32=...)` after the final device move and before tracing,
and recompile after changing its profile. Application wrappers do this for you.

## Full 1024 CUDA benchmark, through batch 64

Use a CUDA VM with enough VRAM, an existing CUDA-enabled PyTorch environment,
the CUDA compiler (`nvcc`) and a host C++ compiler for NVIDIA's custom ops.
Run from this repository's `train_traversals` directory. Copy the `.pkl` to the
VM and substitute its path below. `python` here means the VM's Python environment.

```bash
python -m pip install ninja
python scripts/benchmark_stylegan_inference.py \
  --checkpoint /path/to/stylegan2-ffhq-1024x1024.pkl \
  --device cuda --resolution 1024 \
  --batches 1 2 4 8 16 32 64 \
  --precisions fp32 fp16 bf16 \
  --implementations legacy pytorch optimized \
  --warmup 3 --rounds 7 --iterations 3 \
  --output output/benchmarks/ffhq1024_cuda_eager
```

The three eager implementations are:

1. **legacy:** actual `.pkl` generator classes and their original synthesis,
   with NVIDIA CUDA custom ops.
2. **pytorch:** local optimized synthesis with pure PyTorch op implementations.
3. **optimized:** the same optimized synthesis with automatic CUDA custom ops.
   BF16 tensors use PyTorch fallback; FP32/FP16 tensors still use CUDA kernels.

This separates algorithmic gains from backend effects. Metal kernels are
Apple-specific; they are not part of the CUDA VM comparison.

Optional, separate compiler experiment (CUDA/Inductor execution still needs VM validation):

```bash
python scripts/benchmark_stylegan_inference.py \
  --checkpoint /path/to/stylegan2-ffhq-1024x1024.pkl \
  --device cuda --resolution 1024 \
  --batches 1 2 4 8 16 32 64 \
  --precisions fp32 fp16 bf16 \
  --implementations legacy pytorch \
  --compile --compile-mode default \
  --warmup 3 --rounds 7 --iterations 3 \
  --output output/benchmarks/ffhq1024_cuda_compiled
```

Compilation uses PyTorch op implementations in **both** rows. `optimized` and
`pytorch` would be equivalent in this experiment, so only one is requested.
Each batch may trigger a new graph; lazy compilation and warmup are excluded
from steady-state timing and first-call time is recorded separately.

The driver records synchronized W-to-image latency (median and all samples),
images/s, CUDA peak allocated/reserved memory, preparation time/buffer bytes,
and first-image error against a fixed original FP32 reference. Every worker
uses the same saved W bank and constant noise. Mapping and checkpoint loading
are excluded. TF32 is disabled by default; repeat in a new output directory
with `--tf32` for a separate TF32 experiment. The driver starts a fresh process
per implementation/precision, stops that variant at its first OOM and continues
other variants. It never substitutes smaller microbatches for the requested batch.

Results are written to `results.csv`, `results.json`, per-worker JSONL files,
and `metadata.json`; send that output directory back for CUDA charts/reporting.
Speedup ratios only compare identical batch and precision. Original BF16 rows
are marked unsupported, not relabeled FP16 timings. Existing output directories
with run metadata are rejected to prevent mixing runs.

## Validation

On the development Mac, the new default was checked against the actual full
1024 and 128/256 early-output checkpoints in FP32, mixed FP16 and mixed BF16.
Full-model FP32 output RMSE was approximately 8e-7 for the fixed smoke-test
latent. Prepared-buffer totals were 225.9 MiB (FP32) and 212.4 MiB (mixed
FP16/BF16); these totals describe derived tensors, not peak process memory.
The earlier PDF reports pre-cleanup measurements and remains a historical
benchmark, not a measurement of the new cache policy. CUDA speedups must be
measured on the VM; they have not been inferred from Metal timings.

Local fullgraph Dynamo tracing and latent gradients passed. An optional CPU
Inductor smoke run stalled and was stopped without obtaining compiled timings.
The eager CLI was exercised end to end; CUDA execution and compiler performance
remain VM checks.

Run the portable runtime regression tests on a CUDA VM as well:

```bash
python -m pytest -q tests/test_stylegan_inference_runtime.py tests/test_b64_optimize_ops.py
```
