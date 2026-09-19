# StyleGAN Metal ops

`bias_act` and `upfirdn2d` now use custom Metal kernels for MPS tensors with the
existing default `impl='cuda'`. `impl='mps'` explicitly selects Metal;
`impl='auto'` selects CUDA/Metal on those devices and pure PyTorch elsewhere.
`impl='ref'` and the existing compile-friendly `impl='native'` remain available.
To switch an entire model without editing its call sites:

```python
from models.StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops

with use_native_ops(upfirdn='mps', bias_act='mps'):
    image = G.synthesis(ws, noise_mode='const')
```

## Build and execution

The first MPS invocation builds `stylegan_ops.mm` with PyTorch's C++ extension
loader, using its normal disk cache. This requires a working Apple C++ compiler,
PyTorch with MPS support, and Ninja. The installed PyTorch chooses the C++ language
standard. No CUDA toolkit or standalone `xcrun metal` compiler is needed.

The system Metal runtime compiles the shared header and shader source once per
process, using Metal 2.3 and fast math. Library and specialized pipeline states
are cached. No precompiled metallib tied to one macOS release is shipped.
The Objective-C++ bridge depends on PyTorch's internal `ATen/mps/MPSStream.h` API;
rebuild/revalidate when upgrading PyTorch. Compile failures surface as errors;
select `ref` or `native` explicitly to run without the extension.

All dispatches use PyTorch's current MPS stream/encoder inside its serial queue
and an autorelease pool. There are no per-op command-buffer commits or GPU waits.
Tensor bindings include storage offsets, and absent bias/derivative buffers bind
the input tensor with a flag preventing reads. Scalar metadata uses one shared
host/MSL struct per operation.

## Kernel coverage

- FP32 and FP16 inputs, with FP32 arithmetic and accumulation; FIR filters remain
  FP32. Other activation dtypes, including BF16, are rejected by Metal.
- All nine activations and bias derivatives through second order. Dense physical
  layouts are preserved, including permutations and nonzero storage offsets.
  Nondense bias inputs are materialized; backward restores the saved layout.
- Generic FIR supports arbitrary nonnegative strides (including expanded views),
  rectangular/strided filters, asymmetric up/down factors, signed padding,
  orientation and gain. Separable filters reuse the existing two-pass wrapper.
- FIR backward reuses the same primitive and existing Python padding formulas;
  higher-order differentiation remains available. Filter gradients are rejected.
- Cached function constants specialize 4x4 filters at 1x, up2, and down2 for
  contiguous and channels-last inputs. Full 128-thread groups cooperatively load
  input tiles with a uniform barrier. Threadgroup resource limits are checked.
  Other cases use the generic kernel.

`STYLEGAN_MPS_FIR_KERNEL=tiled` is the default, selected from local measurements.
Set it to `direct` to benchmark specialized direct reads, or `generic` to bypass
specialization. Unsupported shapes always use generic Metal. Performance policy
is not an autotuner and may need different choices on other Apple GPUs.

## Numerical details

The port follows NVIDIA's operation order and saved-reference derivative
contract. Clamp derivatives use an open interval; PyTorch's reference clamp
passes derivatives at exact boundaries. FP16 saved outputs are rounded before
backward, so nonrepresentable clamp thresholds (e.g. 1.3) can produce different
masks from the reference. Use exactly representable thresholds when testing
boundary parity. Reference FP16 also rounds bias/filter intermediate values
that the fused kernels retain in FP32; bitwise general parity is not promised.

Two intentional corrections are included: the shared wrapper saves the output
for clamped linear activations, and Metal evaluates swish derivatives with a
stable sigmoid expression instead of an `exp(x)^3` denominator that can overflow
near x=30. NVIDIA's activation constants and tail cutoffs are retained.

The default synthesis model still forces FP32 on non-CUDA devices. This change
adds FP16 *op* support; it does not alter that model precision policy. Metal calls
are eager custom autograd ops, not registered full-graph `torch.compile` operators;
keep `impl='native'` for the existing compilation workflow.

## Validation and measurements

Run outside a sandbox that hides the MPS device, from the repository root:

```bash
/opt/anaconda3/envs/manip311/bin/python -m pytest tests/test_stylegan2_mps_ops.py -q
PYTHONPATH=. /opt/anaconda3/envs/manip311/bin/python tests/benchmark_stylegan2_mps_ops.py \
  --checkpoint /path/to/stylegan2-ffhq-1024x1024.pkl --output /tmp/stylegan-metal.json
```

343 checks passed on PyTorch `2.15.0.dev20260817`, macOS `26.6.2`. Coverage includes
forward/first/second derivatives, all activations, FP16/FP32, dense and strided
layouts, offsets, noncontiguous filters, asymmetric/negative padding, odd sizes,
separable/large filters, clamp boundaries, extreme swish inputs and optimized
kernel parity. A small 16px generator/discriminator also checks parameter
gradients with path-length and R1 penalties against the reference backend.

On the supplied FFHQ 1024px NVIDIA pickle (strictly loaded through the existing
`load_ffhq_state_dict` helper), fixed-latent/constant-noise FP32 synthesis matched
at every block and at the final image: maximum absolute difference 0.0 for that
sample. Projector-style latent gradients differed by at most `9.31e-10`.

| 1024px synthesis, batch 1 | Median forward time |
| --- | ---: |
| Reference | 215.6 ms |
| Existing pure PyTorch native | 207.1 ms |
| Metal | 160.9 ms |

This is about 1.34x faster than reference and 1.29x faster than native on the tested
machine, excluding compilation and warmup. These are isolated synthesis timings,
not full application or compiled-model timings. Full-resolution training and
long-run convergence have not been benchmarked.

For `[1,64,128,128]`, tiled FIR won all 12 tested dtype/layout/scaling combinations
against the direct variant. FP32 NCHW up2 took 0.302 ms tiled, 0.666 ms direct,
0.479 ms existing native, and 2.905 ms reference. Some native cases remain faster
(e.g. FP32 channels-last up2: 0.393 ms native vs 0.521 ms tiled). Bias/LReLU took
0.101–0.154 ms fused versus 0.201–0.397 ms reference. The benchmark records all
variants so future hardware tuning can compare them without changing semantics.
