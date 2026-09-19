"""Prepared frozen synthesis with shared convolution and composed resampling.

Only the active precision and FIR representation are cached. Parameters and
checkpoint keys remain on the source synthesis; derived buffers are nonpersistent.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bias_act, conv2d_resample, fma, upfirdn2d
from .fused_epilogue import fused_epilogue_composite
from .fused_up_modconv import compose_weight_with_fir_expanded
from .upfirdn2d_native import _filter_2d


_PREFIX = "_inference_"


def _suffix_for_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return "fp32"


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, value.detach().contiguous(), persistent=False)


def clear_prepared_synthesis(synthesis: nn.Module) -> None:
    """Discard derived buffers before changing precision, weights or device."""
    for layer in synthesis.modules():
        for name in list(layer._buffers):
            if name.startswith((_PREFIX, "_b64opt_")):
                delattr(layer, name)
        layer.__dict__.pop("_inference_signature", None)


def _tensor_signature(tensor):
    return (id(tensor), tensor._version, tensor.device, tensor.dtype)


def _prepare_layer(layer: nn.Module, *, dtype: torch.dtype, fir_mode: str | None) -> bool:
    """Keep one representation per layer; rebuild after source weights change."""
    signature = (dtype, fir_mode, _tensor_signature(layer.weight),
                 _tensor_signature(layer.resample_filter))
    if layer.__dict__.get("_inference_signature") == signature:
        return False
    clear_prepared_synthesis(layer)
    # Preparation may first happen inside an inference_mode call. These constants
    # must remain ordinary tensors so subsequent latent-gradient calls can save them.
    with torch.inference_mode(False), torch.no_grad():
        weight32 = layer.weight.detach().float()
        if dtype == torch.float16:
            _, in_channels, kh, kw = weight32.shape
            scale = (weight32.abs().amax(dim=(1, 2, 3), keepdim=True)
                     .clamp_min(1e-8).reciprocal()
                     * float(1.0 / (in_channels * kh * kw) ** 0.5))
            weight32 = weight32 * scale
        suffix = _suffix_for_dtype(dtype)
        _set_buffer(layer, f"{_PREFIX}energy_{suffix}", weight32.square().sum(dim=(2, 3)))
        weight = weight32.to(dtype)
        if fir_mode is None:
            _set_buffer(layer, f"{_PREFIX}weight_{suffix}", weight)
        else:
            if tuple(weight.shape[2:]) != (3, 3) or tuple(layer.resample_filter.shape) != (4, 4) or layer.padding != 1:
                raise ValueError("Composed FIR requires a 3x3 convolution, 4x4 filter and padding=1")
            fir, _ = _filter_2d(layer.resample_filter, device=weight.device,
                               dtype=torch.float32, gain=4.0, force_dense=True)
            composed = compose_weight_with_fir_expanded(weight, fir.to(dtype))
            if fir_mode == "expanded":
                prepared = composed.transpose(0, 1)
            elif fir_mode == "polyphase":
                prepared = torch.cat([composed[:, :, oy::2, ox::2].flip((2, 3))
                                      for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1))])
            else:
                raise ValueError(f"Unsupported FIR mode {fir_mode!r}")
            _set_buffer(layer, f"{_PREFIX}{fir_mode}_{suffix}", prepared)
    layer._inference_signature = signature
    return True


def prepare_optimized_synthesis(synthesis: nn.Module, cfg: Any, *, force_fp32=False) -> bool:
    """Prepare the active profile. Return whether any derived buffer changed.

    Call before torch.compile tracing; eager forwards also check for weight updates.
    Weight training uses the original model, since these kernels are frozen constants.
    """
    if any(p.requires_grad for p in synthesis.parameters()):
        raise RuntimeError("Optimized synthesis requires frozen weights; use the original synthesis for weight training")
    if not (cfg.shared_modconv or cfg.fir_compose):
        return False
    changed = False
    for resolution in synthesis.block_resolutions:
        block = getattr(synthesis, f"b{resolution}")
        dtype = (cfg.low_precision_dtype
                 if (block.use_fp16 or cfg.force_fp16_all_blocks) and not force_fp32
                 else torch.float32)
        for name in ("conv0", "conv1"):
            if hasattr(block, name):
                layer = getattr(block, name)
                mode = cfg.fir_compose_mode if cfg.fir_compose and layer.up == 2 else None
                changed = _prepare_layer(layer, dtype=dtype, fir_mode=mode) or changed
    return changed


def _noise(layer: nn.Module, x: torch.Tensor, noise_mode: str) -> torch.Tensor | None:
    if not layer.use_noise or noise_mode == "none":
        return None
    if noise_mode == "random":
        return (
            torch.randn(
                [x.shape[0], 1, layer.resolution, layer.resolution],
                device=x.device,
            )
            * layer.noise_strength
        )
    if noise_mode == "const":
        return layer.noise_const * layer.noise_strength
    raise ValueError(f"Unsupported noise mode {noise_mode!r}")


def _prepared_up2(
    layer: nn.Module,
    x: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    suffix = _suffix_for_dtype(x.dtype)
    if mode == "expanded":
        weight_t = getattr(layer, f"{_PREFIX}expanded_{suffix}")
        y = F.conv_transpose2d(x, weight_t, stride=2)
        out_h, out_w = x.shape[2] * 2, x.shape[3] * 2
        # Exact common StyleGAN case: 3x3 learned kernel, 4x4 FIR, pad [1]*4.
        return y[:, :, 2 : 2 + out_h, 2 : 2 + out_w]

    weight = getattr(layer, f"{_PREFIX}polyphase_{suffix}")
    out_channels = int(layer.out_channels)
    y = F.conv2d(x, weight, padding=1)
    n, _, h, w = y.shape
    y = (
        y.reshape(n, 4, out_channels, h, w)
        .permute(0, 2, 1, 3, 4)
        .reshape(n, 4 * out_channels, h, w)
    )
    return F.pixel_shuffle(y, 2)


def _shared_layer(
    layer: nn.Module,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    cfg: Any,
    noise_mode: str,
    gain: float,
) -> torch.Tensor:
    styles = layer.affine(w)
    # FP16 needs inf-norm pre-scaling to avoid overflow. BF16 has FP32 range,
    # so the unnormalized weights are used directly.
    if x.dtype == torch.float16:
        styles = styles * (
            styles.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8).reciprocal()
        )
    suffix = _suffix_for_dtype(x.dtype)
    energy = getattr(layer, f"{_PREFIX}energy_{suffix}")

    x = x * styles.to(x.dtype).reshape(x.shape[0], -1, 1, 1)
    if bool(cfg.fir_compose) and int(layer.up) == 2:
        x = _prepared_up2(layer, x, mode=str(cfg.fir_compose_mode))
    else:
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=getattr(layer, f"{_PREFIX}weight_{suffix}"),
            f=layer.resample_filter,
            up=int(layer.up),
            padding=int(layer.padding),
            flip_weight=(int(layer.up) == 1),
        )

    dcoef = (
        styles.float().square() @ energy.float().transpose(0, 1) + 1e-8
    ).rsqrt().to(x.dtype)
    dcoef = dcoef.reshape(x.shape[0], -1, 1, 1)
    noise = _noise(layer, x, noise_mode)
    act_gain = float(layer.act_gain) * float(gain)
    act_clamp = (
        float(layer.conv_clamp) * float(gain)
        if layer.conv_clamp is not None
        else None
    )

    if bool(cfg.fused_epilogue):
        return fused_epilogue_composite(
            x,
            layer.bias.to(x.dtype),
            noise=noise,
            dcoef=dcoef,
            act=layer.activation,
            gain=act_gain,
            clamp=act_clamp,
        )

    if noise is not None:
        x = fma.fma(x, dcoef, noise.to(x.dtype))
    else:
        x = x * dcoef
    return bias_act.bias_act(
        x,
        layer.bias.to(x.dtype),
        act=layer.activation,
        gain=act_gain,
        clamp=act_clamp,
    )


def _layer(
    layer: nn.Module,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    cfg: Any,
    noise_mode: str,
    gain: float = 1.0,
) -> torch.Tensor:
    if bool(cfg.shared_modconv or cfg.fir_compose):
        return _shared_layer(
            layer,
            x,
            w,
            cfg=cfg,
            noise_mode=noise_mode,
            gain=gain,
        )
    # Epilogue-only experiments are intentionally deferred until the shared
    # path is accepted; use the untouched layer otherwise.
    return layer(x, w, noise_mode=noise_mode, gain=gain)


class OptimizedSynthesis(nn.Module):
    """Frozen inference with gradients with respect to styles and inputs."""

    def __init__(self, synthesis: nn.Module, cfg: Any):
        super().__init__()
        self.synthesis = synthesis
        self.cfg = cfg
        self.resolutions = tuple(int(r) for r in synthesis.block_resolutions)
        synthesis.requires_grad_(False)
        self.prepare()

    def prepare(self, *, force_fp32=False):
        return prepare_optimized_synthesis(self.synthesis, self.cfg, force_fp32=force_fp32)

    def _apply(self, fn, recurse=True):
        clear_prepared_synthesis(self.synthesis)
        return super()._apply(fn, recurse=recurse)

    def forward(
        self,
        ws: torch.Tensor,
        noise_mode: str = "const",
        force_fp32: bool = False,
    ) -> torch.Tensor:
        return optimized_synthesis_forward(self.synthesis, ws, self.cfg,
                                           noise_mode=noise_mode, force_fp32=force_fp32)


def optimized_synthesis_forward(synthesis, ws, cfg, *, noise_mode="random", force_fp32=False):
    """Evaluate the source synthesis directly, preserving its registered structure."""
    if not torch.compiler.is_compiling():
        prepare_optimized_synthesis(synthesis, cfg, force_fp32=force_fp32)
    ws = ws.to(torch.float32)
    x = None
    img = None
    w_base = 0

    for resolution in synthesis.block_resolutions:
        block = getattr(synthesis, f"b{resolution}")
        low_dtype = getattr(cfg, "low_precision_dtype", torch.float16)
        dtype = (
            low_dtype
            if (block.use_fp16 or bool(cfg.force_fp16_all_blocks))
            and not force_fp32
            else torch.float32
        )
        memory_format = (
            torch.channels_last
            if block.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if block.in_channels == 0:
            x = block.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)

        w_idx = w_base
        if block.in_channels == 0:
            x = _layer(
                block.conv1,
                x,
                ws[:, w_idx],
                cfg=cfg,
                noise_mode=noise_mode,
            )
            w_idx += 1
        elif block.architecture == "resnet":
            y = block.skip(x, gain=float(0.5**0.5))
            x = _layer(
                block.conv0,
                x,
                ws[:, w_idx],
                cfg=cfg,
                noise_mode=noise_mode,
            )
            w_idx += 1
            x = _layer(
                block.conv1,
                x,
                ws[:, w_idx],
                cfg=cfg,
                noise_mode=noise_mode,
                gain=float(0.5**0.5),
            )
            w_idx += 1
            x = y.add_(x)
        else:
            x = _layer(
                block.conv0,
                x,
                ws[:, w_idx],
                cfg=cfg,
                noise_mode=noise_mode,
            )
            w_idx += 1
            x = _layer(
                block.conv1,
                x,
                ws[:, w_idx],
                cfg=cfg,
                noise_mode=noise_mode,
            )
            w_idx += 1

        if img is not None:
            img = upfirdn2d.upsample2d(img, block.resample_filter)
        if block.is_last or block.architecture == "skip":
            y = block.torgb(x, ws[:, w_idx], fused_modconv=True)
            y = y.to(
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )
            img = img.add_(y) if img is not None else y

        # ToRGB overlaps the next block's first style in StyleGAN2.
        w_base += int(block.num_conv)

    return img

class OptimizedEarlyOutputSynthesis(nn.Module):
    """Prepared StyleGAN prefix followed by an early-output decoder.

    The regular :class:`OptimizedSynthesis` evaluates the StyleGAN RGB skip
    pyramid.  Early-output generators instead keep only the feature path
    through b256 and pass that activation to ``synthesis.decoder``.
    """

    def __init__(self, synthesis: nn.Module, cfg: Any):
        super().__init__()
        if not hasattr(synthesis, "decoder"):
            raise TypeError("Early-output synthesis must define a decoder module")
        self.synthesis = synthesis
        self.cfg = cfg
        self.resolutions = tuple(int(r) for r in synthesis.block_resolutions)
        synthesis.requires_grad_(False)
        self.prepare()

    def prepare(self, *, force_fp32=False):
        return prepare_optimized_synthesis(self.synthesis, self.cfg, force_fp32=force_fp32)

    def _apply(self, fn, recurse=True):
        clear_prepared_synthesis(self.synthesis)
        return super()._apply(fn, recurse=recurse)

    def forward(
        self,
        ws: torch.Tensor,
        noise_mode: str = "const",
        force_fp32: bool = False,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            self.prepare(force_fp32=force_fp32)
        ws = ws.to(torch.float32)
        features = None
        w_base = 0
        low_dtype = getattr(self.cfg, "low_precision_dtype", torch.float16)

        for resolution in self.resolutions:
            block = getattr(self.synthesis, f"b{resolution}")
            dtype = (
                low_dtype
                if (block.use_fp16 or bool(self.cfg.force_fp16_all_blocks))
                and not force_fp32
                else torch.float32
            )
            memory_format = (
                torch.channels_last
                if block.channels_last and not force_fp32
                else torch.contiguous_format
            )
            if block.in_channels == 0:
                features = block.const.to(dtype=dtype, memory_format=memory_format)
                features = features.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            else:
                features = features.to(dtype=dtype, memory_format=memory_format)

            w_idx = w_base
            if block.in_channels == 0:
                features = _layer(
                    block.conv1,
                    features,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
            elif block.architecture == "resnet":
                skip = block.skip(features, gain=float(0.5**0.5))
                features = _layer(
                    block.conv0,
                    features,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
                w_idx += 1
                features = _layer(
                    block.conv1,
                    features,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                    gain=float(0.5**0.5),
                )
                features = skip.add_(features)
            else:
                features = _layer(
                    block.conv0,
                    features,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
                w_idx += 1
                features = _layer(
                    block.conv1,
                    features,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )

            # ToRGB consumes an overlapping style slot in StyleGAN2.  We skip
            # it, so only convolution slots advance the next block's base.
            w_base += int(block.num_conv)

        if features is None:
            raise RuntimeError("Early synthesis produced no prefix activation")

        autocast = (
            torch.autocast(device_type="cuda", dtype=low_dtype)
            if features.device.type == "cuda" and not force_fp32 and low_dtype != torch.float32
            else nullcontext()
        )
        with autocast:
            output = self.synthesis.decoder(features.float())
        return output


def build_optimized_synthesis(
    synthesis: nn.Module,
    cfg: Any,
    *,
    copy_module: bool = True,
) -> OptimizedSynthesis:
    """Build frozen inference, copying the source unless explicitly shared."""
    candidate = copy.deepcopy(synthesis) if copy_module else synthesis
    optimized = OptimizedSynthesis(candidate, cfg)
    optimized.eval()
    for parameter in optimized.parameters():
        parameter.requires_grad_(False)
    return optimized


def build_optimized_early_output_synthesis(
    synthesis: nn.Module,
    cfg: Any,
    *,
    copy_module: bool = True,
) -> OptimizedEarlyOutputSynthesis:
    """Build the optimized prefix path without evaluating StyleGAN toRGBs."""
    candidate = copy.deepcopy(synthesis) if copy_module else synthesis
    optimized = OptimizedEarlyOutputSynthesis(candidate, cfg)
    optimized.eval()
    for parameter in optimized.parameters():
        parameter.requires_grad_(False)
    return optimized


def direct_synthesis_forward(synthesis, ws, *, noise_mode="random", force_fp32=False,
                             low_precision=None, fused_modconv=None, update_emas=False):
    """Per-call inference policy for a directly loaded Generator.

    Unspecified precision follows ambient autocast, otherwise the original
    CUDA-FP16 / non-CUDA-FP32 policy. Explicit low_precision selects the tagged
    high blocks; force_fp32 overrides both it and ambient autocast. The shared
    algorithm does not construct per-sample fused convolution weights, so the
    original fused_modconv hint is accepted but is not needed here.
    """
    from .inference_opt import inference_config
    from .native_backend import get_bias_act_impl, get_upfirdn_impl, use_native_ops

    if low_precision is None:
        if torch.is_autocast_enabled(ws.device.type):
            low_precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}[torch.get_autocast_dtype(ws.device.type)]
        else:
            low_precision = "fp16" if ws.device.type == "cuda" else "fp32"
    cfg = inference_config(low_precision="fp32" if force_fp32 else low_precision)
    # Respect explicit caller backend contexts; default to CUDA/Metal with BF16
    # fallback. Disable outer AMP after reading its requested precision so it
    # cannot silently change FP32 low blocks or cache construction arithmetic.
    with use_native_ops(upfirdn=get_upfirdn_impl("auto"), bias_act=get_bias_act_impl("auto")), \
         torch.autocast(ws.device.type, enabled=False):
        return optimized_synthesis_forward(synthesis, ws, cfg,
                                           noise_mode=noise_mode, force_fp32=force_fp32)
