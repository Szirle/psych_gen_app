"""Standalone inference runtime for integrated StyleGAN2 early-output models.

The supported checkpoints contain the StyleGAN2 mapping network, synthesis
prefix through b128 or b256, and the learned RGB decoder in one ``.pt`` file.
This module intentionally depends only on PyTorch and the StyleGAN2 runtime
already vendored in ``psych_gen_app/stylegan3``.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
STYLEGAN_RUNTIME = PROJECT_ROOT / "stylegan3"
if str(STYLEGAN_RUNTIME) not in sys.path:
    sys.path.insert(0, str(STYLEGAN_RUNTIME))

from stylegan3.training.networks_stylegan2 import (  # noqa: E402
    MappingNetwork,
    SynthesisBlock,
)


CHECKPOINT_FORMAT = "stylegan2_early_output_generator_v2"
SUPPORTED_RESOLUTIONS = (128, 256)
NUM_WS_BY_RESOLUTION = {128: 12, 256: 14}
SOURCE_RESOLUTION = 1024


def select_accelerator(preferred: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Choose CUDA, then MPS, then CPU, or validate an explicit preference."""
    requested = "auto" if preferred is None else str(preferred).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but the PyTorch MPS backend is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported accelerator {device.type!r}; expected cpu, cuda, or mps")
    return device


def _torch_load(path: Union[str, os.PathLike]) -> Mapping:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(path, map_location="cpu")


class FastEarlyOutputDecoder(nn.Module):
    """Deployment decoder stored by the integrated v2 checkpoint format."""

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.linear = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.mix_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.mix_out = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        return self.linear(features) + self.mix_out(
            self.activation(self.mix_in(features))
        )


class EarlyOutputSynthesis(nn.Module):
    """StyleGAN2 synthesis prefix followed by the learned RGB decoder."""

    def __init__(
        self,
        *,
        w_dim: int,
        img_resolution: int,
        img_channels: int = 3,
        decoder_hidden_channels: int = 16,
        channel_base: int = 32768,
        channel_max: int = 512,
        source_num_fp16_res: int = 4,
    ):
        super().__init__()
        if img_resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Early-output resolution must be one of {SUPPORTED_RESOLUTIONS}, "
                f"got {img_resolution}"
            )
        if img_channels != 3:
            raise ValueError("Early-output checkpoints must contain three image channels")

        self.w_dim = int(w_dim)
        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        self.block_resolutions = tuple(
            2**index for index in range(2, int(np.log2(img_resolution)) + 1)
        )
        channels = {
            resolution: min(channel_base // resolution, channel_max)
            for resolution in self.block_resolutions
        }
        source_log2 = int(np.log2(SOURCE_RESOLUTION))
        fp16_resolution = max(2 ** (source_log2 + 1 - source_num_fp16_res), 8)

        self.num_ws = 0
        for resolution in self.block_resolutions:
            block = SynthesisBlock(
                channels[resolution // 2] if resolution > 4 else 0,
                channels[resolution],
                w_dim=w_dim,
                resolution=resolution,
                img_channels=img_channels,
                is_last=resolution == img_resolution,
                use_fp16=resolution >= fp16_resolution,
            )
            self.num_ws += block.num_conv
            if resolution == img_resolution:
                self.num_ws += block.num_torgb
            setattr(self, f"b{resolution}", block)

        expected_num_ws = NUM_WS_BY_RESOLUTION[img_resolution]
        if self.num_ws != expected_num_ws:
            raise RuntimeError(
                f"Expected {expected_num_ws} W slots for b{img_resolution}, got {self.num_ws}"
            )
        decoder_channels = channels[img_resolution]
        self.decoder = FastEarlyOutputDecoder(
            decoder_channels, hidden_channels=decoder_hidden_channels
        )

    def forward(
        self,
        ws: torch.Tensor,
        noise_mode: str = "const",
        force_fp32: bool = True,
        **_kwargs,
    ) -> torch.Tensor:
        dtype = torch.float32 if force_fp32 else torch.float16
        return _SynthesisRuntime(self, dtype)(
            ws, noise_mode=noise_mode, force_fp32=force_fp32
        )


class EarlyOutputGenerator(nn.Module):
    """Checkpoint-compatible integrated b128/b256 generator definition."""

    def __init__(
        self,
        *,
        z_dim: int = 512,
        c_dim: int = 0,
        w_dim: int = 512,
        img_resolution: int,
        img_channels: int = 3,
        decoder_hidden_channels: int = 16,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.c_dim = int(c_dim)
        self.w_dim = int(w_dim)
        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        self.num_ws = NUM_WS_BY_RESOLUTION[self.img_resolution]
        self.mapping = MappingNetwork(
            z_dim=self.z_dim,
            c_dim=self.c_dim,
            w_dim=self.w_dim,
            num_ws=self.num_ws,
        )
        self.synthesis = EarlyOutputSynthesis(
            w_dim=self.w_dim,
            img_resolution=self.img_resolution,
            img_channels=self.img_channels,
            decoder_hidden_channels=decoder_hidden_channels,
        )

    def forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        **synthesis_kwargs,
    ) -> torch.Tensor:
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        return self.synthesis(ws, **synthesis_kwargs)


class _SynthesisRuntime(nn.Module):
    """ToRGB-free synthesis loop with explicit accelerator-safe precision."""

    def __init__(self, synthesis: EarlyOutputSynthesis, compute_dtype: torch.dtype):
        super().__init__()
        self.synthesis = synthesis
        self.compute_dtype = compute_dtype

    def forward(
        self,
        ws: torch.Tensor,
        noise_mode: str = "const",
        force_fp32: bool = False,
    ) -> torch.Tensor:
        if noise_mode not in {"const", "random", "none"}:
            raise ValueError(f"Unsupported noise mode {noise_mode!r}")
        ws = ws.float()
        features = None
        w_base = 0

        for resolution in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f"b{resolution}")
            use_low_precision = (
                self.compute_dtype != torch.float32
                and block.use_fp16
                and not force_fp32
            )
            dtype = self.compute_dtype if use_low_precision else torch.float32
            memory_format = (
                torch.channels_last
                if block.channels_last and use_low_precision
                else torch.contiguous_format
            )

            if block.in_channels == 0:
                features = block.const.to(dtype=dtype, memory_format=memory_format)
                features = features.unsqueeze(0).repeat(ws.shape[0], 1, 1, 1)
            else:
                features = features.to(dtype=dtype, memory_format=memory_format)

            w_index = w_base
            if block.in_channels == 0:
                features = block.conv1(
                    features,
                    ws[:, w_index],
                    noise_mode=noise_mode,
                    fused_modconv=False,
                )
            elif block.architecture == "resnet":
                skip = block.skip(features, gain=float(0.5**0.5))
                features = block.conv0(
                    features,
                    ws[:, w_index],
                    noise_mode=noise_mode,
                    fused_modconv=False,
                )
                w_index += 1
                features = block.conv1(
                    features,
                    ws[:, w_index],
                    noise_mode=noise_mode,
                    fused_modconv=False,
                    gain=float(0.5**0.5),
                )
                features = skip.add_(features)
            else:
                features = block.conv0(
                    features,
                    ws[:, w_index],
                    noise_mode=noise_mode,
                    fused_modconv=False,
                )
                w_index += 1
                features = block.conv1(
                    features,
                    ws[:, w_index],
                    noise_mode=noise_mode,
                    fused_modconv=False,
                )

            # StyleGAN2 ToRGB shares a style slot with the following block.
            w_base += int(block.num_conv)

        if features is None:
            raise RuntimeError("StyleGAN2 prefix produced no activation")
        return self.synthesis.decoder(features.float())


class EarlyOutputStyleGAN:
    """Load and run integrated StyleGAN2 b128/b256 checkpoints.

    ``device='auto'`` selects CUDA, then MPS, then CPU. ``precision='auto'``
    uses FP16 for the high-resolution prefix blocks on CUDA/MPS and FP32 on
    CPU. Mapping, low-resolution synthesis blocks, and the learned decoder
    remain FP32, matching the checkpoint's intended mixed-precision contract.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike],
        *,
        device: Optional[Union[str, torch.device]] = "auto",
        precision: str = "auto",
        compile: bool = False,
        compile_mode: str = "default",
        noise_mode: str = "const",
    ):
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Early-output checkpoint not found: {self.checkpoint_path}")

        self.device = select_accelerator(device)
        self.precision = self._resolve_precision(precision)
        self.compute_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        self.noise_mode = str(noise_mode)
        self.compile_enabled = bool(compile)
        self.compile_mode = str(compile_mode)

        checkpoint = _torch_load(self.checkpoint_path)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Not a {CHECKPOINT_FORMAT} checkpoint: {self.checkpoint_path}"
            )
        config = dict(checkpoint.get("config", {}))
        resolution = int(config.get("img_resolution", 0))
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Checkpoint resolution must be one of {SUPPORTED_RESOLUTIONS}, got {resolution}"
            )

        self.generator = EarlyOutputGenerator(**config)
        self.generator.load_state_dict(checkpoint["generator"], strict=True)
        self.generator.eval().requires_grad_(False).to(self.device)
        self.G = self.generator  # Compatibility with the app's existing backend.
        self.z_dim = self.generator.z_dim
        self.num_ws = self.generator.num_ws
        self.output_resolution = self.generator.img_resolution

        self._runtime = _SynthesisRuntime(
            self.generator.synthesis, self.compute_dtype
        ).eval()
        self._compiled_runtime = None
        if self.compile_enabled:
            self._compiled_runtime = torch.compile(
                self._runtime,
                mode=self.compile_mode,
                fullgraph=False,
            )

    def _resolve_precision(self, precision: str) -> str:
        value = str(precision).lower()
        if value == "auto":
            return "fp16" if self.device.type in {"cuda", "mps"} else "fp32"
        aliases = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
        value = aliases.get(value, value)
        if value not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        return value

    @staticmethod
    def _to_nhwc_uint8(images: torch.Tensor) -> np.ndarray:
        images = (images.float().clamp(-1, 1) + 1) * 127.5
        return (
            images.round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

    def _as_float_tensor(self, value: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a tensor or NumPy array, got {type(value).__name__}")
        return value.to(device=self.device, dtype=torch.float32)

    def map_z(self, z: Union[np.ndarray, torch.Tensor], truncation_psi: float = 1.0) -> torch.Tensor:
        z_tensor = self._as_float_tensor(z)
        with torch.inference_mode():
            return self.generator.mapping(
                z_tensor, None, truncation_psi=float(truncation_psi)
            )

    def synthesize(
        self,
        ws: Union[np.ndarray, torch.Tensor],
        *,
        noise_mode: Optional[str] = None,
    ) -> torch.Tensor:
        ws_tensor = self._as_float_tensor(ws)
        if ws_tensor.ndim == 2:
            ws_tensor = ws_tensor.unsqueeze(1).repeat(1, self.num_ws, 1)
        if ws_tensor.ndim != 3 or ws_tensor.shape[1] != self.num_ws:
            raise ValueError(
                f"Expected W tensor [N, {self.num_ws}, {self.generator.w_dim}], "
                f"got {tuple(ws_tensor.shape)}"
            )
        selected_noise = self.noise_mode if noise_mode is None else str(noise_mode)
        runtime = (
            self._compiled_runtime
            if self._compiled_runtime is not None
            else self._runtime
        )
        try:
            with torch.inference_mode():
                return runtime(ws_tensor, noise_mode=selected_noise, force_fp32=False)
        except Exception:
            if self._compiled_runtime is None:
                raise
            warnings.warn(
                "torch.compile failed for the early-output generator; falling back to eager inference",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_runtime = None
            self.compile_enabled = False
            with torch.inference_mode():
                return self._runtime(
                    ws_tensor, noise_mode=selected_noise, force_fp32=False
                )

    def generate_from_z(
        self,
        z: Union[np.ndarray, torch.Tensor],
        *,
        truncation_psi: float = 1.0,
        noise_mode: Optional[str] = None,
    ) -> torch.Tensor:
        return self.synthesize(
            self.map_z(z, truncation_psi=truncation_psi), noise_mode=noise_mode
        )

    def generate_im_from_w_space(
        self,
        ws: Union[np.ndarray, torch.Tensor],
        resolution: Optional[int] = None,
    ) -> np.ndarray:
        if resolution is not None and int(resolution) != self.output_resolution:
            raise ValueError(
                f"This checkpoint produces {self.output_resolution}x{self.output_resolution}; "
                f"requested {resolution}"
            )
        return self._to_nhwc_uint8(self.synthesize(ws))

    def generate_im_from_z_space(
        self,
        z: Union[np.ndarray, torch.Tensor],
        truncation_psi: float = 0.5,
    ) -> np.ndarray:
        return self._to_nhwc_uint8(
            self.generate_from_z(z, truncation_psi=truncation_psi)
        )

    def generate_im_from_random_seed(
        self,
        seed: int = 22,
        truncation_psi: float = 0.5,
    ) -> np.ndarray:
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        z = torch.randn(
            1, self.z_dim, generator=generator, device=self.device
        )
        return self.generate_im_from_z_space(z, truncation_psi=truncation_psi)

    def warmup(self, batch_size: int = 1) -> None:
        z = torch.zeros(batch_size, self.z_dim, device=self.device)
        self.generate_from_z(z)

    def describe(self) -> dict:
        return {
            "checkpoint": self.checkpoint_path,
            "device": self.device.type,
            "precision": self.precision,
            "compiled": self._compiled_runtime is not None,
            "resolution": self.output_resolution,
            "num_ws": self.num_ws,
        }
