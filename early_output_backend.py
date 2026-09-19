"""Web preview adapter for the 128/256 StyleGAN2 pointwise_style32 models."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from StyleGAN2_mps.early_output_model import (
    load_generator_checkpoint,
    build_optimized_early_output_synthesis,
)
from StyleGAN2_mps.torch_utils.ops.inference_opt import (
    InferenceOptConfig,
    inference_config,
    normalize_scalar_attrs,
)
from StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops


def select_accelerator(
    preferred: Optional[Union[str, torch.device]] = None,
) -> torch.device:
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
            raise RuntimeError(
                "MPS was requested, but the PyTorch MPS backend is unavailable"
            )
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported accelerator {device.type!r}; expected cpu, cuda, or mps"
        )
    return device


class EarlyOutputStyleGAN:
    """Run a trained 128x128 or 256x256 pointwise_style32 generator.

    ``device='auto'`` selects CUDA, then MPS, then CPU. ``precision='auto'``
    uses FP16 high-resolution blocks on CUDA/MPS and FP32 on CPU. Mapping stays
    FP32; the conditioned head uses CUDA autocast and FP32 on MPS/CPU.
    Shared convolution and polyphase FIR are enabled on every device. Eager
    execution uses Metal/CUDA kernels, with PyTorch fallback for BF16 tensors;
    compilation uses traceable PyTorch ops.
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
            raise FileNotFoundError(
                f"Early-output checkpoint not found: {self.checkpoint_path}"
            )

        self.device = select_accelerator(device)
        self.precision = self._resolve_precision(precision)
        self.compute_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        self.force_fp32 = self.compute_dtype == torch.float32
        self.noise_mode = str(noise_mode)
        if self.noise_mode not in {"const", "random", "none"}:
            raise ValueError("noise_mode must be const, random, or none")
        self.compile_enabled = bool(compile)
        self.compile_mode = str(compile_mode)

        self.generator = load_generator_checkpoint(
            self.checkpoint_path, device=self.device
        )
        normalize_scalar_attrs(self.generator)
        self.generator.eval().requires_grad_(False)

        self._runtime = build_optimized_early_output_synthesis(
            self.generator.synthesis,
            self._optimization_config(),
            copy_module=False,
        ).eval()
        normalize_scalar_attrs(self._runtime)

        self.G = self.generator  # Compatibility with FastStyleGANBackend.
        self.z_dim = int(self.generator.z_dim)
        self.num_ws = int(self.generator.num_ws)
        self.output_resolution = int(self.generator.img_resolution)
        self._compiled_runtime = None
        if self.compile_enabled:
            self._prepare_compiler()

    def _resolve_precision(self, precision: str) -> str:
        value = str(precision).lower()
        if value == "auto":
            return "fp16" if self.device.type in {"cuda", "mps"} else "fp32"
        aliases = {
            "float32": "fp32",
            "float16": "fp16",
            "bfloat16": "bf16",
        }
        value = aliases.get(value, value)
        if value not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        return value

    def _optimization_config(self) -> InferenceOptConfig:
        # Match the benchmark's shared-convolution/polyphase policy on MPS
        # as well as CUDA. FP32 must prepare FP32, not unused FP16 kernels.
        return inference_config(low_precision=self.precision)

    def _runtime_forward(self, ws: torch.Tensor) -> torch.Tensor:
        return self._runtime(
            ws,
            noise_mode=self.noise_mode,
            force_fp32=self.force_fp32,
        )

    def _prepare_compiler(self) -> None:
        # Prepare only the active precision before Dynamo starts tracing.
        self._runtime.prepare(force_fp32=self.force_fp32)
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.cache_size_limit = max(
                int(torch._dynamo.config.cache_size_limit), 128
            )
            if hasattr(torch._dynamo.config, "recompile_limit"):
                torch._dynamo.config.recompile_limit = max(
                    int(torch._dynamo.config.recompile_limit), 128
                )
        self._compiled_runtime = torch.compile(
            self._runtime_forward,
            mode=self.compile_mode,
            fullgraph=False,
        )

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

    def _as_float_tensor(
        self, value: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Expected a tensor or NumPy array, got {type(value).__name__}"
            )
        return value.to(device=self.device, dtype=torch.float32)

    def map_z(
        self,
        z: Union[np.ndarray, torch.Tensor],
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        z_tensor = self._as_float_tensor(z)
        with torch.inference_mode(), torch.autocast(
            self.device.type, enabled=False
        ), use_native_ops(upfirdn="auto", bias_act="auto"):
            return self.generator.mapping(
                z_tensor,
                None,
                truncation_psi=float(truncation_psi),
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
        if ws_tensor.ndim != 3 or ws_tensor.shape[1:] != (self.num_ws, self.generator.w_dim):
            raise ValueError(
                f"Expected W tensor [N, {self.num_ws}, {self.generator.w_dim}], "
                f"got {tuple(ws_tensor.shape)}"
            )

        selected_noise = self.noise_mode if noise_mode is None else str(noise_mode)
        if selected_noise not in {"const", "random", "none"}:
            raise ValueError("noise_mode must be const, random, or none")
        use_compiled = (
            self._compiled_runtime is not None and selected_noise == self.noise_mode
        )
        # Preserve explicit block precision even if the webapp caller has an
        # outer autocast context. Only compiled calls require PyTorch-only ops.
        backend = "native" if use_compiled else "auto"
        try:
            with torch.inference_mode(), torch.autocast(
                self.device.type, enabled=False
            ), use_native_ops(upfirdn=backend, bias_act=backend):
                if use_compiled:
                    return self._compiled_runtime(ws_tensor)
                return self._runtime(
                    ws_tensor,
                    noise_mode=selected_noise,
                    force_fp32=self.force_fp32,
                )
        except Exception as error:
            if not use_compiled:
                raise
            warnings.warn(
                "torch.compile failed for StyleGAN2_mps; falling back to eager "
                f"inference ({type(error).__name__}: {error})",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_runtime = None
            self.compile_enabled = False
            with torch.inference_mode(), torch.autocast(
                self.device.type, enabled=False
            ), use_native_ops(upfirdn="auto", bias_act="auto"):
                return self._runtime(
                    ws_tensor,
                    noise_mode=selected_noise,
                    force_fp32=self.force_fp32,
                )

    def generate_from_z(
        self,
        z: Union[np.ndarray, torch.Tensor],
        *,
        truncation_psi: float = 1.0,
        noise_mode: Optional[str] = None,
    ) -> torch.Tensor:
        return self.synthesize(
            self.map_z(z, truncation_psi=truncation_psi),
            noise_mode=noise_mode,
        )

    def generate_im_from_w_space(
        self,
        ws: Union[np.ndarray, torch.Tensor],
        resolution: Optional[int] = None,
    ) -> np.ndarray:
        if resolution is not None and int(resolution) != self.output_resolution:
            raise ValueError(
                f"This checkpoint produces {self.output_resolution}x"
                f"{self.output_resolution}; requested {resolution}"
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
        # MPS has no device-local Generator. CPU sampling also makes a seed
        # identify the same Z input on each inference backend.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(
            1,
            self.z_dim,
            generator=generator,
            device="cpu",
        )
        return self.generate_im_from_z_space(
            z, truncation_psi=truncation_psi
        )

    def warmup(self, batch_size: int = 1) -> None:
        z = torch.zeros(batch_size, self.z_dim, device=self.device)
        self.generate_from_z(z)

    def describe(self) -> dict:
        return {
            "checkpoint": self.checkpoint_path,
            "runtime": "StyleGAN2_mps",
            "architecture": "pointwise_style32",
            "device": self.device.type,
            "precision": self.precision,
            "compiled": self._compiled_runtime is not None,
            "optimized_synthesis": True,
            "fir_compose_mode": self._runtime.cfg.fir_compose_mode,
            "op_backend": "native" if self._compiled_runtime is not None else "auto",
            "resolution": self.output_resolution,
            "num_ws": self.num_ws,
        }
