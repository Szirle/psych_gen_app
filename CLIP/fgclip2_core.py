"""Reusable FG-CLIP 2 inference, region alignment, and differentiable energies.

Tested dependencies (in addition to a matching torch/torchvision installation):
    pip install transformers==4.57.6 huggingface-hub==0.36.2 sentencepiece pillow accelerate psutil

The official checkpoint includes custom Python code. The default So400m revision
is pinned below; weights/tokenizer/code are downloaded into ../models/fgclip2.
No UI dependencies, server, or unrelated project modules are imported here.

Inference:
    fg = FGCLIP2()
    result = fg.agreement(["face.png"], ["a smiling person"])
    maps = fg.dense_alignments("face.png", ["mouth", "eyes"])
    regions = fg.region_alignment("face.png", [[20, 30, 100, 150]], ["smile"])

Differentiable generator -> RGB -> energy (weights frozen by default):
    rgb = generator(z)                         # BCHW, floating RGB in [0, 1]
    inputs = fg.prepare_image_tensor(rgb)      # resize/normalize/patchify in torch
    text = fg.encode_text(["a smiling person"]).detach()  # cache outside loop
    features = fg.encode_image_preprocessed(inputs)
    energy = -fg.pairwise_logits_from_features(features, text).sum()
    energy.backward()                         # gradients reach z

PIL convenience methods use inference_mode. Tensor/preprocessed methods do not.
Text tokenization and box coordinates are discrete; gradients flow through image
pixels/features, not through strings or box coordinates. Scores are uncalibrated
agreement logits, not probabilities. Dense cosine maps are semantic alignment,
not causal attribution or segmentation masks.

Reference: https://github.com/360CVGroup/FG-CLIP (FG-CLIP 2 release).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

DEFAULT_MODEL_ID = "qihoo360/fg-clip2-so400m"
DEFAULT_REVISION = "d57d30fe94a107dd6a2610eb4e9a135004d823a4"
BASE_MODEL_ID = "qihoo360/fg-clip2-base"
BASE_REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "fgclip2"
ImageLike = str | os.PathLike | Image.Image
TensorDict = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class Agreement:
    """Matrices [images (or regions), texts]; remain attached in tensor APIs."""

    cosine: torch.Tensor
    logits: torch.Tensor

    @property
    def energy(self) -> torch.Tensor:
        return -self.logits


@dataclass(frozen=True)
class DenseAlignment:
    similarity: torch.Tensor  # [grid_h, grid_w], CPU in PIL convenience API
    grid_size: tuple[int, int]
    image_size: tuple[int, int]  # original, EXIF-oriented (width, height)


def download_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local model or resume missing Hub files into the explicit cache.

    A complete cached snapshot is reused without a network request. To update a
    model, request a new revision explicitly. Offline mode fails clearly when a
    required file is missing. Hugging Face handles locks and partial downloads.
    """
    if Path(model_id).is_dir():
        return Path(model_id).resolve()
    revision = revision or {
        DEFAULT_MODEL_ID: DEFAULT_REVISION,
        BASE_MODEL_ID: BASE_REVISION,
    }.get(model_id, "main")
    kwargs = {"repo_id": model_id, "revision": revision, "cache_dir": str(cache_dir)}
    try:
        cached = Path(snapshot_download(**kwargs, local_files_only=True))
        required = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "modeling_fgclip2.py",
            "configuration_fgclip2.py",
        ]
        index = cached / "model.safetensors.index.json"
        weights = (
            list(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        if all((cached / name).is_file() for name in required + weights):
            return cached
    except (OSError, ValueError, KeyError):
        pass
    if local_files_only:
        raise FileNotFoundError(
            f"Complete checkpoint {model_id}@{revision} not found in {cache_dir}. Run once online to download it."
        )
    return Path(
        snapshot_download(
            **kwargs,
            allow_patterns=["*.json", "*.py", "*.model", "*.safetensors"],
            max_workers=4,
        )
    )


class FGCLIP2:
    """FG-CLIP 2 wrapper with explicit global/long/box representations.

    Float16 is the default on MPS/CUDA; CPU defaults to float32. Use bfloat16
    on supported hardware for a wider gradient range, or float32 if memory permits.
    Check gradients for finiteness when optimizing in reduced precision.
    Use one instance per worker; the demo serializes access to a shared instance.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device=None,
        dtype=None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        freeze: bool = True,
        memory_reserve_gb: float = 1.25,
    ) -> None:
        if device is None or str(device) == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.dtype = dtype or (
            torch.float16 if self.device.type in {"mps", "cuda"} else torch.float32
        )
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is unavailable. Run outside the sandbox or select device='cpu'."
            )
        self.model_id = model_id
        self.model_path = download_model(
            model_id,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.revision = self.model_path.name
        common = {"local_files_only": True, "trust_remote_code": trust_remote_code}
        self.model = self._load_streamed(common, memory_reserve_gb).eval()
        self.model.requires_grad_(not freeze)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), **common)
        self.image_processor = AutoImageProcessor.from_pretrained(
            str(self.model_path), **common, use_fast=True
        )

    def _load_streamed(self, common: dict, reserve_gb: float):
        """Initialize on meta, then copy safetensor slices directly to the device.

        Avoid simultaneous full CPU float32 and accelerator model copies. Large
        embedding matrices are copied in <=8 MiB slices. No weights are modified
        on disk; the downloaded official safetensors remain reusable.
        """
        import psutil
        from accelerate import init_empty_weights
        from safetensors import safe_open

        if reserve_gb < 0.5:
            raise ValueError("Keep at least 0.5 GB of memory reserve.")
        index = self.model_path / "model.safetensors.index.json"
        shards = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        specs = {}
        for shard in shards:
            with safe_open(self.model_path / shard, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 - safe_open is not a dictionary
                    specs[key] = (shard, f.get_slice(key).get_shape())
        weight_bytes = (
            sum(math.prod(shape) for _, shape in specs.values())
            * torch.empty((), dtype=self.dtype).element_size()
        )
        needed = weight_bytes + reserve_gb * 1024**3
        available = psutil.virtual_memory().available
        if self.device.type == "cuda":
            available = min(available, torch.cuda.mem_get_info(self.device)[0])
        if available < needed:
            raise MemoryError(
                f"Loading {self.model_id} in {self.dtype} needs about {needed / 1024**3:.1f} GiB available including reserve; only {available / 1024**3:.1f} GiB is available. Close memory-heavy apps or select Base/half precision."
            )
        if self.device.type == "mps":
            # Fail allocations in this process before exhausting shared system RAM.
            budget = min(
                available - 0.5 * 1024**3, psutil.virtual_memory().total * 0.30
            )
            torch.mps.set_per_process_memory_fraction(
                min(0.6, budget / torch.mps.recommended_max_memory())
            )
        config = AutoConfig.from_pretrained(str(self.model_path), **common)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=common["trust_remote_code"],
                dtype=self.dtype,
                attn_implementation="sdpa",
            )
        expected = dict(model.named_parameters())
        if set(expected) != set(specs):
            raise RuntimeError(
                f"Checkpoint parameter mismatch: missing={set(expected) - set(specs)}, unexpected={set(specs) - set(expected)}"
            )
        with torch.no_grad():
            for shard in shards:
                with safe_open(
                    self.model_path / shard, framework="pt", device="cpu"
                ) as f:
                    for name in f.keys():  # noqa: SIM118 - safe_open is not a dictionary
                        shape = specs[name][1]
                        if tuple(expected[name].shape) != tuple(shape):
                            raise RuntimeError(f"Checkpoint shape mismatch for {name}.")
                        value = torch.empty(shape, device=self.device, dtype=self.dtype)
                        source = f.get_slice(name)
                        if shape:
                            rows = max(
                                1, 8 * 1024**2 // (max(1, math.prod(shape[1:])) * 4)
                            )
                            for start in range(0, shape[0], rows):
                                value[start : start + rows].copy_(
                                    source[start : start + rows]
                                )
                        else:
                            value.copy_(f.get_tensor(name))
                        module_name, _, leaf = name.rpartition(".")
                        module = (
                            model.get_submodule(module_name) if module_name else model
                        )
                        setattr(
                            module, leaf, torch.nn.Parameter(value, requires_grad=False)
                        )
            for module in model.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    setattr(module, name, buffer.to(self.device))
        return model

    @staticmethod
    def _as_pil(image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(image) as im:
            return ImageOps.exif_transpose(im).convert("RGB")

    @staticmethod
    def _texts(texts: Sequence[str] | str) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Provide at least one nonempty text description.")
        return [t.strip().lower() for t in texts]

    @staticmethod
    def recommended_max_patches(image: Image.Image) -> int:
        w, h = image.size
        count = (w // 16) * (h // 16)
        for threshold, budget in [(784, 1024), (576, 784), (256, 576), (128, 256)]:
            if count > threshold:
                return budget
        return 128

    @staticmethod
    def _patch_budget(value: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 16384
        ):
            raise ValueError("max_num_patches must be an integer between 1 and 16384.")
        return value

    def _to_device(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        return {
            k: v.to(
                device=self.device,
                dtype=self.dtype if v.is_floating_point() else v.dtype,
            )
            for k, v in batch.items()
        }

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        # Similarity/scale arithmetic stays float32 even with low-precision weights.
        return F.normalize(features.float(), dim=-1)

    def prepare_image(
        self, image: ImageLike, *, max_num_patches: int | None = None
    ) -> dict[str, torch.Tensor]:
        return self.prepare_images([image], max_num_patches=max_num_patches)

    def prepare_images(
        self,
        images: Sequence[ImageLike],
        *,
        max_num_patches: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Preprocess an image batch with one shared patch budget."""
        images = list(images)
        if not images:
            raise ValueError("Provide at least one image.")
        pil_images = [self._as_pil(image) for image in images]
        budget = self._patch_budget(
            max_num_patches
            if max_num_patches is not None
            else min(self.recommended_max_patches(image) for image in pil_images)
        )
        return self._to_device(
            self.image_processor(
                images=pil_images, max_num_patches=budget, return_tensors="pt"
            )
        )

    def prepare_image_tensor(
        self, images: torch.Tensor, *, max_num_patches: int = 256
    ) -> dict[str, torch.Tensor]:
        """Differentiable BCHW/CHW RGB [0,1] -> native padded patch tensors.

        Uses the official fast torch processor, including aspect-preserving
        resizing and patch layout. Never clamps or detaches caller pixels.
        Floating resize avoids PIL's uint8 rounding; small numerical differences
        from prepare_image(PIL) are expected. Range validation is caller-owned
        so optimization can explore values outside [0,1] without clipping gradients.
        """
        self._patch_budget(max_num_patches)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if (
            images.ndim != 4
            or images.shape[1] != 3
            or not images.is_floating_point()
            or min(images.shape) == 0
        ):
            raise ValueError(
                "Expected nonempty floating RGB tensor [B,3,H,W] or [3,H,W]."
            )
        batch = self.image_processor(
            images=list(images.unbind(0)),
            input_data_format="channels_first",
            do_rescale=False,
            max_num_patches=max_num_patches,
            return_tensors="pt",
        )
        return self._to_device(batch)

    def text_info(self, texts: Sequence[str] | str, *, mode: str = "auto") -> dict:
        texts = self._texts(texts)
        if mode not in {"auto", "short", "long", "box"}:
            raise ValueError("text mode must be auto, short, long, or box.")
        counts = [
            len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]
        ]
        resolved = ("long" if max(counts) > 64 else "short") if mode == "auto" else mode
        limit = 196 if resolved == "long" else 64
        return {
            "mode": resolved,
            "token_counts": counts,
            "max_length": limit,
            "truncated": [n > limit for n in counts],
        }

    def prepare_text(
        self, texts: Sequence[str] | str, *, mode: str = "auto", truncate: bool = False
    ):
        """Select one text head for the entire batch; reject silent truncation.

        Auto selects long if any description exceeds 64 tokens (including EOS).
        For experiments, explicitly choose a mode so batch composition cannot
        alter a description's representation. truncate=True explicitly opts in.
        """
        texts = self._texts(texts)
        info = self.text_info(texts, mode=mode)
        if any(info["truncated"]) and not truncate:
            raise ValueError(
                f"Text exceeds the {info['max_length']}-token {info['mode']} limit; shorten it, select long mode, or explicitly pass truncate=True."
            )
        batch = self.tokenizer(
            texts,
            padding="max_length",
            max_length=info["max_length"],
            truncation=True,
            return_tensors="pt",
        )
        return self._to_device(batch), info["mode"]

    def encode_text(
        self, texts: Sequence[str] | str, *, mode: str = "auto", truncate: bool = False
    ) -> torch.Tensor:
        inputs, walk_type = self.prepare_text(texts, mode=mode, truncate=truncate)
        return self._normalize(
            self.model.get_text_features(**inputs, walk_type=walk_type)
        )

    def encode_image_preprocessed(self, image_inputs: TensorDict) -> torch.Tensor:
        return self._normalize(
            self.model.get_image_features(**self._to_device(image_inputs))
        )

    def pairwise_logits_from_features(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Scaled cosine matrix. Features may be raw or already normalized."""
        cosine = self._normalize(image_features) @ self._normalize(text_features).T
        return (
            cosine * self.model.logit_scale.float().exp()
            + self.model.logit_bias.float()
        )

    def _agreement(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> Agreement:
        cosine = self._normalize(image_features) @ self._normalize(text_features).T
        return Agreement(
            cosine,
            cosine * self.model.logit_scale.float().exp()
            + self.model.logit_bias.float(),
        )

    def agreement_preprocessed(
        self,
        image_inputs: TensorDict,
        texts: Sequence[str] | str,
        *,
        text_mode: str = "auto",
    ) -> Agreement:
        return self._agreement(
            self.encode_image_preprocessed(image_inputs),
            self.encode_text(texts, mode=text_mode),
        )

    def score_preprocessed(
        self,
        image_inputs: TensorDict,
        texts: Sequence[str] | str,
        *,
        text_mode: str = "auto",
        return_cosine: bool = False,
    ) -> torch.Tensor:
        result = self.agreement_preprocessed(image_inputs, texts, text_mode=text_mode)
        return result.cosine if return_cosine else result.logits

    def score_tensor(
        self,
        images: torch.Tensor,
        texts: Sequence[str] | str,
        *,
        max_num_patches: int = 256,
        text_mode: str = "auto",
        return_cosine: bool = False,
    ) -> torch.Tensor:
        return self.score_preprocessed(
            self.prepare_image_tensor(images, max_num_patches=max_num_patches),
            texts,
            text_mode=text_mode,
            return_cosine=return_cosine,
        )

    @torch.inference_mode()
    def agreement(
        self,
        images: Sequence[ImageLike],
        texts: Sequence[str] | str,
        *,
        text_mode: str = "auto",
        max_num_patches: int | None = None,
    ) -> Agreement:
        images = list(images)
        if not images:
            raise ValueError("Provide at least one image.")
        text_features = self.encode_text(texts, mode=text_mode)
        features = torch.cat(
            [
                self.encode_image_preprocessed(
                    self.prepare_image(im, max_num_patches=max_num_patches)
                )
                for im in images
            ]
        )
        return self._agreement(features, text_features)

    def score(
        self,
        images: Sequence[ImageLike],
        texts: Sequence[str] | str,
        *,
        text_mode: str = "auto",
        return_cosine: bool = False,
        max_num_patches: int | None = None,
    ) -> torch.Tensor:
        result = self.agreement(
            images, texts, text_mode=text_mode, max_num_patches=max_num_patches
        )
        return result.cosine if return_cosine else result.logits

    def compare(
        self,
        image_a: ImageLike,
        image_b: ImageLike,
        text: str,
        *,
        text_mode: str = "auto",
    ) -> tuple[float, float]:
        scores = self.score([image_a, image_b], [text], text_mode=text_mode)
        return float(scores[0, 0]), float(scores[1, 0])

    def attribute_margin_preprocessed(
        self,
        image_inputs: TensorDict,
        positive: str,
        negative: str,
        *,
        text_mode: str = "auto",
    ) -> torch.Tensor:
        scores = self.score_preprocessed(
            image_inputs, [positive, negative], text_mode=text_mode
        )
        return scores[:, 0] - scores[:, 1]

    def attribute_margin(
        self,
        images: Sequence[ImageLike],
        positive: str,
        negative: str,
        *,
        text_mode: str = "auto",
    ) -> torch.Tensor:
        scores = self.score(images, [positive, negative], text_mode=text_mode)
        return scores[:, 0] - scores[:, 1]

    def encode_dense_preprocessed(self, image_inputs: TensorDict) -> list[torch.Tensor]:
        """Differentiable normalized patch features [H,W,D], with padding removed."""
        inputs = self._to_device(image_inputs)
        features = self._normalize(self.model.get_image_dense_feature(**inputs))
        return [
            features[i, : h * w].reshape(h, w, -1)
            for i, (h, w) in enumerate(inputs["spatial_shapes"].tolist())
        ]

    def dense_alignment_preprocessed(
        self, image_inputs: TensorDict, texts: Sequence[str] | str
    ) -> list[torch.Tensor]:
        """One differentiable [texts,H,W] cosine map per image, box text head."""
        text_features = self.encode_text(texts, mode="box")
        return [
            (features @ text_features.T).permute(2, 0, 1)
            for features in self.encode_dense_preprocessed(image_inputs)
        ]

    @torch.inference_mode()
    def dense_alignments(
        self,
        image: ImageLike,
        texts: Sequence[str] | str,
        *,
        max_num_patches: int = 1024,
    ) -> list[DenseAlignment]:
        """Native dense maps. Larger budgets cost quadratic attention memory.

        Default 1024 is suitable for local demos; the paper's visualization uses
        16384 patches and a much larger image. No hidden per-map normalization.
        """
        pil = self._as_pil(image)
        maps = self.dense_alignment_preprocessed(
            self.prepare_image(pil, max_num_patches=max_num_patches), texts
        )[0]
        return [DenseAlignment(m.cpu(), tuple(m.shape), pil.size) for m in maps]

    def dense_alignment(
        self, image: ImageLike, text: str, *, max_num_patches: int = 1024
    ) -> DenseAlignment:
        return self.dense_alignments(image, [text], max_num_patches=max_num_patches)[0]

    def dense_difference(
        self,
        image: ImageLike,
        positive: str,
        negative: str,
        *,
        max_num_patches: int = 1024,
    ) -> DenseAlignment:
        a, b = self.dense_alignments(
            image, [positive, negative], max_num_patches=max_num_patches
        )
        return DenseAlignment(a.similarity - b.similarity, a.grid_size, a.image_size)

    def encode_regions_preprocessed(
        self,
        image_inputs: TensorDict,
        boxes: Sequence[Sequence[Sequence[float]]],
        image_sizes: Sequence[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """Native RoIAlign region features; image_sizes are (height,width).

        Boxes are absolute xyxy pixels in the original, oriented image. On MPS,
        equivalent bilinear grid sampling implements the 1x1 RoIAlign pooling
        because torchvision's native op lacks an MPS kernel. Pooling
        matches the released method: raw dense features, aligned=True, 1x1.
        """
        from torchvision.ops import roi_align

        inputs = self._to_device(image_inputs)
        if (
            len(boxes) != len(image_sizes)
            or len(boxes) != inputs["pixel_values"].shape[0]
        ):
            raise ValueError("Provide one box list and (height,width) size per image.")
        for regions, (height, width) in zip(boxes, image_sizes):
            if height <= 0 or width <= 0 or not regions:
                raise ValueError("Image dimensions and region lists must be nonempty.")
            for box in regions:
                if len(box) != 4:
                    raise ValueError("Each box must contain x1,y1,x2,y2.")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(
                        "Boxes must have positive area and lie inside the image."
                    )
        dense = self.model.get_image_dense_feature(**inputs)
        results = []
        for i, ((h, w), (height, width), regions) in enumerate(
            zip(inputs["spatial_shapes"].tolist(), image_sizes, boxes)
        ):
            fmap = (
                dense[i, : h * w]
                .reshape(h, w, -1)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            rois = torch.tensor(
                [
                    [
                        0,
                        x1 / width * w,
                        y1 / height * h,
                        x2 / width * w,
                        y2 / height * h,
                    ]
                    for x1, y1, x2, y2 in regions
                ],
                device=fmap.device,
                dtype=fmap.dtype,
            )
            if fmap.device.type == "mps":
                samples = []
                for x1, y1, x2, y2 in regions:
                    rw, rh = (x2 - x1) / width * w, (y2 - y1) / height * h
                    nx, ny = max(1, math.ceil(rw)), max(1, math.ceil(rh))
                    xs = (
                        x1 / width * w
                        - 0.5
                        + (torch.arange(nx, device=fmap.device) + 0.5) * rw / nx
                    )
                    ys = (
                        y1 / height * h
                        - 0.5
                        + (torch.arange(ny, device=fmap.device) + 0.5) * rh / ny
                    )
                    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                    grid = torch.stack(
                        ((xx + 0.5) / w * 2 - 1, (yy + 0.5) / h * 2 - 1), dim=-1
                    ).unsqueeze(0)
                    sampled = F.grid_sample(
                        fmap,
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False,
                    )
                    samples.append(sampled.mean(dim=(-2, -1), keepdim=True))
                pooled = torch.cat(samples)
            else:
                pooled = roi_align(
                    fmap,
                    rois,
                    output_size=(1, 1),
                    spatial_scale=1.0,
                    sampling_ratio=-1,
                    aligned=True,
                )
            results.append(self._normalize(pooled[:, :, 0, 0].to(self.device)))
        return results

    def region_alignment_preprocessed(
        self, image_inputs: TensorDict, boxes, image_sizes, texts: Sequence[str] | str
    ) -> list[torch.Tensor]:
        """Differentiable region/text cosine matrices, one per image."""
        text_features = self.encode_text(texts, mode="box")
        return [
            features @ text_features.T
            for features in self.encode_regions_preprocessed(
                image_inputs, boxes, image_sizes
            )
        ]

    @torch.inference_mode()
    def region_alignment(
        self,
        image: ImageLike,
        boxes: Sequence[Sequence[float]],
        texts: Sequence[str] | str,
        *,
        max_num_patches: int = 1024,
    ) -> torch.Tensor:
        pil = self._as_pil(image)
        return self.region_alignment_preprocessed(
            self.prepare_image(pil, max_num_patches=max_num_patches),
            [boxes],
            [(pil.height, pil.width)],
            texts,
        )[0].cpu()


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_ID",
    "DEFAULT_REVISION",
    "FGCLIP2",
    "Agreement",
    "DenseAlignment",
    "download_model",
]
