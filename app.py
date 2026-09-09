# server_fast.py
# A hyper-fast, GPU-locked Flask server compatible with your existing /images route + config.

import hashlib
import os
import io
import json
import time
import base64
import queue
import gc
import threading
import yaml
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import cv2
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from early_output_backend import EarlyOutputStyleGAN
from stimuli_selection_backend import StimuliSelectionBackend, stored_photo_w
from adaptive_batching import (
    balanced_batch_size,
    calibrated_batch_capacity,
    device_memory_snapshot,
    measure_peak_memory,
    prioritize_grid_coordinates,
)
from utils import load_psychGAN_data, ridge_coefs
from api_contract import (
    ApiValidationError,
    dash_to_camel,
    eligible_photo_ids,
    normalized_histogram,
    parse_filters,
    parse_image_encoding,
    parse_image_request,
    parse_num_points,
    parse_requested_dimensions,
    parse_selection_request,
    SELECTION_GRID_SIDE,
    reshape_image_grid_for_api,
)


# -----------------------------
# Performance knobs
# -----------------------------
import torch
# Load GAN once into GPU/MPS/CPU
# -----------------------------

config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
def _download_if_missing(dest_path: str, url: Optional[str]) -> bool:
    """Download a file to dest_path if missing. Returns True if the file exists after call.
    If url is None and the file is missing, returns False.
    """
    if os.path.exists(dest_path):
        return True
    if not url:
        print(f"Missing file {dest_path} and no URL provided to download.")
        return False
    destination_directory = os.path.dirname(dest_path)
    if destination_directory:
        os.makedirs(destination_directory, exist_ok=True)
    print(f"File not found at {dest_path}. Attempting download from {url} ...")
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Download complete: {dest_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url} -> {dest_path}: {e}")
        return False


MODELS_PATH = config["models_path"]
DATA_PATH = config["data_path"]

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configured_early_output_path() -> Optional[str]:
    """Resolve the selected pointwise_style32 preview checkpoint, or disable it."""
    explicit_path = os.environ.get("STYLEGAN_EARLY_OUTPUT_PATH")
    if explicit_path:
        return explicit_path

    configured_resolution = os.environ.get(
        "STYLEGAN_EARLY_OUTPUT_RESOLUTION",
        config.get("stylegan_early_output_resolution"),
    )
    if configured_resolution in (None, "", "none", "off", False):
        return None
    try:
        resolution = int(configured_resolution)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "stylegan_early_output_resolution must be 128, 256 or null"
        ) from error
    if resolution not in (128, 256):
        raise ValueError("pointwise_style32 previews require resolution 128 or 256")
    return config.get(
        f"stylegan_early_output_{resolution}_path",
        os.path.join(MODELS_PATH, f"sg{resolution}_pointwise_style32.pt"),
    )


EARLY_OUTPUT_PATH = _configured_early_output_path()
EARLY_OUTPUT_URL = os.environ.get("STYLEGAN_EARLY_OUTPUT_URL")
NETWORK_PKL = config["stylegan_path"]
STYLEGAN2_PKL_URL = os.environ.get(
    "STYLEGAN2_PKL_URL",
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl",
)

if EARLY_OUTPUT_PATH and _download_if_missing(EARLY_OUTPUT_PATH, EARLY_OUTPUT_URL):
    bm: Any = EarlyOutputStyleGAN(
        EARLY_OUTPUT_PATH,
        device=os.environ.get("GAN_DEVICE", config.get("gan_device", "auto")),
        precision=os.environ.get(
            "GAN_PRECISION", config.get("early_output_precision", "auto")
        ),
        compile=_env_bool(
            "GAN_COMPILE", config.get("early_output_compile", False)
        ),
        compile_mode=os.environ.get(
            "GAN_COMPILE_MODE", config.get("early_output_compile_mode", "default")
        ),
        noise_mode=os.environ.get("GAN_NOISE_MODE", "const"),
    )
    print(f"Loaded early-output StyleGAN2: {bm.describe()}")
else:
    # Import the legacy generator only when its fallback is actually selected.
    from gan_backend import Build_model

    if EARLY_OUTPUT_PATH:
        print(
            "Early-output checkpoint is unavailable; falling back to the original generator."
        )

    # Original StyleGAN2 model (allow URL override via env).
    if not _download_if_missing(NETWORK_PKL, STYLEGAN2_PKL_URL):
        raise FileNotFoundError(
            f"StyleGAN model file not found and could not be downloaded: {NETWORK_PKL}"
        )

    # Distilled/tapped model (optional). Provide URL via env to fetch on-the-fly.
    STYLEGAN_DISTILLED_PATH = config.get("stylegan_distilled_path")
    STYLEGAN_TAPPED_URL = os.environ.get("STYLEGAN_TAPPED_URL")
    TORGB_HEAD_PATH = os.path.join(MODELS_PATH, "torgb_64to128_lpips.pth")
    TORGB_HEAD_URL = os.environ.get("TORGB_HEAD_URL")

    use_distilled = False
    if STYLEGAN_DISTILLED_PATH:
        have_tapped = _download_if_missing(
            STYLEGAN_DISTILLED_PATH, STYLEGAN_TAPPED_URL
        )
        if have_tapped and _download_if_missing(TORGB_HEAD_PATH, TORGB_HEAD_URL):
            use_distilled = True
        elif have_tapped:
            print(
                "Distilled model present but missing ToRGB head checkpoint. "
                "Set TORGB_HEAD_URL to enable it; falling back to the base generator."
            )

    bm = Build_model(
        SimpleNamespace(
            network_pkl=NETWORK_PKL,
            distilled_network_pkl=(
                STYLEGAN_DISTILLED_PATH if use_distilled else None
            ),
        ),
        device=os.environ.get("GAN_DEVICE", config.get("gan_device")),
    )

DEVICE = bm.device
NUM_WS = bm.num_ws
Z_DIM = bm.z_dim
DTYPE = torch.float32

# -----------------------------
# Direction provider (your edit space)
# models: Dict[str, np.ndarray] with shape [512] or [NUM_WS, 512]
# all_labels: List[str]
# -----------------------------
# Expect these to be imported/created exactly as you already do.
# Example (pseudo):
# from my_directions_loader import models, all_labels
models: Dict[str, np.ndarray] = globals().get("models", {})
all_labels: List[str] = globals().get("all_labels", list(models.keys()))
def _get_direction(dim_name: str, backend: Any, alpha: float = 100) -> np.ndarray:
    
    coefs = ridge_coefs(dim_name, alpha, backend=backend)
    coefs = coefs/coefs.norm()
    coefs = coefs.reshape(1, -1).repeat(NUM_WS, 1)
    return coefs

# -----------------------------
# Fast backend (batched, locked)
# -----------------------------
gpu_lock = threading.Lock()
_preview_revision_lock = threading.Lock()
_active_preview_revision = -1


def _payload_preview_revision(payload: Any) -> int:
    if not isinstance(payload, dict):
        raise ApiValidationError("The request body must be a JSON object.")
    revision = payload.get("preview_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ApiValidationError("preview_revision must be a non-negative integer.")
    return revision


def _activate_preview_revision(payload: Any) -> int:
    global _active_preview_revision
    revision = _payload_preview_revision(payload)
    with _preview_revision_lock:
        if revision < _active_preview_revision:
            raise ApiValidationError(
                "This preview request has already been superseded.",
                status_code=409,
            )
        _active_preview_revision = revision
    return revision


def _require_current_preview_revision(payload: Any) -> int:
    revision = _payload_preview_revision(payload)
    with _preview_revision_lock:
        if revision != _active_preview_revision:
            raise ApiValidationError(
                "This request belongs to an obsolete preview.",
                status_code=409,
            )
    return revision

class FastStyleGANBackend:
    def __init__(self, builder: Any):
        self.bm = builder
        self.seed_gen = torch.Generator(device=self.bm.device)
        self.noise_mode = "const"
        self._seed_base = int(time.time())
        self.photo_to_coords, self.dim_to_photo_to_ratings = load_psychGAN_data(DATA_PATH)
        self._average_ratings_cache: Dict[str, Dict[str, float]] = {}
        self.device = self.bm.device
        self.dtype = DTYPE
        self.curr_w = self._sample_w(1, truncation_psi=1)
        self.change_face = False

        # current face latent (in W)
        self.curr_w = self._sample_w(1, truncation_psi=1.0)           # [1, NUM_WS, 512]
        self.w_avg = self.bm.G.mapping.w_avg                          # [512]
        self.change_face = False

        # caches
        self._dir_cache: Dict[Tuple[str, int], torch.Tensor] = {}     # (dim, steps) -> [NUM_WS,512]
        self._img_cache: Dict[Tuple, np.ndarray] = {}                 # per-combo cache
        self._face_hash = self._hash_w(self.curr_w)
        self._current_photo_id: Optional[str] = None
        self._active_filter_signature: Tuple = ()

    def _hash_w(self, w: torch.Tensor) -> str:
        arr = w.detach().to("cpu", dtype=torch.float32).numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()


    def _sample_w(self, n: int, truncation_psi: float = 1.0) -> torch.Tensor:
        # Deterministic but fast; you can pass seed in config if desired.
        self._seed_base += 1
        self.seed_gen.manual_seed(self._seed_base)
        z = torch.randn([n, Z_DIM], generator=self.seed_gen, device=self.bm.device)
        # Mapping -> [n, NUM_WS, 512]
        with torch.inference_mode():
            w = self.bm.G.mapping(z, None, truncation_psi=truncation_psi)
        return w  # [n, NUM_WS, 512]

    def _set_random_base(self) -> None:
        previous_hash = self._face_hash
        for _ in range(3):
            candidate = self._sample_w(1, truncation_psi=1.0)
            candidate_hash = self._hash_w(candidate)
            if candidate_hash != previous_hash:
                self.curr_w = candidate
                self._face_hash = candidate_hash
                break
        self._current_photo_id = None

    def _set_photo_base(self, photo_id: str) -> None:
        self.curr_w = stored_photo_w(
            self.photo_to_coords[photo_id], device=self.device, dtype=self.dtype,
            num_ws=NUM_WS, w_dim=self.curr_w.shape[-1],
        )
        self._face_hash = self._hash_w(self.curr_w)
        self._current_photo_id = photo_id

    def select_base_face(
        self,
        filters: Dict[str, Tuple[float, float]],
        eligible_photos: Optional[set],
        change_face: bool,
    ) -> None:
        """Select a stable random or rating-filtered base latent."""
        signature = tuple(sorted((name, low, high) for name, (low, high) in filters.items()))
        if eligible_photos is None:
            if change_face or self._active_filter_signature:
                self._set_random_base()
            self._active_filter_signature = ()
            return

        if not eligible_photos:
            raise ApiValidationError(
                "No stored face satisfies all selected filters.",
                details={"filters": {key: list(value) for key, value in filters.items()}},
                status_code=422,
            )
        if (
            not change_face
            and self._current_photo_id in eligible_photos
        ):
            self._active_filter_signature = signature
            return

        candidates = sorted(eligible_photos)
        if change_face and self._current_photo_id in candidates and len(candidates) > 1:
            candidates.remove(self._current_photo_id)
        self._seed_base += 1
        self._set_photo_base(candidates[self._seed_base % len(candidates)])
        self._active_filter_signature = signature
        

    def _get_direction_cached(self, dim_name: str, steps: int) -> torch.Tensor:

        key = (dim_name, int(steps))
        d = self._dir_cache.get(key)
        if d is not None:
            return d
        alpha = 2 ** ((steps - 30) / 4.0)
        # new API: pass builder to _get_direction
        d = _get_direction(dim_name, backend=self, alpha=alpha)
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)
        d = d.to(device=self.device, dtype=self.dtype)  # [NUM_WS, 512]
        self._dir_cache[key] = d
        return d

        
    def __call__(
        self,
        *,
        num_faces: int = 1,
        manipulated_dimensions: List[str],
        strengths: List[List[float]],   # list of levels per dimension (ND)
        steps: int = 40,
        latents_from: int = 0,
        latents_to: int = NUM_WS,
        truncation_psi: float = 0.5,
        noise_mode: str = "const",
        strength_scale: float = 0.1,    # replaces older "/10"
        change_face: bool = False,
        **kwargs
    ):
        """
        Returns (images, labels).
        images: uint8 NHWC reshaped to (n1, ..., nK, H, W, 3)
        For each combo of strengths across K dims, we add sum_j w_j * dir_j
        to W[:, latents_from:latents_to, :], then synthesize in one batch.
        """

        if not manipulated_dimensions:
            raise ValueError("manipulated_dimensions must contain at least one name.")
        dims = manipulated_dimensions
        K = len(dims)
        if change_face:
            self._set_random_base()
        device = getattr(self, "device", self.bm.device)
        dtype  = getattr(self, "dtype", torch.float32)

        # --- per-dim levels -> tensors ---
        level_tensors = [torch.tensor(s, device=device, dtype=dtype) * float(strength_scale)
                        for s in strengths]
        grid_shape = [len(t) for t in level_tensors]   # [n1, n2, ... nK]
        # --- ND Cartesian grid of weights -> [N, K] ---
        grids = torch.meshgrid(*level_tensors, indexing="ij")
        weights = torch.stack(grids, dim=-1).reshape(-1, K).to(device=device, dtype=dtype)  # [N, K]
        N = int(weights.shape[0])
        if not (0 <= latents_from < latents_to <= NUM_WS):
            raise ValueError(f"Bad latents_from/to: {latents_from}, {latents_to} (NUM_WS={NUM_WS})")
        L = int(latents_to - latents_from)

        # --- directions stacked on requested slice ---
        # directions_full: [K, NUM_WS, 512]  -> dir_slice: [K, L, 512]
        dir_list = []
        for name in dims:
            d = self._get_direction_cached(name, steps)   # your new API
            if isinstance(d, np.ndarray):
                d = torch.from_numpy(d)
            d = d.to(device=device, dtype=dtype)        # ensure same device/dtype
            dir_list.append(d)
        directions_full = torch.stack(dir_list, dim=0)          # [K, NUM_WS, 512]
        dir_slice = directions_full[:, latents_from:latents_to, :]  # [K, L, 512]

        # --- build W batch and apply summed deltas (ND) ---
        # base W for a single face, repeated to N
        w_base = (self.curr_w-self.w_avg)*truncation_psi + self.w_avg     # [1, NUM_WS, 512]
        w_batch = w_base.repeat(N, 1, 1)                               # [N, NUM_WS, 512]

        # weights_b: [N, K, 1, 1], broadcast with dir_slice: [K, L, 512]
        # delta_slice: [N, L, 512] = sum over K of weights * dir_slice
        weights_b = weights[:, :, None, None]                           # [N, K, 1, 1]
        delta_slice = (weights_b * dir_slice[None, :, :, :]).sum(dim=1) # [N, L, 512]

        # apply to the W slice
        w_batch[:, latents_from:latents_to, :] += delta_slice           # [N, NUM_WS, 512]
        # synthesize once for whole batch; returns NHWC uint8
        img_nhwc = self.bm.generate_im_from_w_space(w_batch)            # [N, H, W, 3] uint8
        
        # --- reshape back to ND grid: (n1,...,nK,H,W,3) ---
        images = reshape_image_grid_for_api(img_nhwc, grid_shape)
        # The 2D preview API is row-major for direct UI consumption:
        # [dimension1 row][dimension0 column].  Keep the true transposed shape;
        # reshaping it back to [n0, n1] scrambles unequal grids such as 3x4.

        labels = {
            "dimensions": dims,
            "strengths": [t.detach().cpu().tolist() for t in level_tensors],
            "grid_shape": grid_shape,
            "latents_from": latents_from,
            "latents_to": latents_to,
            "truncation_psi": truncation_psi,
            "strength_scale": strength_scale,
        }
        return images, labels

backend = FastStyleGANBackend(bm)
selection_backend = StimuliSelectionBackend(bm, backend.photo_to_coords)
_full_resolution_builder: Optional[Any] = None


def _get_full_resolution_builder() -> Any:
    """Return the 1024 generator, loading it only after the first quick look."""
    global _full_resolution_builder
    if getattr(bm, "output_resolution", 1024) == 1024:
        return bm
    if _full_resolution_builder is not None:
        return _full_resolution_builder
    if not _download_if_missing(NETWORK_PKL, STYLEGAN2_PKL_URL):
        raise FileNotFoundError(
            f"Full-resolution StyleGAN model is unavailable: {NETWORK_PKL}"
        )
    from gan_backend import Build_model

    _full_resolution_builder = Build_model(
        SimpleNamespace(network_pkl=NETWORK_PKL, distilled_network_pkl=None),
        device=str(DEVICE),
    )
    return _full_resolution_builder


def _extend_style_layers(value: torch.Tensor, target_layers: int) -> torch.Tensor:
    """Match an early-output W tensor to the full generator's layer count."""
    current_layers = int(value.shape[1])
    if current_layers == target_layers:
        return value
    if current_layers > target_layers:
        return value[:, :target_layers, :]
    return torch.cat(
        [value, value[:, -1:, :].repeat(1, target_layers - current_layers, 1)],
        dim=1,
    )


def _build_full_resolution_latents(
    generation_config: Dict[str, Any], coordinates: List[Tuple[int, ...]]
) -> Tuple[Any, torch.Tensor]:
    """Build the exact full-generator W tensor for each grid coordinate."""
    full_builder = _get_full_resolution_builder()
    target_layers = int(full_builder.num_ws)
    target_device = full_builder.device
    base_w = (
        (backend.curr_w - backend.w_avg) * generation_config["truncation_psi"]
        + backend.w_avg
    )
    base_w = _extend_style_layers(base_w, target_layers).to(
        device=target_device, dtype=torch.float32
    )
    source_to = int(generation_config["latents_to"])
    target_to = target_layers if source_to == NUM_WS else min(source_to, target_layers)
    target_from = min(int(generation_config["latents_from"]), target_to)
    strength_scale = float(generation_config.get("strength_scale", 0.1))

    directions = []
    for dimension in generation_config["manipulated_dimensions"]:
        direction = backend._get_direction_cached(
            dimension, int(generation_config["steps"])
        ).unsqueeze(0)
        direction = _extend_style_layers(direction, target_layers).to(
            device=target_device, dtype=torch.float32
        )
        directions.append(direction[0, target_from:target_to, :])

    weights = torch.tensor(
        [
            [
                float(generation_config["strengths"][axis][level])
                * strength_scale
                for axis, level in enumerate(coordinate)
            ]
            for coordinate in coordinates
        ],
        device=target_device,
        dtype=torch.float32,
    )
    w = base_w.repeat(len(coordinates), 1, 1)
    w[:, target_from:target_to, :] += torch.einsum(
        "nk,kld->nld", weights, torch.stack(directions, dim=0)
    )
    latents = w.detach().to("cpu", dtype=torch.float32)
    if tuple(latents.shape[1:]) != (18, 512):
        raise RuntimeError(
            f"Full-resolution latents have shape {tuple(latents.shape[1:])}; "
            "expected (18, 512)."
        )
    return full_builder, latents


def _synthesize_full_resolution_batch(
    full_builder: Any,
    latents: torch.Tensor,
    cancellation_check: Optional[Any] = None,
) -> np.ndarray:
    images = full_builder.generate_im_from_w_space(
        latents,
        resolution=1024,
        cancellation_check=cancellation_check,
    )
    if images.shape[1:3] != (1024, 1024):
        raise RuntimeError(
            f"Full-resolution generator returned {images.shape[2]}x{images.shape[1]}."
        )
    return images


def _encoded_full_resolution_asset(
    image: np.ndarray, latent: np.ndarray, fmt: str, quality: int
) -> Dict[str, str]:
    latent_buffer = io.BytesIO()
    np.save(latent_buffer, latent.astype(np.float32, copy=False), allow_pickle=False)
    return {
        "image": encode_image_b64(image, fmt=fmt, quality=quality),
        "latent_npy": base64.b64encode(latent_buffer.getvalue()).decode("ascii"),
    }


def _full_resolution_grid_key(
    generation_config: Dict[str, Any], face_hash: str
) -> str:
    signature = json.dumps(
        {"face": face_hash, "config": generation_config},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()


def _all_grid_coordinates(
    generation_config: Dict[str, Any], origin: Tuple[int, ...]
) -> List[Tuple[int, ...]]:
    return prioritize_grid_coordinates(
        [len(levels) for levels in generation_config["strengths"]], origin
    )


@dataclass
class FullResolutionGridJob:
    cache_key: str
    preview_is_low_resolution: bool
    coordinates: List[Tuple[int, ...]] = field(default_factory=list)
    latents: Optional[torch.Tensor] = None
    coordinate_to_index: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    pending: List[Tuple[int, ...]] = field(default_factory=list)
    cached: Dict[Tuple[int, ...], Dict[str, str]] = field(default_factory=dict)
    events: List[Tuple[int, Tuple[int, ...], Dict[str, str]]] = field(
        default_factory=list
    )
    condition: threading.Condition = field(default_factory=threading.Condition)
    initializing: bool = True
    running: bool = False
    complete: bool = False
    error: Optional[str] = None
    next_version: int = 1
    image_format: str = "webp"
    image_quality: int = 95
    calibrated_peak_bytes_per_image: int = 0
    oom_capacity_ceiling: Optional[int] = None
    enqueued: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    terminated: threading.Event = field(default_factory=threading.Event)
    cancellation_reason: Optional[str] = None

    def store(self, coordinate: Tuple[int, ...], asset: Dict[str, str]) -> None:
        with self.condition:
            if self.cancelled.is_set():
                return
            if coordinate not in self.cached:
                self.cached[coordinate] = asset
                self.events.append((self.next_version, coordinate, asset))
                self.next_version += 1
            self.condition.notify_all()

    def promote(self, coordinate: Tuple[int, ...]) -> None:
        with self.condition:
            if coordinate in self.pending:
                self.pending.remove(coordinate)
                self.pending.insert(0, coordinate)
            self.condition.notify_all()

    def updates_after(self, cursor: int, wait_seconds: float = 0) -> Dict[str, Any]:
        """Wait for progress and copy at most two assets out of the shared state."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.next_version - 1 > cursor or self.complete,
                timeout=wait_seconds,
            )
            events = []
            for event in self.events:
                if event[0] > cursor:
                    events.append(event)
                    if len(events) == 2:
                        break
            response_cursor = events[-1][0] if events else cursor
            return {
                "items": [
                    _asset_response(self, coordinate, asset)
                    for _version, coordinate, asset in events
                ],
                "cursor": response_cursor,
                # A completed producer may still have several response pages.
                "complete": self.complete and response_cursor >= self.next_version - 1,
                "error": self.error,
            }

    def cancel(self, reason: str) -> None:
        self.cancellation_reason = reason
        self.cancelled.set()
        with self.condition:
            self.pending.clear()
            self.initializing = False
            self.complete = True
            self.condition.notify_all()

    def raise_if_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise FullResolutionCancelled(
                self.cancellation_reason or "Full-resolution generation cancelled."
            )


class FullResolutionCancelled(RuntimeError):
    pass


_full_grid_jobs: Dict[str, FullResolutionGridJob] = {}
_full_grid_jobs_lock = threading.Lock()
_full_grid_queue: "queue.Queue[Tuple[FullResolutionGridJob, Any]]" = queue.Queue()
_terminating_full_grid_jobs: List[FullResolutionGridJob] = []


def _cancel_active_full_resolution_jobs(
    reason: str,
) -> List[FullResolutionGridJob]:
    with _full_grid_jobs_lock:
        cancelled = [
            (cache_key, job)
            for cache_key, job in _full_grid_jobs.items()
            if not job.complete
        ]
        for cache_key, _job in cancelled:
            _full_grid_jobs.pop(cache_key, None)
        for _cache_key, job in cancelled:
            if all(existing is not job for existing in _terminating_full_grid_jobs):
                _terminating_full_grid_jobs.append(job)
    for _cache_key, job in cancelled:
        job.cancel(reason)
    if cancelled:
        print(f"[FULL-RES] cancelled {len(cancelled)} stale grid job(s): {reason}")
    return [job for _cache_key, job in cancelled]


def _wait_for_cancelled_full_resolution_jobs() -> None:
    """Do not admit preview inference until cancelled GPU work has exited."""
    while True:
        with _full_grid_jobs_lock:
            _terminating_full_grid_jobs[:] = [
                job
                for job in _terminating_full_grid_jobs
                if not job.terminated.is_set()
            ]
            unfinished = list(_terminating_full_grid_jobs)
        if not unfinished:
            return
        for job in unfinished:
            job.terminated.wait()


def _prune_full_grid_jobs() -> None:
    if len(_full_grid_jobs) <= 4:
        return
    removable = [key for key, job in _full_grid_jobs.items() if job.complete]
    while len(_full_grid_jobs) > 4 and removable:
        _full_grid_jobs.pop(removable.pop(0), None)


def _is_oom(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass


def _release_full_resolution_builder() -> None:
    """Offload the separate 1024 model before a new preview inference."""
    global _full_resolution_builder
    full_builder = _full_resolution_builder
    if full_builder is None or full_builder is bm:
        return
    _full_resolution_builder = None
    del full_builder
    gc.collect()
    _clear_accelerator_cache(torch.device(DEVICE))


def _run_full_resolution_grid_job(job: FullResolutionGridJob, builder: Any) -> None:
    try:
        while True:
            job.raise_if_cancelled()
            retry = False
            batch_coordinates = []
            try:
                with gpu_lock, torch.inference_mode():
                    job.raise_if_cancelled()
                    with job.condition:
                        if not job.pending:
                            job.running = False
                            job.enqueued = False
                            job.complete = True
                            job.condition.notify_all()
                            return
                        snapshot = device_memory_snapshot(builder.device)
                        capacity = calibrated_batch_capacity(
                            snapshot,
                            measured_peak_bytes_per_image=(
                                job.calibrated_peak_bytes_per_image
                            ),
                            remaining=len(job.pending),
                        )
                        if job.oom_capacity_ceiling is not None:
                            capacity = min(capacity, job.oom_capacity_ceiling)
                        batch_size = balanced_batch_size(len(job.pending), capacity)
                        planned_batches = (
                            len(job.pending) + batch_size - 1
                        ) // batch_size
                        print(
                            "[FULL-RES] plan: "
                            f"{planned_batches} balanced batch(es), next batch "
                            f"{batch_size}/{len(job.pending)} pending, "
                            f"capacity {capacity} with a 2/3 memory budget "
                            f"on {snapshot.device_type}; "
                            f"{snapshot.available_bytes / 1024**3:.2f} GiB available, "
                            f"{job.calibrated_peak_bytes_per_image / 1024**3:.2f} "
                            "GiB measured peak/image"
                        )
                        batch_coordinates = job.pending[:batch_size]
                        del job.pending[:batch_size]
                    indices = [
                        job.coordinate_to_index[item] for item in batch_coordinates
                    ]
                    batch_latents = job.latents[indices]
                    inference_started = time.perf_counter()
                    images, batch_measurement = measure_peak_memory(
                        builder.device,
                        lambda: _synthesize_full_resolution_batch(
                            builder,
                            batch_latents,
                            job.raise_if_cancelled,
                        ),
                    )
                    inference_seconds = time.perf_counter() - inference_started
                job.raise_if_cancelled()
                measured_per_image = (
                    batch_measurement.peak_increment_bytes
                    + len(batch_coordinates)
                    - 1
                ) // len(batch_coordinates)
                if measured_per_image > job.calibrated_peak_bytes_per_image:
                    job.calibrated_peak_bytes_per_image = measured_per_image
                    print(
                        "[FULL-RES] raised measured peak/image to "
                        f"{measured_per_image / 1024**3:.2f} GiB from the "
                        f"batch={len(batch_coordinates)} observation"
                    )
            except Exception as error:
                if _is_oom(error) and len(batch_coordinates) > 1:
                    job.oom_capacity_ceiling = max(
                        1, len(batch_coordinates) // 2
                    )
                    retry = True
                else:
                    raise

            if retry:
                # The exception/traceback must leave scope before clearing the
                # allocator, otherwise failed activations are still referenced.
                with gpu_lock:
                    _clear_accelerator_cache(builder.device)
                with job.condition:
                    job.raise_if_cancelled()
                    job.pending = batch_coordinates + job.pending
                print(
                    "[FULL-RES] retrying with capacity "
                    f"{job.oom_capacity_ceiling} after OOM"
                )
                continue

            encoding_started = time.perf_counter()
            for coordinate, image, latent in zip(
                batch_coordinates, images, batch_latents.numpy()
            ):
                job.raise_if_cancelled()
                job.store(
                    coordinate,
                    _encoded_full_resolution_asset(
                        image, latent, job.image_format, job.image_quality
                    ),
                )
            encoding_seconds = time.perf_counter() - encoding_started
            driver_growth = max(
                0, batch_measurement.peak_bytes - batch_measurement.baseline_bytes
            )
            print(
                f"[FULL-RES] batch={len(batch_coordinates)} finished: "
                f"generation+readback {inference_seconds:.2f}s, "
                f"encoding {encoding_seconds:.2f}s; "
                f"allocator growth {driver_growth / 1024**3:.2f} GiB, "
                "MPS live tensor peak increment "
                f"{batch_measurement.tensor_peak_increment_bytes / 1024**3:.2f} GiB"
            )
            # Do not retain the previous raw image batch during the next inference.
            del images, image, latent, batch_latents
    except FullResolutionCancelled:
        with job.condition:
            job.running = False
            job.enqueued = False
            job.complete = True
            job.condition.notify_all()
    except Exception as error:
        with job.condition:
            job.running = False
            job.enqueued = False
            job.complete = True
            job.error = str(error)
            job.condition.notify_all()
        print(f"Full-resolution background generation failed: {error}")


def _full_resolution_worker_loop() -> None:
    while True:
        job, builder = _full_grid_queue.get()
        terminated = job.terminated
        try:
            _run_full_resolution_grid_job(job, builder)
        finally:
            # Return unused memory to other applications once the job ends,
            # while allowing allocator reuse between its successful batches.
            try:
                with gpu_lock:
                    _clear_accelerator_cache(builder.device)
            except Exception as error:
                print(f"[FULL-RES] final cache cleanup failed: {error}")
            finally:
                _full_grid_queue.task_done()
                job = None
                builder = None
                terminated.set()


def _enqueue_full_resolution_job(job: FullResolutionGridJob, builder: Any) -> None:
    with job.condition:
        if job.enqueued or job.running or job.complete or not job.pending:
            return
        job.enqueued = True
        job.running = True
    _full_grid_queue.put((job, builder))


threading.Thread(
    target=_full_resolution_worker_loop,
    name="full-resolution-worker",
    daemon=True,
).start()


def _asset_response(
    job: FullResolutionGridJob,
    coordinate: Tuple[int, ...],
    asset: Dict[str, str],
) -> Dict[str, Any]:
    return {
        **asset,
        "selected_levels": list(coordinate),
        "cache_key": job.cache_key,
        "preview_is_low_resolution": job.preview_is_low_resolution,
        "latent_shape": [18, 512],
        "resolution": 1024,
    }
# -----------------------------
# Config parsing (kept compatible)
# -----------------------------
def parse_config(conf):
    """Parse the public preview request into backend kwargs and canonical filters."""
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except json.JSONDecodeError as error:
            raise ApiValidationError("The request body contains invalid JSON.") from error
    available = list(backend.dim_to_photo_to_ratings.keys())
    return parse_image_request(conf, available, NUM_WS)
# -----------------------------
# Flask app + static
# -----------------------------
app = Flask(__name__, static_folder="build/web")
cors_origins = os.environ.get("API_CORS_ALLOW_ORIGINS", "*")
CORS(
    app,
    resources={r"/*": {"origins": cors_origins.split(",")}},
    supports_credentials=True,
)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/charts")
def plotly_charts_iframe():
    # Self-contained HTML page rendering a 2x10 grid of mocked Plotly graphs
    html = """
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Charts</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
      html, body { height: 100%; margin: 0; }
      #container {
        height: 100%;
        width: 100%;
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        grid-auto-rows: 180px;
        gap: 8px;
        padding: 8px;
        box-sizing: border-box;
        background: #ffffff;
      }
      .chart { width: 100%; height: 100%; }
    </style>
  </head>
  <body>
    <div id=\"container\"></div>
    <script>
      function makeData(i) {
        const x = Array.from({ length: 30 }, (_, k) => k);
        const y = x.map(k => Math.sin(k / 3 + i / 4) + (Math.random() - 0.5) * 0.3);
        return [{ x, y, mode: 'lines', line: { color: '#2B3A55', width: 2 } }];
      }
      const layout = { margin: { l: 20, r: 10, t: 10, b: 20 }, showlegend: false };
      const container = document.getElementById('container');
      for (let i = 0; i < 20; i++) {
        const div = document.createElement('div');
        div.className = 'chart';
        div.id = 'chart-' + i;
        container.appendChild(div);
        Plotly.newPlot(div, makeData(i), layout, { displaylogo: false, responsive: true });
      }
      window.addEventListener('resize', () => {
        for (let i = 0; i < 20; i++) {
          const el = document.getElementById('chart-' + i);
          if (el) { Plotly.Plots.resize(el); }
        }
      });
    </script>
  </body>
</html>
    """
    return html

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)



# -----------------------------
# Fast encoder (WEBP default)
# -----------------------------
def encode_image_b64(img: np.ndarray, fmt: str = "webp", quality: int = 90) -> str:
    # img: HWC uint8 RGB
    if fmt.lower() == "png":
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif fmt.lower() == "jpg" or fmt.lower() == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        # webp is usually fastest/smallest
        ok, buf = cv2.imencode(".webp", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    if not ok:
        raise RuntimeError("Image encode failed")
    return base64.b64encode(buf).decode("ascii")


def _api_error(error: Exception, status_code: int = 400):
    details = getattr(error, "details", None)
    body = {"error": str(error)}
    if details is not None:
        body["details"] = details
    return jsonify(body), getattr(error, "status_code", status_code)


@app.route("/images/full/cancel", methods=["POST"])
def cancel_full_resolution_generation():
    """Immediately invalidate queued and active work for obsolete previews."""
    try:
        payload = request.get_json(force=True, silent=True)
        revision = _activate_preview_revision(payload)
        cancelled_jobs = _cancel_active_full_resolution_jobs(
            "The client requested a new preview grid."
        )
        return jsonify(
            {
                "cancelled_jobs": len(cancelled_jobs),
                "preview_revision": revision,
            }
        )
    except ApiValidationError as error:
        return _api_error(error)

@app.route("/images", methods=["POST"])
def generate_images():
    try:
        payload = request.get_json(force=True, silent=True)
        _activate_preview_revision(payload)
        _cancel_active_full_resolution_jobs(
            "A new preview-grid request superseded the old settings."
        )
        _wait_for_cancelled_full_resolution_jobs()
        config, filters = parse_config(payload)
        out_fmt, quality = parse_image_encoding(
            request.args.get("format"), request.args.get("quality")
        )
        change_face = bool(payload.get("change_face", True))
        eligible = _filter_photos(filters, require_coordinates=True)

        t0 = time.time()
        with gpu_lock, torch.inference_mode(), torch.no_grad():
            _require_current_preview_revision(payload)
            _release_full_resolution_builder()
            backend.select_base_face(filters, eligible, change_face)
            images, _labels = backend(**config)
        t1 = time.time()
        print(f"[PROFILE] backend(**config) took {(t1 - t0)*1000:.2f} ms")

        def to_b64(image_array):
            if len(image_array.shape) == 4:
                return [
                    encode_image_b64(image, fmt=out_fmt, quality=quality)
                    for image in image_array
                ]
            return [to_b64(image) for image in image_array]

        return jsonify(to_b64(images))
    except ApiValidationError as error:
        return _api_error(error)
    except Exception as error:
        print(f"/images failed: {error}")
        return _api_error(RuntimeError("Image generation failed."), 500)


@app.route("/stimuli/preview", methods=["POST"])
def generate_selection_preview():
    """Return nine unmanipulated draws without mutating backend.curr_w."""
    try:
        payload = request.get_json(force=True, silent=True)
        selection = parse_selection_request(
            payload, backend.dim_to_photo_to_ratings.keys()
        )
        out_fmt, quality = parse_image_encoding(
            request.args.get("format"), request.args.get("quality")
        )
        eligible = _filter_photos(selection.filters, require_coordinates=True)
        _activate_preview_revision(payload)
        _cancel_active_full_resolution_jobs(
            "A stimuli-selection preview superseded the old preview."
        )
        _wait_for_cancelled_full_resolution_jobs()
        with gpu_lock, torch.inference_mode():
            _require_current_preview_revision(payload)
            _release_full_resolution_builder()
            images, sampling = selection_backend.preview(
                selection,
                eligible,
                lambda: _require_current_preview_revision(payload),
            )
        encoded = [
            encode_image_b64(image, fmt=out_fmt, quality=quality) for image in images
        ]
        _require_current_preview_revision(payload)
        return jsonify(
            {
                "preview_revision": selection.preview_revision,
                "images": [
                    encoded[i : i + SELECTION_GRID_SIDE]
                    for i in range(0, len(encoded), SELECTION_GRID_SIDE)
                ],
                "sampling": sampling,
            }
        )
    except ApiValidationError as error:
        return _api_error(error)
    except Exception as error:
        print(f"/stimuli/preview failed: {error}")
        return _api_error(RuntimeError("Stimuli preview generation failed."), 500)


@app.route("/images/full", methods=["POST"])
def generate_full_resolution_image():
    """Return one full asset, then keep upgrading the nearby grid in batches."""
    try:
        payload = request.get_json(force=True, silent=True)
        _require_current_preview_revision(payload)
        generation_config, _filters = parse_config(payload)
        selected_levels = payload.get("selected_levels")
        expected_count = len(generation_config["strengths"])
        if (
            not isinstance(selected_levels, list)
            or len(selected_levels) != expected_count
            or any(
                isinstance(level, bool) or not isinstance(level, int)
                for level in selected_levels
            )
        ):
            raise ApiValidationError(
                "selected_levels must contain one integer index per dimension."
            )
        for index, (selected_level, levels) in enumerate(
            zip(selected_levels, generation_config["strengths"])
        ):
            if selected_level < 0 or selected_level >= len(levels):
                raise ApiValidationError(
                    f"selected_levels[{index}] is outside the requested grid."
                )

        out_fmt, quality = parse_image_encoding(
            request.args.get("format", "webp"), request.args.get("quality", "95")
        )
        coordinate = tuple(selected_levels)
        preview_is_low_resolution = getattr(bm, "output_resolution", 1024) < 1024
        cache_key = _full_resolution_grid_key(
            generation_config, backend._face_hash
        )
        with _full_grid_jobs_lock:
            job = _full_grid_jobs.get(cache_key)
            owner = job is None
            if owner:
                job = FullResolutionGridJob(
                    cache_key=cache_key,
                    preview_is_low_resolution=preview_is_low_resolution,
                    image_format=out_fmt,
                    image_quality=quality,
                )
                _full_grid_jobs[cache_key] = job
                _prune_full_grid_jobs()

        if not owner:
            if not job.preview_is_low_resolution:
                with job.condition:
                    while job.initializing and job.error is None:
                        job.condition.wait()
                    job.raise_if_cancelled()
                    if coordinate in job.cached:
                        return jsonify(
                            _asset_response(job, coordinate, job.cached[coordinate])
                        )
                    if job.error is not None:
                        raise RuntimeError(job.error)
                preview_image = payload.get("preview_image")
                if not isinstance(preview_image, str):
                    raise ApiValidationError("preview_image is required.")
                raw_preview = base64.b64decode(preview_image, validate=True)
                decoded = cv2.imdecode(
                    np.frombuffer(raw_preview, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None or decoded.shape[:2] != (1024, 1024):
                    raise ApiValidationError(
                        "preview_image must contain a 1024x1024 image."
                    )
                index = job.coordinate_to_index[coordinate]
                asset = _encoded_full_resolution_asset(
                    cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
                    job.latents[index].numpy(),
                    out_fmt,
                    quality,
                )
                job.store(coordinate, asset)
                return jsonify(_asset_response(job, coordinate, asset))
            job.promote(coordinate)
            with job.condition:
                while (
                    coordinate not in job.cached
                    and job.error is None
                    and not job.complete
                ):
                    job.condition.wait()
                if coordinate in job.cached:
                    return jsonify(
                        _asset_response(job, coordinate, job.cached[coordinate])
                    )
                job.raise_if_cancelled()
                raise RuntimeError(job.error or "Full-resolution grid is incomplete.")

        t0 = time.time()
        coordinates = _all_grid_coordinates(generation_config, coordinate)
        with gpu_lock, torch.inference_mode(), torch.no_grad():
            job.raise_if_cancelled()
            full_builder, latents = _build_full_resolution_latents(
                generation_config, coordinates
            )
            if preview_is_low_resolution:
                image_batch, calibration = measure_peak_memory(
                    full_builder.device,
                    lambda: _synthesize_full_resolution_batch(
                        full_builder,
                        latents[0:1],
                        job.raise_if_cancelled,
                    ),
                )
                image = image_batch[0]
                job.calibrated_peak_bytes_per_image = (
                    calibration.peak_increment_bytes
                )
                print(
                    "[FULL-RES] batch=1 calibration measured "
                    f"{calibration.peak_increment_bytes / 1024**3:.2f} GiB "
                    f"peak incremental memory ({calibration.source})"
                )
            else:
                preview_image = payload.get("preview_image")
                if not isinstance(preview_image, str):
                    raise ApiValidationError(
                        "preview_image is required when the preview model is full resolution."
                    )
                raw_preview = base64.b64decode(preview_image, validate=True)
                decoded = cv2.imdecode(
                    np.frombuffer(raw_preview, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None or decoded.shape[:2] != (1024, 1024):
                    raise ApiValidationError(
                        "preview_image must contain a 1024x1024 image."
                    )
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        job.raise_if_cancelled()
        print(
            f"[PROFILE] full-resolution quick look took {(time.time() - t0)*1000:.2f} ms"
        )
        job.coordinates = coordinates
        job.latents = latents
        job.coordinate_to_index = {
            item: index for index, item in enumerate(coordinates)
        }
        job.raise_if_cancelled()
        asset = _encoded_full_resolution_asset(
            image, latents[0].numpy(), out_fmt, quality
        )
        job.raise_if_cancelled()
        job.store(coordinate, asset)
        with job.condition:
            job.raise_if_cancelled()
            job.initializing = False
            if preview_is_low_resolution:
                job.pending = coordinates[1:]
                job.complete = not job.pending
            else:
                job.complete = True
            job.condition.notify_all()
        job.raise_if_cancelled()
        _enqueue_full_resolution_job(job, full_builder)
        if not job.enqueued:
            job.terminated.set()
        return jsonify(_asset_response(job, coordinate, asset))
    except FullResolutionCancelled as error:
        if "full_builder" in locals():
            full_builder = None
        if "job" in locals() and job is not None:
            with job.condition:
                job.initializing = False
                job.running = False
                job.enqueued = False
                job.complete = True
                job.condition.notify_all()
            if not job.enqueued:
                job.terminated.set()
        return jsonify({"cancelled": True, "reason": str(error)}), 409
    except ApiValidationError as error:
        return _api_error(error)
    except Exception as error:
        try:
            if "full_builder" in locals():
                full_builder = None
            if "job" in locals() and job is not None:
                with job.condition:
                    job.initializing = False
                    job.complete = True
                    job.error = str(error)
                    job.condition.notify_all()
                if not job.enqueued:
                    job.terminated.set()
        except Exception:
            pass
        print(f"/images/full failed: {error}")
        return _api_error(RuntimeError("Full-resolution image generation failed."), 500)


@app.route("/images/full/status", methods=["POST"])
def full_resolution_grid_status():
    """Return cached grid upgrades completed after the supplied cursor."""
    try:
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("cache_key"), str
        ):
            raise ApiValidationError("cache_key is required.")
        cursor = payload.get("after", 0)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ApiValidationError("after must be a non-negative integer.")
        wait_ms = payload.get("wait_ms", 0)
        if (
            isinstance(wait_ms, bool)
            or not isinstance(wait_ms, int)
            or not 0 <= wait_ms <= 10000
        ):
            raise ApiValidationError("wait_ms must be an integer between 0 and 10000.")
        with _full_grid_jobs_lock:
            job = _full_grid_jobs.get(payload["cache_key"])
        if job is None:
            return jsonify({"items": [], "cursor": cursor, "complete": True})
        # Serialization can copy megabytes; never hold the worker's condition
        # while encoding JSON or waiting for a client to consume the response.
        return jsonify(job.updates_after(cursor, wait_ms / 1000))
    except ApiValidationError as error:
        return _api_error(error)


@app.route("/images/full/prioritize", methods=["POST"])
def prioritize_full_resolution_coordinate():
    """Move a scheduled coordinate forward without starting generation."""
    try:
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("cache_key"), str
        ):
            raise ApiValidationError("cache_key is required.")
        raw_levels = payload.get("selected_levels")
        if not isinstance(raw_levels, list) or any(
            isinstance(level, bool) or not isinstance(level, int)
            for level in raw_levels
        ):
            raise ApiValidationError("selected_levels must contain integers.")
        coordinate = tuple(raw_levels)
        with _full_grid_jobs_lock:
            job = _full_grid_jobs.get(payload["cache_key"])
        if job is None:
            return jsonify({"found": False, "cached": False, "complete": True})
        if coordinate not in job.coordinate_to_index:
            raise ApiValidationError("selected_levels is outside this grid.")
        job.promote(coordinate)
        with job.condition:
            return jsonify(
                {
                    "found": True,
                    "cached": coordinate in job.cached,
                    "complete": job.complete,
                }
            )
    except ApiValidationError as error:
        return _api_error(error)


# -----------------------------
# Distributions endpoint
# -----------------------------
def _avg_ratings_for_dim(dim_name: str) -> Dict[str, float]:
    """Return mapping photo -> average rating in [0,1] for a given dimension."""
    cached = backend._average_ratings_cache.get(dim_name)
    if cached is not None:
        return cached
    d = backend.dim_to_photo_to_ratings.get(dim_name, {})
    out: Dict[str, float] = {}
    for photo, ratings in d.items():
        try:
            if ratings is None:
                continue
            if isinstance(ratings, (list, tuple)) and len(ratings) > 0:
                # ensure numeric and in [0,1]
                vals = [float(x) for x in ratings if x is not None]
                if len(vals) == 0:
                    continue
                avg = float(np.nanmean(vals))
                if np.isnan(avg):
                    continue
                out[photo] = float(np.clip(avg, 0.0, 1.0))
        except Exception:
            continue
    backend._average_ratings_cache[dim_name] = out
    return out


def _filter_photos(
    filters: Dict[str, Tuple[float, float]], require_coordinates: bool = False
) -> Optional[set]:
    """Return None for no filters or the exact filtered set, including empty."""
    averages = {dimension: _avg_ratings_for_dim(dimension) for dimension in filters}
    coordinate_ids = backend.photo_to_coords.keys() if require_coordinates else None
    return eligible_photo_ids(filters, averages, coordinate_ids)


def _hist_for_dim(
    dim_name: str, photos_subset: Optional[set], num_points: int = 100
) -> list:
    """Compute histogram over [0,1] for dim on photo subset and normalize max to 1."""
    avg_map = _avg_ratings_for_dim(dim_name)
    if photos_subset is not None:
        values = [avg_map[p] for p in photos_subset if p in avg_map]
    else:
        values = list(avg_map.values())
    return normalized_histogram(values, num_points)


@app.route("/distributions", methods=["POST"])
def distributions_endpoint():
    """
    Request JSON:
      {
        "filters": { "attractive": [0.2, 0.8], ... },
        "num_points": 100
      }
    Response JSON:
      {
        "distributions": { "attractive": [..], "dominant": [..], ... }
      }
    """
    try:
        available = list(backend.dim_to_photo_to_ratings.keys())
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            raise ApiValidationError("The request body must be a JSON object.")
        if not isinstance(payload, dict):
            raise ApiValidationError("The request body must be a JSON object.")
        filters = parse_filters(payload.get("filters"), available)
        requested = parse_requested_dimensions(payload.get("variables"), available)
        num_points = parse_num_points(payload.get("num_points", 100))
        photos_subset = _filter_photos(filters)
        result = {
            dash_to_camel(dim): _hist_for_dim(dim, photos_subset, num_points)
            for dim in requested
        }
        return jsonify({"distributions": result})
    except ApiValidationError as error:
        return _api_error(error)
    except Exception as error:
        print(f"/distributions failed: {error}")
        return _api_error(RuntimeError("Distribution calculation failed."), 500)

if __name__ == "__main__":
    # For development: single process is fine. In production:
    #   gunicorn -w 1 -b 0.0.0.0:8000 server_fast:app
    # Keep workers=1 so the GPU lock is enough and the model loads once.
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        threaded=True,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        # The reloader imports this module in a second process, duplicating the
        # model and the process-wide full-resolution queue in memory.
        use_reloader=False,
    )
