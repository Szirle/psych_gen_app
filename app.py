# server_fast.py
# A hyper-fast, GPU-locked Flask server compatible with your existing /images route + config.

import hashlib
import os
import io
import json
import time
import base64
import threading
import yaml
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import cv2
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
# ---- Your fast PyTorch StyleGAN loader (from your port) ----
# Make sure this import path points to the file with Build_model you posted.
from gan_backend import Build_model, _to_nhwc_uint8
from early_output_backend import EarlyOutputStyleGAN
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
    """Resolve an explicit checkpoint or a configured 128/256 selection."""
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
            "stylegan_early_output_resolution must be 128, 256, or null"
        ) from error
    if resolution not in (128, 256):
        raise ValueError("stylegan_early_output_resolution must be 128 or 256")
    return config.get(
        f"stylegan_early_output_{resolution}_path",
        os.path.join(
            MODELS_PATH,
            "stylegan2-ffhq-config-f-early-output-128.pt"
            if resolution == 128
            else "stylegan2-ffhq-config-f-early-output.pt",
        ),
    )


EARLY_OUTPUT_PATH = _configured_early_output_path()
EARLY_OUTPUT_URL = os.environ.get("STYLEGAN_EARLY_OUTPUT_URL")

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
    if EARLY_OUTPUT_PATH:
        print(
            "Early-output checkpoint is unavailable; falling back to the original generator."
        )

    # Original StyleGAN2 model (allow URL override via env).
    NETWORK_PKL = config["stylegan_path"]
    STYLEGAN2_PKL_URL = os.environ.get(
        "STYLEGAN2_PKL_URL",
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl",
    )
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

class FastStyleGANBackend:
    def __init__(self, builder: Build_model):
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
        coords = self.photo_to_coords[photo_id]
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords)
        coords = coords.to(device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        if coords.shape[-1] != self.curr_w.shape[-1]:
            raise ApiValidationError(
                f"Stored latent for {photo_id} has an incompatible shape.",
                status_code=422,
            )
        self.curr_w = coords.repeat(1, NUM_WS, 1)
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

@app.route("/images", methods=["POST"])
def generate_images():
    try:
        payload = request.get_json(force=True, silent=True)
        config, filters = parse_config(payload)
        out_fmt, quality = parse_image_encoding(
            request.args.get("format"), request.args.get("quality")
        )
        change_face = bool(payload.get("change_face", True))
        eligible = _filter_photos(filters, require_coordinates=True)

        t0 = time.time()
        with gpu_lock, torch.inference_mode(), torch.no_grad():
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True, debug=True)
