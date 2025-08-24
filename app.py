# server_fast.py
# A hyper-fast, GPU-locked Flask server compatible with your existing /images route + config.

import os
import io
import json
import time
import base64
import threading
import yaml
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from flask import Flask, send_from_directory, request, jsonify
# ---- Your fast PyTorch StyleGAN loader (from your port) ----
# Make sure this import path points to the file with Build_model you posted.
from gan_backend import Build_model, _to_nhwc_uint8
from utils import load_psychGAN_data, ridge_coefs
# os.chdir("./build")


# -----------------------------
# Performance knobs
# -----------------------------
import torch
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass

# -----------------------------
# Load GAN once into GPU/MPS/CPU
# -----------------------------

config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
NETWORK_PKL = config["stylegan_path"]
MODELS_PATH = config["models_path"]
DATA_PATH = config["data_path"]

bm = Build_model(SimpleNamespace(network_pkl=NETWORK_PKL))  # keeps G hot on device
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

def _get_direction(dim_name: str, backend: Build_model, alpha: float = 100) -> np.ndarray:
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
        self.gen = torch.Generator(device=self.bm.device)
        self.noise_mode = "const"
        self._seed_base = int(time.time())
        self.photo_to_coords, self.dim_to_photo_to_ratings = load_psychGAN_data(DATA_PATH)
        self.device = self.bm.device
        self.dtype = DTYPE
        self.curr_w = self._sample_w(1, truncation_psi=1)
        self.change_face = False
        self.w_avg = self.bm.G.mapping.w_avg


    def _sample_w(self, n: int, truncation_psi: float = 1.0) -> torch.Tensor:
        # Deterministic but fast; you can pass seed in config if desired.
        self._seed_base += 1
        self.gen.manual_seed(self._seed_base)
        z = torch.randn([n, Z_DIM], generator=self.gen, device=self.bm.device)
        # Mapping -> [n, NUM_WS, 512]
        with torch.inference_mode():
            w = self.bm.G.mapping(z, None, truncation_psi=truncation_psi)
        return w  # [n, NUM_WS, 512]
        
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
            self.curr_w = self._sample_w(1, truncation_psi=1.0)
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

        # --- sanity on W slice ---
        if not (0 <= latents_from < latents_to <= NUM_WS):
            raise ValueError(f"Bad latents_from/to: {latents_from}, {latents_to} (NUM_WS={NUM_WS})")
        L = int(latents_to - latents_from)

        # --- directions stacked on requested slice ---
        # directions_full: [K, NUM_WS, 512]  -> dir_slice: [K, L, 512]
        dir_list = []
        for name in dims:
            d = _get_direction(name, backend=self, alpha=2**((steps-30)/4))   # your new API
            if isinstance(d, np.ndarray):
                d = torch.from_numpy(d)
            d = d.to(device=device, dtype=dtype)        # ensure same device/dtype
            dir_list.append(d)
        directions_full = torch.stack(dir_list, dim=0)          # [K, NUM_WS, 512]
        dir_slice = directions_full[:, latents_from:latents_to, :]  # [K, L, 512]

        # --- build W batch and apply summed deltas (ND) ---
        with gpu_lock, torch.inference_mode():
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
        H, W = img_nhwc.shape[1], img_nhwc.shape[2]
        images = img_nhwc.reshape(*grid_shape, H, W, 3)

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
    if isinstance(conf, str):
        conf = json.loads(conf)

    # Map mode -> W-slice
    latents_from, latents_to = {"both": (0, NUM_WS), "color": (9, NUM_WS), "shape": (0, 9)}.get(
        conf.pop("mode", "both"), (0, NUM_WS)
    )

    # strengths: list per dimension (keeps ND structure)
    conf["strengths"] = [
        np.linspace(-1 * dim["strength"], dim["strength"], int(dim["n_levels"])).tolist()
        for dim in conf["manipulated_dimensions"]
    ]

    conf["manipulated_dimensions"] = [dim["name"] for dim in conf["manipulated_dimensions"]]
    conf["steps"] = int(conf.pop("max_steps", 40))
    conf["latents_from"] = latents_from
    conf["latents_to"] = latents_to
    return conf
# -----------------------------
# Flask app + static
# -----------------------------
app = Flask(__name__, static_folder="build/web")
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

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

@app.route("/images", methods=["POST"])
def generate_images():
    config = request.get_json(force=True, silent=False)
    config = parse_config(config)
    config["num_faces"] = 1
    config["change_face"] = True if backend.change_face!=config["preserve_identity"] else False
    backend.change_face = config["preserve_identity"]
    # Call the fast backend (batched + GPU lock)
    images, labels = backend(**config)

    # Your original code sliced images[1:], so keep that behavior:
    image_array = images

    # Allow client to pick format via ?format=png|jpg|webp (default webp)
    out_fmt = request.args.get("format", "webp")
    quality = int(request.args.get("quality", "90"))

    converted_images = [[encode_image_b64(_img, fmt=out_fmt, quality=quality) for _img in img] if len(img.shape)==4 else encode_image_b64(img, fmt=out_fmt, quality=quality) for img in image_array]
    return jsonify(converted_images)

if __name__ == "__main__":
    # For development: single process is fine. In production:
    #   gunicorn -w 1 -b 0.0.0.0:8000 server_fast:app
    # Keep workers=1 so the GPU lock is enough and the model loads once.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True, debug=True)