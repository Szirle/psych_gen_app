# server_fast.py
# A hyper-fast, GPU-locked Flask server compatible with your existing /images route + config.

import os
import io
import json
import time
import base64
import threading
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
NETWORK_PKL = os.environ.get(
    "STYLEGAN_PKL",
    "/Users/adamsobieszek/PycharmProjects/psychGAN/stylegan2-ffhq-1024x1024.pkl"
)

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

def _get_direction(dim_name: str, backend: Build_model) -> np.ndarray:
    coefs = ridge_coefs(dim_name, 100, backend=backend)
    coefs = coefs/coefs.norm()
    coefs = coefs.reshape(1, 1, -1)
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
        self.photo_to_coords, self.dim_to_photo_to_ratings = load_psychGAN_data()
        self.device = self.bm.device
        self.dtype = DTYPE


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
        strengths: List[float],
        steps: int = 40,
        latents_from: int = 0,
        latents_to: int = NUM_WS,
        truncation_psi: float = 0.5,
        noise_mode: str = "const",
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Returns (images, labels). `images` are NHWC uint8 (RGB).
        - We generate one base face per request (num_faces=1), then apply all strengths
          for the first (and only) dimension in `manipulated_dimensions`.
        - Batched synthesis: 1 forward pass for all strengths (big speedup).
        """
        if not manipulated_dimensions:
            raise ValueError("manipulated_dimensions must contain at least one name.")
        dim_name = manipulated_dimensions[0]
        n_levels = len(strengths[0])
        # direction = _get_direction(dim_name)  # [NUM_WS, 512]
        strengths = torch.tensor(strengths[0], dtype=self.dtype, device=self.device).reshape(-1, 1, 1)/10
        # n_levels = strengths.shape[0]
        direction = _get_direction(dim_name, self).repeat(n_levels, 1, 1)
        with gpu_lock, torch.inference_mode():
            # 1) Base W for one face
            w_base = self._sample_w(1, truncation_psi=truncation_psi)  # [1, NUM_WS, 512]=
            w_base = w_base.repeat(n_levels, 1, 1)
            w_base = w_base + strengths * direction
            base_img = self.bm.generate_im_from_w_space(w_base, resolution=256)

        images = base_img
        labels = {"dimension": dim_name, "strengths": strengths.tolist()}
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

    
    # Call the fast backend (batched + GPU lock)
    images, labels = backend(**config)

    # Your original code sliced images[1:], so keep that behavior:
    image_array = images

    # Allow client to pick format via ?format=png|jpg|webp (default webp)
    out_fmt = request.args.get("format", "webp")
    quality = int(request.args.get("quality", "90"))

    converted_images = [encode_image_b64(img, fmt=out_fmt, quality=quality) for img in image_array]
    return jsonify(converted_images)

if __name__ == "__main__":
    # For development: single process is fine. In production:
    #   gunicorn -w 1 -b 0.0.0.0:8000 server_fast:app
    # Keep workers=1 so the GPU lock is enough and the model loads once.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True, debug=True)