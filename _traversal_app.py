"""Temporary traversal explorer reusing the existing Flask/Flutter app contract.

Build the companion Flutter entry point with::

    flutter build web -t lib/main_traversals.dart --output build/traversals_web

Then run this module instead of ``app.py``.  The original application and its
entry point are intentionally left available and unchanged.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import app as orthogonal_app
from api_contract import (
    ApiValidationError,
    parse_image_encoding,
    parse_image_request,
    reshape_image_grid_for_api,
)


TRAVERSALS_ROOT = Path(
    os.environ.get(
        "TRAVERSALS_ROOT",
        "/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals",
    )
).resolve()
MODELS_DIR = Path(
    os.environ.get(
        "TRAVERSAL_MODELS_DIR",
        "/Users/adamsobieszek/PycharmProjects/"
        "new-StyleGAN2-1024-EarlyOutput-W-ResNet-K512-D20__20260817_220919/models",
    )
).resolve()

if str(TRAVERSALS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAVERSALS_ROOT))

# Reuse the checkpoint helpers and model class used by gen_pairs.py.
from gen_pairs import load_traversal_sets, robust_load_waves  # noqa: E402
from lib import TraversalPDE  # noqa: E402


DEVICE = orthogonal_app.DEVICE
DTYPE = torch.float32
NUM_WS = orthogonal_app.NUM_WS
TRAVERSAL_STRENGTH_SCALE = float(
    os.environ.get("TRAVERSAL_STRENGTH_SCALE", 2.0 / 25.0)
)
ROLLOUT_STEPS = int(os.environ.get("TRAVERSAL_ROLLOUT_STEPS", 9))
ROLLOUT_CHUNK_SIZE = int(os.environ.get("TRAVERSAL_ROLLOUT_CHUNK_SIZE", 4))


args, checkpoint, checkpoint_path = load_traversal_sets(str(MODELS_DIR), DEVICE)
traversals = TraversalPDE(
    num_traversal_sets=int(args.num_traversal_sets),
    num_traversal_timesteps=int(args.num_traversal_timesteps),
    traversal_vectors_dim=512,
).to(device=DEVICE, dtype=DTYPE).eval()
robust_load_waves(traversals, checkpoint)
traversals.requires_grad_(False)
NUM_TRAVERSALS = int(traversals.num_traversal_sets)
AVAILABLE_TRAVERSALS = [f"traversal-{index}" for index in range(NUM_TRAVERSALS)]
print(f"Loaded {NUM_TRAVERSALS} traversals from {checkpoint_path}")


class TraversalBackend:
    """Adapt TraversalPDE inference to the existing image-grid API."""

    def __init__(self) -> None:
        self.generator = orthogonal_app.bm
        self.base_backend = orthogonal_app.backend
        self.device = DEVICE
        self.dtype = DTYPE
        self.w_avg = self.generator.G.mapping.w_avg.to(
            device=self.device, dtype=self.dtype
        )

    def change_face(self) -> None:
        # This is the exact base-face sampler used by the original app.
        self.base_backend._set_random_base()

    def _rollout_chunk(
        self, base_w: torch.Tensor, traversal_indices: torch.Tensor, levels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = int(levels.shape[0])
        z = base_w.repeat(batch_size, 1)
        step_dt = levels * (TRAVERSAL_STRENGTH_SCALE / float(ROLLOUT_STEPS))

        for step in range(ROLLOUT_STEPS):
            paths = z[:, None, :].expand(batch_size, NUM_TRAVERSALS, -1).contiguous()
            dt = torch.zeros(
                (batch_size, NUM_TRAVERSALS), device=self.device, dtype=self.dtype
            )
            dt[:, traversal_indices] = step_dt
            t = torch.full(
                (batch_size, 1), float(step), device=self.device, dtype=self.dtype
            )
            _, all_deltas = traversals.inference(paths, t, dt=dt)
            z = (z + all_deltas[:, traversal_indices, :].sum(dim=1)).detach()
        return z

    def __call__(
        self,
        *,
        manipulated_dimensions: List[str],
        strengths: List[List[float]],
        latents_from: int = 0,
        latents_to: int = NUM_WS,
        truncation_psi: float = 0.6,
        **_: Any,
    ) -> np.ndarray:
        indices = torch.tensor(
            [int(name.rsplit("-", 1)[1]) for name in manipulated_dimensions],
            device=self.device,
            dtype=torch.long,
        )
        level_tensors = [
            torch.tensor(values, device=self.device, dtype=self.dtype)
            for values in strengths
        ]
        grids = torch.meshgrid(*level_tensors, indexing="ij")
        levels = torch.stack(grids, dim=-1).reshape(-1, len(indices))

        base_ws = self.base_backend.curr_w.to(device=self.device, dtype=self.dtype)
        base_ws = (base_ws - self.w_avg) * float(truncation_psi) + self.w_avg
        base_w = base_ws[:, 0, :]

        traversed_chunks = []
        for start in range(0, len(levels), ROLLOUT_CHUNK_SIZE):
            traversed_chunks.append(
                self._rollout_chunk(
                    base_w,
                    indices,
                    levels[start : start + ROLLOUT_CHUNK_SIZE],
                )
            )
        traversed_w = torch.cat(traversed_chunks, dim=0)

        ws = base_ws.repeat(len(levels), 1, 1)
        delta = traversed_w - base_w
        ws[:, latents_from:latents_to, :] += delta[:, None, :]
        with torch.inference_mode():
            flat_images = self.generator.generate_im_from_w_space(ws)
        return reshape_image_grid_for_api(
            flat_images, [len(values) for values in strengths]
        )


backend = TraversalBackend()


def parse_config(payload: Any):
    config, _filters = parse_image_request(
        payload, AVAILABLE_TRAVERSALS, NUM_WS
    )
    return config


static_folder = os.environ.get("TRAVERSAL_STATIC_DIR", "build/traversals_web")
app = Flask(__name__, static_folder=static_folder)
CORS(
    app,
    resources={r"/*": {"origins": os.environ.get("API_CORS_ALLOW_ORIGINS", "*").split(",")}},
    supports_credentials=True,
)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path: str):
    return send_from_directory(app.static_folder, path)


@app.route("/images", methods=["POST"])
def generate_images():
    try:
        payload = request.get_json(force=True, silent=True)
        config = parse_config(payload)
        output_format, quality = parse_image_encoding(
            request.args.get("format"), request.args.get("quality")
        )
        change_face = bool(payload.get("change_face", True))

        started = time.time()
        with orthogonal_app.gpu_lock:
            if change_face:
                backend.change_face()
            images = backend(**config)
        print(f"[PROFILE] traversal generation took {(time.time() - started) * 1000:.2f} ms")

        def to_b64(image_array: np.ndarray):
            if image_array.ndim == 4:
                return [
                    orthogonal_app.encode_image_b64(
                        image, fmt=output_format, quality=quality
                    )
                    for image in image_array
                ]
            return [to_b64(image) for image in image_array]

        return jsonify(to_b64(images))
    except ApiValidationError as error:
        return orthogonal_app._api_error(error)
    except Exception as error:
        print(f"/images failed: {error}")
        return orthogonal_app._api_error(
            RuntimeError("Traversal image generation failed."), 500
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        threaded=True,
        debug=False,
    )
