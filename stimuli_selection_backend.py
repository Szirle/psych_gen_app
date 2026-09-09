"""Sampling-only previews using the already-loaded StyleGAN preview generator.

No model loading, Flask state, manipulation directions, or current-face state.
The caller owns the shared GPU lock. Candidate sampling and rendering are kept
separate so future text ranking can render candidate batches and select nine
accepted latents without changing the transport or manipulation API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from api_contract import ApiValidationError, StimuliSelectionRequest


def stored_photo_w(coords, *, device, dtype, num_ws: int, w_dim: int) -> torch.Tensor:
    """Convert the dataset's single W vector to the generator's W+ layout."""
    value = torch.as_tensor(coords, device=device, dtype=dtype).reshape(1, 1, -1)
    if value.shape[-1] != w_dim:
        raise ApiValidationError(
            "Stored face latent has an incompatible shape.", status_code=422
        )
    return value.repeat(1, num_ws, 1)


@dataclass(frozen=True)
class SelectionCandidates:
    latents: torch.Tensor
    source: str
    eligible_count: int | None
    sampled_with_replacement: bool

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "eligible_count": self.eligible_count,
            "sampled_with_replacement": self.sampled_with_replacement,
        }


class StimuliSelectionBackend:
    def __init__(
        self, builder, photo_to_coords: Mapping, *, batch_size: int = 1, rng=None
    ):
        self.builder = builder
        self.photo_to_coords = photo_to_coords
        self.rng = rng if rng is not None else np.random.default_rng()
        # Small bounded batches leave shared MPS memory available and allow
        # cancellation between images, including future candidate-ranking loops.
        self.batch_size = max(1, int(batch_size))
        # Match EarlyOutputStyleGAN's portable Z sampling. Only the tiny random
        # vectors are created on CPU; mapping and synthesis run on the model device.
        self.generator = torch.Generator(device="cpu")

    def sample_candidates(
        self, count: int, eligible_photos: set | None
    ) -> SelectionCandidates:
        builder = self.builder
        if eligible_photos is not None:
            eligible = sorted(set(eligible_photos) & self.photo_to_coords.keys())
            if not eligible:
                raise ApiValidationError(
                    "No stored face satisfies all selected filters.", status_code=422
                )
            replace = len(eligible) < count
            ids = self.rng.choice(eligible, size=count, replace=replace).tolist()
            latents = torch.cat(
                [
                    stored_photo_w(
                        self.photo_to_coords[photo],
                        device=builder.device,
                        dtype=torch.float32,
                        num_ws=builder.num_ws,
                        w_dim=builder.G.mapping.w_avg.shape[-1],
                    )
                    for photo in ids
                ]
            )
            return SelectionCandidates(latents, "ratings", len(eligible), replace)
        self.generator.manual_seed(int(self.rng.integers(0, 2**63 - 1)))
        z = torch.randn(count, builder.z_dim, generator=self.generator).to(
            builder.device
        )
        latents = builder.G.mapping(z, None, truncation_psi=1.0)
        return SelectionCandidates(latents, "generator", None, False)

    @torch.inference_mode()
    def preview(
        self,
        request: StimuliSelectionRequest,
        eligible_photos: set | None,
        check_current: Callable[[], None],
    ) -> tuple[np.ndarray, dict]:
        check_current()
        candidates = self.sample_candidates(request.sample_count, eligible_photos)
        average = self.builder.G.mapping.w_avg
        latents = (candidates.latents - average) * request.truncation_psi + average
        images = []
        for start in range(0, len(latents), self.batch_size):
            check_current()
            images.append(
                self.builder.generate_im_from_w_space(
                    latents[start : start + self.batch_size]
                )
            )
        check_current()
        return np.concatenate(images, axis=0), candidates.metadata()
