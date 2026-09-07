from pathlib import Path

import numpy as np
import pytest
import torch

from early_output_backend import EarlyOutputStyleGAN


@pytest.fixture(scope="module")
def preview():
    if not torch.backends.mps.is_available():
        pytest.skip("Local preview checks use MPS")
    path = Path(__file__).resolve().parents[1] / "models/sg128_pointwise_style32.pt"
    if not path.exists():
        pytest.skip("Install the trained preview checkpoint")
    return EarlyOutputStyleGAN(path, device="mps", precision="fp32")


def test_all_four_styles_match_generator(preview):
    rng = torch.Generator().manual_seed(31)
    base = preview.map_z(torch.randn(1, 512, generator=rng))
    delta = preview.map_z(torch.randn(1, 512, generator=rng))
    layers = torch.arange(12, device="mps")
    for mask in (torch.zeros_like(layers), torch.ones_like(layers), layers < 8, layers % 2):
        ws = base + delta * mask[None, :, None]
        with torch.inference_mode():
            expected = preview.generator.synthesis(
                ws, noise_mode="const", force_fp32=True, fused_modconv=False)
        torch.testing.assert_close(preview.synthesize(ws), expected, rtol=2e-4, atol=2e-4)


def test_seeded_preview_and_w_api(preview):
    first = preview.generate_im_from_random_seed(22)
    second = preview.generate_im_from_random_seed(22)
    assert first.shape == (1, 128, 128, 3) and first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    w = preview.map_z(torch.zeros(1, 512))[:, 0]
    np.testing.assert_array_equal(
        preview.generate_im_from_w_space(w),
        preview.generate_im_from_w_space(w[:, None].repeat(1, 12, 1)))
    with pytest.raises(ValueError, match="Expected W tensor"):
        preview.synthesize(torch.zeros(1, 12, 511))


def test_compile_failure_falls_back_without_losing_style_inputs(preview):
    ws = preview.map_z(torch.zeros(1, 512))
    expected = preview.synthesize(ws)

    def failed_compile(_ws):
        raise RuntimeError("test compiler failure")

    preview._compiled_runtime = failed_compile
    preview.compile_enabled = True
    with pytest.warns(RuntimeWarning, match="falling back to eager"):
        actual = preview.synthesize(ws)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert preview._compiled_runtime is None and not preview.compile_enabled
