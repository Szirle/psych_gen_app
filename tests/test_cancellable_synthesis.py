import pytest
import torch

from gan_backend import _synthesis


class _FakeBlock:
    num_conv = 1
    num_torgb = 1

    def __init__(self, resolution, calls):
        self.resolution = resolution
        self.calls = calls

    def __call__(self, _x, _image, _ws, **_kwargs):
        self.calls.append(self.resolution)
        return torch.tensor(1), torch.tensor(1)


def test_full_resolution_synthesis_stops_between_resolution_blocks():
    calls = []

    class Synthesis:
        block_resolutions = [4, 8]
        b4 = _FakeBlock(4, calls)
        b8 = _FakeBlock(8, calls)

    class Generator:
        synthesis = Synthesis()

    class Cancelled(RuntimeError):
        pass

    checks = 0

    def cancel_after_first_block():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        _synthesis(
            Generator(),
            torch.zeros(1, 3, 512),
            cancellation_check=cancel_after_first_block,
        )

    assert calls == [4]


def test_mps_blocks_finish_before_next_cancellation_check(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    calls = []

    class Synthesis:
        block_resolutions = [4, 8]
        b4 = _FakeBlock(4, calls)
        b8 = _FakeBlock(8, calls)

    class Generator:
        synthesis = Synthesis()

    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("drain"))
    _synthesis(
        Generator(), torch.zeros(1, 3, 512, device="mps"),
        cancellation_check=lambda: calls.append("check"),
    )
    assert calls == ["check", 4, "drain", "check", "check", 8, "drain", "check"]
