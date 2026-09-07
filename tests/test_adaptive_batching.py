import threading
from types import SimpleNamespace

import pytest

import adaptive_batching
from adaptive_batching import (
    DeviceMemorySnapshot,
    balanced_batch_size,
    calibrated_batch_capacity,
    prioritize_grid_coordinates,
)


def test_peak_measurement_samples_the_calibration_operation(monkeypatch):
    state = {"high": False}
    sampled = threading.Event()

    def allocated(_device):
        if state["high"]:
            sampled.set()
            return 100
        return 0

    monkeypatch.setattr(adaptive_batching, "_allocated_bytes", allocated)
    monkeypatch.setattr(adaptive_batching, "_synchronize", lambda _device: None)

    def operation():
        state["high"] = True
        assert sampled.wait(timeout=1)
        state["high"] = False
        return "image"

    result, measurement = adaptive_batching.measure_peak_memory(
        "cpu", operation, sample_interval_seconds=0.0001
    )
    assert result == "image"
    assert measurement.peak_increment_bytes == 100


@pytest.mark.parametrize("system_available, expected", [(3, 3), (20, 10), (0, 0)])
def test_mps_headroom_respects_other_applications(monkeypatch, system_available, expected):
    monkeypatch.setattr(adaptive_batching.torch.mps, "recommended_max_memory", lambda: 16)
    monkeypatch.setattr(adaptive_batching.torch.mps, "driver_allocated_memory", lambda: 6)
    monkeypatch.setattr(
        adaptive_batching.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=system_available),
    )
    assert adaptive_batching.device_memory_snapshot("mps").available_bytes == expected


def test_warm_mps_cache_does_not_hide_the_live_tensor_peak(monkeypatch):
    state = {"active": False}
    sampled = threading.Event()

    def tensor_bytes():
        if state["active"]:
            sampled.set()
            return 700
        return 100

    monkeypatch.setattr(adaptive_batching, "_synchronize", lambda _device: None)
    monkeypatch.setattr(adaptive_batching, "_allocated_bytes", lambda _device: 1000)
    monkeypatch.setattr(adaptive_batching.torch.mps, "current_allocated_memory", tensor_bytes)

    def operation():
        state["active"] = True
        assert sampled.wait(timeout=1)
        state["active"] = False

    _, measurement = adaptive_batching.measure_peak_memory(
        "mps", operation, sample_interval_seconds=0.0001
    )
    assert measurement.peak_bytes == measurement.baseline_bytes == 1000
    assert measurement.peak_increment_bytes == 600
    assert measurement.tensor_peak_increment_bytes == 600


def test_failed_measurement_drains_work_and_stops_sampler(monkeypatch):
    calls = []
    monkeypatch.setattr(adaptive_batching, "_synchronize", lambda _device: calls.append("drain"))
    monkeypatch.setattr(adaptive_batching, "_allocated_bytes", lambda _device: 0)

    def fail():
        calls.append("operation")
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        adaptive_batching.measure_peak_memory("cpu", fail)
    assert calls == ["drain", "operation", "drain"]
    assert not any(t.name == "cpu-memory-calibration" for t in threading.enumerate())


def test_calibrated_capacity_budgets_two_thirds_of_available_memory():
    gib = 1024**3
    snapshot = DeviceMemorySnapshot("cuda", 12 * gib, 16 * gib, 4 * gib, "test")
    assert calibrated_batch_capacity(
        snapshot, measured_peak_bytes_per_image=gib, remaining=20
    ) == 8
    assert calibrated_batch_capacity(
        snapshot, measured_peak_bytes_per_image=2 * gib, remaining=20
    ) == 4


def test_reported_laptop_workload_splits_into_five_and_four():
    gib = 1024**3
    snapshot = DeviceMemorySnapshot(
        "mps", int(10.51 * gib), 16 * gib, 0, "test"
    )
    capacity = calibrated_batch_capacity(
        snapshot, measured_peak_bytes_per_image=int(1.04 * gib), remaining=9
    )
    assert capacity == 6
    assert balanced_batch_size(9, capacity) == 5
    assert balanced_batch_size(4, capacity) == 4


@pytest.mark.parametrize("available_bytes", [0, 1, 2, 3, 4, 5, 10, 17, 100])
@pytest.mark.parametrize("peak_bytes", [1, 2, 3, 7])
def test_capacity_rounds_down_with_single_image_minimum(available_bytes, peak_bytes):
    snapshot = DeviceMemorySnapshot("cpu", available_bytes, 100, 0, "test")
    capacity = calibrated_batch_capacity(
        snapshot, measured_peak_bytes_per_image=peak_bytes, remaining=100
    )
    assert capacity >= 1
    if capacity > 1:
        assert capacity * peak_bytes * 3 <= available_bytes * 2
    assert (capacity + 1) * peak_bytes * 3 > available_bytes * 2


def test_capacity_is_limited_by_actual_remaining_work_not_a_device_cap():
    gib = 1024**3
    snapshot = DeviceMemorySnapshot("mps", 24 * gib, 32 * gib, 8 * gib, "test")
    assert calibrated_batch_capacity(
        snapshot, measured_peak_bytes_per_image=gib, remaining=8
    ) == 8


def test_missing_calibration_falls_back_to_one_instead_of_guessing():
    gib = 1024**3
    snapshot = DeviceMemorySnapshot("cpu", 32 * gib, 64 * gib, 32 * gib, "test")
    assert calibrated_batch_capacity(
        snapshot, measured_peak_bytes_per_image=0, remaining=8
    ) == 1


def test_balanced_batches_use_four_plus_four_instead_of_six_plus_two():
    assert balanced_batch_size(8, 6) == 4
    assert balanced_batch_size(4, 6) == 4
    assert balanced_batch_size(8, 9) == 8
    assert balanced_batch_size(3, 2) == 2


@pytest.mark.parametrize("remaining, capacity", [(9, 6), (20, 6), (13, 5), (8, 6)])
def test_repeated_planning_keeps_batches_within_one_image(remaining, capacity):
    total = remaining
    batches = []
    while remaining:
        size = balanced_batch_size(remaining, capacity)
        batches.append(size)
        remaining -= size
    assert sum(batches) == total
    assert max(batches) <= capacity
    assert max(batches) - min(batches) <= 1


def test_grid_coordinates_are_prioritized_by_distance():
    coordinates = prioritize_grid_coordinates([3, 3], (1, 1))
    distances = [abs(x - 1) + abs(y - 1) for x, y in coordinates]
    assert coordinates[0] == (1, 1)
    assert len(coordinates[1:]) == 8
    assert distances == sorted(distances)
