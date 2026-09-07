"""Reusable inference batch sizing based on current accelerator headroom."""

from __future__ import annotations

import itertools
import os
import threading
from dataclasses import dataclass
from typing import Callable, TypeVar, Union

import torch
import psutil


T = TypeVar("T")


def prioritize_grid_coordinates(
    level_counts: list[int], origin: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """Order an N-D grid from an origin using Manhattan distance."""
    if len(level_counts) != len(origin) or not level_counts:
        raise ValueError("level_counts and origin must describe the same non-empty grid")
    if any(count < 1 for count in level_counts):
        raise ValueError("every level count must be positive")
    if any(
        level < 0 or level >= level_counts[index]
        for index, level in enumerate(origin)
    ):
        raise ValueError("origin is outside the grid")
    coordinates = list(itertools.product(*(range(count) for count in level_counts)))
    coordinates.sort(
        key=lambda coordinate: (
            sum(abs(a - b) for a, b in zip(coordinate, origin)),
            coordinate,
        )
    )
    return coordinates


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    device_type: str
    available_bytes: int
    total_bytes: int
    allocated_bytes: int
    source: str


@dataclass(frozen=True)
class PeakMemoryMeasurement:
    device_type: str
    baseline_bytes: int
    peak_bytes: int
    after_bytes: int
    peak_increment_bytes: int
    source: str
    tensor_peak_increment_bytes: int = 0


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _process_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if os.uname().sysname == "Darwin" else rss * 1024


def _allocated_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return _process_rss_bytes()


def measure_peak_memory(
    device: Union[str, torch.device],
    operation: Callable[[], T],
    *,
    sample_interval_seconds: float = 0.01,
) -> tuple[T, PeakMemoryMeasurement]:
    """Run one calibration operation and measure its incremental peak memory."""
    resolved = torch.device(device)
    _synchronize(resolved)
    baseline = _allocated_bytes(resolved)
    tensor_baseline = (
        int(torch.mps.current_allocated_memory()) if resolved.type == "mps" else 0
    )
    tensor_peak = [tensor_baseline]

    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
        try:
            result = operation()
        finally:
            _synchronize(resolved)
        after = _allocated_bytes(resolved)
        peak = max(after, int(torch.cuda.max_memory_allocated(resolved)))
        source = "cuda.max_memory_allocated"
    else:
        stop = threading.Event()
        peak_holder = [baseline]

        def observe() -> None:
            peak_holder[0] = max(peak_holder[0], _allocated_bytes(resolved))
            if resolved.type == "mps":
                tensor_peak[0] = max(
                    tensor_peak[0], int(torch.mps.current_allocated_memory())
                )

        def sample() -> None:
            while not stop.wait(sample_interval_seconds):
                observe()

        sampler = threading.Thread(
            target=sample,
            name=f"{resolved.type}-memory-calibration",
            daemon=True,
        )
        sampler.start()
        try:
            result = operation()
        finally:
            try:
                # Drain cancelled/failed work before another caller gets the GPU.
                _synchronize(resolved)
            finally:
                stop.set()
                sampler.join()
        after = _allocated_bytes(resolved)
        observe()
        peak = peak_holder[0]
        source = (
            "MPS driver growth / live tensor peak sampler"
            if resolved.type == "mps"
            else "process RSS sampler"
        )

    measurement = PeakMemoryMeasurement(
        device_type=resolved.type,
        baseline_bytes=baseline,
        peak_bytes=peak,
        after_bytes=after,
        # A warm allocator can reuse memory without growing the driver total.
        # Track live tensors too, so reuse cannot masquerade as zero-cost work.
        peak_increment_bytes=max(0, peak - baseline, tensor_peak[0] - tensor_baseline),
        source=source,
        tensor_peak_increment_bytes=max(0, tensor_peak[0] - tensor_baseline),
    )
    return result, measurement


def device_memory_snapshot(
    device: Union[str, torch.device],
) -> DeviceMemorySnapshot:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
        allocated = torch.cuda.memory_allocated(resolved)
        return DeviceMemorySnapshot(
            "cuda",
            int(free_bytes),
            int(total_bytes),
            int(allocated),
            "cuda.mem_get_info",
        )

    if resolved.type == "mps":
        recommended = int(torch.mps.recommended_max_memory())
        allocated = int(torch.mps.driver_allocated_memory())
        # Metal's recommended working set is not the machine's free RAM.
        # MPS shares physical memory with the browser and all other processes.
        system_available = int(psutil.virtual_memory().available)
        return DeviceMemorySnapshot(
            "mps",
            max(0, min(recommended - allocated, system_available)),
            recommended,
            allocated,
            "min(Metal headroom, system available RAM)",
        )

    try:
        memory = psutil.virtual_memory()
        return DeviceMemorySnapshot(
            "cpu",
            int(memory.available),
            int(memory.total),
            int(memory.total - memory.available),
            "psutil.virtual_memory",
        )
    except Exception:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        return DeviceMemorySnapshot(
            "cpu", available, total, max(0, total - available), "os.sysconf"
        )


def calibrated_batch_capacity(
    snapshot: DeviceMemorySnapshot,
    *,
    measured_peak_bytes_per_image: int,
    remaining: int,
) -> int:
    """Budget at most two-thirds of available memory using measured image peaks.

    Keep a minimum of one image so calibration and low-memory jobs can progress,
    even when a single image exceeds the budget.
    """
    if remaining < 1:
        raise ValueError("remaining must be positive")
    if measured_peak_bytes_per_image <= 0:
        return 1
    memory_budget_bytes = max(0, snapshot.available_bytes) * 2 // 3
    capacity = max(1, memory_budget_bytes // measured_peak_bytes_per_image)
    return min(remaining, capacity)


def balanced_batch_size(remaining: int, capacity: int) -> int:
    """Choose equal-ish chunks without ever exceeding remaining or capacity."""
    if remaining < 1 or capacity < 1:
        raise ValueError("remaining and capacity must be positive")
    capacity = min(remaining, capacity)
    batch_count = (remaining + capacity - 1) // capacity
    return (remaining + batch_count - 1) // batch_count
