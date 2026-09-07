"""Exercise production scheduling without app.py's model-loading side effects."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import weakref

import numpy as np
import pytest
import torch
from flask import Flask, jsonify, request

from adaptive_batching import (
    DeviceMemorySnapshot, PeakMemoryMeasurement,
    balanced_batch_size, calibrated_batch_capacity,
)
from api_contract import ApiValidationError


@pytest.fixture
def runtime():
    # Compile the actual definitions, omitting only application bootstrap and
    # route decorators. No model files, server, or generated images are needed.
    names = {
        "FullResolutionGridJob", "FullResolutionCancelled", "_is_oom",
        "_run_full_resolution_grid_job", "_asset_response",
        "full_resolution_grid_status", "_api_error",
    }
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    for node in definitions:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    namespace = dict(globals(), gpu_lock=threading.Lock())
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "app.py", "exec"), namespace)
    namespace["_clear_accelerator_cache"] = lambda _device: None
    namespace["_encoded_full_resolution_asset"] = (
        lambda _image, latent, _fmt, _quality: {"image": str(int(latent[0])), "latent_npy": "x"}
    )

    def snapshot(_device):
        assert namespace["gpu_lock"].locked()
        return DeviceMemorySnapshot("cpu", 10_000, 16_000, 0, "test")

    def measure(_device, operation):
        result = operation()
        return result, PeakMemoryMeasurement("cpu", 0, 1000, 0, 1000, "test")

    namespace["device_memory_snapshot"] = snapshot
    namespace["measure_peak_memory"] = measure
    namespace["_synthesize_full_resolution_batch"] = (
        lambda _builder, latents, _check: np.zeros((len(latents), 2, 2, 3), dtype=np.uint8)
    )
    return namespace


def make_job(runtime, count=9):
    coordinates = [(index,) for index in range(count)]
    return runtime["FullResolutionGridJob"](
        cache_key="test", preview_is_low_resolution=True,
        coordinates=coordinates, pending=list(coordinates),
        coordinate_to_index={coordinate: index for index, coordinate in enumerate(coordinates)},
        latents=torch.arange(count).reshape(count, 1),
        calibrated_peak_bytes_per_image=1000, initializing=False,
    )


def test_worker_balances_without_flushing_successful_batches(runtime):
    batches = []
    raw_images = []

    def synthesize(_builder, latents, _check):
        assert all(ref() is None for ref in raw_images)
        batches.append(len(latents))
        result = np.zeros((len(latents), 2, 2, 3), dtype=np.uint8)
        raw_images.append(weakref.ref(result))
        return result

    def unexpected_clear(_device):
        pytest.fail("Successful batches must reuse the cache")

    runtime["_synthesize_full_resolution_batch"] = synthesize
    runtime["_clear_accelerator_cache"] = unexpected_clear
    job = make_job(runtime)
    runtime["_run_full_resolution_grid_job"](job, SimpleNamespace(device=torch.device("cpu")))
    assert job.error is None
    assert job.complete and len(job.cached) == 9
    assert batches == [5, 4]


def test_oom_retries_after_failed_tensor_references_are_released(runtime):
    failed_allocations = []
    batches = []
    clears = []

    def synthesize(_builder, latents, _check):
        batches.append(len(latents))
        if len(batches) == 1:
            allocation = np.zeros(8)
            failed_allocations.append(weakref.ref(allocation))
            raise RuntimeError("out of memory")
        return np.zeros((len(latents), 2, 2, 3), dtype=np.uint8)

    def clear(_device):
        assert all(ref() is None for ref in failed_allocations)
        clears.append(True)

    runtime["_synthesize_full_resolution_batch"] = synthesize
    runtime["_clear_accelerator_cache"] = clear
    job = make_job(runtime)
    runtime["_run_full_resolution_grid_job"](job, SimpleNamespace(device=torch.device("cpu")))
    assert job.error is None and job.complete
    assert len(job.cached) == 9
    assert batches == [5, 2, 2, 2, 2, 1]
    assert len(clears) == 1


def test_gpu_waiter_can_be_cancelled_without_reserving_work(runtime):
    job = make_job(runtime)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with runtime["gpu_lock"]:
            future = executor.submit(
                runtime["_run_full_resolution_grid_job"], job,
                SimpleNamespace(device=torch.device("cpu")),
            )
            job.cancel("new preview")
        future.result(timeout=2)
    assert job.complete and job.error is None and not job.cached


def test_concurrent_callers_serialize_inference_but_status_can_progress(runtime):
    entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum_active = 0

    def synthesize(_builder, latents, _check):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        entered.set()
        assert release.wait(timeout=2)
        active -= 1
        return np.zeros((len(latents), 2, 2, 3), dtype=np.uint8)

    runtime["_synthesize_full_resolution_batch"] = synthesize
    jobs = [make_job(runtime, 1), make_job(runtime, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime["_run_full_resolution_grid_job"], jobs[0], SimpleNamespace(device="cpu"))
        assert entered.wait(timeout=2)
        second = executor.submit(runtime["_run_full_resolution_grid_job"], jobs[1], SimpleNamespace(device="cpu"))
        try:
            assert jobs[0].updates_after(0)["items"] == []
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert maximum_active == 1
    assert all(job.complete and job.error is None for job in jobs)


def test_status_pages_drain_before_reporting_completion(runtime):
    job = make_job(runtime, 5)
    for coordinate in job.coordinates:
        job.store(coordinate, {"image": "x", "latent_npy": "y"})
    job.complete = True
    pages = [job.updates_after(cursor) for cursor in (0, 2, 4)]
    assert [len(page["items"]) for page in pages] == [2, 2, 1]
    assert [page["cursor"] for page in pages] == [2, 4, 5]
    assert [page["complete"] for page in pages] == [False, False, True]


@pytest.mark.parametrize("cancel", [False, True])
def test_status_wait_wakes_for_image_or_cancellation(runtime, cancel):
    job = make_job(runtime, 1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(job.updates_after, 0, 10)
        if cancel:
            job.cancel("new preview")
        else:
            job.store((0,), {"image": "x", "latent_npy": "y"})
        page = result.result(timeout=2)
    assert page["complete"] == cancel
    assert len(page["items"]) == (0 if cancel else 1)


@pytest.mark.parametrize("wait_ms", [-1, 10001, True, "1000"])
def test_status_rejects_invalid_wait(runtime, wait_ms):
    app = Flask(__name__)
    with app.test_request_context(json={"cache_key": "test", "wait_ms": wait_ms}):
        _, status = runtime["full_resolution_grid_status"]()
    assert status == 400
