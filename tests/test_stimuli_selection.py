"""Real selection policy and Flask contract, without loading GAN checkpoints."""

import ast
import base64
import threading
import typing
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from flask import Flask, jsonify, request

from api_contract import (
    SELECTION_GRID_SIDE,
    ApiValidationError,
    eligible_photo_ids,
    parse_image_encoding,
    parse_selection_request,
)
from stimuli_selection_backend import StimuliSelectionBackend


@pytest.fixture
def builder():
    if not torch.backends.mps.is_available():
        pytest.skip("Run these small tensor checks outside the sandbox on MPS")
    device = torch.device("mps")

    class Mapping:
        w_avg = torch.ones(4, device=device)

        def __call__(self, z, _labels, truncation_psi):
            assert z.device.type == "mps" and truncation_psi == 1
            return z[:, None, :].repeat(1, 12, 1)

    class Builder:
        num_ws, z_dim = 12, 4
        G = SimpleNamespace(mapping=Mapping())

        def __init__(self):
            self.batches = []

        def generate_im_from_w_space(self, ws):
            assert ws.device.type == "mps"
            self.batches.append(ws.clone())
            values = ws[:, 0, 0].detach().cpu().numpy()
            return np.stack(
                [
                    np.full((2, 2, 3), round(float(v) * 10) % 256, dtype=np.uint8)
                    for v in values
                ]
            )

    result = Builder()
    result.device = device
    return result


def selection(filters=None, revision=1, psi=0.6):
    return parse_selection_request(
        {
            "preview_revision": revision,
            "truncation_psi": psi,
            "selection": {"filters": filters or {}},
        },
        ["dominant", "well-groomed"],
    )


def test_rating_sampling_applies_truncation_without_manipulating(builder):
    photos = {f"face-{i}": np.full(4, i, dtype=np.float32) for i in range(12)}
    engine = StimuliSelectionBackend(builder, photos, rng=np.random.default_rng(7))
    images, meta = engine.preview(
        selection({"dominant": [0.2, 0.8]}, psi=0.5), set(photos), lambda: None
    )
    assert images.shape == (9, 2, 2, 3)
    actual = torch.cat(builder.batches).cpu().numpy()
    assert len(builder.batches) == 9 and all(len(b) == 1 for b in builder.batches)
    assert len(np.unique(actual[:, 0, 0])) == 9
    assert np.allclose(actual, actual[:, :1, :])  # no layer/direction edits
    assert all(v in [(i + 1) / 2 for i in range(12)] for v in actual[:, 0, 0])
    assert meta == {
        "source": "ratings",
        "eligible_count": 12,
        "sampled_with_replacement": False,
    }


def test_small_pool_repeats_and_empty_pool_never_falls_back(builder):
    engine = StimuliSelectionBackend(builder, {"only": np.full(4, 2, dtype=np.float32)})
    images, meta = engine.preview(selection(), {"only"}, lambda: None)
    assert len(images) == 9 and meta["sampled_with_replacement"]
    assert meta["eligible_count"] == 1
    builder.batches.clear()
    with pytest.raises(ApiValidationError) as error:
        engine.preview(selection(), set(), lambda: None)
    assert error.value.status_code == 422 and not builder.batches


def test_unfiltered_draws_are_fresh_and_separate(builder):
    engine = StimuliSelectionBackend(builder, {}, rng=np.random.default_rng(42))
    engine.preview(selection(), None, lambda: None)
    first = torch.cat(builder.batches).clone()
    builder.batches.clear()
    _, meta = engine.preview(selection(), None, lambda: None)
    second = torch.cat(builder.batches)
    assert not torch.equal(first, second)
    assert len(torch.unique(second[:, 0, 0])) == 9
    assert meta["source"] == "generator" and meta["eligible_count"] is None


def test_superseded_request_stops_between_batches(builder):
    engine = StimuliSelectionBackend(builder, {})

    def check():
        if len(builder.batches) == 2:
            raise ApiValidationError("superseded", status_code=409)

    with pytest.raises(ApiValidationError):
        engine.preview(selection(), None, check)
    assert len(builder.batches) == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"preview_revision": True},
        {"sample_count": 8},
        {"sample_count": True},
        {"filters": {"dominant": [0.2, 0.8]}},
        {"truncation_psi": float("nan")},
        {"selection": []},
        {"selection": {"description": "a smiling person"}},
        {"selection": {"filters": {"unknown": [0, 1]}}},
    ],
)
def test_invalid_request_rejected(patch):
    payload = {"preview_revision": 1, **patch}
    with pytest.raises(ApiValidationError):
        parse_selection_request(payload, ["dominant"])


def test_selection_reuses_canonical_inclusive_rating_filters():
    parsed = selection({"wellGroomed": [0.8, 0.2]})
    assert parsed.filters == {"well-groomed": (0.2, 0.8)}
    assert eligible_photo_ids(
        parsed.filters,
        {"well-groomed": {"a": 0.2, "b": 0.8, "c": 0.9}},
        ["a", "b", "c"],
    ) == {"a", "b"}


@pytest.fixture
def route(builder):
    # Compile production routes/helpers, omitting only model/server bootstrap.
    names = {
        "generate_selection_preview",
        "_payload_preview_revision",
        "_activate_preview_revision",
        "_require_current_preview_revision",
        "_api_error",
        "_avg_ratings_for_dim",
        "_filter_photos",
        "encode_image_b64",
    }
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    for node in nodes:
        node.decorator_list = []
    photos = {"low": np.zeros(4), "high": np.full(4, 2)}
    backend = SimpleNamespace(
        photo_to_coords=photos,
        curr_w="unchanged",
        dim_to_photo_to_ratings={"dominant": {"low": [0.2], "high": [0.8]}},
        _average_ratings_cache={},
    )
    lock = threading.Lock()
    ns = dict(
        globals(),
        Any=typing.Any,
        Dict=dict,
        Optional=typing.Optional,
        Tuple=tuple,
        jsonify=jsonify,
        request=request,
        SELECTION_GRID_SIDE=SELECTION_GRID_SIDE,
        parse_image_encoding=parse_image_encoding,
        backend=backend,
        selection_backend=StimuliSelectionBackend(builder, photos),
        gpu_lock=lock,
        _preview_revision_lock=threading.Lock(),
        _active_preview_revision=-1,
        _cancel_active_full_resolution_jobs=lambda reason: [],
        _wait_for_cancelled_full_resolution_jobs=lambda: None,
    )

    def release():
        assert lock.locked()

    ns["_release_full_resolution_builder"] = release
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), ns)  # noqa: S102 - local definitions only
    app = Flask(__name__)
    app.add_url_rule(
        "/stimuli/preview", view_func=ns["generate_selection_preview"], methods=["POST"]
    )
    return app.test_client(), ns


def test_route_encodes_nine_filtered_faces_and_preserves_manipulation_base(route):
    client, ns = route
    response = client.post(
        "/stimuli/preview?format=png",
        json={
            "preview_revision": 10,
            "selection": {"filters": {"dominant": [0.8, 0.8]}},
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["preview_revision"] == 10
    assert len(body["images"]) == 3 and all(len(row) == 3 for row in body["images"])
    assert body["sampling"]["eligible_count"] == 1
    pixel = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["images"][0][0]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert pixel.shape == (2, 2, 3)
    assert ns["backend"].curr_w == "unchanged"
    assert (
        client.post("/stimuli/preview", json={"preview_revision": 9}).status_code == 409
    )


def test_route_empty_pool_and_bad_criteria_are_clear_errors(route):
    client, ns = route
    empty = client.post(
        "/stimuli/preview",
        json={
            "preview_revision": 1,
            "selection": {"filters": {"dominant": [0.4, 0.6]}},
        },
    )
    assert empty.status_code == 422 and "error" in empty.get_json()
    invalid = client.post(
        "/stimuli/preview", json={"preview_revision": 2, "selection": {"text": "smile"}}
    )
    assert invalid.status_code == 400
    assert (
        ns["_active_preview_revision"] == 1
    )  # invalid requests do not supersede valid work
