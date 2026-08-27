import pytest
import numpy as np

from api_contract import (
    ApiValidationError,
    camel_to_dash,
    eligible_photo_ids,
    normalized_histogram,
    parse_filters,
    parse_image_encoding,
    parse_image_request,
    parse_requested_dimensions,
    reshape_image_grid_for_api,
)


AVAILABLE = ["dominant", "well-groomed", "looks-like-you"]


def test_name_normalization_and_optional_variables():
    assert camel_to_dash("wellGroomed") == "well-groomed"
    assert parse_requested_dimensions(None, AVAILABLE) == AVAILABLE
    assert parse_requested_dimensions(["looksLikeYou"], AVAILABLE) == [
        "looks-like-you"
    ]


def test_filters_reverse_bounds_and_reject_unknown_dimensions():
    assert parse_filters({"wellGroomed": [0.8, 0.2]}, AVAILABLE) == {
        "well-groomed": (0.2, 0.8)
    }
    with pytest.raises(ApiValidationError) as error:
        parse_filters({"missing": [0.0, 1.0]}, AVAILABLE)
    assert error.value.details == {"unsupported_dimensions": ["missing"]}


def test_empty_filter_intersection_stays_empty():
    averages = {
        "dominant": {"a.jpg": 0.2, "b.jpg": 0.8},
        "well-groomed": {"a.jpg": 0.9, "b.jpg": 0.1},
    }
    assert eligible_photo_ids({}, averages) is None
    assert eligible_photo_ids(
        {"dominant": (0.0, 0.3), "well-groomed": (0.0, 0.2)}, averages
    ) == set()
    assert eligible_photo_ids(
        {"dominant": (0.0, 0.3)}, averages, {"b.jpg"}
    ) == set()


def test_histogram_length_normalization_and_empty_values():
    assert normalized_histogram([], 7) == [0.0] * 7
    histogram = normalized_histogram([0.1, 0.2, 0.2, 0.8], 10)
    assert len(histogram) == 10
    assert max(histogram) == pytest.approx(1.0)


def test_unequal_2d_image_grid_is_returned_as_ui_rows():
    flat = np.arange(12, dtype=np.uint8).reshape(12, 1, 1, 1)
    grid = reshape_image_grid_for_api(flat, [3, 4])

    assert grid.shape == (4, 3, 1, 1, 1)
    assert grid[:, :, 0, 0, 0].tolist() == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]


def test_image_request_is_sanitized_and_uses_dimension_order():
    config, filters = parse_image_request(
        {
            "manipulated_dimensions": [
                {"name": "dominant", "strength": 2, "n_levels": 2},
                {"name": "wellGroomed", "strength": 3, "n_levels": 3},
            ],
            "truncation_psi": 0.6,
            "mode": "shape",
            "change_face": False,
            "filters": {"looksLikeYou": [0.1, 0.9]},
            "num_faces": 999,
            "preserve_identity": True,
            "controlled_variables": ["dominant"],
        },
        AVAILABLE,
        18,
    )
    assert config["manipulated_dimensions"] == ["dominant", "well-groomed"]
    assert config["strengths"] == [[-2.0, 2.0], [-3.0, 0.0, 3.0]]
    assert config["latents_from"] == 0
    assert config["latents_to"] == 9
    assert config["num_faces"] == 1
    assert config["change_face"] is False
    assert filters == {"looks-like-you": (0.1, 0.9)}


@pytest.mark.parametrize(
    ("image_format", "quality", "expected"),
    [("jpeg", "80", ("jpg", 80)), (None, None, ("webp", 90))],
)
def test_image_encoding(image_format, quality, expected):
    assert parse_image_encoding(image_format, quality) == expected


def test_image_request_rejects_invalid_levels_and_mode():
    base = {
        "manipulated_dimensions": [
            {"name": "dominant", "strength": 2, "n_levels": 1}
        ],
        "mode": "both",
    }
    with pytest.raises(ApiValidationError):
        parse_image_request(base, AVAILABLE, 18)
    base["manipulated_dimensions"][0]["n_levels"] = 2
    base["mode"] = "texture"
    with pytest.raises(ApiValidationError):
        parse_image_request(base, AVAILABLE, 18)
