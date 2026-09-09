"""Pure request-validation helpers shared by the Flask API and unit tests."""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Collection, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple


@dataclass
class ApiValidationError(ValueError):
    message: str
    details: Any = None
    status_code: int = 400

    def __str__(self) -> str:
        return self.message


SELECTION_GRID_SIDE = 3
SELECTION_SAMPLE_COUNT = SELECTION_GRID_SIDE ** 2


@dataclass(frozen=True)
class StimuliSelectionRequest:
    """Sampling criteria are independent of per-face manipulation settings.

    Add future text-selection criteria here, not to manipulation dimensions.
    Unknown criteria fail explicitly until their selection policy exists.
    """

    preview_revision: int
    filters: Dict[str, Tuple[float, float]]
    truncation_psi: float
    sample_count: int = SELECTION_SAMPLE_COUNT


def parse_selection_request(
    payload: Any, available: Collection[str]
) -> StimuliSelectionRequest:
    if not isinstance(payload, Mapping):
        raise ApiValidationError("The request body must be a JSON object.")
    unknown_fields = set(payload) - {
        "preview_revision",
        "sample_count",
        "truncation_psi",
        "selection",
    }
    if unknown_fields:
        raise ApiValidationError(
            "Unsupported selection request fields.",
            details={"unsupported_fields": sorted(unknown_fields)},
        )
    revision = payload.get("preview_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ApiValidationError("preview_revision must be a non-negative integer.")
    selection = payload.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ApiValidationError("selection must be an object.")
    unknown = set(selection) - {"filters"}
    if unknown:
        raise ApiValidationError(
            "Unsupported selection criteria.",
            details={"unsupported_criteria": sorted(unknown)},
        )
    count = payload.get("sample_count", SELECTION_SAMPLE_COUNT)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != SELECTION_SAMPLE_COUNT
    ):
        raise ApiValidationError(
            "sample_count must be 9 for the 3x3 selection preview."
        )
    truncation = _finite_number(payload.get("truncation_psi", 0.6), "truncation_psi")
    if not 0.1 <= truncation <= 1.0:
        raise ApiValidationError("truncation_psi must be between 0.1 and 1.0.")
    return StimuliSelectionRequest(
        revision, parse_filters(selection.get("filters"), available), truncation, count
    )


def camel_to_dash(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ApiValidationError("Dimension names must be non-empty strings.")
    value = name.strip()
    result = []
    for index, char in enumerate(value):
        if char.isupper():
            if index and (
                not value[index - 1].isupper()
                or (index + 1 < len(value) and value[index + 1].islower())
            ):
                result.append("-")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def dash_to_camel(name: str) -> str:
    parts = name.split("-")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def canonical_dimension(name: str, available: Collection[str]) -> str:
    canonical = camel_to_dash(name)
    if canonical not in available:
        raise ApiValidationError(
            f"Unsupported dimension: {name}",
            details={"unsupported_dimensions": [name]},
        )
    return canonical


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ApiValidationError(f"{field} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ApiValidationError(f"{field} must be numeric.") from error
    if not math.isfinite(number):
        raise ApiValidationError(f"{field} must be finite.")
    return number


def _integer(value: Any, field: str) -> int:
    number = _finite_number(value, field)
    integer = int(number)
    if number != integer:
        raise ApiValidationError(f"{field} must be an integer.")
    return integer


def parse_filters(
    raw_filters: Any, available: Collection[str]
) -> Dict[str, Tuple[float, float]]:
    if raw_filters in (None, {}):
        return {}
    if not isinstance(raw_filters, Mapping):
        raise ApiValidationError("filters must be an object mapping dimensions to ranges.")

    parsed: Dict[str, Tuple[float, float]] = {}
    unsupported = []
    for raw_name, raw_range in raw_filters.items():
        try:
            name = canonical_dimension(raw_name, available)
        except ApiValidationError:
            unsupported.append(raw_name)
            continue
        if (
            not isinstance(raw_range, Sequence)
            or isinstance(raw_range, (str, bytes))
            or len(raw_range) != 2
        ):
            raise ApiValidationError(
                f"Filter {raw_name} must contain exactly two numeric bounds."
            )
        low = _finite_number(raw_range[0], f"filters.{raw_name}[0]")
        high = _finite_number(raw_range[1], f"filters.{raw_name}[1]")
        if not 0.0 <= low <= 1.0 or not 0.0 <= high <= 1.0:
            raise ApiValidationError(
                f"Filter {raw_name} bounds must be between 0 and 1."
            )
        parsed[name] = (min(low, high), max(low, high))

    if unsupported:
        raise ApiValidationError(
            "One or more filter dimensions are unsupported.",
            details={"unsupported_dimensions": unsupported},
        )
    return parsed


def parse_requested_dimensions(
    raw_variables: Any, available: Collection[str]
) -> list[str]:
    if raw_variables is None:
        return list(available)
    if not isinstance(raw_variables, list):
        raise ApiValidationError("variables must be a list of dimension names.")

    requested: list[str] = []
    unsupported = []
    for raw_name in raw_variables:
        try:
            canonical = canonical_dimension(raw_name, available)
        except ApiValidationError:
            unsupported.append(raw_name)
            continue
        if canonical not in requested:
            requested.append(canonical)
    if unsupported:
        raise ApiValidationError(
            "One or more requested dimensions are unsupported.",
            details={"unsupported_dimensions": unsupported},
        )
    return requested


def parse_num_points(value: Any) -> int:
    num_points = _integer(value, "num_points")
    if not 2 <= num_points <= 1000:
        raise ApiValidationError("num_points must be between 2 and 1000.")
    return num_points


def parse_image_request(
    payload: Any, available: Collection[str], num_ws: int
) -> tuple[Dict[str, Any], Dict[str, Tuple[float, float]]]:
    if not isinstance(payload, Mapping):
        raise ApiValidationError("The request body must be a JSON object.")

    raw_dimensions = payload.get("manipulated_dimensions")
    if not isinstance(raw_dimensions, list) or not 1 <= len(raw_dimensions) <= 3:
        raise ApiValidationError("manipulated_dimensions must contain 1 to 3 items.")

    names: list[str] = []
    strengths: list[list[float]] = []
    for index, raw_dimension in enumerate(raw_dimensions):
        if not isinstance(raw_dimension, Mapping):
            raise ApiValidationError(
                f"manipulated_dimensions[{index}] must be an object."
            )
        name = canonical_dimension(raw_dimension.get("name"), available)
        if name in names:
            raise ApiValidationError(
                f"Dimension {raw_dimension.get('name')} is selected more than once."
            )
        strength = _finite_number(
            raw_dimension.get("strength"),
            f"manipulated_dimensions[{index}].strength",
        )
        levels = _integer(
            raw_dimension.get("n_levels"),
            f"manipulated_dimensions[{index}].n_levels",
        )
        if not 2 <= levels <= 5:
            raise ApiValidationError("Each n_levels value must be between 2 and 5.")
        names.append(name)
        if levels == 2:
            level_values = [-strength, strength]
        else:
            step = (2.0 * strength) / (levels - 1)
            level_values = [-strength + step * level for level in range(levels)]
        strengths.append(level_values)

    truncation = _finite_number(payload.get("truncation_psi", 0.6), "truncation_psi")
    if not 0.1 <= truncation <= 1.0:
        raise ApiValidationError("truncation_psi must be between 0.1 and 1.0.")

    mode = payload.get("mode", "both")
    slices = {"both": (0, num_ws), "color": (9, num_ws), "shape": (0, 9)}
    if mode not in slices:
        raise ApiValidationError("mode must be one of: shape, color, both.")
    latents_from, latents_to = slices[mode]
    if latents_to > num_ws or latents_from >= latents_to:
        raise ApiValidationError(
            f"mode {mode} is unavailable for a generator with {num_ws} W layers."
        )

    steps = _integer(payload.get("max_steps", 40), "max_steps")
    if not 1 <= steps <= 100:
        raise ApiValidationError("max_steps must be between 1 and 100.")

    change_face = payload.get("change_face", True)
    if not isinstance(change_face, bool):
        raise ApiValidationError("change_face must be a boolean.")

    filters = parse_filters(payload.get("filters"), available)
    generation_config = {
        "num_faces": 1,
        "manipulated_dimensions": names,
        "strengths": strengths,
        "steps": steps,
        "latents_from": latents_from,
        "latents_to": latents_to,
        "truncation_psi": truncation,
        "change_face": False,
    }
    return generation_config, filters


def parse_image_encoding(raw_format: Any, raw_quality: Any) -> tuple[str, int]:
    image_format = str(raw_format or "webp").lower()
    if image_format == "jpeg":
        image_format = "jpg"
    if image_format not in {"png", "jpg", "webp"}:
        raise ApiValidationError("format must be one of: png, jpg, webp.")
    quality = _integer(raw_quality if raw_quality is not None else 90, "quality")
    if not 1 <= quality <= 100:
        raise ApiValidationError("quality must be between 1 and 100.")
    return image_format, quality


def eligible_photo_ids(
    filters: Mapping[str, Tuple[float, float]],
    average_ratings: Mapping[str, Mapping[str, float]],
    photos_with_coordinates: Optional[Iterable[str]] = None,
) -> Optional[Set[str]]:
    """Return None for no filters, otherwise the exact intersection (possibly empty)."""
    if not filters:
        return None
    eligible: Optional[Set[str]] = None
    for dimension, (low, high) in filters.items():
        ratings = average_ratings.get(dimension, {})
        subset = {photo for photo, value in ratings.items() if low <= value <= high}
        eligible = subset if eligible is None else eligible & subset
        if not eligible:
            return set()
    if photos_with_coordinates is not None:
        eligible = (eligible or set()) & set(photos_with_coordinates)
    return eligible or set()


def normalized_histogram(values: Iterable[float], num_points: int) -> list[float]:
    values_list = list(values)
    if not values_list:
        return [0.0] * num_points
    histogram, _ = np.histogram(values_list, bins=num_points, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32)
    if num_points >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        histogram = np.convolve(histogram, kernel / kernel.sum(), mode="same")
    maximum = float(histogram.max())
    if maximum > 0:
        histogram = histogram / maximum
    return [float(value) for value in histogram.tolist()]


def reshape_image_grid_for_api(
    flat_images: np.ndarray, grid_shape: Sequence[int]
) -> np.ndarray:
    """Restore the generation grid and put 2D responses in UI row-major order."""
    image_shape = tuple(flat_images.shape[1:])
    images = flat_images.reshape(*grid_shape, *image_shape)
    if len(grid_shape) == 2:
        images = images.swapaxes(0, 1)
    return images
