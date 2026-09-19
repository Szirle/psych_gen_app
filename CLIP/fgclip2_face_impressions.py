"""Reusable FG-CLIP 2 pipelines for face datasets and impression ratings.

This module has no Gradio or experiment-orchestration dependency.
Plotting is optional and imported only by Viz methods. It owns the reusable workflow that future applications need:

* discover and naturally sort a folder of face images;
* batch-encode and persist normalized FG-CLIP image features;
* cache phrase ratings derived from those features;
* load per-photo human ratings and aggregate them to means;
* align complete cases for regression or correlation analysis;
* fit readouts and evaluate arbitrary predictors with one results contract.

Pickle files must be trusted local inputs.

Historical research stages, artifact generation, and reports live in
``CLIP.fgclip2_legacy_research``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .fgclip2_core import DEFAULT_CACHE_DIR, FGCLIP2
except ImportError:
    from fgclip2_core import DEFAULT_CACHE_DIR, FGCLIP2


DEFAULT_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
CACHE_VERSION = 1


@dataclass(frozen=True)
class FaceDatasetConfig:
    directory: Path
    cache_dir: Path = DEFAULT_CACHE_DIR / "dataset-rankings"
    limit: int | None = 1000
    batch_size: int = 64
    image_suffixes: frozenset[str] = DEFAULT_IMAGE_SUFFIXES

    def __post_init__(self):
        object.__setattr__(self, "directory", Path(self.directory).expanduser())
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.limit is not None and (isinstance(self.limit, bool) or self.limit < 1):
            raise ValueError("Dataset limit must be a positive integer or None.")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("Dataset batch size must be a positive integer.")


@dataclass(frozen=True)
class FaceFeatureSet:
    paths: tuple[Path, ...]
    features: torch.Tensor
    cache_key: str
    cache_hit: bool
    elapsed_seconds: float
    effective_batch_size: int | None
    encoded_images: int = 0


@dataclass(frozen=True)
class DatasetScores:
    paths: tuple[Path, ...]
    texts: tuple[str, ...]
    scores: np.ndarray  # [images, texts], float32 agreement logits
    text_mode: str
    rating_cache_hits: tuple[bool, ...]
    runtime: dict[str, Any]

    def details(self) -> dict[str, Any]:
        return dict(self.runtime)


@dataclass(frozen=True)
class HumanRatingVector:
    variable: str
    means: np.ndarray
    rater_counts: np.ndarray
    mean_se: np.ndarray | None = None

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.means)


@dataclass(frozen=True)
class RegressionDataset:
    variable: str
    paths: tuple[Path, ...]
    texts: tuple[str, ...]
    human_means: np.ndarray
    rater_counts: np.ndarray
    predicted_scores: np.ndarray

    def standardized(self) -> tuple[np.ndarray, np.ndarray]:
        """Exploratory standardization; CV must fit scaling on training rows."""
        y, x = self.human_means, self.predicted_scores
        scale = y.std()
        if scale == 0 or np.any(x.std(axis=0) == 0):
            raise ValueError("Cannot standardize a constant target or predictor.")
        return (y-y.mean())/scale, scale_training(x)[0]


@dataclass(frozen=True)
class CorrelationMetrics:
    phrase: str
    images: int
    pearson: float
    spearman: float
    distance_correlation: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _natural_path_key(path: Path):
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def discover_face_images(config: FaceDatasetConfig) -> tuple[Path, ...]:
    directory = config.directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in config.image_suffixes
        ),
        key=_natural_path_key,
    )
    if config.limit is not None:
        paths = paths[: config.limit]
    if not paths:
        raise ValueError(f"No supported face images found in: {directory}")
    return tuple(paths)


def _atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _is_accelerator_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).casefold()


class FaceImpressionPipeline:
    """Efficiently rate one configured face dataset with a shared FG-CLIP model."""

    def __init__(self, engine: FGCLIP2, config: FaceDatasetConfig):
        self.engine = engine
        self.config = config
        self._feature_memory: dict[str, tuple[tuple[Path, ...], torch.Tensor]] = {}

    def image_paths(self) -> tuple[Path, ...]:
        return discover_face_images(self.config)

    def _feature_identity(
        self, paths: tuple[Path, ...], max_num_patches: int
    ) -> tuple[str, dict[str, Any]]:
        directory = paths[0].parent
        files = []
        for path in paths:
            stat = path.stat()
            files.append(
                [str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns]
            )
        manifest = {
            "version": CACHE_VERSION,
            "directory": str(directory),
            "model": self.engine.model_id,
            "revision": self.engine.revision,
            "dtype": str(self.engine.dtype),
            "patches": int(max_num_patches),
            "files": files,
        }
        encoded = json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:24], manifest

    @torch.inference_mode()
    def _encode_features(
        self, paths: tuple[Path, ...], max_num_patches: int
    ) -> tuple[torch.Tensor, int]:
        chunks = []
        offset = 0
        effective_batch_size = self.config.batch_size
        while offset < len(paths):
            current = min(effective_batch_size, len(paths) - offset)
            inputs = None
            try:
                inputs = self.engine.prepare_images(
                    paths[offset : offset + current],
                    max_num_patches=int(max_num_patches),
                )
                features = self.engine.encode_image_preprocessed(inputs).cpu()
            except (MemoryError, RuntimeError) as exc:
                if current == 1 or not (
                    isinstance(exc, MemoryError) or _is_accelerator_oom(exc)
                ):
                    raise
                effective_batch_size = max(1, current // 2)
                del inputs
                if self.engine.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.engine.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            chunks.append(features)
            offset += current
            del inputs, features
        return torch.cat(chunks), effective_batch_size

    def features(self, max_num_patches: int = 128) -> FaceFeatureSet:
        started = time.perf_counter()
        paths = self.image_paths()
        key, manifest = self._feature_identity(paths, max_num_patches)
        memory_entry = self._feature_memory.get(key)
        if memory_entry is not None and memory_entry[0] == paths:
            return FaceFeatureSet(
                paths,
                memory_entry[1],
                key,
                True,
                time.perf_counter() - started,
                None,
            )

        cache_path = self.config.cache_dir / f"features-{key}.npz"
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"]))
                relative_paths = cache["paths"].tolist()
                features_array = cache["features"]
            expected_paths = [str(path.relative_to(paths[0].parent)) for path in paths]
            if metadata != manifest or relative_paths != expected_paths:
                raise ValueError("Dataset feature cache identity mismatch.")
            features = torch.from_numpy(features_array.copy()).float()
            if features.ndim != 2 or features.shape[0] != len(paths):
                raise ValueError("Dataset feature cache has an invalid shape.")
            self._feature_memory[key] = (paths, features)
            return FaceFeatureSet(
                paths,
                features,
                key,
                True,
                time.perf_counter() - started,
                None,
            )
        except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)

        # Reuse an exact, manifest-verified prefix when expanding a dataset.
        # Labels never participate in this cache or in frozen image encoding.
        prefix = None
        prefix_length = 0
        for candidate in sorted(self.config.cache_dir.glob("features-*.npz")):
            try:
                with np.load(candidate, allow_pickle=False) as cache:
                    previous = json.loads(str(cache["metadata"]))
                    count = len(previous["files"])
                    if not prefix_length < count < len(paths):
                        continue
                    if any(previous.get(k) != v for k, v in manifest.items() if k != "files"):
                        continue
                    if previous["files"] != manifest["files"][:count]:
                        continue
                    if cache["paths"].tolist() != [p.name for p in paths[:count]]:
                        continue
                    array = cache["features"]
                    if array.ndim != 2 or len(array) != count or not np.isfinite(array).all():
                        continue
                    prefix, prefix_length = torch.from_numpy(array.copy()).float(), count
            except (KeyError, OSError, ValueError):
                continue
        features, effective_batch_size = self._encode_features(paths[prefix_length:], max_num_patches)
        if prefix is not None:
            features = torch.cat((prefix, features))
        relative_paths = [str(path.relative_to(paths[0].parent)) for path in paths]
        _atomic_save_npz(
            cache_path,
            features=features.numpy(),
            paths=np.asarray(relative_paths),
            metadata=np.asarray(json.dumps(manifest, ensure_ascii=False)),
        )
        self._feature_memory[key] = (paths, features)
        return FaceFeatureSet(
            paths,
            features,
            key,
            False,
            time.perf_counter() - started,
            effective_batch_size,
            len(paths) - prefix_length,
        )

    @staticmethod
    def _rating_cache_key(feature_key: str, text: str, text_mode: str) -> str:
        encoded = json.dumps(
            [feature_key, text.lower().strip(), text_mode],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]

    def rate(
        self,
        texts: list[str] | tuple[str, ...],
        *,
        text_mode: str = "auto",
        max_num_patches: int = 128,
    ) -> DatasetScores:
        texts = tuple(texts)
        text_details = self.engine.text_info(texts, mode=text_mode)
        resolved_mode = text_details["mode"]
        started = time.perf_counter()
        feature_set = self.features(max_num_patches)
        scores = np.empty((len(feature_set.paths), len(texts)), dtype=np.float32)
        cache_hits = [False] * len(texts)
        missing = []
        rating_paths = []

        for index, text in enumerate(texts):
            rating_key = self._rating_cache_key(
                feature_set.cache_key, text, resolved_mode
            )
            rating_path = self.config.cache_dir / f"ratings-{rating_key}.npz"
            rating_paths.append(rating_path)
            try:
                with np.load(rating_path, allow_pickle=False) as cache:
                    cached_scores = cache["scores"].astype(np.float32, copy=True)
                if (
                    cached_scores.shape != (len(feature_set.paths),)
                    or not np.isfinite(cached_scores).all()
                ):
                    raise ValueError("Dataset rating cache has invalid scores.")
                scores[:, index] = cached_scores
                cache_hits[index] = True
            except (FileNotFoundError, KeyError, OSError, ValueError):
                rating_path.unlink(missing_ok=True)
                missing.append(index)

        if missing:
            with torch.inference_mode():
                text_features = torch.cat([
                    self.engine.encode_text([texts[index] for index in missing[start:start+32]],
                                            mode=resolved_mode).cpu()
                    for start in range(0, len(missing), 32)
                ])
                scale = self.engine.model.logit_scale.float().exp().item()
                bias = self.engine.model.logit_bias.float().item()
                new_scores = feature_set.features @ text_features.T
                new_scores = new_scores * scale + bias
            scores[:, missing] = new_scores.numpy()
            for index in missing:
                _atomic_save_npz(rating_paths[index], scores=scores[:, index])

        if not np.isfinite(scores).all():
            raise RuntimeError("FGCLIP produced non-finite dataset scores.")
        runtime = {
            "images_rated": len(feature_set.paths),
            "dataset": str(feature_set.paths[0].parent),
            "feature_cache_hit": feature_set.cache_hit,
            "images_encoded": feature_set.encoded_images,
            "rating_cache_hits": sum(cache_hits),
            "phrases_rated": len(texts),
            "feature_seconds": round(feature_set.elapsed_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "images_per_second": round(
                feature_set.encoded_images / max(feature_set.elapsed_seconds, 1e-9), 2
            )
            if not feature_set.cache_hit
            else None,
            "requested_batch_size": self.config.batch_size,
            "effective_batch_size": feature_set.effective_batch_size or "cached",
            "model": self.engine.model_id,
            "revision": self.engine.revision,
            "device": str(self.engine.device),
            "dtype": str(self.engine.dtype),
            "text": text_details,
        }
        return DatasetScores(
            feature_set.paths,
            texts,
            scores,
            resolved_mode,
            tuple(cache_hits),
            runtime,
        )


class HumanRatingsStore:
    """Load and aggregate dimension -> photo -> individual ratings mappings."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._data: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is not None:
            return self._data
        if not self.path.is_file():
            raise ValueError(f"Human ratings file does not exist: {self.path}")
        with self.path.open("rb") as source:
            data = pickle.load(source)
        if not isinstance(data, dict) or not data:
            raise ValueError("Human ratings file must contain a nonempty dictionary.")
        self._data = data
        return data

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key, value in self._load().items()
                if isinstance(key, str) and isinstance(value, dict) and value
            )
        )

    def means(
        self, variable: str, paths: tuple[Path, ...] | list[Path], *,
        min_ratings: int = 1, reject_nonfinite: bool = False,
    ) -> HumanRatingVector:
        data = self._load()
        if variable not in data or not isinstance(data[variable], dict):
            raise ValueError(f"Unknown human ratings variable: {variable}")
        ratings_by_photo = data[variable]
        means = np.full(len(paths), np.nan, dtype=np.float64)
        counts = np.zeros(len(paths), dtype=np.int32)
        se = np.full(len(paths), np.nan)
        for index, path in enumerate(paths):
            ratings = ratings_by_photo.get(path.name)
            if ratings is None:
                continue
            try:
                values = np.asarray(ratings, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if reject_nonfinite and not np.isfinite(values).all():
                continue
            values = values[np.isfinite(values)]
            if values.size >= min_ratings:
                means[index] = values.mean()
                counts[index] = values.size
                if values.size > 1:
                    se[index] = values.std(ddof=1) / np.sqrt(values.size)
        return HumanRatingVector(variable, means, counts, se)

    def align(self, variable: str, dataset_scores: DatasetScores) -> RegressionDataset:
        ratings = self.means(variable, dataset_scores.paths)
        valid = ratings.valid_mask & np.isfinite(dataset_scores.scores).all(axis=1)
        if valid.sum() < 3:
            raise ValueError(
                f"Only {int(valid.sum())} dataset images have complete {variable} data."
            )
        valid_indices = np.flatnonzero(valid)
        return RegressionDataset(
            variable,
            tuple(dataset_scores.paths[index] for index in valid_indices),
            dataset_scores.texts,
            ratings.means[valid],
            ratings.rater_counts[valid],
            dataset_scores.scores[valid],
        )


def pearson_correlation(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def average_ranks(values) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(values, method="average")


def centered_distances(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    distances = np.abs(values[:, None] - values[None, :])
    row_means = distances.mean(axis=1, keepdims=True)
    column_means = distances.mean(axis=0, keepdims=True)
    return distances - row_means - column_means + distances.mean()


def distance_correlation(x, y) -> float:
    return _distance_correlation_from_centered(centered_distances(x), y)


def _distance_correlation_from_centered(centered_x, y) -> float:
    centered_y = centered_distances(y)
    covariance_squared = float(np.mean(centered_x * centered_y))
    variance_x_squared = float(np.mean(centered_x * centered_x))
    variance_y_squared = float(np.mean(centered_y * centered_y))
    denominator = np.sqrt(variance_x_squared * variance_y_squared)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(max(0.0, covariance_squared / denominator)))


def correlate_regression_data(
    regression: RegressionDataset,
) -> tuple[CorrelationMetrics, ...]:
    centered_human = centered_distances(regression.human_means)
    human_ranks = average_ranks(regression.human_means)
    results = []
    for index, phrase in enumerate(regression.texts):
        predicted = regression.predicted_scores[:, index]
        results.append(
            CorrelationMetrics(
                phrase=phrase,
                images=len(regression.paths),
                pearson=pearson_correlation(regression.human_means, predicted),
                spearman=pearson_correlation(human_ranks, average_ranks(predicted)),
                distance_correlation=_distance_correlation_from_centered(
                    centered_human, predicted
                ),
            )
        )
    return tuple(results)


def regression_metrics(y, prediction) -> dict[str, float]:
    """Scalar-target metrics; undefined correlations/R² are NaN."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape:
        raise ValueError("Expected aligned one-dimensional targets/predictions.")
    if not len(y):
        return dict.fromkeys(("pearson", "spearman", "r2", "mse", "rmse", "mae"), float("nan"))
    error = y - prediction
    variance = np.sum((y - y.mean()) ** 2)
    mse = float(np.mean(error ** 2))
    return dict(pearson=pearson_correlation(y, prediction),
                spearman=pearson_correlation(average_ranks(y), average_ranks(prediction)),
                r2=float(1 - error @ error / variance) if variance else float("nan"),
                mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(error))))


@dataclass
class RegressionResults:
    """OOF contract: truth [rows, targets], predictions {method: [rows, targets]}.

    One-dimensional arrays are promoted to one target. Metrics are always
    indexed by method, then target; summary() is JSON-ready, without raw arrays.
    """
    y_true: np.ndarray
    predictions: dict[str, np.ndarray]
    targets: tuple[str, ...]
    fold_ids: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        def matrix(a):
            a = np.asarray(a, dtype=float)
            return a[:, None] if a.ndim == 1 else a
        self.y_true = matrix(self.y_true)
        self.targets = tuple(self.targets)
        self.predictions = {name: matrix(p) for name, p in self.predictions.items()}
        if (self.y_true.ndim != 2 or not len(self.y_true) or not self.predictions
                or self.y_true.shape[1] != len(self.targets) or len(set(self.targets)) != len(self.targets)
                or not np.isfinite(self.y_true).all()
                or any(p.shape != self.y_true.shape or not np.isfinite(p).all() for p in self.predictions.values())):
            raise ValueError("Expected finite aligned predictions and unique target names.")
        if self.fold_ids is not None:
            self.fold_ids = np.asarray(self.fold_ids)
            if self.fold_ids.shape != (len(self.y_true),) or not np.issubdtype(self.fold_ids.dtype, np.integer) or np.any(self.fold_ids < 0):
                raise ValueError("fold_ids must contain one nonnegative integer per row.")

    def _metrics(self, rows):
        return {name: {target: regression_metrics(self.y_true[rows, j], p[rows, j])
                       for j, target in enumerate(self.targets)} for name, p in self.predictions.items()}

    @property
    def metrics(self):
        return self._metrics(slice(None))

    @property
    def fold_metrics(self):
        return {} if self.fold_ids is None else {
            int(fold): self._metrics(self.fold_ids == fold) for fold in np.unique(self.fold_ids)}

    def summary(self):
        """Serializable metrics/metadata; undefined numerical values become null."""
        value = dict(n=len(self.y_true), targets=list(self.targets), methods=list(self.predictions),
                     metrics=self.metrics, fold_metrics=self.fold_metrics,
                     macro_mse={n: float(np.mean((p-self.y_true)**2)) for n,p in self.predictions.items()},
                     metadata=self.metadata)
        def clean(v):
            if isinstance(v, dict): return {k: clean(x) for k,x in v.items()}
            if isinstance(v, (list, tuple, np.ndarray)): return [clean(x) for x in v]
            if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
            if isinstance(v, np.integer): return int(v)
            return v
        return clean(value)

    def metric_table(self, methods=None, metrics=("pearson", "spearman", "r2", "rmse", "mae")):
        """Compact Markdown table; the same metrics contract serves every model."""
        values = self.metrics
        rows = [[name, target, *[f"{values[name][target][m]:.4f}" for m in metrics]]
                for name in (self.predictions if methods is None else methods) for target in self.targets]
        return "\n".join("| " + " | ".join(row) + " |" for row in
                         [["Method", "Target", *metrics], ["---"]*(len(metrics)+2), *rows])


class Viz:
    """Optional plots of RegressionResults; callers own/close returned figures."""
    @staticmethod
    def oof_scatter(results: RegressionResults, method: str, *, columns=4, title=None):
        import matplotlib.pyplot as plt
        if columns < 1:
            raise ValueError("columns must be positive.")
        columns = min(columns, len(results.targets))
        rows = (len(results.targets) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.8*columns, 3.5*rows),
                                 squeeze=False, constrained_layout=True)
        prediction, metrics = results.predictions[method], results.metrics[method]
        for j, (target, ax) in enumerate(zip(results.targets, axes.flat)):
            y, p = results.y_true[:, j], prediction[:, j]
            lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
            ax.scatter(y, p, s=9, alpha=.35, edgecolors="none")
            ax.plot([lo, hi], [lo, hi], color="gray", lw=1)
            ax.set(title=f"{target}\nr={metrics[target]['pearson']:.3f}  RMSE={metrics[target]['rmse']:.4f}",
                   xlabel="Observed rating", ylabel="Held-out prediction")
        for ax in list(axes.flat)[len(results.targets):]: ax.set_visible(False)
        if title: fig.suptitle(title)
        return fig

    @staticmethod
    def export(fig, path, *, formats=("png", "svg"), dpi=170, close=True):
        import matplotlib.pyplot as plt
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in formats: fig.savefig(path.with_suffix("." + suffix), dpi=dpi)
        if close: plt.close(fig)


def make_cv_splits(n: int, folds: int, seed: int, groups=None):
    """Use groups for related faces/latent traversals; defaults to shuffled faces."""
    from sklearn.model_selection import GroupKFold, KFold

    if groups is None:
        return list(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n)))
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError("groups must have one entry per face.")
    return list(GroupKFold(folds, shuffle=True, random_state=seed).split(np.arange(n), groups=groups))


def cross_validate_predictor(x, y, fit_predict, *, splits=None, folds=10, seed=20260908):
    """Small architecture-independent full-CV runner; no approximated CV scores.

    fit_predict(X_train, y_train, X_test, fold) must do ALL supervised fitting
    and preprocessing using its training arguments. Targets may be [rows] or
    [rows, targets]; predictions may additionally include a methods axis.
    Supplied splits can also be LeaveOneOut/GroupKFold. Each face must be tested
    exactly once; repeated CV is performed with separate calls/seeds.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim not in (1, 2) or len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite X[n,p] and y[n] or y[n,targets].")
    splits = list(splits) if splits is not None else make_cv_splits(len(y), folds, seed)
    coverage = np.zeros(len(y), dtype=int)
    for train, test in splits:
        train, test = np.asarray(train), np.asarray(test)
        if len(train) == 0 or len(test) == 0 or np.intersect1d(train, test).size:
            raise ValueError("CV train/test sets must be nonempty and disjoint.")
        if min(train.min(), test.min()) < 0 or max(train.max(), test.max()) >= len(y):
            raise ValueError("CV index outside dataset.")
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise ValueError("Duplicate indices in a CV fold.")
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise ValueError("Every face must have exactly one held-out prediction.")
    prediction, fold_ids = None, np.empty(len(y), dtype=int)
    for fold, (train, test) in enumerate(splits):
        values = np.asarray(fit_predict(x[train], y[train], x[test], fold))
        if values.ndim not in (1, 2, 3) or len(values) != len(test) or not np.isfinite(values).all():
            raise ValueError("Predictor returned invalid held-out values.")
        if prediction is None:
            prediction = np.empty((len(y),) + values.shape[1:], dtype=float)
        if values.shape[1:] != prediction.shape[1:]:
            raise ValueError("Prediction dimensions changed between folds.")
        prediction[test], fold_ids[test] = values, fold
    return prediction, fold_ids


def scale_training(x):
    x = np.asarray(x, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - mean) / scale, mean, scale




def fit_score_model(x, y, kind="ridge", columns=None, alpha=1.0, *,
                    estimator=None, feature_groups=None, sample_weight=None):
    """Fit raw-score linear readouts or a supplied sklearn estimator.

    Scaling and target centering are training-only. feature_groups optionally
    removes each row's mean within feature groups before scaling. A supplied
    estimator is cloned; its constructor owns kernel/optimizer hyperparameters.
    Ridge/linear support multiple outputs; lasso and isotonic are scalar-only.
    """
    from sklearn.base import clone
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Lasso
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim not in (1, 2) or len(x) != len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite aligned X[n,p], y[n] or y[n,targets].")
    if feature_groups is not None: x = center_groups(x, feature_groups, axis=1)
    columns = list(range(x.shape[1])) if columns is None else list(columns)
    state = dict(kind=kind, columns=columns,
                 feature_groups=None if feature_groups is None else np.asarray(feature_groups).tolist())
    if kind == "isotonic" and estimator is None:
        if y.ndim != 1 or len(columns) != 1: raise ValueError("Isotonic needs one feature and one target.")
        model = IsotonicRegression(out_of_bounds="clip").fit(x[:, columns[0]], y, sample_weight=sample_weight)
        return dict(state, x=model.X_thresholds_.tolist(), y=model.y_thresholds_.tolist())
    z, mean, scale = scale_training(x[:, columns])
    target_mean = np.average(y, axis=0, weights=sample_weight)
    centered = y-target_mean
    if estimator is not None:
        model = clone(estimator)
        model.fit(z, centered, **({} if sample_weight is None else {"sample_weight":sample_weight}))
        return dict(state, kind="estimator", estimator=model, mean=mean, scale=scale, target_mean=target_mean)
    if sample_weight is not None:
        raise ValueError("Use a supplied estimator for weighted linear/ridge/lasso fits.")
    if kind == "ridge":
        weights = np.linalg.solve(z.T@z + alpha*np.eye(len(columns)), z.T@centered)
    elif kind == "linear":
        weights = np.linalg.lstsq(z, centered, rcond=None)[0]
    elif kind == "lasso" and y.ndim == 1:
        alpha_max = max(np.max(np.abs(z.T@centered))/len(y), 1e-12)
        weights = Lasso(alpha=alpha*alpha_max, fit_intercept=False, max_iter=50000, tol=1e-7).fit(z, centered).coef_
    else: raise ValueError(f"Unsupported readout {kind!r}; alternatively supply an estimator.")
    raw = weights/(scale[:, None] if weights.ndim == 2 else scale)
    return dict(state, alpha=float(alpha), weights=raw.tolist(),
                intercept=np.asarray(target_mean-mean@raw).tolist(), standardized_weights=weights.tolist())


def predict_score_model(model, x):
    """Predict a fitted readout or a weighted ensemble of such readouts."""
    if "components" in model:
        if len(model["components"]) != len(model["weights"]): raise ValueError("Ensemble weights/components differ.")
        return sum(w*predict_score_model(m,x) for w,m in zip(model["weights"],model["components"]))
    x = np.asarray(x, dtype=float)
    if model.get("feature_groups") is not None: x = center_groups(x, model["feature_groups"], axis=1)
    x = x[:, model["columns"]]
    if model["kind"] == "isotonic": return np.interp(x[:, 0], model["x"], model["y"])
    if model["kind"] == "estimator": return model["estimator"].predict((x-model["mean"])/model["scale"])+model["target_mean"]
    return x@np.asarray(model["weights"])+model["intercept"]


def paired_prediction_bootstrap(y, baseline, prediction, *, samples=2000, seed=20260908, groups=None):
    """Conditional paired bootstrap, optionally sampling whole identity groups.

    Accepts [rows] or [rows, targets]. Each interval is [low, high] or
    [targets, 2]. It does not measure model-refitting/search uncertainty.
    Existing delta keys are retained; rmse gives the prediction's interval.
    """
    y, baseline, prediction = [np.asarray(v, dtype=float) for v in (y, baseline, prediction)]
    if y.ndim not in (1, 2) or not len(y) or y.shape != baseline.shape or y.shape != prediction.shape or samples < 1:
        raise ValueError("Expected aligned targets/predictions and a positive sample count.")
    scalar = y.ndim == 1
    if scalar: y, baseline, prediction = [v[:, None] for v in (y, baseline, prediction)]
    labels = np.arange(len(y)) if groups is None else np.asarray(groups)
    if labels.shape != (len(y),): raise ValueError("groups must align with rows.")
    rows = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    rng = np.random.default_rng(seed)
    values = np.empty((samples, 4, y.shape[1]))
    for i in range(samples):
        index = np.concatenate([rows[j] for j in rng.integers(len(rows), size=len(rows))])
        target, base, pred = y[index], baseline[index], prediction[index]
        bm, pm = np.mean((target-base)**2, axis=0), np.mean((target-pred)**2, axis=0)
        variance = np.var(target, axis=0)
        values[i, 0] = [pearson_correlation(target[:, j], pred[:, j]) - pearson_correlation(target[:, j], base[:, j]) for j in range(y.shape[1])]
        values[i, 1] = np.divide(bm-pm, variance, out=np.full_like(variance, np.nan), where=variance > 0)
        values[i, 2], values[i, 3] = np.sqrt(bm)-np.sqrt(pm), np.sqrt(pm)
    bounds = np.quantile(values, [.025, .975], axis=0).transpose(1, 2, 0)
    return {key: (bounds[j, 0] if scalar else bounds[j]).tolist()
            for j,key in enumerate(("delta_pearson", "delta_r2", "rmse_reduction", "rmse"))}


def extract_impression_features(pipeline, ratings, prompt_bank, output_dir, *, text_mode="short", patches=128):
    """Frozen encoding/cache stage, separate from cheap CPU-only research runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_texts = tuple(dict.fromkeys(t for phrases in prompt_bank.values() for t in phrases))
    scores = pipeline.rate(all_texts, text_mode=text_mode, max_num_patches=patches)
    for variable, phrases in prompt_bank.items():
        indices = [all_texts.index(t) for t in phrases]
        selected = DatasetScores(scores.paths, tuple(phrases), scores.scores[:, indices],
                                 scores.text_mode, tuple(scores.rating_cache_hits[i] for i in indices), scores.runtime)
        data = ratings.align(variable, selected)
        metadata = {"variable": variable, "encoding": scores.runtime,
                    "patches": patches, "ratings_path": str(ratings.path),
                    "ratings_sha256": hashlib.sha256(ratings.path.read_bytes()).hexdigest(),
                    "unmatched_faces": len(scores.paths) - len(data.paths),
                    "feature_identity": pipeline.features(patches).cache_key}
        _atomic_save_npz(output_dir / f"{variable}-features.npz", x=data.predicted_scores,
                         y=data.human_means, counts=data.rater_counts,
                         paths=np.asarray([str(p) for p in data.paths]), texts=np.asarray(data.texts),
                         metadata=np.asarray(json.dumps(metadata)))
    return scores.runtime


def load_impression_features(path):
    """Architecture-neutral exchange format: score matrix + aligned ratings."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        regression = RegressionDataset(metadata["variable"], tuple(Path(p) for p in data["paths"]),
                                       tuple(data["texts"].tolist()), data["y"].copy(),
                                       data["counts"].copy(), data["x"].copy())
    return regression, metadata


def slice_regression(data, indices, columns=None):
    """Preserve face/text order when forming discovery or validation subsets."""
    indices = np.asarray(indices, dtype=int)
    columns = np.arange(len(data.texts)) if columns is None else np.asarray(columns, dtype=int)
    return RegressionDataset(data.variable, tuple(data.paths[i] for i in indices),
                             tuple(data.texts[j] for j in columns), data.human_means[indices],
                             data.rater_counts[indices], data.predicted_scores[np.ix_(indices, columns)])


def anchored_cv_splits(n, development, evaluation, *, folds=10, seed=20260908):
    """Indices into the full dataset; evaluate each non-development face once."""
    development, evaluation = np.asarray(development, dtype=int), np.asarray(evaluation, dtype=int)
    if len(np.unique(np.r_[development, evaluation])) != n or set(np.r_[development, evaluation]) != set(range(n)):
        raise ValueError('Development and evaluation must be disjoint and cover all faces.')
    return [(np.r_[development, evaluation[train]], evaluation[test])
            for train, test in make_cv_splits(len(evaluation), folds, seed)]




def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, str(array.dtype))).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def write_json(path, value):
    """Atomically replace a JSON file; reject nonfinite values."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def merge_groups(*labels):
    """Union row groups when any supplied label set identifies related rows."""
    if not labels or len({len(x) for x in labels}) != 1:
        raise ValueError("Provide aligned group labels.")
    parents = list(range(len(labels[0])))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for keys in labels:
        seen = {}
        for i,key in enumerate(keys):
            if key in seen: parents[find(i)] = find(seen[key])
            else: seen[key] = i
    return np.asarray([str(find(i)) for i in range(len(parents))])


def center_groups(values, groups, *, axis=0):
    """Subtract each group's mean along rows (identities) or columns (banks)."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    out = np.moveaxis(values, axis, 0).copy()
    if groups.shape != (len(out),): raise ValueError("Group labels must match the selected axis.")
    for group in np.unique(groups):
        mask = groups == group
        out[mask] -= out[mask].mean(axis=0)
    return np.moveaxis(out, 0, axis)


def evaluate_predictors(x, y, predictors, *, targets, splits=None, folds=10, seed=20260908, groups=None):
    """Evaluate {name: fit_predict(X_train,y_train,X_test,fold)} on shared folds.

    Callbacks own training-only preprocessing/tuning. Returns RegressionResults,
    directly usable with summary(), metric_table(), and Viz.oof_scatter().
    """
    splits = list(splits) if splits is not None else make_cv_splits(len(y), folds, seed, groups)
    predictions = {}
    for name, callback in predictors.items():
        predictions[name], fold_ids = cross_validate_predictor(x, y, callback, splits=splits)
    if not predictions: raise ValueError("Provide at least one predictor.")
    return RegressionResults(y, predictions, tuple(targets), fold_ids)


class ImpressionPredictor:
    """Any frozen phrase readout with a common, trusted-local bundle contract.

    Required keys: phrases, patches, encoding (model/revision/dtype), targets,
    readout. Custom readout_predict(model, scores) supports specialized kernels.
    """
    def __init__(self, bundle, *, readout_predict=predict_score_model):
        self.bundle = bundle
        self.targets = tuple(bundle["targets"])
        if not self.targets or not bundle["phrases"] or not bundle["patches"]:
            raise ValueError("Bundle requires targets, phrases and patch budgets.")
        self._predict = readout_predict
        self._engine = self._text_features = None

    @classmethod
    def load(cls, path):
        import joblib
        return cls(joblib.load(Path(path)))

    def predict_scores(self, scores):
        scores = np.asarray(scores, float)
        width = len(self.bundle["phrases"])*len(self.bundle["patches"])
        if scores.ndim != 2 or not len(scores) or scores.shape[1] != width or not np.isfinite(scores).all():
            raise ValueError(f"Expected finite scores [faces, {width}], ordered by patch bank then phrase.")
        return self._predict(self.bundle["readout"], scores)

    @torch.inference_mode()
    def predict_images(self, paths, *, engine=None, batch_size=16, return_scores=False, device="mps"):
        paths = list(paths)
        if not paths or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Provide images and a positive integer batch size.")
        expected = self.bundle["encoding"]
        engine = engine or self._engine
        if engine is None:
            engine = FGCLIP2(expected["model"], revision=expected["revision"], device=device,
                             dtype=getattr(torch, expected["dtype"].removeprefix("torch.")), local_files_only=True)
        if (engine.model_id, engine.revision, str(engine.dtype)) != (expected["model"], expected["revision"], expected["dtype"]):
            raise ValueError("Encoder model/revision/dtype differs from training.")
        if engine is not self._engine or self._text_features is None:
            phrases = self.bundle["phrases"]
            self._text_features = torch.cat([engine.encode_text(phrases[i:i+32], mode=expected.get("text_mode", "short")).cpu()
                                             for i in range(0,len(phrases),32)])
            self._engine = engine
        scale, bias = engine.model.logit_scale.float().exp().item(), engine.model.logit_bias.float().item()
        banks = []
        for patches in self.bundle["patches"]:
            pieces = []
            for i in range(0,len(paths),batch_size):
                features = engine.encode_image_preprocessed(engine.prepare_images(paths[i:i+batch_size], max_num_patches=patches)).cpu()
                pieces.append((features@self._text_features.T*scale+bias).numpy())
            banks.append(np.concatenate(pieces))
        scores = np.column_stack(banks)
        prediction = self.predict_scores(scores)
        return (prediction,scores) if return_scores else prediction

    def predict_directory(self, directory, output):
        import csv
        paths = discover_face_images(FaceDatasetConfig(Path(directory), limit=None))
        predictions = self.predict_images(paths)
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["face", *[f"predicted_{t}_mean" for t in self.targets]])
            writer.writerows([p.name,*row] for p,row in zip(paths,np.asarray(predictions).reshape(len(paths),-1)))
        return predictions


__all__ = [
    'CorrelationMetrics',
    'DatasetScores',
    'FaceDatasetConfig',
    'FaceFeatureSet',
    'FaceImpressionPipeline',
    'HumanRatingVector',
    'HumanRatingsStore',
    'RegressionDataset', 'RegressionResults', 'Viz',
    'ImpressionPredictor', 'evaluate_predictors', 'center_groups',
    'scale_training', 'array_hash', 'write_json', 'merge_groups',
    'DEFAULT_IMAGE_SUFFIXES',
    'average_ranks',
    'centered_distances',
    'correlate_regression_data',
    'discover_face_images',
    'distance_correlation',
    'pearson_correlation',
    'cross_validate_predictor',
    'make_cv_splits',
    'regression_metrics',
    'fit_score_model',
    'predict_score_model',
    'paired_prediction_bootstrap',
    'extract_impression_features',
    'load_impression_features',
    'slice_regression',
    'anchored_cv_splits',
]

