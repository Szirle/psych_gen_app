"""Standalone local FG-CLIP 2 lab; all UI/plot/export code lives in this file.

Install with the chosen Python environment:
    python -m pip install transformers==4.57.6 gradio==5.49.1 sentencepiece accelerate psutil

Run (safe default: Base, half precision on MPS, one queued request at a time):
    python CLIP/fgclip2_demo.py
    python CLIP/fgclip2_demo.py --model qihoo360/fg-clip2-so400m --dtype float16
    python CLIP/fgclip2_demo.py --offline --port 7860
    python CLIP/fgclip2_demo.py --self-test --test-image /path/to/face.png

The core still defaults to So400m. This UI defaults to Base for 16 GB Macs.
Set FGCLIP2_MODEL or --model to choose another checkpoint. Weights live in the
project's models/fgclip2, regardless of the current working directory. No Flutter
or other app code is used. Importing this file neither loads weights nor starts
a server. There is no public share link. Ctrl-C stops the local server.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageOps

try:
    from .fgclip2_core import BASE_MODEL_ID, DEFAULT_CACHE_DIR, FGCLIP2
    from .fgclip2_face_impressions import (
        FaceDatasetConfig,
        FaceImpressionPipeline,
        HumanRatingsStore,
        correlate_regression_data,
    )
except ImportError:
    from fgclip2_core import BASE_MODEL_ID, DEFAULT_CACHE_DIR, FGCLIP2
    from fgclip2_face_impressions import (
        FaceDatasetConfig,
        FaceImpressionPipeline,
        HumanRatingsStore,
        correlate_regression_data,
    )

_ENGINE = None
_FACE_PIPELINE = None
_HUMAN_RATINGS_STORE = None
_LOCK = threading.RLock()
_OPTIONS = {"model_id": os.environ.get("FGCLIP2_MODEL", BASE_MODEL_ID)}
_DATASET_OPTIONS = {
    "directory": Path(
        os.environ.get(
            "FGCLIP2_DATASET_DIR",
            "/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images",
        )
    ).expanduser(),
    "limit": int(os.environ.get("FGCLIP2_DATASET_LIMIT", "1000")),
    "batch_size": int(os.environ.get("FGCLIP2_DATASET_BATCH_SIZE", "64")),
    "cache_dir": DEFAULT_CACHE_DIR / "dataset-rankings",
}
_HUMAN_RATINGS = {
    "path": Path(
        os.environ.get(
            "FGCLIP2_HUMAN_RATINGS",
            Path(__file__).resolve().parents[1]
            / "data"
            / "dim_to_photo_to_ratings.pkl",
        )
    ).expanduser()
}
LOGGER = logging.getLogger(__name__)


def get_engine() -> FGCLIP2:
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = FGCLIP2(**_OPTIONS)
        return _ENGINE


def get_face_pipeline() -> FaceImpressionPipeline:
    global _FACE_PIPELINE
    with _LOCK:
        if _FACE_PIPELINE is None:
            _FACE_PIPELINE = FaceImpressionPipeline(
                get_engine(),
                FaceDatasetConfig(
                    directory=_DATASET_OPTIONS["directory"],
                    limit=int(_DATASET_OPTIONS["limit"]),
                    batch_size=int(_DATASET_OPTIONS["batch_size"]),
                    cache_dir=_DATASET_OPTIONS["cache_dir"],
                ),
            )
        return _FACE_PIPELINE


def get_human_ratings_store() -> HumanRatingsStore:
    global _HUMAN_RATINGS_STORE
    with _LOCK:
        if _HUMAN_RATINGS_STORE is None:
            _HUMAN_RATINGS_STORE = HumanRatingsStore(_HUMAN_RATINGS["path"])
        return _HUMAN_RATINGS_STORE


def reset_face_impression_helpers():
    global _FACE_PIPELINE, _HUMAN_RATINGS_STORE
    _FACE_PIPELINE = None
    _HUMAN_RATINGS_STORE = None


def action(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _LOCK:
            try:
                return fn(*args, **kwargs)
            except (ValueError, OSError, RuntimeError, MemoryError) as exc:
                LOGGER.exception("FG-CLIP action failed")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                raise gr.Error(str(exc)) from exc

    return wrapped


def require_image(image):
    if image is None:
        raise ValueError("Upload an image first.")
    return FGCLIP2._as_pil(image)


def descriptions(text, limit=16):
    lines = [s.strip() for s in (text or "").splitlines() if s.strip()]
    if not lines:
        raise ValueError("Enter at least one description.")
    if len(lines) > limit:
        raise ValueError(f"Use at most {limit} descriptions, one per line.")
    return lines


def one_text(text):
    if not text or not text.strip():
        raise ValueError("Enter a description.")
    return text.strip()


def _gallery_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@action
def rank_dataset_images(text, mode="auto", patches=128):
    text = one_text(text)
    dataset_scores = get_face_pipeline().rate(
        [text], text_mode=mode, max_num_patches=int(patches)
    )
    paths = dataset_scores.paths
    if len(paths) < 8:
        raise ValueError("The face dataset must contain at least 8 images.")
    scores = dataset_scores.scores[:, 0]
    details = dataset_scores.details()

    order = np.argsort(scores)
    lowest = order[:4].tolist()
    highest = order[-4:][::-1].tolist()
    selected = highest + lowest
    gallery = [
        (
            _gallery_image(paths[index]),
            (
                f"Highest · {paths[index].name} · {scores[index]:+.4f}"
                if position < 4
                else f"Lowest · {paths[index].name} · {scores[index]:+.4f}"
            ),
        )
        for position, index in enumerate(selected)
    ]
    details.update(
        description=text,
        rating_cache_hit=details["rating_cache_hits"] == 1,
        score_min=float(scores[order[0]]),
        score_max=float(scores[order[-1]]),
    )
    return gallery, details


def _human_rating_variables():
    return list(get_human_ratings_store().variables)


def _correlation_bar_plot(rows):
    fig = Figure(
        figsize=(max(8, len(rows) * 1.15), 4.8),
        layout="constrained",
    )
    axis = fig.subplots()
    positions = np.arange(len(rows))
    width = 0.25
    axis.bar(
        positions - width,
        [row["pearson"] for row in rows],
        width,
        label="Pearson r",
    )
    axis.bar(
        positions,
        [row["spearman"] for row in rows],
        width,
        label="Spearman ρ",
    )
    axis.bar(
        positions + width,
        [row["distance_correlation"] for row in rows],
        width,
        label="Distance correlation",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Correlation")
    axis.set_xticks(
        positions,
        [row["phrase"][:32] for row in rows],
        rotation=25,
        ha="right",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return fig


def _correlation_scatter_plot(variable, human_means, scores, rows):
    best_index = int(np.nanargmax([abs(row["pearson"]) for row in rows]))
    phrase = rows[best_index]["phrase"]
    predicted = scores[:, best_index]
    fig = Figure(figsize=(6.5, 5), layout="constrained")
    axis = fig.subplots()
    axis.scatter(human_means, predicted, alpha=0.35, s=18)
    if np.std(human_means) > 0:
        slope, intercept = np.polyfit(human_means, predicted, 1)
        line_x = np.linspace(human_means.min(), human_means.max(), 100)
        axis.plot(line_x, slope * line_x + intercept, color="#0f9d8a", linewidth=2)
    axis.set_xlabel(f"Mean human rating · {variable}")
    axis.set_ylabel("FGCLIP agreement logit")
    axis.set_title(f"Strongest |Pearson| phrase: {phrase}", wrap=True)
    axis.grid(alpha=0.2)
    return fig


@action
def correlate_human_ratings(text, variable, mode="auto", patches=128):
    texts = descriptions(text)
    dataset_scores = get_face_pipeline().rate(
        texts, text_mode=mode, max_num_patches=int(patches)
    )
    regression = get_human_ratings_store().align(variable, dataset_scores)
    rows = [result.as_dict() for result in correlate_regression_data(regression)]
    human = regression.human_means
    scores = regression.predicted_scores

    best_linear = max(rows, key=lambda row: abs(row["pearson"]))
    best_nonlinear = max(rows, key=lambda row: row["distance_correlation"])
    summary = (
        f"### Human variable: `{variable}`\n"
        f"Strongest linear relationship: **{best_linear['phrase']}** "
        f"(Pearson r = **{best_linear['pearson']:+.3f}**).  \n"
        f"Strongest general dependence: **{best_nonlinear['phrase']}** "
        f"(distance correlation = **{best_nonlinear['distance_correlation']:.3f}**)."
    )
    table = [
        [
            row["phrase"],
            row["images"],
            round(row["pearson"], 4),
            round(row["spearman"], 4),
            round(row["distance_correlation"], 4),
        ]
        for row in rows
    ]
    details = {
        "human_variable": variable,
        "human_ratings_file": str(get_human_ratings_store().path),
        "matched_images": len(regression.paths),
        "median_raters_per_image": float(np.median(regression.rater_counts)),
        "human_mean_min": float(human.min()),
        "human_mean_max": float(human.max()),
        "metrics": {
            "pearson": "Signed linear correlation.",
            "spearman": "Signed rank correlation; captures monotonic nonlinear relationships.",
            "distance_correlation": "Unsigned dependence score from 0 to 1; captures non-monotonic relationships.",
        },
        "results": rows,
        **dataset_scores.details(),
    }
    return (
        summary,
        table,
        _correlation_bar_plot(rows),
        _correlation_scatter_plot(variable, human, scores, rows),
        details,
    )


def metadata(engine, texts, mode):
    return {
        "model": engine.model_id,
        "revision": engine.revision,
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "text": engine.text_info(texts, mode=mode),
    }


@action
def model_status():
    e = get_engine()
    return {
        **metadata(e, ["example"], "short"),
        "checkpoint": str(e.model_path),
        "mps_allocated_GiB": round(torch.mps.driver_allocated_memory() / 2**30, 3)
        if e.device.type == "mps"
        else None,
    }


@action
def single_score(image, text, mode="auto", patches=128):
    image, text = require_image(image), one_text(text)
    e = get_engine()
    r = e.agreement([image], [text], text_mode=mode, max_num_patches=int(patches))
    return {
        "agreement_logit": float(r.logits[0, 0]),
        "cosine_similarity": float(r.cosine[0, 0]),
        "energy": float(r.energy[0, 0]),
        **metadata(e, [text], mode),
    }


@action
def compare_images(image_a, image_b, text, mode="auto", patches=128):
    images, text = [require_image(image_a), require_image(image_b)], one_text(text)
    e = get_engine()
    r = e.agreement(images, [text], text_mode=mode, max_num_patches=int(patches))
    a, b = r.logits[:, 0].tolist()
    margin = a - b
    winner = "Tie" if abs(margin) < 1e-6 else "Image A" if margin > 0 else "Image B"
    return f"### {winner}\nA − B agreement margin: **{margin:+.4f}**", {
        "A_logit": a,
        "B_logit": b,
        "A_minus_B": margin,
        "A_cosine": float(r.cosine[0, 0]),
        "B_cosine": float(r.cosine[1, 0]),
        **metadata(e, [text], mode),
    }


@action
def rank_descriptions(image, text, mode="auto", patches=128):
    image, texts = require_image(image), descriptions(text)
    e = get_engine()
    r = e.agreement([image], texts, text_mode=mode, max_num_patches=int(patches))
    indices = r.logits[0].argsort(descending=True).tolist()
    rows = [
        [texts[i], float(r.cosine[0, i]), float(r.logits[0, i]), float(r.energy[0, i])]
        for i in indices
    ]
    return rows, metadata(e, texts, mode)


def plot_maps(image, maps, labels, opacity, *, difference=False):
    """Only visualization rescales colors; the underlying cosine maps stay raw."""
    fig = Figure(figsize=(5 * len(maps), 5), layout="constrained")
    axes = fig.subplots(1, len(maps), squeeze=False)[0]
    unsigned_maps = maps[:-1] if difference else maps
    shared_min = min(float(np.min(m)) for m in unsigned_maps)
    shared_max = max(float(np.max(m)) for m in unsigned_maps)
    for i, (ax, heat, label) in enumerate(zip(axes, maps, labels)):
        signed = difference and i == len(maps) - 1
        if signed:
            extent = max(float(np.abs(heat).max()), 1e-6)
            vmin, vmax, cmap = -extent, extent, "coolwarm"
        else:
            vmin, vmax, cmap = shared_min, max(shared_max, shared_min + 1e-6), "viridis"
        up = np.asarray(
            Image.fromarray(heat.astype(np.float32)).resize(
                image.size, Image.Resampling.BILINEAR
            )
        )
        ax.imshow(image)
        im = ax.imshow(up, alpha=float(opacity), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label[:70], wrap=True)
        ax.set_axis_off()
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.02,
            label="Δ cosine" if signed else "cosine",
        )
    return fig


def export_maps(maps, labels, engine, image, patches, *, texts=None):
    info = {
        **metadata(engine, texts or labels, "box"),
        "image_size": image.size,
        "max_num_patches": int(patches),
        "grid_size": maps[0].shape,
        "labels": labels,
    }
    # Gradio copies returned exports into its own expiring cache.
    for old in Path(tempfile.gettempdir()).glob("fgclip2-*.npz"):
        if old.stat().st_mtime < time.time() - 3600:
            old.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", prefix="fgclip2-", delete=False
    ) as f:
        np.savez_compressed(f, similarity=np.stack(maps), metadata=json.dumps(info))
        return f.name


@action
def dense_heatmap(image, text, opacity=0.55, patches=128):
    image, texts = require_image(image), descriptions(text, limit=4)
    e = get_engine()
    maps = [
        m.similarity.numpy()
        for m in e.dense_alignments(image, texts, max_num_patches=int(patches))
    ]
    stats = {
        "maps": [
            {
                "description": t,
                "min": float(m.min()),
                "max": float(m.max()),
                "mean": float(m.mean()),
            }
            for t, m in zip(texts, maps)
        ],
        "grid_size": maps[0].shape,
        **metadata(e, texts, "box"),
    }
    return (
        plot_maps(image, maps, texts, opacity),
        stats,
        export_maps(maps, texts, e, image, patches),
    )


@action
def contrast(image, positive, negative, mode="auto", patches=128, opacity=0.55):
    image, positive, negative = (
        require_image(image),
        one_text(positive),
        one_text(negative),
    )
    e = get_engine()
    texts = [positive, negative]
    # Validate both heads before doing either image forward.
    e.prepare_text(texts, mode="box")
    r = e.agreement([image], texts, text_mode=mode, max_num_patches=int(patches))
    a, b = [
        m.similarity.numpy()
        for m in e.dense_alignments(image, texts, max_num_patches=int(patches))
    ]
    margin = float(r.logits[0, 0] - r.logits[0, 1])
    maps = [a, b, a - b]
    labels = [positive, negative, "Positive − negative"]
    fig = plot_maps(image, maps, labels, opacity, difference=True)
    data = {
        "positive_logit": float(r.logits[0, 0]),
        "negative_logit": float(r.logits[0, 1]),
        "attribute_margin": margin,
        "contrastive_energy": -margin,
        **metadata(e, texts, mode),
    }
    return data, fig, export_maps(maps, labels, e, image, patches, texts=texts)


@action
def score_region(image, text, x1, y1, x2, y2, patches=128):
    image, texts = require_image(image), descriptions(text)
    if not (0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100):
        raise ValueError(
            "Region coordinates must satisfy 0 ≤ left < right ≤ 100 and 0 ≤ top < bottom ≤ 100."
        )
    box = [
        x1 / 100 * image.width,
        y1 / 100 * image.height,
        x2 / 100 * image.width,
        y2 / 100 * image.height,
    ]
    e = get_engine()
    scores = e.region_alignment(image, [box], texts, max_num_patches=int(patches))[0]
    indices = scores.argsort(descending=True).tolist()
    preview = image.copy()
    ImageDraw.Draw(preview).rectangle(
        box, outline="#00d6af", width=max(2, image.width // 150)
    )
    return (
        preview,
        [[texts[i], float(scores[i])] for i in indices],
        {"box_xyxy_pixels": box, "image_size": image.size, **metadata(e, texts, "box")},
    )


CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
#hero {padding: 24px 10px 12px;}
#hero h1 {font-size: 36px; letter-spacing: -1.2px; color: var(--body-text-color);}
#hero p {font-size: 16px; color: var(--body-text-color-subdued); max-width: 850px;}
footer {display: none !important;}
"""


def build_demo():
    human_variables = _human_rating_variables()
    with gr.Blocks(
        title="FG-CLIP 2 · Face & attribute lab",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=CSS,
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as demo:
        gr.Markdown(
            "# Face & attribute lab\nExplore continuous agreement, compare photographs, and inspect where descriptions align with an image.",
            elem_id="hero",
        )
        gr.Markdown(
            f"**FG-CLIP 2** · `{_OPTIONS['model_id']}` · local inference · English & Chinese"
        )
        with gr.Row():
            mode = gr.Dropdown(
                ["auto", "short", "long"],
                value="auto",
                label="Global text head",
                info="Auto uses token count. Long: up to 196 tokens; short: 64.",
            )
            patches = gr.Dropdown(
                [64, 128, 256, 576, 1024],
                value=128,
                label="Image patch budget",
                info="128 is the low-memory default. Higher values cost more memory; dense attention grows quadratically.",
            )
        gr.Markdown(
            "Larger logits mean stronger agreement; energy is −logit. These are **not calibrated probabilities**. Dense maps show semantic similarity, not causal explanations or verified attribute labels."
        )

        def bind(button, fn, inputs, outputs, name):
            button.click(
                fn,
                inputs,
                outputs,
                api_name=name,
                concurrency_id="model",
                concurrency_limit=1,
            )

        with gr.Tab("Agreement"):
            with gr.Row():
                im = gr.Image(
                    type="pil", label="Photograph", sources=["upload", "clipboard"]
                )
                with gr.Column():
                    text = gr.Textbox(
                        label="Description",
                        placeholder="A person with a subtle closed-mouth smile",
                        lines=4,
                    )
                    run = gr.Button("Measure agreement", variant="primary")
                    out = gr.JSON(label="Agreement & energy")
            bind(run, single_score, [im, text, mode, patches], out, "agreement")

        with gr.Tab("Compare images"):
            with gr.Row():
                a = gr.Image(
                    type="pil", label="Image A", sources=["upload", "clipboard"]
                )
                b = gr.Image(
                    type="pil", label="Image B", sources=["upload", "clipboard"]
                )
            text = gr.Textbox(
                label="Shared description",
                placeholder="A face with pronounced cheekbones",
            )
            run = gr.Button("Compare A and B", variant="primary")
            winner, out = gr.Markdown(), gr.JSON(label="Independent scores")
            bind(
                run,
                compare_images,
                [a, b, text, mode, patches],
                [winner, out],
                "compare",
            )

        with gr.Tab("Rank descriptions"):
            with gr.Row():
                im = gr.Image(
                    type="pil", label="Photograph", sources=["upload", "clipboard"]
                )
                text = gr.Textbox(
                    label="Descriptions · one per line",
                    placeholder="A smiling person\nA person with a neutral expression\nA person wearing glasses",
                    lines=6,
                )
            run = gr.Button("Rank descriptions", variant="primary")
            table = gr.Dataframe(
                headers=["Description", "Cosine", "Logit", "Energy"], interactive=False
            )
            out = gr.JSON(label="Run details")
            bind(
                run, rank_descriptions, [im, text, mode, patches], [table, out], "rank"
            )

        with gr.Tab("Rank face dataset"):
            gr.Markdown(
                "Rate the configured face dataset against one phrase. The first run batch-encodes image features; later phrases reuse those features, and repeated ratings are loaded from disk."
            )
            text = gr.Textbox(
                label="Description",
                placeholder="A person with a subtle closed-mouth smile",
                lines=2,
            )
            run = gr.Button("Rate dataset", variant="primary")
            gallery = gr.Gallery(
                label="Four highest (top) and four lowest (bottom)",
                columns=4,
                rows=2,
                object_fit="cover",
                height=650,
                allow_preview=True,
            )
            out = gr.JSON(label="Dataset run details")
            bind(
                run,
                rank_dataset_images,
                [text, mode, patches],
                [gallery, out],
                "rank_dataset",
            )

        with gr.Tab("Human rating correlations"):
            gr.Markdown(
                "Compare FGCLIP predictions for multiple phrases with mean human ratings. Pearson measures linear correlation, Spearman measures monotonic association, and distance correlation detects broader nonlinear dependence."
            )
            with gr.Row():
                text = gr.Textbox(
                    label="Descriptions · one per line",
                    placeholder="an older-looking face\na youthful-looking face",
                    lines=6,
                )
                variable = gr.Dropdown(
                    human_variables,
                    value="age" if "age" in human_variables else human_variables[0],
                    label="Human ratings variable",
                )
            run = gr.Button("Calculate correlations", variant="primary")
            summary = gr.Markdown()
            table = gr.Dataframe(
                headers=[
                    "Description",
                    "Matched images",
                    "Pearson r",
                    "Spearman ρ",
                    "Distance correlation",
                ],
                interactive=False,
            )
            with gr.Row():
                correlation_plot = gr.Plot(label="Correlation comparison")
                scatter_plot = gr.Plot(label="Strongest linear relationship")
            out = gr.JSON(label="Analysis details")
            bind(
                run,
                correlate_human_ratings,
                [text, variable, mode, patches],
                [summary, table, correlation_plot, scatter_plot, out],
                "human_rating_correlations",
            )

        with gr.Tab("Dense alignment"):
            gr.Markdown(
                "Enter up to four local descriptors, one per line (64 tokens each). Maps share a color scale. Download the original patch values for analysis."
            )
            with gr.Row():
                im = gr.Image(
                    type="pil", label="Photograph", sources=["upload", "clipboard"]
                )
                with gr.Column():
                    text = gr.Textbox(
                        label="Local descriptors",
                        placeholder="mouth\neyebrows\nhair",
                        lines=4,
                    )
                    alpha = gr.Slider(0, 1, 0.55, step=0.05, label="Overlay opacity")
                    run = gr.Button("Map alignment", variant="primary")
            plot, out, file = (
                gr.Plot(label="Native dense similarities"),
                gr.JSON(label="Raw map statistics"),
                gr.File(label="Raw maps (.npz)"),
            )
            bind(
                run,
                dense_heatmap,
                [im, text, alpha, patches],
                [plot, out, file],
                "dense",
            )

        with gr.Tab("Attribute contrast"):
            gr.Markdown(
                "Subtract two descriptions to inspect relative evidence. Blue in the difference map favors the negative description; red favors the positive."
            )
            with gr.Row():
                im = gr.Image(
                    type="pil", label="Photograph", sources=["upload", "clipboard"]
                )
                with gr.Column():
                    positive = gr.Textbox(
                        label="Positive description", placeholder="a smiling face"
                    )
                    negative = gr.Textbox(
                        label="Negative description", placeholder="a neutral face"
                    )
                    alpha = gr.Slider(0, 1, 0.55, step=0.05, label="Overlay opacity")
                    run = gr.Button("Measure attribute contrast", variant="primary")
            out, plot, file = (
                gr.JSON(label="Signed agreement"),
                gr.Plot(label="Positive, negative & difference"),
                gr.File(label="Raw maps (.npz)"),
            )
            bind(
                run,
                contrast,
                [im, positive, negative, mode, patches, alpha],
                [out, plot, file],
                "contrast",
            )

        with gr.Tab("Region scoring"):
            gr.Markdown(
                "Score a rectangle using the model's dense region features. Coordinates are percentages of the oriented image; the preview marks the exact region."
            )
            with gr.Row():
                im = gr.Image(
                    type="pil", label="Photograph", sources=["upload", "clipboard"]
                )
                text = gr.Textbox(
                    label="Region descriptions · one per line",
                    placeholder="smiling mouth\nclosed mouth",
                    lines=4,
                )
            with gr.Row():
                coords = [
                    gr.Number(value=v, minimum=0, maximum=100, label=label)
                    for v, label in zip(
                        [0, 0, 100, 100], ["Left %", "Top %", "Right %", "Bottom %"]
                    )
                ]
            run = gr.Button("Score region", variant="primary")
            preview = gr.Image(label="Scored rectangle")
            table, out = (
                gr.Dataframe(
                    headers=["Description", "Region cosine"], interactive=False
                ),
                gr.JSON(label="Region details"),
            )
            bind(
                run,
                score_region,
                [im, text, *coords, patches],
                [preview, table, out],
                "region",
            )

        with gr.Accordion("Model status & Python API", open=False):
            gr.Markdown(
                "The first request loads the selected model. Missing files download automatically; complete cached weights work offline. The application processes one request at a time."
            )
            load, status = (
                gr.Button("Load model / show status"),
                gr.JSON(label="Runtime"),
            )
            bind(load, model_status, [], status, "status")
            gr.Code(
                'from CLIP.fgclip2_core import FGCLIP2\nfg = FGCLIP2("qihoo360/fg-clip2-base", device="mps")\n# RGB tensor: [B,3,H,W], floating values in [0,1]\nscores = fg.score_tensor(rgb, ["a smiling face"], max_num_patches=128)\nenergy = -scores.sum()\nenergy.backward()',
                language="python",
            )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def self_test(image_path=None):
    """Small real-checkpoint regression test; no server, no CPU inference."""
    e = get_engine()
    if e.device.type != "mps":
        raise RuntimeError("This local validation is configured for MPS only.")
    image = (
        require_image(image_path) if image_path else Image.new("RGB", (96, 80), "red")
    )
    texts = (
        ["a smiling person", "a neutral face"]
        if image_path
        else ["a red square", "a blue square"]
    )
    inputs = e.prepare_image(image, max_num_patches=64)
    with torch.inference_mode():
        ours = e.agreement_preprocessed(inputs, texts)
        raw_i = e.model.get_image_features(**inputs).float()
        ti, walk = e.prepare_text(texts)
        raw_t = e.model.get_text_features(**ti, walk_type=walk).float()
        cosine = (
            torch.nn.functional.normalize(raw_i, dim=-1)
            @ torch.nn.functional.normalize(raw_t, dim=-1).T
        )
        assert torch.allclose(ours.cosine, cosine, atol=2e-5)
        assert torch.allclose(
            ours.logits,
            cosine * e.model.logit_scale.float().exp() + e.model.logit_bias.float(),
            atol=2e-5,
        )
    report = {
        "model": e.model_id,
        "dtype": str(e.dtype),
        "device": str(e.device),
        "cosine": ours.cosine.tolist(),
    }
    del ours, raw_i, raw_t, cosine, ti, inputs
    torch.mps.empty_cache()
    print("PASS official global score parity", flush=True)
    with torch.inference_mode():
        long_features = e.encode_text(["a photograph of a person " * 16], mode="long")
        chinese_features = e.encode_text(["微笑的人"], mode="short")
        assert (
            torch.isfinite(long_features).all()
            and torch.isfinite(chinese_features).all()
        )
    del long_features, chinese_features
    maps = e.dense_alignments(image, texts, max_num_patches=64)
    assert all(torch.isfinite(m.similarity).all() for m in maps)
    difference = e.dense_difference(image, texts[0], texts[0], max_num_patches=64)
    assert torch.count_nonzero(difference.similarity) == 0
    region = e.region_alignment(
        image, [[0, 0, image.width, image.height]], texts, max_num_patches=64
    )
    assert region.shape == (1, 2) and torch.isfinite(region).all()
    # Full-image aligned RoI with adaptive 1x1 pooling samples every patch center.
    with torch.inference_mode():
        region_inputs = e.prepare_image(image, max_num_patches=64)
        dense_raw = e.model.get_image_dense_feature(**region_inputs).float()
        h, w = region_inputs["spatial_shapes"][0].tolist()
        expected_region = torch.nn.functional.normalize(
            dense_raw[:, : h * w].mean(1), dim=-1
        )
        actual_region = e.encode_regions_preprocessed(
            region_inputs,
            [[[0, 0, image.width, image.height]]],
            [(image.height, image.width)],
        )[0]
        assert torch.allclose(expected_region, actual_region, atol=3e-4)
    del region_inputs, dense_raw, expected_region, actual_region
    print("PASS dense maps, identical-text difference and region pooling", flush=True)
    del maps, difference, region
    torch.mps.empty_cache()
    rgb = (
        torch.tensor(
            np.asarray(image.resize((64, 64))).copy(), device="mps", dtype=torch.float32
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255
    )
    rgb.requires_grad_(True)
    score = e.score_tensor(rgb, texts[:1], max_num_patches=16).sum()
    grad = torch.autograd.grad(score, rgb)[0]
    assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0
    report["gradient_max"] = float(grad.abs().max())
    print("PASS raw RGB -> score gradients", flush=True)
    del score, grad
    # The dense and region tensor APIs must also preserve the input graph.
    tensor_inputs = e.prepare_image_tensor(rgb, max_num_patches=16)
    dense_score = e.dense_alignment_preprocessed(tensor_inputs, texts[:1])[0].sum()
    dense_grad = torch.autograd.grad(dense_score, rgb)[0]
    assert torch.isfinite(dense_grad).all() and float(dense_grad.abs().max()) > 0
    del tensor_inputs, dense_score, dense_grad
    tensor_inputs = e.prepare_image_tensor(rgb, max_num_patches=16)
    region_score = e.region_alignment_preprocessed(
        tensor_inputs, [[[8, 8, 56, 56]]], [(64, 64)], texts[:1]
    )[0].sum()
    region_grad = torch.autograd.grad(region_score, rgb)[0]
    assert torch.isfinite(region_grad).all() and float(region_grad.abs().max()) > 0
    del tensor_inputs, region_score, region_grad, rgb
    torch.mps.empty_cache()
    assert e.text_info(["hello " * 80])["mode"] == "long"
    try:
        e.prepare_text(["hello " * 250])
        raise AssertionError("Expected overlong text rejection")
    except ValueError:
        pass
    # Exercise every UI callback with small patch budgets.
    assert np.isfinite(single_score(image, texts[0], "short", 64)["agreement_logit"])
    assert (
        abs(compare_images(image, image, texts[0], "short", 64)[1]["A_minus_B"]) < 1e-5
    )
    assert len(rank_descriptions(image, "\n".join(texts), "short", 64)[0]) == 2
    for output in [
        dense_heatmap(image, texts[0], 0.5, 64),
        contrast(image, texts[0], texts[1], "short", 64, 0.5),
    ]:
        with np.load(output[-1]) as export:
            assert np.isfinite(export["similarity"]).all()
        Path(output[-1]).unlink()
    assert len(score_region(image, "\n".join(texts), 20, 20, 80, 80, 64)[1]) == 2
    old_dataset_options = _DATASET_OPTIONS.copy()
    with tempfile.TemporaryDirectory(prefix="fgclip2-self-test-") as directory:
        directory = Path(directory)
        for index in range(8):
            image.save(directory / f"{index + 1}.png")
        _DATASET_OPTIONS.update(
            directory=directory,
            limit=8,
            batch_size=4,
            cache_dir=directory / "cache",
        )
        reset_face_impression_helpers()
        try:
            first_gallery, first_rank = rank_dataset_images(texts[0], "short", 64)
            second_gallery, second_rank = rank_dataset_images(texts[0], "short", 64)
            assert len(first_gallery) == len(second_gallery) == 8
            assert (
                not first_rank["rating_cache_hit"] and second_rank["rating_cache_hit"]
            )
        finally:
            _DATASET_OPTIONS.clear()
            _DATASET_OPTIONS.update(old_dataset_options)
            reset_face_impression_helpers()
    report["checks"] = (
        "official score parity; long/Chinese text; dense; difference; region pooling; global/dense/region RGB gradients; text limits; core UI callbacks; NPZ exports and dataset caches"
    )
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=os.environ.get("FGCLIP2_MODEL", BASE_MODEL_ID)
    )
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_OPTIONS["directory"],
        help="Directory containing face images for the dataset-ranking tab.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=_DATASET_OPTIONS["limit"],
        help="Natural-sort prefix to rate (default: 1000).",
    )
    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        default=_DATASET_OPTIONS["batch_size"],
        help="Initial image inference batch size; halves automatically after OOM.",
    )
    parser.add_argument(
        "--human-ratings",
        type=Path,
        default=_HUMAN_RATINGS["path"],
        help="Pickle mapping rating dimensions to per-photo rating lists.",
    )
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test-image")
    args = parser.parse_args()
    _OPTIONS.update(
        model_id=args.model,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.offline,
    )
    _DATASET_OPTIONS.update(
        directory=args.dataset_dir,
        limit=args.dataset_limit,
        batch_size=args.dataset_batch_size,
        cache_dir=args.cache_dir / "dataset-rankings",
    )
    _HUMAN_RATINGS["path"] = args.human_ratings
    if args.self_test:
        self_test(args.test_image)
    else:
        build_demo().launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
            show_error=True,
            max_file_size="20mb",
        )


if __name__ == "__main__":
    main()
