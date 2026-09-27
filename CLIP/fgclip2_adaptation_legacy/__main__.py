"""Isolated research CLI. See README.md for protocols and runnable examples."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..fgclip2_core import FGCLIP2, DEFAULT_MODEL_ID, BASE_MODEL_ID
from ..fgclip2_face_impressions import (
    FaceDatasetConfig, FaceImpressionPipeline, HumanRatingsStore, RegressionResults,
    discover_face_images, fit_score_model, predict_score_model, make_cv_splits,
    cross_validate_predictor, array_hash, merge_groups, write_json, _atomic_save_npz,
)
from ..race_perception import (
    candidate_grid, inner_predictions, select_strategies,
    apply_strategies, fit_bundle_readout, predict_readout,
)
from ..race_perception.phrases import TARGETS, build_phrase_bank
from .models import BackboneSession, SoftContext, RatingNetwork, level_prompts, objective


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("ridge", "kernel", "phrase-kernel", "hybrid", "adapter", "prototypes",
           "phrase-adapter", "soft-prompt", "local", "lora", "partial", "pool")
CLASSICAL = METHODS[:4]


@dataclass
class RatingData:
    paths: tuple
    y: np.ndarray
    se: np.ndarray
    hist: np.ndarray
    groups: np.ndarray


def load_data(args):
    paths = discover_face_images(FaceDatasetConfig(args.images, limit=None))
    store = HumanRatingsStore(args.ratings)
    vectors = [store.means(t, paths, min_ratings=2, reject_nonfinite=True) for t in args.targets]
    y, se = [np.column_stack([getattr(v, key) for v in vectors]) for key in ("means", "mean_se")]
    eligible = np.flatnonzero(np.isfinite(y).all(1) & np.isfinite(se).all(1))
    if args.limit:
        eligible = np.sort(np.random.default_rng(args.seed).permutation(eligible)[:args.limit])
    paths = tuple(paths[i] for i in eligible)
    y, se = y[eligible], se[eligible]
    if len(paths) < 8 or np.any((y < 0) | (y > 1)):
        raise ValueError("Need at least eight complete faces with ratings in [0, 1].")
    # Isolated adapter to the existing trusted-local store; no new core API.
    raw = store._load()
    hist = np.zeros((len(paths), len(args.targets), args.levels), np.float32)
    for i, path in enumerate(paths):
        for t, name in enumerate(args.targets):
            values = np.asarray(raw[name][path.name], dtype=float).reshape(-1)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"Invalid individual ratings: {path.name}/{name}")
            scaled = values * (args.levels - 1)
            lo = np.floor(scaled).astype(int)
            hi = np.minimum(lo + 1, args.levels - 1)
            np.add.at(hist[i, t], lo, (1 - (scaled-lo)) / len(values))
            np.add.at(hist[i, t], hi, (scaled-lo) / len(values))
    hashes = []
    for path in paths:
        with Image.open(path) as im:
            hashes.append(array_hash(np.asarray(im.convert("RGB"))))
    groups = np.asarray(hashes)
    if args.groups:
        supplied = json.loads(args.groups.read_text())
        groups = merge_groups(groups, [str(supplied[p.name]) for p in paths])
    return RatingData(paths, y.astype(np.float32), se.astype(np.float32), hist, groups)


class SelectedPipeline(FaceImpressionPipeline):
    def __init__(self, engine, config, paths):
        super().__init__(engine, config)
        self.paths = paths

    def image_paths(self):
        return self.paths


class Assets:
    """Only label-independent frozen features persist across supervised folds."""
    def __init__(self, engine, data, args):
        self.engine, self.data, self.args = engine, data, args
        self.pipeline = SelectedPipeline(engine, FaceDatasetConfig(args.images, cache_dir=args.cache_dir,
                                         limit=None, batch_size=args.encode_batch), data.paths)
        self.views = [self.pipeline.features(p).features.float().numpy() for p in args.patches]
        self.full = np.concatenate(self.views, axis=1)
        self.global_features = F.normalize(torch.tensor(np.mean(self.views, axis=0)), dim=-1)
        self.text_cache, self.dense = {}, None

    def text(self, prompts):
        key = tuple(prompts)
        if key not in self.text_cache:
            with torch.no_grad():
                features = torch.cat([self.engine.encode_text(key[i:i+8], mode="short").float().cpu()
                                      for i in range(0, len(key), 8)])
            self.text_cache[key] = features
        return self.text_cache[key]

    def phrases(self):
        text = self.text(build_phrase_bank()["phrases"]).numpy()
        # Training-fold scaling in the reused kernel cancels global scale/bias.
        return np.concatenate([view @ text.T for view in self.views], axis=1)

    def local(self):
        if self.dense is None:
            chunks = []
            with torch.no_grad():
                for start in range(0, len(self.data.paths), self.args.encode_batch):
                    inputs = self.engine.prepare_images(self.data.paths[start:start+self.args.encode_batch],
                                                       max_num_patches=self.args.patches[-1])
                    chunks.extend(x.reshape(-1, x.shape[-1]).half().cpu()
                                  for x in self.engine.encode_dense_preprocessed(inputs))
            padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
            mask = torch.arange(padded.shape[1])[None] < torch.tensor([len(x) for x in chunks])[:, None]
            self.dense = padded, mask
        return self.dense


def fit_classical(method, assets, data, train, test, args, fold):
    x = assets.phrases() if method in ("phrase-kernel", "hybrid") else assets.full
    splits = make_cv_splits(len(train), args.inner_folds, args.seed+fold+1, data.groups[train])
    if method == "ridge":
        alphas = (1., 10., 100., 1000.)
        def fit(a, alpha):
            return fit_score_model(x[a], data.y[a], estimator=Ridge(alpha=alpha, solver="lsqr"))
        candidates = []
        for alpha in alphas:
            prediction, _ = cross_validate_predictor(train[:, None], data.y[train],
                lambda a, y, b, f: predict_score_model(fit(a[:, 0].astype(int), alpha), x[b[:, 0].astype(int)]), splits=splits)
            candidates.append(prediction)
        losses = np.array([np.mean((p-data.y[train])**2) for p in candidates])
        fitted = fit(train, alphas[int(losses.argmin())])
        return predict_score_model(fitted, x[test]), {"readout": fitted, "alpha": alphas[int(losses.argmin())]}
    recipes = [r for r in candidate_grid() if len(assets.views) == 2 or
               (r["bank"] == "both" and not r.get("centered"))]
    if args.smoke:
        recipes = [dict(family=f, bank="both", centered=False, alpha=.1, **({"gamma":.1} if f == "rbf" else {}))
                   for f in ("linear", "poly2", "rbf")]
    inner, _ = inner_predictions(x[train], data.y[train], recipes, args.inner_folds, args.seed+fold+1, data.groups[train])
    strategies, _, _ = select_strategies(inner, data.y[train], recipes)
    fitted = fit_bundle_readout(x[train], data.y[train], recipes, strategies)
    prediction = predict_readout(fitted, x[test])
    state = {"readout": fitted}
    if method == "hybrid":
        # Complement of the exact fixed phrase span; SVD uses no labels.
        text = assets.text(build_phrase_bank()["phrases"]).numpy()
        _, singular, vh = np.linalg.svd(text, full_matrices=False)
        basis = vh[singular > singular[0]*1e-6]
        complement = np.concatenate([v - (v @ basis.T) @ basis for v in assets.views], axis=1)
        selected_inner = apply_strategies(inner, strategies)
        residual_oof = np.empty_like(data.y[train])
        # Each residual learner sees baseline residuals cross-fitted *within its
        # own training subset*, not labels from its validation subset.
        for a, b in splits:
            sub = train[a]
            sub_inner, _ = inner_predictions(x[sub], data.y[sub], recipes, args.inner_folds,
                                            args.seed+fold+17, data.groups[sub])
            sub_strategies, _, _ = select_strategies(sub_inner, data.y[sub], recipes)
            residuals = data.y[sub] - apply_strategies(sub_inner, sub_strategies)
            model = fit_score_model(complement[sub], residuals, estimator=Ridge(alpha=100., solver="lsqr"))
            residual_oof[b] = predict_score_model(model, complement[train[b]])
        weights = np.array([0., .25, .5, 1.])
        losses = np.mean((selected_inner[None] + weights[:, None, None]*residual_oof - data.y[train])**2, axis=1)
        weight = weights[losses.argmin(0)]
        residual = fit_score_model(complement[train], data.y[train]-selected_inner,
                                   estimator=Ridge(alpha=100., solver="lsqr"))
        prediction += weight * predict_score_model(residual, complement[test])
        state.update(basis=basis, residual=residual, weight=weight)
    return prediction, state


def train_network(method, assets, data, train, validation, args, *, epochs, seed, capture=False, desc=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    engine, device = assets.engine, assets.engine.device
    kind = "lora" if args.visual_lora or method == "lora" else method if method in ("partial", "pool") else "frozen"
    live = kind != "frozen"
    head_method = method if method in ("adapter", "prototypes", "phrase-adapter", "soft-prompt", "local") else "direct"
    if head_method == "local" and not live:
        assets.local()
    prompts = level_prompts(args.targets, args.levels)
    prototype = assets.text(prompts).to(device) if head_method in ("prototypes", "soft-prompt") else None
    phrase = assets.text(build_phrase_bank()["phrases"]).to(device) if head_method == "phrase-adapter" else None
    with BackboneSession(engine, kind, args.blocks, args.rank, not args.no_checkpointing):
        soft = SoftContext(engine, prompts, args.context_tokens, not args.no_checkpointing) if head_method == "soft-prompt" else None
        network = RatingNetwork(assets.global_features.shape[1], len(args.targets), head_method,
                                rank=args.rank, bottleneck=args.bottleneck, levels=args.levels,
                                prototype_init=prototype, phrase_init=phrase, soft_context=soft).to(device)
        # Starting mean is training-only; no held-out target enters a head.
        with torch.no_grad():
            network.head.weight.mul_(.01)
            network.head.bias.copy_(torch.tensor(data.y[train].mean(0), device=device))
        backbone = [(n, p) for n, p in engine.model.named_parameters() if p.requires_grad]
        head_params = [p for p in network.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{"params":head_params, "lr":args.head_lr},
                                      {"params":[p for _, p in backbone], "lr":args.encoder_lr}], weight_decay=args.weight_decay)
        all_params = head_params + [p for _, p in backbone]
        y = torch.tensor(data.y, device=device)
        se, hist = torch.tensor(data.se, device=device), torch.tensor(data.hist, device=device)
        global_features = assets.global_features.to(device)

        def forward(indices, cached_text=None):
            ix = torch.as_tensor(indices, device=device, dtype=torch.long)
            dense = mask = None
            if live:
                views = []
                for patches in args.patches:
                    inputs = engine.prepare_images([data.paths[i] for i in indices], max_num_patches=patches)
                    views.append(engine.encode_image_preprocessed(inputs).float())
                z = F.normalize(torch.stack(views).mean(0), dim=-1)
                if head_method == "local":
                    features = engine.encode_dense_preprocessed(inputs)
                    dense = torch.nn.utils.rnn.pad_sequence([f.reshape(-1, f.shape[-1]) for f in features], batch_first=True)
                    mask = torch.arange(dense.shape[1], device=device)[None] < torch.tensor([f.shape[0]*f.shape[1] for f in features], device=device)[:, None]
            else:
                z = global_features[ix]
                if head_method == "local":
                    dense, mask = (v[indices].to(device) for v in assets.dense)
            return network(z, dense, mask, text_override=cached_text), ix

        @torch.no_grad()
        def predict(indices):
            network.eval()
            engine.model.eval()
            text = network.soft_context() if network.soft_context else None
            return np.concatenate([forward(indices[start:start+args.batch_size], text)[0][0].cpu().numpy()
                                   for start in range(0, len(indices), args.batch_size)])

        initial = {n: p.detach().cpu().clone() for n, p in network.named_parameters()} if args.smoke else None
        initial_backbone = {n: p.detach().cpu().clone() for n, p in backbone} if args.smoke else None
        history, best, best_epoch, stale, steps = [], float("inf"), 0, 0, 0
        batches = -(-len(train) // args.batch_size)
        planned = epochs * batches
        progress = tqdm(total=min(planned, args.max_steps) if args.max_steps else planned,
                        desc=desc or method, leave=True)
        for epoch in range(epochs):
            network.train()
            # Enable checkpointing only in trainable visual blocks; frozen text
            # remains deterministic even while differentiating soft context.
            engine.model.vision_model.train(live)
            optimizer.param_groups[1]["lr"] = 0. if epoch < args.warmup_epochs else args.encoder_lr
            order = rng.permutation(train)
            total, count = 0., 0
            for start in range(0, len(order), args.batch_size):
                if args.max_steps and steps >= args.max_steps:
                    break
                indices = order[start:start+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                outputs, ix = forward(indices)
                loss = objective(outputs, y[ix], se[ix], hist[ix], auxiliary=args.auxiliary,
                                 aux_weight=args.aux_weight, anchor_weight=args.anchor_weight,
                                 preserve=args.preserve, preserve_weight=args.preserve_weight, teacher=global_features[ix])
                if not torch.isfinite(loss):
                    progress.close()
                    raise FloatingPointError("Nonfinite loss; no checkpoint was written.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1., error_if_nonfinite=True)
                optimizer.step()
                total += loss.item()*len(indices)
                count += len(indices)
                steps += 1
                progress.set_postfix(epoch=epoch+1, loss=f"{loss.item():.4f}", refresh=False)
                progress.update()
            score = float(np.mean((predict(validation)-data.y[validation])**2)) if validation is not None else None
            history.append({"epoch":epoch+1, "train_loss":total/max(count, 1), "validation_mse":score})
            if score is not None:
                progress.set_postfix(epoch=epoch+1, loss=f"{total/max(count, 1):.4f}", val=f"{score:.4f}")
            if score is not None and (not live or epoch >= args.warmup_epochs):
                if score < best:
                    best, best_epoch, stale = score, epoch+1, 0
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            if args.max_steps and steps >= args.max_steps:
                break
        progress.close()
        predictions = predict(args.predict_indices) if args.predict_indices is not None else None
        details = {"history":history, "best_epoch":best_epoch or len(history), "steps":steps,
                   "trainable_head":sum(p.numel() for p in head_params),
                   "trainable_backbone":sum(p.numel() for _, p in backbone)}
        if args.smoke:
            details["head_changed"] = any(not torch.equal(p.detach().cpu(), initial[n]) for n, p in network.named_parameters())
            details["backbone_changed"] = any(not torch.equal(p.detach().cpu(), initial_backbone[n]) for n, p in backbone)
            adapted = {"adapter":"adapter.2.weight", "prototypes":"text_up", "phrase-adapter":"text_up",
                       "soft-prompt":"soft_context.context", "local":"queries"}.get(head_method, "head.weight")
            details["adaptation_changed"] = not torch.equal(dict(network.named_parameters())[adapted].detach().cpu(), initial[adapted])
            if not details["head_changed"] or not details["adaptation_changed"] or (live and not details["backbone_changed"]):
                raise RuntimeError("Smoke training did not update the intended parameters.")
        state = None
        if capture:
            state = {"head":{n: v.detach().cpu().clone() for n, v in network.state_dict().items()},
                     "backbone":{n: p.detach().cpu().clone() for n, p in backbone},
                     "prompts":build_phrase_bank()["phrases"] if head_method == "phrase-adapter" else prompts}
        del optimizer, all_params, head_params, backbone, network
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return predictions, details, state


def run_method(method, assets, data, args):
    details, fitted = [], None
    def fit_predict(a, unused_y, b, fold):
        nonlocal fitted
        train, test = a[:, 0].astype(int), b[:, 0].astype(int)
        if method in CLASSICAL:
            prediction, fitted = fit_classical(method, assets, data, train, test, args, fold)
            details.append({"train":train.tolist(), "test":test.tolist()})
            return prediction
        inner_train, inner_valid = make_cv_splits(len(train), args.inner_folds, args.seed+fold+1, data.groups[train])[0]
        args.predict_indices = None
        _, selection, _ = train_network(method, assets, data, train[inner_train], train[inner_valid], args,
                                         epochs=args.epochs, seed=args.seed+fold,
                                         desc=f"{method} fold{fold} select")
        args.predict_indices = test
        prediction, fit, fitted = train_network(method, assets, data, train, None, args,
                                                 epochs=selection["best_epoch"], seed=args.seed+fold,
                                                 capture=args.save_checkpoint, desc=f"{method} fold{fold} refit")
        details.append({"train":train.tolist(), "test":test.tolist(),
                        "inner_train":train[inner_train].tolist(), "inner_validation":train[inner_valid].tolist(),
                        "selection":selection, "refit":fit})
        return prediction

    if args.mode == "cv":
        splits = make_cv_splits(len(data.y), args.folds, args.seed, data.groups)
        prediction, fold_ids = cross_validate_predictor(np.arange(len(data.y))[:, None], data.y, fit_predict, splits=splits)
        rows = np.arange(len(data.y))
    else:
        train, rows = make_cv_splits(len(data.y), args.folds, args.seed, data.groups)[0]
        prediction = fit_predict(train[:, None], data.y[train], rows[:, None], 0)
        fold_ids = np.zeros(len(rows), dtype=int)
    if args.mode == "fit":
        all_rows = np.arange(len(data.y))
        if method in CLASSICAL:
            _, fitted = fit_classical(method, assets, data, all_rows, rows, args, 0)
        else:
            args.predict_indices = None
            _, final, fitted = train_network(method, assets, data, all_rows, None, args,
                                              epochs=details[0]["selection"]["best_epoch"], seed=args.seed,
                                              capture=args.save_checkpoint, desc=f"{method} final")
            details.append({"final_fit":final})
    results = RegressionResults(data.y[rows], {method:prediction}, tuple(args.targets), fold_ids,
                                metadata={"mode":args.mode, "smoke":args.smoke, "rows":rows.tolist(), "fits":details})
    return results, fitted, rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=["adapter"], choices=(*METHODS, "all"))
    p.add_argument("--mode", choices=("holdout", "cv", "fit"), default="holdout")
    p.add_argument("--images", type=Path, default=Path("/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images"))
    p.add_argument("--ratings", type=Path, default=ROOT/"data/dim_to_photo_to_ratings.pkl")
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--groups", type=Path)
    p.add_argument("--model", choices=("so400m", "base"), default="so400m")
    p.add_argument("--device", default="auto")
    p.add_argument("--patches", nargs="+", type=int, default=[256])
    p.add_argument("--limit", type=int)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--save-checkpoint", action="store_true", help="Only with --mode fit; never in smoke/CV.")
    p.add_argument("--smoke", action="store_true", help="32 faces, one update per fit, no weights saved.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--encode-batch", type=int, default=4)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--encoder-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--context-tokens", type=int, default=4)
    p.add_argument("--levels", type=int, choices=(3, 5), default=5)
    p.add_argument("--auxiliary", choices=("none", "pairwise", "ordinal", "distribution"), default="none")
    p.add_argument("--aux-weight", type=float, default=.05)
    p.add_argument("--anchor-weight", type=float, default=.1)
    p.add_argument("--preserve", choices=("none", "feature", "gram"), default="feature")
    p.add_argument("--preserve-weight", type=float, default=.1)
    p.add_argument("--visual-lora", action="store_true", help="Combine LoRA with prototypes, soft prompts, or local pooling.")
    p.add_argument("--no-checkpointing", action="store_true", help="Disable activation checkpointing (unrelated to saving weights).")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260919)
    return p


def main():
    args = parser().parse_args()
    args.methods = list(METHODS) if "all" in args.methods else args.methods
    if args.save_checkpoint and (args.smoke or args.mode != "fit" or not args.output):
        raise ValueError("Saving weights requires --mode fit --output PATH and no --smoke.")
    if args.visual_lora and any(m not in ("prototypes", "soft-prompt", "local", "adapter", "phrase-adapter") for m in args.methods):
        raise ValueError("--visual-lora requires an adaptable neural method.")
    if args.smoke:
        args.limit, args.epochs, args.max_steps = args.limit or 32, 1, 1
        args.encode_batch, args.warmup_epochs = min(args.encode_batch, 2), 0
        args.folds, args.inner_folds = 4, 2
    for field in ("epochs", "patience", "batch_size", "encode_batch", "rank", "blocks", "bottleneck", "context_tokens"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if args.folds < 2 or args.inner_folds < 2 or len(set(args.targets)) != len(args.targets):
        raise ValueError("Need at least two folds and unique targets.")
    if args.context_tokens > 8 or args.max_steps < 0 or args.warmup_epochs < 0:
        raise ValueError("Context supports up to eight tokens; steps and warm-up must be nonnegative.")
    if (args.visual_lora or any(m in ("lora", "partial", "pool") for m in args.methods)) and args.epochs <= args.warmup_epochs:
        raise ValueError("Backbone adaptation needs epochs greater than warmup-epochs.")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.output:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Choose an empty output directory; existing experiments are never overwritten.")
        args.output.mkdir(parents=True, exist_ok=True)
    print("Loading ratings and frozen FG-CLIP2…", flush=True)
    data = load_data(args)
    with tempfile.TemporaryDirectory(prefix="fgclip2-smoke-") if args.smoke and args.cache_dir is None else nullcontext(None) as temporary:
        args.cache_dir = args.cache_dir or (Path(temporary) if temporary else Path(__file__).parent/".cache")
        engine = FGCLIP2(BASE_MODEL_ID if args.model == "base" else DEFAULT_MODEL_ID, device=args.device,
                         local_files_only=True, memory_reserve_gb=.5 if args.smoke else 1.25)
        assets = Assets(engine, data, args)
        manifest = {"config":config, "model":engine.model_id, "revision":engine.revision, "dtype":str(engine.dtype),
                    "paths":[str(p) for p in data.paths], "groups":data.groups.tolist(),
                    "phrases":build_phrase_bank()["phrases"] if any(m in ("phrase-kernel", "hybrid", "phrase-adapter") for m in args.methods) else None,
                    "data_hash":array_hash(data.y, data.se, data.hist, data.groups),
                    "source_hash":hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name("models.py").read_bytes()).hexdigest()}
        if args.output:
            write_json(args.output/"manifest.json", manifest)
        for method in args.methods:
            print(f"{method}: starting {'smoke' if args.smoke else args.mode}", flush=True)
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                results, fitted, rows = run_method(method, assets, data, args)
            report = results.summary()
            report["seconds"] = time.perf_counter()-started
            rmse = report["macro_mse"][method]**.5
            print(f"{method}: held-out RMSE={rmse:.5f}, n={len(rows)}, {report['seconds']:.1f}s", flush=True)
            if args.output:
                write_json(args.output/f"{method}.json", report)
                _atomic_save_npz(args.output/f"{method}-predictions.npz", predictions=results.predictions[method],
                                 y=results.y_true, rows=rows, fold_ids=results.fold_ids)
                if args.save_checkpoint:
                    torch.save({"format":"fgclip2_adaptation_v1", "method":method, "manifest":manifest, "state":fitted},
                               args.output/f"{method}.pt")
            del fitted
    print("Complete. " + ("Final weights saved." if args.save_checkpoint else "No model checkpoints saved."), flush=True)


if __name__ == "__main__":
    main()
