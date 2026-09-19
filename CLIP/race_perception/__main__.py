"""Reproducible prepare/train/predict CLI for generated-face rating regression.

Run from the repository root using /opt/anaconda3/envs/manip311/bin/python.
Run outside the sandbox: shared FG-CLIP utilities import torch.
Numerical kernel fitting uses CPU; image encoding uses MPS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from . import (
    RacePerceptionPredictor, apply_strategies, array_hash, candidate_grid, cv_splits,
    fit_bundle_readout, inner_predictions, predict_candidates, predict_readout,
    reference_predict, select_strategies, validate_xy,
)
from .phrases import TARGETS, build_phrase_bank
from ..fgclip2_face_impressions import (
    HumanRatingsStore, RegressionResults, Viz, merge_groups,
    paired_prediction_bootstrap, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "CLIP/research/race_perception"
DEFAULT_IMAGES = Path("/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images")
DEFAULT_RATINGS = ROOT / "data/dim_to_photo_to_ratings.pkl"


def prepare(args):
    import torch
    from PIL import Image
    from ..fgclip2_core import FGCLIP2, DEFAULT_MODEL_ID, DEFAULT_REVISION
    from ..fgclip2_face_impressions import FaceDatasetConfig, FaceImpressionPipeline

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    bank = build_phrase_bank()
    bank_path = out / "phrase_bank.json"
    if bank_path.exists() and json.loads(bank_path.read_text()) != bank:
        raise ValueError("Existing frozen phrase bank differs; use a new output directory.")
    write_json(bank_path, bank)
    print(f"Frozen bank: {len(bank['phrases'])} phrases, {len(bank['groups'])} groups, {len(TARGETS)} targets", flush=True)
    engine = FGCLIP2(DEFAULT_MODEL_ID, revision=DEFAULT_REVISION, device="mps",
                     dtype=torch.float16, local_files_only=True)
    pipeline = FaceImpressionPipeline(engine, FaceDatasetConfig(args.images, limit=None, batch_size=16))
    paths = pipeline.image_paths()
    ratings = HumanRatingsStore(args.ratings)
    missing_targets = set(TARGETS) - set(ratings.variables)
    if missing_targets:
        raise ValueError(f"Missing expected targets: {missing_targets}")
    vectors = [ratings.means(t, paths, min_ratings=2, reject_nonfinite=True) for t in TARGETS]
    y, counts, se = [np.column_stack([getattr(v, key) for v in vectors])
                     for key in ("means", "rater_counts", "mean_se")]
    invalid = [dict(face=paths[i].name, target=TARGETS[t], reason="missing, too few, or nonfinite ratings")
               for i,t in zip(*np.where(~np.isfinite(y)))]
    valid = np.isfinite(y).all(axis=1)
    aligned_paths = [p for p, keep in zip(paths, valid) if keep]
    if not aligned_paths:
        raise ValueError("No complete cases.")
    # Group exact decoded-image duplicates even across file encodings. No latent
    # family metadata is assumed. An optional explicit map takes precedence for
    # related identities, but is unioned with pixel duplicates.
    files, pixel_hashes = [], []
    for path in aligned_paths:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        pixel_hashes.append(array_hash(rgb))
        files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          pixel_sha256=pixel_hashes[-1]))
    groups = np.asarray(pixel_hashes)
    identity_description = "Exact decoded-image duplicates grouped. Otherwise original generated faces are assumed independent; latent-family metadata unavailable."
    if args.groups:
        mapping = json.loads(args.groups.read_text())
        given = [str(mapping[p.name]) for p in aligned_paths]
        groups = merge_groups(given, pixel_hashes)
        identity_description = "User-provided identity groups unioned with exact decoded-image duplicates."
    audit = dict(discovered_faces=len(paths), complete_faces=int(valid.sum()), exclusions=invalid,
                 excluded_paths=[str(p) for p, keep in zip(paths, valid) if not keep],
                 unique_groups=len(np.unique(groups)), identity_assumption=identity_description,
                 targets={t: dict(count_min=int(counts[valid, j].min()), count_median=float(np.median(counts[valid, j])),
                                  count_max=int(counts[valid, j].max()), mean_min=float(y[valid, j].min()),
                                  mean_max=float(y[valid, j].max()), median_mean_se=float(np.median(se[valid, j])),
                                  rms_mean_se=float(np.sqrt(np.mean(se[valid, j] ** 2)))) for j, t in enumerate(TARGETS)},
                 input_files=files, ratings_path=str(args.ratings),
                 ratings_sha256=hashlib.sha256(args.ratings.read_bytes()).hexdigest())
    write_json(out / "data_audit.json", audit)
    banks, encoding = [], []
    for patches in (128, 256):
        print(f"Encoding/scoring {patches} patches", flush=True)
        scores = pipeline.rate(bank["phrases"], text_mode="short", max_num_patches=patches)
        if scores.paths != paths or scores.texts != tuple(bank["phrases"]):
            raise ValueError("Image or phrase order changed between banks.")
        banks.append(scores.scores[valid])
        encoding.append(dict(patches=patches, feature_identity=pipeline.features(patches).cache_key, runtime=scores.runtime))
        print(scores.runtime, flush=True)
    metadata = dict(encoding=dict(model=engine.model_id, revision=engine.revision, dtype=str(engine.dtype),
                                 text_mode="short", score_transform="exp(logit_scale) * dot(normalized_image, normalized_text) + logit_bias"),
                    banks=encoding, phrase_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                    ratings_sha256=audit["ratings_sha256"], identity_assumption=identity_description)
    x = np.column_stack(banks)
    validate_xy(x, y[valid])
    np.savez_compressed(out / "features.npz", x=x, y=y[valid], counts=counts[valid], mean_se=se[valid],
                        paths=np.asarray([str(p) for p in aligned_paths]), groups=groups,
                        targets=np.asarray(TARGETS), phrases=np.asarray(bank["phrases"]), metadata=json.dumps(metadata))
    print(f"Saved aligned features {x.shape}, targets {y[valid].shape}", flush=True)


def train(args):
    out = args.output
    with np.load(out / "features.npz", allow_pickle=False) as data:
        x, y = validate_xy(data["x"], data["y"])
        paths, groups, counts, se = [data[k].copy() for k in ("paths", "groups", "counts", "mean_se")]
        metadata = json.loads(str(data["metadata"]))
        if tuple(data["targets"]) != TARGETS:
            raise ValueError("Unexpected target order.")
        phrases = data["phrases"].tolist()
    bank = json.loads((out / "phrase_bank.json").read_text())
    if bank["phrases"] != phrases or hashlib.sha256((out / "phrase_bank.json").read_bytes()).hexdigest() != metadata["phrase_sha256"]:
        raise ValueError("Frozen phrase bank does not match feature cache.")
    recipes = candidate_grid()
    config = dict(outer_folds=args.outer_folds, inner_folds=args.inner_folds, seed=args.seed,
                  candidates=recipes, selection="Output-specific inner MSE, optionally blending two kernel winners at weights .25/.5/.75.",
                  target_contract=bank["semantics"], evaluation="Internal nested CV; historical Asian prompt work used this dataset. No pristine external-test claim.",
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      Path(__file__), Path(__file__).with_name("__init__.py"), Path(__file__).with_name("phrases.py"),
                      ROOT / "CLIP/fgclip2_face_impressions.py")})
    fingerprint = hashlib.sha256((array_hash(x, y, groups, paths) + json.dumps(config, sort_keys=True)).encode()).hexdigest()
    config["fingerprint"] = fingerprint
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Data/code/protocol changed. Preserve the experiment and use a fresh output directory.")
    write_json(protocol_path, config)
    checkpoint = out / "folds"
    checkpoint.mkdir(exist_ok=True)
    splits = cv_splits(len(y), args.outer_folds, args.seed, groups)
    predictions = {name: np.full_like(y, np.nan) for name in ("mean", "reference", "ridge", "selected")}
    fold_ids = np.full(len(y), -1, int)
    started = time.perf_counter()
    for fold, (train_ix, test_ix) in enumerate(splits):
        save = checkpoint / f"fold_{fold:02d}.npz"
        record_path = checkpoint / f"fold_{fold:02d}.json"
        if save.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if record["fingerprint"] != fingerprint:
                raise ValueError("Stale fold checkpoint.")
            with np.load(save) as saved:
                np.testing.assert_array_equal(saved["test"], test_ix)
                for name in predictions:
                    predictions[name][test_ix] = saved[name]
            print(f"Fold {fold+1}/{len(splits)} resumed", flush=True)
        else:
            step = time.perf_counter()
            tx, ty = x[train_ix], y[train_ix]
            inner, inner_splits = inner_predictions(tx, ty, recipes, args.inner_folds, args.seed + fold + 1, groups[train_ix])
            strategies, ridge, losses = select_strategies(inner, ty, recipes)
            needed = sorted(set(ridge) | {j for s in strategies for j in s["indices"]})
            selected = predict_candidates(tx, ty, x[test_ix], [recipes[j] for j in needed])
            candidate_predictions = np.full((len(test_ix), len(recipes), y.shape[1]), np.nan)
            candidate_predictions[:, needed, :] = selected
            predictions["selected"][test_ix] = apply_strategies(candidate_predictions, strategies)
            predictions["ridge"][test_ix] = np.column_stack([candidate_predictions[:, j, t] for t, j in enumerate(ridge)])
            predictions["reference"][test_ix] = reference_predict(tx, ty, x[test_ix], bank["reference_indices"])
            predictions["mean"][test_ix] = ty.mean(axis=0)
            record = dict(fold=fold, fingerprint=fingerprint, train=train_ix.tolist(), test=test_ix.tolist(),
                          inner_splits=[dict(train=train_ix[a].tolist(), test=train_ix[b].tolist()) for a, b in inner_splits],
                          strategies=strategies, ridge_indices=ridge, inner_mse=losses.tolist(),
                          seconds=time.perf_counter()-step)
            np.savez_compressed(save, test=test_ix, **{k: v[test_ix] for k, v in predictions.items()})
            write_json(record_path, record)
            print(f"Fold {fold+1}/{len(splits)} finished; train={len(train_ix)}, test={len(test_ix)}, {record['seconds']:.1f}s", flush=True)
        fold_ids[test_ix] = fold
    if np.any(fold_ids < 0) or any(not np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Incomplete outer predictions.")
    np.savez_compressed(out / "oof_predictions.npz", y=y, paths=paths, groups=groups, counts=counts, mean_se=se,
                        targets=np.asarray(TARGETS), folds=fold_ids, **predictions)
    with (out / "oof_predictions.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "group", "fold", "target", "human_mean", "raters", "mean_se", *predictions])
        for i, path in enumerate(paths):
            for t, target in enumerate(TARGETS):
                writer.writerow([path, groups[i], fold_ids[i], target, y[i,t], counts[i,t], se[i,t], *[p[i,t] for p in predictions.values()]])
    results = RegressionResults(y, predictions, TARGETS, fold_ids, metadata=dict(
        protocol=config, phrases=len(phrases), features=x.shape[1],
        elapsed_cv_seconds=time.perf_counter()-started))
    intervals = paired_prediction_bootstrap(y, predictions["ridge"], predictions["selected"],
                                           groups=groups, seed=args.seed)
    # Only RMSE intervals are needed here; preserve undefined correlation CIs in shared callers.
    results.metadata["bootstrap_vs_ridge"] = {k: intervals[k] for k in ("rmse", "rmse_reduction")}
    write_json(out / "results.json", results.summary())
    print("All outer folds complete. Final inner tuning and full-data fit.", flush=True)
    inner, final_splits = inner_predictions(x, y, recipes, args.inner_folds, args.seed+10000, groups)
    strategies, ridge, losses = select_strategies(inner, y, recipes)
    readout = fit_bundle_readout(x, y, recipes, strategies)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch")}
    versions["python"] = platform.python_version()
    bundle = dict(format="fgclip2_subjective_race_v1", targets=list(TARGETS), phrases=phrases, patches=[128,256],
                  encoding=metadata["encoding"], metadata=metadata, readout=readout, recipes=recipes,
                  protocol_fingerprint=fingerprint, versions=versions, training_paths=paths.tolist(),
                  target_semantics=bank["semantics"])
    joblib.dump(bundle, out / "race_perception_model.joblib", compress=3)
    write_json(out / "model_manifest.json", dict(format=bundle["format"], targets=list(TARGETS), phrases=len(phrases),
               patches=[128,256], training_images=len(y), encoding=metadata["encoding"], versions=versions,
               strategies={t: dict(recipes=[recipes[j] for j in s["indices"]], weights=s["weights"]) for t,s in zip(TARGETS,strategies)},
               protocol_fingerprint=fingerprint, model_sha256=hashlib.sha256((out / "race_perception_model.joblib").read_bytes()).hexdigest(),
               final_inner_splits=[dict(train=a.tolist(),test=b.tolist()) for a,b in final_splits], final_inner_mse=losses.tolist()))
    # Independent path implementation comparison catches serialization/selection
    # contract errors, rather than treating fitted accuracy as validation.
    check_ix = np.arange(min(7,len(y)))
    needed = sorted({j for s in strategies for j in s["indices"]})
    direct = np.full((len(check_ix),len(recipes),y.shape[1]),np.nan)
    direct[:,needed,:] = predict_candidates(x,y,x[check_ix],[recipes[j] for j in needed])
    expected = apply_strategies(direct,strategies)
    loaded = RacePerceptionPredictor.load(out / "race_perception_model.joblib")
    actual = loaded.predict_scores(x[check_ix])
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    write_json(out / "serialization_check.json", dict(images=len(check_ix), max_absolute_difference=float(np.max(np.abs(actual-expected)))))
    table = results.metric_table()
    (out / "REPORT.md").write_text(table + "\n")
    Viz.export(Viz.oof_scatter(results, "selected"), out / "oof_scatter")
    print(results.metric_table(methods=("selected",)), flush=True)
    print(f"Saved model and report in {out}", flush=True)


def predict(args):
    from ..fgclip2_face_impressions import FaceDatasetConfig, discover_face_images
    model = RacePerceptionPredictor.load(args.output / "race_perception_model.joblib")
    paths = discover_face_images(FaceDatasetConfig(args.images, limit=None))
    prediction = model.predict_images(paths)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", *[f"predicted_{t}_mean" for t in model.targets]])
        writer.writerows([str(path), *row] for path,row in zip(paths,prediction))
    print(f"Saved {len(paths)} × {len(model.targets)} predictions to {args.csv}", flush=True)


def verify(out):
    with np.load(out / "features.npz") as data:
        x, y, paths, groups = [data[k].copy() for k in ("x", "y", "paths", "groups")]
    protocol = json.loads((out / "protocol.json").read_text())
    records = sorted((out / "folds").glob("fold_*.json"))
    assert len(records) == protocol["outer_folds"]
    coverage = np.zeros(len(y), int)
    expected_fold_ids = np.full(len(y), -1, int)
    for path in records:
        record = json.loads(path.read_text())
        assert record["fingerprint"] == protocol["fingerprint"]
        train, test = set(record["train"]), set(record["test"])
        assert not train & test and train | test == set(range(len(y)))
        assert not set(groups[list(train)]) & set(groups[list(test)])
        coverage[list(test)] += 1
        expected_fold_ids[list(test)] = record["fold"]
        inner_coverage = {i: 0 for i in train}
        for split in record["inner_splits"]:
            a, b = set(split["train"]), set(split["test"])
            assert not a & b and a | b == train
            assert not set(groups[list(a)]) & set(groups[list(b)])
            for i in b:
                inner_coverage[i] += 1
        assert set(inner_coverage.values()) == {1}
    np.testing.assert_array_equal(coverage, 1)
    with np.load(out / "oof_predictions.npz") as oof:
        np.testing.assert_array_equal(oof["folds"], expected_fold_ids)
        np.testing.assert_array_equal(oof["paths"], paths)
        np.testing.assert_array_equal(oof["y"], y)
        for method in ("mean", "reference", "ridge", "selected"):
            assert oof[method].shape == y.shape and np.isfinite(oof[method]).all()
    manifest = json.loads((out / "model_manifest.json").read_text())
    assert hashlib.sha256((out / "race_perception_model.joblib").read_bytes()).hexdigest() == manifest["model_sha256"]
    model = RacePerceptionPredictor.load(out / "race_perception_model.joblib")
    assert tuple(model.targets) == TARGETS
    assert model.bundle["training_paths"] == paths.tolist()
    indices = np.linspace(0, len(y)-1, 7, dtype=int)
    expected = model.predict_scores(x[indices])
    print("All nested splits and OOF coverage verified. Checking fresh-image inference.", flush=True)
    actual, scores = model.predict_images(paths[indices].tolist(), batch_size=7, return_scores=True)
    # Float32 normalized-feature matrix products may differ very slightly with
    # batch size; tolerances remain far below slider-mean uncertainty.
    np.testing.assert_allclose(scores, x[indices], atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
    one = model.predict_images([paths[indices[0]]], batch_size=1)
    np.testing.assert_allclose(one[0], actual[0], atol=2e-5, rtol=1e-5)
    report = dict(outer_folds=len(records), inner_folds_per_outer=protocol["inner_folds"],
                  exact_oof_coverage=True, group_disjoint_both_levels=True, target_order=list(TARGETS),
                  fresh_image_paths=paths[indices].tolist(),
                  max_logit_difference=float(np.max(np.abs(scores-x[indices]))),
                  max_prediction_difference=float(np.max(np.abs(actual-expected))),
                  batch_size_prediction_difference=float(np.max(np.abs(one[0]-actual[0]))),
                  saved_model_hash_verified=True)
    write_json(out / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "predict", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--groups", type=Path, help="Optional filename -> identity group JSON; exact duplicates are also grouped.")
    parser.add_argument("--outer-folds", type=int, default=50)
    parser.add_argument("--inner-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUT / "new_predictions.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"prepare": prepare, "train": train, "predict": predict,
         "verify": lambda a: verify(a.output)}[args.stage](args)


if __name__ == "__main__":
    main()
