"""Multi-output regression of mean subjective ratings of generated faces.

Frozen FG-CLIP features are shared
across targets; output-specific regularization/ensembles are selected strictly
inside training folds. Outputs are continuous rating units, not probabilities.

API: from CLIP.race_perception import RacePerceptionPredictor
CLI: python -m CLIP.race_perception {prepare,train,predict,verify}
"""
from __future__ import annotations

import json

import numpy as np
from scipy.linalg import eigh

from .phrases import TARGETS
from ..fgclip2_face_impressions import (
    ImpressionPredictor, array_hash, cross_validate_predictor, make_cv_splits as cv_splits, scale_training,
)


def candidate_grid():
    """Predeclared candidates; kernel alpha uses kernels normalized by width."""
    bases = [dict(family="linear", bank="both", centered=False),
             dict(family="linear", bank="p256", centered=False),
             dict(family="poly2", bank="both", centered=False),
             dict(family="poly2", bank="p256", centered=False)]
    bases += [dict(family="rbf", bank="both", centered=centered, gamma=gamma)
              for centered, gamma in ((False, .1), (False, 1.),
                                     (True, .03), (True, .1), (True, .3), (True, 1.))]
    return [dict(base, alpha=float(alpha)) for base in bases
            for alpha in np.logspace(-4, 1, 11)]


def validate_xy(x, y=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or x.shape[1] < 2 or x.shape[1] % 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty X[n, 2 * phrases], p128 then p256.")
    if y is None:
        return x
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != len(x) or y.shape[1] == 0 or not np.isfinite(y).all():
        raise ValueError("Expected finite Y[n, targets] aligned with X.")
    return x, y


def base_key(recipe):
    return json.dumps({k: v for k, v in recipe.items() if k != "alpha"}, sort_keys=True)


def feature_view(x, recipe):
    p = x.shape[1] // 2
    if recipe["bank"] not in ("both", "p128", "p256"):
        raise ValueError("Unknown feature bank.")
    if recipe.get("centered", False):
        x = np.column_stack([x[:, :p] - x[:, :p].mean(axis=1, keepdims=True),
                             x[:, p:] - x[:, p:].mean(axis=1, keepdims=True)])
    return x if recipe["bank"] == "both" else x[:, :p] if recipe["bank"] == "p128" else x[:, p:]


def kernel(a, b, recipe):
    dot = a @ b.T / a.shape[1]
    if recipe["family"] == "linear":
        return dot
    if recipe["family"] == "poly2":
        return (1. + dot) ** 2
    if recipe["family"] == "rbf":
        distance = ((a * a).mean(axis=1)[:, None] + (b * b).mean(axis=1)[None, :] - 2 * dot)
        return np.exp(-recipe["gamma"] * np.maximum(distance, 0.))
    raise ValueError("Unknown kernel family.")


def prepare_kernel(train_x, recipe):
    z, mean, scale = scale_training(feature_view(train_x, recipe))
    gram = kernel(z, z, recipe)
    # Center nonlinear kernels as well: this gives a properly fitted intercept.
    km = gram.mean(axis=0)
    kg = float(km.mean())
    gram = gram - km[None, :] - km[:, None] + kg
    eigenvalues, vectors = eigh(gram, check_finite=False, driver="evr")
    state = dict(recipe=recipe, mean=mean, scale=scale, z=z, kernel_mean=km, kernel_grand_mean=kg)
    return state, np.maximum(eigenvalues, 0.), vectors


def test_kernel(state, x):
    z = (feature_view(x, state["recipe"]) - state["mean"]) / state["scale"]
    cross = kernel(z, state["z"], state["recipe"])
    return cross - cross.mean(axis=1, keepdims=True) - state["kernel_mean"][None, :] + state["kernel_grand_mean"]


def predict_candidates(train_x, train_y, test_x, recipes):
    """One eigendecomposition per kernel serves every alpha and every output."""
    train_x, train_y = validate_xy(train_x, train_y)
    test_x = validate_xy(test_x)
    if test_x.shape[1] != train_x.shape[1]:
        raise ValueError("Train and test feature widths differ.")
    out = np.empty((len(test_x), len(recipes), train_y.shape[1]))
    groups = {}
    for j, recipe in enumerate(recipes):
        if not np.isfinite(recipe["alpha"]) or recipe["alpha"] <= 0:
            raise ValueError("Regularization must be positive and finite.")
        groups.setdefault(base_key(recipe), []).append(j)
    target_mean = train_y.mean(axis=0)
    for indices in groups.values():
        state, values, vectors = prepare_kernel(train_x, recipes[indices[0]])
        left = test_kernel(state, test_x) @ vectors
        right = vectors.T @ (train_y - target_mean)
        for j in indices:
            out[:, j, :] = left @ (right / (values[:, None] + recipes[j]["alpha"])) + target_mean
    return out


def inner_predictions(x, y, recipes, folds, seed, groups):
    splits = cv_splits(len(y), folds, seed, groups)
    predictions, _ = cross_validate_predictor(
        x, y, lambda tx, ty, test, fold: predict_candidates(tx, ty, test, recipes), splits=splits)
    return predictions, splits


def select_strategies(predictions, y, recipes):
    """Output-specific MSE tuning, including predeclared two-family blends.

    A shared feature space need not imply equal penalties, a simplex, or an
    imposed correlation structure on these subjective dimensions.
    """
    losses = ((predictions - y[:, None, :]) ** 2).mean(axis=0)
    base_groups = {}
    for j, recipe in enumerate(recipes):
        base_groups.setdefault(base_key(recipe), []).append(j)
    strategies, ridge = [], []
    ridge_ids = [j for j, r in enumerate(recipes) if r["family"] == "linear"]
    for t in range(y.shape[1]):
        best = int(np.argmin(losses[:, t]))
        strategy = dict(indices=[best], weights=[1.], inner_mse=float(losses[best, t]))
        shortlist = [min(ids, key=lambda j: losses[j, t]) for ids in base_groups.values()]
        for a_pos, a in enumerate(shortlist):
            for b in shortlist[a_pos + 1:]:
                for weight in (.25, .5, .75):
                    pred = weight * predictions[:, a, t] + (1 - weight) * predictions[:, b, t]
                    loss = float(np.mean((pred - y[:, t]) ** 2))
                    if loss < strategy["inner_mse"]:
                        strategy = dict(indices=[a, b], weights=[weight, 1 - weight], inner_mse=loss)
        strategies.append(strategy)
        ridge.append(int(min(ridge_ids, key=lambda j: losses[j, t])))
    return strategies, ridge, losses


def apply_strategies(predictions, strategies):
    return np.column_stack([sum(w * predictions[:, j, t] for j, w in zip(s["indices"], s["weights"]))
                            for t, s in enumerate(strategies)])


def reference_predict(train_x, train_y, test_x, indices):
    result = np.empty((len(test_x), train_y.shape[1]))
    for t, column in enumerate(indices):
        a = np.column_stack([np.ones(len(train_x)), train_x[:, column]])
        coef = np.linalg.lstsq(a, train_y[:, t], rcond=None)[0]
        result[:, t] = coef[0] + coef[1] * test_x[:, column]
    return result


def fit_bundle_readout(x, y, recipes, strategies):
    """Fit all selected outputs together, sharing each kernel decomposition."""
    x, y = validate_xy(x, y)
    needed = sorted({j for s in strategies for j in s["indices"]})
    groups = {}
    for j in needed:
        groups.setdefault(base_key(recipes[j]), []).append(j)
    components = {}
    for indices in groups.values():
        state, values, vectors = prepare_kernel(x, recipes[indices[0]])
        target_mean = y.mean(axis=0)
        projected = vectors.T @ (y - target_mean)
        # Store the common training features once per kernel; coefficients differ
        # by alpha and output. All arrays are in original target units.
        coefficients = {j: vectors @ (projected / (values[:, None] + recipes[j]["alpha"])) for j in indices}
        components[base_key(recipes[indices[0]])] = dict(state=state, coefficients=coefficients, target_mean=target_mean)
    return dict(components=components, strategies=strategies, width=x.shape[1], outputs=y.shape[1])


def predict_readout(model, x):
    x = validate_xy(x)
    if x.shape[1] != model["width"]:
        raise ValueError("Feature width does not match training.")
    result = np.zeros((len(x), model["outputs"]))
    for component in model["components"].values():
        cross = test_kernel(component["state"], x)
        for j, coefficients in component["coefficients"].items():
            prediction = cross @ coefficients + component["target_mean"]
            for t, strategy in enumerate(model["strategies"]):
                for index, weight in zip(strategy["indices"], strategy["weights"]):
                    if index == j:
                        result[:, t] += weight * prediction[:, t]
    return result


class RacePerceptionPredictor(ImpressionPredictor):
    """Original kernel bundle; encoding and inference plumbing are shared."""
    def __init__(self, bundle):
        if bundle.get("format") != "fgclip2_subjective_race_v1" or tuple(bundle.get("targets", [])) != TARGETS:
            raise ValueError("Unrecognized subjective race-perception bundle.")
        super().__init__(bundle, readout_predict=predict_readout)


