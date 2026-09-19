"""Legacy FG-CLIP 2 research experiments, artifact generation, and reports.

This module intentionally contains the dataset-specific orchestration kept out of
``fgclip2_face_impressions``. Reusable encoding, CV, fitting, metrics, and
inference APIs remain in that module.

Run research stages from the project root with::

    python -m CLIP.fgclip2_legacy_research <stage> [options]
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .fgclip2_core import BASE_MODEL_ID, DEFAULT_MODEL_ID, FGCLIP2
    from .fgclip2_face_impressions import (
        DEFAULT_IMAGE_SUFFIXES,
        ImpressionPredictor,
        RegressionResults,
        RegressionDataset,
        center_groups,
        cross_validate_predictor,
        make_cv_splits,
        anchored_cv_splits,
        scale_training,
        predict_score_model,

        FaceDatasetConfig,
        FaceImpressionPipeline,
        HumanRatingsStore,
        _atomic_save_npz,
        extract_impression_features,
        fit_score_model,
        load_impression_features,
        paired_prediction_bootstrap,
        pearson_correlation,
        regression_metrics,
        slice_regression,
    )
except ImportError:
    from fgclip2_core import BASE_MODEL_ID, DEFAULT_MODEL_ID, FGCLIP2
    from fgclip2_face_impressions import (
        DEFAULT_IMAGE_SUFFIXES,
        ImpressionPredictor,
        RegressionResults,
        center_groups,
        cross_validate_predictor,
        make_cv_splits,
        anchored_cv_splits,
        scale_training,
        predict_score_model,

        FaceDatasetConfig,
        FaceImpressionPipeline,
        HumanRatingsStore,
        _atomic_save_npz,
        extract_impression_features,
        fit_score_model,
        load_impression_features,
        paired_prediction_bootstrap,
        pearson_correlation,
        regression_metrics,
        slice_regression,
    )


# Labels for the reusable readout methods returned by ``evaluate_impression``.
METHOD_LABELS = {
    "reference_linear": "Reference phrase · linear",
    "reference_isotonic": "Reference phrase · isotonic",
    "mean_ensemble": "Fixed signed mean · linear",
    "best_single": "Inner-selected single phrase",
    "ridge": "All 16 · ridge",
    "lasso": "All 16 · lasso",
    "evolved_ridge": "Evolved subset · ridge",
}

@dataclass(frozen=True)
class CVConfig:
    """Predeclared search budget. Every outer face is held out exactly once."""

    outer_folds: int = 10
    inner_folds: int = 5
    seed: int = 20260908
    ridge_alphas: tuple[float, ...] = tuple(np.logspace(-3, 4, 15))
    lasso_fractions: tuple[float, ...] = tuple(np.logspace(0, -4, 17))
    population: int = 32
    generations: int = 6
    elite: int = 8
    bootstrap_samples: int = 2000

    def __post_init__(self):
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Both CV levels need at least two folds.")
        if not 2 <= self.elite < self.population or self.generations < 1:
            raise ValueError("Require 2 <= elite < population and generations >= 1.")
        if not self.ridge_alphas or not all(np.isfinite(a) and a > 0 for a in self.ridge_alphas):
            raise ValueError("Ridge alphas must be finite and positive.")
        if not self.lasso_fractions or not all(0 < a <= 1 for a in self.lasso_fractions):
            raise ValueError("Lasso fractions must be in (0, 1].")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive.")


class InnerCVSearch:
    """Training-only search; cache standardized fold sufficient statistics.

    Independent of FG-CLIP: any frozen architecture's score matrix is accepted.
    Subset fitness is exact inner held-out MSE, with alpha selected jointly.
    """

    def __init__(self, x, y, config: CVConfig, seed: int, groups=None):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.x, self.y, self.config = x, y, config
        self.splits = make_cv_splits(len(y), config.inner_folds, seed, groups)
        self.folds = []
        for train, test in self.splits:
            z, mean, scale = scale_training(x[train])
            self.folds.append((z.T @ z, z.T @ (y[train] - y[train].mean()),
                               (x[test] - mean) / scale, y[train].mean(), y[test]))
        self.cache = {}

    def ridge(self, columns):
        columns = tuple(sorted(columns))
        if columns not in self.cache:
            losses = np.zeros(len(self.config.ridge_alphas))
            for gram, cross, test, mean_y, target in self.folds:
                eigenvalues, vectors = np.linalg.eigh(gram[np.ix_(columns, columns)])
                weights = vectors @ ((vectors.T @ cross[list(columns)])[:, None] /
                                     (np.maximum(eigenvalues, 0)[:, None] + self.config.ridge_alphas))
                error = test[:, columns] @ weights + mean_y - target[:, None]
                losses += np.sum(error ** 2, axis=0)
            index = int(np.argmin(losses))
            self.cache[columns] = (float(losses[index] / len(self.y)), float(self.config.ridge_alphas[index]))
        return self.cache[columns]

    def lasso(self):
        from sklearn.linear_model import lasso_path

        fractions = np.asarray(sorted(self.config.lasso_fractions, reverse=True))
        losses = np.zeros(len(fractions))
        for train, test in self.splits:
            z, mean, scale = scale_training(self.x[train])
            centered = self.y[train] - self.y[train].mean()
            alpha_max = max(np.max(np.abs(z.T @ centered)) / len(train), 1e-12)
            _, weights, gaps = lasso_path(z, centered, alphas=fractions * alpha_max,
                                          tol=1e-7, max_iter=50000)
            error = (self.x[test] - mean) / scale @ weights + self.y[train].mean() - self.y[test, None]
            losses += np.sum(error ** 2, axis=0)
        best = int(np.argmin(losses))
        return float(losses[best] / len(self.y)), float(fractions[best])

    def best_single(self):
        losses = []
        for column in range(self.x.shape[1]):
            loss = 0.0
            for train, test in self.splits:
                model = fit_score_model(self.x[train], self.y[train], "linear", [column])
                loss += np.sum((predict_score_model(model, self.x[test]) - self.y[test]) ** 2)
            losses.append(loss / len(self.y))
        return int(np.argmin(losses))

    def evolve(self, seed: int, initial_subsets=()):
        """Elitism + crossover + bit-flip mutation; retain reference phrase 0.

        Search is budgeted, not exhaustive over all 2**(p-1) subsets. Trace values
        describe inner optimization only; outer labels are never available here.
        """
        rng = np.random.default_rng(seed)
        p, cfg = self.x.shape[1], self.config
        def repair(mask):
            mask[0] = True
            return tuple(np.flatnonzero(mask).tolist())
        population = {tuple(range(p)), (0,)}
        for subset in initial_subsets:
            if not subset or min(subset) < 0 or max(subset) >= p:
                raise ValueError("Initial evolutionary subset outside feature space.")
            population.add(tuple(sorted(set(subset) | {0})))
        capacity = min(cfg.population, 2 ** (p - 1))
        while len(population) < capacity:
            population.add(repair(rng.random(p) < rng.uniform(.2, .9)))
        history = []
        for generation in range(cfg.generations):
            ranked = sorted(population, key=lambda c: (self.ridge(c)[0], len(c), c))
            best = ranked[0]
            history.append({"generation": generation, "inner_mse": self.ridge(best)[0],
                            "features": len(best), "evaluated_subsets": len(self.cache)})
            if generation == cfg.generations - 1:
                break
            elites = ranked[:min(cfg.elite, len(ranked))]
            population = set(elites)
            attempts = 0
            while len(population) < capacity:
                a, b = [elites[int(rng.integers(len(elites)))] for _ in range(2)]
                first, second = np.isin(np.arange(p), a), np.isin(np.arange(p), b)
                child = np.where(rng.random(p) < .5, first, second)
                child ^= rng.random(p) < 1 / p
                if attempts > capacity * 10:
                    child = rng.random(p) < .5
                population.add(repair(child))
                attempts += 1
        return best, self.ridge(best)[1], history


def fit_impression_readouts(x, y, config: CVConfig, *, seed=None, groups=None, directions=None):
    """Tune/final-fit all declared approaches on the supplied training data only."""
    seed = config.seed if seed is None else seed
    search = InnerCVSearch(x, y, config, seed, groups)
    columns = tuple(range(x.shape[1]))
    _, ridge_alpha = search.ridge(columns)
    _, lasso_fraction = search.lasso()
    best_single = search.best_single()
    subset, evolved_alpha, trace = search.evolve(seed)
    models = {
        "reference_linear": fit_score_model(x, y, "linear", [0]),
        "reference_isotonic": fit_score_model(x, y, "isotonic", [0]),
        "best_single": fit_score_model(x, y, "linear", [best_single]),
        "ridge": fit_score_model(x, y, "ridge", alpha=ridge_alpha),
        "lasso": fit_score_model(x, y, "lasso", alpha=lasso_fraction),
        "evolved_ridge": fit_score_model(x, y, "ridge", subset, evolved_alpha),
    }
    # Equal-weight signed mean of training-standardized score columns; calibrate
    # its slope/intercept on training ratings, then collapse to raw coefficients.
    z, mean, scale = scale_training(x)
    directions = np.ones(x.shape[1]) if directions is None else np.asarray(directions)
    if directions.shape != (x.shape[1],):
        raise ValueError("One fixed direction is required per phrase.")
    pooled = (z @ directions / x.shape[1])[:, None]
    pooled_model = fit_score_model(pooled, y, "linear")
    raw = pooled_model["weights"][0] * directions / (x.shape[1] * scale)
    models["mean_ensemble"] = {"kind": "linear", "columns": list(columns),
                               "weights": raw.tolist(),
                               "intercept": float(pooled_model["intercept"] - mean @ raw),
                               "standardized_weights": (raw * scale).tolist()}
    return models, trace


def evaluate_impression(regression: RegressionDataset, config=CVConfig(), *, groups=None, directions=None):
    """Leakage-safe nested CV of seven prespecified score-space approaches."""
    started = time.perf_counter()
    x, y = regression.predicted_scores.astype(float), regression.human_means.astype(float)
    if np.var(y) == 0 or x.shape[1] < 1:
        raise ValueError("A variable target and at least one predictor are required.")
    splits = make_cv_splits(len(y), config.outer_folds, config.seed, groups)
    details, methods = [], list(METHOD_LABELS)
    def train_predict(train_x, train_y, test_x, fold):
        train_groups = None if groups is None else np.asarray(groups)[splits[fold][0]]
        models, trace = fit_impression_readouts(train_x, train_y, config,
                                                seed=config.seed + 1009 * (fold + 1),
                                                groups=train_groups, directions=directions)
        details.append({"fold": fold, "models": models, "evolution": trace})
        return np.column_stack([predict_score_model(models[m], test_x) for m in methods])
    predictions, fold_ids = cross_validate_predictor(x, y, train_predict, splits=splits)
    cv_seconds = time.perf_counter() - started
    result = RegressionResults(y, dict(zip(methods, predictions.T)), (regression.variable,), fold_ids)
    bootstrap = {m: paired_prediction_bootstrap(y, predictions[:, 0], predictions[:, j],
                                                 samples=config.bootstrap_samples, seed=config.seed, groups=groups)
                 for j, m in enumerate(methods) if j > 0}
    versus_isotonic = {m: paired_prediction_bootstrap(y, predictions[:, 1], predictions[:, methods.index(m)],
                                                      samples=config.bootstrap_samples, seed=config.seed, groups=groups)
                        for m in ("ridge", "lasso", "evolved_ridge")}
    evolved_vs_ridge = paired_prediction_bootstrap(
        y, predictions[:, methods.index("ridge")], predictions[:, methods.index("evolved_ridge")],
        samples=config.bootstrap_samples, seed=config.seed, groups=groups)
    return {**_legacy_result(result), "variable": regression.variable, "texts": list(regression.texts),
            "config": asdict(config),
            "bootstrap_vs_reference": bootstrap, "bootstrap_vs_isotonic": versus_isotonic,
            "evolved_vs_ridge": evolved_vs_ridge,
            "fold_details": details, "cv_seconds": cv_seconds,
            "raw_reference_pearson": pearson_correlation(y, x[:, 0])}


def phrase_contributions(data, config=CVConfig()):
    """Retuned drop-one CV contribution, single-phrase fit and add-to-reference.

    Positive drop_delta_r2 means the phrase helps conditional on the remaining
    bank. Retraining and alpha retuning let correlated alternatives substitute;
    this is predictive utility, not causal importance. Call on discovery faces
    ONLY if the results will guide wording evaluated on a separate set.
    """
    x, y = np.asarray(data.predicted_scores, dtype=float), data.human_means
    p = x.shape[1]
    details = []
    def fit_predict(train_x, train_y, test_x, fold):
        search = InnerCVSearch(train_x, train_y, config, config.seed + 1009 * (fold+1))
        selections = [tuple(range(p))] + [tuple(k for k in range(p) if k != j) for j in range(p)]
        selections += [tuple(sorted({0, j})) for j in range(p)]
        predictions = []
        for columns in selections:
            _, alpha = search.ridge(columns)
            model = fit_score_model(train_x, train_y, 'ridge', columns, alpha)
            predictions.append(predict_score_model(model, test_x))
        for j in range(p):
            model = fit_score_model(train_x, train_y, 'linear', [j])
            predictions.append(predict_score_model(model, test_x))
        details.append({'fold': fold, 'full_ridge_alpha': search.ridge(tuple(range(p)))[1]})
        return np.column_stack(predictions)
    predictions, folds = cross_validate_predictor(x, y, fit_predict, folds=config.outer_folds, seed=config.seed)
    full = regression_metrics(y, predictions[:, 0])
    reference = regression_metrics(y, predictions[:, 1+p])
    records = []
    for j, phrase in enumerate(data.texts):
        drop = regression_metrics(y, predictions[:, 1+j])
        alone = regression_metrics(y, predictions[:, 1+2*p+j])
        added = regression_metrics(y, predictions[:, 1+p+j])
        # Sign convention is improvement from putting the dropped phrase back.
        ci = paired_prediction_bootstrap(y, predictions[:, 1+j], predictions[:, 0],
                                        samples=config.bootstrap_samples, seed=config.seed)
        records.append({'id': j+1, 'phrase': phrase, 'drop_delta_r2': full['r2']-drop['r2'],
                        'drop_delta_r2_ci': ci['delta_r2'], 'single_r': alone['pearson'],
                        'single_r2': alone['r2'], 'added_to_reference_delta_r2': added['r2']-reference['r2']})
    return {'full_metrics': full, 'rows': records, 'folds': folds.tolist(), 'n': len(y),
            'face_names': [p.name for p in data.paths], 'config': asdict(config)}


def development_protocol(paths, *, development_size=400, seed=20260909):
    """Freeze a label-independent set for adaptive wording; remainder is evaluation."""
    if not 2 <= development_size < len(paths)-10:
        raise ValueError('Development split leaves too few evaluation faces.')
    order = np.random.default_rng(seed).permutation(len(paths))
    development = sorted(order[:development_size].tolist())
    evaluation = sorted(order[development_size:].tolist())
    return {'seed': seed, 'development_indices': development, 'evaluation_indices': evaluation,
            'face_names': [p.name for p in paths],
            'primary': 'CV on evaluation faces; development faces always available for training',
            'secondary': 'Full-dataset CV is exploratory because wording used development labels',
            'previous_exposure': 'All faces appeared in the preceding Base experiment; this is not an untouched external test.'}


def compare_prompt_spaces(data, spaces, config=CVConfig(), *, protocol=None):
    """Matched folds for original, exploitative and exploratory prompt banks.

    With protocol, development faces are used in each outer training set but
    never scored. All tuning/elitism sees only that fold's training labels.
    Without protocol, ordinary all-face nested CV is descriptive after adaptation.
    """
    x, y = np.asarray(data.predicted_scores, dtype=float), np.asarray(data.human_means, dtype=float)
    if 'original16' not in spaces:
        raise ValueError('Provide the original16 parent bank for matched comparisons.')
    parent = list(spaces['original16'])
    for bank, columns in spaces.items():
        if not columns or len(set(columns)) != len(columns) or min(columns) < 0 or max(columns) >= x.shape[1]:
            raise ValueError(f'Invalid columns for {bank}.')
        if list(columns[:len(parent)]) != parent:
            raise ValueError('Every expanded bank must start with the same original columns.')
    if protocol is None:
        splits = make_cv_splits(len(y), config.outer_folds, config.seed)
        evaluated = np.arange(len(y))
    else:
        if protocol['face_names'] != [p.name for p in data.paths]:
            raise ValueError('Protocol face order differs from the score matrix.')
        splits = anchored_cv_splits(len(y), protocol['development_indices'], protocol['evaluation_indices'],
                                   folds=config.outer_folds, seed=config.seed)
        evaluated = np.asarray(protocol['evaluation_indices'])
    methods = ['reference_linear', 'reference_isotonic'] + [f'{bank}/{kind}' for bank in spaces for kind in ('ridge','lasso','evolved_ridge')]
    predictions = np.full((len(y), len(methods)), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    details, start = [], time.perf_counter()
    for fold, (train, test) in enumerate(splits):
        fold_models, histories = {}, {}
        for method, kind in [('reference_linear','linear'), ('reference_isotonic','isotonic')]:
            model = fit_score_model(x[train], y[train], kind, [0])
            predictions[test, methods.index(method)] = predict_score_model(model, x[test])
            fold_models[method] = model
        for bank, columns in spaces.items():
            xx = x[:, columns]
            search = InnerCVSearch(xx[train], y[train], config, config.seed+1009*(fold+1))
            _, alpha = search.ridge(tuple(range(len(columns))))
            _, fraction = search.lasso()
            subset, evolved_alpha, trace = search.evolve(config.seed+1009*(fold+1),
                                                        initial_subsets=[tuple(range(len(parent)))])
            for kind, selected, parameter in [('ridge',None,alpha),('lasso',None,fraction),('evolved_ridge',subset,evolved_alpha)]:
                model = fit_score_model(xx[train], y[train], 'ridge' if kind=='evolved_ridge' else kind, selected, parameter)
                # Convert local bank columns back to global phrase IDs for reuse.
                model['columns'] = [columns[j] for j in model['columns']]
                method = f'{bank}/{kind}'
                predictions[test, methods.index(method)] = predict_score_model(model,x[test])
                fold_models[method] = model
            histories[bank] = trace
        details.append({'fold':fold, 'models':fold_models, 'evolution':histories})
        fold_ids[test] = fold
    cv_seconds = time.perf_counter()-start
    if not np.isfinite(predictions[evaluated]).all() or np.any(fold_ids[evaluated]<0):
        raise RuntimeError('Incomplete outer validation predictions.')
    result = RegressionResults(y[evaluated], dict(zip(methods, predictions[evaluated].T)),
                               (data.variable,), fold_ids[evaluated])
    intervals = {}
    for bank in spaces:
        if bank == 'original16': continue
        for kind in ('ridge','lasso','evolved_ridge'):
            method, base = f'{bank}/{kind}', f'original16/{kind}'
            intervals[method] = paired_prediction_bootstrap(y[evaluated],predictions[evaluated,methods.index(base)],
                                                            predictions[evaluated,methods.index(method)],
                                                            samples=config.bootstrap_samples,seed=config.seed)
    direct = {}
    if 'similar32' in spaces and 'exploratory32' in spaces:
        direct = paired_prediction_bootstrap(
            y[evaluated],predictions[evaluated,methods.index('similar32/ridge')],
            predictions[evaluated,methods.index('exploratory32/ridge')],
            samples=config.bootstrap_samples,seed=config.seed)
    return {**_legacy_result(result), 'delta_ci_vs_original':intervals,
            'exploratory_vs_similar_ridge_ci':direct,
            'evaluated_indices':evaluated.tolist(), 'fold_details':details, 'cv_seconds':cv_seconds,
            'config':asdict(config), 'spaces':{k:list(v) for k,v in spaces.items()}}


def prompt_space_geometry(x, spaces):
    """Label-free descriptive novelty relative to the original phrase span."""
    original, _, _ = scale_training(x[:, spaces['original16']])
    values = {}
    for bank, columns in spaces.items():
        z, _, _ = scale_training(x[:, columns])
        eigenvalues = np.maximum(np.linalg.eigvalsh(z.T@z/len(z)),0)
        row = {'participation_rank':float(eigenvalues.sum()**2/np.sum(eigenvalues**2))}
        if bank!='original16':
            new = z[:,16:]
            residual = new-original@np.linalg.lstsq(original,new,rcond=None)[0]
            row['new_residual_variance_fraction'] = np.mean(residual**2,axis=0).tolist()
            row['median_new_residual_variance_fraction'] = float(np.median(np.mean(residual**2,axis=0)))
        values[bank] = row
    return values


def trust_recipe_grid():
    """Finite development budget; no tuning against hidden test stimuli."""
    recipes=[]
    def add(family,bank,**params):
        recipes.append(dict(family=family,bank=bank,**params))
    for bank in ('p128','p256','both'):
        for alpha in (1.,10.,100.,1000.,10000.):add('ridge',bank,alpha=alpha)
        for gamma in (.1,.3,1.,3.):
            for alpha in (.001,.01,.1,1.):add('rbf',bank,gamma=gamma,alpha=alpha)
        for gamma in (.1,.3,1.):
            for alpha in (.001,.01,.1):add('centered_rbf',bank,gamma=gamma,alpha=alpha)
    for k in (32,64,128):
        for alpha in (1.,10.,100.,1000.):add('screened_ridge','both',alpha=alpha,screen_k=k)
    for bank in ('p256','both'):
        for alpha in (.001,.01,.1,1.,10.):add('poly2',bank,alpha=alpha)
        for gamma in (.1,1.):
            for c in (.1,1.,10.):add('svr',bank,gamma=gamma,C=c,epsilon=.01)
    for leaves in (7,15):
        for l2 in (1.,10.):add('hgb','p256',leaves=leaves,l2=l2)
    return recipes


def trust_feature_columns(x,y,recipe):
    p=x.shape[1]//2
    columns=np.asarray(recipe.get('columns', list(range(p)) if recipe['bank']=='p128'
                                  else list(range(p,2*p)) if recipe['bank']=='p256'
                                  else list(range(2*p))),dtype=int)
    if len(columns)==0 or min(columns)<0 or max(columns)>=x.shape[1]:
        raise ValueError('Invalid trustworthiness feature indices.')
    if recipe.get('screen_k'):
        z,_,_=scale_training(x[:,columns])
        relevance=np.abs(z.T@(y-y.mean()))
        columns=columns[np.argsort(-relevance,kind='stable')[:recipe['screen_k']]]
    return columns


def fit_trust_recipe(x,y,recipe):
    """Fit training-only preprocessing and a small linear/nonlinear readout."""
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.ensemble import HistGradientBoostingRegressor
    x,y=np.asarray(x,float),np.asarray(y,float)
    columns=trust_feature_columns(x,y,recipe)
    family=recipe['family']
    if family in ('ridge','screened_ridge','evolved_groups'):
        estimator=Ridge(alpha=recipe['alpha'],fit_intercept=False,solver='cholesky')
    elif family in ('rbf','centered_rbf'):
        estimator=KernelRidge(alpha=recipe['alpha'],kernel='rbf',gamma=recipe['gamma']/len(columns))
    elif family=='poly2':
        estimator=KernelRidge(alpha=recipe['alpha'],kernel='polynomial',degree=2,gamma=1/len(columns),coef0=1.)
    elif family=='svr':
        estimator=SVR(C=recipe['C'],epsilon=recipe['epsilon'],gamma=recipe['gamma']/len(columns))
    elif family=='hgb':
        estimator=HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=recipe['leaves'],
                    l2_regularization=recipe['l2'],min_samples_leaf=20,early_stopping=False,random_state=20260910)
    else:raise ValueError(f'Unknown trustworthiness family {family}')
    feature_groups=np.repeat(np.arange(2),x.shape[1]//2) if recipe['family']=='centered_rbf' else None
    return dict(fit_score_model(x,y,columns=columns,estimator=estimator,feature_groups=feature_groups),recipe=recipe)


def predict_trust_readout(model,x):
    """Raw mean-rating prediction, including recursively stored convex ensembles."""
    if "kind" in model: return predict_score_model(model,x)
    if "transfer_kind" in model:
        return predict_trust_transfer(model,x)
    if 'components' in model:
        return sum(w*predict_trust_readout(m,x) for w,m in zip(model['weights'],model['components']))
    x=np.asarray(x,float)
    if model['recipe']['family']=='centered_rbf':
        x=trust_center_scores(x)
    z=(x[:,model['columns']]-model['mean'])/model['scale']
    return model['estimator'].predict(z)+model['target_mean']


def trust_center_scores(x):
    return center_groups(x, np.repeat(np.arange(2), x.shape[1]//2), axis=1)


def trust_candidate_cv(x,y,recipes,*,folds=5,seed=20260910,progress=False):
    def fit_predict(tx,ty,test,fold):
        return np.column_stack([predict_trust_readout(fit_trust_recipe(tx,ty,r),test) for r in recipes])
    return cross_validate_predictor(x,y,fit_predict,folds=folds,seed=seed)[0]


def choose_trust_strategy(predictions,y,recipes):
    """Inner-only selection of a recipe or a linear/nonlinear convex blend.

    Blend weights are picked on the same inner OOF matrix, so only the outer
    evaluation estimates performance of this whole selection procedure.
    """
    losses=np.mean((predictions-y[:,None])**2,axis=0)
    family_best={}
    for j,r in enumerate(recipes):
        family=r['family']
        if family not in family_best or losses[j]<losses[family_best[family]]:family_best[family]=j
    best=int(np.argmin(losses))
    choices=[{'indices':[best],'weights':[1.],'mse':float(losses[best])}]
    ranked=sorted(family_best.values(),key=lambda j:losses[j])[:3]
    if len(ranked)>1:
        for i in range(len(ranked)):
            for j in range(i+1,len(ranked)):
                for w in (.25,.5,.75):
                    p=w*predictions[:,ranked[i]]+(1-w)*predictions[:,ranked[j]]
                    choices.append({'indices':[ranked[i],ranked[j]],'weights':[w,1-w],
                                    'mse':float(np.mean((p-y)**2))})
        p=predictions[:,ranked].mean(axis=1)
        choices.append({'indices':ranked,'weights':[1/len(ranked)]*len(ranked),'mse':float(np.mean((p-y)**2))})
    return min(choices,key=lambda c:c['mse']),family_best,losses


def fit_trust_strategy(x,y,recipes,strategy):
    return {'components':[fit_trust_recipe(x,y,recipes[j]) for j in strategy['indices']],
            'weights':strategy['weights'],'strategy':strategy}


def trust_tail_metrics(y,prediction,fold_ids,high_mask,se=None):
    """Upper-tail errors and same-model close-rating pair concordance.

    These are rating-near pairs among distinct images, NOT within-identity
    latent traversals. Pair counts share faces and are not independent trials.
    """
    high_mask=np.asarray(high_mask,bool)
    result={'high_n':int(high_mask.sum()),'high_metrics':regression_metrics(y[high_mask],prediction[high_mask])}
    a,b=np.triu_indices(len(y),1)
    gap=y[b]-y[a]
    eligible=(fold_ids[a]==fold_ids[b])&high_mask[a]&high_mask[b]&(np.abs(gap)>=.01)&(np.abs(gap)<=.05)
    margin=(prediction[b]-prediction[a])*np.sign(gap)
    result['close_pairs']=int(eligible.sum())
    result['close_pair_concordance']=float(np.mean((margin[eligible]>0)+.5*(margin[eligible]==0))) if eligible.any() else None
    if se is not None:
        reliable=eligible&(np.abs(gap)>1.96*np.sqrt(se[a]**2+se[b]**2))
        result['noise_separated_pairs']=int(reliable.sum())
        result['noise_separated_concordance']=float(np.mean((margin[reliable]>0)+.5*(margin[reliable]==0))) if reliable.any() else None
    return result


def evolve_trust_groups(x,y,groups,*,seed=20260910):
    """Small development-only genetic search over the 16 literature cue families."""
    rng=np.random.default_rng(seed)
    p=x.shape[1]//2
    offsets=np.cumsum([0]+[len(g['phrases']) for g in groups])
    cache={}
    def columns(mask):
        local=[j for g in mask for j in range(offsets[g],offsets[g+1])]
        return local+[j+p for j in local]
    def fitness(mask):
        mask=tuple(sorted(mask))
        if mask not in cache:
            candidates=[dict(family='evolved_groups',bank='both',columns=columns(mask),alpha=a) for a in (10.,100.,1000.)]
            pred=trust_candidate_cv(x,y,candidates,folds=3,seed=seed)
            mse=np.mean((pred-y[:,None])**2,axis=0);best=int(np.argmin(mse))
            cache[mask]=(float(mse[best]),candidates[best])
        return cache[mask]
    population={tuple(range(len(groups)))}
    while len(population)<12:
        mask=tuple(np.flatnonzero(rng.random(len(groups))<.65).tolist())
        if len(mask)>=2:population.add(mask)
    trace=[]
    for generation in range(4):
        ranked=sorted(population,key=lambda m:fitness(m)[0])
        trace.append({'generation':generation+1,'mse':fitness(ranked[0])[0],'groups':list(ranked[0]),'evaluated':len(cache)})
        if generation==3:break
        population=set(ranked[:3])
        while len(population)<12:
            parents=[ranked[int(rng.integers(3))] for _ in range(2)]
            bits=np.where(rng.random(len(groups))<.5,np.isin(range(len(groups)),parents[0]),np.isin(range(len(groups)),parents[1]))
            bits^=rng.random(len(groups))<1/len(groups)
            mask=tuple(np.flatnonzero(bits).tolist())
            if len(mask)>=2:population.add(mask)
    best=min(cache,key=lambda m:cache[m][0])
    return cache[best][1],trace


IMPRESSION_DIRECTIONS = {name: (1,) * 12 + (-1,) * 4 for name in IMPRESSION_PROMPTS}



def research_main():
    import argparse
    import gc
    import platform

    parser = argparse.ArgumentParser(description="Frozen FG-CLIP prompt ensemble research; no UI required.")
    parser.add_argument("stage", choices=("encode",))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "research" / "impression_ensembles")
    parser.add_argument("--images", type=Path, default=Path("/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images"))
    parser.add_argument("--ratings", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "dim_to_photo_to_ratings.pkl")
    parser.add_argument("--models", nargs="+", choices=("base", "so400m"), default=["base"])
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--patches", type=int, default=128)
    parser.add_argument("--memory-reserve-gb", type=float, default=1.25)
    parser.add_argument("--bundle", type=Path, help="Frozen trustworthiness model bundle for trust-predict.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = CVConfig(seed=args.seed)
    if args.stage == "encode":
        for name in args.models:
            print(f"Loading {name} on MPS", flush=True)
            engine = FGCLIP2(BASE_MODEL_ID if name == "base" else DEFAULT_MODEL_ID,
                            device="mps", local_files_only=True,
                            memory_reserve_gb=args.memory_reserve_gb)
            pipeline = FaceImpressionPipeline(engine, FaceDatasetConfig(args.images, limit=None))
            runtime = extract_impression_features(pipeline, HumanRatingsStore(args.ratings), IMPRESSION_PROMPTS,
                                                   args.output / name, patches=args.patches)
            print(json.dumps(runtime), flush=True)
            del pipeline, engine
            gc.collect()
            torch.mps.empty_cache()
        return
    records = []
    from threadpoolctl import threadpool_limits
    # Small 16x16 systems are faster without a large BLAS thread pool.
    with threadpool_limits(limits=1):
        for name in args.models:
            for variable in IMPRESSION_PROMPTS:
                data, metadata = load_impression_features(args.output / name / f"{variable}-features.npz")
                directory = args.output / name / variable
                if args.stage == "evaluate":
                    print(f"Nested CV: {name}/{variable}, n={len(data.paths)}", flush=True)
                    result = evaluate_impression(data, config, directions=IMPRESSION_DIRECTIONS[variable])
                    models, _ = fit_impression_readouts(data.predicted_scores, data.human_means, config,
                                                        directions=IMPRESSION_DIRECTIONS[variable])
                    save_research_result(data, result, metadata, directory, final_models=models)
                    print(json.dumps({"cv_seconds": result["cv_seconds"], "metrics": result["metrics"]}), flush=True)
                else:
                    result = json.loads((directory / "results.json").read_text())
                    import csv
                    rows = list(csv.DictReader((directory / "oof_predictions.csv").open()))
                    result["predictions"] = np.asarray([[float(row[m]) for m in result["methods"]] for row in rows])
                    result["fold_ids"] = np.asarray([int(row["fold"]) for row in rows])
                records.append((f"{name.upper()} / {variable}", data, result))
    if args.stage == "evaluate":
        import sklearn, scipy
        manifest = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                    "sklearn": sklearn.__version__, "scipy": scipy.__version__, "config": asdict(config),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "command": __import__("sys").argv, "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
        (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_research_report(records, args.output)
    print(f"Report: {args.output / 'report.html'}", flush=True)





__all__ = [
    'research_main',
    'IMPRESSION_PROMPTS',
    'IMPRESSION_DIRECTIONS',
]


if __name__ == "__main__":
    research_main()
