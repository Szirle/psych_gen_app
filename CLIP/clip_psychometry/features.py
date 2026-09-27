"""Fold-local response scales and deliberate metric weights; chapters 4 and 7."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Representation:
    columns: tuple[int, ...]
    transform: str = 'raw'  # raw, rank, hinge
    quantile: float = .5
    balance_blocks: bool = False
    # Optional explicit partition; otherwise registry cue names define blocks.
    blocks: tuple[str, ...] | None = None
    # Squared-distance weights aligned with columns; normalized to mean one.
    metric_weights: tuple[float, ...] | None = None
    # Add standardized differences, preserving every original coordinate.
    contrast_pairs: tuple[tuple[int, int], ...] = ()
    contrast_weight: float = 1.
    normalize_contrast_energy: bool = False
    # Return metric coordinates followed by untouched standardized coordinates.
    separate_metric: bool = False


class FeatureMap:
    def __init__(self, spec, registry):
        self.spec, self.registry = spec, registry
        if spec.transform not in ('raw', 'rank', 'hinge') or not 0 <= spec.quantile <= 1:
            raise ValueError('Unknown transform or invalid quantile.')
        if len(set(spec.columns)) != len(spec.columns) or any(c < 0 or c >= len(registry.items) for c in spec.columns):
            raise ValueError('Representation columns must be unique valid indices.')

    def _raw(self, x):
        z = np.asarray(x[:, self.spec.columns], float).copy()
        if self.spec.transform == 'hinge':
            return np.maximum(z-self.threshold, 0.)
        if self.spec.transform == 'rank':
            return np.column_stack([ecdf(self.reference[:, j], z[:, j]) for j in range(z.shape[1])])
        return z

    def fit(self, x):
        selected = np.asarray(x[:, self.spec.columns], float)
        self.reference = np.sort(selected, axis=0)
        self.threshold = np.quantile(selected, self.spec.quantile, axis=0) if selected.shape[1] else np.empty(0)
        z = self._raw(x) if selected.shape[1] else selected
        self.mean = z.mean(0)
        scale = z.std(0)
        self.scale = np.where(scale > 1e-12, scale, 1.)
        self.weight = np.ones(z.shape[1])
        if self.spec.balance_blocks and z.shape[1]:
            blocks = self.spec.blocks or tuple(self.registry.items[c].cue for c in self.spec.columns)
            if len(blocks) != z.shape[1]:
                raise ValueError('One disjoint metric block per selected column is required.')
            names, counts = np.unique(blocks, return_counts=True)
            count = dict(zip(names, counts))
            # Kernel divides squared distance/dot product by total width.
            self.weight = np.sqrt([z.shape[1]/(len(names)*count[b]) for b in blocks])
        if self.spec.metric_weights is not None:
            weights = np.asarray(self.spec.metric_weights, float)
            if weights.shape != (z.shape[1],) or not np.isfinite(weights).all() or np.any(weights < 0) or not weights.sum():
                raise ValueError('Need aligned nonnegative metric weights with positive total.')
            self.weight *= np.sqrt(weights/weights.mean())
        if self.spec.contrast_pairs:
            position = {c:i for i,c in enumerate(self.spec.columns)}
            self.pairs = [(position[a], position[b]) for a,b in self.spec.contrast_pairs]
            standard = (z-self.mean)/self.scale
            delta = np.column_stack([standard[:,a]-standard[:,b] for a,b in self.pairs])
            self.delta_scale = np.maximum(delta.std(0), 1e-8)
            if not np.isfinite(self.spec.contrast_weight) or self.spec.contrast_weight < 0:
                raise ValueError('Contrast weight must be finite and nonnegative.')
        if self.spec.separate_metric and self.spec.contrast_pairs:
            raise ValueError('Separate metric currently requires equal metric/raw widths.')
        return self

    def transform(self, x):
        if not self.spec.columns:
            return np.empty((len(x), 0))
        raw = (self._raw(x)-self.mean)/self.scale
        metric = raw*self.weight
        if self.spec.separate_metric:
            return np.column_stack((metric, raw))
        if self.spec.contrast_pairs:
            delta = np.column_stack([raw[:,a]-raw[:,b] for a,b in self.pairs])/self.delta_scale
            augmented = np.column_stack((metric, delta*np.sqrt(self.spec.contrast_weight)))
            if self.spec.normalize_contrast_energy:
                augmented *= np.sqrt(augmented.shape[1]/(np.sum(self.weight**2)+len(self.pairs)*self.spec.contrast_weight))
            return augmented
        return metric


def ecdf(training, values):
    """Mid-distribution ranks with training ties; range [0,1], not probabilities."""
    reference = np.sort(np.asarray(training, float))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError('A finite nonempty training reference is required.')
    return (np.searchsorted(reference, values, side='left')+
            np.searchsorted(reference, values, side='right'))/(2*len(reference))


def conjunction(training_a, training_b, a, b):
    return np.minimum(ecdf(training_a, a), ecdf(training_b, b))
