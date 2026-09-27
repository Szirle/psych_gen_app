"""Two-level measurement diagnostics for visual cues and human impressions.

Pure analysis imports do not load PyTorch or a vision model. See docs/METHODOLOGY.md.
"""
from .schema import Column, Dataset, Registry
from .features import Representation
from .evaluation import ComparisonConfig, compare

__all__ = ['Column', 'Dataset', 'Registry', 'Representation', 'ComparisonConfig', 'compare']
