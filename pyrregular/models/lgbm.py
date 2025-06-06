"""LGBM Pipeline.
Simple LightGBM classifier that treats time series as tabular data.
"""

from lightgbm import LGBMClassifier
from sktime.pipeline import make_pipeline
from sktime.transformations.panel.reduce import Tabularizer

lgbm_pipeline = make_pipeline(
    Tabularizer(),
    LGBMClassifier(
        n_jobs=1,
    ),
)
"""This pipeline applies Tabularizer → LGBMClassifier."""
