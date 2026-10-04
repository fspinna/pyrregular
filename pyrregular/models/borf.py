"""BORF pipelines.
Bag-Of-Receptive-Fields features (fast_borf) with a LightGBM classifier.
``iborf_pipeline`` is the time-aware variant for irregular series (ridge head).
"""

import numpy as np
from fast_borf import BORF
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from pyrregular.models.nodes import _to_float
from pyrregular.models.ridge_cv import _RidgeClassifierCVFix

borf_pipeline = make_pipeline(
    BORF(),
    FunctionTransformer(func=_to_float),
    LGBMClassifier(
        n_jobs=1,
    ),
)
"""This pipeline applies BORF → to_float → LGBMClassifier. Input: X from to_dense()."""

iborf_pipeline = make_pipeline(
    BORF(time_channel=True, min_window_to_signal_std_ratio=0.15),
    FunctionTransformer(func=np.arcsinh),
    _RidgeClassifierCVFix(),
)
"""This pipeline applies time-aware BORF (I-BORF) → arcsinh → RidgeClassifierCV, as in
Spinnato, "A Time-Aware Bag-of-Receptive-Fields for Interpretable Irregular Time
Series Classification". Input: X from to_dense(concatenate_time=True,
normalize_time=True), where the last channel holds the timestamps."""
