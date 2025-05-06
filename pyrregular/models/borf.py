"""BORF pipeline.
multivariate dictionary based transformer based on Bag-Of-Receptive-Fields transform.
"""

from aeon.transformations.collection.dictionary_based import BORF
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from pyrregular.models.nodes import to_float

borf_pipeline = make_pipeline(
    BORF(),
    FunctionTransformer(func=to_float),
    LGBMClassifier(
        n_jobs=1,
    ),
)
"""This pipeline applies BORF → to_float → LGBMClassifier."""
