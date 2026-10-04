"""BORF pipeline (aeon implementation, as used in the pyrregular paper).
multivariate dictionary based transformer based on Bag-Of-Receptive-Fields transform.
"""

from aeon.transformations.collection.dictionary_based import BORF
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from pyrregular.models.nodes import _to_float

aeon_borf_pipeline = make_pipeline(
    BORF(),
    FunctionTransformer(func=_to_float),
    LGBMClassifier(
        n_jobs=1,
        random_state=0,
    ),
)
"""This pipeline applies BORF → to_float → LGBMClassifier."""
