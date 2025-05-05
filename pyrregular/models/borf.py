from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMClassifier
from aeon.transformations.collection.dictionary_based import BORF
from pyrregular.models.nodes import to_float


borf_pipeline = make_pipeline(
    BORF(),
    FunctionTransformer(func=to_float),
    LGBMClassifier(
        n_jobs=1,
    ),
)
