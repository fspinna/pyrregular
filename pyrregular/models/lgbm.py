from lightgbm import LGBMClassifier
from sktime.pipeline import make_pipeline
from sktime.transformations.panel.reduce import Tabularizer

lgbm_pipeline = make_pipeline(
    Tabularizer(),
    LGBMClassifier(
        n_jobs=1,
    ),
)
