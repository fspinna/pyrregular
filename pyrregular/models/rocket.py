from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
from lightgbm import LGBMClassifier
from sktime.pipeline import make_pipeline


rocket_pipeline = make_pipeline(
    MiniRocketMultivariateVariable(
        pad_value_short_series=0,
    ),
    LGBMClassifier(n_jobs=1)
)
