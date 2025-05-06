import numpy as np
from sklearn.pipeline import Pipeline
from sktime.classification.kernel_based import TimeSeriesSVC
from sktime.datatypes import convert_to
from sktime.dists_kernels.lcss import LcssTslearn

from pyrregular.models.nodes import ApplyFunc, DropNATransformer, standardize


class TimeSeriesSVCFix(TimeSeriesSVC):

    def predict_proba(self, X):
        return np.eye(len(self.classes_))[self.predict(X)]


svm_pipeline = Pipeline(
    [
        ("standardize", ApplyFunc(func=standardize)),
        (
            "convert_to_nested",
            ApplyFunc(func=convert_to, fn_kwargs={"to_type": "nested_univ"}),
        ),
        ("drop_na", DropNATransformer()),
        (
            "svc",
            TimeSeriesSVCFix(
                kernel=LcssTslearn(
                    global_constraint="sakoe_chiba", sakoe_chiba_radius=10
                ),
                max_iter=1000,
            ),
        ),
    ]
)
