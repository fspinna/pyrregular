"""SVM Pipeline.
Supports LCSS kernel and uses a custom TimeSeriesSVC class to handle the kernel.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sktime.classification.kernel_based import TimeSeriesSVC
from sktime.datatypes import convert_to
from sktime.dists_kernels.lcss import LcssTslearn

from pyrregular.models.nodes import ApplyFunc, DropNATransformer, _standardize


class TimeSeriesSVCFix(TimeSeriesSVC):

    def predict_proba(self, X):
        # one-hot of the predicted class, at the position of its label in classes_
        positions = np.searchsorted(self.classes_, self.predict(X))
        return np.eye(len(self.classes_))[positions]


class _SktimeClassifier(ClassifierMixin, BaseEstimator):
    """Run an sktime classifier as the last step of a scikit-learn Pipeline."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = self.estimator.clone()
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


svm_pipeline = Pipeline(
    [
        ("standardize", ApplyFunc(func=_standardize)),
        (
            "convert_to_nested",
            ApplyFunc(func=convert_to, fn_kwargs={"to_type": "nested_univ"}),
        ),
        ("drop_na", DropNATransformer()),
        (
            "svc",
            _SktimeClassifier(
                TimeSeriesSVCFix(
                    kernel=LcssTslearn(
                        global_constraint="sakoe_chiba", sakoe_chiba_radius=10
                    ),
                    max_iter=1000,
                )
            ),
        ),
    ]
)
"""This pipeline applies standardize → convert_to_nested → drop_na → TimeSeriesSVC with LCSS kernel."""
