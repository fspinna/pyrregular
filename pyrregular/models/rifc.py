"""RIFC Pipeline.
Baseline classifier that uses RandomIntervalFeatureExtractor to extract simple statistical features from windows.
"""

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor


def _nanskew(x):
    return skew(x, nan_policy="omit")


def _nankurtosis(x):
    return kurtosis(x, nan_policy="omit")


class RandomIntervalFeatureClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        features=(
            np.nanmean,
            np.nanstd,
            np.nanmin,
            np.nanmax,
            np.nanmedian,
            _nanskew,
            _nankurtosis,
        ),
        random_state=None,
        n_intervals="log",
    ):
        self.features = features
        self.random_state = random_state
        self.n_intervals = n_intervals

    def fit(self, X, y):
        # built here, not in __init__, so that set_params/clone take effect
        self.transformer_ = RandomIntervalFeatureExtractor(
            features=list(self.features),
            random_state=self.random_state,
            n_intervals=self.n_intervals,
        )
        self.clf_ = LGBMClassifier(n_jobs=1, random_state=self.random_state)
        X = self.transformer_.fit_transform(X)
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        X = self.transformer_.transform(X)
        return self.clf_.predict(X)

    def predict_proba(self, X):
        X = self.transformer_.transform(X)
        return self.clf_.predict_proba(X)


rifc_pipeline = make_pipeline(RandomIntervalFeatureClassifier())
"""This pipeline applies RandomIntervalFeatureClassifier."""
