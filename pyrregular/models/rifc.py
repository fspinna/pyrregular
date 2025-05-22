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
        self.transformer = RandomIntervalFeatureExtractor(
            features=list(self.features),
            random_state=self.random_state,
            n_intervals=n_intervals,
        )
        self.clf = LGBMClassifier(n_jobs=1, random_state=self.random_state)

    def fit(self, X, y):
        X = self.transformer.fit_transform(X)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = self.transformer.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = self.transformer.transform(X)
        return self.clf.predict_proba(X)


rifc_pipeline = make_pipeline(RandomIntervalFeatureClassifier())
"""This pipeline applies RandomIntervalFeatureClassifier."""
