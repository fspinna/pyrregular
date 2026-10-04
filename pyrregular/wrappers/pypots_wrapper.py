import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pyrregular.conversion_utils import _to_pypots


class PyPOTSWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, model_params, random_state=None):
        self.model = model
        self.model_params = model_params
        self.random_state = random_state

        self.n_classes_ = None
        self.n_steps_ = None
        self.n_features_ = None

    def fit(self, X, y):
        # pypots needs labels 0..k-1: encode here, decode in predict
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        y = self.label_encoder_.transform(y)
        self.n_classes_ = len(self.classes_)
        self.n_steps_ = X.shape[2]
        self.n_features_ = X.shape[1]
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        # the fitted model goes in model_, so that `model` stays the class and
        # fit can be called again
        self.model_ = self.model(**self.model_params)
        self.model_.fit(_to_pypots(X, y))

    def predict_proba(self, X):
        out = self.model_.predict(_to_pypots(X))["classification_proba"]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _split(self, X, y):
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, stratify=y, random_state=self.random_state
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, random_state=self.random_state
            )
        return _to_pypots(X_train, y_train), _to_pypots(X_val, y_val)
