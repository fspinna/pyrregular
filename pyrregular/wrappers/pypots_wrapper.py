import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from pyrregular.conversion_utils import to_pypots


class PyPOTSWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, model_params, random_state=None):
        self.model = model
        self.model_params = model_params
        self.random_state = random_state

        self.n_classes_ = None
        self.n_steps_ = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_steps_ = X.shape[2]
        self.n_features_ = X.shape[1]
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        self.model = self.model(**self.model_params)
        self.model.fit(to_pypots(X, y))

    def predict_proba(self, X):
        out = self.model.predict(to_pypots(X))["classification_proba"]
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _split(self, X, y):
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, stratify=y, random_state=self.random_state
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, random_state=self.random_state
            )
        return to_pypots(X_train, y_train), to_pypots(X_val, y_val)
