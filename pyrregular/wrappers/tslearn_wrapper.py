from sklearn.base import BaseEstimator, ClassifierMixin

from pyrregular.conversion_utils import _to_tslearn


class TslearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(_to_tslearn(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(_to_tslearn(X))

    def predict(self, X):
        return self.model.predict(_to_tslearn(X))
