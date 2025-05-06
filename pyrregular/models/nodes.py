import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def dropna(x):
    x = x.dropna()
    if len(x) == 0:
        # return a series of zeros for compatibility as models expect a non-empty series
        return pd.Series(pd.Series([0] * 1))
    return x


def to_float(x):
    return x.astype(np.float_)


def standardize(X):
    return (X - np.nanmean(X, axis=2, keepdims=True)) / (
        np.nanstd(X, axis=2, keepdims=True) + 1e-8
    )


class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class DropNATransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.applymap(dropna)


class ApplyFunc(BaseEstimator, TransformerMixin):
    def __init__(self, func, fn_kwargs=None):
        self.func = func
        self.fn_kwargs = fn_kwargs if fn_kwargs is not None else dict()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X, **self.fn_kwargs)
