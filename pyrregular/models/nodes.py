"""Nodes.
This module contains various pipeline transformer classes for data preprocessing and transformation.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _dropna(x):
    x = x.dropna()
    if len(x) == 0:
        # return a series of zeros for compatibility as models expect a non-empty series
        return pd.Series(pd.Series([0] * 1))
    return x


def _to_float(x):
    return x.astype(np.float_)


def _standardize(X):
    return (X - np.nanmean(X, axis=2, keepdims=True)) / (
        np.nanstd(X, axis=2, keepdims=True) + 1e-8
    )


class PassthroughTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that returns the input data unchanged.

    This transformer is useful as a placeholder or when no transformation is needed.
    """

    def fit(self, X, y=None):
        """
        Fit the transformer. This transformer does not learn anything.

        Args:
            X: Input data.
            y: Target values (default is None).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Return the input data unchanged.

        Args:
            X: Input data.

        Returns:
            The unchanged input data.
        """
        return X


class DropNATransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a function to drop or handle NA values in a DataFrame.

    This transformer applies the `_dropna` function element-wise to the input DataFrame.
    """

    def fit(self, X, y=None):
        """
        Fit the transformer. This transformer does not learn anything.

        Args:
            X: Input data.
            y: Target values (default is None).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Apply the `_dropna` function element-wise to the DataFrame.

        Args:
            X: Input pandas DataFrame.

        Returns:
            Transformed DataFrame with `_dropna` applied to each element.
        """
        return X.applymap(_dropna)


class ApplyFunc(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a specified function to the input data.

    Args:
        func (callable): The function to apply to the input data.
        fn_kwargs (dict, optional): Additional keyword arguments to pass to the function.
    """

    def __init__(self, func, fn_kwargs=None):
        """
        Initialize the ApplyFunc transformer.

        Args:
            func (callable): The function to apply.
            fn_kwargs (dict, optional): Keyword arguments for the function (default is None).
        """
        self.func = func
        self.fn_kwargs = fn_kwargs if fn_kwargs is not None else dict()

    def fit(self, X, y=None):
        """
        Fit the transformer. This transformer does not learn anything.

        Args:
            X: Input data.
            y: Target values (default is None).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Apply the specified function to the input data.

        Args:
            X: Input data.

        Returns:
            The result of applying `func` to `X` with the specified keyword arguments.
        """
        return self.func(X, **self.fn_kwargs)
