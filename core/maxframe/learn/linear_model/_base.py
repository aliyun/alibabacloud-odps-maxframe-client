# Copyright 1999-2025 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
from abc import ABCMeta, abstractmethod

from ... import tensor as mt
from ...tensor.datasource import tensor as astensor
from ..core import BaseEstimator, ClassifierMixin
from ..preprocessing import normalize as f_normalize
from ..utils.validation import FLOAT_DTYPES, check_array

try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:
    check_is_fitted = lambda *_: True


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    sample_weight=None,
    return_mean=False,
    check_input=True,
):
    """Center and scale data.

    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output

        X = (X - X_offset) / X_scale

    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).

    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = astensor(sample_weight)

    X = astensor(X)
    y = astensor(y, dtype=X.dtype)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    elif copy:
        if X.issparse():
            X = X.copy()
        else:
            X = X.copy(order="K")

    if fit_intercept:
        if X.issparse():
            raise NotImplementedError("Does not support sparse input!")
        else:
            X_offset = mt.average(X, axis=0, weights=sample_weight)
            X = X - X_offset
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False, return_norm=True)
            else:
                X_scale = mt.ones(X.shape[1], dtype=X.dtype)
        y_offset = mt.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        if X.issparse():
            raise NotImplementedError("Does not support sparse input!")
        X_offset = mt.zeros(X.shape[1], dtype=X.dtype)
        X_scale = mt.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = mt.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.

    For many linear models, this enables easy support for sample_weight.

    Returns
    -------
    X_rescaled : {array-like, sparse matrix}

    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight = mt.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = mt.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight = mt.sqrt(sample_weight)
    sw_matrix = mt.diag(sample_weight, sparse=True)
    X = mt.dot(sw_matrix, X)
    y = mt.dot(sw_matrix, y)
    return X, y


class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(
            X, y="no_validation", accept_sparse=["csr", "csc", "coo"], reset=False
        )
        return mt.dot(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    _preprocess_data = staticmethod(_preprocess_data)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - mt.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = mt.tensor(0.0)

    def _more_tags(self):  # noqa: R0201  # pylint: disable=no-self-use
        return {"requires_y": True}


class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.

    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse="csr")

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (X.shape[1], n_features)
            )

        scores = mt.dot(X, self.coef_.T) + self.intercept_
        return scores

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        indices = scores.argmax(axis=1)
        return self.classes_[indices].execute()
