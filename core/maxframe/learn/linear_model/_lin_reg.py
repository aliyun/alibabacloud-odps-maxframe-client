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

from numpy.linalg import LinAlgError

from ... import tensor as mt
from ..core import RegressorMixin
from ..utils.validation import _check_sample_weight
from ._base import LinearModel, _rescale_data

try:
    from sklearn.base import MultiOutputMixin
except ImportError:
    MultiOutputMixin = type("MultiOutputMixin", (object,), {})


class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.
    """

    _data_attributes = ["coef_", "intercept_"]

    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize=False,
        copy_X=True,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive

    def fit(
        self, X, y, sample_weight=None, session=None, run_kwargs=None, execute=False
    ):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
            return_mean=True,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if self.positive:
            # TODO: implement optimize.nnls first
            raise NotImplementedError("Does not support positive coefficients!")
        elif X.issparse():
            # TODO: implement sparse.linalg.lsqr first
            raise NotImplementedError("Does not support sparse input!")
        else:
            # TODO: implement linalg.lstsq first
            # In numpy:
            #   Mat mul does NOT always satisfy associative law
            # Tyipical mistake:
            #   (mt.linalg.inv(X.T @ X) @ (X.T @ y)).T
            self.coef_ = (mt.linalg.inv(X.T @ X) @ X.T @ y).T

        if y.ndim == 1:
            self.coef_ = mt.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        if execute:
            run_kwargs = run_kwargs or {}
            try:
                self.execute(session=session, **run_kwargs)
            except LinAlgError:
                # TODO: implement linalg.lstsq first
                raise NotImplementedError("Does not support singular matrix!")
        return self
