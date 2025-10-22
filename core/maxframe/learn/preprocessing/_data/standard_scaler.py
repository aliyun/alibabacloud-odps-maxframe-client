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

import numpy as np

from .... import tensor as mt
from ....core import ENTITY_TYPE
from ....lib import sparse
from ...core import BaseEstimator, TransformerMixin
from ...utils.extmath import _incremental_mean_and_var
from ...utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_array,
    check_is_fitted,
)
from .utils import _handle_zeros_in_scale, _is_constant_feature


class StandardScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

    .. code-block:: text

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    `StandardScaler` is sensitive to outliers, and the features may scale
    differently from each other in the presence of outliers. For an example
    visualization, refer to :ref:`Compare StandardScaler with other scalers
    <plot_all_scaling_standard_scaler_section>`.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False`` and ``with_std=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_mean=False`` and
        ``with_std=False``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    See Also
    --------
    scale : Equivalent function without the estimator API.

    :class:`~sklearn.decomposition.PCA` : Further removes the linear
        correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    Examples
    --------
    >>> from maxframe.learn.preprocessing import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_.execute())
    [0.5 0.5]
    >>> print(scaler.transform(data).execute())
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]).execute())
    [[3. 3.]]
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True, validate=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.validate = validate

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(
        self,
        X,
        y=None,
        sample_weight=None,
        execute=False,
        session=None,
        run_kwargs=None,
    ) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(
            X, y, sample_weight, execute=execute, session=session, run_kwargs=run_kwargs
        )

    def partial_fit(
        self,
        X,
        y=None,
        sample_weight=None,
        execute=False,
        session=None,
        run_kwargs=None,
    ):
        """Online computation of mean and std on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        first_call = not hasattr(self, "n_samples_seen_")
        if self.validate:
            X = self._validate_data(
                X,
                accept_sparse=("csr", "csc"),
                dtype=FLOAT_DTYPES,
                force_all_finite="allow-nan",
                reset=first_call,
            )
        n_features = X.shape[1] if X.ndim == 2 else 1

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        # if n_samples_seen_ is an integer (i.e. no missing values), we need to
        # transform it to a NumPy array of shape (n_features,) required by
        # incr_mean_variance_axis and _incremental_variance_axis
        dtype = np.int64 if sample_weight is None else X.dtype
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = (
                mt.zeros(n_features, dtype=dtype) if X.ndim == 2 else 0
            )
        # elif np.size(self.n_samples_seen_) == 1:
        #     self.n_samples_seen_ = np.repeat(self.n_samples_seen_, X.shape[1])
        #     self.n_samples_seen_ = self.n_samples_seen_.astype(dtype, copy=False)

        if sparse.issparse(X):
            raise NotImplementedError("Scaling on sparse tensors is not supported")
        else:
            # First pass
            if not hasattr(self, "scale_"):
                self.mean_ = 0.0
                if self.with_std:
                    self.var_ = 0.0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - mt.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = _incremental_mean_and_var(
                    X,
                    self.mean_,
                    self.var_,
                    self.n_samples_seen_,
                    sample_weight=sample_weight,
                )

        # # for backward-compatibility, reduce n_samples_seen_ to an integer
        # # if the number of samples is the same for each feature (i.e. no
        # # missing values)
        # if np.ptp(self.n_samples_seen_) == 0:
        #     self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_std:
            # Extract the list of near constant features on the raw variances,
            # before taking the square root.
            constant_mask = _is_constant_feature(
                self.var_, self.mean_, self.n_samples_seen_
            )
            self.scale_ = mt.sqrt(self.var_)
            if self.validate:
                self.scale_ = _handle_zeros_in_scale(
                    self.scale_, copy=False, constant_mask=constant_mask
                )
        else:
            self.scale_ = None

        if execute:
            self.execute(session=session, **(run_kwargs or dict()))
        return self

    def transform(self, X, copy=None, execute=False, session=None, run_kwargs=None):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        if self.validate:
            X = self._validate_data(
                X,
                reset=False,
                accept_sparse="csr",
                copy=copy,
                dtype=FLOAT_DTYPES,
                force_all_finite="allow-nan",
            )

        if sparse.issparse(X):
            raise NotImplementedError("Scaling on sparse tensors is not supported")
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        if execute:
            X.execute(session=session, **(run_kwargs or dict()))
        return X

    def inverse_transform(
        self, X, copy=None, execute=False, session=None, run_kwargs=None
    ):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(
            X,
            accept_sparse="csr",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            raise NotImplementedError("Scaling on sparse tensors is not supported")
        else:
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        if execute:
            X.execute(session=session, **(run_kwargs or dict()))
        return X


def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True, validate=True):
    """Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    StandardScaler : Performs scaling to unit variance using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSC matrix.

    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.StandardScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(StandardScaler(), LogisticRegression())`.

    Examples
    --------
    >>> from maxframe.learn.preprocessing import scale
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    >>> scale(X, axis=0).execute()  # scaling each column independently
    array([[-1.,  1.,  1.],
           [ 1., -1., -1.]])
    >>> scale(X, axis=1).execute()  # scaling each row independently
    array([[-1.37...,  0.39...,  0.98...],
           [-1.22...,  0.     ,  1.22...]])
    """
    if not isinstance(X, ENTITY_TYPE):
        X = mt.tensor(X)

    ndim = X.ndim
    if validate and ndim == 1:
        X = X.reshape((X.shape[0], 1))
    if axis == 1:
        X = X.T

    scaler = StandardScaler(
        with_mean=with_mean, with_std=with_std, copy=copy, validate=validate
    )
    transformed = scaler.fit_transform(X)

    if axis == 1:
        transformed = transformed.T
    if validate and ndim == 1:
        transformed = transformed.reshape(transformed.shape[0])
    return transformed
