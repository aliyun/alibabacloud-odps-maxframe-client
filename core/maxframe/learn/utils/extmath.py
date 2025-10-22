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

from ... import tensor as mt
from ...core import ENTITY_TYPE
from ...dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ...tensor.datasource import TensorZeros


# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A accumulator function such as np.mean or np.sum
    x : numpy array
        A tensor to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function

    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if (
        hasattr(x, "dtype")
        and np.issubdtype(x.dtype, np.floating)
        and x.dtype.itemsize < 8
    ):
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def logsumexp_real(a, axis=None, keepdims=False):
    """Simplified logsumexp for real arrays without biases"""
    from ... import tensor as mt

    x = mt.tensor(a)
    x_max = mt.amax(a, axis=axis, keepdims=True)
    exp_x_shifted = mt.exp(x - x_max)
    ret = mt.log(mt.sum(exp_x_shifted, axis=axis, keepdims=True)) + x_max
    if keepdims:
        return ret
    return mt.squeeze(ret, axis=1)


def _incremental_mean_and_var(
    X, last_mean, last_variance, last_sample_count, sample_weight=None
):
    """Calculate mean update and a Youngs and Cramer variance update.

    If sample_weight is given, the weighted mean and variance is computed.

    Update a given mean and (possibly) variance according to new data given
    in X. last_mean is always required to compute the new mean.
    If last_variance is None, no variance is computed and None return for
    updated_variance.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to use for variance update.

    last_mean : array-like of shape (n_features,)

    last_variance : array-like of shape (n_features,)

    last_sample_count : array-like of shape (n_features,)
        The number of samples encountered until now if sample_weight is None.
        If sample_weight is not None, this is the sum of sample_weight
        encountered.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights. If None, compute the unweighted mean/variance.

    Returns
    -------
    updated_mean : ndarray of shape (n_features,)

    updated_variance : ndarray of shape (n_features,)
        None if last_variance was None.

    updated_sample_count : ndarray of shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    has_last_sample = isinstance(last_sample_count, ENTITY_TYPE) and not isinstance(
        last_sample_count.op, TensorZeros
    )
    is_df_type = isinstance(X, (DATAFRAME_TYPE, SERIES_TYPE))

    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count if has_last_sample else 0
    X_nan_mask = mt.isnan(X)
    # if mt.any(X_nan_mask):
    #     sum_op = mt.nansum
    # else:
    #     sum_op = mt.sum

    def df_sum(val, **kw):
        if "dtype" in kw:
            val = val.astype(kw.pop("dtype"))
        return val.sum(**kw)

    if is_df_type:
        sum_op = df_sum
    else:
        sum_op = mt.nansum

    if sample_weight is not None:
        # equivalent to np.nansum(X * sample_weight, axis=0)
        # safer because np.float64(X*W) != np.float64(X)*np.float64(W)
        new_sum = _safe_accumulator_op(
            mt.matmul, sample_weight, mt.where(X_nan_mask, 0, X)
        )
        new_sample_count = _safe_accumulator_op(
            mt.sum, sample_weight[:, None] * (~X_nan_mask), axis=0
        )
    else:
        new_sum = _safe_accumulator_op(sum_op, X, axis=0)
        if is_df_type:
            new_sample_count = X.count()
        else:
            n_samples = X.shape[0]
            new_sample_count = n_samples - mt.sum(X_nan_mask, axis=0)

    if not has_last_sample:
        updated_sample_count = new_sample_count
    else:
        updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T
        if sample_weight is not None:
            # equivalent to np.nansum((X-T)**2 * sample_weight, axis=0)
            # safer because np.float64(X*W) != np.float64(X)*np.float64(W)
            correction = _safe_accumulator_op(
                mt.matmul, sample_weight, mt.where(X_nan_mask, 0, temp)
            )
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(
                mt.matmul, sample_weight, mt.where(X_nan_mask, 0, temp)
            )
        else:
            correction = _safe_accumulator_op(sum_op, temp, axis=0)
            temp **= 2
            new_unnormalized_variance = _safe_accumulator_op(sum_op, temp, axis=0)

        # correction term of the corrected 2 pass algorithm.
        # See "Algorithms for computing the sample variance: analysis
        # and recommendations", by Chan, Golub, and LeVeque.
        new_unnormalized_variance -= correction**2 / new_sample_count

        last_unnormalized_variance = (
            last_variance * last_sample_count if has_last_sample else 0
        )

        with mt.errstate(divide="ignore", invalid="ignore"):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum) ** 2
            )

        if not has_last_sample:
            updated_unnormalized_variance = new_unnormalized_variance
        else:
            zeros = last_sample_count == 0
            updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Performs no input validation.

    Parameters
    ----------
    X : array_like
        The input tensor
    squared : bool, optional (default = False)
        If True, return squared norms.

    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """

    norms = (X**2).sum(axis=1)
    if not squared:
        norms = mt.sqrt(norms)
    return norms
