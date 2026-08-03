# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe import tensor as mt
from maxframe.core import ENTITY_TYPE
from maxframe.dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from maxframe.tensor.datasource import TensorZeros
from maxframe.tensor.linalg.lu import lu
from maxframe.tensor.linalg.qr import qr
from maxframe.tensor.linalg.svd import svd
from maxframe.tensor.utils import check_random_state


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
    from maxframe import tensor as mt

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


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : Tensor
        u and v are the output of `linalg.svd` or
        `randomized_svd`, with matching inner dimensions
        so one can compute `mt.dot(u * s, v)`.

    v : Tensor
        u and v are the output of `linalg.svd` or
        `randomized_svd`, with matching inner dimensions
        so one can compute `mt.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = mt.argmax(mt.abs(u), axis=0)
        signs = mt.sign(u[max_abs_cols, np.arange(u.shape[1])])
        u *= signs
        v *= signs[:, mt.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = mt.argmax(mt.abs(v), axis=1)
        signs = mt.sign(v[np.arange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, mt.newaxis]
    return u, v


def randomized_range_finder(
    A, size, n_iter, power_iteration_normalizer="auto", random_state=None
):
    r"""Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D tensor
        The input data tensor

    size : integer
        Size of the return tensor

    n_iter : integer
        Number of power iterations used to stabilize the result

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Returns
    -------
    Q : 2D tensor
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----
    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == "f":
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == "auto":
        if n_iter <= 2:
            power_iteration_normalizer = "none"
        else:
            power_iteration_normalizer = "LU"

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        if power_iteration_normalizer == "none":
            Q = A.dot(Q)
            Q = A.T.dot(Q)
        elif power_iteration_normalizer == "LU":
            # TODO: directly get Q when lu supports `permute_l`
            p, l, _ = lu(A.dot(Q))
            Q = p.dot(l)
            p, l, _ = lu(A.T.dot(Q))
            Q = p.dot(l)
        elif power_iteration_normalizer == "QR":
            Q, _ = qr(A.dot(Q))
            Q, _ = qr(A.T.dot(Q))

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = qr(A.dot(Q))
    return Q


def randomized_svd(
    M,
    n_components,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=0,
):
    r"""Computes a truncated randomized SVD.

    Parameters
    ----------
    M : Tensor
        Tensor to decompose.
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    M = mt.asarray(M)
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Check if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA.
        # https://github.com/scikit-learn/scikit-learn/pull/5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(
        M, n_random, n_iter, power_iteration_normalizer, random_state
    )
    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.T.dot(M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = svd(B)

    U = Q.dot(Uhat)

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]
