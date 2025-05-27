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

from ... import opcodes
from ...serialization.serializables import Float64Field, NDArrayField, StringField
from ..utils import gen_random_seeds
from .core import TensorDistribution, TensorRandomOperatorMixin


class TensorMultivariateNormal(TensorDistribution, TensorRandomOperatorMixin):
    _op_type_ = opcodes.RAND_MULTIVARIATE_NORMAL

    _fields_ = "mean", "cov", "size", "check_valid", "tol"
    mean = NDArrayField("mean")
    cov = NDArrayField("cov")
    check_valid = StringField("check_valid")
    tol = Float64Field("tol")
    _func_name = "multivariate_normal"

    def __call__(self, chunk_size=None):
        N = self.mean.size
        if self.size is None:
            shape = (N,)
        else:
            try:
                shape = tuple(self.size) + (N,)
            except TypeError:
                shape = (self.size, N)

        return self.new_tensor(None, shape, raw_chunk_size=chunk_size)


def multivariate_normal(
    random_state,
    mean,
    cov,
    size=None,
    check_valid=None,
    tol=None,
    chunk_size=None,
    gpu=None,
    dtype=None,
):
    """
    Draw random samples from a multivariate normal distribution.

    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions.  Such a distribution is specified by its mean and
    covariance matrix.  These parameters are analogous to the mean
    (average or "center") and variance (standard deviation, or "width,"
    squared) of the one-dimensional normal distribution.

    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    size : int or tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
        each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
        If no shape is specified, a single (`N`-D) sample is returned.
    check_valid : { 'warn', 'raise', 'ignore' }, optional
        Behavior when the covariance matrix is not positive semidefinite.
    tol : float, optional
        Tolerance when checking the singular values in covariance matrix.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.

        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Notes
    -----
    The mean is a coordinate in N-dimensional space, which represents the
    location where samples are most likely to be generated.  This is
    analogous to the peak of the bell curve for the one-dimensional or
    univariate normal distribution.

    Covariance indicates the level to which two variables vary together.
    From the multivariate normal distribution, we draw N-dimensional
    samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
    element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
    The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
    "spread").

    Instead of specifying the full covariance matrix, popular
    approximations include:

      - Spherical covariance (`cov` is a multiple of the identity matrix)
      - Diagonal covariance (`cov` has non-negative elements, and only on
        the diagonal)

    This geometrical property can be seen in two dimensions by plotting
    generated data-points:

    >>> mean = [0, 0]
    >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

    Diagonal covariance means that points are oriented along x or y-axis:

    >>> import matplotlib.pyplot as plt
    >>> import maxframe.tensor as mt
    >>> x, y = mt.random.multivariate_normal(mean, cov, 5000).T
    >>> plt.plot(x.execute(), y.execute(), 'x')
    >>> plt.axis('equal')
    >>> plt.show()

    Note that the covariance matrix must be positive semidefinite (a.k.a.
    nonnegative-definite). Otherwise, the behavior of this method is
    undefined and backwards compatibility is not guaranteed.

    References
    ----------
    .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
           Processes," 3rd ed., New York: McGraw-Hill, 1991.
    .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
           Classification," 2nd ed., New York: Wiley, 2001.

    Examples
    --------
    >>> mean = (1, 2)
    >>> cov = [[1, 0], [0, 1]]
    >>> x = mt.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)

    The following is probably true, given that 0.6 is roughly twice the
    standard deviation:

    >>> list(((x[0,0,:] - mean) < 0.6).execute())
    [True, True]
    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    if mean.ndim != 1:
        raise ValueError("mean must be 1 dimensional")
    if cov.ndim != 2:
        raise ValueError("cov must be 1 dimensional")
    if len(set(mean.shape + cov.shape)) != 1:
        raise ValueError("mean and cov must have same length")

    if dtype is None:
        small_kw = {}
        if check_valid:
            small_kw["check_valid"] = check_valid
        if tol:
            small_kw["tol"] = tol
        dtype = np.random.multivariate_normal(mean, cov, size=(0,), **small_kw).dtype

    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorMultivariateNormal(
        mean=mean,
        cov=cov,
        size=size,
        check_valid=check_valid,
        tol=tol,
        seed=seed,
        gpu=gpu,
        dtype=dtype,
    )
    return op(chunk_size=chunk_size)
