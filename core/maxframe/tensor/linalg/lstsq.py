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

from ...core import ExecutableTuple
from ...udf import builtin_function
from .svd import svd


@builtin_function
def _cut_residuals(residuals, full_rank):
    return residuals if full_rank else np.array([], dtype=residuals.dtype)


def lstsq(a, b, rcond=None):
    """
    Return the least-squares solution to a linear matrix equation.

    Computes the vector `x` that approximately solves the equation
    ``a @ x = b``. The equation may be under-, well-, or over-determined
    (i.e., the number of linearly independent rows of `a` can be less than,
    equal to, or greater than its number of linearly independent columns).
    If `a` is square and of full rank, then `x` (but for round-off error)
    is the "exact" solution of the equation. Else, `x` minimizes the
    Euclidean 2-norm :math:`||b - ax||`. If there are multiple minimizing
    solutions, the one with the smallest 2-norm :math:`||x||` is returned.

    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`.
        The default uses the machine precision times ``max(M, N)``.  Passing
        ``-1`` will use machine precision.

    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(1,), (K,), (0,)} ndarray
        Sums of squared residuals: Squared Euclidean 2-norm for each column in
        ``b - a @ x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) ndarray
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.
    """
    # fixme when has_unknown_shape(a)

    rcond = rcond if rcond is not None and rcond < 1 else None
    if rcond is None:
        rcond = max(a.shape) * np.finfo(a.dtype).eps
    elif rcond == -1:
        rcond = np.finfo(np.float64).eps

    u, s, vt = svd(a)
    cutoff = (s > s.max() * rcond).astype(int)
    # v^T diag(1.0 / s) u_1^T b
    x = vt.T * (1.0 / s * cutoff) @ u.T[: a.shape[1], :] @ b
    residuals = b - a.dot(x)
    residuals = abs(residuals**2).sum(axis=0, keepdims=b.ndim == 1)

    rank = cutoff.sum()
    residuals = residuals.mf.apply_chunk(
        _cut_residuals, dtype=residuals.dtype, full_rank=rank == min(a.shape)
    )
    return ExecutableTuple((x, residuals, rank, s))
