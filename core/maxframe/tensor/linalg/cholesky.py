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
from numpy.linalg import LinAlgError

from ... import opcodes
from ...serialization.serializables import BoolField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorCholesky(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.CHOLESKY

    lower = BoolField("lower")

    def __call__(self, a):
        return self.new_tensor([a], a.shape, order=TensorOrder.C_ORDER)


def cholesky(a, lower=False):
    """
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.H`, of the square matrix `a`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `a` is real-valued).  `a` must be
    Hermitian (symmetric if real-valued) and positive-definite.  Only `L` is
    actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.
    lower : bool
        Whether to compute the upper or lower triangular Cholesky
        factorization.  Default is upper-triangular.

    Returns
    -------
    L : (..., M, M) array_like
        Upper or lower-triangular Cholesky factor of `a`.

    Raises
    ------
    LinAlgError
       If the decomposition fails, for example, if `a` is not
       positive-definite.

    Notes
    -----

    Broadcasting rules apply, see the `mt.linalg` documentation for
    details.

    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \\mathbf{x} = \\mathbf{b}

    (when `A` is both Hermitian/symmetric and positive-definite).

    First, we solve for :math:`\\mathbf{y}` in

    .. math:: L \\mathbf{y} = \\mathbf{b},

    and then for :math:`\\mathbf{x}` in

    .. math:: L.H \\mathbf{x} = \\mathbf{y}.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> A = mt.array([[1,-2j],[2j,5]])
    >>> A.execute()
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = mt.linalg.cholesky(A, lower=True)
    >>> L.execute()
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> mt.dot(L, L.T.conj()).execute() # verify that L * L.H = A
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?
    >>> mt.linalg.cholesky(A, lower=True).execute()
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])

    """
    a = astensor(a)

    if a.ndim != 2:  # pragma: no cover
        raise LinAlgError(
            f"{a.ndim}-dimensional array given. Tensor must be two-dimensional"
        )
    if a.shape[0] != a.shape[1]:  # pragma: no cover
        raise LinAlgError("Input must be square")

    cho = np.linalg.cholesky(np.array([[1, 2], [2, 5]], dtype=a.dtype))

    op = TensorCholesky(lower=lower, dtype=cho.dtype)
    return op(a)
