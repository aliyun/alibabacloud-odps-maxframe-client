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
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorInv(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.INV

    def __call__(self, a):
        a = astensor(a)
        return self.new_tensor([a], a.shape, order=TensorOrder.C_ORDER)


def inv(a, sparse=None):
    """
    Compute the (multiplicative) inverse of a matrix.
    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.
    sparse: bool, optional
        Return sparse value or not.

    Returns
    -------
    ainv : (..., M, M) ndarray or matrix
        (Multiplicative) inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = mt.linalg.inv(a)
    >>> mt.allclose(mt.dot(a, ainv), mt.eye(2)).execute()
    True

    >>> mt.allclose(mt.dot(ainv, a), mt.eye(2)).execute()
    True

    >>> ainv.execute()
    array([[ -2. ,  1. ],
           [ 1.5, -0.5]])
    """

    # TODO: using some parallel algorithm for matrix inversion.
    a = astensor(a)
    if a.ndim != 2:
        raise LinAlgError(
            f"{a.ndim}-dimensional array given. Tensor must be two-dimensional"
        )
    if a.shape[0] != a.shape[1]:
        raise LinAlgError("Input must be square")

    tiny_inv = np.linalg.inv(np.array([[1, 2], [2, 5]], dtype=a.dtype))
    sparse = sparse if sparse is not None else a.issparse()
    op = TensorInv(dtype=tiny_inv.dtype, sparse=sparse)
    return op(a)
