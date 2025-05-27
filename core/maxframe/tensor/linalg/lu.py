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
from ...core import ExecutableTuple
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin


class TensorLU(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.LU

    @property
    def output_limit(self):
        return 3

    def __call__(self, a):
        import scipy.linalg

        a = astensor(a)
        if a.ndim != 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional array given. Tensor must be two-dimensional"
            )

        if a.shape[0] > a.shape[1]:
            p_shape = (a.shape[0],) * 2
            l_shape = a.shape
            u_shape = (a.shape[1],) * 2
        elif a.shape[0] < a.shape[1]:
            p_shape = (a.shape[0],) * 2
            l_shape = (a.shape[0],) * 2
            u_shape = a.shape
        else:
            p_shape, l_shape, u_shape = (a.shape,) * 3

        tiny_p, tiny_l, tiny_u = scipy.linalg.lu(
            np.array([[1, 2], [2, 5]], dtype=a.dtype)
        )

        order = a.order
        p, l, u = self.new_tensors(
            [a],
            kws=[
                {"side": "p", "dtype": tiny_p.dtype, "shape": p_shape, "order": order},
                {"side": "l", "dtype": tiny_l.dtype, "shape": l_shape, "order": order},
                {"side": "u", "dtype": tiny_u.dtype, "shape": u_shape, "order": order},
            ],
        )
        return ExecutableTuple([p, l, u])


def lu(a):
    """
    LU decomposition

    The decomposition is::
        A = P L U
    where P is a permutation matrix, L lower triangular with unit diagonal elements,
    and U upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Array to decompose

    Returns
    -------
    p : (M, M) ndarray
        Permutation matrix
    l : (M, K) ndarray
        Lower triangular or trapezoidal matrix with unit diagonal.
        K = min(M, N)
    u : (K, N) ndarray
        Upper triangular or trapezoidal matrix

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> A = mt.array([[1,2],[2,3]])
    >>> A.execute()
    array([[ 1,  2],
           [ 2,  3]])
    >>> P, L, U = mt.linalg.lu(A)
    >>> P.execute()
    array([[ 0,  1],
           [ 1,  0]])
    >>> L.execute()
    array([[ 1,  0],
           [ 0.5,  1]])
    >>> U.execute()
    array([[ 2,  3],
           [ 0,  0.5]])
    >>> mt.dot(P.dot(L), U).execute() # verify that PL * U = A
    array([[ 1,  2],
           [ 2,  3]])

    """
    op = TensorLU(sparse=a.issparse())
    return op(a)
