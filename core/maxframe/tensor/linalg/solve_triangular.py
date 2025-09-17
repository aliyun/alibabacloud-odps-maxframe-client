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

from typing import List

import numpy as np
from numpy.linalg import LinAlgError

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import BoolField, KeyField
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin


class TensorSolveTriangular(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.SOLVE_TRIANGULAR

    a = KeyField("a")
    b = KeyField("b")
    lower = BoolField("lower", default=None)
    strict = BoolField("strict", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorSolveTriangular", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.a, op.b = op._inputs

    def __call__(self, a, b):
        shape = (a.shape[1],) if len(b.shape) == 1 else (a.shape[1], b.shape[1])
        return self.new_tensor([a, b], shape, order=TensorOrder.F_ORDER)


def solve_triangular(a, b, lower=False, sparse=None):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in `a x = b`
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    sparse: bool, optional
        Return sparse value or not.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system `a x = b`.  Shape of return matches `b`.

    Examples
    --------
    Solve the lower triangular system a x = b, where::
             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> import maxframe.tensor as mt
    >>> a = mt.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = mt.array([4, 2, 4, 2])
    >>> x = mt.linalg.solve_triangular(a, b, lower=True)
    >>> x.execute()
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])

    >>> a.dot(x).execute()  # Check the result
    array([ 4.,  2.,  4.,  2.])
    """
    import scipy.linalg

    a = astensor(a)
    b = astensor(b)

    if a.ndim != 2:
        raise LinAlgError("a must be 2 dimensional")
    if b.ndim <= 2:
        if a.shape[1] != b.shape[0]:
            raise LinAlgError("a.shape[1] and b.shape[0] must be equal")
    else:
        raise LinAlgError("b must be 1 or 2 dimensional")

    tiny_x = scipy.linalg.solve_triangular(
        np.array([[2, 0], [2, 1]], dtype=a.dtype), np.array([[2], [3]], dtype=b.dtype)
    )
    sparse = sparse if sparse is not None else a.issparse()
    op = TensorSolveTriangular(lower=lower, dtype=tiny_x.dtype, sparse=sparse)
    return op(a, b)
