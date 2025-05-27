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

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import KeyField, StringField
from ..core import Tensor, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import broadcast_shape, check_order, check_out_param


class TensorMatmul(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.MATMUL

    a = KeyField("a")
    b = KeyField("b")
    casting = StringField("casting", default="same_kind")
    order = StringField("order", default="K")

    def __init__(self, **kw):
        super().__init__(**kw)
        check_order(self.order)

    @classmethod
    def _set_inputs(cls, op: "TensorMatmul", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.a = op._inputs[0]
        op.b = op._inputs[1]

    def _calc_order(self, a, b, out):
        if out is not None:
            return out.order

        if self.order in "A":
            if a.order == TensorOrder.C_ORDER or b.order == TensorOrder.C_ORDER:
                return TensorOrder.C_ORDER
            else:
                return TensorOrder.F_ORDER
        elif self.order in "CK":
            return TensorOrder.C_ORDER
        else:
            return TensorOrder.F_ORDER

    def __call__(self, a, b, out=None):
        from ..misc import broadcast_to

        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        if out is not None and not isinstance(out, Tensor):
            raise TypeError(f"out must be a Tensor, got {type(out)} instead")

        a_is_1d = False
        if a.ndim == 1:
            a_is_1d = True
            a = a[np.newaxis, :]

        b_is_1d = False
        if b.ndim == 1:
            b_is_1d = True
            b = b[:, np.newaxis]

        if a.ndim < b.ndim:
            a = a[(b.ndim - a.ndim) * (np.newaxis,)]
        elif a.ndim > b.ndim:
            b = b[(a.ndim - b.ndim) * (np.newaxis,)]

        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                f"shape {a.shape} and {b.shape} not aligned: "
                f"{a.shape[-1]} (dim {a.ndim - 1}) != {b.shape[-2]} (dim {b.ndim - 2})"
            )

        shape = broadcast_shape(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
        order = self._calc_order(a, b, out)
        t = self.new_tensor([a, b], shape, order=order)

        if a_is_1d:
            t = t[..., 0, :]
        if b_is_1d:
            t = t[..., 0]

        if out is not None:
            check_out_param(out, t, self.casting)
            t = broadcast_to(t, out.shape)
            out.data = t.data
            return out

        return t


def matmul(a, b, sparse=None, out=None, **kw):
    """
    Matrix product of two tensors.

    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional
      matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of
      matrices residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by
      prepending a 1 to its dimensions. After matrix multiplication
      the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by
      appending a 1 to its dimensions. After matrix multiplication
      the appended 1 is removed.

    Multiplication by a scalar is not allowed, use ``*`` instead. Note that
    multiplying a stack of matrices with a vector will result in a stack of
    vectors, but matmul will not recognize it as such.

    ``matmul`` differs from ``dot`` in two important ways.

    - Multiplication by scalars is not allowed.
    - Stacks of matrices are broadcast together as if the matrices
      were elements.

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : Tensor, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type,
        and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : Tensor
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        1-D arrays then a scalar is returned; otherwise an array is
        returned.  If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

        If scalar value is passed.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    dot : alternative matrix product with different broadcasting rules.

    Notes
    -----
    The matmul function implements the semantics of the `@` operator introduced
    in Python 3.5 following PEP465.

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import maxframe.tensor as mt

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> mt.matmul(a, b).execute()
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = [[1, 0], [0, 1]]
    >>> b = [1, 2]
    >>> mt.matmul(a, b).execute()
    array([1, 2])
    >>> mt.matmul(b, a).execute()
    array([1, 2])


    Broadcasting is conventional for stacks of arrays

    >>> a = mt.arange(2*2*4).reshape((2,2,4))
    >>> b = mt.arange(2*2*4).reshape((2,4,2))
    >>> mt.matmul(a,b).shape
    (2, 2, 2)
    >>> mt.matmul(a,b)[0,1,1].execute()
    98
    >>> mt.sum(a[0,1,:] * b[0,:,1]).execute()
    98

    Vector, vector returns the scalar inner product, but neither argument
    is complex-conjugated:

    >>> mt.matmul([2j, 3j], [2j, 3j]).execute()
    (-13+0j)

    Scalar multiplication raises an error.

    >>> mt.matmul([1,2], 3)
    Traceback (most recent call last):
    ...
    ValueError: Scalar operands are not allowed, use '*' instead
    """
    a = astensor(a)
    b = astensor(b)

    sparse = sparse if sparse is not None else a.issparse() and b.issparse()
    op = TensorMatmul(dtype=np.promote_types(a.dtype, b.dtype), sparse=sparse, **kw)
    return op(a, b, out=out)
