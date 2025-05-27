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

from ... import opcodes
from ...serialization.serializables import Int32Field
from ..core import TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import reverse_order, validate_axis


class TensorSwapAxes(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.SWAPAXES

    axis1 = Int32Field("axis1")
    axis2 = Int32Field("axis2")

    def __init__(self, **kw):
        super().__init__(create_view=True, **kw)

    def __call__(self, a):
        axis1, axis2 = self.axis1, self.axis2
        if (axis1 == 0 and axis2 == a.ndim - 1) or (axis1 == a.ndim - 1 and axis2 == 0):
            tensor_order = reverse_order(a.order)
        else:
            tensor_order = TensorOrder.C_ORDER
        shape = self._swap(a.shape, self.axis1, self.axis2)
        return self.new_tensor([a], shape, order=tensor_order)

    @staticmethod
    def _swap(it, axis1, axis2):
        new_it = list(it)
        new_it[axis1], new_it[axis2] = it[axis2], it[axis1]

        return tuple(new_it)

    def on_output_modify(self, new_output):
        op = TensorSwapAxes(
            axis1=self.axis2,
            axis2=self.axis1,
            dtype=new_output.dtype,
            sparse=new_output.issparse(),
        )
        return op(new_output)

    def on_input_modify(self, new_input):
        op = self.copy().reset_key()
        return op(new_input)


def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : Tensor
        If `a` is a Tensor, then a view of `a` is
        returned; otherwise a new tensor is created.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.array([[1,2,3]])
    >>> mt.swapaxes(x,0,1).execute()
    array([[1],
           [2],
           [3]])

    >>> x = mt.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x.execute()
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> mt.swapaxes(x,0,2).execute()
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])

    """
    a = astensor(a)
    axis1 = validate_axis(a.ndim, axis1)
    axis2 = validate_axis(a.ndim, axis2)

    if axis1 == axis2:
        return a

    op = TensorSwapAxes(axis1=axis1, axis2=axis2, dtype=a.dtype, sparse=a.issparse())
    return op(a)
