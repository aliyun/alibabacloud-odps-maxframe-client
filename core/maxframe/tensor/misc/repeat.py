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

from numbers import Integral
from typing import List

import numpy as np

from ... import opcodes
from ...serialization.serializables import AnyField, Int32Field
from ...typing_ import EntityType
from ..core import Tensor, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import broadcast_shape
from .ravel import ravel


class TensorRepeat(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.REPEAT

    repeats = AnyField("repeats", default=None)
    axis = Int32Field("axis", default=None)

    def __init__(self, sparse=False, **kw):
        super().__init__(sparse=sparse, **kw)

    @classmethod
    def _set_inputs(cls, op: "TensorRepeat", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if len(inputs) > 1:
            op.repeats = inputs[1]

    def __call__(self, a, repeats):
        axis = self.axis
        a = astensor(a)
        if axis is None:
            a = ravel(a)

        ax = axis or 0

        if not isinstance(repeats, Integral):
            if not isinstance(repeats, Tensor):
                repeats = np.asarray(repeats)
                if repeats.size == 1:
                    repeats = int(repeats[0])
                    size = repeats * a.shape[axis or 0]
                elif a.shape[ax] == 1:
                    size = repeats = int(repeats.sum())
                else:
                    size = int(repeats.sum())
            else:
                size = np.nan
            if not isinstance(repeats, Integral):
                if repeats.ndim != 1:
                    raise ValueError("repeats should be 1-d tensor")
                broadcast_shape(repeats.shape, a.shape[ax : ax + 1])
        else:
            size = a.shape[axis or 0] * repeats

        shape = a.shape[:ax] + (size,) + a.shape[ax + 1 :]
        self.dtype = a.dtype
        self.sparse = a.issparse()

        inputs = [a]
        if isinstance(repeats, Tensor):
            inputs.append(repeats)
        else:
            self.repeats = repeats

        return self.new_tensor(inputs, shape, order=TensorOrder.C_ORDER)


def repeat(a, repeats, axis=None):
    """
    Repeat elements of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.
    repeats : int or tensor of ints
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input tensor, and return a flat output tensor.

    Returns
    -------
    repeated_tensor : Tensor
        Output array which has the same shape as `a`, except along
        the given axis.

    See Also
    --------
    tile : Tile a tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.repeat(3, 4).execute()
    array([3, 3, 3, 3])
    >>> x = mt.array([[1,2],[3,4]])
    >>> mt.repeat(x, 2).execute()
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> mt.repeat(x, 3, axis=1).execute()
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> mt.repeat(x, [1, 2], axis=0).execute()
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    op = TensorRepeat(axis=axis)
    return op(a, repeats)
