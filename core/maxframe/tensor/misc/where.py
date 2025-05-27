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
from ...serialization.serializables import KeyField
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import broadcast_shape
from .broadcast_to import broadcast_to


class TensorWhere(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.WHERE

    condition = KeyField("condition", default=None)
    x = KeyField("x", default=None)
    y = KeyField("y", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorWhere", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.condition = op._inputs[0]
        op.x = op._inputs[1]
        op.y = op._inputs[2]

    def __call__(self, condition, x, y, shape=None):
        shape = shape or broadcast_shape(condition.shape, x.shape, y.shape)
        return self.new_tensor([condition, x, y], shape)


def where(condition, x=None, y=None):
    """
    Return elements, either from `x` or `y`, depending on `condition`.

    If only `condition` is given, return ``condition.nonzero()``.

    Parameters
    ----------
    condition : array_like, bool
        When True, yield `x`, otherwise yield `y`.
    x, y : array_like, optional
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : Tensor or tuple of Tensors
        If both `x` and `y` are specified, the output tensor contains
        elements of `x` where `condition` is True, and elements from
        `y` elsewhere.

        If only `condition` is given, return the tuple
        ``condition.nonzero()``, the indices where `condition` is True.

    See Also
    --------
    nonzero, choose

    Notes
    -----
    If `x` and `y` are given and input arrays are 1-D, `where` is
    equivalent to::

        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]]).execute()
    array([[1, 8],
           [3, 4]])

    >>> mt.where([[0, 1], [1, 0]]).execute()
    (array([0, 1]), array([1, 0]))

    >>> x = mt.arange(9.).reshape(3, 3)
    >>> mt.where( x > 5 ).execute()
    (array([2, 2, 2]), array([0, 1, 2]))
    >>> mt.where(x < 5, x, -1).execute()               # Note: broadcasting.
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -1.],
           [-1., -1., -1.]])

    Find the indices of elements of `x` that are in `goodvalues`.

    >>> goodvalues = [3, 4, 7]
    >>> ix = mt.isin(x, goodvalues)
    >>> ix.execute()
    array([[False, False, False],
           [ True,  True, False],
           [False,  True, False]])
    >>> mt.where(ix).execute()
    (array([1, 1, 2]), array([0, 1, 1]))
    """
    if (x is None) != (y is None):
        raise ValueError("either both or neither of x and y should be given")

    if x is None and y is None:
        return astensor(condition).nonzero()

    x, y = astensor(x), astensor(y)
    dtype = np.result_type(x.dtype, y.dtype)
    shape = broadcast_shape(x.shape, y.shape)

    if np.isscalar(condition):
        return broadcast_to(x if condition else y, shape).astype(dtype)
    else:
        condition = astensor(condition)
        op = TensorWhere(dtype=dtype, sparse=condition.issparse())
        return op(condition, x, y, shape=shape)
