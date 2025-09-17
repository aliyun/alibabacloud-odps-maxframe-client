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
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField, BoolField, Int32Field
from ..core import Tensor
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin


class TensorFillDiagonal(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.FILL_DIAGONAL

    val = AnyField("val", default=None)
    wrap = BoolField("wrap", default=None)
    # used for chunk
    k = Int32Field("k", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorFillDiagonal", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if len(op._inputs) == 2:
            op.val = op._inputs[1]

    def __call__(self, a, val=None):
        inputs = [a]
        if val is not None:
            inputs.append(val)
        return self.new_tensor(inputs, shape=a.shape, order=a.order)


def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given tensor of any dimensionality.

    For a tensor `a` with ``a.ndim >= 2``, the diagonal is the list of
    locations with indices ``a[i, ..., i]`` all identical. This function
    modifies the input tensor in-place, it does not return a value.

    Parameters
    ----------
    a : Tensor, at least 2-D.
      Tensor whose diagonal is to be filled, it gets modified in-place.

    val : scalar
      Value to be written on the diagonal, its type must be compatible with
      that of the tensor a.

    wrap : bool
      For tall matrices in NumPy version up to 1.6.2, the
      diagonal "wrapped" after N columns. You can have this behavior
      with this option. This affects only tall matrices.

    See also
    --------
    diag_indices, diag_indices_from

    Notes
    -----

    This functionality can be obtained via `diag_indices`, but internally
    this version uses a much faster implementation that never constructs the
    indices and uses simple slicing.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = mt.zeros((3, 3), int)
    >>> mt.fill_diagonal(a, 5)
    >>> a.execute()
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-D tensor:

    >>> a = mt.zeros((3, 3, 3, 3), int)
    >>> mt.fill_diagonal(a, 4)

    We only show a few blocks for clarity:

    >>> a[0, 0].execute()
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1, 1].execute()
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2, 2].execute()
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    The wrap option affects only tall matrices:

    >>> # tall matrices no wrap
    >>> a = mt.zeros((5, 3), int)
    >>> mt.fill_diagonal(a, 4)
    >>> a.execute()
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [0, 0, 0]])

    >>> # tall matrices wrap
    >>> a = mt.zeros((5, 3), int)
    >>> mt.fill_diagonal(a, 4, wrap=True)
    >>> a.execute()
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [4, 0, 0]])

    >>> # wide matrices
    >>> a = mt.zeros((3, 5), int)
    >>> mt.fill_diagonal(a, 4, wrap=True)
    >>> a.execute()
    array([[4, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 4, 0, 0]])

    The anti-diagonal can be filled by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = mt.zeros((3, 3), int)
    >>> mt.fill_diagonal(mt.fliplr(a), [1,2,3])  # Horizontal flip
    >>> a.execute()
    array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    >>> mt.fill_diagonal(mt.flipud(a), [1,2,3])  # Vertical flip
    >>> a.execute()
    array([[0, 0, 3],
           [0, 2, 0],
           [1, 0, 0]])

    Note that the order in which the diagonal is filled varies depending
    on the flip function.
    """

    if not isinstance(a, Tensor):
        raise TypeError(f"`a` should be a tensor, got {type(a)}")
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if a.ndim > 2 and len(set(a.shape)) != 1:
        raise ValueError("All dimensions of input must be of equal length")

    # process val
    if isinstance(val, ENTITY_TYPE):
        val = astensor(val)
        if val.ndim > 1:
            val = val.ravel()
        val_input = val
    else:
        val = np.asarray(val)
        if val.ndim > 1:
            val = val.ravel()
        val_input = None

    op = TensorFillDiagonal(val=val, wrap=wrap, dtype=a.dtype)
    t = op(a, val=val_input)
    a.data = t.data
