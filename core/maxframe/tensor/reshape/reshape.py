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

import logging
from typing import List

import numpy as np

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import FieldTypes, KeyField, StringField, TupleField
from ..datasource import tensor as astensor
from ..operators import TensorMapReduceOperator, TensorOperatorMixin
from ..utils import get_order

logger = logging.getLogger(__name__)


class TensorReshape(TensorMapReduceOperator, TensorOperatorMixin):
    _op_type_ = opcodes.RESHAPE

    _input = KeyField("input")
    newshape = TupleField("newshape", FieldTypes.int64, default=None)
    order = StringField("order", default=None)

    axis_offsets = TupleField("axis_offsets", FieldTypes.uint64, default=None)
    oldshape = TupleField("oldshape", FieldTypes.uint64, default=None)
    new_chunk_size = TupleField("new_chunk_size", FieldTypes.uint64, default=None)

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "TensorReshape", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]

    def on_output_modify(self, new_output):
        return reshape(new_output, self._input.shape)

    def on_input_modify(self, new_input):
        op = self.copy().reset_key()
        return op(new_input)

    def __call__(self, a, order, out_shape):
        return self.new_tensor([a], out_shape, order=order)


def calc_shape(size, newshape):
    if isinstance(newshape, int):
        newshape = (newshape,)
    else:
        newshape = tuple(int(s) for s in newshape)

    known_shape = [s for s in newshape if s >= 0]
    missing_dim = len(newshape) - len(known_shape)
    if missing_dim > 1:
        raise ValueError("can only specify one unknown dimension")
    if missing_dim == 1:
        known_size = np.prod(known_shape)
        newshape = tuple(
            int(size / known_size) if s < 0 and known_size > 0 else s for s in newshape
        )

    return newshape


def reshape(a, newshape, order="C"):
    """
    Gives a new shape to a tensor without changing its data.

    Parameters
    ----------
    a : array_like
        Tensor to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D tensor of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the tensor and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    Returns
    -------
    reshaped_array : Tensor
        This will be a new view object if possible; otherwise, it will
        be a copy.

    See Also
    --------
    Tensor.reshape : Equivalent method.

    Notes
    -----
    It is not always possible to change the shape of a tensor without
    copying the data. If you want an error to be raised when the data is copied,
    you should assign the new shape to the shape attribute of the array::

    >>> import maxframe.tensor as mt

    >>> a = mt.arange(6).reshape((3, 2))
    >>> a.execute()
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the tensor (using the given
    index order), then inserting the elements from the raveled tensor into the
    new tensor using the same kind of index ordering as was used for the
    raveling.

    >>> mt.reshape(a, (2, 3)).execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.reshape(mt.ravel(a), (2, 3)).execute()
    array([[0, 1, 2],
           [3, 4, 5]])

    Examples
    --------
    >>> a = mt.array([[1,2,3], [4,5,6]])
    >>> mt.reshape(a, 6).execute()
    array([1, 2, 3, 4, 5, 6])

    >>> mt.reshape(a, (3,-1)).execute()       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    a = astensor(a)

    if np.isnan(sum(a.shape)):
        # some shape is nan
        new_shape = [newshape] if isinstance(newshape, int) else list(newshape)
        # if -1 exists in newshape, just treat it as unknown shape
        new_shape = [s if s != -1 else np.nan for s in new_shape]
        out_shape = tuple(new_shape)
    else:
        out_shape = newshape = calc_shape(a.size, newshape)
        if a.size != np.prod(newshape):
            raise ValueError(
                f"cannot reshape array of size {a.size} into shape {newshape}"
            )

    tensor_order = get_order(order, a.order, available_options="CFA")

    if a.shape == newshape and (
        a.ndim <= 1 or (a.ndim > 1 and tensor_order == a.order)
    ):
        # does not need to reshape
        return a
    return _reshape(
        a, newshape, order=order, tensor_order=tensor_order, out_shape=out_shape
    )


def _reshape(a, newshape, order="C", tensor_order=None, out_shape=None):
    if tensor_order is None:
        tensor_order = get_order(order, a.order, available_options="CFA")
    op = TensorReshape(
        newshape=newshape,
        order=order,
        dtype=a.dtype,
        create_view=tensor_order == a.order,
    )
    if out_shape is None:
        out_shape = newshape
    return op(a, tensor_order, out_shape)
