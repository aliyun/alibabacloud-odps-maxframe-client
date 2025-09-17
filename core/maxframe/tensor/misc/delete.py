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

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import AnyField, Int32Field, Int64Field
from ..datasource import tensor as astensor
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import calc_object_length, filter_inputs, validate_axis


class TensorDelete(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.DELETE

    index_obj = AnyField("index_obj", default=None)
    axis = Int32Field("axis", default=None)

    # for chunk
    offset_on_axis = Int64Field("offset_on_axis", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorDelete", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if len(inputs) > 1:
            op.index_obj = inputs[1]

    def __call__(self, arr, obj, shape):
        return self.new_tensor(filter_inputs([arr, obj]), shape=shape, order=arr.order)


def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int or array of ints
        Indicate indices of sub-arrays to remove along the specified axis.
    axis : int, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : maxframe.tensor.Tensor
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> arr = mt.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr.execute()
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> mt.delete(arr, 1, 0).execute()
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])
    >>> mt.delete(arr, np.s_[::2], 1).execute()
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> mt.delete(arr, [1,3,5], None).execute()
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])
    """
    arr = astensor(arr)
    if getattr(obj, "ndim", 0) > 1:  # pragma: no cover
        raise ValueError(
            "index array argument obj to insert must be one dimensional or scalar"
        )

    if axis is None:
        # if axis is None, array will be flatten
        arr_size = arr.size
        idx_length = calc_object_length(obj, size=arr_size)
        shape = (arr_size - idx_length,)
    else:
        validate_axis(arr.ndim, axis)
        idx_length = calc_object_length(obj, size=arr.shape[axis])
        shape = tuple(
            s - idx_length if i == axis else s for i, s in enumerate(arr.shape)
        )

    op = TensorDelete(index_obj=obj, axis=axis, dtype=arr.dtype)
    return op(arr, obj, shape)
