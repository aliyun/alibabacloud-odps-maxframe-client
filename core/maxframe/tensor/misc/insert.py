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
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField, Int32Field, TupleField
from ..operators import TensorHasInput, TensorOperatorMixin
from ..utils import calc_object_length, filter_inputs, validate_axis


class TensorInsert(TensorHasInput, TensorOperatorMixin):
    _op_type_ = opcodes.INSERT

    index_obj = AnyField("index_obj", default=None)
    values = AnyField("values", default=None)
    axis = Int32Field("axis", default=None)

    # for chunk
    range_on_axis = TupleField("range_on_axis", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorInsert", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(inputs[1:])
        if isinstance(op.index_obj, ENTITY_TYPE):
            op.index_obj = next(inputs_iter)
        if isinstance(op.values, ENTITY_TYPE):
            op.values = next(inputs_iter)

    def __call__(self, arr, obj, values, shape):
        return self.new_tensor(
            filter_inputs([arr, obj, values]), shape=shape, order=arr.order
        )


def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : array like
        Input array.
    obj : int, slice or sequence of ints
        Object that defines the index or indices before which `values` is
        inserted.
    values : array_like
        Values to insert into `arr`. If the type of `values` is different
        from that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that ``arr[...,obj,...] = values``
        is legal.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.
    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.
    See Also
    --------
    append : Append elements at the end of an array.
    concatenate : Join a sequence of arrays along an existing axis.
    delete : Delete elements from an array.
    Notes
    -----
    Note that for higher dimensional inserts `obj=0` behaves very different
    from `obj=[0]` just like `arr[:,0,:] = values` is different from
    `arr[:,[0],:] = values`.
    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = mt.array([[1, 1], [2, 2], [3, 3]])
    >>> a.execute()
    array([[1, 1],
           [2, 2],
           [3, 3]])
    >>> mt.insert(a, 1, 5).execute()
    array([1, 5, 1, ..., 2, 3, 3])
    >>> mt.insert(a, 1, 5, axis=1).execute()
    array([[1, 5, 1],
           [2, 5, 2],
           [3, 5, 3]])
    Difference between sequence and scalars:
    >>> mt.insert(a, [1], [[1],[2],[3]], axis=1).execute()
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> b = a.flatten()
    >>> b.execute()
    array([1, 1, 2, 2, 3, 3])
    >>> mt.insert(b, [2, 2], [5, 6]).execute()
    array([1, 1, 5, ..., 2, 3, 3])
    >>> mt.insert(b, slice(2, 4), [5, 6]).execute()
    array([1, 1, 5, ..., 2, 3, 3])
    >>> mt.insert(b, [2, 2], [7.13, False]).execute() # type casting
    array([1, 1, 7, ..., 2, 3, 3])
    >>> x = mt.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> mt.insert(x, idx, 999, axis=1).execute()
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])
    """
    from ..datasource import tensor as astensor

    arr = astensor(arr)
    if getattr(obj, "ndim", 0) > 1:  # pragma: no cover
        raise ValueError(
            "index array argument obj to insert must be one dimensional or scalar"
        )

    if axis is None:
        # if axis is None, array will be flatten
        arr_size = arr.size
        idx_length = calc_object_length(obj, size=arr_size)
        shape = (arr_size + idx_length,)
    else:
        validate_axis(arr.ndim, axis)
        idx_length = calc_object_length(obj, size=arr.shape[axis])
        shape = tuple(
            s + idx_length if i == axis else s for i, s in enumerate(arr.shape)
        )

    op = TensorInsert(index_obj=obj, values=values, axis=axis, dtype=arr.dtype)
    return op(arr, obj, values, shape)
