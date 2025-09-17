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
from ...core import ENTITY_TYPE, ExecutableTuple
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Int32Field,
    ListField,
    StringField,
)
from ...typing_ import EntityType
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import validate_axis, validate_order


class TensorPartition(TensorOperatorMixin, TensorOperator):
    _op_type_ = opcodes.PARTITION

    kth = AnyField("kth")
    axis = Int32Field("axis")
    kind = StringField("kind")
    order = ListField("order", FieldTypes.string)
    need_align = BoolField("need_align")
    return_value = BoolField("return_value")
    return_indices = BoolField("return_indices")

    @classmethod
    def _set_inputs(cls, op: "TensorPartition", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        if len(op._inputs) > 1:
            op.kth = op._inputs[1]

    @property
    def psrs_kinds(self):
        # to keep compatibility with PSRS
        # remember when merging data in PSRSShuffle(reduce),
        # we don't need sort, thus set psrs_kinds[2] to None
        return ["quicksort", "mergesort", None]

    @property
    def output_limit(self):
        return int(bool(self.return_value)) + int(bool(self.return_indices))

    def __call__(self, a, kth):
        inputs = [a]
        if isinstance(kth, TENSOR_TYPE):
            inputs.append(kth)
        kws = []
        if self.return_value:
            kws.append(
                {
                    "shape": a.shape,
                    "order": a.order,
                    "type": "sorted",
                    "dtype": a.dtype,
                }
            )
        if self.return_indices:
            kws.append(
                {
                    "shape": a.shape,
                    "order": TensorOrder.C_ORDER,
                    "type": "argsort",
                    "dtype": np.dtype(np.int64),
                }
            )
        ret = self.new_tensors(inputs, kws=kws)
        if len(kws) == 1:
            return ret[0]
        return ExecutableTuple(ret)


def _check_kth_dtype(dtype):
    if not np.issubdtype(dtype, np.integer):
        raise TypeError("Partition index must be integer")


def _validate_kth_value(kth, size):
    kth = np.where(kth < 0, kth + size, kth)
    if np.any((kth < 0) | (kth >= size)):
        invalid_kth = next(k for k in kth if k < 0 or k >= size)
        raise ValueError(f"kth(={invalid_kth}) out of bounds ({size})")
    return kth


def _validate_partition_arguments(a, kth, axis, kind, order, kw):
    a = astensor(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    else:
        axis = validate_axis(a.ndim, axis)
    if isinstance(kth, ENTITY_TYPE):
        kth = astensor(kth)
        _check_kth_dtype(kth.dtype)
    else:
        kth = np.atleast_1d(kth)
        kth = _validate_kth_value(kth, a.shape[axis])
    if kth.ndim > 1:
        raise ValueError("object too deep for desired array")
    if kind != "introselect":
        raise ValueError(f"{kind} is an unrecognized kind of select")
    # if a is structure type and order is not None
    order = validate_order(a.dtype, order)
    need_align = kw.pop("need_align", None)
    if len(kw) > 0:
        raise TypeError(
            f"partition() got an unexpected keyword argument '{next(iter(kw))}'"
        )

    return a, kth, axis, kind, order, need_align


def partition(a, kth, axis=-1, kind="introselect", order=None, **kw):
    r"""
    Return a partitioned copy of a tensor.

    Creates a copy of the tensor with its elements rearranged in such a
    way that the value of the element in k-th position is in the
    position it would be in a sorted tensor. All elements smaller than
    the k-th element are moved before this element and all equal or
    greater are moved behind it. The ordering of the elements in the two
    partitions is undefined.

    Parameters
    ----------
    a : array_like
        Tensor to be sorted.
    kth : int or sequence of ints
        Element index to partition by. The k-th value of the element
        will be in its final sorted position and all smaller elements
        will be moved before it and all equal or greater elements behind
        it. The order of all elements in the partitions is undefined. If
        provided with a sequence of k-th it will partition all elements
        indexed by k-th  of them into their sorted position at once.
    axis : int or None, optional
        Axis along which to sort. If None, the tensor is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'.
    order : str or list of str, optional
        When `a` is a tensor with fields defined, this argument
        specifies which fields to compare first, second, etc.  A single
        field can be specified as a string.  Not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    partitioned_tensor : Tensor
        Tensor of the same type and shape as `a`.

    See Also
    --------
    Tensor.partition : Method to sort a tensor in-place.
    argpartition : Indirect partition.
    sort : Full sorting

    Notes
    -----
    The various selection algorithms are characterized by their average
    speed, worst case performance, work space size, and whether they are
    stable. A stable sort keeps items with the same key in the same
    relative order. The available algorithms have the following
    properties:

    ================= ======= ============= ============ =======
       kind            speed   worst case    work space  stable
    ================= ======= ============= ============ =======
    'introselect'        1        O(n)           0         no
    ================= ======= ============= ============ =======

    All the partition algorithms make temporary copies of the data when
    partitioning along any but the last axis.  Consequently,
    partitioning along the last axis is faster and uses less space than
    partitioning along any other axis.

    The sort order for complex numbers is lexicographic. If both the
    real and imaginary parts are non-nan then the order is determined by
    the real parts except when they are equal, in which case the order
    is determined by the imaginary parts.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> a = mt.array([3, 4, 2, 1])
    >>> mt.partition(a, 3).execute()
    array([2, 1, 3, 4])

    >>> mt.partition(a, (1, 3)).execute()
    array([1, 2, 3, 4])
    """
    return_indices = kw.pop("return_index", False)
    a, kth, axis, kind, order, need_align = _validate_partition_arguments(
        a, kth, axis, kind, order, kw
    )
    op = TensorPartition(
        kth=kth,
        axis=axis,
        kind=kind,
        order=order,
        need_align=need_align,
        return_value=True,
        return_indices=return_indices,
        dtype=a.dtype,
        gpu=a.op.gpu,
    )
    return op(a, kth)
