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

import numpy as np

from ... import opcodes
from ...serialization.serializables import Int32Field
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import validate_axis


class TensorConcatenate(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.CONCATENATE

    axis = Int32Field("axis", default=None)

    def __call__(self, tensors):
        axis = self.axis
        shape = _calc_concatenate_shape(tensors, axis)
        shape[axis] = sum(t.shape[axis] for t in tensors)
        return self.new_tensor(tensors, shape=tuple(shape))


def concatenate(tensors, axis=0):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The tensors must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the tensors will be joined.  Default is 0.

    Returns
    -------
    res : Tensor
        The concatenated tensor.

    See Also
    --------
    stack : Stack a sequence of tensors along a new axis.
    vstack : Stack tensors in sequence vertically (row wise)

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> a = mt.array([[1, 2], [3, 4]])
    >>> b = mt.array([[5, 6]])
    >>> mt.concatenate((a, b), axis=0).execute()
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> mt.concatenate((a, b.T), axis=1).execute()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    if axis is None:
        axis = 0
    tensors = [astensor(t) for t in tensors]
    axis = validate_axis(tensors[0].ndim, axis)

    if len(set(t.ndim for t in tensors)) != 1:
        raise ValueError("all the input tensors must have same number of dimensions")

    shapes = [t.shape[:axis] + t.shape[axis + 1 :] for t in tensors]
    if len(set(shapes)) != 1:
        raise ValueError(
            "all the input tensor dimensions "
            "except for the concatenation axis must match exactly"
        )
    shape = _calc_concatenate_shape(tensors, axis)
    if any(np.isnan(s) for i, s in enumerate(shape) if i != axis):
        raise ValueError("cannot concatenate tensor with unknown shape")

    return _concatenate(tensors, axis)


def _concatenate(tensors, axis=0):
    dtype = np.result_type(*(t.dtype for t in tensors))
    sparse = all(t.issparse() for t in tensors)

    op = TensorConcatenate(axis=axis, dtype=dtype, sparse=sparse)
    return op(tensors)


def _calc_concatenate_shape(tensors, axis):
    return [0 if i == axis else tensors[0].shape[i] for i in range(tensors[0].ndim)]
