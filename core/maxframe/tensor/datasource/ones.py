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
from ...serialization.serializables import (
    AnyField,
    FieldTypes,
    KeyField,
    StringField,
    TupleField,
)
from ...utils import on_deserialize_shape, on_serialize_shape
from ..utils import get_order
from .array import tensor
from .core import TensorLike, TensorNoInput


class TensorOnes(TensorNoInput):
    _op_type_ = opcodes.TENSOR_ONES

    order = StringField("order")
    shape = TupleField(
        "shape",
        FieldTypes.int64,
        on_serialize=on_serialize_shape,
        on_deserialize=on_deserialize_shape,
    )
    chunk_size = AnyField("chunk_size")

    def __init__(self, shape=None, **kwargs):
        if type(shape) is int:
            shape = (shape,)
        super().__init__(shape=shape, **kwargs)


def ones(shape, dtype=None, chunk_size=None, gpu=None, order="C"):
    """
    Return a new tensor of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the tensor, e.g., `mt.int8`.  Default is
        `mt.float64`.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    out : Tensor
        Tensor of ones with the given shape, dtype, and order.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.ones(5).execute()
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> mt.ones((5,), dtype=int).execute()
    array([1, 1, 1, 1, 1])

    >>> mt.ones((2, 1)).execute()
    array([[ 1.],
           [ 1.]])

    >>> s = (2,2)
    >>> mt.ones(s).execute()
    array([[ 1.,  1.],
           [ 1.,  1.]])

    """
    tensor_order = get_order(
        order,
        None,
        available_options="CF",
        err_msg="only 'C' or 'F' order is permitted",
    )
    dtype = np.dtype(dtype or "f8")
    op = TensorOnes(
        dtype=dtype, shape=shape, chunk_size=chunk_size, gpu=gpu, order=order
    )
    return op(shape, chunk_size=chunk_size, order=tensor_order)


class TensorOnesLike(TensorLike):
    _op_type_ = opcodes.TENSOR_ONES_LIKE

    _input = KeyField("input")
    order = StringField("order")

    def __init__(self, dtype=None, sparse=False, **kw):
        dtype = np.dtype(dtype) if dtype is not None else None
        super().__init__(dtype=dtype, sparse=sparse, **kw)


def ones_like(a, dtype=None, gpu=None, order="K"):
    """
    Return a tensor of ones with the same shape and type as a given tensor.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned tensor.
    dtype : data-type, optional
        Overrides the data type of the result.
    gpu : bool, optional
        Allocate the tensor on GPU if True, None as default
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.

    Returns
    -------
    out : Tensor
        Tensor of ones with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return a tensor of zeros with shape and type of input.
    empty_like : Return a empty tensor with shape and type of input.
    zeros : Return a new tensor setting values to zero.
    ones : Return a new tensor setting values to one.
    empty : Return a new uninitialized tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.ones_like(x).execute()
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = mt.arange(3, dtype=float)
    >>> y.execute()
    array([ 0.,  1.,  2.])
    >>> mt.ones_like(y).execute()
    array([ 1.,  1.,  1.])

    """
    a = tensor(a)
    tensor_order = get_order(order, a.order)
    gpu = a.op.gpu if gpu is None else gpu
    op = TensorOnesLike(dtype=dtype, gpu=gpu, sparse=a.issparse(), order=order)
    return op(a, order=tensor_order)
