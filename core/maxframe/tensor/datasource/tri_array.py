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
from .array import tensor as astensor
from .core import TensorHasInput


class TensorTriArray(TensorHasInput):
    k = Int32Field("k")

    def __call__(self, m, order=None):
        order = TensorOrder.C_ORDER if order is None else order
        return self.new_tensor([m], shape=m.shape, order=order)


class TensorTriu(TensorTriArray):
    _op_type_ = opcodes.TENSOR_TRIU


def triu(m, k=0, gpu=None):
    """
    Upper triangle of a tensor.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    tril : lower triangle of a tensor

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1).execute()
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    """
    m = astensor(m)
    gpu = m.op.gpu if gpu is None else gpu
    op = TensorTriu(k=k, dtype=m.dtype, sparse=m.issparse(), gpu=gpu)
    return op(m)


class TensorTril(TensorTriArray):
    _op_type_ = opcodes.TENSOR_TRIL


def tril(m, k=0, gpu=None):
    """
    Lower triangle of a tensor.

    Return a copy of a tensor with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like, shape (M, N)
        Input tensor.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.
    gpu : bool, optional
        Allocate the tensor on GPU if True, None as default

    Returns
    -------
    tril : Tensor, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1).execute()
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    """
    m = astensor(m)
    gpu = m.op.gpu if gpu is None else gpu
    op = TensorTril(k=k, dtype=m.dtype, sparse=m.issparse(), gpu=gpu)
    return op(m)
