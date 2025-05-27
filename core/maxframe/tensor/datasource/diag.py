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
from ...lib.sparse import diag as sparse_diag
from ...lib.sparse import get_sparse_module, issparse
from ...lib.sparse.core import get_array_module
from ...serialization.serializables import Int32Field
from ..core import TENSOR_TYPE, TensorOrder
from .array import tensor
from .core import TensorHasInput


def _get_diag_shape(v_shape, k):
    size_0, size_1 = 0, 0
    if k > 0:
        size_1 += k
    elif k < 0:
        size_0 -= k
    size = min(v_shape[0] - size_0, v_shape[1] - size_1)
    return (size,)


class TensorDiag(TensorHasInput):
    _op_type_ = opcodes.TENSOR_DIAG

    k = Int32Field("k", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorDiag", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.dtype is None:
            op._dtype = op.inputs[0].dtype

    def __call__(self, v, shape, chunk_size=None):
        return self.new_tensor(
            [v], shape, raw_chunk_size=chunk_size, order=TensorOrder.C_ORDER
        )


def diag(v, k=0, sparse=None, gpu=None, chunk_size=None):
    """
    Extract a diagonal or construct a diagonal tensor.

    See the more detailed documentation for ``mt.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting tensor

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D tensor, return its `k`-th diagonal.
        If `v` is a 1-D tensor, return a 2-D tensor with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.
    sparse: bool, optional
        Create sparse tensor if True, False as default
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    out : Tensor
        The extracted diagonal or constructed diagonal tensor.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of a tensor.
    tril : Lower triangle of a tensor.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> x = mt.arange(9).reshape((3,3))
    >>> x.execute()
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> mt.diag(x).execute()
    array([0, 4, 8])
    >>> mt.diag(x, k=1).execute()
    array([1, 5])
    >>> mt.diag(x, k=-1).execute()
    array([3, 7])

    >>> mt.diag(mt.diag(x)).execute()
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    if not isinstance(v, TENSOR_TYPE):
        tensor_v = tensor(v)
        if tensor_v.issparse():
            xps = get_sparse_module(tensor_v.data)
            v = xps.csr_matrix(
                (tensor_v.op.data, tensor_v.op.indices, tensor_v.op.indptr),
                tensor_v.shape,
            )
            diag_v = sparse_diag(v, k=k)
        else:
            v = tensor(v).op.data
            diag_v = get_array_module(v).diag(v, k=k)
        sparse = sparse if sparse is not None else issparse(v)
        return tensor(diag_v, gpu=gpu, sparse=sparse, chunk_size=chunk_size)

    sparse = sparse if sparse is not None else v.issparse()

    if v.ndim == 1:
        shape = (v.size + abs(k),) * 2
    elif v.ndim == 2:
        shape = _get_diag_shape(v.shape, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")

    op = TensorDiag(k=k, dtype=v.dtype, gpu=gpu, sparse=sparse)
    return op(v, shape)
