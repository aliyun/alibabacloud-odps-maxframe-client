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
from ...serialization.serializables import Int32Field, StringField
from ..utils import get_order
from .core import TensorNoInput


class TensorEye(TensorNoInput):
    _op_type_ = opcodes.TENSOR_EYE

    k = Int32Field("k", default=0)
    order = StringField("order", default=None)

    def __init__(self, dtype=None, **kw):
        dtype = np.dtype(dtype or "f8")
        super().__init__(dtype=dtype, **kw)


def eye(N, M=None, k=0, dtype=None, sparse=False, gpu=None, chunk_size=None, order="C"):
    """
    Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned tensor.
    sparse: bool, optional
        Create sparse tensor if True, False as default
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    I : Tensor of shape (N,M)
      An tensor where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D tensor from a 1-D tensor specified by the user.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> mt.eye(2, dtype=int).execute()
    array([[1, 0],
           [0, 1]])
    >>> mt.eye(3, k=1).execute()
    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])

    """
    if M is None:
        M = N

    shape = (N, M)
    tensor_order = get_order(
        order,
        None,
        available_options="CF",
        err_msg="only 'C' or 'F' order is permitted",
    )
    op = TensorEye(k=k, dtype=dtype, gpu=gpu, sparse=sparse, order=order)
    return op(shape, chunk_size=chunk_size, order=tensor_order)
