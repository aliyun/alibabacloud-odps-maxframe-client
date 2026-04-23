# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from typing import Tuple, Union

from ... import opcodes
from ...serialization.serializables import AnyField
from ..core import Tensor
from ..operators import TensorOperator, TensorOperatorMixin

chunk_size_type = Union[int, Tuple[int], Tuple[Tuple[int], ...]]


class TensorRechunk(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.RECHUNK

    chunk_size = AnyField("chunk_size")

    def __call__(self, tensor: Tensor):
        return self.new_tensor([tensor], tensor.shape, order=tensor.order)


def rechunk(
    tensor: Tensor, chunk_size: chunk_size_type, reassign_worker=False
) -> Tensor:
    """
    Rechunk tensor data.

    This function is used to reorganize the chunk structure of data, which can
    change how data is distributed in distributed computing. By adjusting chunk
    sizes, memory usage and computational performance can be optimized. Note that
    this function is only effective in DPE for now.

    Parameters
    ----------
    chunk_size : chunk_size_type
        New chunk size configuration. Can be one of the following forms:
        - Integer: Same chunk size for all dimensions
        - Tuple: Different chunk sizes for each dimension. For instance, (100, 100)
          rechunks dimension 0 and 1 into chunks with size 100 respectively, and
          ((100, 100, 100), (100,)) rechunks dimension 0 into 3 chunks with size
          100 for each chunk, and rechunks dimension 1 into one single chunk with
          size 100.
        - Dictionary: Chunk sizes for specific dimensions, for instance,
          `{0: 100}` rechunks dimension 0 into 100 elements per chunk, and
          `{0: (100, 100, 100)}` rechunks dimension 0 into 3 chunks where each
          chunk has size 100.
    reassign_worker : bool, default False
        Reserved for future use.

    Returns
    -------
    TileableType
        The rechunked tensor object.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> t = mt.random.rand(1000, 1000)
    >>> # Rechunk to 100 rows and columns per chunk
    >>> t_rechunked = t.rechunk(100)
    >>> # Specify different chunk sizes for different dimensions
    >>> t_rechunked = t.rechunk((50, 60))
    """
    op = TensorRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
        dtype=tensor.dtype,
        sparse=tensor.issparse(),
    )
    return op(tensor)
