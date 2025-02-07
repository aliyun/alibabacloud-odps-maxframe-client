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
    op = TensorRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
        dtype=tensor.dtype,
        sparse=tensor.issparse(),
    )
    return op(tensor)
