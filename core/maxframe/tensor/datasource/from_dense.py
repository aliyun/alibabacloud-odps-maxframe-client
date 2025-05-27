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
from ...serialization.serializables import AnyField, KeyField
from .array import tensor
from .core import TensorHasInput


class DenseToSparse(TensorHasInput):
    _op_type_ = opcodes.DENSE_TO_SPARSE

    _input = KeyField("input")
    missing = AnyField("missing", default=None)

    def __init__(self, **kw):
        super().__init__(sparse=True, **kw)


def fromdense(a, missing=None):
    a = tensor(a)
    if a.issparse():
        return a

    op = DenseToSparse(dtype=a.dtype, gpu=a.op.gpu, missing=missing)
    return op(a)
