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
from ...serialization.serializables import AnyField, KeyField, StringField
from ..utils import get_order
from .array import tensor
from .core import TensorHasInput


class SparseToDense(TensorHasInput):
    _op_type_ = opcodes.SPARSE_TO_DENSE

    _input = KeyField("input")
    order = StringField("order", default=None)
    fill_value = AnyField("fill_value", default=None)

    def __init__(self, **kw):
        super().__init__(sparse=False, **kw)


def fromsparse(a, order="C", fill_value=None):
    a = tensor(a)
    if not a.issparse():
        return a.astype(a.dtype, order=order, copy=False)

    tensor_order = get_order(
        order,
        None,
        available_options="CF",
        err_msg="only 'C' or 'F' order is permitted",
    )
    op = SparseToDense(dtype=a.dtype, gpu=a.op.gpu, order=order, fill_value=fill_value)
    return op(a, order=tensor_order)
