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
from ...serialization.serializables import AnyField
from .core import TensorNoInput


class Scalar(TensorNoInput):
    """
    Operator represents scalar type.
    """

    _op_type_ = opcodes.SCALAR

    data = AnyField("data", default=None)


def scalar(data, dtype=None, gpu=None):
    try:
        arr = np.array(data, dtype=dtype)
        op = Scalar(data=arr, dtype=arr.dtype, gpu=gpu)
        shape = ()
        return op(shape)
    except ValueError:
        raise TypeError(f"Expect scalar, got: {data}")
