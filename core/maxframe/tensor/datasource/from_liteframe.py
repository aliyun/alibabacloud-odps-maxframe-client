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

import numpy as np

from maxframe import opcodes
from maxframe.serialization.serializables import FieldTypes, KeyField, ListField
from maxframe.tensor.core import TensorOrder
from maxframe.tensor.datasource.core import TensorHasInput
from maxframe.utils import make_dtype


class TensorFromLiteFrame(TensorHasInput):
    """Represent tensor from LiteFrame."""

    _op_type_ = opcodes.TENSOR_FROM_LITEFRAME
    _input = KeyField("_input")
    visible_columns = ListField("visible_columns", FieldTypes.string, default=None)


def from_liteframe(in_lf, dtype=None):
    if dtype is None:
        np_dtypes = [make_dtype(dt, return_pd_dtype=False) for dt in in_lf.dtypes]
        dtype = np.result_type(*np_dtypes) if np_dtypes else np.dtype(np.float64)
    else:
        dtype = np.dtype(dtype)
    visible_columns = list(in_lf.columns)
    op = TensorFromLiteFrame(
        dtype=dtype, gpu=in_lf.op.gpu, visible_columns=visible_columns
    )
    return op(in_lf, order=TensorOrder.F_ORDER)
