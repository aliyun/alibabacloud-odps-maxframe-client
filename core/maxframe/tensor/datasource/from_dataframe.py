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
from ...serialization.serializables import BoolField, KeyField
from ..core import TensorOrder
from ..utils import to_numpy
from .core import TensorHasInput


class TensorFromDataFrame(TensorHasInput):
    """represent tensor from DataFrame"""

    _op_type_ = opcodes.TENSOR_FROM_DATAFRAME
    _input = KeyField("_input")
    extract_multi_index = BoolField("extract_multi_index", default=None)

    def __call__(self, a, order=None):
        from ...dataframe.core import INDEX_TYPE, IndexValue

        if (
            self.extract_multi_index
            and isinstance(a, INDEX_TYPE)
            and isinstance(a.index_value.value, IndexValue.MultiIndex)
        ):
            order = a.order if order is None else order
            return self.new_tensor(
                [a], (a.shape[0], len(a.index_value.value.names)), order=order
            )
        else:
            self.extract_multi_index = False

        return super().__call__(a, order=order)


def from_dataframe(in_df, dtype=None):
    from ...dataframe.utils import build_empty_df

    if dtype is None:
        empty_pdf = build_empty_df(in_df.dtypes)
        dtype = to_numpy(empty_pdf).dtype
    op = TensorFromDataFrame(dtype=dtype, gpu=in_df.op.gpu)
    return op(in_df, order=TensorOrder.F_ORDER)  # return tensor with F-order always


def from_series(in_series, dtype=None):
    op = TensorFromDataFrame(dtype=dtype or in_series.dtype, gpu=in_series.op.gpu)
    return op(in_series, order=TensorOrder.F_ORDER)  # return tensor with F-order always


def from_index(in_index, dtype=None, extract_multi_index=False):
    op = TensorFromDataFrame(
        dtype=dtype or in_index.dtype,
        gpu=in_index.op.gpu,
        extract_multi_index=extract_multi_index,
    )
    return op(in_index, order=TensorOrder.F_ORDER)  # return tensor with F-order always
