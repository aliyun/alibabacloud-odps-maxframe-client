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

from maxframe import opcodes
from maxframe.serialization.serializables import BoolField, KeyField
from maxframe.tensor.core import TensorOrder
from maxframe.tensor.datasource.core import TensorHasInput
from maxframe.tensor.utils import to_numpy
from maxframe.utils import make_dtype


class TensorFromDataFrame(TensorHasInput):
    """represent tensor from DataFrame"""

    _op_type_ = opcodes.TENSOR_FROM_DATAFRAME
    _input = KeyField("_input")
    extract_multi_index = BoolField("extract_multi_index", default=None)

    def __call__(self, a, order=None):
        from maxframe.dataframe.core import INDEX_TYPE, IndexValue

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
    from maxframe.dataframe.utils import build_empty_df

    if dtype is None:
        empty_pdf = build_empty_df(in_df.dtypes)
        dtype = to_numpy(empty_pdf).dtype
    op = TensorFromDataFrame(dtype=dtype, gpu=in_df.op.gpu)
    return op(in_df, order=TensorOrder.F_ORDER)  # return tensor with F-order always


def from_series(in_series, dtype=None):
    dtype = make_dtype(dtype or in_series.dtype, return_pd_dtype=False)
    op = TensorFromDataFrame(dtype=dtype, gpu=in_series.op.gpu)
    return op(in_series, order=TensorOrder.F_ORDER)  # return tensor with F-order always


def from_index(in_index, dtype=None, extract_multi_index=False):
    dtype = make_dtype(dtype or in_index.dtype, return_pd_dtype=False)
    op = TensorFromDataFrame(
        dtype=dtype,
        gpu=in_index.op.gpu,
        extract_multi_index=extract_multi_index,
    )
    return op(in_index, order=TensorOrder.F_ORDER)  # return tensor with F-order always
