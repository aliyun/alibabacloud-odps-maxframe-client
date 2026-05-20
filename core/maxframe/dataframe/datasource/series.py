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
from maxframe.core import OutputType
from maxframe.dataframe.datasource.core import PandasDataSourceOperator
from maxframe.dataframe.operators import DataFrameOperatorMixin
from maxframe.dataframe.utils import is_cudf, parse_index
from maxframe.serialization.serializables import DataTypeField, SeriesField


class SeriesDataSource(PandasDataSourceOperator, DataFrameOperatorMixin):
    """
    Represents data from pandas Series
    """

    _op_type_ = opcodes.SERIES_DATA_SOURCE

    data = SeriesField("data")
    dtype = DataTypeField("dtype")

    def __init__(self, data=None, dtype=None, gpu=None, **kw):
        if dtype is None and data is not None:
            dtype = data.dtype
        if gpu is None and is_cudf(data):  # pragma: no cover
            gpu = True
        super().__init__(
            data=data, dtype=dtype, gpu=gpu, _output_types=[OutputType.series], **kw
        )

    def __call__(self, shape, chunk_size=None):
        return self.new_series(
            None,
            shape=shape,
            dtype=self.dtype,
            index_value=parse_index(self.data.index),
            name=self.data.name,
            raw_chunk_size=chunk_size,
        )


def from_pandas(data, chunk_size=None, gpu=None, sparse=False):
    op = SeriesDataSource(data=data, gpu=gpu, sparse=sparse)
    return op(data.shape, chunk_size=chunk_size)
