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
from ...core import OutputType
from ...serialization.serializables import DataFrameField, SeriesField
from ..operators import DataFrameOperatorMixin
from ..utils import is_cudf, parse_index
from .core import PandasDataSourceOperator


class DataFrameDataSource(PandasDataSourceOperator, DataFrameOperatorMixin):
    """
    Represents data from pandas DataFrame
    """

    _op_type_ = opcodes.DATAFRAME_DATA_SOURCE

    data = DataFrameField("data")
    dtypes = SeriesField("dtypes")

    def __init__(self, data=None, dtypes=None, gpu=None, **kw):
        if dtypes is None and data is not None:
            dtypes = data.dtypes
        if gpu is None and is_cudf(data):  # pragma: no cover
            gpu = True
        super().__init__(
            data=data,
            dtypes=dtypes,
            gpu=gpu,
            _output_types=[OutputType.dataframe],
            **kw
        )

    def __call__(self, shape, chunk_size=None):
        return self.new_dataframe(
            None,
            shape,
            dtypes=self.dtypes,
            index_value=parse_index(self.data.index),
            columns_value=parse_index(self.data.columns, store_data=True),
            raw_chunk_size=chunk_size,
        )


def from_pandas(data, chunk_size=None, gpu=None, sparse=False):
    op = DataFrameDataSource(data=data, gpu=gpu, sparse=sparse)
    return op(data.shape, chunk_size=chunk_size)
