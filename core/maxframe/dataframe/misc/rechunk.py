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
from ...serialization.serializables import AnyField
from ...tensor.rechunk.rechunk import chunk_size_type
from ...typing_ import TileableType
from ..core import DATAFRAME_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin


class DataFrameRechunk(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.RECHUNK

    chunk_size = AnyField("chunk_size")

    def __call__(self, x):
        if isinstance(x, DATAFRAME_TYPE):
            return self.new_dataframe(
                [x],
                shape=x.shape,
                dtypes=x.dtypes,
                columns_value=x.columns_value,
                index_value=x.index_value,
            )
        else:
            self.output_types = x.op.output_types
            f = (
                self.new_series
                if self.output_types[0] == OutputType.series
                else self.new_index
            )
            return f(
                [x],
                shape=x.shape,
                dtype=x.dtype,
                index_value=x.index_value,
                name=x.name,
            )


def rechunk(a: TileableType, chunk_size: chunk_size_type, reassign_worker=False):
    op = DataFrameRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
    )
    return op(a)
