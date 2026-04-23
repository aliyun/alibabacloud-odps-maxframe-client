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
    """
    Rechunk DataFrame, Series or Index data.

    This function is used to reorganize the chunk structure of data, which can
    change how data is distributed in distributed computing. By adjusting chunk
    sizes, memory usage and computational performance can be optimized. Note that
    this function is only effective in DPE for now.

    Parameters
    ----------
    chunk_size : chunk_size_type
        New chunk size configuration. Can be one of the following forms:
        - Integer: Same chunk size for all dimensions
        - Tuple: Different chunk sizes for each dimension. For instance, (100, 100)
          rechunks dimension 0 and 1 into chunks with size 100 respectively, and
          ((100, 100, 100), (100,)) rechunks dimension 0 into 3 chunks with size
          100 for each chunk, and rechunks dimension 1 into one single chunk with
          size 100.
        - Dictionary: Chunk sizes for specific dimensions, for instance,
          `{0: 100}` rechunks dimension 0 into 100 elements per chunk, and
          `{0: (100, 100, 100)}` rechunks dimension 0 into 3 chunks where each
          chunk has size 100.
    reassign_worker : bool, default False
        Reserved for future use.

    Returns
    -------
    TileableType
        The rechunked DataFrame, Series or Index object.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': range(1000), 'B': range(1000)})
    >>> # Rechunk to 100 rows and columns per chunk
    >>> df_rechunked = df.rechunk(100)
    >>> # Specify different chunk sizes for different dimensions
    >>> df_rechunked = df.rechunk((50, 2))
    """
    op = DataFrameRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
    )
    return op(a)
