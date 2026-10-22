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
from maxframe.dataframe.core import DATAFRAME_TYPE
from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.dataframe.utils import validate_axis
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int64Field,
)
from maxframe.tensor.rechunk.rechunk import TensorRechunk, chunk_size_type
from maxframe.typing_ import TileableType


class DataFrameRechunk(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.RECHUNK

    chunk_size = AnyField("chunk_size", default=None)
    axis = Int64Field("axis", default=None)
    factor = Float64Field("factor", default=None)
    num_partitions = Int64Field("num_partitions", default=None)
    rough = BoolField("rough", default=False)

    validate_rebalance_args = staticmethod(TensorRechunk.validate_rebalance_args)
    normalize_rechunk_axis = staticmethod(TensorRechunk.normalize_rechunk_axis)

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


def rechunk(
    a: TileableType, chunk_size: chunk_size_type, reassign_worker=False, axis=None
):
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
    axis : int, optional
        If specified, ``chunk_size`` applies only to that axis. Internally
        normalized to the dict form ``{axis: chunk_size}``. Cannot be used
        together with a dict ``chunk_size``.

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
    >>> # Rechunk only axis 0
    >>> df_rechunked = df.rechunk(100, axis=0)
    """
    chunk_size, _ = DataFrameRechunk.normalize_rechunk_axis(
        axis, chunk_size, validate_axis, axis, a
    )
    op = DataFrameRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
    )
    return op(a)


def rebalance(df_or_series, axis=0, factor=None, num_partitions=None, rough=False):
    """
    Make data more balanced across entire cluster.

    Parameters
    ----------
    axis : int
        The axis to rebalance.
    factor : float
        Specified so the number of chunks after balance along the selected
        axis per plane is ``chunk_shape[axis] * factor``.
    num_partitions : int
        Specified so the number of chunks are at most
        num_partitions.
    rough : bool, default False
        Hint to rebalance without first resolving unknown axis shapes.

        When True, rebalance may partition the pivoting axis using a per-chunk
        size estimate instead of waiting for real (resolved) shapes. This only
        has an effect when the pivoting axis shape is unknown **and** the source
        chunks carry such an estimate; currently that estimate is produced only
        by SIZE-mode ``read_odps_table`` reads along the row axis (axis 0). The
        rough path is therefore restricted to ``axis=0`` -- on any other axis
        ``rough=True`` falls back to the normal path even when shapes are
        unknown, because the row-share estimate would collapse to an
        uninformative equal split. In every other case -- known shapes, or an
        unknown row axis with no estimate -- ``rough=True`` silently falls back
        to the normal path (resolving real shapes then rechunking). The estimate
        is approximate, so the output is balanced by estimated share, not
        exactly equal in size.

        .. note:: ``rough`` is infrastructure for the upcoming SIZE-mode
           ``read_odps_table`` -> rough rebalance pipeline. No production call
           site in the current corpus passes ``rough=True``; it is exercised by
           the producer + downstream-engagement tests until a consumer lands.

    Returns
    -------
    Series or DataFrame
        Result of DataFrame or Series after rebalanced.
    """
    axis = validate_axis(axis, df_or_series)

    DataFrameRechunk.validate_rebalance_args(factor, num_partitions)

    op = DataFrameRechunk(
        chunk_size=None,
        axis=axis,
        factor=factor,
        num_partitions=num_partitions,
        rough=rough,
    )
    return op(df_or_series)
