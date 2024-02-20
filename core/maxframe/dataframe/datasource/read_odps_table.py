# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from odps import ODPS
from odps.models import Table
from odps.utils import to_timestamp

from ... import opcodes
from ...core import OutputType
from ...odpsio import odps_schema_to_pandas_dtypes
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from ..core import DataFrame  # noqa: F401
from ..utils import parse_index
from .core import ColumnPruneSupportedDataSourceMixin, IncrementalIndexDatasource

logger = logging.getLogger(__name__)


class DataFrameReadODPSTable(
    IncrementalIndexDatasource,
    ColumnPruneSupportedDataSourceMixin,
):
    _op_type_ = opcodes.READ_ODPS_TABLE

    table_name = StringField("table_name")
    partitions = ListField("partitions", field_type=FieldTypes.string, default=None)
    dtypes = SeriesField("dtypes", default=None)
    columns = AnyField("columns", default=None)
    nrows = Int64Field("nrows", default=None)
    use_arrow_dtype = BoolField("use_arrow_dtype", default=None)
    string_as_binary = BoolField("string_as_binary", default=None)
    append_partitions = BoolField("append_partitions", default=None)
    last_modified_time = Int64Field("last_modified_time", default=None)
    index_columns = ListField("index_columns", FieldTypes.string, default=None)
    index_dtypes = SeriesField("index_dtypes", default=None)

    def __init__(self, memory_scale=None, **kw):
        output_type = kw.get("output_type", OutputType.dataframe)
        super(DataFrameReadODPSTable, self).__init__(
            memory_scale=memory_scale, _output_types=[output_type], **kw
        )

    @property
    def partition(self):
        return getattr(self, "partition_spec", None)

    def get_columns(self):
        return self.columns

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        self.columns = columns

    def __call__(self, shape, chunk_bytes=None, chunk_size=None):
        if not self.index_columns:
            if np.isnan(shape[0]):
                index_value = parse_index(pd.RangeIndex(0))
            else:
                index_value = parse_index(pd.RangeIndex(shape[0]))
        elif len(self.index_columns) == 1:
            index_value = parse_index(
                pd.Index([], name=self.index_columns[0]).astype(self.index_dtypes[0])
            )
        else:
            idx = pd.MultiIndex.from_frame(
                pd.DataFrame([], columns=self.index_columns).astype(self.index_dtypes)
            )
            index_value = parse_index(idx)

        if self.output_types[0] == OutputType.dataframe:
            columns_value = parse_index(self.dtypes.index, store_data=True)
            return self.new_tileable(
                [],
                None,
                shape=shape,
                dtypes=self.dtypes,
                index_value=index_value,
                columns_value=columns_value,
                chunk_bytes=chunk_bytes,
                chunk_size=chunk_size,
            )
        elif self.output_types[0] == OutputType.series:
            return self.new_tileable(
                [],
                None,
                shape=shape[:1],
                name=self.dtypes.index[0],
                dtype=self.dtypes.iloc[0],
                index_value=index_value,
                chunk_bytes=chunk_bytes,
                chunk_size=chunk_size,
            )
        elif self.output_types[0] == OutputType.index:
            return self.new_tileable(
                [],
                None,
                shape=shape,
                name=getattr(index_value, "name", None),
                names=getattr(index_value, "names", None),
                index_value=index_value,
                chunk_bytes=chunk_bytes,
                chunk_size=chunk_size,
            )


def read_odps_table(
    table_name: Union[str, Table],
    partitions: Union[None, str, List[str]] = None,
    columns: Optional[List[str]] = None,
    index_col: Union[None, str, List[str]] = None,
    odps_entry: ODPS = None,
    string_as_binary: bool = None,
    append_partitions: bool = False,
    **kw,
):
    """
    Read data from a MaxCompute (ODPS) table into DataFrame.

    Supports specifying some columns as indexes. If not specified, RangeIndex
    will be generated.

    Parameters
    ----------
    table_name: Union[str, Table]
        Name of the table to read from.
    partitions: Union[None, str, List[str]]
        Table partition or list of partitions to read from.
    columns: Optional[List[str]]
        Table columns to read from. You may also specify partition columns here.
        If not specified, all table columns (or include partition columns if
        `append_partitions` is True) will be included.
    index_col: Union[None, str, List[str]]
        Columns to be specified as indexes.
    append_partitions: bool
        If True, will add all partition columns as selected columns when
        `columns` is not specified,

    Returns
    -------
    result: DataFrame
        DataFrame read from MaxCompute (ODPS) table
    """
    odps_entry = odps_entry or ODPS.from_global() or ODPS.from_environments()
    if isinstance(table_name, Table):
        table = table_name
    else:
        table = odps_entry.get_table(table_name)

    if not table.table_schema.partitions and (
        partitions is not None or append_partitions
    ):
        raise ValueError("Cannot handle partitions on non-partition tables")

    cols = (
        table.table_schema.columns
        if append_partitions or columns is not None
        else table.table_schema.simple_columns
    )
    table_columns = [c.name.lower() for c in cols]
    table_dtypes = odps_schema_to_pandas_dtypes(
        table.table_schema, with_partitions=True
    )
    df_types = [table_dtypes[c] for c in table_columns]

    if isinstance(index_col, str):
        index_col = [index_col]
    if index_col and columns is None:
        index_col_set = set([c.lower() for c in index_col])
        col_diff = sorted(index_col_set - set(table_columns))
        if col_diff:
            raise ValueError(
                f"Cannot found column {', '.join(col_diff)} specified "
                f"in index_columns argument in source table"
            )
        columns = [c for c in table_columns if c not in index_col_set]

    if not index_col:
        index_dtypes = None
    else:
        table_index_types = [df_types[table_columns.index(col)] for col in index_col]
        index_dtypes = pd.Series(table_index_types, index=index_col)

    if columns is not None:
        table_col_set = set([c.lower() for c in columns])
        col_diff = sorted(table_col_set - set(table_columns))
        if col_diff:
            raise ValueError(
                f"Cannot found column {', '.join(col_diff)} specified "
                f"in columns argument in source table"
            )
        if any(col in table_col_set for col in index_col or ()):
            raise ValueError("Index columns and columns shall not overlap.")

        # reorder columns
        new_columns = [c for c in table_columns if c in table_col_set]
        df_types = [df_types[table_columns.index(col)] for col in new_columns]
        table_columns = new_columns
        columns = new_columns

    shape = kw.pop("shape", None) or (np.nan, len(df_types))
    dtypes = pd.Series(df_types, index=table_columns)
    chunk_bytes = kw.pop("chunk_bytes", None)
    chunk_size = kw.pop("chunk_size", None)
    use_arrow_dtype = kw.pop("use_arrow_dtype", True)

    partitions = partitions or kw.get("partition")
    if isinstance(partitions, str):
        partitions = [partitions]

    append_partitions = append_partitions or any(
        pt.name in (columns or ()) for pt in (table.table_schema.partitions or ())
    )
    op = DataFrameReadODPSTable(
        table_name=table.full_table_name,
        partitions=partitions,
        dtypes=dtypes,
        columns=columns,
        use_arrow_dtype=use_arrow_dtype,
        string_as_binary=string_as_binary,
        append_partitions=append_partitions,
        last_modified_time=to_timestamp(table.last_data_modified_time),
        index_columns=index_col,
        index_dtypes=index_dtypes,
        **kw,
    )
    return op(shape, chunk_bytes=chunk_bytes, chunk_size=chunk_size)
