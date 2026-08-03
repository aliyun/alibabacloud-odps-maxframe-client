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

import pandas as pd
from odps import ODPS

from maxframe.config import options
from maxframe.io.odpsio import odps_schema_to_pandas_dtypes
from maxframe.liteframe.operators.source import LiteFrameReadODPSTable
from maxframe.liteframe.utils import normalize_col_name, normalize_dtypes_index


def read_odps_table(
    table_name,
    partitions=None,
    columns=None,
    default_index_type=None,
    **kwargs,
):
    """Read an ODPS table into a LiteFrame with Arrow dtypes."""
    odps_entry = ODPS.from_global() or ODPS.from_environments()
    if odps_entry is None:
        raise ValueError("Missing odps_entry parameter")
    schema = options.session.default_schema or odps_entry.schema
    table = odps_entry.get_table(table_name, schema=schema)
    table.reload()

    # Resolve dtypes from table schema using pyarrow backend
    col_objs = (
        table.table_schema.columns
        if partitions is not None
        else table.table_schema.simple_columns
    )
    table_columns = [normalize_col_name(c.name) for c in col_objs]

    table_dtypes = odps_schema_to_pandas_dtypes(
        table.table_schema,
        with_partitions=partitions is not None,
        dtype_backend="pyarrow",
    )
    # Build lookup with normalized (lowercased) keys
    table_dtypes_lowered = {normalize_col_name(k): v for k, v in table_dtypes.items()}
    df_types = [table_dtypes_lowered[c] for c in table_columns]

    if columns is not None:
        # Normalize the user-supplied column filter and do case-insensitive
        # membership check against the lowered table columns
        columns_lowered = [normalize_col_name(c) for c in columns]
        missing_columns = sorted(set(columns_lowered) - set(table_columns))
        if missing_columns:
            raise ValueError(
                f"Column(s) {missing_columns} not found in ODPS table {table_name}"
            )
        df_types = [table_dtypes_lowered[c] for c in columns_lowered]
        table_columns = columns_lowered
        columns = columns_lowered

    dtypes = pd.Series(df_types, index=table_columns)
    dtypes = normalize_dtypes_index(dtypes)

    if isinstance(partitions, str):
        partitions = [partitions]

    op = LiteFrameReadODPSTable(
        table_name=table.full_table_name,
        dtypes=dtypes,
        partitions=partitions,
        columns=columns,
        default_index_type=default_index_type,
    )
    return op._new_liteframe_from_source()
