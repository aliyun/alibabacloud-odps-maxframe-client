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
from maxframe.liteframe.operators.datasource import LiteFrameReadODPSTable
from maxframe.liteframe.utils import normalize_col_name, normalize_dtypes_index


def read_odps_table(
    table_name,
    partitions=None,
    columns=None,
    append_partitions=False,
    default_index_type=None,
    filters=None,
    **kwargs,
):
    """Read an ODPS table into a LiteFrame with Arrow dtypes.

    The ODPS entry is resolved from the global context or environment
    variables. Column names are normalized to be case-insensitive and
    dtypes are derived from the ODPS table schema using the pyarrow
    backend.

    Parameters
    ----------
    table_name : str
        Name of the ODPS table to read.
    partitions : str or list of str, optional
        Partition specs to read (e.g. ``'pt=00'``). If None, all
        partitions are read for partitioned tables.
    columns : list of str, optional
        Column names to select. Matching is case-insensitive. If None,
        all columns are read.
    append_partitions : bool, default False
        Whether to include partition spec columns (e.g. ``pt`` for
        ``pt=00``) as physical columns in the output. Automatically
        enabled when a partition column is named explicitly in
        ``columns``.
    default_index_type : str, optional
        Index type for the resulting LiteFrame. If None, a default is
        chosen by the backend.
    filters : list of tuple or list of list of tuple, optional
        Predicate filters to apply when reading the table.
    **kwargs
        Additional keyword arguments forwarded to the reader.

    Returns
    -------
    LiteFrame
        A lazy LiteFrame with Arrow dtypes representing the ODPS table
        data. Call ``.execute()`` to materialize the results.

    See Also
    --------
    LiteFrame.to_odps_table : Write a LiteFrame to an ODPS table.
    read_parquet : Read a Parquet file into a LiteFrame.
    """
    odps_entry = ODPS.from_global() or ODPS.from_environments()
    if odps_entry is None:
        raise ValueError("Missing odps_entry parameter")
    schema = options.session.default_schema or odps_entry.schema
    table = odps_entry.get_table(table_name, schema=schema)
    table.reload()

    # ``partitions`` only selects which partitions to read; ``append_partitions``
    # decides whether partition spec columns (e.g. ``pt`` for ``pt=00``) are
    # included as physical columns in the output dtypes.  Auto-enable it when a
    # partition column is named explicitly in ``columns``.
    table_partition_names = {
        normalize_col_name(p.name) for p in (table.table_schema.partitions or ())
    }
    append_partitions = bool(append_partitions) or (
        columns is not None
        and any(normalize_col_name(c) in table_partition_names for c in columns)
    )
    if not table.table_schema.partitions and (
        partitions is not None or append_partitions
    ):
        raise ValueError("Cannot handle partitions on non-partition tables")

    # Resolve dtypes from table schema using pyarrow backend.  Partition
    # columns are included only when append_partitions is set.
    col_objs = (
        table.table_schema.columns
        if append_partitions or columns is not None
        else table.table_schema.simple_columns
    )
    table_columns = [normalize_col_name(c.name) for c in col_objs]

    table_dtypes = odps_schema_to_pandas_dtypes(
        table.table_schema,
        with_partitions=True,
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
        append_partitions=append_partitions,
        default_index_type=default_index_type,
        filters=filters,
    )
    return op._new_liteframe_from_source()
