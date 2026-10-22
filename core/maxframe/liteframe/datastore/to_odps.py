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

from typing import Dict, List, Optional, Union

from odps import ODPS
from odps.models import Table as ODPSTable
from odps.types import OdpsSchema

from maxframe.config import options
from maxframe.io.odpsio import odps_schema_to_pandas_dtypes
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.core import LiteFrame as MFLiteFrame
from maxframe.liteframe.datastore.core import validate_write_mode
from maxframe.liteframe.operators.datastore import LiteFrameToODPSTable
from maxframe.liteframe.utils import normalize_col_name
from maxframe.typing_ import TileableType


def to_odps_table(
    lf: TileableType,
    table: Union[ODPSTable, str],
    partition: Optional[str] = None,
    partition_col: Union[None, str, List[str]] = None,
    mode: str = "overwrite",
    unknown_as_string: Optional[bool] = True,
    lifecycle: Optional[int] = None,
    table_properties: Optional[dict] = None,
    primary_key: Union[None, str, List[str]] = None,
    odps_types: Optional[Dict[str, str]] = None,
):
    """
    Write LiteFrame object into a MaxCompute (ODPS) table.

    Parameters
    ----------
    table : odps.models.Table or str
        ODPS table or table name to write to.
    partition : str, optional
        Partition spec for the target table (e.g. "ds=20240101").
    partition_col : str or list of str, optional
        Column name(s) to use as partition columns. The data is partitioned
        by these columns.
    mode : {'overwrite', 'append'}, default 'overwrite'
        Write mode. 'overwrite' drops the existing data; 'append' adds to it.
    unknown_as_string : bool, default True
        If True, columns with unknown ODPS types are written as strings.
    lifecycle : int, optional
        Table lifecycle in days.
    table_properties : dict, optional
        ODPS table properties.
    primary_key : str or list of str, optional
        Column name(s) to set as the table primary key.
    odps_types : dict, optional
        Mapping of column name to ODPS type string, used to override
        auto-detected types.

    Returns
    -------
    LiteFrame
        Stub LiteFrame for execution.

    See Also
    --------
    read_odps_table : Read an ODPS table into a LiteFrame.
    to_parquet : Write a LiteFrame to a Parquet file.
    """
    validate_write_mode(mode)
    if mode == "merge":
        raise NotImplementedError(
            "merge mode is not supported for ODPS tables; "
            "use 'overwrite' or 'append' instead."
        )

    odps_entry = ODPS.from_global() or ODPS.from_environments()
    is_schema_enabled = (
        options.session.enable_schema or odps_entry.is_schema_namespace_enabled()
    )

    if not isinstance(lf, LITEFRAME_TYPE):
        lf = MFLiteFrame(lf)

    # Drop hidden columns before writing
    if lf._hidden_columns:
        from maxframe.liteframe.indexing.select import drop as lf_drop

        lf = lf_drop(lf, list(lf._hidden_columns))

    if isinstance(table, ODPSTable):
        table = table.full_table_name
    elif is_schema_enabled and "." not in table:
        default_schema = (
            options.session.default_schema or odps_entry.schema or "default"
        )
        table = default_schema + "." + table

    if isinstance(partition_col, str):
        partition_col = [normalize_col_name(partition_col)]
    elif partition_col is not None:
        partition_col = [normalize_col_name(c) for c in partition_col]

    # Validate partition columns
    table_cols = set(lf.columns)  # already lowercased
    if partition_col:
        partition_diff = set(partition_col) - table_cols
        if partition_diff:
            raise ValueError(
                f"Partition column(s) {partition_diff}"
                " is not the data column(s) of the input liteframe."
            )

    # Handle primary_key
    table_properties = table_properties or {}
    if primary_key is not None:
        table_properties["transactional"] = "true"

    if primary_key and not isinstance(primary_key, (list, tuple)):
        primary_key = [normalize_col_name(primary_key)]
    elif primary_key is not None:
        primary_key = [normalize_col_name(c) for c in primary_key]

    # Determine target dtypes (use physical dtypes only, not range columns)
    if odps_types is None:
        target_dtypes = lf._data._physical_dtypes
    else:
        odps_schema = OdpsSchema.from_dict(odps_types)
        target_dtypes = odps_schema_to_pandas_dtypes(
            odps_schema, dtype_backend="pyarrow"
        )

    op = LiteFrameToODPSTable(
        dtypes=target_dtypes,
        table_name=table,
        unknown_as_string=unknown_as_string,
        partition_spec=partition,
        partition_columns=partition_col,
        mode=mode,
        lifecycle=lifecycle
        if lifecycle is not None
        else options.session.table_lifecycle,
        table_properties=table_properties or None,
        primary_key=primary_key or None,
    )
    return op(lf)
