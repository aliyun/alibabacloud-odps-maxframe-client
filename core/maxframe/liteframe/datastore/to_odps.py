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

from maxframe import opcodes
from maxframe.config import options
from maxframe.core import OutputType
from maxframe.io.odpsio import odps_schema_to_pandas_dtypes
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.core import LiteFrame as MFLiteFrame
from maxframe.liteframe.datastore.core import LiteFrameDataStore, validate_write_mode
from maxframe.liteframe.utils import normalize_col_name
from maxframe.serialization.serializables import (
    BoolField,
    DictField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from maxframe.typing_ import TileableType


class LiteFrameToODPSTable(LiteFrameDataStore):
    _op_type_ = opcodes.TO_ODPS_TABLE

    dtypes = SeriesField("dtypes")
    table_name = StringField("table_name", default=None)
    partition_spec = StringField("partition_spec", default=None)
    partition_columns = ListField("partition_columns", FieldTypes.string, default=None)
    mode = StringField("mode", default=None)
    unknown_as_string = BoolField("unknown_as_string", default=None)
    lifecycle = Int64Field("lifecycle", default=None)
    table_properties = DictField("table_properties", default=None)
    primary_key = ListField("primary_key", FieldTypes.string, default=None)

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.liteframe], **kw)

    def can_fuse_with_custom_code(self) -> bool:
        return False

    def __call__(self, lf: TileableType):
        shape = (0,) * len(lf.shape)
        return self.new_liteframe(
            [lf],
            shape=shape,
            physical_dtypes=lf._data._physical_dtypes[:0],
        )


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

    Simplified version for LiteFrame with no index semantics.

    Parameters
    ----------
    lf: LiteFrame
        LiteFrame object to write
    table: str
        Name of the table to write LiteFrame into
    partition: Optional[str]
        Spec of the partition to write to, can be 'pt1=xxx,pt2=yyy'
    partition_col: Union[None, str, List[str]]
        Name of columns in LiteFrame as partition columns.
    mode: str
        Write mode. One of 'overwrite', 'append'. 'overwrite' replaces
        existing data, 'append' adds to existing data.
    unknown_as_string: bool
        If True, object type in the LiteFrame will be treated as strings.
    lifecycle: Optional[int]
        Specify lifecycle of the output table.
    table_properties: Optional[dict]
        Specify properties of the output table.
    primary_key: Union[None, str, List[str]]
        If provided and target table does not exist, target table
        will be a delta table with columns specified in this argument
        as primary key.
    odps_types: Optional[Dict[str, str]]
        Specify ODPS column types as a dict mapping column names to ODPS type strings.

    Returns
    -------
    result: LiteFrame
        Stub LiteFrame for execution.
    """
    validate_write_mode(mode)

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
