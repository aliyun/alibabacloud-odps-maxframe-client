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

import itertools
import logging
from typing import Any, List, Optional, Union

from odps import ODPS
from odps.models import Table as ODPSTable
from odps.types import PartitionSpec

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...io.odpsio import build_dataframe_table_meta
from ...serialization.serializables import (
    BoolField,
    DictField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from ...typing_ import TileableType
from ..core import DataFrame  # noqa: F401
from ..utils import parse_index
from .core import DataFrameDataStore

logger = logging.getLogger(__name__)


class DataFrameToODPSTable(DataFrameDataStore):
    _op_type_ = opcodes.TO_ODPS_TABLE

    dtypes = SeriesField("dtypes")

    table_name = StringField("table_name", default=None)
    partition_spec = StringField("partition_spec", default=None)
    partition_columns = ListField("partition_columns", FieldTypes.string, default=None)
    overwrite = BoolField("overwrite", default=None)
    write_batch_size = Int64Field("write_batch_size", default=None)
    unknown_as_string = BoolField("unknown_as_string", default=None)
    index = BoolField("index", default=True)
    index_label = ListField("index_label", FieldTypes.string, default=None)
    lifecycle = Int64Field("lifecycle", default=None)
    table_properties = DictField("table_properties", default=None)
    primary_key = ListField("primary_key", FieldTypes.string, default=None)
    use_generated_table_meta = BoolField("use_generated_table_meta", default=False)

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.dataframe], **kw)

    def check_inputs(self, inputs: List[TileableType]):
        if self.use_generated_table_meta:
            return None
        return super().check_inputs(inputs)

    def __call__(self, x):
        shape = (0,) * len(x.shape)
        index_value = parse_index(x.index_value.to_pandas()[:0], x.key, "index")
        columns_value = parse_index(
            x.columns_value.to_pandas()[:0], x.key, "columns", store_data=True
        )
        return self.new_dataframe(
            [x],
            shape=shape,
            dtypes=x.dtypes[:0],
            index_value=index_value,
            columns_value=columns_value,
        )

    @classmethod
    def get_index_mapping(
        cls,
        index_label: Optional[List[str]],
        raw_index_levels: List[Any],
    ) -> List[Any]:
        def_labels = index_label or itertools.repeat(None)
        def_labels = itertools.chain(def_labels, itertools.repeat(None))
        names = raw_index_levels
        if len(names) == 1:
            default_labels = ["index"]
        else:
            default_labels = [f"level_{i}" for i in range(len(names))]
        indexes = [
            def_label or name or label
            for def_label, name, label in zip(def_labels, names, default_labels)
        ]
        return [x.lower() for x in indexes]


def to_odps_table(
    df: TileableType,
    table: Union[ODPSTable, str],
    partition: Optional[str] = None,
    partition_col: Union[None, str, List[str]] = None,
    overwrite: bool = False,
    unknown_as_string: Optional[bool] = True,
    index: bool = True,
    index_label: Union[None, str, List[str]] = None,
    lifecycle: Optional[int] = None,
    table_properties: Optional[dict] = None,
    primary_key: Union[None, str, List[str]] = None,
):
    """
    Write DataFrame object into a MaxCompute (ODPS) table.

    You need to provide the name of the table to write to. If you want to store
    data into a specific partitioned of a table, argument `partition` can be used.
    You can also use `partition_col` to specify DataFrame columns as partition
    columns, and data in the DataFrame will be grouped by these columns and
    inserted into partitions the values of these columns.

    If the table does not exist, `to_odps_table` will create one.

    Column names for indexes is determined by `index_label` argument. If the
    argument is absent, names of the levels is used if they are not None, or
    default names will be used. The default name for indexes with only one level
    will be `index`, and for indexes with multiple levels, the name will be
    `level_x` while x is the index of the level.

    Parameters
    ----------
    table: str
        Name ot the table to write DataFrame into
    partition: Optional[str]
        Spec of the partition to write to, can be 'pt1=xxx,pt2=yyy'
    partition_col: Union[None, str, List[str]]
        Name of columns in DataFrame as partition columns.
    overwrite: bool
        Overwrite data if the table / partition already exists.
    unknown_as_string: bool
        If True, object type in the DataFrame will be treated as strings.
        Otherwise errors might be raised.
    index: bool
        If True, indexes will be stored. Otherwise they are ignored.
    index_label: Union[None, str, List[str]]
        Specify column names for index levels. If absent, level names or default
        names will be used.
    lifecycle: Optional[int]
        Specify lifecycle of the output table.
    table_properties: Optional[dict]
        Specify properties of the output table.
    primary_key: Union[None, str, List[str]]
        If provided and target table does not exist, target table
        will be a delta table with columns specified in this argument
        as primary key.

    Returns
    -------
    result: DataFrame
        Stub DataFrame for execution.

    Notes
    -----
    `to_odps_table` returns a stub object for execution. The result returned is
    not reusable.

    Examples
    --------

    """
    odps_entry = ODPS.from_global() or ODPS.from_environments()
    if isinstance(table, ODPSTable):
        table = table.full_table_name
    elif options.session.enable_schema and "." not in table:
        default_schema = (
            options.session.default_schema or odps_entry.schema or "default"
        )
        table = default_schema + "." + table

    if isinstance(index_label, str):
        index_label = [index_label]
    if isinstance(partition_col, str):
        partition_col = [partition_col]

    if index_label and len(index_label) != len(df.index.names):
        raise ValueError(
            f"index_label needs {len(df.index.nlevels)} labels "
            f"but it only have {len(index_label)}"
        )

    # check if table partition columns conflicts with dataframe columns
    table_cols = set(build_dataframe_table_meta(df).table_column_names)
    partition_col_set = (
        set(x.lower() for x in PartitionSpec(partition).keys()) if partition else set()
    )
    if partition:
        partition_intersect = partition_col_set & table_cols
        if partition_intersect:
            raise ValueError(
                f"Data column(s) {partition_intersect} in the dataframe"
                " cannot be used in parameter 'partition'."
                " Use 'partition_col' instead."
            )

    if index:
        index_cols = set(
            DataFrameToODPSTable.get_index_mapping(index_label, df.index.names)
        )
        index_table_intersect = index_cols & table_cols
        if index_table_intersect:
            raise ValueError(
                f"Index column(s) {index_table_intersect} conflict with "
                f"column(s) of the input dataframe."
            )
        index_partition_intersect = index_cols & partition_col_set
        if index_partition_intersect:
            raise ValueError(
                f"Index column(s) {index_partition_intersect} conflict "
                f"with partition column(s)."
            )

    if partition_col:
        partition_diff = set(x.lower() for x in partition_col) - table_cols
        if partition_diff:
            raise ValueError(
                f"Partition column(s) {partition_diff}"
                " is not the data column(s) of the input dataframe."
            )

    table_properties = table_properties or {}
    if primary_key is not None:
        table_properties["transactional"] = "true"
    if odps_entry.exist_table(table):
        table_obj = odps_entry.get_table(table)
        if table_obj.is_transactional:
            table_properties = table_properties or {}
            table_properties["transactional"] = "true"
            primary_key = primary_key or table_obj.primary_key or ()
            if set(primary_key) != set(table_obj.primary_key or ()):
                raise ValueError(
                    f"Primary keys between existing table {table} and "
                    f"provided arguments are not same."
                )
    if primary_key and not isinstance(primary_key, (list, tuple)):
        primary_key = [primary_key]

    op = DataFrameToODPSTable(
        dtypes=df.dtypes,
        table_name=table,
        unknown_as_string=unknown_as_string,
        partition_spec=partition,
        partition_columns=partition_col,
        overwrite=overwrite,
        index=index,
        index_label=index_label,
        lifecycle=lifecycle or options.session.table_lifecycle,
        table_properties=table_properties or None,
        primary_key=primary_key or None,
    )
    return op(df)
