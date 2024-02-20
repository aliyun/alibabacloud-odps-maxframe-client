#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from odps.models import Table as ODPSTable

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...serialization.serializables import (
    BoolField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from ...typing_ import TileableType
from ..core import DataFrame  # noqa: F401
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index

logger = logging.getLogger(__name__)


class DataFrameToODPSTable(DataFrameOperator, DataFrameOperatorMixin):
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

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.dataframe], **kw)

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


def to_odps_table(
    df: TileableType,
    table: Union[ODPSTable, str],
    partition: Optional[str] = None,
    partition_col: Union[None, str, List[str]] = None,
    overwrite: bool = False,
    unknown_as_string: Optional[bool] = None,
    index: bool = True,
    index_label: Union[None, str, List[str]] = None,
    lifecycle: Optional[int] = None,
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
    if isinstance(table, ODPSTable):
        table = table.full_table_name

    if isinstance(index_label, str):
        index_label = [index_label]
    if isinstance(partition_col, str):
        partition_col = [partition_col]

    if index_label and len(index_label) != len(df.index.names):
        raise ValueError(
            f"index_label needs {len(df.index.nlevels)} labels "
            f"but it only have {len(index_label)}"
        )

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
    )
    return op(df)
