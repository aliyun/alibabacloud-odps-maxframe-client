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

import string
from collections import defaultdict
from typing import Any, Dict, Tuple

import pandas as pd
import pyarrow as pa
from odps import types as odps_types
from pandas.api import types as pd_types

from ..core import TILEABLE_TYPE, OutputType
from ..protocol import DataFrameTableMeta
from ..tensor.core import TENSOR_TYPE

_TEMP_TABLE_PREFIX = "tmp_mf_"

_arrow_to_odps_types = {
    pa.string(): odps_types.string,
    pa.binary(): odps_types.binary,
    pa.int8(): odps_types.tinyint,
    pa.int16(): odps_types.smallint,
    pa.int32(): odps_types.int_,
    pa.int64(): odps_types.bigint,
    pa.bool_(): odps_types.boolean,
    pa.float32(): odps_types.float_,
    pa.float64(): odps_types.double,
    pa.date32(): odps_types.date,
    pa.timestamp("ms"): odps_types.datetime,
    pa.timestamp("ns"): odps_types.timestamp,
}

_odps_type_to_arrow = {
    odps_types.string: pa.string(),
    odps_types.binary: pa.binary(),
    odps_types.tinyint: pa.int8(),
    odps_types.smallint: pa.int16(),
    odps_types.int_: pa.int32(),
    odps_types.bigint: pa.int64(),
    odps_types.boolean: pa.bool_(),
    odps_types.float_: pa.float32(),
    odps_types.double: pa.float64(),
    odps_types.date: pa.date32(),
    odps_types.datetime: pa.timestamp("ms"),
    odps_types.timestamp: pa.timestamp("ns"),
}


def arrow_type_to_odps_type(
    arrow_type: pa.DataType, col_name: str, unknown_as_string: bool = False
) -> odps_types.DataType:
    if arrow_type in _arrow_to_odps_types:
        return _arrow_to_odps_types[arrow_type]
    elif isinstance(arrow_type, pa.ListType):
        return odps_types.Array(
            arrow_type_to_odps_type(arrow_type.value_type, col_name, unknown_as_string)
        )
    elif isinstance(arrow_type, pa.MapType):
        return odps_types.Map(
            arrow_type_to_odps_type(arrow_type.key_type, col_name, unknown_as_string),
            arrow_type_to_odps_type(arrow_type.item_type, col_name, unknown_as_string),
        )
    elif isinstance(arrow_type, pa.StructType):
        type_dict = {}
        for idx in range(arrow_type.num_fields):
            field = arrow_type[idx]
            type_dict[field.name] = arrow_type_to_odps_type(
                field.type, col_name, unknown_as_string
            )
        return odps_types.Struct(type_dict)
    elif isinstance(arrow_type, pa.Decimal128Type):
        return odps_types.Decimal(arrow_type.precision, arrow_type.scale)

    if unknown_as_string:
        return odps_types.string
    else:
        raise TypeError(
            "Unknown type {}, column name is {},"
            "specify `unknown_as_string=True` "
            "or `as_type` to set column dtype".format(arrow_type, col_name)
        )


def arrow_schema_to_odps_schema(
    arrow_schema: pa.Schema, unknown_as_string: bool = False
) -> odps_types.OdpsSchema:
    odps_cols = []
    for col_name, col_type in zip(arrow_schema.names, arrow_schema.types):
        col_odps_type = arrow_type_to_odps_type(
            col_type, col_name, unknown_as_string=unknown_as_string
        )
        odps_cols.append(odps_types.Column(col_name, col_odps_type))

    return odps_types.OdpsSchema(odps_cols)


def odps_type_to_arrow_type(
    odps_type: odps_types.DataType, col_name: str
) -> pa.DataType:
    if odps_type in _odps_type_to_arrow:
        col_type = _odps_type_to_arrow[odps_type]
    else:
        if isinstance(odps_type, odps_types.Array):
            col_type = pa.list_(odps_type_to_arrow_type(odps_type.value_type, col_name))
        elif isinstance(odps_type, odps_types.Map):
            col_type = pa.map_(
                odps_type_to_arrow_type(odps_type.key_type, col_name),
                odps_type_to_arrow_type(odps_type.value_type, col_name),
            )
        elif isinstance(odps_type, odps_types.Struct):
            fields = [
                (k, odps_type_to_arrow_type(v, col_name))
                for k, v in odps_type.field_types.items()
            ]
            col_type = pa.struct(fields)
        elif isinstance(odps_type, odps_types.Decimal):
            col_type = pa.decimal128(
                odps_type.precision or odps_types.Decimal._max_precision,
                odps_type.scale or odps_types.Decimal._max_scale,
            )
        elif isinstance(odps_type, (odps_types.Varchar, odps_types.Char)):
            col_type = pa.string()
        else:
            raise TypeError(
                "Unsupported type {}, column name is {}".format(odps_type, col_name)
            )
    return col_type


def odps_schema_to_arrow_schema(
    odps_schema: odps_types.OdpsSchema, with_partitions: bool = False
) -> pa.Schema:
    arrow_schema = []
    cols = odps_schema.columns if with_partitions else odps_schema.simple_columns
    for col in cols:
        col_name = col.name
        col_type = odps_type_to_arrow_type(col.type, col_name)
        arrow_schema.append(pa.field(col_name, col_type))

    return pa.schema(arrow_schema)


def odps_schema_to_pandas_dtypes(
    odps_schema: odps_types.OdpsSchema, with_partitions: bool = False
) -> pd.Series:
    arrow_schema = odps_schema_to_arrow_schema(
        odps_schema, with_partitions=with_partitions
    )
    return arrow_schema.empty_table().to_pandas().dtypes


def _is_scalar_object(df_obj: Any) -> bool:
    return (
        isinstance(df_obj, TENSOR_TYPE) and df_obj.shape == ()
    ) or pd_types.is_scalar(df_obj)


def _scalar_as_index(df_obj: Any) -> pd.Index:
    if isinstance(df_obj, TILEABLE_TYPE):
        return pd.Index([], dtype=df_obj.dtype)
    else:
        return pd.Index([df_obj])[:0]


def pandas_to_odps_schema(
    df_obj: Any, unknown_as_string: bool = False
) -> Tuple[odps_types.OdpsSchema, DataFrameTableMeta]:
    from .. import dataframe as md
    from .arrow import pandas_to_arrow

    if _is_scalar_object(df_obj):
        empty_index = None
    elif hasattr(df_obj, "index_value"):
        empty_index = df_obj.index_value.to_pandas()[:0]
    elif not isinstance(df_obj, pd.Index):
        empty_index = df_obj.index[:0]
    else:
        empty_index = df_obj[:0]

    if hasattr(df_obj, "columns_value"):
        empty_columns = df_obj.dtypes.index
    elif hasattr(df_obj, "columns"):
        empty_columns = df_obj.columns
    else:
        empty_columns = None

    if isinstance(df_obj, (md.DataFrame, pd.DataFrame)):
        empty_df_obj = pd.DataFrame(
            [], columns=empty_columns, index=empty_index
        ).astype(df_obj.dtypes)
    elif isinstance(df_obj, (md.Series, pd.Series)):
        empty_df_obj = pd.Series([], name=df_obj.name, index=empty_index).astype(
            df_obj.dtype
        )
    elif isinstance(df_obj, (md.Index, pd.Index)):
        empty_df_obj = empty_index
    else:
        empty_df_obj = df_obj

    arrow_data, table_meta = pandas_to_arrow(empty_df_obj)
    return (
        arrow_schema_to_odps_schema(
            arrow_data.schema, unknown_as_string=unknown_as_string
        ),
        table_meta,
    )


def build_table_column_name(
    col_idx: int, pd_col_name: Any, records: Dict[str, str]
) -> str:
    """
    Convert column name to MaxCompute acceptable names

    Parameters
    ----------
    col_idx:
        index of the column
    pd_col_name:
        column name in pandas
    records:
        record for existing columns

    Returns
    -------
        converted column name
    """

    def _is_col_name_legal(name: str):
        if len(name) < 1 or len(name) > 128:
            return False
        if name[0] not in string.ascii_letters and name[0] != "_":
            return False
        for ch in name:
            if ch not in string.digits and ch not in string.ascii_letters and ch != "_":
                return False
        return True

    try:
        return records[pd_col_name]
    except KeyError:
        pass

    if isinstance(pd_col_name, str):
        col_name = pd_col_name
    elif isinstance(pd_col_name, tuple):
        col_name = "_".join(str(x) for x in pd_col_name)
    else:
        col_name = str(pd_col_name)

    col_name = col_name.lower()
    if not _is_col_name_legal(col_name):
        col_name = f"_column_{col_idx}"

    records[pd_col_name] = col_name
    return col_name


def build_dataframe_table_meta(df_obj: Any) -> DataFrameTableMeta:
    from .. import dataframe as md

    col_to_count = defaultdict(lambda: 0)
    col_to_idx = defaultdict(lambda: 0)
    pd_col_to_col_name = dict()
    if isinstance(df_obj, (md.DataFrame, pd.DataFrame)):
        obj_type = OutputType.dataframe
    elif isinstance(df_obj, (md.Series, pd.Series)):
        obj_type = OutputType.series
    elif isinstance(df_obj, (md.Index, pd.Index)):
        obj_type = OutputType.index
    elif _is_scalar_object(df_obj):
        obj_type = OutputType.scalar
    else:  # pragma: no cover
        raise TypeError(f"Cannot accept type {type(df_obj)}")

    if obj_type == OutputType.scalar:
        pd_dtypes = pd.Series([])
        column_index_names = []
        index_obj = _scalar_as_index(df_obj)
    elif obj_type == OutputType.index:
        pd_dtypes = pd.Series([])
        column_index_names = []
        index_obj = df_obj
    elif obj_type == OutputType.series:
        pd_dtypes = pd.Series([df_obj.dtype], index=[df_obj.name])
        column_index_names = [None]
        index_obj = df_obj.index
    else:
        pd_dtypes = df_obj.dtypes
        column_index_names = list(pd_dtypes.index.names)
        index_obj = df_obj.index

    if isinstance(df_obj, TILEABLE_TYPE):
        table_name = _TEMP_TABLE_PREFIX + str(df_obj.key)
    else:
        table_name = None

    sql_columns = [None] * len(pd_dtypes)
    pd_col_names = pd_dtypes.index
    if obj_type == OutputType.series and df_obj.name is None:
        # use special table column name for series
        pd_col_names = ["_data"]
    for idx, col in enumerate(pd_col_names):
        sql_columns[idx] = col_name = build_table_column_name(
            idx, col, pd_col_to_col_name
        )
        col_to_count[col_name] += 1

    final_sql_columns = []
    for col in sql_columns:
        if col_to_count[col] > 1:
            col_name = f"{col}_{col_to_idx[col]}"
            col_to_idx[col] += 1
            while col_name in col_to_count:
                col_name = f"{col}_{col_to_idx[col]}"
                col_to_idx[col] += 1
            final_sql_columns.append(col_name)
        else:
            final_sql_columns.append(col)

    if hasattr(index_obj, "index_value"):
        pd_index_val = index_obj.index_value.to_pandas()
    else:
        pd_index_val = index_obj

    if hasattr(pd_index_val, "dtypes"):
        index_dtypes = pd.Series(pd_index_val.dtypes.values, index=pd_index_val.names)
    else:
        index_dtypes = pd.Series([pd_index_val.dtype], index=pd_index_val.names)

    return DataFrameTableMeta(
        table_name=table_name,
        type=obj_type,
        table_column_names=final_sql_columns,
        table_index_column_names=[f"_idx_{i}" for i in range(len(index_obj.names))],
        pd_column_dtypes=pd_dtypes,
        pd_column_level_names=column_index_names,
        pd_index_dtypes=index_dtypes,
    )
