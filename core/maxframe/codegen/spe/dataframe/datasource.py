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

from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from odps import ODPS

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.config import options
from maxframe.dataframe.datasource.dataframe import DataFrameDataSource
from maxframe.dataframe.datasource.date_range import DataFrameDateRange
from maxframe.dataframe.datasource.from_records import DataFrameFromRecords
from maxframe.dataframe.datasource.from_tensor import (
    DataFrameFromTensor,
    SeriesFromTensor,
)
from maxframe.dataframe.datasource.index import IndexDataSource
from maxframe.dataframe.datasource.read_odps_query import DataFrameReadODPSQuery
from maxframe.dataframe.datasource.read_odps_table import DataFrameReadODPSTable
from maxframe.io.odpsio import ODPSTableIO, arrow_to_pandas
from maxframe.protocol import DataFrameTableMeta


@register_op_adapter(DataFrameDataSource)
class DataFrameDataSourceAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameDataSource, context: SPECodeContext
    ) -> List[str]:
        data_var = context.register_operator_constants(op.data)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {data_var}"]


@register_op_adapter(DataFrameDateRange)
class DataFrameDateRangeAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameDateRange, context: SPECodeContext
    ) -> List[str]:
        context.register_import("pandas", "pd")

        kw_keys = [
            "start",
            "end",
            "periods",
            "freq",
            "tz",
            "normalize",
            "name",
            "inclusive",
        ]
        args = self.generate_call_args_with_attributes(
            op, context, kw_keys=kw_keys, skip_none=True
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = pd.date_range({args})"]


@register_op_adapter(DataFrameFromRecords)
class DataFrameFromRecordsAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameFromRecords, context: SPECodeContext
    ) -> List[str]:
        context.register_import("pandas", "pd")

        args = self.generate_call_args_with_attributes(
            op,
            context,
            "input",
            kw_keys=["columns", "exclude", "coerce_float", "nrows"],
            skip_none=True,
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = pd.DataFrame.from_records({args})"]


@register_op_adapter(DataFrameFromTensor)
class DataFrameFromTensorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameFromTensor, context: SPECodeContext
    ) -> List[str]:
        context.register_import("pandas", "pd")

        args = self.generate_call_args_with_attributes(
            op, context, "input", kw_keys=["index", "columns"], skip_none=True
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = pd.DataFrame({args})"]


@register_op_adapter(DataFrameReadODPSTable)
class DataFrameReadODPSTableAdapter(SPEOperatorAdapter):
    @staticmethod
    def _read_as_pandas(
        table_name: str,
        partitions: Optional[List[str]],
        columns: Optional[List[str]],
        index_columns: Optional[List[str]],
        append_partitions: bool,
        table_meta: DataFrameTableMeta,
    ):
        o = ODPS.from_environments() or ODPS.from_global()

        if columns is not None:
            all_cols = (index_columns or []) + (columns or [])
        else:
            all_cols = None

        with ODPSTableIO(o).open_reader(
            table_name, partitions, all_cols, partition_columns=append_partitions
        ) as reader:
            arrow_result = reader.read_all()
        if not index_columns:
            # need prepending an index column
            arrow_result = arrow_result.add_column(
                0,
                table_meta.table_index_column_names[0],
                pa.array(np.arange(len(arrow_result))),
            )
        else:
            name_mapping = dict(zip(index_columns, table_meta.table_index_column_names))
            new_names = [name_mapping.get(c, c) for c in arrow_result.column_names]
            arrow_result = arrow_result.rename_columns(new_names)

        res = arrow_to_pandas(arrow_result, table_meta)
        if not index_columns:
            res.index = pd.RangeIndex(len(arrow_result))
        return res

    def generate_code(
        self, op: DataFrameReadODPSTable, context: SPECodeContext
    ) -> List[str]:
        table_meta = context.get_pandas_data_table_meta(op.outputs[0])

        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args = self._translate_call_args(
            context,
            op.table_name,
            op.partitions,
            op.columns,
            op.index_columns,
            op.append_partitions,
            table_meta,
        )
        return [f"{res_var_name} = {cls_name}._read_as_pandas({', '.join(args)})"]


@register_op_adapter(DataFrameReadODPSQuery)
class DataFrameReadODPSQueryAdapter(SPEOperatorAdapter):
    @staticmethod
    def _build_hints(odps_entry: ODPS):
        hints = options.sql.settings.copy() or {}
        if options.session.enable_schema or odps_entry.is_schema_namespace_enabled():
            hints["odps.namespace.schema"] = "true"
            hints["odps.sql.allow.namespace.schema"] = "true"
        hints["odps.sql.submit.mode"] = "script"
        hints["odps.sql.object.table.split.by.object.size.enabled"] = "false"
        return hints

    @staticmethod
    def _read_as_pandas(
        query: str,
        columns: Optional[List[str]],
        index_columns: Optional[List[str]],
        column_renames: Optional[dict],
        nrows: Optional[int],
        table_meta: DataFrameTableMeta,
    ):
        odps_entry = ODPS.from_environments() or ODPS.from_global()
        hints = DataFrameReadODPSQueryAdapter._build_hints(odps_entry)
        instance = odps_entry.execute_sql(query, hints=hints)

        with instance.open_reader(tunnel=True) as reader:
            result = reader.to_pandas()

        if nrows is not None:
            result = result.iloc[:nrows]

        column_renames = column_renames or {}
        if column_renames:
            result = result.rename(columns=column_renames)

        selected_columns = []
        if index_columns:
            selected_columns.extend(index_columns)
        if columns is None:
            selected_columns.extend(table_meta.pd_column_names)
        else:
            selected_columns.extend(columns)
        result = result[selected_columns]

        if not index_columns:
            result.insert(
                0,
                table_meta.table_index_column_names[0],
                np.arange(len(result)),
            )
        else:
            name_mapping = dict(zip(index_columns, table_meta.table_index_column_names))
            result = result.rename(columns=name_mapping)

        result = result.rename(
            columns=dict(zip(table_meta.pd_column_names, table_meta.table_column_names))
        )
        res = arrow_to_pandas(
            pa.Table.from_pandas(result, preserve_index=False), table_meta
        )
        if not index_columns:
            res.index = pd.RangeIndex(len(result))
        return res

    def generate_code(
        self, op: DataFrameReadODPSQuery, context: SPECodeContext
    ) -> List[str]:
        table_meta = context.get_pandas_data_table_meta(op.outputs[0])

        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args = self._translate_call_args(
            context,
            op.query,
            op.columns,
            op.index_columns,
            op.column_renames,
            op.nrows,
            table_meta,
        )
        return [f"{res_var_name} = {cls_name}._read_as_pandas({', '.join(args)})"]


@register_op_adapter(SeriesFromTensor)
class SeriesFromTensorAdapter(SPEOperatorAdapter):
    def generate_code(self, op: SeriesFromTensor, context: SPECodeContext) -> List[str]:
        context.register_import("pandas", "pd")
        tensor_var_name = context.get_input_tileable_variable(op.input)
        call_args = self._translate_call_args(
            context,
            index=op.index,
            name=op.outputs[0].name,
            dtype=op.outputs[0].dtype,
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [
            f"{res_var_name} = pd.Series({tensor_var_name}, {', '.join(call_args)})"
        ]


@register_op_adapter(IndexDataSource)
class IndexDataSourceAdapter(SPEOperatorAdapter):
    def generate_code(self, op: IndexDataSource, context: SPECodeContext) -> List[str]:
        context.register_import("pandas")
        data_var = self.translate_var(context, op.data)
        res_var = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var} = {data_var}"]
