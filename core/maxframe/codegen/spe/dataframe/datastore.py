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

from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
from odps import ODPS
from odps.types import Column, OdpsSchema, PartitionSpec

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.config import options
from maxframe.dataframe.datastore.to_lance import DataFrameToLance
from maxframe.dataframe.datastore.to_odps import DataFrameToODPSTable
from maxframe.io.odpsio import ODPSTableIO, pandas_to_arrow, pandas_to_odps_schema
from maxframe.protocol import DataFrameTableMeta, ResultInfo
from maxframe.typing_ import PandasObjectTypes

try:
    import lance
except (ImportError, AttributeError):  # pragma: no cover
    # lance not installed or not compatible
    lance = None


@register_op_adapter(DataFrameToODPSTable)
class DataFrameToODPSTableAdapter(SPEOperatorAdapter):
    @classmethod
    def _get_index_mapping(
        cls,
        index: bool,
        index_label: Optional[List[str]],
        input_meta: DataFrameTableMeta,
    ) -> Dict[str, str]:
        if not index:
            return {}
        labels = DataFrameToODPSTable.get_index_mapping(
            index_label, input_meta.pd_index_level_names
        )
        return {
            src_col: idx_col
            for src_col, idx_col in zip(input_meta.table_index_column_names, labels)
        }

    @classmethod
    def _get_result_schema(
        cls,
        partition_spec: Optional[str],
        partition_columns: Optional[List[str]],
        src_schema: OdpsSchema,
        exclude_cols: List[str],
        index_mappings: Dict[str, str],
    ) -> OdpsSchema:
        part_spec = (
            PartitionSpec(partition_spec) if partition_spec is not None else None
        )
        exclude_col_set = set(exclude_cols or ())

        parts = []
        if part_spec is not None:
            parts.extend(Column(k, "string") for k in part_spec.keys())
        if partition_columns is not None:
            part_column_set = set(partition_columns)
            cols = [
                col
                for col in src_schema.columns
                if col.name not in part_column_set and col.name not in exclude_col_set
            ]
            parts.extend(
                col for col in src_schema.columns if col.name in part_column_set
            )
        else:
            cols = [
                col for col in src_schema.columns if col.name not in exclude_col_set
            ]
        odps_schema = OdpsSchema(cols, parts)

        for col in odps_schema.columns:
            col.name = index_mappings.get(col.name, col.name)
        return odps_schema

    @classmethod
    def _check_table_schema(cls, table_schema: OdpsSchema, data_schema: OdpsSchema):
        table_cols = [col.name for col in table_schema.simple_columns]
        data_cols = [col.name for col in data_schema.simple_columns]
        if set(table_cols) != set(data_cols):
            raise ValueError(
                f"Source and dest columns does not match: "
                f"{sorted(data_cols)} does not equal to {sorted(table_cols)}"
            )
        table_part_cols = [col.name for col in table_schema.partitions]
        data_part_cols = [col.name for col in data_schema.partitions]
        if set(table_part_cols) != set(data_part_cols):
            raise ValueError(
                f"Source and dest partitions does not match: "
                f"{sorted(data_part_cols)} does not equal to {sorted(table_part_cols)}"
            )

    @classmethod
    def _write_as_table(
        cls,
        df_obj: PandasObjectTypes,
        table_name: str,
        partition_spec: Optional[str] = None,
        partition_columns: Optional[List[str]] = None,
        index: bool = True,
        index_label: Optional[List[str]] = None,
        overwrite: bool = True,
        lifecycle: Optional[int] = None,
        table_properties: Optional[Dict] = None,
    ):
        partition_spec = (
            PartitionSpec(partition_spec) if partition_spec is not None else None
        )
        odps_schema, odps_meta = pandas_to_odps_schema(df_obj, unknown_as_string=True)
        index_mapping = cls._get_index_mapping(index, index_label, odps_meta)
        exclude_cols = [] if index else odps_meta.table_index_column_names
        result_schema = cls._get_result_schema(
            partition_spec, partition_columns, odps_schema, exclude_cols, index_mapping
        )

        odps_entry = ODPS.from_environments() or ODPS.from_global()
        if not odps_entry.exist_table(table_name):
            odps_entry.create_table(
                table_name,
                result_schema,
                lifecycle=lifecycle,
                table_properties=table_properties,
            )
            part_cols = [col.name for col in result_schema.partitions]
        else:
            table = odps_entry.get_table(table_name)
            cls._check_table_schema(table.table_schema, result_schema)
            result_schema = table.table_schema
            part_cols = [col.name for col in table.table_schema.partitions]

        part_and_group = []
        if partition_columns is not None:
            groupby_obj = df_obj.groupby(partition_columns)
            for group in groupby_obj.groups:
                group = (group,) if len(partition_columns) == 1 else group
                try:
                    group_data = groupby_obj.get_group(group)
                except KeyError:
                    if len(group) != 1:
                        raise
                    group_data = groupby_obj.get_group(group[0])

                group_data = group_data.loc[
                    :, ~group_data.columns.isin(partition_columns)
                ]

                part_dict = dict(zip(partition_columns, group))

                part_dict.update(dict(partition_spec.items() if partition_spec else {}))
                part_spec = ",".join(f"{col}={part_dict[col]}" for col in part_cols)
                part_and_group.append((part_spec, group_data))
        else:
            part_and_group.append((partition_spec, df_obj))

        for part, data in part_and_group:
            if part is not None:
                odps_entry.get_table(table_name).create_partition(
                    part, if_not_exists=True
                )

            with ODPSTableIO(odps_entry).open_writer(
                table_name, part, overwrite=overwrite
            ) as writer:
                arrow_val, _ = pandas_to_arrow(data)
                data_array_dict = dict(zip(arrow_val.column_names, arrow_val.columns))
                cols, arrays = [], []
                col_name_to_src_name = {v: k for k, v in index_mapping.items()}
                for col in result_schema.simple_columns:
                    cols.append(col.name)
                    arrays.append(
                        data_array_dict[col_name_to_src_name.get(col.name, col.name)]
                    )
                writer.write(pa.Table.from_arrays(arrays, cols))

    def generate_code(
        self, op: DataFrameToODPSTable, context: SPECodeContext
    ) -> List[str]:
        # to_odps_table should return None
        context.put_tileable_result_info(op.outputs[0], ResultInfo())

        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        context.register_import(__name__, from_item=type(self).__name__)

        args = self._translate_call_args(
            context,
            op.table_name,
            op.partition_spec,
            op.partition_columns,
            op.index,
            op.index_label,
            overwrite=op.overwrite,
            lifecycle=op.lifecycle or options.session.table_lifecycle,
            table_properties=op.table_properties,
        )
        return [
            f"{type(self).__name__}._write_as_table("
            f"{input_var_name}, {', '.join(args)})",
            f"{res_var_name} = None",
        ]


@register_op_adapter(DataFrameToLance)
class DataFrameToLanceAdapter(SPEOperatorAdapter):
    @classmethod
    def _write_as_table(
        cls,
        df_obj,
        path,
        mode="create",
        index=None,
        index_label=None,
        schema=None,
        lance_kwargs=None,
    ):
        if lance is None:
            raise ImportError(
                "Please install pylance which is required to write Lance datasets"
            )

        # Apply index_label renaming
        if index is not False and index_label is not None:
            if isinstance(df_obj.index, pd.MultiIndex):
                if len(index_label) == len(df_obj.index.names):
                    df_obj.index.names = index_label
            else:
                label = index_label[0] if isinstance(index_label, list) else index_label
                df_obj.index.name = label

        preserve_index = index is not False
        if preserve_index and isinstance(df_obj.index, pd.RangeIndex):
            df_obj.index = pd.Index(df_obj.index.to_numpy(), name=df_obj.index.name)

        table = pa.Table.from_pandas(df_obj, preserve_index=preserve_index)

        if schema is not None:
            user_schema = pa.ipc.read_schema(pa.BufferReader(schema))
            merged_fields = []
            user_fields = {f.name: f for f in user_schema}
            for field in table.schema:
                if field.name in user_fields:
                    merged_fields.append(user_fields[field.name])
                else:
                    merged_fields.append(field)
            table = table.cast(pa.schema(merged_fields, metadata=table.schema.metadata))

        lance.write_dataset(
            table,
            path,
            schema=table.schema,
            mode=mode or "create",
            **(lance_kwargs or {}),
        )

        return pd.DataFrame()

    def generate_code(self, op: DataFrameToLance, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)

        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        args = self._translate_call_args(
            context,
            op.path,
            mode=op.mode,
            index=op.index,
            index_label=op.index_label,
            schema=op.schema,
            lance_kwargs=op.lance_kwargs,
        )
        return [
            f"{res_var_name} = {cls_name}._write_as_table("
            f"{input_var_name}, {', '.join(args)})"
        ]
