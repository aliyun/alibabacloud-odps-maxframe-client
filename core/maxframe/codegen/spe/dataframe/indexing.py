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

from typing import List, Union

from ....dataframe.core import DATAFRAME_TYPE, INDEX_TYPE
from ....dataframe.indexing.align import DataFrameAlign
from ....dataframe.indexing.droplevel import DataFrameDropLevel
from ....dataframe.indexing.filter import DataFrameFilter
from ....dataframe.indexing.get_level_values import IndexGetLevelValues
from ....dataframe.indexing.getitem import DataFrameIndex, SeriesIndex
from ....dataframe.indexing.iloc import (
    DataFrameIlocGetItem,
    DataFrameIlocSetItem,
    IndexIlocGetItem,
    SeriesIlocGetItem,
    SeriesIlocSetItem,
)
from ....dataframe.indexing.insert import DataFrameInsert
from ....dataframe.indexing.loc import DataFrameLocGetItem, DataFrameLocSetItem
from ....dataframe.indexing.reindex import DataFrameReindex
from ....dataframe.indexing.rename import DataFrameRename
from ....dataframe.indexing.rename_axis import DataFrameRenameAxis
from ....dataframe.indexing.reorder_levels import DataFrameReorderLevels
from ....dataframe.indexing.reset_index import DataFrameResetIndex
from ....dataframe.indexing.sample import DataFrameSample
from ....dataframe.indexing.set_axis import DataFrameSetAxis
from ....dataframe.indexing.set_index import DataFrameSetIndex
from ....dataframe.indexing.setitem import DataFrameSetitem
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter


@register_op_adapter(DataFrameAlign)
class DataFrameAlignAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameAlign, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        out_lhs_var = context.get_output_tileable_variable(op.outputs[0])
        out_rhs_var = context.get_output_tileable_variable(op.outputs[1])
        kw_keys = [
            "join",
            "axis",
            "level",
            "fill_value",
            "method",
            "limit",
            "fill_axis",
            "broadcast_axis",
        ]
        args_str = self.generate_call_args_with_attributes(
            op, context, "rhs", skip_none=True, kw_keys=kw_keys
        )
        return [f"{out_lhs_var}, {out_rhs_var} = {input_var}.align({args_str})"]


DataFrameDropLevelAdapter = build_method_call_adapter(
    DataFrameDropLevel, "droplevel", skip_none=True, kw_keys=["level", "axis"]
)
IndexGetLevelValuesAdapter = build_method_call_adapter(
    IndexGetLevelValues, "get_level_values", "level"
)
DataFrameInsertAdapter = build_method_call_adapter(
    DataFrameInsert, "insert", kw_keys=["loc", "column", "value"], allow_duplicates=None
)


@register_op_adapter(SeriesIndex)
class SeriesIndexAdapter(SPEOperatorAdapter):
    def generate_code(self, op: SeriesIndex, context: SPECodeContext) -> List[str]:
        args = self._translate_call_args(context, op.labels)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        return [f"{output_var} = {input_var}[{', '.join(args)}]"]


@register_op_adapter(DataFrameIndex)
class DataFrameIndexAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameIndex, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        if op.col_names is not None:
            args = ", ".join(self._translate_call_args(context, op.col_names))
        else:
            args = context.get_input_tileable_variable(op.mask)
        output_var = context.get_output_tileable_variable(op.outputs[0])
        return [f"{output_var} = {input_var}[{args}]"]


@register_op_adapter(DataFrameRename)
class DataFrameRenameAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameRename, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])

        if isinstance(op.outputs[0], INDEX_TYPE):
            func_name = "set_names"
            arg_list = [self.translate_var(context, op.index_mapper)]
            arg_dict = {"level": self.translate_var(context, op.level)}
        else:
            func_name = "rename"
            new_name_var = self.translate_var(context, op.new_name)
            arg_list = []
            arg_dict = {
                "columns": self.translate_var(context, op.columns_mapper),
                "index": self.translate_var(context, op.index_mapper),
                "level": self.translate_var(context, op.level),
                "errors": self.translate_var(context, op.errors),
            }
            if (
                not isinstance(op.outputs[0], DATAFRAME_TYPE)
                and op.new_name is not None
            ):
                arg_dict["index"] = new_name_var

        arg_list += [f"{k}={v}" for k, v in arg_dict.items() if v != "None"]
        arg_str = ", ".join(arg_list)
        return [f"{output_var} = {input_var}.{func_name}({arg_str})"]


@register_op_adapter(DataFrameSetAxis)
class DataFrameSetAxisAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameSetAxis, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.input)
        value_var = self.translate_var(context, op.value)
        axis = self.translate_var(context, op.axis)
        output_var = context.get_output_tileable_variable(op.outputs[0])
        return [
            f"{output_var} = {input_var}.set_axis({value_var}, axis={axis})",
        ]


@register_op_adapter(DataFrameSetitem)
class DataFrameSetitemAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameSetitem, context: SPECodeContext) -> List[str]:
        target_var = context.get_input_tileable_variable(op.target)
        source_var = self.translate_var(context, op.value)
        output_var = context.get_output_tileable_variable(op.outputs[0])
        indexes = self.translate_var(context, op.indexes)
        return [
            f"{target_var}[{indexes}] = {source_var}",
            f"{output_var} = {target_var}",
        ]


@register_op_adapter([DataFrameIlocGetItem, SeriesIlocGetItem, IndexIlocGetItem])
class DataFrameIlocGetItemAdapter(SPEOperatorAdapter):
    def generate_code(
        self,
        op: Union[DataFrameIlocGetItem, SeriesIlocGetItem, IndexIlocGetItem],
        context: SPECodeContext,
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        row_exp = self.translate_var(context, op.indexes[0])
        col_exp = ""
        if len(op.indexes) > 1:
            col_exp = f", {self.translate_var(context, op.indexes[1])}"
        return [f"{output_var} = {input_var}.iloc[{row_exp}{col_exp}]"]


@register_op_adapter([DataFrameIlocSetItem, SeriesIlocSetItem])
class DataFrameIlocSetItemAdapter(SPEOperatorAdapter):
    def generate_code(
        self,
        op: Union[DataFrameIlocSetItem, SeriesIlocSetItem],
        context: SPECodeContext,
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        value_exp = self.translate_var(context, op.value)
        row_exp = self.translate_var(context, op.indexes[0])
        col_exp = ""
        if len(op.indexes) > 1:
            col_exp = f", {self.translate_var(context, op.indexes[1])}"
        return [
            f"{output_var} = {input_var}.copy()",
            f"{output_var}.iloc[{row_exp}{col_exp}] = {value_exp}",
        ]


@register_op_adapter(DataFrameLocGetItem)
class DataFrameLocGetItemAdapter(SPEOperatorAdapter):
    def generate_code(
        self,
        op: DataFrameLocGetItem,
        context: SPECodeContext,
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        row_exp = self.translate_var(context, op.indexes[0])
        col_exp = ""
        if len(op.indexes) > 1:
            col_exp = f", {self.translate_var(context, op.indexes[1])}"
        return [f"{output_var} = {input_var}.loc[{row_exp}{col_exp}]"]


@register_op_adapter(DataFrameLocSetItem)
class DataFrameLocSetItemAdapter(SPEOperatorAdapter):
    def generate_code(
        self,
        op: DataFrameLocSetItem,
        context: SPECodeContext,
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        value_exp = self.translate_var(context, op.value)
        row_exp = self.translate_var(context, op.indexes[0])
        col_exp = ""
        if len(op.indexes) > 1:
            col_exp = f", {self.translate_var(context, op.indexes[1])}"
        return [
            f"{output_var} = {input_var}.copy()",
            f"{output_var}.loc[{row_exp}{col_exp}] = {value_exp}",
        ]


@register_op_adapter(DataFrameSample)
class DataFrameSampleAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameSample, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        kwargs = {
            "n": op.size,
            "frac": op.frac,
            "replace": op.replace,
            "weights": op.weights,
            "axis": op.axis,
        }
        codes = list()
        context.register_import("numpy", "np")
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args_str = ", ".join(self._translate_call_args(context, **kwargs))
        if op.random_state is not None:
            rs_var = f"{res_var_name}_rs"
            codes.append(f"{rs_var} = np.random.RandomState()")
            rs_param = self.translate_var(context, op.random_state.get_state())
            codes.append(f"{rs_var}.set_state({rs_param})")
            args_str += f", random_state={rs_var}"
        codes.append(f"{res_var_name} = {input_var_name}.sample({args_str})")
        return codes


@register_op_adapter(DataFrameResetIndex)
class DataFrameResetIndexAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameResetIndex, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])

        kwargs = {"level": op.level, "drop": op.drop, "inplace": False}

        if isinstance(op.inputs[0], DATAFRAME_TYPE):
            kwargs.update(
                {
                    "col_level": op.col_level,
                    "col_fill": op.col_fill,
                }
            )
        else:
            kwargs.update(
                {
                    "name": op.name,
                }
            )

        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args_str = ", ".join(self._translate_call_args(context, **kwargs))

        return [f"{res_var_name} = {input_var_name}.reset_index({args_str})"]


@register_op_adapter(DataFrameSetIndex)
class DataFrameSetIndexAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameSetIndex, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])

        kwargs = {
            "keys": op.keys,
            "drop": op.drop,
            "append": op.append,
            "inplace": False,
            "verify_integrity": op.verify_integrity,
        }

        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args_str = ", ".join(self._translate_call_args(context, **kwargs))

        return [f"{res_var_name} = {input_var_name}.set_index({args_str})"]


DataFrameReindexAdapter = build_method_call_adapter(
    DataFrameReindex,
    "reindex",
    skip_none=True,
    kw_keys=[
        "index",
        "columns",
        "method",
        "level",
        "fill_value",
        "limit",
    ],
)

DataFrameFilterAdapter = build_method_call_adapter(
    DataFrameFilter,
    "filter",
    skip_none=True,
    kw_keys=["items", "like", "regex", "axis"],
)


DataFrameReorderLevelsAdapter = build_method_call_adapter(
    DataFrameReorderLevels, "reorder_levels", kw_keys=["order", "axis"]
)

DataFrameRenameAxisAdapter = build_method_call_adapter(
    DataFrameRenameAxis,
    "rename_axis",
    skip_none=True,
    kw_keys=["mapper", "index", "columns", "axis", "copy", "level", "inplace"],
)
