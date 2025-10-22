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

from typing import Any, List

from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE, TILEABLE_TYPE
from ....dataframe.misc.apply import DataFrameApply
from ....dataframe.misc.astype import DataFrameAstype
from ....dataframe.misc.case_when import DataFrameCaseWhen
from ....dataframe.misc.check_monotonic import DataFrameCheckMonotonic
from ....dataframe.misc.clip import DataFrameClip
from ....dataframe.misc.cut import DataFrameCut
from ....dataframe.misc.describe import DataFrameDescribe
from ....dataframe.misc.diff import DataFrameDiff
from ....dataframe.misc.drop import DataFrameDrop
from ....dataframe.misc.drop_duplicates import DataFrameDropDuplicates
from ....dataframe.misc.duplicated import DataFrameDuplicated
from ....dataframe.misc.eval import DataFrameEval
from ....dataframe.misc.explode import DataFrameExplode
from ....dataframe.misc.get_dummies import DataFrameGetDummies
from ....dataframe.misc.isin import DataFrameIsin
from ....dataframe.misc.map import DataFrameMap
from ....dataframe.misc.memory_usage import DataFrameMemoryUsage
from ....dataframe.misc.repeat import DataFrameRepeat
from ....dataframe.misc.shift import DataFrameShift
from ....dataframe.misc.to_numeric import DataFrameToNumeric
from ....dataframe.misc.transform import DataFrameTransform
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

DataFrameAstypeAdapter = build_method_call_adapter(
    DataFrameAstype, "astype", "dtype_values"
)


@register_op_adapter(DataFrameCheckMonotonic)
class DataFrameCheckMonotonicAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameCheckMonotonic, context: SPECodeContext
    ) -> List[str]:
        inst_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        if op.decreasing:
            prop = "is_monotonic_decreasing"
        else:
            prop = "is_monotonic_increasing"
        return [f"{res_var_name} = {inst_var_name}.{prop}"]


DataFrameClipAdapter = build_method_call_adapter(
    DataFrameClip, "clip", skip_none=True, kw_keys=["lower", "upper", "axis"]
)


@register_op_adapter(DataFrameCut)
class DataFrameCutAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameCut, context: SPECodeContext) -> List[str]:
        """ """
        context.register_import("pandas", "pd")

        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_keys = [
            "skip_none",
            "right",
            "labels",
            "retbins",
            "precision",
            "include_lowest",
            "duplicates",
            "ordered",
        ]
        args = self.generate_call_args_with_attributes(
            op, context, "_input", "bins", kw_keys=kw_keys
        )
        return [f"{res_var_name} = pd.cut({args})"]


DataFrameDescribeAdapter = build_method_call_adapter(
    DataFrameDescribe, "describe", kw_keys=["percentiles", "include", "exclude"]
)
DataFrameDiffAdapter = build_method_call_adapter(
    DataFrameDiff, "diff", kw_keys=["periods", "axis"]
)
DataFrameDuplicatedAdapter = build_method_call_adapter(
    DataFrameDuplicated, "duplicated", kw_keys=["subset", "keep"]
)


@register_op_adapter(DataFrameEval)
class DataFrameEvalAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameEval, context: SPECodeContext) -> List[str]:
        func_name = "query" if op.is_query else "eval"
        inst_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args_list = self._translate_call_args(context, op.expr)
        args = ", ".join(args_list)
        return [f"{res_var_name} = {inst_var_name}.{func_name}({args})"]


DataFrameExplodeAdapter = build_method_call_adapter(
    DataFrameExplode, "explode", "column", ignore_index=None
)
DataFrameMemoryUsageAdapter = build_method_call_adapter(
    DataFrameMemoryUsage, "memory_usage", kw_keys=["index", "deep"]
)


@register_op_adapter(DataFrameIsin)
class DataFrameIsinAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameIsin, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.input)
        filter_var = self.translate_var(context, op.values)
        output_var = context.get_output_tileable_variable(op.outputs[0])
        return [f"{output_var} = {input_var}.isin({filter_var})"]


DataFrameMapAdapter = build_method_call_adapter(
    DataFrameMap, "map", "arg", na_action=None
)
DataFrameRepeatAdapter = build_method_call_adapter(DataFrameRepeat, "repeat", "repeats")
DataFrameShiftAdapter = build_method_call_adapter(
    DataFrameShift, "shift", kw_keys=["periods", "freq", "axis", "fill_value"]
)


@register_op_adapter(DataFrameToNumeric)
class DataFrameToNumericAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameToNumeric, context: SPECodeContext
    ) -> List[str]:
        context.register_import("pandas", "pd")

        kw_keys = ["errors", "downcast"]
        args = self.generate_call_args_with_attributes(
            op, context, "input", kw_keys=kw_keys
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = pd.to_numeric({args})"]


@register_op_adapter(DataFrameTransform)
class TransformOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameTransform, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        funcs_var = self.translate_var(context, op.func)
        return [f"{output_var} = {input_var}.transform({funcs_var}, axis={op.axis})"]


@register_op_adapter(DataFrameDrop)
class DataFrameDropAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameDrop, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        args_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["index", "columns", "level", "errors"]
        )
        return [f"{output_var} = {input_var}.drop({args_str})"]


@register_op_adapter(DataFrameApply)
class DataFrameApplyAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameApply, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        applied_var = f"{output_var}_applied"
        renamed_var = f"{output_var}_renamed"
        if isinstance(op.inputs[0], DATAFRAME_TYPE):
            apply_arg_list = ("func", "axis", "raw", "result_type")
        else:
            apply_arg_list = ("func", "convert_dtype")
        apply_args = self.generate_call_args_with_attributes(
            op, context, kw_keys=apply_arg_list
        )

        apply_code = f"{applied_var} = {input_var}.apply({apply_args})"

        if isinstance(op.outputs[0], SERIES_TYPE):
            rename_arg = self.translate_var(context, op.outputs[0].name)
            rename_code = f"{renamed_var} = {applied_var}.rename({rename_arg})"
            astype_arg = self.translate_var(context, op.outputs[0].dtype)
            astype_node = f"{output_var} = {renamed_var}.astype({astype_arg})"
        else:
            rename_arg = self.translate_var(context, list(op.outputs[0].dtypes.index))
            rename_code = f"{applied_var}.columns = {rename_arg}"
            astype_arg = self.translate_var(context, dict(op.outputs[0].dtypes))
            astype_node = f"{output_var} = {applied_var}.astype({astype_arg})"

        return [apply_code, rename_code, astype_node]


@register_op_adapter(DataFrameDropDuplicates)
class DropDuplicatesOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameDropDuplicates, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        if isinstance(op.inputs[0], DATAFRAME_TYPE):
            arg_list = ("subset", "keep", "ignore_index")
        elif isinstance(op.inputs[0], SERIES_TYPE):
            arg_list = ("keep", "ignore_index")
        else:
            arg_list = ("keep",)
        args = self.generate_call_args_with_attributes(op, context, kw_keys=arg_list)
        return [f"{output_var} = {input_var}.drop_duplicates({args})"]


@register_op_adapter(DataFrameCaseWhen)
class CaseWhenOperatorAdapter(SPEOperatorAdapter):
    @classmethod
    def _get_case_expression(cls, expr: Any, context: SPECodeContext) -> str:
        if isinstance(expr, TILEABLE_TYPE):
            return context.get_input_tileable_variable(expr)
        return cls.translate_var(context, expr)

    def generate_code(
        self, op: DataFrameCaseWhen, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])

        cond_exprs = [
            self._get_case_expression(cond, context) for cond in op.conditions
        ]
        repl_exprs = [
            self._get_case_expression(repl, context) for repl in op.replacements
        ]
        case_list_items = [
            f"({cond_expr}, {repr_expr})"
            for cond_expr, repr_expr in zip(cond_exprs, repl_exprs)
        ]
        case_list = ", ".join(case_list_items)
        return [f"{output_var} = {input_var}.case_when(caselist=[{case_list}])"]


DataFrameGetDummiesAdapter = build_method_call_adapter(
    DataFrameGetDummies,
    "get_dummies",
    "data",
    kw_keys=[
        "prefix",
        "prefix_sep",
        "dummy_na",
        "columns",
        "sparse",
        "drop_first",
        "dtype",
    ],
    source_module="pd",
)
