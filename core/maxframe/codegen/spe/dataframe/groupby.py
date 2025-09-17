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

from typing import Any, Dict, List

import pandas as pd

from ....dataframe.groupby.aggregation import DataFrameGroupByAgg
from ....dataframe.groupby.apply import GroupByApply
from ....dataframe.groupby.apply_chunk import GroupByApplyChunk
from ....dataframe.groupby.core import DataFrameGroupByOp
from ....dataframe.groupby.cum import GroupByCumcount, GroupByCumReductionOperator
from ....dataframe.groupby.expanding import GroupByExpandingAgg
from ....dataframe.groupby.fill import GroupByFill
from ....dataframe.groupby.getitem import GroupByIndex
from ....dataframe.groupby.head import GroupByHead
from ....dataframe.groupby.rolling import GroupByRollingAgg
from ....dataframe.groupby.sample import GroupBySample
from ....dataframe.groupby.transform import GroupByTransform
from ....dataframe.utils import make_column_list
from ....dataframe.window.rolling import Rolling
from ....lib.version import parse as parse_version
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

_need_enforce_group_keys = parse_version(pd.__version__) < parse_version("1.5.0")


class SPEGroupByOperatorAdapter(SPEOperatorAdapter):
    def build_groupby_call(
        self, context: SPECodeContext, groupby_params: Dict[str, Any]
    ) -> str:
        groupby_params = {k: v for k, v in groupby_params.items() if v is not None}
        selection = groupby_params.pop("selection", None)
        by_args_list = self._translate_call_args(context, **groupby_params)
        groupby_args = ", ".join(by_args_list)
        groupby_call = f"groupby({groupby_args})"
        if selection is not None:
            sel_str = ", ".join(repr(s) for s in selection)
            groupby_call += f"[[{sel_str}]]"
        return groupby_call


@register_op_adapter(DataFrameGroupByAgg)
class DataFrameGroupByAggAdapter(SPEGroupByOperatorAdapter):
    def generate_code(
        self, op: DataFrameGroupByAgg, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        groupby_call = self.build_groupby_call(context, op.groupby_params)
        agg_func_str = self.translate_var(context, op.raw_func)

        return [f"{res_var_name} = {input_var_name}.{groupby_call}.agg({agg_func_str})"]


DataFrameGroupByOpAdapter = build_method_call_adapter(
    DataFrameGroupByOp,
    "groupby",
    kw_keys=["by", "level", "as_index", "sort", "group_keys"],
)


@register_op_adapter(GroupByApply)
class GroupByApplyAdapter(SPEOperatorAdapter):
    def generate_code(self, op: GroupByApply, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        args_list = self._translate_call_args(context, op.func, *op.args, **op.kwds)
        args = ", ".join(args_list)
        return [f"{res_var_name} = {input_var_name}.apply({args})"]


@register_op_adapter(GroupByApplyChunk)
class GroupByApplyChunkAdapter(SPEGroupByOperatorAdapter):
    def generate_code(
        self, op: GroupByApplyChunk, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        func_var = self.translate_var(context, op.func)

        by_cols = make_column_list(op.groupby_params.get("by", []), op.inputs[0].dtypes)

        need_group_keys = False
        if isinstance(op.inputs[0].op, (GroupByIndex, DataFrameGroupByOp)):
            input_obj_code = input_var
        else:
            assert op.groupby_params
            groupby_params = (op.groupby_params or {}).copy()
            if op.groupby_params.get("selection") is None:
                by_col_set = set(by_cols)
                sel = [
                    col for col in op.inputs[0].dtypes.index if col not in by_col_set
                ]

                if _need_enforce_group_keys and groupby_params.get("group_keys"):
                    need_group_keys = True
                    sel.extend(by_cols)

                groupby_params["selection"] = sel
            groupby_call = self.build_groupby_call(context, groupby_params)
            input_obj_code = f"{input_var}.{groupby_call}"

        args_list = self._translate_call_args(context, *op.args, **op.kwargs)
        args = ", ".join(args_list)
        if args:
            args = f", {args}"

        if not op.batch_rows:
            lines = [f"{output_var} = {input_obj_code}.apply({func_var}{args})"]
        else:
            lines = [
                f"def batch_apply_{input_var}(frame, *args, **kw):",
                f"    batches = [{func_var}(frame.iloc[i : i + {op.batch_rows}], *args, **kw) "
                f"    for i in range(0, len(frame), {op.batch_rows})]",
                f"    if len(batches) > 1:",
                f"        return pd.concat(batches)",
                f"    else:",
                f"        return batches[0]",
                f"{output_var} = {input_obj_code}.apply(batch_apply_{input_var}{args})",
            ]
        if need_group_keys:
            by_cols_str = ", ".join(repr(col) for col in by_cols)
            lines.append(
                f"{output_var} = {output_var}.set_index("
                f"[{by_cols_str}, {output_var}.index])",
            )
        return lines


@register_op_adapter(GroupByCumReductionOperator)
class GroupByCumReductionAdapter(SPEOperatorAdapter):
    # legacy adapter kept for compatibility

    def generate_code(
        self, op: GroupByCumReductionOperator, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        if isinstance(op, GroupByCumcount):
            kw_keys = ["ascending"]
        else:
            kw_keys = ["axis"]

        args = self.generate_call_args_with_attributes(
            op, context, kw_keys=kw_keys, skip_none=True
        )
        return [f"{res_var_name} = {input_var_name}.{op._func_name}({args})"]


@register_op_adapter(GroupByFill)
class GroupByFillOperatorAdapter(SPEOperatorAdapter):
    def generate_code(self, op: GroupByFill, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        kw_keys = ["method", "axis", "limit", "downcast"]
        if op._func_name == "fillna":
            kw_args = ["value"]
        else:
            kw_args = []
        args = self.generate_call_args_with_attributes(
            op, context, *kw_args, kw_keys=kw_keys, skip_none=True
        )
        return [f"{res_var_name} = {input_var_name}.{op._func_name}({args})"]


@register_op_adapter(GroupByIndex)
class GroupByIndexAdapter(SPEOperatorAdapter):
    def generate_code(self, op: GroupByIndex, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        selection = self.translate_var(context, op.selection)
        return [f"{res_var_name} = {input_var_name}[{selection}]"]


@register_op_adapter(GroupByHead)
class GroupByHeadAdapter(SPEGroupByOperatorAdapter):
    def generate_code(self, op: GroupByHead, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        groupby_call = self.build_groupby_call(context, op.groupby_params)
        args = self.generate_call_args_with_attributes(op, context, "row_count")
        return [f"{res_var_name} = {input_var_name}.{groupby_call}.head({args})"]


@register_op_adapter(GroupBySample)
class GroupBySampleAdapter(SPEGroupByOperatorAdapter):
    def generate_code(self, op: GroupBySample, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        groupby_call = self.build_groupby_call(context, op.groupby_params)
        args = self.generate_call_args_with_attributes(
            op,
            context,
            size="n",
            kw_keys=["frac", "replace", "weights"],
            skip_none=True,
        )
        return [f"{res_var_name} = {input_var_name}.{groupby_call}.agg({args})"]


@register_op_adapter(GroupByTransform)
class GroupByTransformAdapter(SPEOperatorAdapter):
    def generate_code(self, op: GroupByTransform, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        args_list = self._translate_call_args(context, op.func, *op.args, **op.kwds)
        args = ", ".join(args_list)
        return [f"{res_var_name} = {input_var_name}.transform({args})"]


@register_op_adapter(GroupByExpandingAgg)
class GroupByExpandingAggAdapter(SPEGroupByOperatorAdapter):
    def generate_code(
        self, op: GroupByExpandingAgg, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        by_call = self.build_groupby_call(context, op.groupby_params)
        op_kwargs = (op.kwargs or dict()).copy()
        op_kwargs.pop("cumcount", None)
        func_kw_str = ", ".join(self._translate_call_args(context, **op_kwargs))

        min_periods = op.window_params.get("min_periods", 1)
        shift = op.window_params.get("shift", 0)
        reverse_range = op.window_params.get("reverse_range", False)

        if (
            min_periods == 1
            and shift == 0
            and not reverse_range
            and op.func in ("count", "sum", "prod", "min", "max")
        ):
            # can be simplified as cumxxx function
            cumfunc = "cum" + op.func
            return [
                f"{res_var_name} = {input_var_name}.{by_call}.{cumfunc}({func_kw_str})"
            ]

        agg_func_str = self.translate_var(context, op.func)
        min_periods_str = self.translate_var(context, min_periods)
        inv_shift_str = self.translate_var(context, -shift)

        # need to call groupby.apply() as groupby.expanding() is not a standard API
        func_kw_str = (", " + func_kw_str) if func_kw_str else ""
        lines = [
            f"def _exp_fun_{res_var_name}(frame, **_):",
            f'    func = {agg_func_str} if func != "prod" else lambda x: x.prod()',
            f"    frame = frame.shift({inv_shift_str})" if shift else None,
            f"    frame = frame.iloc[::-1]" if reverse_range else None,
            f"    out_frame = frame.expanding(min_periods={min_periods_str}).agg(func{func_kw_str})",
            f"    out_frame = out_frame.iloc[::-1]" if reverse_range else None,
            f"    return out_frame",
            f"{res_var_name} = {input_var_name}.{by_call}.apply("
            f"_exp_fun_{res_var_name}, include_groups=False)",
        ]
        return [line for line in lines if line is not None]


@register_op_adapter(GroupByRollingAgg)
class GroupByRollingAggAdapter(SPEGroupByOperatorAdapter):
    def generate_code(
        self, op: GroupByRollingAgg, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        by_call = self.build_groupby_call(context, op.groupby_params)
        agg_func_str = self.translate_var(context, op.func)

        window_params = op.window_params.copy()
        shift = window_params.pop("shift", 0)
        for key in Rolling._mf_specific_fields:
            window_params.pop(key, None)

        window_params_kw_str = ", ".join(
            self._translate_call_args(context, **window_params)
        )
        func_kw_str = ", ".join(self._translate_call_args(context, **op.kwargs))
        func_kw_str = (", " + func_kw_str) if func_kw_str else ""
        inv_shift_str = self.translate_var(context, -shift)

        # need to call groupby.apply() as groupby.rolling() is not a standard API
        return [
            f"def _roll_fun_{res_var_name}(frame, **_):",
            f'    func = {agg_func_str} if func != "prod" else lambda x: x.prod()',
            f"    frame = frame.shift({inv_shift_str})",
            f"    return frame.rolling({window_params_kw_str}).agg(func{func_kw_str})",
            f"{res_var_name} = {input_var_name}.{by_call}.apply("
            f"_roll_fun_{res_var_name}, include_groups=False)",
        ]
