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

from typing import List

from ....dataframe.window.ewm import DataFrameEwmAgg
from ....dataframe.window.expanding import DataFrameExpandingAgg
from ....dataframe.window.rolling import DataFrameRollingAgg
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(DataFrameExpandingAgg)
class DataFrameExpandingAggAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameExpandingAgg, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        window_args = self.generate_call_args_with_attributes(
            op, context, kw_keys=["min_periods", "axis"], skip_none=True
        )
        return [
            f"{res_var_name} = {input_var_name}.expanding({window_args}).{op.func}()"
        ]


@register_op_adapter(DataFrameEwmAgg)
class DataFrameEwmAggAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameEwmAgg, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_keys = ["alpha", "min_periods", "adjust", "axis"]
        window_args = self.generate_call_args_with_attributes(
            op, context, kw_keys=kw_keys, alpha_ignore_na="ignore_na", skip_none=True
        )
        return [f"{res_var_name} = {input_var_name}.rolling({window_args}).{op.func}()"]


@register_op_adapter(DataFrameRollingAgg)
class DataFrameRollingAggAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameRollingAgg, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        kw_keys = ["min_periods", "center", "win_type", "on", "axis", "closed"]
        window_args = self.generate_call_args_with_attributes(
            op, context, "window", kw_keys=kw_keys, skip_none=True
        )
        args_list = self._translate_call_args(context, *op.func_args, **op.func_kwargs)
        args = ", ".join(args_list)
        return [
            f"{res_var_name} = {input_var_name}.rolling({window_args}).{op.func}({args})"
        ]
