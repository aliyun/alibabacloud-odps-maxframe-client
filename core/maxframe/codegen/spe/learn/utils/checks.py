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

from .....learn.utils.checks import AssertAllFinite, CheckNonNegative
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(AssertAllFinite)
class AssertAllFiniteAdapter(SPEOperatorAdapter):
    def generate_code(self, op: AssertAllFinite, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.utils", from_item="assert_all_finite", alias="sk_assert_all_finite"
        )

        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "x",
            skip_none=True,
            kw_keys=["allow_nan", "msg_dtype", "check_only"],
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_assert_all_finite({call_args})"]


@register_op_adapter(CheckNonNegative)
class CheckNonNegativeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: CheckNonNegative, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.utils.validation",
            from_item="check_non_negative",
            alias="sk_check_non_negative",
        )
        call_args = self.generate_call_args_with_attributes(
            op, context, "input", skip_none=True, kw_keys=["whom"]
        )
        val_var_name = context.get_input_tileable_variable(op.value)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [
            f"sk_check_non_negative({call_args})",
            f"{res_var_name} = {val_var_name}",
        ]
