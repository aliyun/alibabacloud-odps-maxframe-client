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

from ....tensor.indexing import (
    TensorIndex,
    TensorIndexSetValue,
    TensorNonzero,
    TensorUnravelIndex,
)
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter


@register_op_adapter(TensorIndex)
class TensorIndexAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorIndex, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        indexes_var_name = self.translate_var(context, op.indexes)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        return [f"{res_var_name} = {input_var_name}[tuple({indexes_var_name})]"]


@register_op_adapter(TensorIndexSetValue)
class TensorIndexSetValueAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: TensorIndexSetValue, context: SPECodeContext
    ) -> List[str]:
        context.register_import("numpy", "np")
        indexes_var_name = self.translate_var(context, op.indexes)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        value_expr = self.translate_var(context, op.value)
        return [
            f"{res_var_name} = {input_var_name}.copy()",
            f"{res_var_name}[tuple({indexes_var_name})] = {value_expr}",
        ]


TensorNonzeroAdapter = build_method_call_adapter(
    TensorNonzero, "nonzero", 0, source_module="np"
)
TensorUnravelIndexAdapter = build_method_call_adapter(
    TensorUnravelIndex,
    "unravel_index",
    0,
    skip_none=True,
    kw_keys=["dims", "order"],
    source_module="np",
)
