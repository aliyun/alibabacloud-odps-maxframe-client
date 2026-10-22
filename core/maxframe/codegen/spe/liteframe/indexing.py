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

from typing import List

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.liteframe.operators.indexing import (
    LiteFrameIlocGetItem,
    LiteFrameIlocSetItem,
)


@register_op_adapter(LiteFrameIlocGetItem)
class LiteFrameIlocGetItemAdapter(SPEOperatorAdapter):
    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        row_index = op.indexes[0]
        # An integer row index must select a one-row DataFrame, not a Series.
        if isinstance(row_index, int):
            row_exp = f"[{self.translate_var(context, row_index)}]"
        else:
            row_exp = self.translate_var(context, row_index)
        col_exp = ""
        if len(op.indexes) > 1:
            col_exp = f", {self.translate_var(context, op.indexes[1])}"
        return [f"{output_var} = {input_var}.iloc[{row_exp}{col_exp}]"]


@register_op_adapter(LiteFrameIlocSetItem)
class LiteFrameIlocSetItemAdapter(SPEOperatorAdapter):
    def generate_code(self, op, context: SPECodeContext) -> List[str]:
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
