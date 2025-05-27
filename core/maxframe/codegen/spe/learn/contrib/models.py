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

from .....learn.contrib.models import ModelApplyChunk, ModelDataSource
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(ModelDataSource)
class ModelDataSourceAdapter(SPEOperatorAdapter):
    def generate_code(self, op: ModelDataSource, context: SPECodeContext) -> List[str]:
        out_var = context.get_output_tileable_variable(op.outputs[0])
        model_var = context.register_operator_constants(op.data)
        return [f"{out_var} = {model_var}"]


@register_op_adapter(ModelApplyChunk)
class ModelApplyChunkAdapter(SPEOperatorAdapter):
    def generate_code(self, op: ModelApplyChunk, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        func_var = context.register_operator_constants(op.func)
        call_args = self._translate_call_args(context, *op.args, **op.kwargs)
        output_var_names = [
            context.get_output_tileable_variable(out) for out in op.outputs
        ]
        return [
            f"{', '.join(output_var_names)} = {func_var}("
            f"{input_var_name}, {', '.join(call_args)})"
        ]
