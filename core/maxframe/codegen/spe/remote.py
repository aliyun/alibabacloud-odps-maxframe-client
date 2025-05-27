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

from ...remote.core import RemoteFunction
from .core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(RemoteFunction)
class RemoteFunctionAdapter(SPEOperatorAdapter):
    def generate_code(self, op: RemoteFunction, context: SPECodeContext) -> List[str]:
        args_str = ", ".join(
            self._translate_call_args(context, *op.function_args, **op.function_kwargs)
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        func_var_name = context.register_operator_constants(op.function)
        return [f"{res_var_name} = {func_var_name}({args_str})"]
