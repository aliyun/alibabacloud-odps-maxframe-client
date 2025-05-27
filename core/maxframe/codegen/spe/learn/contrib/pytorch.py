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

import sys
from typing import List, Union

from .....learn.contrib.pytorch.run_script import RunPyTorch
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(RunPyTorch)
class RunPyTorchAdapter(SPEOperatorAdapter):
    @staticmethod
    def _run_script(code: Union[bytes, str], data: dict, cmd_args: List[str]):
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["script"] + cmd_args
            locals_dict = {"__name__": "__main__"}
            locals_dict.update(data)

            exec(code, locals_dict)

            return {"status": "ok"}
        finally:
            sys.argv = old_argv

    def generate_code(self, op: RunPyTorch, context: SPECodeContext) -> List[str]:
        data_args = ", ".join(self._translate_call_args(context, **(op.data or {})))
        cmd_args_str = ", ".join(f'"{v}"' for v in (op.command_args or []))
        code_var = context.register_operator_constants(op.code)
        output_var = context.get_output_tileable_variable(op.outputs[0])

        cls_name = type(self).__name__
        context.register_import(__name__, from_item=type(self).__name__)
        return [
            f"{output_var} = {cls_name}._run_script("
            f"{code_var}, dict({data_args}), [{cmd_args_str}])"
        ]
