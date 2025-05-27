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

from ....tensor.extensions.apply_chunk import TensorApplyChunk
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(TensorApplyChunk)
class TensorApplyChunkAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorApplyChunk, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        func_var = self.translate_var(context, op.func)
        kwargs = op.kwargs.copy()
        if op.with_chunk_index:
            kwargs["chunk_index"] = (0,) * op.inputs[0].ndim
        args = self._translate_call_args(context, *op.args, **kwargs)
        if args:
            args_str = ", " + ", ".join(args)
        else:
            args_str = ""
        return [f"{output_var} = {func_var}({input_var}{args_str})"]
