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

from ....tensor.sort.sort import TensorSort
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(TensorSort)
class TensorSortAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorSort, context: SPECodeContext) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_names = [
            context.get_output_tileable_variable(out) for out in op.outputs
        ]
        res_name_iter = iter(res_var_names)

        axes_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["axis", "kind", "order"]
        )
        lines = []
        if op.return_value:
            lines.append(
                f"{next(res_name_iter)} = np.sort({input_var_name}, {axes_str})"
            )
        if op.return_indices:
            lines.append(
                f"{next(res_name_iter)} = np.argsort({input_var_name}, {axes_str})"
            )
        return lines
