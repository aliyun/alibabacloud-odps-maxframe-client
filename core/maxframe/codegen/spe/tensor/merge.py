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

from ....tensor.core import Tensor
from ....tensor.merge.concatenate import TensorConcatenate
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(TensorConcatenate)
class TensorConcatenateAdapter(SPEOperatorAdapter):
    def generate_code(self, op: Tensor, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        ndarr_str = ", ".join(context.get_input_tileable_variable(k) for k in op.inputs)
        if len(op.inputs) == 1:
            ndarr_str += ","
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        axis_var = context.register_operator_constants(op.axis)
        return [f"{res_var_name} = np.concatenate(({ndarr_str}), axis={axis_var})"]
