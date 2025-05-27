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

from ....tensor.random.rand import TensorRand
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(TensorRand)
class TensorRandAdapter(SPEOperatorAdapter):
    def generate_code(self, op: TensorRand, context: SPECodeContext) -> List[str]:
        context.register_import("numpy", "np")
        # Use shape to achieve backward compatibility for 1.0 SDK
        shape = op.size or op.outputs[0].shape
        shape_str = ", ".join(str(x) for x in shape) if shape else ""
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = np.random.rand({shape_str})"]
