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

from .....learn.preprocessing._data.normalize import TensorNormalize
from .....typing_ import OperatorType
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(TensorNormalize)
class TensorNormalizeAdapter(SPEOperatorAdapter):
    def generate_code(self, op: OperatorType, context: SPECodeContext) -> List[str]:
        context.register_import(
            "sklearn.preprocessing", from_item="normalize", alias="sk_normalize"
        )

        call_args = self.generate_call_args_with_attributes(
            op,
            context,
            "input",
            skip_none=True,
            kw_keys=["norm", "axis", "return_norm"],
        )
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = sk_normalize({call_args})"]
