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

from .....learn.utils.validation import CheckConsistentLength
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(CheckConsistentLength)
class CheckConsistentLengthAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: CheckConsistentLength, context: SPECodeContext
    ) -> List[str]:
        context.register_import(
            "maxframe.learn.utils.validation", from_item="check_consistent_length"
        )
        inp_vars = [context.get_input_tileable_variable(inp) for inp in op.inputs]
        inp_vars_str = ", ".join(inp_vars)
        if op.ref_length is not None:
            inp_vars_str += f", ref_length={op.ref_length}"
        out_vars = [context.get_output_tileable_variable(out) for out in op.outputs]
        out_vars_str = ", ".join(out_vars)
        return [f"{out_vars_str} = check_consistent_length({inp_vars_str})"]
