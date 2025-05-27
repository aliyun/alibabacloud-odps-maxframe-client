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

from ....dataframe.misc.value_counts import DataFrameValueCounts
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(DataFrameValueCounts)
class DataFrameValueCountsAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameValueCounts, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["ascending", "bins", "dropna", "normalize", "sort"]
        )
        return [f"{res_var_name} = {input_var_name}.value_counts({args_str})"]
