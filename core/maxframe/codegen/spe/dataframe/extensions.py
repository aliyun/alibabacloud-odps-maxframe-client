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

from ....dataframe.extensions import DataFrameApplyChunk, DataFrameReshuffle
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(DataFrameReshuffle)
class DataFrameReshuffleAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameReshuffle, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        if not op.group_by:
            kwargs = {}
            if op.ignore_index:
                kwargs["ignore_index"] = op.ignore_index
            args = self._translate_call_args(context, frac=1, **kwargs)
            shuffled = f"{input_var}.sample({', '.join(args)})"
        else:
            sort_keys = list(op.group_by) + list(op.sort_by or [])
            kwargs = {"ascending": op.ascending}
            if op.ignore_index:
                kwargs["ignore_index"] = op.ignore_index
            args = self._translate_call_args(context, sort_keys, **kwargs)
            shuffled = f"{input_var}.sort_values({', '.join(args)})"
        output_var = context.get_output_tileable_variable(op.outputs[0])
        return [f"{output_var} = {shuffled}"]


@register_op_adapter(DataFrameApplyChunk)
class DataFrameApplyChunkAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameApplyChunk, context: SPECodeContext
    ) -> List[str]:
        context.register_import("pandas", "pd")
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        func_var = self.translate_var(context, op.func)
        if not op.batch_rows:
            return [f"{output_var} = {func_var}({input_var})"]
        else:
            return [
                f"{input_var}_batches = [{func_var}({input_var}.iloc[i : i + {op.batch_rows}]) "
                f"for i in range(0, len({input_var}), {op.batch_rows})]",
                f"if len({input_var}_batches) > 1:",
                f"    {output_var} = pd.concat({input_var}_batches)",
                f"else:",
                f"    {output_var} = {input_var}_batches[0]",
            ]
