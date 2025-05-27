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

from ....dataframe.missing.checkna import DataFrameCheckNA
from ....dataframe.missing.dropna import DataFrameDropNA
from ....dataframe.missing.fillna import DataFrameFillNA
from ....dataframe.missing.replace import DataFrameReplace
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter


@register_op_adapter(DataFrameDropNA)
class DataFrameDropNAAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameDropNA, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        args_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["axis", "how", "thresh", "subset"], skip_none=True
        )
        drop_idx_code = "" if not op.ignore_index else ".reset_index(drop=True)"
        return [f"{output_var} = {input_var}.dropna({args_str}){drop_idx_code}"]


DataFrameFillNAAdapter = build_method_call_adapter(
    DataFrameFillNA,
    "fillna",
    skip_none=False,
    kw_keys=["value", "method", "axis", "limit", "downcast"],
)


@register_op_adapter(DataFrameCheckNA)
class DataFrameCheckNAAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameCheckNA, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        if op.positive:
            return [f"{output_var} = {input_var}.isna()"]
        else:
            return [f"{output_var} = {input_var}.notna()"]


@register_op_adapter(DataFrameReplace)
class DataFrameReplaceAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameReplace, context: SPECodeContext) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        args_str = self.generate_call_args_with_attributes(
            op, context, kw_keys=["to_replace", "value", "limit", "regex", "method"]
        )
        return [f"{output_var} = {input_var}.replace({args_str})"]
