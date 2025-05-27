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

from ....dataframe.statistics.corr import DataFrameCorr
from ....dataframe.statistics.quantile import DataFrameQuantile
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter


@register_op_adapter(DataFrameCorr)
class DataFrameCorrAdapter(SPEOperatorAdapter):
    def generate_code(self, op: DataFrameCorr, context: SPECodeContext) -> List[str]:
        if op.other is None:
            func_name = "corr"
            args = []
            kw_keys = ["method", "min_periods"]
        else:
            func_name = "corrwith"
            args = ["other"]
            kw_keys = ["axis", "drop", "method"]
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args = self.generate_call_args_with_attributes(
            op, context, *args, kw_keys=kw_keys
        )
        return [f"{res_var_name} = {input_var_name}.{func_name}({args})"]


DataFrameQuantileAdapter = build_method_call_adapter(
    DataFrameQuantile,
    "quantile",
    kw_keys=["q", "axis", "numeric_only", "interpolation"],
)
