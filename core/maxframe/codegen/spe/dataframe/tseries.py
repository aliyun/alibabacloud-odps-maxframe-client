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

from ....dataframe.tseries.between_time import DataFrameBetweenTime
from ....dataframe.tseries.to_datetime import DataFrameToDatetime
from ....utils import no_default
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

DataFrameBetweenTimeAdapter = build_method_call_adapter(
    DataFrameBetweenTime,
    "between_time",
    kw_keys=["start_time", "end_time", "inclusive", "axis"],
    skip_none=True,
)


@register_op_adapter(DataFrameToDatetime)
class DataFrameToDatetimeAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameToDatetime, context: SPECodeContext
    ) -> List[str]:
        context.register_import("pandas", "pd")

        inst_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        raw_kw = dict(
            errors=op.errors,
            dayfirst=op.dayfirst,
            yearfirst=op.yearfirst,
            utc=op.utc,
            format=op.format,
            exact=op.exact,
            unit=op.unit,
            infer_datetime_format=op.infer_datetime_format,
            origin=op.origin,
            cache=op.cache,
        )
        kw = {k: v for k, v in raw_kw.items() if v is not None and v != no_default}
        args_list = self._translate_call_args(context, **kw)
        args = ", ".join([inst_var_name] + args_list)
        return [f"{res_var_name} = pd.to_datetime({args})"]
