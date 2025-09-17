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

from ....dataframe.reshape.melt import DataFrameMelt
from ....dataframe.reshape.pivot import DataFramePivot
from ....dataframe.reshape.pivot_table import DataFramePivotTable
from ....dataframe.reshape.stack import DataFrameStack
from ..utils import build_method_call_adapter

DataFrameMeltAdapter = build_method_call_adapter(
    DataFrameMelt,
    "melt",
    kw_keys=["id_vars", "value_vars", "var_name", "value_name", "col_level"],
)
DataFrameStackAdapter = build_method_call_adapter(
    DataFrameStack, "stack", kw_keys=["level", "dropna"]
)
DataFramePivotAdapter = build_method_call_adapter(
    DataFramePivot, "pivot", kw_keys=["columns", "index", "values"]
)
DataFramePivotTableAdapter = build_method_call_adapter(
    DataFramePivotTable,
    "pivot_table",
    kw_keys=[
        "values",
        "index",
        "columns",
        "aggfunc",
        "fill_value",
        "margins",
        "dropna",
        "margins_name",
        "sort",
    ],
)
