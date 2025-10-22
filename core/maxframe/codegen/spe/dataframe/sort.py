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

from abc import abstractmethod
from typing import Any, Dict, List

from ....dataframe.core import DATAFRAME_TYPE
from ....dataframe.sort import DataFrameRank, DataFrameSortIndex, DataFrameSortValues
from ....dataframe.sort.core import DataFrameSortOperator
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter
from ..utils import build_method_call_adapter

DataFrameRankAdapter = build_method_call_adapter(
    DataFrameRank,
    "rank",
    kw_keys=["axis", "method", "numeric_only", "na_option", "ascending", "pct"],
    skip_none=True,
)


class _DataFrameSortAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameSortOperator, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        if op.keep_kind == "head" or not op.nrows:
            args = []
            kwargs = {
                "axis": op.axis,
                "ascending": op.ascending,
                "inplace": op.inplace,
                "na_position": op.na_position,
                "ignore_index": op.ignore_index,
            }
        else:
            args = [op.nrows]
            kwargs = {"keep": op.keep_kind}
        kwargs.update(self.extra_args(op, context))
        args_str = ", ".join(self._translate_call_args(context, *args, **kwargs))
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        ret = [
            f"{res_var_name} = {input_var_name}.{self.get_method_name(op)}({args_str})"
        ]
        if op.keep_kind == "head" and op.nrows:
            ret.append(f"{res_var_name} = {res_var_name}.head({op.nrows})")
        return ret

    @abstractmethod
    def extra_args(
        self, op: DataFrameSortOperator, context: SPECodeContext
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_method_name(self, op: DataFrameSortOperator) -> str:
        raise NotImplementedError


@register_op_adapter(DataFrameSortIndex)
class DataFrameSortIndexAdapter(_DataFrameSortAdapter):
    _method_name = "sort_index"

    def get_method_name(self, op: DataFrameSortIndex) -> str:
        return self._method_name

    def extra_args(
        self, op: DataFrameSortIndex, context: SPECodeContext
    ) -> Dict[str, Any]:
        return {
            "level": op.level,
            "sort_remaining": op.sort_remaining,
        }


@register_op_adapter(DataFrameSortValues)
class DataFrameSortValuesAdapter(_DataFrameSortAdapter):
    _method_name = "sort_values"

    def get_method_name(self, op: DataFrameSortValues) -> str:
        if op.keep_kind == "head" or not op.nrows:
            return self._method_name
        else:
            return "nsmallest" if op.ascending else "nlargest"

    def extra_args(
        self, op: DataFrameSortValues, context: SPECodeContext
    ) -> Dict[str, Any]:
        if isinstance(op.outputs[0], DATAFRAME_TYPE):
            if op.keep_kind == "head" or not op.nrows:
                return {"by": op.by}
            else:
                return {"columns": op.by}
        return {}
