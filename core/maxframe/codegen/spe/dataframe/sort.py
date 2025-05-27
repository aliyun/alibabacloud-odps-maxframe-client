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
from ....dataframe.sort import DataFrameSortIndex, DataFrameSortValues
from ....dataframe.sort.core import DataFrameSortOperator
from ..core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


class _DataFrameSortAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: DataFrameSortOperator, context: SPECodeContext
    ) -> List[str]:
        input_var_name = context.get_input_tileable_variable(op.inputs[0])
        kwargs = {
            "axis": op.axis,
            "ascending": op.ascending,
            "inplace": op.inplace,
            "na_position": op.na_position,
            "ignore_index": op.ignore_index,
        }
        kwargs.update(self.extra_args(op, context))
        args_str = ", ".join(self._translate_call_args(context, **kwargs))
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {input_var_name}.{self.method_name}({args_str})"]

    @abstractmethod
    def extra_args(
        self, op: DataFrameSortOperator, context: SPECodeContext
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def method_name(self) -> str:
        raise NotImplementedError


@register_op_adapter(DataFrameSortIndex)
class DataFrameSortIndexAdapter(_DataFrameSortAdapter):
    _method_name = "sort_index"

    @property
    def method_name(self) -> str:
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

    @property
    def method_name(self) -> str:
        return self._method_name

    def extra_args(
        self, op: DataFrameSortValues, context: SPECodeContext
    ) -> Dict[str, Any]:
        if isinstance(op.outputs[0], DATAFRAME_TYPE):
            return {"by": op.by}
        return {}
