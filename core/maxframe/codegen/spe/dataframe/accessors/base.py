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

from .....dataframe.accessors.datetime_ import SeriesDatetimeMethod
from .....dataframe.accessors.string_ import SeriesStringMethod
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter


@register_op_adapter(SeriesDatetimeMethod)
class SeriesDatetimeMethodAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesDatetimeMethod, context: SPECodeContext
    ) -> List[str]:
        inst_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        method_str = f"{inst_var_name}.dt.{op.method}"
        if op.is_property:
            return [f"{res_var_name} = {method_str}"]
        else:
            args_list = self._translate_call_args(
                context, *op.method_args, **op.method_kwargs
            )
            args = ", ".join(args_list)
            return [f"{res_var_name} = {method_str}({args})"]


@register_op_adapter(SeriesStringMethod)
class SeriesStringMethodAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesStringMethod, context: SPECodeContext
    ) -> List[str]:
        inst_var_name = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        method_str = f"{inst_var_name}.str.{op.method}"

        args_list = self._translate_call_args(
            context, *op.method_args, **op.method_kwargs
        )
        args = ", ".join(args_list)
        return [f"{res_var_name} = {method_str}({args})"]


class SeriesTemplateMethodAdapter(SPEOperatorAdapter):
    _templates = None

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        kw = {
            "input_var": context.get_input_tileable_variable(op.inputs[0]),
            "output_var": context.get_output_tileable_variable(op.outputs[0]),
            "output_name_var": self.translate_var(context, op.outputs[0].name),
            "output_dtype_var": context.register_operator_constants(
                op.outputs[0].dtype
            ),
        }
        kw.update(
            {k: self.translate_var(context, v) for k, v in op.method_kwargs.items()}
        )
        return [self._templates.get(op.method).format(**kw)]
