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

from .....dataframe.accessors.list_.getitem import SeriesListGetItemOperator
from .....dataframe.accessors.list_.length import SeriesListLengthOperator
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter

_get_template = """
def _inner_get(data):
    try:
        return data[{index_var}]
    except IndexError:
        if {ignore_index_error_var}:
            return None
        else:
            raise

{output_var} = {input_var}.map(_inner_get, na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""

_length_template = """
{output_var} = {input_var}.map(len, na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""


@register_op_adapter(SeriesListGetItemOperator)
class SeriesListGetItemOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesListGetItemOperator, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        index_var = self.translate_var(context, op.query_index)
        output_name_var = self.translate_var(context, op.outputs[0].name)
        ignore_index_error_var = self.translate_var(context, op.ignore_index_error)
        output_dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        return [
            _get_template.format(
                input_var=input_var,
                output_var=output_var,
                index_var=index_var,
                ignore_index_error_var=ignore_index_error_var,
                output_name_var=output_name_var,
                output_dtype_var=output_dtype_var,
            )
        ]


@register_op_adapter(SeriesListLengthOperator)
class SeriesListLengthOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesListLengthOperator, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        output_name_var = self.translate_var(context, op.outputs[0].name)
        output_dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        return [
            _length_template.format(
                input_var=input_var,
                output_var=output_var,
                output_name_var=output_name_var,
                output_dtype_var=output_dtype_var,
            )
        ]
