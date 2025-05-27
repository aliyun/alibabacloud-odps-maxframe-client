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

from .....dataframe.accessors.dict_.contains import SeriesDictContainsOperator
from .....dataframe.accessors.dict_.getitem import SeriesDictGetItemOperator
from .....dataframe.accessors.dict_.length import SeriesDictLengthOperator
from .....dataframe.accessors.dict_.remove import SeriesDictRemoveOperator
from .....dataframe.accessors.dict_.setitem import SeriesDictSetItemOperator
from ...core import SPECodeContext, SPEOperatorAdapter, register_op_adapter

_get_template = """
def _inner_get(data):
    found = False
    for tup in data:
        if tup[0] == {key_var}:
            found = True
            return tup[1]
    if not found:
        if {ignore_key_error_var}:
            return {default_value_var}
        else:
            raise KeyError({key_var})

{output_var} = {input_var}.map(_inner_get, na_action="ignore").astype({output_dtype_var})
{output_var}.name = {key_var}
"""

_set_template = """
def _inner_set(row):
    found = False
    value = list()
    for tup in row:
        if tup[0] == {key_var}:
            value.append((tup[0], {value_var}))
            found = True
        else:
            value.append(tup)
    if not found:
        value.append(({key_var}, {value_var}))
    return value

{output_var} = {input_var}.map(_inner_set, na_action="ignore").astype({output_dtype_var})
"""

_length_template = """
{output_var} = {input_var}.map(len, na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""


_remove_template = """
def _inner_remove(value):
    row = list()
    found = False
    for tup in value:
        if tup[0] == {key_var}:
            found = True
        else:
            row.append(tup)
    if not found and not {ignore_key_error_var}:
        raise KeyError({key_var})
    return row

{output_var} = {input_var}.map(_inner_remove, na_action="ignore").astype({output_dtype_var})
"""

_contains_template = """
{output_var} = {input_var}.map(lambda x: any({key_var} in tup[0] for tup in x), na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""


@register_op_adapter(SeriesDictGetItemOperator)
class SeriesDictGetItemOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesDictGetItemOperator, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        key_var = self.translate_var(context, op.query_key)
        output_name_var = self.translate_var(context, op.outputs[0].name)
        default_value_var = self.translate_var(context, op.default_value)
        ignore_key_error_var = self.translate_var(context, op.ignore_key_error)
        output_dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        return [
            _get_template.format(
                input_var=input_var,
                output_var=output_var,
                key_var=key_var,
                default_value_var=default_value_var,
                ignore_key_error_var=ignore_key_error_var,
                output_name_var=output_name_var,
                output_dtype_var=output_dtype_var,
            )
        ]


@register_op_adapter(SeriesDictSetItemOperator)
class SeriesDictSetItemOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesDictSetItemOperator, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        key_var = self.translate_var(context, op.query_key)
        value_var = self.translate_var(context, op.value)
        output_name_var = self.translate_var(context, op.outputs[0].name)
        output_dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        return [
            _set_template.format(
                input_var=input_var,
                output_var=output_var,
                key_var=key_var,
                value_var=value_var,
                output_name_var=output_name_var,
                output_dtype_var=output_dtype_var,
            )
        ]


@register_op_adapter(SeriesDictLengthOperator)
class SeriesDictLengthOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesDictLengthOperator, context: SPECodeContext
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


@register_op_adapter(SeriesDictRemoveOperator)
class SeriesDictRemoveOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesDictRemoveOperator, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        output_name_var = self.translate_var(context, op.outputs[0].name)
        output_dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        key_var = self.translate_var(context, op.query_key)
        ignore_key_error_var = self.translate_var(context, op.ignore_key_error)
        return [
            _remove_template.format(
                input_var=input_var,
                output_var=output_var,
                key_var=key_var,
                ignore_key_error_var=ignore_key_error_var,
                output_name_var=output_name_var,
                output_dtype_var=output_dtype_var,
            )
        ]


@register_op_adapter(SeriesDictContainsOperator)
class SeriesDictContainsOperatorAdapter(SPEOperatorAdapter):
    def generate_code(
        self, op: SeriesDictContainsOperator, context: SPECodeContext
    ) -> List[str]:
        input_var = context.get_input_tileable_variable(op.inputs[0])
        output_var = context.get_output_tileable_variable(op.outputs[0])
        output_name_var = self.translate_var(context, op.outputs[0].name)
        output_dtype_var = context.register_operator_constants(op.outputs[0].dtype)
        key_var = self.translate_var(context, op.query_key)
        return [
            _contains_template.format(
                input_var=input_var,
                output_var=output_var,
                key_var=key_var,
                output_name_var=output_name_var,
                output_dtype_var=output_dtype_var,
            )
        ]
