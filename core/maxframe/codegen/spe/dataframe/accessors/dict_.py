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

from .....dataframe.accessors.dict_.core import SeriesDictMethod
from ...core import register_op_adapter
from .base import SeriesTemplateMethodAdapter

_get_template = """
def _inner_get(data):
    found = False
    for tup in data:
        if tup[0] == {query_key}:
            found = True
            return tup[1]
    if not found:
        if {ignore_key_error}:
            return {default_value}
        else:
            raise KeyError({query_key})

{output_var} = {input_var}.map(_inner_get, na_action="ignore").astype({output_dtype_var})
{output_var}.name = {query_key}
"""

_set_template = """
def _inner_set(row):
    found = False
    value = list()
    for tup in row:
        if tup[0] == {query_key}:
            value.append((tup[0], {value}))
            found = True
        else:
            value.append(tup)
    if not found:
        value.append(({query_key}, {value}))
    return value

{output_var} = {input_var}.map(_inner_set, na_action="ignore").astype({output_dtype_var})
"""

_len_template = """
{output_var} = {input_var}.map(len, na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""


_remove_template = """
def _inner_remove(value):
    row = list()
    found = False
    for tup in value:
        if tup[0] == {query_key}:
            found = True
        else:
            row.append(tup)
    if not found and not {ignore_key_error}:
        raise KeyError({query_key})
    return row

{output_var} = {input_var}.map(_inner_remove, na_action="ignore").astype({output_dtype_var})
"""

_contains_template = """
{output_var} = {input_var}.map(lambda x: any({query_key} in tup[0] for tup in x), na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""


@register_op_adapter(SeriesDictMethod)
class SeriesDictMethodAdapter(SeriesTemplateMethodAdapter):
    _templates = {
        "contains": _contains_template,
        "getitem": _get_template,
        "len": _len_template,
        "setitem": _set_template,
        "remove": _remove_template,
    }
