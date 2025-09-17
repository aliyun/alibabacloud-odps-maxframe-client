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

from .....dataframe.accessors.list_.core import SeriesListMethod
from ...core import register_op_adapter
from .base import SeriesTemplateMethodAdapter

_get_template = """
def _inner_get(data):
    try:
        return data[{query_index}]
    except IndexError:
        if {ignore_index_error}:
            return None
        else:
            raise

{output_var} = {input_var}.map(_inner_get, na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""

_len_template = """
{output_var} = {input_var}.map(len, na_action="ignore").astype({output_dtype_var})
{output_var}.name = None
"""


@register_op_adapter(SeriesListMethod)
class SeriesListMethodAdapter(SeriesTemplateMethodAdapter):
    _templates = {
        "getitem": _get_template,
        "len": _len_template,
    }
