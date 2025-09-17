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

from .....dataframe.accessors.struct_.core import SeriesStructMethod
from ...core import register_op_adapter
from .base import SeriesTemplateMethodAdapter

_field_template = """
{output_var} = {input_var}.struct.field({name_or_index})
"""


@register_op_adapter(SeriesStructMethod)
class SeriesStructMethodAdapter(SeriesTemplateMethodAdapter):
    _templates = {
        "field": _field_template,
    }
