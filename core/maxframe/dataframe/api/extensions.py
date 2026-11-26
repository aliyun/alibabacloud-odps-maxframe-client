# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

# Public shortcut for the DataFrame accessor registrators, which are defined in
# maxframe.dataframe.core next to the type tuples they install on.
from maxframe.dataframe.core import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)

__all__ = [
    "register_dataframe_accessor",
    "register_index_accessor",
    "register_series_accessor",
]
