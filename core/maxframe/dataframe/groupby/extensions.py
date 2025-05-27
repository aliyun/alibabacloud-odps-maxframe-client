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

from ...core import BaseMaxFrameAccessor
from ..core import DataFrameGroupBy, SeriesGroupBy


class DataFrameGroupByMaxFrameAccessor(BaseMaxFrameAccessor):
    obj: "DataFrameGroupBy"
    _api_count: int = 0


class SeriesGroupByMaxFrameAccessor(BaseMaxFrameAccessor):
    obj: "SeriesGroupBy"
    _api_count: int = 0
