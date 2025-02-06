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
from typing import TYPE_CHECKING

from ...core import BaseMaxFrameAccessor

if TYPE_CHECKING:
    from ..core import DataFrame, Index, Series


class DataFrameMaxFrameAccessor(BaseMaxFrameAccessor):
    obj: "DataFrame"
    _api_count: int = 0


class SeriesMaxFrameAccessor(BaseMaxFrameAccessor):
    obj: "Series"
    _api_count: int = 0


class IndexMaxFrameAccessor(BaseMaxFrameAccessor):
    obj: "Index"
    _api_count: int = 0
