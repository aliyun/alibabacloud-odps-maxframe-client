# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import DataFrame, Index, Series


class BaseMaxFrameAccessor:
    def __init__(self, df_obj):
        self._df_obj = df_obj

    @classmethod
    def _register(cls, name, func):
        @functools.wraps(func)
        def wrapped(self, *args, **kw):
            return func(self._df_obj, *args, **kw)

        wrapped.__name__ = name
        setattr(cls, name, wrapped)
        if hasattr(cls, "_api_count"):  # pragma: no branch
            cls._api_count += 1


class DataFrameMaxFrameAccessor(BaseMaxFrameAccessor):
    _df_obj: "DataFrame"
    _api_count: int = 0


class SeriesMaxFrameAccessor(BaseMaxFrameAccessor):
    _df_obj: "Series"
    _api_count: int = 0


class IndexMaxFrameAccessor(BaseMaxFrameAccessor):
    _df_obj: "Index"
    _api_count: int = 0
