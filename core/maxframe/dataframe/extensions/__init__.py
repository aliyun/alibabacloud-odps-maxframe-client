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

from ..misc.accessor import CachedAccessor
from .accessor import (
    DataFrameMaxFrameAccessor,
    IndexMaxFrameAccessor,
    SeriesMaxFrameAccessor,
)
from .reshuffle import DataFrameReshuffle, df_reshuffle


def _install():
    from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE

    DataFrameMaxFrameAccessor._register("reshuffle", df_reshuffle)

    if DataFrameMaxFrameAccessor._api_count:
        for t in DATAFRAME_TYPE:
            t.mf = CachedAccessor("mf", DataFrameMaxFrameAccessor)
    if SeriesMaxFrameAccessor._api_count:
        for t in SERIES_TYPE:
            t.mf = CachedAccessor("mf", SeriesMaxFrameAccessor)
    if IndexMaxFrameAccessor._api_count:
        for t in INDEX_TYPE:
            t.mf = CachedAccessor("mf", IndexMaxFrameAccessor)


_install()
del _install
