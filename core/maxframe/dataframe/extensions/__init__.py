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

from ...core import CachedAccessor
from .accessor import (
    DataFrameMaxFrameAccessor,
    IndexMaxFrameAccessor,
    SeriesMaxFrameAccessor,
)
from .apply_chunk import (
    DataFrameApplyChunk,
    DataFrameApplyChunkOperator,
    df_apply_chunk,
    series_apply_chunk,
)
from .cartesian_chunk import cartesian_chunk
from .collect_kv import collect_kv
from .extract_kv import extract_kv
from .flatjson import series_flatjson
from .flatmap import df_flatmap, series_flatmap
from .map_reduce import map_reduce
from .rebalance import DataFrameRebalance, rebalance
from .reshuffle import DataFrameReshuffle, df_reshuffle


def _install():
    from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE

    DataFrameMaxFrameAccessor._register("apply_chunk", df_apply_chunk)
    DataFrameMaxFrameAccessor._register("cartesian_chunk", cartesian_chunk)
    DataFrameMaxFrameAccessor._register("collect_kv", collect_kv)
    DataFrameMaxFrameAccessor._register("extract_kv", extract_kv)
    DataFrameMaxFrameAccessor._register("flatmap", df_flatmap)
    DataFrameMaxFrameAccessor._register("map_reduce", map_reduce)
    DataFrameMaxFrameAccessor._register("rebalance", rebalance)
    DataFrameMaxFrameAccessor._register("reshuffle", df_reshuffle)

    SeriesMaxFrameAccessor._register("apply_chunk", series_apply_chunk)
    SeriesMaxFrameAccessor._register("cartesian_chunk", cartesian_chunk)
    SeriesMaxFrameAccessor._register("extract_kv", extract_kv)
    SeriesMaxFrameAccessor._register("flatjson", series_flatjson)
    SeriesMaxFrameAccessor._register("flatmap", series_flatmap)
    SeriesMaxFrameAccessor._register("rebalance", rebalance)

    IndexMaxFrameAccessor._register("rebalance", rebalance)

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
