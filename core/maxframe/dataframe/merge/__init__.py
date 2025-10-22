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

from .append import append
from .combine import DataFrameCombine, df_combine, series_combine
from .combine_first import df_combine_first, series_combine_first
from .compare import DataFrameCompare, df_compare, series_compare
from .concat import DataFrameConcat, concat
from .merge import (
    DataFrameMerge,
    DistributedMapJoinHint,
    MapJoinHint,
    SkewJoinHint,
    join,
    merge,
)
from .update import DataFrameUpdate, df_update, series_update


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE

    for cls in DATAFRAME_TYPE:
        setattr(cls, "combine_first", df_combine_first)
        setattr(cls, "compare", df_compare)
        setattr(cls, "join", join)
        setattr(cls, "merge", merge)
        setattr(cls, "update", df_update)
        setattr(cls, "combine", df_combine)

    for cls in SERIES_TYPE:
        setattr(cls, "combine_first", series_combine_first)
        setattr(cls, "compare", series_compare)
        setattr(cls, "update", series_update)
        setattr(cls, "combine", series_combine)

    for cls in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(cls, "append", append)


_install()
del _install
