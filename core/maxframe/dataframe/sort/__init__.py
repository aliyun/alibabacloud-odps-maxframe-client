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

from .rank import DataFrameRank
from .sort_index import DataFrameSortIndex
from .sort_values import DataFrameSortValues


def _install():
    from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
    from .argsort import index_argsort, series_argsort
    from .nlargest import df_nlargest, series_nlargest
    from .nsmallest import df_nsmallest, series_nsmallest
    from .rank import rank
    from .sort_index import sort_index
    from .sort_values import dataframe_sort_values, series_sort_values

    for cls in DATAFRAME_TYPE:
        setattr(cls, "nlargest", df_nlargest)
        setattr(cls, "nsmallest", df_nsmallest)
        setattr(cls, "rank", rank)
        setattr(cls, "sort_values", dataframe_sort_values)
        setattr(cls, "sort_index", sort_index)

    for cls in SERIES_TYPE:
        setattr(cls, "argsort", series_argsort)
        setattr(cls, "nlargest", series_nlargest)
        setattr(cls, "nsmallest", series_nsmallest)
        setattr(cls, "rank", rank)
        setattr(cls, "sort_values", series_sort_values)
        setattr(cls, "sort_index", sort_index)

    for cls in INDEX_TYPE:
        setattr(cls, "argsort", index_argsort)


_install()
del _install
