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

from maxframe.dataframe import (
    accessors,
    arithmetic,
    datasource,
    datastore,
    extensions,
    fetch,
    groupby,
    indexing,
    merge,
    misc,
    missing,
    multimodal,
    reduction,
    sort,
    statistics,
    tseries,
    ufunc,
    window,
)
from maxframe.dataframe.datasource.date_range import date_range
from maxframe.dataframe.datasource.direct import read_clipboard
from maxframe.dataframe.datasource.from_index import series_from_index
from maxframe.dataframe.datasource.from_records import from_records
from maxframe.dataframe.datasource.from_tensor import (
    dataframe_from_tensor,
    series_from_tensor,
)
from maxframe.dataframe.datasource.read_csv import read_csv
from maxframe.dataframe.datasource.read_json import read_json
from maxframe.dataframe.datasource.read_lance import read_lance
from maxframe.dataframe.datasource.read_odps_query import read_odps_query
from maxframe.dataframe.datasource.read_odps_table import read_odps_table
from maxframe.dataframe.datasource.read_parquet import read_parquet
from maxframe.dataframe.datastore.to_lance import to_lance
from maxframe.dataframe.datastore.to_odps import to_odps_table
from maxframe.dataframe.initializer import DataFrame, Index, Series, read_pandas
from maxframe.dataframe.merge import concat, merge
from maxframe.dataframe.misc.cut import cut
from maxframe.dataframe.misc.eval import (
    maxframe_eval as eval,  # pylint: disable=redefined-builtin
)
from maxframe.dataframe.misc.factorize import factorize
from maxframe.dataframe.misc.get_dummies import get_dummies
from maxframe.dataframe.misc.qcut import qcut
from maxframe.dataframe.misc.to_numeric import to_numeric
from maxframe.dataframe.missing import isna, isnull, notna, notnull
from maxframe.dataframe.reduction import CustomReduction, NamedAgg, unique
from maxframe.dataframe.reshape.melt import melt
from maxframe.dataframe.reshape.pivot import pivot
from maxframe.dataframe.reshape.pivot_table import pivot_table
from maxframe.dataframe.tseries.to_datetime import to_datetime

try:
    from pandas import NA, NaT, Timestamp
except ImportError:  # pragma: no cover
    pass

try:
    from maxframe.dataframe import _internal
except ImportError:  # pragma: no cover
    pass

del (
    accessors,
    arithmetic,
    datasource,
    datastore,
    extensions,
    fetch,
    groupby,
    indexing,
    multimodal,
    reduction,
    sort,
    tseries,
    ufunc,
)

from_pandas = read_pandas

# isort: off
from maxframe.lib.compat import patch_pandas
from maxframe.dataframe.typing_ import dtype, infer_dtype, register_pandas_typing_funcs

register_pandas_typing_funcs()
del register_pandas_typing_funcs
patch_pandas()
del patch_pandas
