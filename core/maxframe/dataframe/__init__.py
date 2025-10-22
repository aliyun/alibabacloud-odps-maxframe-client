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

from . import (
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
    reduction,
    sort,
    statistics,
    tseries,
    ufunc,
    window,
)
from .datasource.date_range import date_range
from .datasource.direct import read_clipboard
from .datasource.from_index import series_from_index
from .datasource.from_records import from_records
from .datasource.from_tensor import dataframe_from_tensor, series_from_tensor
from .datasource.read_csv import read_csv
from .datasource.read_odps_query import read_odps_query
from .datasource.read_odps_table import read_odps_table
from .datasource.read_parquet import read_parquet
from .datastore.to_odps import to_odps_table
from .initializer import DataFrame, Index, Series, read_pandas
from .merge import concat, merge
from .misc.cut import cut
from .misc.eval import maxframe_eval as eval  # pylint: disable=redefined-builtin
from .misc.get_dummies import get_dummies
from .misc.qcut import qcut
from .misc.to_numeric import to_numeric
from .missing import isna, isnull, notna, notnull
from .reduction import CustomReduction, NamedAgg, unique
from .reshape.melt import melt
from .reshape.pivot import pivot
from .reshape.pivot_table import pivot_table
from .tseries.to_datetime import to_datetime

try:
    from pandas import NA, NaT, Timestamp
except ImportError:  # pragma: no cover
    pass

try:
    from . import _internal
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
    reduction,
    sort,
    tseries,
    ufunc,
)

from_pandas = read_pandas

# isort: off
from .typing_ import register_pandas_typing_funcs

register_pandas_typing_funcs()
del register_pandas_typing_funcs
