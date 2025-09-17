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

import pandas as pd

from .... import opcodes
from ....serialization.serializables.field import AnyField, BoolField
from .core import LegacySeriesListOperator, SeriesListMethod


class SeriesListGetItemOperator(LegacySeriesListOperator):
    # operator class deprecated since v2.3.0
    _op_type_ = opcodes.SERIES_LIST_GETITEM
    _method_name = "getitem"
    query_index = AnyField("query_index", default=None)
    ignore_index_error = BoolField("ignore_index_error", default=False)


def series_list_getitem(series, query_index):
    """
    Get the value by the index of each list in the Series.

    Parameters
    ----------
    query_index : Any
        The key to check, must be index.

    Returns
    -------
    Series :
        A Series with the list value's data type. The value will be
        ``None`` if the list is None.

    Examples
    --------
    Create a series with list type data.

    >>> import maxframe.dataframe as md
    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import list_
    >>> s = md.Series(
    ...     data=[[1, 2, 3], [4, 5, 6], None],
    ...     index=[1, 2, 3],
    ...     dtype=list_(pa.int64()),
    ... )
    >>> s.execute()
    1    [1, 2, 3]
    2    [4, 5, 6]
    3         <NA>
    dtype: list<int64>[pyarrow]

    >>> s.list.get(0).execute()
    1       1
    2       4
    3    <NA>
    dtype: int64[pyarrow]
    """
    kw = dict(query_index=query_index, ignore_index_error=True)
    arrow_list_type = series.dtype.pyarrow_dtype
    return SeriesListMethod(method="getitem", method_kwargs=kw)(
        series, dtype=pd.ArrowDtype(arrow_list_type.value_type)
    )


def series_list_getitem_with_index_error(series, query_index):
    """
    Get the value by the index of each list in the Series. If the index
    is not in the list, raise IndexError.

    Parameters
    ----------
    query_index : Any
        The index to check, must be integer.

    Returns
    -------
    Series :
        A Series with the list value's data type. Return ``None`` if the list is None.

    Raises
    ------
    KeyError
        If the index is not in one list.

    See Also
    --------
    Series.list.get: Get the value by the index of each list in the Series.

    Examples
    --------
    Create a series with list type data.

    >>> import maxframe.dataframe as md
    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import list_
    >>> s = md.Series(
    ...     data=[[1, 2, 3], [4, 5, 6], None],
    ...     index=[1, 2, 3],
    ...     dtype=list_(pa.int64()),
    ... )
    >>> s.execute()
    1    [1, 2, 3]
    2    [4, 5, 6]
    3         <NA>
    dtype: list<int64>[pyarrow]

    >>> s.list.get(0).execute()
    1       1
    2       4
    3    <NA>
    dtype: int64[pyarrow]
    """
    kw = dict(query_index=query_index, ignore_index_error=False)
    arrow_list_type = series.dtype.pyarrow_dtype
    return SeriesListMethod(method="getitem", method_kwargs=kw)(
        series, dtype=pd.ArrowDtype(arrow_list_type.value_type)
    )
