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
from .core import LegacySeriesDictOperator, SeriesDictMethod


class SeriesDictGetItemOperator(LegacySeriesDictOperator):
    # operator class deprecated since v2.3.0
    _op_type_ = opcodes.SERIES_DICT_GETITEM
    _method_name = "getitem"
    query_key = AnyField("query_key", default=None)
    default_value = AnyField("default_value", default=None)
    ignore_key_error = BoolField("ignore_key_error", default=False)


def series_dict_getitem(series, query_key, default_value=None):
    """
    Get the value by the key of each dict in the Series.

    Parameters
    ----------
    query_key : Any
        The key to check, must be in the same key type of the dict.
    default_value : Any, optional
        The value to return if the key is not in the dict, by default None.

    Returns
    -------
    Series :
        A Series with the dict value's data type. The value will be ``default_value``
        if the key is not in the dict, or ``None`` if the dict is None.

    See Also
    --------
    Series.dict.__getitem__: Get the value by the key of each dict in the Series.

    Examples
    --------
    Create a series with dict type data.

    >>> import maxframe.dataframe as md
    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import dict_
    >>> s = md.Series(
    ...     data=[[("k1", 1), ("k2", 2)], [("k1", 3)], None],
    ...     index=[1, 2, 3],
    ...     dtype=dict_(pa.string(), pa.int64()),
    ... )
    >>> s.execute()
    1    [('k1', 1), ('k2', 2)]
    2               [('k1', 3)]
    3                      <NA>
    dtype: map<string, int64>[pyarrow]

    >>> s.dict.get("k2", 9).execute()
    1       2
    2       9
    3    <NA>
    Name: k2, dtype: int64[pyarrow]
    """
    method_kw = dict(
        query_key=query_key, default_value=default_value, ignore_key_error=True
    )
    arrow_map_type = series.dtype.pyarrow_dtype
    dtype = pd.ArrowDtype(arrow_map_type.item_type)
    return SeriesDictMethod(method="getitem", method_kwargs=method_kw)(
        series, name=str(query_key), dtype=dtype
    )


def series_dict_getitem_with_key_error(series, query_key):
    """
    Get the value by the key of each dict in the Series. If the key is not in the dict,
    raise KeyError.

    Parameters
    ----------
    query_key : Any
        The key to check, must be in the same key type of the dict.

    Returns
    -------
    Series :
        A Series with the dict value's data type. Return ``None`` if the dict is None.

    Raises
    ------
    KeyError
        If the key is not in one dict.

    See Also
    --------
    Series.dict.get: Get the value by the key of each dict in the Series with an optional
    default value.

    Examples
    --------
    Create a series with dict type data.

    >>> import maxframe.dataframe as md
    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import dict_
    >>> s = md.Series(
    ...     data=[[("k1", 1), ("k2", 2)], [("k1", 3)], None],
    ...     index=[1, 2, 3],
    ...     dtype=dict_(pa.string(), pa.int64()),
    ... )
    >>> s.execute()
    1    [('k1', 1), ('k2', 2)]
    2               [('k1', 3)]
    3                      <NA>
    dtype: map<string, int64>[pyarrow]

    >>> s.dict["k1"].execute()
    1       1
    2       3
    3    <NA>
    Name: k1, dtype: int64[pyarrow]
    """
    method_kw = dict(query_key=query_key, default_value=None, ignore_key_error=False)
    arrow_map_type = series.dtype.pyarrow_dtype
    dtype = pd.ArrowDtype(arrow_map_type.item_type)
    return SeriesDictMethod(method="getitem", method_kwargs=method_kw)(
        series, name=str(query_key), dtype=dtype
    )
