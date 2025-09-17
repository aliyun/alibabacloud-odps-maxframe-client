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

from .... import opcodes
from ....serialization.serializables.field import AnyField, BoolField
from .core import LegacySeriesDictOperator, SeriesDictMethod


class SeriesDictRemoveOperator(LegacySeriesDictOperator):
    # operator class deprecated since v2.3.0
    _op_type_ = opcodes.SERIES_DICT_REMOVE
    _method_name = "remove"
    query_key = AnyField("query_key", default=None)
    ignore_key_error = BoolField("ignore_key_error", default=False)


def series_dict_remove(series, query_key, ignore_key_error: bool = False):
    """
    Remove the item by the key from each dict of the Series.

    Parameters
    ----------
    query_key : Any
        The key to remove, must be in the same key type of the dict.
    ignore_key_error : bool, optional, default False
        When the ``query_key`` is not in the dict, if ``ignore_key_error`` is True,
        nothing will happen in the dict. If ``ignore_key_error`` is ``False``, an
        ``KeyError`` will be raised. If the dict is ``None``, returns ``None``.

    Returns
    -------
    Series :
        A Series with the same data type. If the dict is ``None``.
    Raises
    -------
    KeyError :
        If the ``query_key`` is not in one dict and ``ignore_key_error`` is ``False``.

    Examples
    --------
    Create a series with dict type data.

    >>> import maxframe.dataframe as md
    >>> import pyarrow as pa
    >>> from maxframe.lib.dtypes_extension import dict_
    >>> s = md.Series(
    ...     data=[[("k1", 1), ("k2", 2)], [("k1", 3)], None],
    ...     index=[1, 2, 3],
    ...     dtype=map_(pa.string(), pa.int64()),
    ... )
    >>> s.execute()
    1    [('k1', 1), ('k2', 2)]
    2               [('k1', 3)]
    3                      <NA>
    dtype: map<string, int64>[pyarrow]

    >>> s.dict.remove("k2", ignore_key_error=True).execute()
    1    [('k1', 1)]
    2    [('k1', 3)]
    3           <NA>
    dtype: map<string, int64>[pyarrow]
    """
    method_kwargs = dict(query_key=query_key, ignore_key_error=ignore_key_error)
    return SeriesDictMethod(method="remove", method_kwargs=method_kwargs)(series)
