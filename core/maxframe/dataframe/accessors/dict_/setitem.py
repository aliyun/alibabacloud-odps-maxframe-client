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
from ....core.entity.output_types import OutputType
from ....serialization.serializables.field import AnyField
from ...core import TILEABLE_TYPE
from ...operators import DataFrameOperator, DataFrameOperatorMixin


class SeriesDictSetItemOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_DICT_SETITEM
    query_key = AnyField("query_key", default=None)
    value = AnyField("default_value", default=None)

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.series], **kw)

    def __call__(self, series):
        new_series = self.new_series(
            [series],
            shape=series.shape,
            index_value=series.index_value,
            dtype=series.dtype,
            name=series.name,
        )
        series.data = new_series.data


def series_dict_setitem(series, query_key, value):
    """
    Set the value with the key to each dict of the Series.

    Parameters
    ----------
    query_key : Any
        The key of the value to set to, must be in the same key type of the dict.
    value : Any
        The value to set, must be in the same value type of the dict. If the ``query_key``
        exists, the value will be replaced. Otherwise, the value will be added. A dict
        will be skipped if it's ``None``.

    Returns
    -------
    Series :
        A Series with the same data type.


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

    >>> s.dict["k2"] = 4
    >>> s.execute()
    1    [('k1', 1), ('k2', 4)]
    2    [('k1', 3), ('k2', 4)]
    3                      <NA>
    dtype: map<string, int64>[pyarrow]
    """
    if isinstance(value, TILEABLE_TYPE):
        raise NotImplementedError(
            "Set the value with a dataframe/series is not allowed yet"
        )
    return SeriesDictSetItemOperator(query_key=query_key, value=value)(series)
