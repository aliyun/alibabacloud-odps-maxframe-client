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
import pyarrow as pa

from .... import opcodes
from ....serialization.serializables.field import AnyField
from .core import LegacySeriesDictOperator, SeriesDictMethod


class SeriesDictContainsOperator(LegacySeriesDictOperator):
    # operator class deprecated since v2.3.0
    _op_type_ = opcodes.SERIES_DICT_CONTAINS
    _method_name = "contains"
    query_key = AnyField("query_key", default=None)


def series_dict_contains(series, query_key):
    """
    Check whether the key is in each dict of the Series.

    Parameters
    ----------
    query_key : Any
        The key to check, must be in the same key type of the dict.

    Returns
    -------
    Series :
        A Series with data type ``pandas.ArrowDtype(pyarrow.bool_)``. The value will
        be ``True`` if the key is in the dict, ``False`` otherwise, or ``None`` if the
        dict is None.

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

    >>> s.dict.contains("k2").execute()
    1     True
    2    False
    3     <NA>
    dtype: bool[pyarrow]
    """
    return SeriesDictMethod(method="contains", method_kwargs=dict(query_key=query_key))(
        series, dtype=pd.ArrowDtype(pa.bool_()), name=None
    )
