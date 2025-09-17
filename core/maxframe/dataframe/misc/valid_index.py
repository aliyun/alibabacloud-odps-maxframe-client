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

from ...udf import builtin_function


@builtin_function
def _item_or_none(item):
    if len(item) > 0:
        return item[0]
    return None


def _valid_index(df_or_series, slc: slice):
    from ... import tensor as mt

    idx = df_or_series.dropna(how="all").index[slc]
    return mt.array(idx).mf.apply_chunk(_item_or_none, dtype=idx.dtype)


_doc = """
Return index for %(pos)s non-NA value or None, if no non-NA value is found.

Returns
-------
type of index

Examples
--------
For Series:

>>> import maxframe.dataframe as md
>>> s = md.Series([None, 3, 4])
>>> s.first_valid_index().execute()
1
>>> s.last_valid_index().execute()
2

>>> s = md.Series([None, None])
>>> print(s.first_valid_index()).execute()
None
>>> print(s.last_valid_index()).execute()
None

If all elements in Series are NA/null, returns None.

>>> s = md.Series()
>>> print(s.first_valid_index()).execute()
None
>>> print(s.last_valid_index()).execute()
None

If Series is empty, returns None.

For DataFrame:

>>> df = md.DataFrame({'A': [None, None, 2], 'B': [None, 3, 4]})
>>> df.execute()
     A      B
0  NaN    NaN
1  NaN    3.0
2  2.0    4.0
>>> df.first_valid_index().execute()
1
>>> df.last_valid_index().execute()
2

>>> df = md.DataFrame({'A': [None, None, None], 'B': [None, None, None]})
>>> df.execute()
     A      B
0  None   None
1  None   None
2  None   None
>>> print(df.first_valid_index()).execute()
None
>>> print(df.last_valid_index()).execute()
None

If all elements in DataFrame are NA/null, returns None.

>>> df = md.DataFrame()
>>> df.execute()
Empty DataFrame
Columns: []
Index: []
>>> print(df.first_valid_index()).execute()
None
>>> print(df.last_valid_index()).execute()
None

If DataFrame is empty, returns None.
"""


def first_valid_index(df_or_series):
    return _valid_index(df_or_series, slice(None, 1))


def last_valid_index(df_or_series):
    return _valid_index(df_or_series, slice(-1, None))


first_valid_index.__doc__ = _doc % dict(pos="first")
last_valid_index.__doc__ = _doc % dict(pos="last")
