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

from ..utils import validate_axis


def truncate(df, before=None, after=None, axis=0, copy=None):
    """
    Truncate a Series or DataFrame before and after some index value.

    This is a useful shorthand for boolean indexing based on index
    values above or below certain thresholds.

    Parameters
    ----------
    before : date, str, int
        Truncate all rows before this index value.
    after : date, str, int
        Truncate all rows after this index value.
    axis : {0 or 'index', 1 or 'columns'}, optional
        Axis to truncate. Truncates the index (rows) by default.
        For `Series` this parameter is unused and defaults to 0.
    copy : bool, default is True,
        This parameter is only kept for compatibility with pandas.

    Returns
    -------
    type of caller
        The truncated Series or DataFrame.

    See Also
    --------
    DataFrame.loc : Select a subset of a DataFrame by label.
    DataFrame.iloc : Select a subset of a DataFrame by position.

    Notes
    -----
    If the index being truncated contains only datetime values,
    `before` and `after` may be specified as strings instead of
    Timestamps.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
    ...                    'B': ['f', 'g', 'h', 'i', 'j'],
    ...                    'C': ['k', 'l', 'm', 'n', 'o']},
    ...                   index=[1, 2, 3, 4, 5])
    >>> df.execute()
       A  B  C
    1  a  f  k
    2  b  g  l
    3  c  h  m
    4  d  i  n
    5  e  j  o

    >>> df.truncate(before=2, after=4).execute()
       A  B  C
    2  b  g  l
    3  c  h  m
    4  d  i  n

    The columns of a DataFrame can be truncated.

    >>> df.truncate(before="A", after="B", axis="columns").execute()
       A  B
    1  a  f
    2  b  g
    3  c  h
    4  d  i
    5  e  j

    For Series, only rows can be truncated.

    >>> df['A'].truncate(before=2, after=4).execute()
    2    b
    3    c
    4    d
    Name: A, dtype: object

    The index values in ``truncate`` can be datetimes or string
    dates.

    >>> dates = md.date_range('2016-01-01', '2016-02-01', freq='s')
    >>> df = md.DataFrame(index=dates, data={'A': 1})
    >>> df.tail().execute()
                         A
    2016-01-31 23:59:56  1
    2016-01-31 23:59:57  1
    2016-01-31 23:59:58  1
    2016-01-31 23:59:59  1
    2016-02-01 00:00:00  1

    >>> df.truncate(before=md.Timestamp('2016-01-05'),
    ...             after=md.Timestamp('2016-01-10')).tail().execute()
                         A
    2016-01-09 23:59:56  1
    2016-01-09 23:59:57  1
    2016-01-09 23:59:58  1
    2016-01-09 23:59:59  1
    2016-01-10 00:00:00  1

    Because the index is a DatetimeIndex containing only dates, we can
    specify `before` and `after` as strings. They will be coerced to
    Timestamps before truncation.

    >>> df.truncate('2016-01-05', '2016-01-10').tail().execute()
                         A
    2016-01-09 23:59:56  1
    2016-01-09 23:59:57  1
    2016-01-09 23:59:58  1
    2016-01-09 23:59:59  1
    2016-01-10 00:00:00  1

    Note that ``truncate`` assumes a 0 value for any unspecified time
    component (midnight). This differs from partial string slicing, which
    returns any partially matching dates.

    >>> df.loc['2016-01-05':'2016-01-10', :].tail().execute()
                         A
    2016-01-10 23:59:55  1
    2016-01-10 23:59:56  1
    2016-01-10 23:59:57  1
    2016-01-10 23:59:58  1
    2016-01-10 23:59:59  1
    """
    _ = copy
    axis = validate_axis(axis, df)
    return df.loc[before:after, :] if axis == 0 else df.loc[:, before:after]
