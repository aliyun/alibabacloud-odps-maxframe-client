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


def at_time(df_or_series, time, axis=0):
    """
    Select values at particular time of day (e.g., 9:30AM).

    Parameters
    ----------
    time : datetime.time or str
        The values to select.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        For `Series` this parameter is unused and defaults to 0.

    Returns
    -------
    Series or DataFrame

    Raises
    ------
    TypeError
        If the index is not  a :class:`DatetimeIndex`

    See Also
    --------
    between_time : Select values between particular times of the day.
    first : Select initial periods of time series based on a date offset.
    last : Select final periods of time series based on a date offset.
    DatetimeIndex.indexer_at_time : Get just the index locations for
        values at particular time of the day.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> i = md.date_range('2018-04-09', periods=4, freq='12h')
    >>> ts = md.DataFrame({'A': [1, 2, 3, 4]}, index=i)
    >>> ts.execute()
                         A
    2018-04-09 00:00:00  1
    2018-04-09 12:00:00  2
    2018-04-10 00:00:00  3
    2018-04-10 12:00:00  4

    >>> ts.at_time('12:00').execute()
                         A
    2018-04-09 12:00:00  2
    2018-04-10 12:00:00  4
    """
    return df_or_series.between_time(time, time, inclusive="both", axis=axis)
