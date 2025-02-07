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


def pct_change(
    df_or_series, periods=1, fill_method="pad", limit=None, freq=None, **kwargs
):
    # FIXME: https://github.com/aliyun/alibabacloud-odps-maxframe-client/issues/32
    """
    Percentage change between the current and a prior element.

    Computes the percentage change from the immediately previous row by
    default. This is useful in comparing the percentage of change in a time
    series of elements.

    Parameters
    ----------
    periods : int, default 1
        Periods to shift for forming percent change.
    fill_method : str, default 'pad'
        How to handle NAs before computing percent changes.
    limit : int, default None
        The number of consecutive NAs to fill before stopping.
    freq : DateOffset, timedelta, or str, optional
        Increment to use from time series API (e.g. 'M' or BDay()).
    **kwargs
        Additional keyword arguments are passed into
        `DataFrame.shift` or `Series.shift`.

    Returns
    -------
    chg : Series or DataFrame
        The same type as the calling object.

    See Also
    --------
    Series.diff : Compute the difference of two elements in a Series.
    DataFrame.diff : Compute the difference of two elements in a DataFrame.
    Series.shift : Shift the index by some number of periods.
    DataFrame.shift : Shift the index by some number of periods.
    """

    axis = validate_axis(kwargs.pop("axis", 0))
    if fill_method is None:
        data = df_or_series
    else:
        data = df_or_series.fillna(method=fill_method, axis=axis, limit=limit)

    rs = data.div(data.shift(periods=periods, freq=freq, axis=axis, **kwargs)) - 1
    if freq is not None:
        # Shift method is implemented differently when freq is not None
        # We want to restore the original index
        rs = rs.loc[~rs.index.duplicated()]
        rs = rs.reindex_like(data)
    return rs
