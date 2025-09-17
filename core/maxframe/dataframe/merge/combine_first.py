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


def df_combine_first(df, other):
    """
    Update null elements with value in the same location in `other`.

    Combine two DataFrame objects by filling null values in one DataFrame
    with non-null values from other DataFrame. The row and column indexes
    of the resulting DataFrame will be the union of the two. The resulting
    dataframe contains the 'first' dataframe values and overrides the
    second one values where both first.loc[index, col] and
    second.loc[index, col] are not missing values, upon calling
    first.combine_first(second).

    Parameters
    ----------
    other : DataFrame
        Provided DataFrame to use to fill null values.

    Returns
    -------
    DataFrame
        The result of combining the provided DataFrame with the other object.

    See Also
    --------
    DataFrame.combine : Perform series-wise operation on two DataFrames
        using a given function.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df1 = md.DataFrame({'A': [None, 0], 'B': [None, 4]})
    >>> df2 = md.DataFrame({'A': [1, 1], 'B': [3, 3]})
    >>> df1.combine_first(df2).execute()
         A    B
    0  1.0  3.0
    1  0.0  4.0

    Null values still persist if the location of that null value
    does not exist in `other`

    >>> df1 = md.DataFrame({'A': [None, 0], 'B': [4, None]})
    >>> df2 = md.DataFrame({'B': [3, 3], 'C': [1, 1]}, index=[1, 2])
    >>> df1.combine_first(df2).execute()
         A    B    C
    0  NaN  4.0  NaN
    1  0.0  3.0  1.0
    2  NaN  3.0  1.0
    """
    ret = df.copy()
    ret.update(other, join="outer", overwrite=False)
    return ret


def series_combine_first(series, other):
    """
    Update null elements with value in the same location in 'other'.

    Combine two Series objects by filling null values in one Series with
    non-null values from the other Series. Result index will be the union
    of the two indexes.

    Parameters
    ----------
    other : Series
        The value(s) to be used for filling null values.

    Returns
    -------
    Series
        The result of combining the provided Series with the other object.

    See Also
    --------
    Series.combine : Perform element-wise operation on two Series
        using a given function.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s1 = md.Series([1, mt.nan])
    >>> s2 = md.Series([3, 4, 5])
    >>> s1.combine_first(s2).execute()
    0    1.0
    1    4.0
    2    5.0
    dtype: float64

    Null values still persist if the location of that null value
    does not exist in `other`

    >>> s1 = md.Series({'falcon': mt.nan, 'eagle': 160.0})
    >>> s2 = md.Series({'eagle': 200.0, 'duck': 30.0})
    >>> s1.combine_first(s2).execute()
    duck       30.0
    eagle     160.0
    falcon      NaN
    dtype: float64
    """
    ret = series.copy()
    # as Series.update does not have other args, we add them manually
    #  to the operator object
    ret.update(other)
    ret.op.join, ret.op.overwrite = "outer", False
    return ret
