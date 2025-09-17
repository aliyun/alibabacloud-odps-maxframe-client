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


def cov_dataframe(df, min_periods=None, ddof=1, numeric_only=True):
    """
    Compute pairwise covariance of columns, excluding NA/null values.

    Compute the pairwise covariance among the series of a DataFrame.
    The returned data frame is the `covariance matrix
    <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
    of the DataFrame.

    Both NA and null values are automatically excluded from the
    calculation. (See the note below about bias from missing values.)
    A threshold can be set for the minimum number of
    observations for each value created. Comparisons with observations
    below this threshold will be returned as ``NaN``.

    This method is generally used for the analysis of time series data to
    understand the relationship between different measures
    across time.

    Parameters
    ----------
    min_periods : int, optional
        Minimum number of observations required per pair of columns
        to have a valid result.

    ddof : int, default 1
        Delta degrees of freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        This argument is applicable only when no ``nan`` is in the dataframe.

    numeric_only : bool, default True
        Include only `float`, `int` or `boolean` data.

    Returns
    -------
    DataFrame
        The covariance matrix of the series of the DataFrame.

    See Also
    --------
    Series.cov : Compute covariance with another Series.
    core.window.ewm.ExponentialMovingWindow.cov : Exponential weighted sample
        covariance.
    core.window.expanding.Expanding.cov : Expanding sample covariance.
    core.window.rolling.Rolling.cov : Rolling sample covariance.

    Notes
    -----
    Returns the covariance matrix of the DataFrame's time series.
    The covariance is normalized by N-ddof.

    For DataFrames that have Series that are missing data (assuming that
    data is `missing at random
    <https://en.wikipedia.org/wiki/Missing_data#Missing_at_random>`__)
    the returned covariance matrix will be an unbiased estimate
    of the variance and covariance between the member Series.

    However, for many applications this estimate may not be acceptable
    because the estimate covariance matrix is not guaranteed to be positive
    semi-definite. This could lead to estimate correlations having
    absolute values which are greater than one, and/or a non-invertible
    covariance matrix. See `Estimation of covariance matrices
    <https://en.wikipedia.org/w/index.php?title=Estimation_of_covariance_
    matrices>`__ for more details.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([(1, 2), (0, 3), (2, 0), (1, 1)],
    ...                   columns=['dogs', 'cats'])
    >>> df.cov().execute()
              dogs      cats
    dogs  0.666667 -1.000000
    cats -1.000000  1.666667

    >>> mt.random.seed(42)
    >>> df = md.DataFrame(mt.random.randn(1000, 5),
    ...                   columns=['a', 'b', 'c', 'd', 'e'])
    >>> df.cov().execute()
              a         b         c         d         e
    a  0.998438 -0.020161  0.059277 -0.008943  0.014144
    b -0.020161  1.059352 -0.008543 -0.024738  0.009826
    c  0.059277 -0.008543  1.010670 -0.001486 -0.000271
    d -0.008943 -0.024738 -0.001486  0.921297 -0.013692
    e  0.014144  0.009826 -0.000271 -0.013692  0.977795

    **Minimum number of periods**

    This method also supports an optional ``min_periods`` keyword
    that specifies the required minimum number of non-NA observations for
    each column pair in order to have a valid result:

    >>> mt.random.seed(42)
    >>> df = md.DataFrame(mt.random.randn(20, 3),
    ...                   columns=['a', 'b', 'c'])
    >>> df.loc[df.index[:5], 'a'] = mt.nan
    >>> df.loc[df.index[5:10], 'b'] = mt.nan
    >>> df.cov(min_periods=12).execute()
              a         b         c
    a  0.316741       NaN -0.150812
    b       NaN  1.248003  0.191417
    c -0.150812  0.191417  0.895202
    """
    from ..statistics.corr import DataFrameCorr

    if not numeric_only:
        raise NotImplementedError("numeric_only==True not supported")
    op = DataFrameCorr(method="cov", min_periods=min_periods, ddof=ddof)
    return op(df)


def cov_series(series, other, min_periods=None, ddof=1):
    """
    Compute covariance with Series, excluding missing values.

    The two `Series` objects are not required to be the same length and
    will be aligned internally before the covariance is calculated.

    Parameters
    ----------
    other : Series
        Series with which to compute the covariance.
    min_periods : int, optional
        Minimum number of observations needed to have a valid result.
    ddof : int, default 1
        Delta degrees of freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.

    Returns
    -------
    float
        Covariance between Series and other normalized by N-1
        (unbiased estimator).

    See Also
    --------
    DataFrame.cov : Compute pairwise covariance of columns.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s1 = md.Series([0.90010907, 0.13484424, 0.62036035])
    >>> s2 = md.Series([0.12528585, 0.26962463, 0.51111198])
    >>> s1.cov(s2).execute()
    -0.01685762652715874
    """
    from ..statistics.corr import DataFrameCorr

    op = DataFrameCorr(other=other, method="cov", min_periods=min_periods, ddof=ddof)
    return op(series)
