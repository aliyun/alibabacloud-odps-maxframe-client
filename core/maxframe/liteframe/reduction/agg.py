# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import AggCall


def agg(
    self,
    func=None,
    *,
    method="auto",
    numeric_only=None,
    input_frame_type="pandas",
    **kwargs
):
    """Aggregate using one or more operations over columns.

    Each column is reduced independently to a single scalar, and the
    results are combined into a single-row LiteFrame.

    Parameters
    ----------
    func : str, list of str, or dict, optional
        Aggregation specification. A string names a single reduction
        (e.g. ``'sum'``); a list applies several reductions to each
        column, producing a single row with ``{column}_{function}``
        suffixed columns (e.g. ``a_sum``, ``a_mean``); a dict maps
        column names to reductions. By default None, which supports the
        kwargs-only named-aggregation form (e.g.
        ``lf.agg(a_sum=('a', 'sum'))``).
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    input_frame_type : str, default 'pandas'
        Frame type hint for the operator.
    **kwargs
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column aggregate results.

    See Also
    --------
    sum : Return the sum of each column.
    mean : Return the mean of each column.
    count : Return the count of non-null values in each column.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.agg('sum')
    >>> result.execute()
       a  b
    0  6  15
    >>> result = lf.agg(['sum', 'mean'])
    >>> result.execute()
       a_sum  a_mean  b_sum  b_mean
    0      6     2.0     15     5.0
    """
    op = LiteFrameAgg(
        raw_func=func,
        raw_func_kw=kwargs if kwargs else None,
        groupby_params=None,
        method=method,
        numeric_only=numeric_only,
        input_frame_type=input_frame_type,
    )
    return op(self)


def sum(self, method="auto", numeric_only=None, **kw):
    """Return the sum of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column sums.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.sum()
    >>> result.execute()
       a   b
    0  6  15
    """
    return self.agg("sum", method=method, numeric_only=numeric_only, **kw)


def mean(self, method="auto", numeric_only=None, **kw):
    """Return the mean of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column means.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.mean()
    >>> result.execute()
         a    b
    0  2.0  5.0
    """
    return self.agg("mean", method=method, numeric_only=numeric_only, **kw)


def min(self, method="auto", numeric_only=None, **kw):
    """Return the minimum of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column minima.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.min()
    >>> result.execute()
       a  b
    0  1  4
    """
    return self.agg("min", method=method, numeric_only=numeric_only, **kw)


def max(self, method="auto", numeric_only=None, **kw):
    """Return the maximum of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column maxima.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.max()
    >>> result.execute()
       a  b
    0  3  6
    """
    return self.agg("max", method=method, numeric_only=numeric_only, **kw)


def prod(self, method="auto", numeric_only=None, **kw):
    """Return the product of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column products.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.prod()
    >>> result.execute()
       a   b
    0  6  120
    """
    return self.agg("prod", method=method, numeric_only=numeric_only, **kw)


def count(self, method="auto", **kw):
    """Return the count of non-null values in each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column non-null counts.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
    >>> result = lf.count()
    >>> result.execute()
         a  b
    0  2  3
    """
    return self.agg("count", method=method, **kw)


def size(self, method="auto", **kw):
    """Return the total number of rows.

    Unlike ``count``, this includes null values.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame with the row count.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.size()
    >>> result.execute()
           size
    0      3
    """
    return self.agg("size", method=method, **kw)


def var(self, method="auto", numeric_only=None, **kw):
    """Return the unbiased variance of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column variances.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.var()
    >>> result.execute()
         a    b
    0  1.0  1.0
    """
    return self.agg("var", method=method, numeric_only=numeric_only, **kw)


def std(self, method="auto", numeric_only=None, **kw):
    """Return the sample standard deviation of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column standard deviations.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.std()
    >>> result.execute()
             a           b
    0  1.000000    1.000000
    """
    return self.agg("std", method=method, numeric_only=numeric_only, **kw)


def sem(self, method="auto", numeric_only=None, **kw):
    """Return the standard error of the mean of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column standard errors.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.sem()
    >>> result.execute()
             a           b
    0  0.577350    0.577350
    """
    return self.agg("sem", method=method, numeric_only=numeric_only, **kw)


def skew(self, method="auto", numeric_only=None, **kw):
    """Return the unbiased skewness of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column skewness values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.skew()
    >>> result.execute()
         a    b
    0  0.0  0.0
    """
    return self.agg("skew", method=method, numeric_only=numeric_only, **kw)


def kurt(self, method="auto", numeric_only=None, **kw):
    """Return the unbiased kurtosis of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column kurtosis values.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4], 'b': [4, 5, 6, 7]})
    >>> result = lf.kurt()
    >>> result.execute()
         a    b
    0 -1.2 -1.2
    """
    return self.agg("kurt", method=method, numeric_only=numeric_only, **kw)


def nunique(self, method="auto", numeric_only=None, dropna=True, **kw):
    """Return the number of distinct values in each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    dropna : bool, default True
        When True, exclude NaN values from the count.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column distinct value counts.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 2], 'b': [4, 4, 5]})
    >>> result = lf.nunique()
    >>> result.execute()
       a  b
    0  2  2
    """
    return self.agg(
        AggCall("nunique", dropna=dropna),
        method=method,
        numeric_only=numeric_only,
        **kw,
    )


def median(self, method="auto", numeric_only=None, **kw):
    """Return the median of each column.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column medians.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> result = lf.median()
    >>> result.execute()
         a    b
    0  2.0  5.0
    """
    return self.agg("median", method=method, numeric_only=numeric_only, **kw)


def all(self, method="auto", numeric_only=None, **kw):
    """Return whether all elements in each column are truthy.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column boolean results.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [0, 5, 6]})
    >>> result = lf.all()
    >>> result.execute()
           a      b
    0   True  False
    """
    return self.agg("all", method=method, numeric_only=numeric_only, **kw)


def any(self, method="auto", numeric_only=None, **kw):
    """Return whether any element in each column is truthy.

    Parameters
    ----------
    method : {'auto', 'tree', 'shuffle'}, default 'auto'
        Execution strategy. ``'auto'`` selects based on data size.
    numeric_only : bool, optional
        When True, restrict to numeric columns. By default None.
    **kw
        Additional keyword arguments forwarded to the reduction.

    Returns
    -------
    LiteFrame
        Single-row LiteFrame of per-column boolean results.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [0, 0, 0], 'b': [0, 5, 6]})
    >>> result = lf.any()
    >>> result.execute()
            a     b
    0  False  True
    """
    return self.agg("any", method=method, numeric_only=numeric_only, **kw)
