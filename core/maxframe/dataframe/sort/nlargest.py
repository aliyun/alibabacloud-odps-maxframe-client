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

from ...core import OutputType
from .sort_values import DataFrameSortValues


def _nlargest(df, n, columns=None, keep="first"):
    op = DataFrameSortValues(
        output_types=[OutputType.dataframe],
        axis=0,
        by=columns,
        ignore_index=False,
        ascending=False,
        nrows=n,
        keep_kind=keep,
    )
    return op(df)


def df_nlargest(df, n, columns, keep="first"):
    """
    Return the first `n` rows ordered by `columns` in descending order.

    Return the first `n` rows with the largest values in `columns`, in
    descending order. The columns that are not specified are returned as
    well, but not used for ordering.

    This method is equivalent to
    ``df.sort_values(columns, ascending=False).head(n)``, but more
    performant.

    Parameters
    ----------
    n : int
        Number of rows to return.
    columns : label or list of labels
        Column label(s) to order by.
    keep : {'first', 'last', 'all'}, default 'first'
        Where there are duplicate values:

        - `first` : prioritize the first occurrence(s)
        - `last` : prioritize the last occurrence(s)
        - ``all`` : do not drop any duplicates, even it means
                    selecting more than `n` items.

    Returns
    -------
    DataFrame
        The first `n` rows ordered by the given columns in descending
        order.

    See Also
    --------
    DataFrame.nsmallest : Return the first `n` rows ordered by `columns` in
        ascending order.
    DataFrame.sort_values : Sort DataFrame by the values.
    DataFrame.head : Return the first `n` rows without re-ordering.

    Notes
    -----
    This function cannot be used with all column types. For example, when
    specifying columns with `object` or `category` dtypes, ``TypeError`` is
    raised.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'population': [59000000, 65000000, 434000,
    ...                                   434000, 434000, 337000, 11300,
    ...                                   11300, 11300],
    ...                    'GDP': [1937894, 2583560 , 12011, 4520, 12128,
    ...                            17036, 182, 38, 311],
    ...                    'alpha-2': ["IT", "FR", "MT", "MV", "BN",
    ...                                "IS", "NR", "TV", "AI"]},
    ...                   index=["Italy", "France", "Malta",
    ...                          "Maldives", "Brunei", "Iceland",
    ...                          "Nauru", "Tuvalu", "Anguilla"])
    >>> df.execute()
              population      GDP alpha-2
    Italy       59000000  1937894      IT
    France      65000000  2583560      FR
    Malta         434000    12011      MT
    Maldives      434000     4520      MV
    Brunei        434000    12128      BN
    Iceland       337000    17036      IS
    Nauru          11300      182      NR
    Tuvalu         11300       38      TV
    Anguilla       11300      311      AI

    In the following example, we will use ``nlargest`` to select the three
    rows having the largest values in column "population".

    >>> df.nlargest(3, 'population').execute()
            population      GDP alpha-2
    France    65000000  2583560      FR
    Italy     59000000  1937894      IT
    Malta       434000    12011      MT

    When using ``keep='last'``, ties are resolved in reverse order:

    >>> df.nlargest(3, 'population', keep='last').execute()
            population      GDP alpha-2
    France    65000000  2583560      FR
    Italy     59000000  1937894      IT
    Brunei      434000    12128      BN

    When using ``keep='all'``, all duplicate items are maintained:

    >>> df.nlargest(3, 'population', keep='all').execute()
              population      GDP alpha-2
    France      65000000  2583560      FR
    Italy       59000000  1937894      IT
    Malta         434000    12011      MT
    Maldives      434000     4520      MV
    Brunei        434000    12128      BN

    To order by the largest values in column "population" and then "GDP",
    we can specify multiple columns like in the next example.

    >>> df.nlargest(3, ['population', 'GDP']).execute()
            population      GDP alpha-2
    France    65000000  2583560      FR
    Italy     59000000  1937894      IT
    Brunei      434000    12128      BN
    """
    return _nlargest(df, n, columns, keep=keep)


def series_nlargest(df, n, keep="first"):
    """
    Return the largest `n` elements.

    Parameters
    ----------
    n : int, default 5
        Return this many descending sorted values.
    keep : {'first', 'last', 'all'}, default 'first'
        When there are duplicate values that cannot all fit in a
        Series of `n` elements:

        - ``first`` : return the first `n` occurrences in order
            of appearance.
        - ``last`` : return the last `n` occurrences in reverse
            order of appearance.
        - ``all`` : keep all occurrences. This can result in a Series of
            size larger than `n`.

    Returns
    -------
    Series
        The `n` largest values in the Series, sorted in decreasing order.

    See Also
    --------
    Series.nsmallest: Get the `n` smallest elements.
    Series.sort_values: Sort Series by values.
    Series.head: Return the first `n` rows.

    Notes
    -----
    Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
    relative to the size of the ``Series`` object.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> countries_population = {"Italy": 59000000, "France": 65000000,
    ...                         "Malta": 434000, "Maldives": 434000,
    ...                         "Brunei": 434000, "Iceland": 337000,
    ...                         "Nauru": 11300, "Tuvalu": 11300,
    ...                         "Anguilla": 11300, "Montserrat": 5200}
    >>> s = md.Series(countries_population)
    >>> s.execute()
    Italy       59000000
    France      65000000
    Malta         434000
    Maldives      434000
    Brunei        434000
    Iceland       337000
    Nauru          11300
    Tuvalu         11300
    Anguilla       11300
    Montserrat      5200
    dtype: int64

    The `n` largest elements where ``n=5`` by default.

    >>> s.nlargest().execute()
    France      65000000
    Italy       59000000
    Malta         434000
    Maldives      434000
    Brunei        434000
    dtype: int64

    The `n` largest elements where ``n=3``. Default `keep` value is 'first'
    so Malta will be kept.

    >>> s.nlargest(3).execute()
    France    65000000
    Italy     59000000
    Malta       434000
    dtype: int64

    The `n` largest elements where ``n=3`` and keeping the last duplicates.
    Brunei will be kept since it is the last with value 434000 based on
    the index order.

    >>> s.nlargest(3, keep='last').execute()
    France      65000000
    Italy       59000000
    Brunei        434000
    dtype: int64

    The `n` largest elements where ``n=3`` with all duplicates kept. Note
    that the returned Series has five elements due to the three duplicates.

    >>> s.nlargest(3, keep='all').execute()
    France      65000000
    Italy       59000000
    Malta         434000
    Maldives      434000
    Brunei        434000
    dtype: int64
    """
    return _nlargest(df, n, keep=keep)
