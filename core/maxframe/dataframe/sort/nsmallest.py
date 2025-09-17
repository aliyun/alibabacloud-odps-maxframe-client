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


def _nsmallest(df, n, columns=None, keep="first"):
    op = DataFrameSortValues(
        output_types=[OutputType.dataframe],
        axis=0,
        by=columns,
        ignore_index=False,
        ascending=True,
        nrows=n,
        keep_kind=keep,
    )
    return op(df)


def df_nsmallest(df, n, columns, keep="first"):
    """
    Return the first `n` rows ordered by `columns` in ascending order.

    Return the first `n` rows with the smallest values in `columns`, in
    ascending order. The columns that are not specified are returned as
    well, but not used for ordering.

    This method is equivalent to
    ``df.sort_values(columns, ascending=True).head(n)``, but more
    performant.

    Parameters
    ----------
    n : int
        Number of items to retrieve.
    columns : list or str
        Column name or names to order by.
    keep : {'first', 'last', 'all'}, default 'first'
        Where there are duplicate values:

        - ``first`` : take the first occurrence.
        - ``last`` : take the last occurrence.
        - ``all`` : do not drop any duplicates, even it means
          selecting more than `n` items.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.nlargest : Return the first `n` rows ordered by `columns` in
        descending order.
    DataFrame.sort_values : Sort DataFrame by the values.
    DataFrame.head : Return the first `n` rows without re-ordering.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'population': [59000000, 65000000, 434000,
    ...                                   434000, 434000, 337000, 337000,
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
    Nauru         337000      182      NR
    Tuvalu         11300       38      TV
    Anguilla       11300      311      AI

    In the following example, we will use ``nsmallest`` to select the
    three rows having the smallest values in column "population".

    >>> df.nsmallest(3, 'population').execute()
              population    GDP alpha-2
    Tuvalu         11300     38      TV
    Anguilla       11300    311      AI
    Iceland       337000  17036      IS

    When using ``keep='last'``, ties are resolved in reverse order:

    >>> df.nsmallest(3, 'population', keep='last').execute()
              population  GDP alpha-2
    Anguilla       11300  311      AI
    Tuvalu         11300   38      TV
    Nauru         337000  182      NR

    When using ``keep='all'``, all duplicate items are maintained:

    >>> df.nsmallest(3, 'population', keep='all').execute()
              population    GDP alpha-2
    Tuvalu         11300     38      TV
    Anguilla       11300    311      AI
    Iceland       337000  17036      IS
    Nauru         337000    182      NR

    To order by the smallest values in column "population" and then "GDP", we can
    specify multiple columns like in the next example.

    >>> df.nsmallest(3, ['population', 'GDP']).execute()
              population  GDP alpha-2
    Tuvalu         11300   38      TV
    Anguilla       11300  311      AI
    Nauru         337000  182      NR
    """
    return _nsmallest(df, n, columns, keep=keep)


def series_nsmallest(df, n, keep="first"):
    """
    Return the smallest `n` elements.

    Parameters
    ----------
    n : int, default 5
        Return this many ascending sorted values.
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
        The `n` smallest values in the Series, sorted in increasing order.

    See Also
    --------
    Series.nlargest: Get the `n` largest elements.
    Series.sort_values: Sort Series by values.
    Series.head: Return the first `n` rows.

    Notes
    -----
    Faster than ``.sort_values().head(n)`` for small `n` relative to
    the size of the ``Series`` object.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> countries_population = {"Italy": 59000000, "France": 65000000,
    ...                         "Brunei": 434000, "Malta": 434000,
    ...                         "Maldives": 434000, "Iceland": 337000,
    ...                         "Nauru": 11300, "Tuvalu": 11300,
    ...                         "Anguilla": 11300, "Montserrat": 5200}
    >>> s = md.Series(countries_population)
    >>> s.execute()
    Italy       59000000
    France      65000000
    Brunei        434000
    Malta         434000
    Maldives      434000
    Iceland       337000
    Nauru          11300
    Tuvalu         11300
    Anguilla       11300
    Montserrat      5200
    dtype: int64

    The `n` smallest elements where ``n=5`` by default.

    >>> s.nsmallest().execute()
    Montserrat    5200
    Nauru        11300
    Tuvalu       11300
    Anguilla     11300
    Iceland     337000
    dtype: int64

    The `n` smallest elements where ``n=3``. Default `keep` value is
    'first' so Nauru and Tuvalu will be kept.

    >>> s.nsmallest(3).execute()
    Montserrat   5200
    Nauru       11300
    Tuvalu      11300
    dtype: int64

    The `n` smallest elements where ``n=3`` and keeping the last
    duplicates. Anguilla and Tuvalu will be kept since they are the last
    with value 11300 based on the index order.

    >>> s.nsmallest(3, keep='last').execute()
    Montserrat   5200
    Anguilla    11300
    Tuvalu      11300
    dtype: int64

    The `n` smallest elements where ``n=3`` with all duplicates kept. Note
    that the returned Series has four elements due to the three duplicates.

    >>> s.nsmallest(3, keep='all').execute()
    Montserrat   5200
    Nauru       11300
    Tuvalu      11300
    Anguilla    11300
    dtype: int64
    """
    return _nsmallest(df, n, keep=keep)
