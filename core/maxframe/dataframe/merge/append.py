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

from ..datasource.dataframe import from_pandas


def append(df, other, ignore_index=False, verify_integrity=False, sort=False):
    """
    Append rows of `other` to the end of caller, returning a new object.

    Columns in `other` that are not in the caller are added as new columns.

    Parameters
    ----------
    other : DataFrame or Series/dict-like object, or list of these
        The data to append.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.
    verify_integrity : bool, default False
        If True, raise ValueError on creating index with duplicates.
    sort : bool, default False
        Sort columns if the columns of `self` and `other` are not aligned.

    Returns
    -------
    DataFrame
        A new DataFrame consisting of the rows of caller and the rows of `other`.

    See Also
    --------
    concat : General function to concatenate DataFrame or Series objects.

    Notes
    -----
    If a list of dict/series is passed and the keys are all contained in
    the DataFrame's index, the order of the columns in the resulting
    DataFrame will be unchanged.

    Iteratively appending rows to a DataFrame can be more computationally
    intensive than a single concatenate. A better solution is to append
    those rows to a list and then concatenate the list with the original
    DataFrame all at once.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2], [3, 4]], columns=list('AB'), index=['x', 'y'])
    >>> df.execute()
       A  B
    x  1  2
    y  3  4
    >>> df2 = md.DataFrame([[5, 6], [7, 8]], columns=list('AB'), index=['x', 'y'])
    >>> df.append(df2).execute()
       A  B
    x  1  2
    y  3  4
    x  5  6
    y  7  8

    With `ignore_index` set to True:

    >>> df.append(df2, ignore_index=True).execute()
       A  B
    0  1  2
    1  3  4
    2  5  6
    3  7  8

    The following, while not recommended methods for generating DataFrames,
    show two ways to generate a DataFrame from multiple data sources.

    Less efficient:

    >>> df = md.DataFrame(columns=['A'])
    >>> for i in range(5):
    ...     df = df.append({'A': i}, ignore_index=True)
    >>> df.execute()
       A
    0  0
    1  1
    2  2
    3  3
    4  4

    More efficient:

    >>> md.concat([md.DataFrame([i], columns=['A']) for i in range(5)],
    ...           ignore_index=True).execute()
       A
    0  0
    1  1
    2  2
    3  3
    4  4
    """
    from .concat import concat

    if isinstance(other, dict):
        other = from_pandas(pd.DataFrame(dict((k, [v]) for k, v in other.items())))
    if not isinstance(other, list):
        other = [other]
    return concat(
        [df] + other,
        ignore_index=ignore_index,
        verify_integrity=verify_integrity,
        sort=sort,
    )
