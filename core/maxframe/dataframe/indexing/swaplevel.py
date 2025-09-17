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

from ..utils import validate_axis


def _swaplevel(df_or_series, i=-2, j=-1, axis=0):
    axis = validate_axis(axis, df_or_series)

    if axis == 0:
        src_idx_value = df_or_series.index_value
    else:
        src_idx_value = df_or_series.columns_value

    pd_index = src_idx_value.to_pandas()
    if not isinstance(pd_index, pd.MultiIndex):
        raise ValueError("swaplevel can only be used with MultiIndex")

    name_to_idx = {name: idx for idx, name in enumerate(pd_index.names)}
    src_idx, dest_idx = name_to_idx.get(i, i), name_to_idx.get(j, j)

    new_idxes = list(range(len(pd_index.names)))
    new_idxes[src_idx], new_idxes[dest_idx] = new_idxes[dest_idx], new_idxes[src_idx]
    if df_or_series.ndim == 2:
        return df_or_series.reorder_levels(new_idxes, axis=axis)
    else:
        return df_or_series.reorder_levels(new_idxes)


def df_swaplevel(df, i=-2, j=-1, axis=0):
    """
    Swap levels i and j in a :class:`MultiIndex`.

    Default is to swap the two innermost levels of the index.

    Parameters
    ----------
    i, j : int or str
        Levels of the indices to be swapped. Can pass level name as string.
    axis : {0 or 'index', 1 or 'columns'}, default 0
                The axis to swap levels on. 0 or 'index' for row-wise, 1 or
                'columns' for column-wise.

    Returns
    -------
    DataFrame
        DataFrame with levels swapped in MultiIndex.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     {"Grade": ["A", "B", "A", "C"]},
    ...     index=[
    ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
    ...         ["History", "Geography", "History", "Geography"],
    ...         ["January", "February", "March", "April"],
    ...     ],
    ... )
    >>> df.execute()
                                        Grade
    Final exam  History     January      A
                Geography   February     B
    Coursework  History     March        A
                Geography   April        C

    In the following example, we will swap the levels of the indices.
    Here, we will swap the levels column-wise, but levels can be swapped row-wise
    in a similar manner. Note that column-wise is the default behaviour.
    By not supplying any arguments for i and j, we swap the last and second to
    last indices.

    >>> df.swaplevel().execute()
                                        Grade
    Final exam  January     History         A
                February    Geography       B
    Coursework  March       History         A
                April       Geography       C

    By supplying one argument, we can choose which index to swap the last
    index with. We can for example swap the first index with the last one as
    follows.

    >>> df.swaplevel(0).execute()
                                        Grade
    January     History     Final exam      A
    February    Geography   Final exam      B
    March       History     Coursework      A
    April       Geography   Coursework      C

    We can also define explicitly which indices we want to swap by supplying values
    for both i and j. Here, we for example swap the first and second indices.

    >>> df.swaplevel(0, 1).execute()
                                        Grade
    History     Final exam  January         A
    Geography   Final exam  February        B
    History     Coursework  March           A
    Geography   Coursework  April           C
    """
    return _swaplevel(df, i, j, axis=axis)


def series_swaplevel(series, i=-2, j=-1):
    """
    Swap levels i and j in a :class:`MultiIndex`.

    Default is to swap the two innermost levels of the index.

    Parameters
    ----------
    i, j : int or str
        Levels of the indices to be swapped. Can pass level name as string.

    Returns
    -------
    Series
        Series with levels swapped in MultiIndex.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series(
    ...     ["A", "B", "A", "C"],
    ...     index=[
    ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
    ...         ["History", "Geography", "History", "Geography"],
    ...         ["January", "February", "March", "April"],
    ...     ],
    ... )
    >>> s.execute()
    Final exam  History     January      A
                Geography   February     B
    Coursework  History     March        A
                Geography   April        C
    dtype: object

    In the following example, we will swap the levels of the indices.
    Here, we will swap the levels column-wise, but levels can be swapped row-wise
    in a similar manner. Note that column-wise is the default behaviour.
    By not supplying any arguments for i and j, we swap the last and second to
    last indices.

    >>> s.swaplevel().execute()
    Final exam  January     History         A
                February    Geography       B
    Coursework  March       History         A
                April       Geography       C
    dtype: object

    By supplying one argument, we can choose which index to swap the last
    index with. We can for example swap the first index with the last one as
    follows.

    >>> s.swaplevel(0).execute()
    January     History     Final exam      A
    February    Geography   Final exam      B
    March       History     Coursework      A
    April       Geography   Coursework      C
    dtype: object

    We can also define explicitly which indices we want to swap by supplying values
    for both i and j. Here, we for example swap the first and second indices.

    >>> s.swaplevel(0, 1).execute()
    History     Final exam  January         A
    Geography   Final exam  February        B
    History     Coursework  March           A
    Geography   Coursework  April           C
    dtype: object
    """
    return _swaplevel(series, i, j)
