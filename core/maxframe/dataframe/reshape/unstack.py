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

from ...udf import builtin_function
from ..core import SERIES_TYPE


@builtin_function
def _adjust_names(pivoted, idx_names=None, col_names=None, remove_col_level=False):
    if remove_col_level:
        pivoted = pivoted.droplevel(0, axis=1)
    if idx_names:
        pivoted = pivoted.rename_axis(idx_names, axis=0)
    if col_names:
        pivoted = pivoted.rename_axis(col_names, axis=1)
    return pivoted


def unstack(df_or_series, level=-1, fill_value=None):
    """
    Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.

    Parameters
    ----------
    level : int, str, or list of these, default last level
        Level(s) to unstack, can pass level name.
    fill_value : scalar value, default None
        Value to use when replacing NaN values.

    Returns
    -------
    DataFrame
        Unstacked Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3, 4],
    ...               index=md.MultiIndex.from_product([['one', 'two'],
    ...                                                 ['a', 'b']]))
    >>> s.execute()
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1).execute()
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0).execute()
         one  two
    a    1    3
    b    2    4
    """
    if df_or_series.index.nlevels == 1:
        if isinstance(df_or_series, SERIES_TYPE):
            raise ValueError("index must be a MultiIndex to unstack")
        raw_idx_nlevels = df_or_series.index.nlevels
        stacked = df_or_series.stack(level=level, fill_value=fill_value)
        orig_order = list(stacked.index.nlevels)
        new_order = orig_order[raw_idx_nlevels:] + orig_order[:raw_idx_nlevels]
        return stacked.reorder_levels(new_order)

    nlevels = df_or_series.index.nlevels
    idx_names = list(df_or_series.index.names)
    if df_or_series.ndim == 1:
        col_names = []
    else:
        col_names = list(df_or_series.columns.names)

    name_to_idx = {name: i for i, name in enumerate(idx_names)}
    level = level if isinstance(level, list) else [level]
    level_idxes = [(nlevels + name_to_idx.get(lv, lv)) % nlevels for lv in level]
    level_idx_set = set(level_idxes)
    level_cols = [f"__level_{idx}" for idx in range(nlevels)]

    if df_or_series.ndim == 1:
        data = df_or_series.to_frame(name="__data")
    else:
        data = df_or_series
    pivot_val_col = list(data.dtypes.index)
    pivot_col_col = [level_cols[i] for i in level_idxes]
    pivot_idx_col = [level_cols[i] for i in range(nlevels) if i not in level_idx_set]

    new_idx_names = [idx_names[i] for i in range(nlevels) if i not in level_idx_set]
    new_col_names = col_names + [idx_names[i] for i in level_idxes]

    data = data.reset_index(names=level_cols)
    pivoted = data.pivot(
        index=pivot_idx_col, columns=pivot_col_col, values=pivot_val_col
    )
    pivoted.op.fill_value = fill_value
    return pivoted.mf.apply_chunk(
        _adjust_names,
        col_names=new_col_names,
        idx_names=new_idx_names,
        remove_col_level=df_or_series.ndim == 1,
        skip_infer=True,
        output_type="dataframe",
    )
